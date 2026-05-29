import gc
import inspect
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from vllm.forward_context import set_forward_context
except ImportError:
    set_forward_context = None


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """Methods used by the ES trainer and merge pipeline."""

    _VISUAL_PREFIXES = ("visual.", "model.visual.")

    def cleanup_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def _should_perturb(self, name: str) -> bool:
        if os.environ.get("PERTURB_VISUAL", "0") == "1":
            return True
        return not name.startswith(self._VISUAL_PREFIXES)

    def _set_seed(self, seed):
        self.local_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _forward_model_logits(self, input_ids):
        model = self.model_runner.model
        model.eval()
        ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(ids_tensor.numel(), dtype=torch.long, device=self.device)

        with torch.no_grad():
            if set_forward_context is not None and hasattr(self.model_runner, "vllm_config"):
                with set_forward_context(attn_metadata=None, vllm_config=self.model_runner.vllm_config):
                    outputs = model(input_ids=ids_tensor, positions=positions)
            else:
                if "positions" in inspect.signature(model.forward).parameters:
                    outputs = model(input_ids=ids_tensor, positions=positions)
                else:
                    outputs = model(input_ids=ids_tensor.unsqueeze(0))

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if logits.ndim == 3:
            logits = logits[0]
        return logits, ids_tensor, positions, outputs

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            if self._should_perturb(name):
                p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, sigma, negate=False):
        self._set_seed(seed)
        sign = -1.0 if negate else 1.0
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            if self._should_perturb(name):
                p.data.add_(-sign * float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        param_count = 0
        for name, p in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                param_count += 1
                continue
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)
            for i, seed in enumerate(seeds):
                self._set_seed(seed)
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                noise_fp32 = noise.to(torch.float32)
                del noise
                noise_fp32.mul_(coeffs[i])
                update_accumulator.add_(noise_fp32)
                del noise_fp32
            update_accumulator.div_(population_size)
            update_accumulator.mul_(alpha)
            p.data.add_(update_accumulator.to(p.dtype))
            del update_accumulator
            param_count += 1
            if param_count % 50 == 0:
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        return True

    def get_worker_ip(self):
        from vllm.utils import get_ip
        return get_ip()

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True

    def dump_noise_for_seed(self, seed: int, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        noise_state = {}
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            noise_state[name] = noise.detach().cpu()
            del noise
        torch.save(noise_state, os.path.join(out_dir, f"noise_seed_{int(seed)}.pt"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def print_model_weights_stats(self):
        for name, p in self.model_runner.model.named_parameters():
            print(f"Param: {name}, Shape: {p.shape}")
        return True

    def store_base_weights(self):
        self._base_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self._base_weights[name] = p.data.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def _apply_components_from_base(self, components):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")

        param_count = 0
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
            if self._should_perturb(name):
                perturbation = torch.zeros_like(p.data, dtype=torch.float32)
                for seed, sigma, weight in components:
                    gen = torch.Generator(device=p.device)
                    gen.manual_seed(int(seed))
                    noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                    noise_fp32 = noise.to(torch.float32)
                    del noise
                    noise_fp32.mul_(float(weight) * float(sigma))
                    perturbation.add_(noise_fp32)
                    del noise_fp32
                p.data.add_(perturbation.to(p.dtype))
                del perturbation
            param_count += 1
            if param_count % 50 == 0:
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        return True

    def apply_perturbation(self, seed, sigma):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        self._set_seed(seed)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            if self._should_perturb(name):
                p.data.add_(float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def reset_to_base_weights(self):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def clear_base_weights(self):
        if hasattr(self, "_base_weights"):
            del self._base_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def apply_averaged_perturbations(self, seeds_sigmas, weights=None):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        num_models = len(seeds_sigmas)
        if weights is None:
            weights = [1.0 / num_models] * num_models
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        components = [
            (int(seed), float(sigma), float(weight))
            for (seed, sigma), weight in zip(seeds_sigmas, weights)
        ]
        return self._apply_components_from_base(components)

    def apply_direction_from_base(self, components):
        return self._apply_components_from_base(components)

    def commit_current_as_base(self):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        for name, p in self.model_runner.model.named_parameters():
            self._base_weights[name] = p.data.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def commit_direction_from_base(self, components):
        self._apply_components_from_base(components)
        return self.commit_current_as_base()

    def apply_weighted_delta_from_base(self, seeds_sigmas, weights, alpha):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        if len(seeds_sigmas) != len(weights):
            raise ValueError("seeds_sigmas and weights must have the same length")
        components = [
            (int(seed), float(sigma), float(weight) * float(alpha))
            for (seed, sigma), weight in zip(seeds_sigmas, weights)
        ]
        return self._apply_components_from_base(components)

    def _coordinate_sketch_plan(self, points_per_tensor: int):
        plan = []
        for name, p in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue
            take = min(int(points_per_tensor), int(p.numel()))
            if take <= 0:
                continue
            plan.append(
                {
                    "name": name,
                    "numel": int(p.numel()),
                    "selected_numel": int(take),
                }
            )
        return plan

    def _coordinate_sketch_for_seed(self, seed: int, sigma: float, points_per_tensor: int):
        sketch_parts = []
        for name, p in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue
            take = min(int(points_per_tensor), int(p.numel()))
            if take <= 0:
                continue
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            prefix = torch.randn((take,), dtype=p.dtype, device=p.device, generator=gen)
            prefix = prefix.to(torch.float32)
            prefix.mul_(float(sigma))
            sketch_parts.append(prefix.cpu())
            del prefix
        if not sketch_parts:
            return np.zeros((0,), dtype=np.float32)
        sketch = torch.cat(sketch_parts, dim=0).numpy().astype(np.float32, copy=False)
        del sketch_parts
        return sketch

    def get_isotropic_coordinate_sketch_batch(self, seed_sigma_pairs, points_per_tensor=32):
        plan = self._coordinate_sketch_plan(points_per_tensor)
        sketches = []
        for seed, sigma in seed_sigma_pairs:
            sketches.append(self._coordinate_sketch_for_seed(int(seed), float(sigma), int(points_per_tensor)))
        if sketches:
            sketch_matrix = np.stack(sketches, axis=0).astype(np.float32, copy=False)
        else:
            sketch_matrix = np.zeros((0, 0), dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {
            "sketch_matrix": sketch_matrix,
            "sketch_plan": plan,
            "sketch_dim": int(sketch_matrix.shape[1]) if sketch_matrix.ndim == 2 else 0,
            "points_per_tensor": int(points_per_tensor),
        }

    def get_logits_for_prompt(self, input_ids_list):
        results = []
        with torch.no_grad():
            for input_ids in input_ids_list:
                logits, ids_tensor, positions, outputs = self._forward_model_logits(input_ids)
                last_logits = logits[-1, :].detach().cpu()
                results.append(last_logits)
                del logits, ids_tensor, positions, outputs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return results

    def score_answer_candidates(self, prompt_token_ids_list, candidate_token_ids_list):
        if len(prompt_token_ids_list) != len(candidate_token_ids_list):
            raise ValueError("prompt_token_ids_list and candidate_token_ids_list must have same length")

        all_scores = []
        with torch.no_grad():
            for prompt_token_ids, candidates in zip(prompt_token_ids_list, candidate_token_ids_list):
                example_scores = []
                for candidate_token_ids in candidates:
                    if not candidate_token_ids:
                        example_scores.append(float("-inf"))
                        continue
                    full_input_ids = list(prompt_token_ids) + list(candidate_token_ids)
                    logits, ids_tensor, positions, outputs = self._forward_model_logits(full_input_ids)
                    prompt_len = len(prompt_token_ids)
                    candidate_len = len(candidate_token_ids)
                    selected_logits = logits[prompt_len - 1:prompt_len - 1 + candidate_len, :].float()
                    log_probs = F.log_softmax(selected_logits, dim=-1)
                    candidate_tensor = torch.tensor(candidate_token_ids, dtype=torch.long, device=log_probs.device)
                    token_scores = log_probs.gather(1, candidate_tensor.unsqueeze(1)).squeeze(1)
                    example_scores.append(float(token_scores.mean().item()))
                    del logits, ids_tensor, positions, outputs, selected_logits, log_probs, candidate_tensor, token_scores
                all_scores.append(example_scores)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return all_scores

    def generate_with_logits_callback(self, input_ids, max_new_tokens, temperature=1.0):
        model = self.model_runner.model
        model.eval()
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        all_logits = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=current_ids)
                last_logits = outputs.logits[0, -1, :]
                all_logits.append(last_logits.cpu())
                if temperature > 0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                del outputs
        generated = current_ids[0].cpu().tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return generated, all_logits
