"""Corp: constructive population collapse + adaptive local recenter."""

import argparse
from datetime import datetime
import gc
import json
import math
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
try:
    import ray
except ModuleNotFoundError:
    ray = None
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from core import cleanup_engines, launch_engines
from core import collapse_ops, recenter_ops, corp_ops, artifact_ops, eval_ops
from data_handlers import get_dataset_handler, list_datasets

NEWLINE = chr(10)
USE_RAY = True
DirectionSpec = corp_ops.DirectionSpec


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

def _unwrap(result):
    return result[0] if isinstance(result, list) and len(result) == 1 else result


def engine_generate(engine, prompts, sampling_params):
    if USE_RAY:
        return ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))
    return engine.generate(prompts, sampling_params, use_tqdm=False)


def engines_generate(engines, prompts, sampling_params):
    if USE_RAY:
        return ray.get([e.generate.remote(prompts, sampling_params, use_tqdm=False) for e in engines])
    return [e.generate(prompts, sampling_params, use_tqdm=False) for e in engines]


def engine_collective(engine, method, args=(), kwargs=None, timeout=None):
    if USE_RAY:
        kw = {}
        if timeout is not None:
            kw["timeout"] = timeout
        if args:
            kw["args"] = args
        if kwargs:
            kw["kwargs"] = kwargs
        return _unwrap(ray.get(engine.collective_rpc.remote(method, **kw)))
    return _unwrap(engine.collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs))


def engines_collective(engines, method, args_list):
    if USE_RAY:
        results = ray.get([e.collective_rpc.remote(method, args=a) for e, a in zip(engines, args_list)])
        return [_unwrap(r) for r in results]
    return [_unwrap(e.collective_rpc(method, args=a)) for e, a in zip(engines, args_list)]


def call_all_engines(engines, method, args=()):
    return engines_collective(engines, method, [args] * len(engines))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.strip().lower() in {"1", "true", "t", "yes", "y"}:
        return True
    if v.strip().lower() in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def _floatlist(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _intlist(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="gsm8k", choices=list_datasets())
    p.add_argument("--train_data_path", default=None)
    p.add_argument("--test_data_path", default=None)
    p.add_argument("--soup_train_samples", type=int, default=500)
    p.add_argument("--test_samples", type=int, default=None)
    p.add_argument("--probe_start_index", type=int, default=-1)
    p.add_argument("--probe_samples", type=int, default=64)

    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--precision", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--max_tokens", type=int, default=None)

    p.add_argument("--sigma_values", default="0.0001,0.0005,0.001,0.002,0.005,0.01")
    p.add_argument("--population_size", type=int, default=30)
    p.add_argument("--num_engines", type=int, default=4)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cuda_devices", default="0,1,2,3")
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default="corp-outputs")
    p.add_argument("--resume_dir", default=None)

    p.add_argument("--alpha_grid", default="0.5,1,2,4,8,16")
    p.add_argument("--q_grid", default="0.5,0.7,0.9")
    p.add_argument("--beta_grid", default="0.5,1,2,5,10,20,50")

    p.add_argument("--recenter_attempts", type=int, default=3)
    p.add_argument("--recenter_population_size", type=int, default=None)
    p.add_argument("--recenter_rank", type=int, default=8)
    p.add_argument("--max_accepted_recenters", type=int, default=2)
    p.add_argument("--recenter_iso_lambda", type=float, default=0.5)
    p.add_argument("--recenter_sigma_up", type=float, default=1.25)
    p.add_argument("--recenter_sigma_down", type=float, default=0.5)
    p.add_argument("--recenter_accept_epsilon", type=float, default=0.0)
    p.add_argument("--recenter_patience", type=int, default=2)

    p.add_argument("--stage2_regression_lambda", type=float, default=1.0)
    p.add_argument("--stage2_regression_lambda_final", type=float, default=None)
    p.add_argument("--stage2_target_ess_initial", type=float, default=16.0)
    p.add_argument("--stage2_target_ess_final", type=float, default=4.0)
    p.add_argument("--stage2_gate_lcb_z", type=float, default=1.0)

    p.add_argument("--consolidation_method", choices=["corp", "stage1_only"], default="corp")
    p.add_argument("--consolidation_budget", choices=["fast", "sweep"], default="fast")
    p.add_argument("--reuse_resume_stage1", type=_str2bool, default=True)

    args = p.parse_args()
    args.sigma_list = _floatlist(args.sigma_values)
    args.scale_grid_list = _floatlist(args.alpha_grid)
    args.beta_grid_list = _floatlist(args.beta_grid)
    args.q_grid_list = _floatlist(args.q_grid)
    if args.recenter_population_size is None:
        args.recenter_population_size = min(512, args.population_size)
    if args.stage2_regression_lambda_final is None:
        args.stage2_regression_lambda_final = args.stage2_regression_lambda
    if not str(args.cuda_devices).strip():
        args.cuda_devices = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_slice(handler, path, split, max_samples=None, start_index=0):
    try:
        return handler.load_data(path, split=split, max_samples=max_samples, start_index=start_index)
    except TypeError:
        data = handler.load_data(path, split=split, max_samples=None)
        end = None if max_samples is None else start_index + max_samples
        return data[start_index:end]


def load_data_splits(handler, args):
    train_path = args.train_data_path or handler.default_train_path
    test_path = args.test_data_path or handler.default_test_path
    support_end = int(args.soup_train_samples)

    print(f"Loading {handler.name} data...")
    if train_path == test_path:
        all_data = _load_slice(handler, train_path, split="train", max_samples=None)
        effective_support, effective_probe = corp_ops.resolve_shared_support_probe_budgets(
            len(all_data), support_end, args.probe_samples
        )
        probe_start = effective_support if args.probe_start_index < 0 else max(args.probe_start_index, effective_support)
        probe_end = min(probe_start + effective_probe, len(all_data))
        support_datas = all_data[:effective_support]
        probe_datas = all_data[probe_start:probe_end]
        test_datas = all_data[probe_end:] if args.test_samples is None else all_data[probe_end:probe_end + args.test_samples]
    else:
        support_datas = _load_slice(handler, train_path, "train", max_samples=support_end)
        probe_start = support_end if args.probe_start_index < 0 else max(args.probe_start_index, support_end)
        probe_datas = _load_slice(handler, train_path, "train", max_samples=args.probe_samples, start_index=probe_start)
        test_datas = _load_slice(handler, test_path, "test", max_samples=args.test_samples)

    print(f"  Support: {len(support_datas)} | Probe: {len(probe_datas)} | Test: {len(test_datas)}")
    return support_datas, probe_datas, test_datas


def build_prompt_formatter(model_name, tokenizer):
    is_instruct = any(t in model_name.lower() for t in ["instruct", "chat", "it"])

    def format_prompt(messages):
        if is_instruct and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return NEWLINE.join(m["content"] for m in messages) + NEWLINE

    return format_prompt


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_outputs(handler, outputs, datas):
    accuracy = float(handler.postprocess_outputs(outputs, datas) * 100.0)
    return {"accuracy": accuracy, "correct": int(round(accuracy / 100.0 * len(datas)))}


def _eval_outputs_with_details(handler, outputs, datas, include_responses=False):
    metrics = _eval_outputs(handler, outputs, datas)
    metrics["example_rows"] = eval_ops.build_example_rows(handler, outputs, datas, include_responses=include_responses)
    return metrics


def _eval_model(engine, handler, prompts, datas, sampling_params):
    return _eval_outputs(handler, engine_generate(engine, prompts, sampling_params), datas)


def _eval_model_with_details(engine, handler, prompts, datas, sampling_params, include_responses=False):
    outputs = engine_generate(engine, prompts, sampling_params)
    return _eval_outputs_with_details(handler, outputs, datas, include_responses=include_responses)


def _eval_direction(engine, handler, prompts, datas, sampling_params, direction_spec, step_size, *, with_details=False):
    scaled = corp_ops.scale_direction_spec(direction_spec, float(step_size))
    engine_collective(engine, "apply_direction_from_base", args=(corp_ops.serialize_direction_spec(scaled),))
    outputs = engine_generate(engine, prompts, sampling_params)
    metrics = _eval_outputs_with_details(handler, outputs, datas) if with_details else _eval_outputs(handler, outputs, datas)
    engine_collective(engine, "reset_to_base_weights")
    return metrics


def _commit_direction(engines, direction_spec):
    serialized = corp_ops.serialize_direction_spec(direction_spec)
    return call_all_engines(engines, "commit_direction_from_base", args=(serialized,))


def _score_population_with_details(engines, handler, prompts, datas, direction_specs, sampling_params, label):
    if not direction_specs:
        return []
    runs = []
    total_batches = (len(direction_specs) + len(engines) - 1) // len(engines)
    print(f"Scoring {label}: {len(direction_specs)} candidates")
    for batch_idx in range(total_batches):
        start = batch_idx * len(engines)
        end = min(start + len(engines), len(direction_specs))
        batch_specs = direction_specs[start:end]
        eng_subset = engines[:len(batch_specs)]
        serialized_batch = [(corp_ops.serialize_direction_spec(s),) for s in batch_specs]
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  batch {batch_idx + 1}/{total_batches}", flush=True)
        engines_collective(eng_subset, "apply_direction_from_base", serialized_batch)
        outputs = engines_generate(eng_subset, prompts, sampling_params)
        call_all_engines(eng_subset, "reset_to_base_weights")
        for spec, output in zip(batch_specs, outputs):
            metrics = _eval_outputs_with_details(handler, output, datas)
            runs.append({
                "direction_spec": spec,
                "mean_rss": float(corp_ops.direction_rss_norm(spec)),
                "accuracy": float(metrics["accuracy"]),
                "correct": int(metrics["correct"]),
                "example_rows": metrics["example_rows"],
            })
        del outputs
        gc.collect()
    return runs


# ---------------------------------------------------------------------------
# Isotropic sampling
# ---------------------------------------------------------------------------

def run_sampling(args, engines, handler, support_prompts, support_datas, sampling_params):
    print("\n" + "=" * 60 + "\nISOTROPIC CANDIDATE DISCOVERY\n" + "=" * 60)
    print(f"Budget: {args.population_size} | Support: {len(support_datas)} | Sigmas: {args.sigma_list}")

    rng = np.random.default_rng(seed=args.global_seed)
    all_seeds = rng.choice(2**31, size=args.population_size, replace=False).tolist()
    all_sigmas = rng.choice(args.sigma_list, size=args.population_size).tolist()
    perf: Dict[Tuple[int, float], float] = {}
    seed_idx = 0
    evaluated = 0
    batch_num = 0

    while evaluated < args.population_size:
        n = min(args.num_engines, args.population_size - evaluated)
        batch = [(all_seeds[seed_idx + i], all_sigmas[seed_idx + i]) for i in range(n)]
        seed_idx += n
        eng_sub = engines[:n]
        engines_collective(eng_sub, "perturb_self_weights", [(int(s), sig, False) for s, sig in batch])
        outputs = engines_generate(eng_sub, support_prompts, sampling_params)
        engines_collective(eng_sub, "restore_self_weights", [(int(s), sig, False) for s, sig in batch])
        rewards = []
        for i, (seed, sigma) in enumerate(batch):
            r = float(np.mean(handler.compute_rewards(outputs[i], support_datas)))
            perf[(seed, sigma)] = r
            rewards.append(r)
        evaluated += n
        batch_num += 1
        print(f"  batch {batch_num} | {evaluated}/{args.population_size} | {['%.3f' % r for r in rewards]}")

    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (_, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)
    best_sigma = max(args.sigma_list, key=lambda s: np.mean(sigma_rewards[s]) if sigma_rewards[s] else 0.0)
    print(f"Best sigma: {best_sigma}")
    return perf, best_sigma


# ---------------------------------------------------------------------------
# Stage 1 — constructive collapse
# ---------------------------------------------------------------------------

def _serializable_merge_fields(merge_info):
    return {k: merge_info[k] for k in (
        "tail_size", "reward_threshold", "tail_reward_mean", "tail_reward_min", "tail_reward_max",
        "pass1_weight_entropy", "pass1_effective_sample_size", "weight_entropy", "effective_sample_size",
        "max_weight", "top5_weight_mass", "top10_weight_mass", "provisional_mean_rss",
        "mean_rss", "merged_rss", "trace_cov", "effective_rank", "top_eigvals",
        "subspace_rank", "kappa", "num_direction_components", "basis_size",
        "alignment_stats", "alignment_score_stats", "dispersion_stats", "dispersion_score_stats",
    )}


def _collapse_state(best):
    return {
        "q": float(best.get("q", 0.0)),
        "beta": float(best.get("beta", 0.0)),
        "direction_spec": best["direction_spec"],
        "mean_spec": best["mean_spec"],
        "principal_directions": best["principal_directions"],
        "principal_eigvals": best["principal_eigvals"],
        "kappa": float(best["kappa"]),
        "trace_cov": float(best["trace_cov"]),
        "effective_rank": float(best["effective_rank"]),
        "top_eigvals": list(best["top_eigvals"]),
        "subspace_rank": int(best["subspace_rank"]),
        "mean_rss": float(best["mean_rss"]),
        "elite_rows": [dict(r) for r in best["elite_rows"]],
        "pass1_weights": [float(w) for w in best.get("pass1_weights", best.get("weights", []))],
        "pass2_weights": [float(w) for w in best.get("pass2_weights", best.get("weights", []))],
        "elite_direction_specs": list(best["elite_direction_specs"]),
    }


def _run_collapse_sweep(engine, handler, probe_prompts, probe_datas, sampling_params,
                        direction_specs, rewards, q_grid, beta_grid, scale_grid, recenter_rank,
                        label, tuning_path=None):
    tuning_rows = []
    best = None
    total = len(q_grid) * len(beta_grid) * len(scale_grid)
    print(f"  {label}: {total} probe evals ({len(q_grid)} q × {len(beta_grid)} beta × {len(scale_grid)} scale)")

    for q_i, q in enumerate(q_grid, 1):
        print(f"  q sweep {q_i}/{len(q_grid)} | q={q}")
        best_probe_q = None
        best_tail_q = None
        for beta in beta_grid:
            merge = collapse_ops.build_compressibility_aware_merge(direction_specs, rewards, q, beta, rank=recenter_rank)
            for scale in scale_grid:
                eta = float(scale) * float(merge["mean_rss"])
                metrics = _eval_direction(engine, handler, probe_prompts, probe_datas, sampling_params, merge["direction_spec"], eta)
                record = {
                    "q": float(q), "beta": float(beta), "scale": float(scale), "eta": float(eta),
                    **_serializable_merge_fields(merge),
                    "probe_accuracy": float(metrics["accuracy"]),
                    "probe_correct": int(metrics["correct"]),
                }
                tuning_rows.append(record)
                if best is None or metrics["accuracy"] > best["probe_accuracy"]:
                    best = {**record, "direction_spec": merge["direction_spec"], "mean_spec": merge["mean_spec"],
                            "principal_directions": merge["principal_directions"], "principal_eigvals": merge["principal_eigvals"],
                            "elite_rows": merge["elite_rows"], "pass1_weights": merge["pass1_weights"],
                            "pass2_weights": merge["pass2_weights"],
                            "elite_direction_specs": [direction_specs[idx] for idx in merge["elite_indices"]]}
                if best_probe_q is None or metrics["accuracy"] > best_probe_q:
                    best_probe_q = metrics["accuracy"]
                    best_tail_q = merge["tail_size"]
        print(f"  q={q}: tail≈{best_tail_q}/{len(direction_specs)} | best probe={best_probe_q:.2f}%")

    if tuning_path is not None:
        corp_ops.save_json(tuning_path, tuning_rows)
    return best, tuning_rows


def run_stage1(engine, handler, probe_prompts, probe_datas, test_prompts, test_datas,
               sampling_params, direction_specs, rewards, q_grid, beta_grid, scale_grid,
               base_test, artifacts_dir, recenter_rank):
    print("\n" + "=" * 60 + "\nSTAGE 1 — CONSTRUCTIVE COLLAPSE\n" + "=" * 60)
    print(f"  N={len(direction_specs)} | Q={list(q_grid)} | Beta={list(beta_grid)} | Scale={list(scale_grid)}")

    best, _ = _run_collapse_sweep(
        engine, handler, probe_prompts, probe_datas, sampling_params,
        direction_specs, rewards, q_grid, beta_grid, scale_grid, recenter_rank,
        label="Stage 1",
        tuning_path=os.path.join(artifacts_dir, "stage1_tuning.json"),
    )
    test_metrics = _eval_direction(engine, handler, test_prompts, test_datas, sampling_params, best["direction_spec"], best["eta"])

    # top-1 reference (best individual candidate)
    best_idx = int(np.argmax(np.asarray(rewards, dtype=np.float64)))
    top1_spec = direction_specs[best_idx]
    top1_best = None
    for scale in scale_grid:
        eta = float(scale) * float(corp_ops.direction_rss_norm(top1_spec))
        m = _eval_direction(engine, handler, probe_prompts, probe_datas, sampling_params, top1_spec, eta)
        if top1_best is None or m["accuracy"] > top1_best["probe_accuracy"]:
            top1_best = {"scale": float(scale), "eta": float(eta), "probe_accuracy": float(m["accuracy"])}
    top1_test = _eval_direction(engine, handler, test_prompts, test_datas, sampling_params, top1_spec, top1_best["eta"])

    result = {
        "q": best["q"], "beta": best["beta"], "scale": best["scale"], "eta": best["eta"],
        **{k: best[k] for k in ("tail_size", "reward_threshold", "weight_entropy", "effective_sample_size",
                                "kappa", "trace_cov", "effective_rank", "top_eigvals", "subspace_rank",
                                "mean_rss", "merged_rss", "num_direction_components")},
        "probe_accuracy": float(best["probe_accuracy"]),
        "probe_correct": int(best["probe_correct"]),
        "accuracy": float(test_metrics["accuracy"]),
        "correct": int(test_metrics["correct"]),
        "delta_vs_base": float(test_metrics["accuracy"] - base_test * 100.0),
    }
    collapse_st = _collapse_state(best)

    corp_ops.save_json(os.path.join(artifacts_dir, "stage1_summary.json"), {
        "best_result": result,
        "top1_reference": {
            "scale": top1_best["scale"], "eta": top1_best["eta"],
            "probe_accuracy": top1_best["probe_accuracy"],
            "accuracy": float(top1_test["accuracy"]),
            "delta_vs_base": float(top1_test["accuracy"] - base_test * 100.0),
        },
        "population_size": len(direction_specs),
    })
    corp_ops.save_json(os.path.join(artifacts_dir, "stage1_elite_cloud.json"), {
        "q": result["q"], "beta": result["beta"], "eta": result["eta"],
        "kappa": result["kappa"], "trace_cov": result["trace_cov"],
        "elite_rows": best["elite_rows"],
    })
    print(f"  Best: q={result['q']}, beta={result['beta']}, scale={result['scale']:.2g}, "
          f"tail={result['tail_size']}/{len(direction_specs)}")
    print(f"  probe={result['probe_accuracy']:.2f}%  test={result['accuracy']:.2f}%  "
          f"(Δ={result['delta_vs_base']:+.2f}%,  κ={result['kappa']:.4f})")
    return result, collapse_st


# ---------------------------------------------------------------------------
# Stage 2 — adaptive constructive recenter
# ---------------------------------------------------------------------------

def _binary_scores(example_rows):
    return np.asarray([1 if float(r.get("example_score", 0.0)) > 0.5 else 0 for r in example_rows], dtype=np.int8)


def _effect_profile(center_scores, candidate_rows, *, regression_lambda):
    total = int(center_scores.size)
    if total <= 0:
        return {"fixes": 0, "regressions": 0, "net_gain": 0.0, "constructive_score": 0.0, "constructive_score_raw": 0.0}
    cand = _binary_scores(candidate_rows)
    fixes = int(np.sum((center_scores == 0) & (cand == 1)))
    regressions = int(np.sum((center_scores == 1) & (cand == 0)))
    raw = float(fixes - float(regression_lambda) * regressions)
    return {
        "fixes": fixes, "regressions": regressions,
        "retained_correct": int(np.sum((center_scores == 1) & (cand == 1))),
        "still_wrong": int(np.sum((center_scores == 0) & (cand == 0))),
        "net_gain": float((fixes - regressions) / total),
        "constructive_score": float(raw / total),
        "constructive_score_raw": float(raw),
        "delta_pp": float(100.0 * (fixes - regressions) / total),
    }


def _paired_delta_summary(center_scores, candidate_rows, *, lcb_z):
    total = int(center_scores.size)
    if total <= 0:
        return {"delta_mean": 0.0, "delta_lcb": 0.0, "delta_lcb_pp": 0.0}
    cand = _binary_scores(candidate_rows)
    deltas = cand.astype(np.float64) - center_scores.astype(np.float64)
    dmean = float(deltas.mean())
    dse = float(deltas.std(ddof=1) / math.sqrt(float(total))) if total > 1 else 0.0
    return {"delta_mean": dmean, "delta_lcb": dmean - float(lcb_z) * dse, "delta_lcb_pp": 100.0 * (dmean - float(lcb_z) * dse)}


def _stage2_progress(args, round_idx):
    total = max(int(args.recenter_attempts), 1)
    return float(np.clip((int(round_idx) - 1) / float(total - 1), 0.0, 1.0)) if total > 1 else 1.0


def _stage2_lambda(args, round_idx):
    t = _stage2_progress(args, round_idx)
    return float((1.0 - t) * args.stage2_regression_lambda + t * args.stage2_regression_lambda_final)


def _stage2_ess(args, round_idx):
    t = _stage2_progress(args, round_idx)
    start = max(float(args.stage2_target_ess_initial), 1.0)
    end = max(float(args.stage2_target_ess_final), 1.0)
    return float(math.exp((1.0 - t) * math.log(start) + t * math.log(end)))


def _crossfit_split(total, seed):
    if total <= 1:
        idx = list(range(total))
        return idx, idx
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(total)
    mid = total // 2
    sel = sorted(int(i) for i in perm[:mid].tolist())
    gate = sorted(int(i) for i in perm[mid:].tolist())
    return (sel, gate) if sel and gate else (list(range(total)), list(range(total)))


def _subset(items, indices):
    return [items[i] for i in indices]


def _run_recenter_proposal_sweep(
    args, *, engine, handler, select_prompts, select_datas, center_select_scores,
    gate_prompts, gate_datas, center_gate_scores, sampling_params,
    direction_specs, candidate_rows, eta_grid, round_regression_lambda,
):
    """Sweep (q, beta, eta) proposals; select best by gate-pass constructive criterion."""
    support_rewards = [float(r["support_reward"]) for r in candidate_rows]
    proposals = []

    for q in args.q_grid_list:
        for beta in args.beta_grid_list:
            merge = collapse_ops.build_compressibility_aware_merge(direction_specs, support_rewards, q, beta, rank=args.recenter_rank)
            for eta in eta_grid:
                sel_m = _eval_direction(engine, handler, select_prompts, select_datas, sampling_params, merge["direction_spec"], eta, with_details=True)
                sel_eff = _effect_profile(center_select_scores, sel_m["example_rows"], regression_lambda=round_regression_lambda)
                sel_delta = _paired_delta_summary(center_select_scores, sel_m["example_rows"], lcb_z=args.stage2_gate_lcb_z)
                gate_m = _eval_direction(engine, handler, gate_prompts, gate_datas, sampling_params, merge["direction_spec"], eta, with_details=True)
                gate_eff = _effect_profile(center_gate_scores, gate_m["example_rows"], regression_lambda=round_regression_lambda)
                gate_delta = _paired_delta_summary(center_gate_scores, gate_m["example_rows"], lcb_z=args.stage2_gate_lcb_z)
                gate_pass = bool(gate_eff["constructive_score"] > 0.0 and gate_delta["delta_lcb"] > 0.0)
                proposals.append({
                    "q": float(q), "beta": float(beta), "eta": float(eta),
                    "direction_spec": merge["direction_spec"], "mean_spec": merge["mean_spec"],
                    "principal_directions": merge["principal_directions"], "principal_eigvals": merge["principal_eigvals"],
                    "elite_direction_specs": [direction_specs[i] for i in merge["elite_indices"]],
                    **_serializable_merge_fields(merge),
                    "select_constructive_score": float(sel_eff["constructive_score"]),
                    "select_fixes": int(sel_eff["fixes"]), "select_regressions": int(sel_eff["regressions"]),
                    "select_delta_lcb_pp": float(sel_delta["delta_lcb_pp"]),
                    "gate_constructive_score": float(gate_eff["constructive_score"]),
                    "gate_fixes": int(gate_eff["fixes"]), "gate_regressions": int(gate_eff["regressions"]),
                    "gate_delta_lcb_pp": float(gate_delta["delta_lcb_pp"]),
                    "gate_reality_pass": gate_pass,
                })

    if not proposals:
        return None

    def key(r):
        return (float(r["select_constructive_score"]), float(r["select_fixes"]) - float(r["select_regressions"]))

    sorted_props = sorted(proposals, key=key, reverse=True)
    gate_pass_props = [r for r in sorted_props if r["gate_reality_pass"]]
    return gate_pass_props[0] if gate_pass_props else sorted_props[0]


def run_stage2(args, engines, handler, support_prompts, support_datas, probe_prompts, probe_datas,
               test_prompts, test_datas, sampling_params, base_test, initial_result, initial_state,
               initial_sigma, artifacts_dir):
    print("\n" + "=" * 60 + "\nSTAGE 2 — ADAPTIVE CONSTRUCTIVE RECENTER\n" + "=" * 60)
    accept_threshold = float(args.recenter_accept_epsilon)
    probe_alpha_grid = [0.5, 1.0, 2.0] if args.consolidation_budget == "fast" else [0.5, 1.0, 2.0, 4.0]

    current_center_spec = corp_ops.scale_direction_spec(initial_state["direction_spec"], initial_result["eta"])
    current_state = dict(initial_state)
    search_state = dict(initial_state)
    current_summary = dict(initial_result)
    current_eta = float(initial_result["eta"])
    current_sigma = float(initial_sigma)
    current_probe_accuracy = float(initial_result["probe_accuracy"])
    current_probe_correct = int(initial_result["probe_correct"])
    search_iso_lambda = float(args.recenter_iso_lambda)
    accepted_rounds = 0
    consecutive_rejects = 0

    # Evaluate current center on all splits
    full_support_details = _eval_model_with_details(engines[0], handler, support_prompts, support_datas, sampling_params)
    current_probe_details = _eval_model_with_details(engines[0], handler, probe_prompts, probe_datas, sampling_params) if probe_datas else {
        "accuracy": current_probe_accuracy, "correct": current_probe_correct, "example_rows": []
    }
    sel_indices, gate_indices = _crossfit_split(len(support_datas), args.global_seed + 17)
    sel_prompts = _subset(support_prompts, sel_indices)
    sel_datas = _subset(support_datas, sel_indices)
    gate_prompts = _subset(support_prompts, gate_indices)
    gate_datas = _subset(support_datas, gate_indices)
    center_sel_details = _eval_model_with_details(engines[0], handler, sel_prompts, sel_datas, sampling_params)
    center_gate_details = _eval_model_with_details(engines[0], handler, gate_prompts, gate_datas, sampling_params)

    print(f"  local_pop={args.recenter_population_size} | rank={args.recenter_rank} | iso_lambda={search_iso_lambda:.3g} | "
          f"support_split={len(sel_datas)}/{len(gate_datas)} | probe_alpha={probe_alpha_grid}")
    print(f"  stage1 support={full_support_details['accuracy']:.2f}% | probe={current_probe_accuracy:.2f}% | "
          f"test={initial_result['accuracy']:.2f}%")

    if args.recenter_attempts <= 0 or args.max_accepted_recenters <= 0:
        return _build_stage2_result(current_summary, base_test=base_test, sigma=current_sigma,
                                    state=current_state, accepted=0, completed=0, applied=False,
                                    reason="stage2_disabled"), current_center_spec

    round_rows = []
    for round_idx in range(1, int(args.recenter_attempts) + 1):
        sigma_before = float(current_sigma)
        probe_before = float(current_probe_accuracy)
        eta_grid = corp_ops.build_local_eta_grid(current_eta)
        round_lambda = _stage2_lambda(args, round_idx)
        search_target_ess = _stage2_ess(args, round_idx)
        search_iso_before = float(search_iso_lambda)

        print(f"  round {round_idx}/{args.recenter_attempts} | sigma={sigma_before:.6g} | eta={current_eta:.6g} | "
              f"kappa={current_state.get('kappa', 0.0):.4f} | ess={search_target_ess:.2f} | lambda={round_lambda:.3g}", flush=True)

        # Sample low-rank local population
        local_specs, proposal_meta = recenter_ops.sample_low_rank_population(
            search_state.get("principal_directions", []),
            search_state.get("principal_eigvals", []),
            sigma=sigma_before,
            population_size=int(args.recenter_population_size),
            iso_lambda=search_iso_before,
            rank=int(args.recenter_rank),
            seed=int(args.global_seed + 1000 + round_idx),
        )

        # Score local candidates (with per-example detail for constructive scoring)
        local_runs = _score_population_with_details(engines, handler, sel_prompts, sel_datas, local_specs, sampling_params,
                                                    label=f"round {round_idx} selection")
        sel_scores = _binary_scores(center_sel_details["example_rows"])
        candidate_rows = []
        for ci, (run, meta) in enumerate(zip(local_runs, proposal_meta)):
            eff = _effect_profile(sel_scores, run["example_rows"], regression_lambda=round_lambda)
            candidate_rows.append({
                "candidate_index": ci, "seed": int(meta["seed"]),
                "support_accuracy": float(run["accuracy"]),
                "support_reward": float(run["accuracy"] / 100.0),
                "mean_rss": float(run["mean_rss"]),
                **eff,
            })

        # Update search distribution
        search_update = recenter_ops.build_search_update(
            local_specs, candidate_rows,
            current_iso_lambda=search_iso_before, target_ess=search_target_ess, rank=int(args.recenter_rank),
        )
        if search_update["merge_info"] is not None:
            search_state = _collapse_state(search_update["merge_info"])
        search_iso_lambda = float(search_update["iso_lambda_after"])

        # Sweep proposals on select+gate splits
        gate_scores = _binary_scores(center_gate_details["example_rows"])
        local_best = _run_recenter_proposal_sweep(
            args, engine=engines[0], handler=handler,
            select_prompts=sel_prompts, select_datas=sel_datas, center_select_scores=sel_scores,
            gate_prompts=gate_prompts, gate_datas=gate_datas, center_gate_scores=gate_scores,
            sampling_params=sampling_params, direction_specs=local_specs,
            candidate_rows=candidate_rows, eta_grid=eta_grid, round_regression_lambda=round_lambda,
        )

        accepted = False
        rejection_reason = None
        if local_best is None:
            rejection_reason = "no_local_proposal"
            consecutive_rejects += 1
            current_sigma = float(max(sigma_before * args.recenter_sigma_down, 1e-8))
            print("    rejected | no valid proposal", flush=True)
        else:
            if not local_best["gate_reality_pass"]:
                rejection_reason = "crossfit_gate_failed"
                consecutive_rejects += 1
                current_sigma = float(max(sigma_before * args.recenter_sigma_down, 1e-8))
                print(f"    rejected | gate failed | select_cs={local_best['select_constructive_score']:.4f}", flush=True)
            else:
                # Probe line search on probe split
                probe_run = eval_ops.probe_line_search(
                    engines[0], handler, probe_prompts, probe_datas, sampling_params,
                    direction_spec=local_best["direction_spec"],
                    mean_rss=local_best["mean_rss"],
                    search_grid=probe_alpha_grid,
                    eval_fn=lambda *a, **kw: _eval_direction(*a, **kw, with_details=True),
                    eta_reference=local_best["eta"],
                )
                probe_delta_summary = _paired_delta_summary(
                    _binary_scores(current_probe_details["example_rows"]),
                    probe_run["example_rows"],
                    lcb_z=args.stage2_gate_lcb_z,
                )
                probe_reality_pass = bool(
                    probe_delta_summary["delta_lcb"] > 0.0
                    and float(probe_run["probe_accuracy"]) >= probe_before + accept_threshold
                )
                if probe_reality_pass:
                    final_run = eval_ops.finalize_on_test(
                        engines[0], handler, test_prompts, test_datas, sampling_params, probe_run,
                        eval_fn=lambda *a, **kw: _eval_direction(*a, **kw, with_details=True),
                    )
                    deployed_delta = corp_ops.scale_direction_spec(probe_run["direction_spec"], float(probe_run["eta"]))
                    _commit_direction(engines, deployed_delta)
                    current_center_spec = corp_ops.combine_direction_specs([current_center_spec, deployed_delta])

                    # Re-evaluate center on all sub-splits after accept
                    full_support_details = _eval_model_with_details(engines[0], handler, support_prompts, support_datas, sampling_params)
                    center_sel_details = _eval_model_with_details(engines[0], handler, sel_prompts, sel_datas, sampling_params)
                    center_gate_details = _eval_model_with_details(engines[0], handler, gate_prompts, gate_datas, sampling_params)
                    current_probe_details = _eval_model_with_details(engines[0], handler, probe_prompts, probe_datas, sampling_params)

                    current_eta = float(probe_run["eta"])
                    current_probe_accuracy = float(probe_run["probe_accuracy"])
                    current_probe_correct = int(probe_run["probe_correct"])
                    current_state = search_state  # inherit new subspace
                    accepted_support_eff = _effect_profile(sel_scores, center_sel_details["example_rows"], regression_lambda=round_lambda)
                    current_summary = {
                        "q": float(local_best["q"]), "beta": float(local_best["beta"]),
                        "eta": float(probe_run["eta"]), "alpha": float(probe_run["alpha"]),
                        "probe_accuracy": float(probe_run["probe_accuracy"]),
                        "probe_correct": int(probe_run["probe_correct"]),
                        "accuracy": float(final_run["accuracy"]),
                        "correct": int(final_run["correct"]),
                        "selected_population_sigma": float(sigma_before),
                        "selected_local_q": float(local_best["q"]),
                        "selected_local_beta": float(local_best["beta"]),
                        "selected_gate_delta_lcb_pp": float(local_best["gate_delta_lcb_pp"]),
                        "selected_probe_delta_lcb_pp": float(probe_delta_summary["delta_lcb_pp"]),
                        "selected_local_constructive_score": float(accepted_support_eff["constructive_score"]),
                        "selected_local_fixes": int(accepted_support_eff["fixes"]),
                        "selected_local_regressions": int(accepted_support_eff["regressions"]),
                    }
                    accepted_rounds += 1
                    consecutive_rejects = 0
                    accepted = True
                    print(f"    accepted | probe={probe_run['probe_accuracy']:.2f}% | test={final_run['accuracy']:.2f}%", flush=True)
                else:
                    rejection_reason = "probe_not_better"
                    current_sigma = float(max(sigma_before * args.recenter_sigma_down, 1e-8))
                    consecutive_rejects += 1
                    print(f"    rejected | probe={probe_run['probe_accuracy']:.2f}% | "
                          f"lcb={probe_delta_summary['delta_lcb_pp']:.2f}pp | current={probe_before:.2f}%", flush=True)

        round_rows.append({"round": round_idx, "accepted": accepted, "rejection_reason": rejection_reason,
                           "sigma_before": sigma_before, "sigma_after": current_sigma,
                           "probe_before": probe_before, "probe_after": current_probe_accuracy})

        stop = recenter_ops.compute_stop_reason(
            accepted=accepted, accepted_recenters=accepted_rounds,
            max_accepted_recenters=args.max_accepted_recenters,
            consecutive_rejects=consecutive_rejects, recenter_patience=args.recenter_patience,
            round_idx=round_idx, recenter_attempts=args.recenter_attempts,
        )
        if stop is not None:
            print(f"  stopping: {stop}")
            break

    corp_ops.save_json(os.path.join(artifacts_dir, "recenter_rounds.json"), {"rounds": round_rows})

    abstain_reason = None
    if accepted_rounds <= 0:
        abstain_reason = round_rows[-1].get("rejection_reason") if round_rows else "no_rounds"
    return _build_stage2_result(current_summary, base_test=base_test, sigma=current_sigma,
                                state=current_state, accepted=accepted_rounds, completed=len(round_rows),
                                applied=accepted_rounds > 0, reason=abstain_reason), current_center_spec


def _build_stage2_result(current_result, *, base_test, sigma, state, accepted, completed, applied, reason=None):
    out = {
        "stage2_applied": bool(applied),
        "accepted_rounds": int(accepted),
        "rounds_completed": int(completed),
        "q": current_result.get("q", 0.0),
        "beta": current_result.get("beta", 0.0),
        "eta": current_result.get("eta", 0.0),
        "alpha": current_result.get("alpha", 0.0),
        "probe_accuracy": float(current_result.get("probe_accuracy", 0.0)),
        "probe_correct": int(current_result.get("probe_correct", 0)),
        "accuracy": float(current_result.get("accuracy", 0.0)),
        "correct": int(current_result.get("correct", 0)),
        "delta_vs_base": float(current_result.get("accuracy", 0.0) - base_test * 100.0),
        "final_sigma": float(sigma),
        "basis_rank": int(state.get("subspace_rank", 0) or 0),
        "kappa": float(state.get("kappa", current_result.get("kappa", 0.0))),
        **{k: current_result.get(k) for k in (
            "selected_population_sigma", "selected_local_q", "selected_local_beta",
            "selected_gate_delta_lcb_pp", "selected_probe_delta_lcb_pp",
            "selected_local_constructive_score", "selected_local_fixes", "selected_local_regressions",
        )},
    }
    if reason is not None:
        out["stage2_abstain_reason"] = reason
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    start_time = time.time()
    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens
    is_resume = args.resume_dir is not None

    print("=" * 60)
    print(f"Corp — {handler.name.upper()} {'[RESUME]' if is_resume else ''}")
    print("=" * 60)
    print(f"Model:   {args.model_name}")
    print(f"Pop:     {args.population_size} | Engines: {args.num_engines} | TP: {args.tp}")
    print(f"Method:  {args.consolidation_method} | Budget: {args.consolidation_budget}")

    global USE_RAY

    def ensure_ray(n):
        global USE_RAY
        USE_RAY = not (int(n) == 1 and args.tp == 1)
        if USE_RAY:
            if ray is None:
                raise ModuleNotFoundError("ray not installed. Use --num_engines 1 --tp 1 for local mode.")
            if not ray.is_initialized():
                addr = "auto" if os.environ.get("RAY_ADDRESS") else "local"
                ray.init(address=addr, ignore_reinit_error=True)

    if is_resume:
        base_model_path, base_train_from_pop, best_sigma, perf = artifact_ops.load_sampling_population(args.resume_dir)
        previous_results = artifact_ops.load_resume_results(args.resume_dir, base_train=base_train_from_pop)
        base_train = float(previous_results["base_train_accuracy"])
        base_test = float(previous_results["base_test_accuracy"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging_dir = os.path.join(args.experiment_dir, f"{args.dataset}_resume_{timestamp}")
    else:
        base_model_path = args.model_name
        previous_results = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging_dir = os.path.join(args.experiment_dir, f"{args.dataset}_{timestamp}")

    model_saves_dir = os.path.join(logging_dir, "model_saves")
    artifacts_dir = os.path.join(logging_dir, "artifacts")
    os.makedirs(model_saves_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(logging_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    support_datas, probe_datas, test_datas = load_data_splits(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    format_prompt = build_prompt_formatter(base_model_path, tokenizer)
    support_prompts = [format_prompt(item["messages"]) for item in support_datas]
    probe_prompts = [format_prompt(item["messages"]) for item in probe_datas]
    test_prompts = [format_prompt(item["messages"]) for item in test_datas]
    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)

    ensure_ray(args.num_engines)
    engines, pgs = launch_engines(
        args.num_engines, base_model_path,
        precision=args.precision, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization, use_ray=USE_RAY,
    )
    try:
        if not is_resume:
            # Base model evaluation
            print("\n" + "=" * 60 + "\nBASE MODEL EVALUATION\n" + "=" * 60)
            support_out = engine_generate(engines[0], support_prompts, sampling_params)
            base_train = handler.postprocess_outputs(support_out, support_datas)
            print(f"Support: {base_train * 100:.2f}%")
            test_out = engine_generate(engines[0], test_prompts, sampling_params)
            base_test = handler.postprocess_outputs(test_out, test_datas)
            print(f"Test:    {base_test * 100:.2f}%")

            perf, best_sigma = run_sampling(args, engines, handler, support_prompts, support_datas, sampling_params)
            artifact_ops.save_sampling_population(model_saves_dir, base_model_path, base_train, best_sigma, perf)
        else:
            print(f"\nRESUME: reusing population from {args.resume_dir}")

        sorted_perf = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        all_direction_specs = [corp_ops.make_isotropic_spec(seed, sigma) for (seed, sigma), _ in sorted_perf]
        all_rewards = [reward for _, reward in sorted_perf]

        call_all_engines(engines, "store_base_weights")
        try:
            if is_resume and args.reuse_resume_stage1:
                print("\n" + "=" * 60 + "\nSTAGE 1 [REUSED]\n" + "=" * 60)
                stage1_result, collapse_state = artifact_ops.load_resume_stage1(
                    args.resume_dir, previous_results, target_artifacts_dir=artifacts_dir
                )
                print(f"  q={stage1_result['q']}, beta={stage1_result['beta']}, "
                      f"eta={stage1_result['eta']:.6g}, tail={int(stage1_result['tail_size'])}/{len(all_direction_specs)}")
                print(f"  probe={stage1_result['probe_accuracy']:.2f}%  test={stage1_result['accuracy']:.2f}%")
            else:
                stage1_result, collapse_state = run_stage1(
                    engines[0], handler, probe_prompts, probe_datas, test_prompts, test_datas,
                    sampling_params, all_direction_specs, all_rewards,
                    args.q_grid_list, args.beta_grid_list, args.scale_grid_list,
                    base_test, artifacts_dir, args.recenter_rank,
                )

            initial_center = corp_ops.scale_direction_spec(collapse_state["direction_spec"], stage1_result["eta"])
            _commit_direction(engines, initial_center)

            if args.consolidation_method == "stage1_only":
                stage2_result = {
                    "stage2_applied": False, "accepted_rounds": 0, "rounds_completed": 0,
                    "q": stage1_result["q"], "beta": stage1_result["beta"],
                    "eta": stage1_result["eta"], "alpha": stage1_result.get("scale", stage1_result.get("alpha", 1.0)),
                    "probe_accuracy": stage1_result["probe_accuracy"],
                    "probe_correct": stage1_result["probe_correct"],
                    "accuracy": stage1_result["accuracy"],
                    "correct": stage1_result["correct"],
                    "delta_vs_base": stage1_result["delta_vs_base"],
                    "final_sigma": float(best_sigma),
                    "basis_rank": 0,
                }
                final_direction_spec = initial_center
            else:
                stage2_result, final_direction_spec = run_stage2(
                    args, engines, handler,
                    support_prompts, support_datas, probe_prompts, probe_datas,
                    test_prompts, test_datas, sampling_params, base_test,
                    stage1_result, collapse_state, best_sigma, artifacts_dir,
                )

            artifact_ops.save_direction_specs(artifacts_dir, collapse_state["direction_spec"], stage1_result["eta"], final_direction_spec)

            corp_summary = {
                "method": "corp",
                "stage2_applied": bool(stage2_result.get("stage2_applied", False)),
                "accepted_rounds": int(stage2_result.get("accepted_rounds", 0)),
                "stage1_accuracy": float(stage1_result["accuracy"]),
                "final_accuracy": float(stage2_result["accuracy"]),
                "delta_stage2_vs_stage1": float(stage2_result["accuracy"] - stage1_result["accuracy"]),
            }
            corp_ops.save_json(os.path.join(artifacts_dir, "corp_summary.json"), corp_summary)

        finally:
            call_all_engines(engines, "clear_base_weights")

        artifact_ops.save_results(
            args, logging_dir, base_model_path, base_train, base_test,
            time.time() - start_time, perf, best_sigma,
            corp_summary=corp_summary,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
        )
        print(f"\nFinal test accuracy: {stage2_result['accuracy']:.2f}%  "
              f"(stage1={stage1_result['accuracy']:.2f}%, base={base_test * 100:.2f}%)")
    finally:
        cleanup_engines(engines, pgs)


if __name__ == "__main__":
    main(parse_args())
