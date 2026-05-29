"""vLLM engine setup and management."""
import gc
import os
import time
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM


class RandOptNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(
    num_engines: int,
    model_name: str,
    precision: str = "bfloat16",
    batch_size: int = 25,
    tensor_parallel_size: int = 1,
    enable_prefix_caching: bool = False,
    gpu_memory_utilization: float = 0.75,
    multimodal: bool = False,
    use_ray: bool = True,
):
    required_gpus = num_engines * tensor_parallel_size
    if not use_ray:
        engines = []
        for _ in range(num_engines):
            kw = dict(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="mp",
                worker_extension_cls="utils.worker_extn.WorkerExtension",
                dtype=precision,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=True,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_log_stats=True,
            )
            if multimodal:
                kw["limit_mm_per_prompt"] = {"image": 1}
            engines.append(RandOptNcclLLM(**kw))
        return engines, []

    cluster_resources = ray.cluster_resources()
    available_gpus = int(cluster_resources.get("GPU", 0))
    print(f"Cluster: {available_gpus} GPUs available, {required_gpus} required")

    if available_gpus < required_gpus:
        max_engines = available_gpus // tensor_parallel_size
        print(f"WARNING: Reducing num_engines {num_engines} -> {max_engines}")
        num_engines = max_engines
        if num_engines == 0:
            raise RuntimeError(f"No GPUs available (need at least {tensor_parallel_size})")

    pg_bundles = [{"GPU": 1, "CPU": 0} for _ in range(tensor_parallel_size)]
    pgs = [placement_group(pg_bundles, lifetime="detached") for _ in range(num_engines)]
    try:
        ray.get([pg.ready() for pg in pgs], timeout=120)
    except ray.exceptions.GetTimeoutError:
        from ray.util.placement_group import remove_placement_group
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        raise RuntimeError("Timeout waiting for placement groups")

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    kw = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype=precision,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )
    if multimodal:
        kw["limit_mm_per_prompt"] = {"image": 1}

    engines = []
    num_batches = (num_engines + batch_size - 1) // batch_size
    t0 = time.time()
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_engines)
        print(f"Launching engines {start}-{end - 1}...")
        batch = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=s)(RandOptNcclLLM).remote(**kw)
            for s in strategies[start:end]
        ]
        ray.get([e.collective_rpc.remote("store_base_weights", args=()) for e in batch])
        engines.extend(batch)

    print(f"All {num_engines} engines launched in {time.time() - t0:.1f}s")
    return engines, pgs


def cleanup_engines(engines: list, pgs: list):
    if not pgs:
        for llm in engines:
            try:
                if hasattr(llm, "llm_engine"):
                    eng = llm.llm_engine
                    for attr in ("model_executor", "engine_core"):
                        obj = getattr(eng, attr, None)
                        if obj is not None and hasattr(obj, "shutdown"):
                            obj.shutdown()
                    if hasattr(eng, "shutdown"):
                        eng.shutdown()
            except Exception:
                pass
        engines.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(0.25)
        return

    from ray.util.placement_group import remove_placement_group
    for llm in engines:
        try:
            ray.kill(llm)
        except Exception:
            pass
    for pg in pgs:
        try:
            remove_placement_group(pg)
        except Exception:
            pass
    engines.clear()
    gc.collect()
    ray.shutdown()
