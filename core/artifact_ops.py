"""Artifact persistence and resume helpers."""
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import collapse_ops, corp_ops


def save_results(
    args,
    logging_dir: str,
    base_model_path: str,
    base_train: float,
    base_test: float,
    runtime_seconds: float,
    perf: dict,
    best_sigma: float,
    *,
    corp_summary=None,
    stage1_result=None,
    stage2_result=None,
):
    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (_, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)
    sigma_stats = {
        str(s): {
            "mean": float(np.mean(sigma_rewards[s])) if sigma_rewards[s] else None,
            "count": len(sigma_rewards[s]),
        }
        for s in args.sigma_list
    }

    results = {
        "dataset": args.dataset,
        "model": base_model_path,
        "train_samples": args.soup_train_samples,
        "test_samples": args.test_samples,
        "base_train_accuracy": float(base_train),
        "base_test_accuracy": float(base_test),
        "runtime_seconds": float(runtime_seconds),
        "best_sigma": float(best_sigma),
        "sigma_stats": sigma_stats,
        "corp_summary": corp_summary,
        "stage1_result": stage1_result,
        "stage2_result": stage2_result,
    }
    path = os.path.join(logging_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {path}")


def save_sampling_population(model_saves_dir: str, base_model_path: str, base_train: float, best_sigma: float, perf: dict):
    rows = [
        {"seed": int(seed), "sigma": float(sigma), "support_reward": float(reward)}
        for (seed, sigma), reward in perf.items()
    ]
    payload = {
        "base_model_path": base_model_path,
        "base_train_accuracy": float(base_train),
        "best_sigma": float(best_sigma),
        "population_models": rows,
    }
    path = os.path.join(model_saves_dir, "sampling_population.json")
    corp_ops.save_json(path, payload)


def load_sampling_population(resume_dir: str) -> Tuple[str, float, float, dict]:
    path = os.path.join(resume_dir, "model_saves", "sampling_population.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot resume: missing {path}")
    with open(path) as f:
        payload = json.load(f)
    base_model_path = str(payload["base_model_path"])
    base_train = float(payload.get("base_train_accuracy", 0.0))
    best_sigma = float(payload["best_sigma"])
    perf = {}
    for row in payload.get("population_models", []):
        perf[(int(row["seed"]), float(row["sigma"]))] = float(row.get("support_reward", row.get("train_reward", 0.0)))
    return base_model_path, base_train, best_sigma, perf


def load_resume_results(resume_dir: str, *, base_train: float) -> dict:
    results_path = os.path.join(resume_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        # normalise old key names
        if "stage1_result" not in data:
            data["stage1_result"] = data.get("legacy_randsoup_result") or data.get("randsoup_result")
        if "stage2_result" not in data:
            data["stage2_result"] = data.get("legacy_guided_randsoup_result") or data.get("guided_randsoup_result")
        return data

    summary_path = os.path.join(resume_dir, "artifacts", "stage1_summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(resume_dir, "artifacts", "randsoup_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Cannot resume {resume_dir}: no results.json or stage1 summary")
    with open(summary_path) as f:
        summary = json.load(f)
    stage1 = dict(summary.get("best_result") or {})
    if not stage1:
        raise ValueError(f"Missing best_result in {summary_path}")
    base_test = float(stage1["accuracy"] - stage1["delta_vs_base"]) / 100.0
    return {
        "base_train_accuracy": float(base_train),
        "base_test_accuracy": float(base_test),
        "stage1_result": stage1,
    }


def save_direction_specs(artifacts_dir: str, stage1_spec, stage1_eta: float, final_spec):
    payload = {
        "stage1_direction_spec": corp_ops.serialize_direction_spec(stage1_spec),
        "stage1_eta": float(stage1_eta),
        "final_direction_spec": corp_ops.serialize_direction_spec(final_spec),
    }
    corp_ops.save_json(os.path.join(artifacts_dir, "direction_specs.json"), payload)


def load_direction_specs(artifacts_dir: str) -> dict:
    path = os.path.join(artifacts_dir, "direction_specs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = json.load(f)

    def _deser(raw):
        if not raw:
            return []
        return [corp_ops.DirectionComponent(seed=int(s), sigma=float(sig), weight=float(w)) for s, sig, w in raw]

    return {
        "stage1_direction_spec": _deser(payload.get("stage1_direction_spec")),
        "stage1_eta": float(payload.get("stage1_eta", 0.0)),
        "final_direction_spec": _deser(payload.get("final_direction_spec") or payload.get("guided_final_direction_spec")),
    }


def load_resume_stage1(resume_dir: str, previous_results: dict, *, target_artifacts_dir: str | None = None) -> Tuple[dict, dict]:
    source_dir = os.path.join(resume_dir, "artifacts")
    summary_path = next(
        (os.path.join(source_dir, fn) for fn in ("stage1_summary.json", "randsoup_summary.json") if os.path.exists(os.path.join(source_dir, fn))),
        None,
    )
    if summary_path is None:
        raise FileNotFoundError(f"Missing saved stage-1 summary in {source_dir}")

    with open(summary_path) as f:
        summary_payload = json.load(f)
    stage1_result = dict(summary_payload.get("best_result") or {})
    prev = dict(previous_results.get("stage1_result") or previous_results.get("legacy_randsoup_result") or previous_results.get("randsoup_result") or {})
    for key, val in prev.items():
        if val is not None:
            stage1_result[key] = val

    # Try to load saved direction spec
    saved = load_direction_specs(source_dir)
    stage1_spec = list(saved.get("stage1_direction_spec") or [])
    stage1_mean_rss = _infer_mean_rss(stage1_result)

    # Load elite rows to reconstruct cloud
    elite_rows = []
    for fn in ("stage1_tail_candidates.json", "randsoup_tail_candidates.json", "stage1_elite_cloud.json"):
        candidate_path = os.path.join(source_dir, fn)
        if os.path.exists(candidate_path):
            with open(candidate_path) as f:
                tail = json.load(f)
            if tail.get("elite_rows"):
                elite_rows = [dict(r) for r in tail["elite_rows"]]
                break

    reconstructed_cloud = None
    if elite_rows:
        elite_specs = [corp_ops.make_isotropic_spec(int(r["seed"]), float(r["sigma"])) for r in elite_rows]
        pass2_weights = [float(r.get("pass2_weight", 0.0)) for r in elite_rows]
        if not any(abs(w) > 1e-12 for w in pass2_weights):
            pass2_weights = [1.0] * len(elite_specs)
        cloud_rank = max(int(stage1_result.get("subspace_rank", 0) or 0), 1)
        reconstructed_cloud = collapse_ops.build_weighted_cloud(elite_specs, pass2_weights, rank=cloud_rank)

    if stage1_spec:
        stage1_spec = corp_ops.normalize_direction_spec(stage1_spec)
        if stage1_mean_rss <= 1e-12 and reconstructed_cloud is not None:
            stage1_mean_rss = float(reconstructed_cloud["mean_rss"])
        stage1_mean_spec = corp_ops.scale_direction_spec(stage1_spec, stage1_mean_rss)
    elif reconstructed_cloud is not None:
        stage1_mean_spec = reconstructed_cloud["mean_spec"]
        stage1_mean_rss = float(reconstructed_cloud["mean_rss"])
        stage1_spec = reconstructed_cloud["direction_spec"]
    else:
        raise ValueError(f"Cannot reconstruct stage-1 from {source_dir}")

    stage1_result["stage1_source"] = "resume_artifacts"
    if stage1_mean_rss > 1e-12:
        stage1_result.setdefault("mean_rss", stage1_mean_rss)

    collapse_state = {
        "q": float(stage1_result.get("q", 0.0)),
        "beta": float(stage1_result.get("beta", 0.0)),
        "direction_spec": stage1_spec,
        "mean_spec": stage1_mean_spec,
        "principal_directions": [] if reconstructed_cloud is None else reconstructed_cloud["principal_directions"],
        "principal_eigvals": [] if reconstructed_cloud is None else reconstructed_cloud["principal_eigvals"],
        "kappa": float((reconstructed_cloud or stage1_result).get("kappa", 0.0)),
        "trace_cov": float((reconstructed_cloud or stage1_result).get("trace_cov", 0.0)),
        "effective_rank": float((reconstructed_cloud or stage1_result).get("effective_rank", 0.0)),
        "top_eigvals": list((reconstructed_cloud or stage1_result).get("top_eigvals", [])),
        "subspace_rank": int(len(reconstructed_cloud["principal_directions"]) if reconstructed_cloud else stage1_result.get("subspace_rank", 0) or 0),
        "mean_rss": float(stage1_mean_rss),
        "elite_rows": elite_rows,
        "elite_direction_specs": [] if not elite_rows else [corp_ops.make_isotropic_spec(int(r["seed"]), float(r["sigma"])) for r in elite_rows],
    }

    if target_artifacts_dir is not None:
        _copy_stage1_artifacts(source_dir, target_artifacts_dir)

    return stage1_result, collapse_state


def _infer_mean_rss(stage1_result: dict) -> float:
    for key in ("mean_rss", "merged_rss"):
        val = stage1_result.get(key)
        if val is not None:
            val = float(val)
            if val > 1e-12:
                return val
    eta = stage1_result.get("eta")
    alpha = stage1_result.get("scale_multiplier") or stage1_result.get("alpha")
    if eta is None or not alpha:
        return 0.0
    return float(eta) / float(alpha)


def _copy_stage1_artifacts(source_dir: str, target_dir: str):
    for fn in ("stage1_summary.json", "randsoup_summary.json", "stage1_tail_candidates.json",
               "randsoup_tail_candidates.json", "stage1_elite_cloud.json", "direction_specs.json"):
        src = os.path.join(source_dir, fn)
        if not os.path.exists(src):
            continue
        with open(src) as f:
            data = json.load(f)
        corp_ops.save_json(os.path.join(target_dir, fn), data)


