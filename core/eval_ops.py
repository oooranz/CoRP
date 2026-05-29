"""Per-example evaluation helpers and probe line-search utilities."""
from typing import Dict, List, Sequence

import numpy as np


def build_example_rows(handler, outputs, task_datas, *, include_responses: bool = False) -> List[dict]:
    rows = []
    for i, (output, data) in enumerate(zip(outputs, task_datas)):
        response = output.outputs[0].text
        ground_truth = data.get("ground_truth")
        score = float(handler.is_answer_correct(response, ground_truth)) if ground_truth is not None else 0.0
        row = {"example_index": i, "example_score": score}
        if include_responses:
            row["response"] = response
        rows.append(row)
    return rows


def build_population_rows(perf: Dict[tuple, float]) -> List[Dict]:
    sorted_entries = sorted(perf.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "rank": rank + 1,
            "candidate_index": rank,
            "seed": int(seed),
            "sigma": float(sigma),
            "support_reward": float(reward),
        }
        for rank, ((seed, sigma), reward) in enumerate(sorted_entries)
    ]


def probe_line_search(
    engine,
    handler,
    probe_prompts,
    probe_datas,
    sampling_params,
    *,
    direction_spec: Sequence,
    mean_rss: float,
    search_grid: Sequence[float],
    eval_fn,
    eta_reference: float | None = None,
    metadata: dict | None = None,
):
    """Line-search the step size on the probe split; return the best result."""
    grid = list(search_grid) if search_grid else [1.0]
    eta_base = float(mean_rss if eta_reference is None else eta_reference)
    best = None
    best_rows = None
    for alpha in grid:
        eta = float(alpha) * eta_base
        metrics = eval_fn(engine, handler, probe_prompts, probe_datas, sampling_params, direction_spec, eta)
        if best is None or metrics["accuracy"] > best["probe_accuracy"]:
            best = {"alpha": float(alpha), "eta": float(eta), "probe_accuracy": float(metrics["accuracy"]), "probe_correct": int(metrics["correct"])}
            best_rows = metrics.get("example_rows", [])
    return {
        **(metadata or {}),
        "direction_spec": direction_spec,
        "mean_rss": float(mean_rss),
        "eta_reference": float(eta_base),
        "alpha": float(best["alpha"]),
        "eta": float(best["eta"]),
        "probe_accuracy": float(best["probe_accuracy"]),
        "probe_correct": int(best["probe_correct"]),
        "example_rows": best_rows,
    }


def finalize_on_test(engine, handler, test_prompts, test_datas, sampling_params, run: dict, *, eval_fn):
    """Evaluate the accepted direction on the test split."""
    metrics = eval_fn(engine, handler, test_prompts, test_datas, sampling_params, run["direction_spec"], run["eta"])
    return {**run, "accuracy": float(metrics["accuracy"]), "correct": int(metrics["correct"]), "example_rows": metrics.get("example_rows", [])}
