import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DirectionComponent:
    seed: int
    sigma: float
    weight: float = 1.0


DirectionSpec = List[DirectionComponent]
DirectionBasis = List[Tuple[int, float]]


def save_json(path: str, payload) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=4)


def summarize_values(values: Sequence[float]) -> Dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def summarize_improvements(
    values: Sequence[float],
    baseline: float,
    *,
    eps: float = 1e-12,
    thresholds_pp: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 5.0),
) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    deltas = arr - float(baseline)
    positive = deltas > eps
    flat = np.abs(deltas) <= eps
    negative = deltas < -eps

    buckets = {}
    delta_pp = deltas * 100.0
    for threshold in thresholds_pp:
        key = f">={float(threshold):g}pp"
        count = int(np.sum(delta_pp >= float(threshold)))
        buckets[key] = {
            "count": count,
            "fraction": float(count / arr.size) if arr.size else 0.0,
        }

    return {
        "baseline": float(baseline),
        "positive_count": int(positive.sum()),
        "positive_fraction": float(positive.mean()) if arr.size else 0.0,
        "flat_count": int(flat.sum()),
        "flat_fraction": float(flat.mean()) if arr.size else 0.0,
        "negative_count": int(negative.sum()),
        "negative_fraction": float(negative.mean()) if arr.size else 0.0,
        "delta_stats": summarize_values(deltas.tolist()),
        "delta_stats_pp": summarize_values(delta_pp.tolist()),
        "gain_buckets_pp": buckets,
    }


def safe_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return None
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def build_results_payload(
    *,
    args,
    base_model_path: str,
    base_train: float,
    base_test: float,
    runtime_seconds: float,
    sigma_stats,
    best_sigma: float,
    consolidation_suite_result=None,
    consensus_signal_result=None,
    mean_result=None,
    lowrank_result=None,
    score_result=None,
    ridge_result=None,
    legacy_randsoup_result=None,
    legacy_guided_result=None,
):
    config = {
        "probe_start_index": args.probe_start_index,
        "probe_samples": args.probe_samples,
        "soup_train_samples": args.soup_train_samples,
        "scale_grid": getattr(args, "scale_grid_list", args.alpha_grid_list),
        "alpha_grid": args.alpha_grid_list,
        "step_grid": getattr(args, "step_grid_list", args.alpha_grid_list),
        "beta_grid": args.beta_grid_list,
        "q_grid": args.q_grid_list,
        "recenter_attempts": args.recenter_attempts,
        "recenter_population_size": args.recenter_population_size,
        "recenter_rank": args.recenter_rank,
        "recenter_iso_lambda": args.recenter_iso_lambda,
        "recenter_sigma_up": args.recenter_sigma_up,
        "recenter_sigma_down": args.recenter_sigma_down,
        "recenter_accept_epsilon": args.recenter_accept_epsilon,
        "recenter_patience": args.recenter_patience,
        "max_accepted_recenters": args.max_accepted_recenters,
        "guided_rounds": args.recenter_attempts,
        "guided_population_size": args.recenter_population_size,
        "guided_lambda": args.recenter_iso_lambda,
        "guided_sigma_up": args.recenter_sigma_up,
        "guided_sigma_down": args.recenter_sigma_down,
        "guided_accept_epsilon": args.recenter_accept_epsilon,
        "guided_patience": args.recenter_patience,
        "consolidation_method": getattr(args, "consolidation_method", None),
        "consolidation_budget": getattr(args, "consolidation_budget", None),
        "reuse_resume_stage1": getattr(args, "reuse_resume_stage1", None),
        "family_score_mode": getattr(args, "family_score_mode", None),
        "family_lcb_z": getattr(args, "family_lcb_z", None),
        "family_topm_grid": getattr(args, "family_topm_grid", None),
        "family_weights_mode": getattr(args, "family_weights_mode", None),
        "stage2_regression_lambda": getattr(args, "stage2_regression_lambda", None),
        "stage2_regression_lambda_final": getattr(args, "stage2_regression_lambda_final", None),
        "stage2_target_ess_initial": getattr(args, "stage2_target_ess_initial", None),
        "stage2_target_ess_final": getattr(args, "stage2_target_ess_final", None),
        "stage2_gate_lcb_z": getattr(args, "stage2_gate_lcb_z", None),
        "run_consensus_signal_analysis": getattr(args, "run_consensus_signal_analysis", None),
        "consensus_diag_top_m": getattr(args, "consensus_diag_top_m", None),
        "consensus_diag_rank": getattr(args, "consensus_diag_rank", None),
    }
    payload = {
        "dataset": args.dataset,
        "model": base_model_path,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "base_train_accuracy": base_train,
        "base_test_accuracy": base_test,
        "runtime_seconds": runtime_seconds,
        "sigma_stats": sigma_stats,
        "best_sigma": best_sigma,
        "config": config,
        "consensus_signal_result": consensus_signal_result,
        "consolidation_suite_result": consolidation_suite_result,
        "consolidation_mean_result": mean_result,
        "consolidation_lowrank_result": lowrank_result,
        "consolidation_score_result": score_result,
        "consolidation_ridge_result": ridge_result,
        "legacy_randsoup_result": legacy_randsoup_result,
        "legacy_guided_randsoup_result": legacy_guided_result,
        "randsoup_result": legacy_randsoup_result,
        "guided_randsoup_result": legacy_guided_result if legacy_guided_result is not None else legacy_randsoup_result,
    }
    return payload


def make_isotropic_spec(seed: int, sigma: float) -> DirectionSpec:
    return [DirectionComponent(seed=int(seed), sigma=float(sigma), weight=1.0)]


def serialize_direction_spec(spec: Sequence[DirectionComponent]) -> List[Tuple[int, float, float]]:
    return [(int(component.seed), float(component.sigma), float(component.weight)) for component in spec]


def flatten_direction_spec(spec: Sequence[DirectionComponent], atol: float = 1e-12) -> DirectionSpec:
    merged = {}
    for component in spec:
        key = (int(component.seed), float(component.sigma))
        merged[key] = merged.get(key, 0.0) + float(component.weight)
    flattened = [
        DirectionComponent(seed=seed, sigma=sigma, weight=weight)
        for (seed, sigma), weight in sorted(merged.items())
        if abs(weight) > atol
    ]
    return flattened


def scale_direction_spec(spec: Sequence[DirectionComponent], scale: float) -> DirectionSpec:
    return flatten_direction_spec(
        [
            DirectionComponent(
                seed=component.seed,
                sigma=component.sigma,
                weight=float(component.weight) * float(scale),
            )
            for component in spec
        ]
    )


def combine_direction_specs(
    specs: Sequence[Sequence[DirectionComponent]],
    coefficients: Sequence[float] | None = None,
) -> DirectionSpec:
    if coefficients is None:
        coefficients = [1.0] * len(specs)
    if len(specs) != len(coefficients):
        raise ValueError("specs and coefficients must have the same length")

    components: DirectionSpec = []
    for spec, coeff in zip(specs, coefficients):
        if abs(float(coeff)) <= 1e-12:
            continue
        components.extend(scale_direction_spec(spec, float(coeff)))
    return flatten_direction_spec(components)


def weighted_average_specs(
    specs: Sequence[Sequence[DirectionComponent]],
    weights: Sequence[float],
) -> DirectionSpec:
    return combine_direction_specs(specs, weights)


def direction_rss_norm(spec: Sequence[DirectionComponent]) -> float:
    return math.sqrt(sum((float(component.weight) * float(component.sigma)) ** 2 for component in spec))


def direction_to_amplitude_map(spec: Sequence[DirectionComponent]) -> Dict[Tuple[int, float], float]:
    amplitudes: Dict[Tuple[int, float], float] = {}
    for component in spec:
        key = (int(component.seed), float(component.sigma))
        amplitudes[key] = amplitudes.get(key, 0.0) + float(component.weight) * float(component.sigma)
    return {key: value for key, value in amplitudes.items() if abs(value) > 1e-12}


def direction_basis(specs: Sequence[Sequence[DirectionComponent]]) -> DirectionBasis:
    keys = set()
    for spec in specs:
        keys.update(direction_to_amplitude_map(spec).keys())
    return sorted(keys)


def vectorize_direction_spec(spec: Sequence[DirectionComponent], basis: DirectionBasis) -> np.ndarray:
    amplitude_map = direction_to_amplitude_map(spec)
    return np.asarray([amplitude_map.get(key, 0.0) for key in basis], dtype=np.float64)


def vectorize_direction_specs(
    specs: Sequence[Sequence[DirectionComponent]],
    basis: DirectionBasis | None = None,
) -> Tuple[np.ndarray, DirectionBasis]:
    if basis is None:
        basis = direction_basis(specs)
    if not basis:
        return np.zeros((len(specs), 0), dtype=np.float64), []
    matrix = np.stack([vectorize_direction_spec(spec, basis) for spec in specs], axis=0)
    return matrix, list(basis)


def direction_from_amplitudes(basis: DirectionBasis, amplitudes: Sequence[float], atol: float = 1e-12) -> DirectionSpec:
    components: DirectionSpec = []
    for (seed, sigma), amplitude in zip(basis, amplitudes):
        amplitude_value = float(amplitude)
        if abs(amplitude_value) <= atol:
            continue
        components.append(
            DirectionComponent(
                seed=int(seed),
                sigma=float(sigma),
                weight=amplitude_value / float(sigma),
            )
        )
    return flatten_direction_spec(components, atol=atol)


def direction_inner_product(
    first: Sequence[DirectionComponent],
    second: Sequence[DirectionComponent],
) -> float:
    first_map = direction_to_amplitude_map(first)
    second_map = direction_to_amplitude_map(second)
    if len(first_map) > len(second_map):
        first_map, second_map = second_map, first_map
    return float(sum(value * second_map.get(key, 0.0) for key, value in first_map.items()))


def direction_cosine_similarity(
    first: Sequence[DirectionComponent],
    second: Sequence[DirectionComponent],
    eps: float = 1e-12,
) -> float:
    first_norm = direction_rss_norm(first)
    second_norm = direction_rss_norm(second)
    if first_norm <= eps or second_norm <= eps:
        return 0.0
    return float(direction_inner_product(first, second) / (first_norm * second_norm))


def project_direction_onto(
    spec: Sequence[DirectionComponent],
    target: Sequence[DirectionComponent],
    eps: float = 1e-12,
) -> Dict[str, object]:
    target_norm = direction_rss_norm(target)
    if target_norm <= eps:
        zero_spec: DirectionSpec = []
        return {
            "coefficient": 0.0,
            "projection_spec": zero_spec,
            "residual_spec": flatten_direction_spec(spec),
            "residual_norm_sq": float(direction_rss_norm(spec) ** 2),
        }

    target_unit = normalize_direction_spec(target, eps=eps)
    coefficient = direction_inner_product(spec, target_unit)
    projection_spec = scale_direction_spec(target_unit, coefficient)
    residual_spec = combine_direction_specs([spec, scale_direction_spec(projection_spec, -1.0)])
    residual_norm_sq = max(direction_rss_norm(spec) ** 2 - coefficient ** 2, 0.0)
    return {
        "coefficient": float(coefficient),
        "projection_spec": projection_spec,
        "residual_spec": residual_spec,
        "residual_norm_sq": float(residual_norm_sq),
    }


def normalize_direction_spec(spec: Sequence[DirectionComponent], eps: float = 1e-12) -> DirectionSpec:
    norm = direction_rss_norm(spec)
    if norm <= eps:
        return flatten_direction_spec(spec)
    return scale_direction_spec(spec, 1.0 / norm)


def build_local_eta_grid(
    center_eta: float,
    factors: Sequence[float] = (0.5, 1.0, 2.0),
    min_eta: float = 1e-8,
    eps: float = 1e-12,
) -> List[float]:
    center = max(float(center_eta), float(min_eta))
    candidates = sorted({max(center * float(factor), float(min_eta)) for factor in factors if float(factor) > eps})
    if not candidates:
        raise ValueError("local eta grid must contain at least one positive value")
    return candidates


def tail_softmax_weights(
    rewards: Sequence[float],
    q: float,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    reward_arr = np.asarray(rewards, dtype=np.float64)
    if reward_arr.size == 0:
        return reward_arr, np.zeros(0, dtype=bool)

    threshold = float(np.quantile(reward_arr, q))
    mask = reward_arr >= threshold
    if not mask.any():
        mask = np.zeros(len(reward_arr), dtype=bool)
        mask[int(reward_arr.argmax())] = True

    tail = reward_arr[mask]
    shifted = beta * tail - (beta * tail).max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    return weights, mask


def resolve_shared_split_ranges(
    total_size: int,
    train_samples: int,
    soup_train_samples: int,
    probe_start_index: int,
    probe_samples: int,
    test_samples: int | None,
) -> Tuple[slice, slice, slice, slice]:
    train_end = max(0, min(int(train_samples), total_size))
    soup_end = max(train_end, min(int(soup_train_samples), total_size))
    probe_start = soup_end if int(probe_start_index) < 0 else max(soup_end, min(int(probe_start_index), total_size))
    probe_end = max(probe_start, min(probe_start + int(probe_samples), total_size))
    test_start = max(soup_end, probe_end)
    if test_samples is None:
        test_end = total_size
    else:
        test_end = max(test_start, min(test_start + int(test_samples), total_size))

    return slice(0, train_end), slice(0, soup_end), slice(probe_start, probe_end), slice(test_start, test_end)


def resolve_shared_support_probe_budgets(
    total_size: int,
    requested_support: int,
    requested_probe: int,
    *,
    small_dataset_threshold: int = 1000,
    shared_support_cap: int = 150,
    shared_probe_cap: int = 50,
) -> Tuple[int, int]:
    """Choose support/probe budgets for datasets that only expose one evaluation file.

    For small shared-file benchmarks such as MATH-500, we keep the official-style
    full test set intact and shrink the support/probe budgets instead of consuming
    the entire file for training-time selection.
    """
    total_size = max(0, int(total_size))
    support = max(0, int(requested_support))
    probe = max(0, int(requested_probe))

    if total_size <= small_dataset_threshold:
        support = min(support, int(shared_support_cap))
        probe = min(probe, int(shared_probe_cap))

    support = min(support, total_size)
    probe = min(probe, max(0, total_size - support))
    return support, probe
