import math
from typing import Dict, List, Sequence

import numpy as np

from . import corp_ops as randsoup_ops


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def _effective_sample_size(weights: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.square(weights).sum())
    if denom <= eps:
        return 0.0
    return float(1.0 / denom)


def _zscore(values: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std <= eps:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def _select_elite(rewards: Sequence[float], q: float) -> Dict[str, object]:
    reward_arr = np.asarray(rewards, dtype=np.float64)
    threshold = float(np.quantile(reward_arr, q))
    mask = reward_arr >= threshold
    if not mask.any():
        mask = np.zeros(len(reward_arr), dtype=bool)
        mask[int(reward_arr.argmax())] = True
        threshold = float(reward_arr.max())
    indices = np.flatnonzero(mask).tolist()
    return {
        "threshold": threshold,
        "mask": mask,
        "indices": indices,
        "rewards": reward_arr[mask],
    }


def _cloud_statistics(
    specs: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    weights: Sequence[float],
    *,
    rank: int,
    eps: float = 1e-12,
) -> Dict[str, object]:
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = weight_arr / max(float(weight_arr.sum()), eps)
    matrix, basis = randsoup_ops.vectorize_direction_specs(specs)
    if matrix.shape[1] == 0:
        zero_spec: randsoup_ops.DirectionSpec = []
        return {
            "basis": basis,
            "mean_vector": np.zeros(0, dtype=np.float64),
            "mean_spec": zero_spec,
            "direction_spec": zero_spec,
            "mean_rss": 0.0,
            "trace_cov": 0.0,
            "effective_rank": 0.0,
            "top_eigvals": [],
            "principal_eigvals": [],
            "principal_directions": [],
            "kappa": 0.0,
            "num_direction_components": 0,
        }

    mean_vector = weight_arr @ matrix
    mean_spec = randsoup_ops.direction_from_amplitudes(basis, mean_vector)
    direction_spec = randsoup_ops.normalize_direction_spec(mean_spec)
    mean_rss = randsoup_ops.direction_rss_norm(mean_spec)

    centered = matrix - mean_vector
    weighted_centered = np.sqrt(weight_arr)[:, None] * centered
    if weighted_centered.size == 0:
        eigvals = np.zeros(0, dtype=np.float64)
        eigvecs = np.zeros((0, matrix.shape[1]), dtype=np.float64)
    else:
        _, singular_values, vh = np.linalg.svd(weighted_centered, full_matrices=False)
        eigvals = singular_values ** 2
        eigvecs = vh

    trace_cov = float(eigvals.sum()) if eigvals.size else 0.0
    if trace_cov <= eps:
        effective_rank = 0.0
    else:
        mass = eigvals / trace_cov
        effective_rank = float(np.exp(-(mass * np.log(np.clip(mass, eps, None))).sum()))

    top_rank = min(int(rank), len(eigvals))
    principal_eigvals = [float(value) for value in eigvals[:top_rank]]
    principal_directions = []
    for idx in range(top_rank):
        principal_spec = randsoup_ops.direction_from_amplitudes(basis, eigvecs[idx])
        principal_directions.append(randsoup_ops.normalize_direction_spec(principal_spec))

    kappa = float((mean_rss ** 2) / (trace_cov + eps))
    return {
        "basis": basis,
        "mean_vector": mean_vector,
        "mean_spec": mean_spec,
        "direction_spec": direction_spec,
        "mean_rss": float(mean_rss),
        "trace_cov": trace_cov,
        "effective_rank": effective_rank,
        "top_eigvals": [float(value) for value in eigvals[: min(8, len(eigvals))]],
        "principal_eigvals": principal_eigvals,
        "principal_directions": principal_directions,
        "kappa": kappa,
        "num_direction_components": len(direction_spec),
    }


def build_weighted_cloud(
    direction_specs: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    weights: Sequence[float],
    *,
    rank: int,
    eps: float = 1e-12,
) -> Dict[str, object]:
    if len(direction_specs) != len(weights):
        raise ValueError("direction_specs and weights must have the same length")
    return _cloud_statistics(direction_specs, weights, rank=rank, eps=eps)


def build_compressibility_aware_merge(
    direction_specs: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    rewards: Sequence[float],
    q: float,
    beta: float,
    *,
    rank: int,
    eps: float = 1e-12,
) -> Dict[str, object]:
    reward_arr = np.asarray(rewards, dtype=np.float64)
    elite = _select_elite(reward_arr, q)
    elite_indices = elite["indices"]
    elite_rewards = reward_arr[elite["mask"]]
    elite_specs = [direction_specs[idx] for idx in elite_indices]

    pass1_logits = float(beta) * elite_rewards
    pass1_weights = _softmax(pass1_logits)
    provisional = _cloud_statistics(elite_specs, pass1_weights.tolist(), rank=rank, eps=eps)
    provisional_mean = provisional["mean_spec"]

    alignments = [randsoup_ops.direction_cosine_similarity(spec, provisional_mean, eps=eps) for spec in elite_specs]
    dispersions = [
        randsoup_ops.project_direction_onto(spec, provisional_mean, eps=eps)["residual_norm_sq"]
        for spec in elite_specs
    ]
    alignment_z = _zscore(alignments, eps=eps)
    dispersion_z = _zscore(dispersions, eps=eps)

    pass2_logits = float(beta) * elite_rewards + alignment_z - dispersion_z
    pass2_weights = _softmax(pass2_logits)
    final_cloud = _cloud_statistics(elite_specs, pass2_weights.tolist(), rank=rank, eps=eps)

    sorted_pass2 = np.sort(pass2_weights)[::-1]
    elite_rows = []
    for row_index, (elite_index, reward, pass1_weight, pass2_weight, alignment, alignment_score, dispersion, dispersion_score) in enumerate(
        zip(
            elite_indices,
            elite_rewards.tolist(),
            pass1_weights.tolist(),
            pass2_weights.tolist(),
            alignments,
            alignment_z.tolist(),
            dispersions,
            dispersion_z.tolist(),
        ),
        start=1,
    ):
        component_spec = direction_specs[elite_index]
        first_component = component_spec[0] if component_spec else None
        elite_rows.append(
            {
                "rank": row_index,
                "candidate_index": int(elite_index),
                "seed": int(first_component.seed) if first_component is not None else None,
                "sigma": float(first_component.sigma) if first_component is not None else None,
                "support_reward": float(reward),
                "pass1_weight": float(pass1_weight),
                "pass2_weight": float(pass2_weight),
                "alignment": float(alignment),
                "alignment_score": float(alignment_score),
                "dispersion": float(dispersion),
                "dispersion_score": float(dispersion_score),
            }
        )

    return {
        "q": float(q),
        "beta": float(beta),
        "tail_size": int(len(elite_indices)),
        "reward_threshold": float(elite["threshold"]),
        "tail_reward_mean": float(elite_rewards.mean()),
        "tail_reward_min": float(elite_rewards.min()),
        "tail_reward_max": float(elite_rewards.max()),
        "pass1_weight_entropy": float(-(pass1_weights * np.log(np.clip(pass1_weights, eps, None))).sum()),
        "pass1_effective_sample_size": float(1.0 / np.square(pass1_weights).sum()),
        "weight_entropy": float(-(pass2_weights * np.log(np.clip(pass2_weights, eps, None))).sum()),
        "effective_sample_size": float(1.0 / np.square(pass2_weights).sum()),
        "max_weight": float(pass2_weights.max()),
        "top5_weight_mass": float(sorted_pass2[: min(5, len(sorted_pass2))].sum()),
        "top10_weight_mass": float(sorted_pass2[: min(10, len(sorted_pass2))].sum()),
        "provisional_mean_rss": float(provisional["mean_rss"]),
        "alignment_stats": randsoup_ops.summarize_values(alignments),
        "alignment_score_stats": randsoup_ops.summarize_values(alignment_z.tolist()),
        "dispersion_stats": randsoup_ops.summarize_values(dispersions),
        "dispersion_score_stats": randsoup_ops.summarize_values(dispersion_z.tolist()),
        "basis_size": len(final_cloud["basis"]),
        "mean_rss": float(final_cloud["mean_rss"]),
        "merged_rss": float(final_cloud["mean_rss"]),
        "trace_cov": float(final_cloud["trace_cov"]),
        "effective_rank": float(final_cloud["effective_rank"]),
        "top_eigvals": final_cloud["top_eigvals"],
        "subspace_rank": len(final_cloud["principal_directions"]),
        "kappa": float(final_cloud["kappa"]),
        "direction_spec": final_cloud["direction_spec"],
        "mean_spec": final_cloud["mean_spec"],
        "principal_directions": final_cloud["principal_directions"],
        "principal_eigvals": final_cloud["principal_eigvals"],
        "num_direction_components": int(final_cloud["num_direction_components"]),
        "elite_indices": elite_indices,
        "pass1_weights": [float(value) for value in pass1_weights.tolist()],
        "pass2_weights": [float(value) for value in pass2_weights.tolist()],
        "elite_rows": elite_rows,
    }


def build_positive_recenter_merge(
    direction_specs: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    scores: Sequence[float],
    *,
    target_ess: float,
    rank: int,
    eps: float = 1e-12,
) -> Dict[str, object] | None:
    score_arr = np.asarray(scores, dtype=np.float64)
    positive_mask = score_arr > 0.0
    if not positive_mask.any():
        return None

    positive_indices = np.flatnonzero(positive_mask).tolist()
    positive_scores = score_arr[positive_mask]
    positive_specs = [direction_specs[idx] for idx in positive_indices]
    num_positive = int(len(positive_scores))

    target_ess = float(np.clip(float(target_ess), 1.0, float(num_positive)))
    uniform_weights = np.full(num_positive, 1.0 / float(num_positive), dtype=np.float64)

    if num_positive == 1 or float(positive_scores.std()) <= eps or target_ess >= float(num_positive) - 1e-6:
        temperature = float("inf")
        weights = uniform_weights
    else:
        def weights_for_temperature(temp: float) -> np.ndarray:
            return _softmax(positive_scores / max(float(temp), eps))

        def ess_for_temperature(temp: float) -> float:
            return _effective_sample_size(weights_for_temperature(temp), eps=eps)

        low = float(eps)
        high = 1.0
        while ess_for_temperature(high) < target_ess and high < 1e6:
            high *= 2.0

        if ess_for_temperature(high) < target_ess:
            temperature = float("inf")
            weights = uniform_weights
        else:
            for _ in range(60):
                mid = 0.5 * (low + high)
                if ess_for_temperature(mid) >= target_ess:
                    high = mid
                else:
                    low = mid
            temperature = float(high)
            weights = weights_for_temperature(temperature)

    cloud = _cloud_statistics(positive_specs, weights.tolist(), rank=rank, eps=eps)
    sorted_weights = np.sort(weights)[::-1]
    score_stats = randsoup_ops.summarize_values(positive_scores.tolist())

    elite_rows = []
    for row_index, (candidate_index, score, weight, spec) in enumerate(
        zip(positive_indices, positive_scores.tolist(), weights.tolist(), positive_specs),
        start=1,
    ):
        first_component = spec[0] if spec else None
        elite_rows.append(
            {
                "rank": int(row_index),
                "candidate_index": int(candidate_index),
                "seed": int(first_component.seed) if first_component is not None else None,
                "sigma": float(first_component.sigma) if first_component is not None else None,
                "selection_score": float(score),
                "weight": float(weight),
            }
        )

    return {
        "tail_size": int(num_positive),
        "positive_count": int(num_positive),
        "target_ess": float(target_ess),
        "temperature": float(temperature),
        "score_threshold": 0.0,
        "score_stats": score_stats,
        "weight_entropy": float(-(weights * np.log(np.clip(weights, eps, None))).sum()),
        "effective_sample_size": float(_effective_sample_size(weights, eps=eps)),
        "max_weight": float(weights.max()),
        "top5_weight_mass": float(sorted_weights[: min(5, len(sorted_weights))].sum()),
        "top10_weight_mass": float(sorted_weights[: min(10, len(sorted_weights))].sum()),
        "mean_rss": float(cloud["mean_rss"]),
        "merged_rss": float(cloud["mean_rss"]),
        "trace_cov": float(cloud["trace_cov"]),
        "effective_rank": float(cloud["effective_rank"]),
        "top_eigvals": cloud["top_eigvals"],
        "subspace_rank": len(cloud["principal_directions"]),
        "kappa": float(cloud["kappa"]),
        "direction_spec": cloud["direction_spec"],
        "mean_spec": cloud["mean_spec"],
        "principal_directions": cloud["principal_directions"],
        "principal_eigvals": cloud["principal_eigvals"],
        "num_direction_components": int(cloud["num_direction_components"]),
        "basis_size": int(len(cloud["basis"])),
        "elite_indices": positive_indices,
        "elite_direction_specs": positive_specs,
        "weights": [float(value) for value in weights.tolist()],
        "elite_rows": elite_rows,
    }
