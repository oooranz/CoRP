from typing import Dict, Sequence

import numpy as np

from . import collapse_ops, corp_ops as randsoup_ops


def sample_low_rank_population(
    principal_directions: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    principal_eigvals: Sequence[float],
    sigma: float,
    population_size: int,
    iso_lambda: float,
    rank: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    seeds = rng.choice(2**31, size=population_size, replace=False).tolist()
    active_rank = min(int(rank), len(principal_directions), len(principal_eigvals))
    sigma_perp = float(sigma) * np.sqrt(max(float(iso_lambda), 0.0))
    low_rank_scale = float(sigma) * np.sqrt(max(0.0, 1.0 - float(iso_lambda)))

    direction_specs = []
    metadata = []
    for sample_seed in seeds:
        iso_spec = randsoup_ops.make_isotropic_spec(int(sample_seed), float(sigma_perp))
        coeffs = []
        if active_rank > 0 and low_rank_scale > 0.0:
            z = rng.standard_normal(active_rank)
            coeffs = [
                float(low_rank_scale * np.sqrt(max(float(principal_eigvals[idx]), 0.0)) * z[idx])
                for idx in range(active_rank)
            ]
            low_rank_spec = randsoup_ops.combine_direction_specs(principal_directions[:active_rank], coeffs)
            full_spec = randsoup_ops.combine_direction_specs([iso_spec, low_rank_spec])
        else:
            full_spec = iso_spec
        direction_specs.append(full_spec)
        metadata.append(
            {
                "seed": int(sample_seed),
                "sigma_perp": float(sigma_perp),
                "low_rank_coeffs": coeffs,
                "low_rank_coeff_norm": float(np.linalg.norm(coeffs)) if coeffs else 0.0,
                "isotropic_energy": float(sigma_perp ** 2),
                "low_rank_energy": float(sum(value * value for value in coeffs)),
            }
        )
    return direction_specs, metadata


def build_search_update(
    direction_specs: Sequence[Sequence[randsoup_ops.DirectionComponent]],
    candidate_rows: Sequence[dict],
    *,
    current_iso_lambda: float,
    target_ess: float,
    rank: int,
) -> Dict[str, object]:
    scores = [float(row["constructive_score"]) for row in candidate_rows]
    score_arr = np.asarray(scores, dtype=np.float64)
    positive_fraction = float(np.mean(score_arr > 0.0)) if score_arr.size else 0.0
    positive_count = int(np.sum(score_arr > 0.0)) if score_arr.size else 0
    search_target_ess = None
    search_merge = None
    if positive_count > 0:
        search_target_ess = float(min(float(positive_count), max(float(target_ess), 2.0 * float(target_ess))))
        search_merge = collapse_ops.build_positive_recenter_merge(
            direction_specs,
            scores,
            target_ess=search_target_ess,
            rank=rank,
        )

    target_iso_lambda = float(np.clip(0.15 + 0.8 * (1.0 - positive_fraction), 0.15, 0.95))
    next_iso_lambda = float(np.clip(0.5 * float(current_iso_lambda) + 0.5 * target_iso_lambda, 0.05, 0.95))
    return {
        "positive_count": int(positive_count),
        "positive_fraction": float(positive_fraction),
        "search_target_ess": None if search_target_ess is None else float(search_target_ess),
        "iso_lambda_before": float(current_iso_lambda),
        "target_iso_lambda": float(target_iso_lambda),
        "iso_lambda_after": float(next_iso_lambda),
        "merge_info": search_merge,
    }


def compute_stop_reason(
    *,
    accepted: bool,
    accepted_recenters: int,
    max_accepted_recenters: int,
    consecutive_rejects: int,
    recenter_patience: int,
    round_idx: int,
    recenter_attempts: int,
) -> str | None:
    if accepted and accepted_recenters >= max_accepted_recenters:
        return "max_accepted_recenters"
    if not accepted and recenter_patience > 0 and consecutive_rejects >= recenter_patience:
        return "patience_exhausted"
    if round_idx >= recenter_attempts:
        return "max_attempts"
    return None
