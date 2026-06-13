"""Diagnostic algorithm-v1.1 PD-IPM outer/inner continuation driver.

This module is explicit-import only. It uses the existing diagnostic reduced
step as an inner Newton direction provider, adds P-based Armijo trial
selection, and separates fixed-barrier inner iterations from barrier-parameter
outer continuation. It does not change production behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.optimize.condensate_algorithm_v11_direction import (
    AlgorithmV11Direction,
    blend_algorithm_v11_directions,
    build_active_condensate_budget_correction_direction,
    build_linear_budget_total_density_amount_gas_direction,
    build_linear_budget_total_density_restoration_direction,
)
from exogibbs.optimize.condensate_algorithm_v11_filter import (
    select_filter_restoration_trial,
    select_soft_restoration_trial,
)
from exogibbs.optimize.condensate_ipopt_filter import (
    IpoptFilterEntry,
    augment_persistent_filter,
    is_acceptable_to_persistent_filter,
    select_ipopt_h_type_filter_trial,
)
from exogibbs.optimize.condensate_algorithm_v11_merit import (
    compute_algorithm_v11_barrier_penalty_merit,
    select_p_based_armijo_trial,
)
from exogibbs.optimize.pdipm_rgie_cond import (
    PdipmRgieCondensateState,
    _algorithm_v11_residuals,
    _stable_l2_norm,
    build_pdipm_rgie_condensate_state,
    solve_pdipm_rgie_algorithm_v11_reduced_step,
)


@dataclass(frozen=True)
class AlgorithmV11ContinuationReport:
    """Report for diagnostic PD-IPM outer/inner continuation."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    outer_iteration_count: int
    inner_iteration_count: int
    reached_final_barrier: bool
    converged_at_final_barrier: bool
    stopped_reason: str
    initial_epsilon: float
    final_epsilon_target: float
    final_epsilon: float
    tau: float
    barrier_schedule_policy: str
    ipopt_mu_linear_decrease_factor: float
    ipopt_mu_superlinear_decrease_power: float
    ipopt_enable_superlinear_decrease: bool
    center_tolerance_multiplier: float
    final_residual_l2: float
    final_p_merit: float
    final_state: PdipmRgieCondensateState
    outer_records: tuple[Mapping[str, Any], ...]
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_state"] = self.final_state.as_dict()
        return payload


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _residual_norm(
    *,
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    external_condensate_budget: np.ndarray | None,
    gas_stationarity_source: np.ndarray,
    condensate_standard_source: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    rho: np.ndarray,
    qtot: float,
    epsilon: float,
) -> tuple[float, dict[str, float]]:
    residuals = _algorithm_v11_residuals(
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=condensate_standard_source,
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        qtot=qtot,
        epsilon=epsilon,
    )
    components = {
        key: _stable_l2_norm(residuals[key])
        for key in ("gas", "condensate", "budget", "complementarity", "total_density")
    }
    n = np.exp(q)
    m = np.exp(r)
    components["amount_weighted_gas"] = _stable_l2_norm(n * residuals["gas"])
    components["sqrt_amount_weighted_gas"] = _stable_l2_norm(np.sqrt(n) * residuals["gas"])
    components["amount_weighted_condensate"] = float(
        _stable_l2_norm(m * residuals["condensate"])
    )
    components["sqrt_amount_weighted_condensate"] = float(
        _stable_l2_norm(np.sqrt(m) * residuals["condensate"])
    )
    return _stable_l2_norm(residuals["combined"]), components


def _p_merit(
    *,
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    external_condensate_budget: np.ndarray | None,
    gas_stationarity_source: np.ndarray,
    condensate_standard_source: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    qtot: float,
    qtot_reference: float,
    epsilon: float,
    equality_penalty_weight: float,
    total_density_penalty_weight: float,
) -> dict[str, Any]:
    return compute_algorithm_v11_barrier_penalty_merit(
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=condensate_standard_source,
        q=q,
        r=r,
        qtot=qtot,
        qtot_reference=qtot_reference,
        epsilon=epsilon,
        equality_penalty_weight=equality_penalty_weight,
        total_density_penalty_weight=total_density_penalty_weight,
    ).as_dict()


def _weighted_filter_theta(
    components: Mapping[str, float],
    component_weights: Mapping[str, float],
) -> float:
    total = 0.0
    for name, weight in component_weights.items():
        value = float(components.get(name, math.nan))
        weight_value = float(weight)
        if not math.isfinite(value) or not math.isfinite(weight_value) or weight_value < 0.0:
            return math.inf
        total += weight_value * value
    return float(total)


def _strict_barrier_components_met(
    components: Mapping[str, float],
    component_names: Sequence[str],
    threshold: float,
) -> bool:
    limit = float(threshold)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("strict_barrier_update_threshold must be finite and non-negative.")
    for name in component_names:
        value = float(components.get(name, math.inf))
        if not math.isfinite(value) or value > limit:
            return False
    return True


def _center_metric_from_components(
    residual_l2: float,
    components: Mapping[str, float],
    *,
    center_metric_policy: str,
    center_component_weights: Mapping[str, float] | None,
    center_component_scales: Mapping[str, float] | None,
) -> float:
    if center_metric_policy == "raw_l2":
        return float(residual_l2)
    if center_metric_policy == "component_max":
        names = tuple(components)
    elif center_metric_policy == "amount_weighted_kkt_max":
        names = (
            "budget",
            "total_density",
            "amount_weighted_gas",
            "amount_weighted_condensate",
            "complementarity",
        )
    else:
        raise ValueError(
            "center_metric_policy must be raw_l2, component_max, or "
            "amount_weighted_kkt_max."
        )
    values = []
    for name in names:
        value = float(components.get(name, math.inf))
        weight = (
            1.0
            if center_component_weights is None
            else float(center_component_weights.get(name, 1.0))
        )
        scale = (
            1.0
            if center_component_scales is None
            else float(center_component_scales.get(name, 1.0))
        )
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        values.append(abs(weight) * value / scale)
    return float(max(values)) if values else math.inf


def _next_barrier_epsilon(
    *,
    epsilon: float,
    final_epsilon: float,
    tau: float,
    barrier_schedule_policy: str,
    ipopt_mu_linear_decrease_factor: float,
    ipopt_mu_superlinear_decrease_power: float,
    ipopt_enable_superlinear_decrease: bool,
) -> float:
    if barrier_schedule_policy == "fixed_tau":
        return float(max(final_epsilon, epsilon + math.log(tau)))
    if barrier_schedule_policy != "ipopt_like_monotone":
        raise ValueError("barrier_schedule_policy must be fixed_tau or ipopt_like_monotone.")
    mu = math.exp(float(epsilon))
    target_mu = math.exp(float(final_epsilon))
    linear_mu = float(ipopt_mu_linear_decrease_factor) * mu
    if ipopt_enable_superlinear_decrease:
        superlinear_mu = mu ** float(ipopt_mu_superlinear_decrease_power)
        candidate_mu = min(linear_mu, superlinear_mu)
    else:
        candidate_mu = linear_mu
    new_mu = max(target_mu, candidate_mu)
    if not math.isfinite(new_mu) or new_mu <= 0.0:
        raise ValueError("computed barrier parameter must be finite and positive.")
    return float(math.log(new_mu))


def _select_dedicated_restoration_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_components: Mapping[str, float],
    component_weights: Mapping[str, float],
    max_proximity: float | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    current_theta = _weighted_filter_theta(current_components, component_weights)
    proximity_limit = None if max_proximity is None else float(max_proximity)
    if proximity_limit is not None and (
        not math.isfinite(proximity_limit) or proximity_limit < 0.0
    ):
        raise ValueError("dedicated_restoration_max_proximity must be finite and non-negative.")
    accepted: list[dict[str, Any]] = []
    finite_count = 0
    rejected_count = 0
    for index, trial in enumerate(trials):
        try:
            theta = _weighted_filter_theta(trial["residual_components"], component_weights)
            proximity = float(trial.get("proximity", 0.0))
            alpha = float(trial["alpha"])
        except (KeyError, TypeError, ValueError):
            rejected_count += 1
            continue
        finite = bool(
            trial.get("all_finite", True)
            and math.isfinite(theta)
            and math.isfinite(proximity)
            and math.isfinite(alpha)
        )
        if not finite:
            rejected_count += 1
            continue
        finite_count += 1
        proximity_accepts = proximity_limit is None or proximity <= proximity_limit
        theta_accepts = theta < current_theta
        if theta_accepts and proximity_accepts:
            accepted.append(
                {
                    "index": index,
                    "alpha": alpha,
                    "theta": theta,
                    "theta_ratio": theta / max(current_theta, 1.0e-300),
                    "proximity": proximity,
                }
            )
    if accepted:
        selected = min(accepted, key=lambda row: (float(row["theta"]), -float(row["alpha"])))
        return trials[int(selected["index"])], {
            "selection_schema": "exogibbs_dedicated_restoration_filter_selection_v1",
            "selected": True,
            "selected_index": int(selected["index"]),
            "selected_alpha": float(selected["alpha"]),
            "selected_theta": float(selected["theta"]),
            "current_theta": float(current_theta),
            "selected_theta_ratio": float(selected["theta_ratio"]),
            "selected_proximity": float(selected["proximity"]),
            "finite_trial_count": finite_count,
            "accepted_trial_count": len(accepted),
            "rejected_trial_count": rejected_count,
            "component_weights": dict(component_weights),
            "max_proximity": proximity_limit,
            "selected_reason": "dedicated_restoration_theta_progress",
            "diagnostic_only": True,
            "production_behavior_change": False,
        }
    return None, {
        "selection_schema": "exogibbs_dedicated_restoration_filter_selection_v1",
        "selected": False,
        "selected_index": None,
        "selected_alpha": None,
        "selected_theta": None,
        "current_theta": float(current_theta),
        "selected_theta_ratio": None,
        "selected_proximity": None,
        "finite_trial_count": finite_count,
        "accepted_trial_count": 0,
        "rejected_trial_count": rejected_count,
        "component_weights": dict(component_weights),
        "max_proximity": proximity_limit,
        "selected_reason": None,
        "diagnostic_only": True,
        "production_behavior_change": False,
    }


def fraction_to_boundary_alpha(
    *,
    delta_r: Sequence[float],
    delta_rho: Sequence[float],
    gamma: float = 0.995,
) -> float:
    """Return a diagnostic physical-space fraction-to-boundary alpha."""

    dr = _as_vector(delta_r, "delta_r")
    drho = _as_vector(delta_rho, "delta_rho")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0 or gamma_value >= 1.0:
        raise ValueError("gamma must be finite and in the interval (0, 1).")
    candidates = [1.0]
    negative_dr = dr[dr < 0.0]
    if negative_dr.size:
        candidates.append(float(np.min(-1.0 / negative_dr)))
    negative_drho = drho[drho < 0.0]
    if negative_drho.size:
        candidates.append(float(np.min(-1.0 / negative_drho)))
    return float(max(0.0, min(1.0, gamma_value * min(candidates))))


def _apply_condensate_capacity_cap(
    *,
    r: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
) -> tuple[np.ndarray, int]:
    with np.errstate(divide="ignore", invalid="ignore"):
        per_element_limits = np.where(
            formula_matrix_cond_active > 0.0,
            element_inventory_target[:, None] / formula_matrix_cond_active,
            np.inf,
        )
    cond_capacity = np.min(per_element_limits, axis=0)
    finite_positive_capacity = np.isfinite(cond_capacity) & (cond_capacity > 0.0)
    if not np.any(finite_positive_capacity):
        return r, 0
    log_capacity = np.full_like(r, np.inf)
    log_capacity[finite_positive_capacity] = np.log(
        cond_capacity[finite_positive_capacity]
    )
    capped = np.minimum(r, log_capacity)
    return capped, int(np.count_nonzero(capped < r))


def _trial_rows(
    *,
    alpha_grid: Sequence[float],
    alpha_max: float,
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    external_condensate_budget: np.ndarray | None,
    gas_stationarity_source: np.ndarray,
    condensate_standard_source: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    rho: np.ndarray,
    qtot: float,
    epsilon: float,
    delta_q: np.ndarray,
    delta_r: np.ndarray,
    delta_lambda: np.ndarray,
    delta_rho: np.ndarray,
    delta_qtot: float,
    equality_penalty_weight: float,
    total_density_penalty_weight: float,
    enforce_condensate_capacity: bool = False,
) -> list[dict[str, Any]]:
    alphas = sorted(
        {
            float(alpha)
            for alpha in alpha_grid
            if 0.0 < float(alpha) <= max(alpha_max, 0.0) and math.isfinite(float(alpha))
        },
        reverse=True,
    )
    if alpha_max > 0.0 and alpha_max not in alphas:
        alphas.append(float(alpha_max))
        alphas = sorted(set(alphas), reverse=True)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        q_trial = q + alpha * delta_q
        r_trial = r + alpha * delta_r
        capacity_cap_count = 0
        if enforce_condensate_capacity:
            r_trial, capacity_cap_count = _apply_condensate_capacity_cap(
                r=r_trial,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
            )
        lam_trial = lam + alpha * delta_lambda
        rho_trial = rho + alpha * delta_rho
        qtot_trial = float(qtot + alpha * delta_qtot)
        residual_l2, residual_components = _residual_norm(
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            external_condensate_budget=external_condensate_budget,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=condensate_standard_source,
            q=q_trial,
            r=r_trial,
            lam=lam_trial,
            rho=rho_trial,
            qtot=qtot_trial,
            epsilon=epsilon,
        )
        merit = _p_merit(
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            external_condensate_budget=external_condensate_budget,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=condensate_standard_source,
            q=q_trial,
            r=r_trial,
            qtot=qtot_trial,
            qtot_reference=qtot,
            epsilon=epsilon,
            equality_penalty_weight=equality_penalty_weight,
            total_density_penalty_weight=total_density_penalty_weight,
        )
        rows.append(
            {
                "alpha": float(alpha),
                "p_merit": float(merit["total_merit"]),
                "p_merit_breakdown": merit,
                "residual_l2": residual_l2,
                "residual_components": residual_components,
                "all_finite": bool(merit["finite"] and math.isfinite(residual_l2)),
                "condensate_capacity_cap_count": capacity_cap_count,
            }
        )
    return rows


def _make_direction_from_step(
    *,
    delta_q: np.ndarray,
    delta_r: np.ndarray,
    delta_lambda: np.ndarray,
    delta_rho: np.ndarray,
    delta_qtot: float,
    direction_kind: str,
) -> AlgorithmV11Direction:
    return AlgorithmV11Direction(
        delta_q=np.asarray(delta_q, dtype=np.float64),
        delta_r=np.asarray(delta_r, dtype=np.float64),
        delta_lambda=np.asarray(delta_lambda, dtype=np.float64),
        delta_rho=np.asarray(delta_rho, dtype=np.float64),
        delta_qtot=float(delta_qtot),
        direction_kind=direction_kind,
    )


def _budget_total_density_null_projected_gas_direction(
    *,
    formula_matrix: np.ndarray,
    gas_stationarity_source: np.ndarray,
    q: np.ndarray,
    lam: np.ndarray,
    r_size: int,
    rho_size: int,
    max_abs_delta_q: float,
) -> AlgorithmV11Direction:
    gas_residual = q + gas_stationarity_source - formula_matrix.T @ lam
    raw_direction = -gas_residual
    n = np.exp(q)
    constraint = np.vstack([formula_matrix * n[np.newaxis, :], n[np.newaxis, :]])
    gram = constraint @ constraint.T
    rhs = constraint @ raw_direction
    try:
        multiplier = np.linalg.solve(gram + 1.0e-20 * np.eye(gram.shape[0]), rhs)
    except np.linalg.LinAlgError:
        multiplier = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    projected = raw_direction - constraint.T @ multiplier
    norm_inf = float(np.max(np.abs(projected))) if projected.size else 0.0
    if not math.isfinite(norm_inf) or norm_inf == 0.0:
        delta_q = np.zeros_like(projected)
    else:
        delta_q = projected * min(1.0, float(max_abs_delta_q) / norm_inf)
    return AlgorithmV11Direction(
        delta_q=np.asarray(delta_q, dtype=np.float64),
        delta_r=np.zeros(r_size, dtype=np.float64),
        delta_lambda=np.zeros(lam.shape[0], dtype=np.float64),
        delta_rho=np.zeros(rho_size, dtype=np.float64),
        delta_qtot=0.0,
        direction_kind="budget_total_density_null_projected_gas_stationarity_direction",
    )


def _direction_candidates(
    *,
    direction_policy: str,
    algorithm_fraction_grid: Sequence[float],
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    external_condensate_budget: np.ndarray | None,
    gas_stationarity_source: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    rho: np.ndarray,
    qtot: float,
    delta_q: np.ndarray,
    delta_r: np.ndarray,
    delta_lambda: np.ndarray,
    delta_rho: np.ndarray,
    delta_qtot: float,
    max_abs_delta_q: float,
    max_abs_delta_r: float,
) -> list[AlgorithmV11Direction]:
    algorithm_direction = _make_direction_from_step(
        delta_q=delta_q,
        delta_r=delta_r,
        delta_lambda=delta_lambda,
        delta_rho=delta_rho,
        delta_qtot=delta_qtot,
        direction_kind="algorithm_v11_reduced_direction",
    )
    if direction_policy == "algorithm_v11_reduced":
        return [algorithm_direction]
    gas_direction = _budget_total_density_null_projected_gas_direction(
        formula_matrix=formula_matrix,
        gas_stationarity_source=gas_stationarity_source,
        q=q,
        lam=lam,
        r_size=r.shape[0],
        rho_size=rho.shape[0],
        max_abs_delta_q=max_abs_delta_q,
    )
    if direction_policy == "budget_safe_gas_stationarity":
        return [gas_direction]
    if direction_policy == "algorithm_v11_with_budget_safe_gas_stationarity":
        return [algorithm_direction, gas_direction]
    condensate_budget_direction = build_active_condensate_budget_correction_direction(
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        q=q,
        r=r,
        lambda_size=lam.shape[0],
        rho_size=rho.shape[0],
        max_abs_delta_r=max_abs_delta_r,
    )
    relative_condensate_budget_direction = (
        build_active_condensate_budget_correction_direction(
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            external_condensate_budget=external_condensate_budget,
            q=q,
            r=r,
            lambda_size=lam.shape[0],
            rho_size=rho.shape[0],
            max_abs_delta_r=max_abs_delta_r,
            relative_budget_weighting=True,
            enforce_condensate_capacity=True,
        )
    )
    if direction_policy == "condensate_budget_correction":
        return [condensate_budget_direction]
    if direction_policy == "algorithm_v11_with_condensate_budget_correction":
        return [algorithm_direction, condensate_budget_direction]
    if direction_policy == "relative_condensate_budget_correction":
        return [relative_condensate_budget_direction]
    if direction_policy == "algorithm_v11_with_relative_condensate_budget_correction":
        return [algorithm_direction, relative_condensate_budget_direction]
    if direction_policy == "algorithm_v11_gas_stationarity_blend":
        candidates: list[AlgorithmV11Direction] = []
        for beta in algorithm_fraction_grid:
            beta_value = float(beta)
            candidates.append(
                AlgorithmV11Direction(
                    delta_q=beta_value * algorithm_direction.delta_q
                    + (1.0 - beta_value) * gas_direction.delta_q,
                    delta_r=beta_value * algorithm_direction.delta_r
                    + (1.0 - beta_value) * gas_direction.delta_r,
                    delta_lambda=beta_value * algorithm_direction.delta_lambda
                    + (1.0 - beta_value) * gas_direction.delta_lambda,
                    delta_rho=beta_value * algorithm_direction.delta_rho
                    + (1.0 - beta_value) * gas_direction.delta_rho,
                    delta_qtot=beta_value * algorithm_direction.delta_qtot
                    + (1.0 - beta_value) * gas_direction.delta_qtot,
                    direction_kind=(
                        "algorithm_v11_gas_stationarity_blend_"
                        f"algorithm_fraction_{beta_value:.6g}"
                    ),
                )
            )
        return candidates
    if direction_policy == "joint_budget_amount_gas_linearized":
        return [
            build_linear_budget_total_density_amount_gas_direction(
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
                external_condensate_budget=external_condensate_budget,
                gas_stationarity_source=gas_stationarity_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                target_direction=algorithm_direction,
                max_abs_delta_q=max_abs_delta_q,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_linearized_no_prior":
        return [
            build_linear_budget_total_density_amount_gas_direction(
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
                external_condensate_budget=external_condensate_budget,
                gas_stationarity_source=gas_stationarity_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                target_direction=None,
                target_direction_weight=0.0,
                max_abs_delta_q=max_abs_delta_q,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_condensate_prior":
        return [
            build_linear_budget_total_density_amount_gas_direction(
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
                external_condensate_budget=external_condensate_budget,
                gas_stationarity_source=gas_stationarity_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                target_direction=algorithm_direction,
                target_direction_weight=1.0e-2,
                target_direction_component_mode="condensate_dual_only",
                max_abs_delta_q=max_abs_delta_q,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_condensate_prior_budget_strong":
        return [
            build_linear_budget_total_density_amount_gas_direction(
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
                external_condensate_budget=external_condensate_budget,
                gas_stationarity_source=gas_stationarity_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                target_direction=algorithm_direction,
                budget_weight=100.0,
                total_density_weight=100.0,
                amount_gas_weight=1.0,
                target_direction_weight=1.0e-2,
                target_direction_component_mode="condensate_dual_only",
                max_abs_delta_q=max_abs_delta_q,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_condensate_prior_balanced":
        return [
            build_linear_budget_total_density_amount_gas_direction(
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
                external_condensate_budget=external_condensate_budget,
                gas_stationarity_source=gas_stationarity_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                target_direction=algorithm_direction,
                budget_weight=30.0,
                total_density_weight=30.0,
                amount_gas_weight=3.0,
                target_direction_weight=1.0e-2,
                target_direction_component_mode="condensate_dual_only",
                max_abs_delta_q=max_abs_delta_q,
            )
        ]
    if direction_policy != "residual_norm_blend":
        raise ValueError(
            "direction_policy must be algorithm_v11_reduced, residual_norm_blend, "
            "budget_safe_gas_stationarity, or "
            "algorithm_v11_with_budget_safe_gas_stationarity, or "
            "condensate_budget_correction, or "
            "algorithm_v11_with_condensate_budget_correction, or "
            "relative_condensate_budget_correction, or "
            "algorithm_v11_with_relative_condensate_budget_correction, or "
            "algorithm_v11_gas_stationarity_blend, or "
            "joint_budget_amount_gas_linearized, or "
            "joint_budget_amount_gas_linearized_no_prior, or "
            "joint_budget_amount_gas_condensate_prior, or "
            "joint_budget_amount_gas_condensate_prior_budget_strong, or "
            "joint_budget_amount_gas_condensate_prior_balanced."
        )

    restoration_direction = build_linear_budget_total_density_restoration_direction(
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        q=q,
        r=r,
        lambda_size=lam.shape[0],
        rho_size=rho.shape[0],
        qtot=qtot,
    )
    candidates: list[AlgorithmV11Direction] = []
    for beta in algorithm_fraction_grid:
        blended = blend_algorithm_v11_directions(
            algorithm_direction=algorithm_direction,
            restoration_direction=restoration_direction,
            algorithm_fraction=float(beta),
        )
        candidates.append(
            AlgorithmV11Direction(
                delta_q=blended.delta_q,
                delta_r=blended.delta_r,
                delta_lambda=blended.delta_lambda,
                delta_rho=blended.delta_rho,
                delta_qtot=blended.delta_qtot,
                direction_kind=f"residual_norm_blend_algorithm_fraction_{float(beta):.6g}",
            )
        )
    return candidates


def run_algorithm_v11_pdipm_continuation(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    initial_epsilon: float,
    final_epsilon: float,
    tau: float = 0.5,
    barrier_schedule_policy: str = "fixed_tau",
    ipopt_mu_linear_decrease_factor: float = 0.2,
    ipopt_mu_superlinear_decrease_power: float = 1.5,
    ipopt_enable_superlinear_decrease: bool = True,
    max_outer_iterations: int = 4,
    max_inner_iterations: int = 6,
    center_tolerance_multiplier: float = 10.0,
    alpha_grid: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625, 1.0e-2, 1.0e-3),
    armijo_c1: float = 1.0e-4,
    fraction_to_boundary_gamma: float = 0.995,
    equality_penalty_weight: float = 100.0,
    total_density_penalty_weight: float = 100.0,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_rho: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    direction_policy: str = "algorithm_v11_reduced",
    algorithm_fraction_grid: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    trial_acceptance_policy: str = "p_armijo_or_best_residual",
    filter_component_weights: Mapping[str, float] | None = None,
    filter_component_scale_policy: str = "none",
    ipopt_h_type_component_weights: Mapping[str, float] | None = None,
    ipopt_h_type_theta_reduction_fraction: float = 1.0e-4,
    ipopt_h_type_protected_components: Sequence[str] = (),
    ipopt_h_type_protected_component_max_normalized_increase: float | None = None,
    persistent_filter_gamma_p: float = 1.0e-8,
    persistent_filter_gamma_theta: float = 1.0e-5,
    persistent_filter_theta_max_factor: float = 1.0e4,
    strict_barrier_update_components: Sequence[str] = (),
    strict_barrier_update_threshold: float = 1.0e-6,
    center_metric_policy: str = "raw_l2",
    center_component_weights: Mapping[str, float] | None = None,
    center_component_scales: Mapping[str, float] | None = None,
    enable_native_soft_restoration_fallback: bool = False,
    soft_restoration_component_weights: Mapping[str, float] | None = None,
    soft_restoration_proximity_weight: float = 1.0e-2,
    soft_restoration_max_proximity: float | None = 10.0,
    enable_dedicated_restoration_filter_phase: bool = False,
    dedicated_restoration_component_weights: Mapping[str, float] | None = None,
    dedicated_restoration_max_proximity: float | None = 10.0,
    require_residual_nonworsening: bool = False,
    residual_worsening_tolerance: float = 0.0,
) -> AlgorithmV11ContinuationReport:
    """Run diagnostic outer/inner PD-IPM continuation for algorithm-v1.1."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for continuation diagnostics.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    if state.rho is None:
        raise ValueError("state.rho is required for continuation diagnostics.")
    eps = float(initial_epsilon)
    eps_final = float(final_epsilon)
    tau_value = float(tau)
    schedule_policy = str(barrier_schedule_policy)
    ipopt_linear_factor = float(ipopt_mu_linear_decrease_factor)
    ipopt_superlinear_power = float(ipopt_mu_superlinear_decrease_power)
    ipopt_superlinear_enabled = bool(ipopt_enable_superlinear_decrease)
    if not math.isfinite(eps) or not math.isfinite(eps_final):
        raise ValueError("epsilon values must be finite.")
    if eps_final > eps:
        raise ValueError("final_epsilon must be less than or equal to initial_epsilon.")
    if not math.isfinite(tau_value) or tau_value <= 0.0 or tau_value >= 1.0:
        raise ValueError("tau must be finite and in the interval (0, 1).")
    if schedule_policy not in {"fixed_tau", "ipopt_like_monotone"}:
        raise ValueError("barrier_schedule_policy must be fixed_tau or ipopt_like_monotone.")
    if (
        not math.isfinite(ipopt_linear_factor)
        or ipopt_linear_factor <= 0.0
        or ipopt_linear_factor >= 1.0
    ):
        raise ValueError(
            "ipopt_mu_linear_decrease_factor must be finite and in the interval (0, 1)."
        )
    if (
        not math.isfinite(ipopt_superlinear_power)
        or ipopt_superlinear_power <= 1.0
        or ipopt_superlinear_power > 2.0
    ):
        raise ValueError(
            "ipopt_mu_superlinear_decrease_power must be finite and in the interval (1, 2]."
        )
    if max_outer_iterations < 1 or max_inner_iterations < 1:
        raise ValueError("iteration limits must be positive.")
    if trial_acceptance_policy not in {
        "p_armijo_or_best_residual",
        "filter_restoration",
        "ipopt_fastchem4_h_type",
        "ipopt_persistent_h_type",
    }:
        raise ValueError(
            "trial_acceptance_policy must be p_armijo_or_best_residual, "
            "filter_restoration, ipopt_fastchem4_h_type, or ipopt_persistent_h_type."
        )
    if filter_component_scale_policy not in {"none", "current"}:
        raise ValueError("filter_component_scale_policy must be none or current.")
    soft_proximity_weight = float(soft_restoration_proximity_weight)
    if not math.isfinite(soft_proximity_weight) or soft_proximity_weight < 0.0:
        raise ValueError("soft_restoration_proximity_weight must be finite and non-negative.")
    soft_max_proximity = (
        None if soft_restoration_max_proximity is None else float(soft_restoration_max_proximity)
    )
    if soft_max_proximity is not None and (
        not math.isfinite(soft_max_proximity) or soft_max_proximity < 0.0
    ):
        raise ValueError("soft_restoration_max_proximity must be finite and non-negative.")
    dedicated_max_proximity = (
        None
        if dedicated_restoration_max_proximity is None
        else float(dedicated_restoration_max_proximity)
    )
    if dedicated_max_proximity is not None and (
        not math.isfinite(dedicated_max_proximity) or dedicated_max_proximity < 0.0
    ):
        raise ValueError("dedicated_restoration_max_proximity must be finite and non-negative.")
    protected_max_increase = (
        None
        if ipopt_h_type_protected_component_max_normalized_increase is None
        else float(ipopt_h_type_protected_component_max_normalized_increase)
    )
    h_type_theta_fraction = float(ipopt_h_type_theta_reduction_fraction)
    if (
        not math.isfinite(h_type_theta_fraction)
        or h_type_theta_fraction < 0.0
        or h_type_theta_fraction >= 1.0
    ):
        raise ValueError(
            "ipopt_h_type_theta_reduction_fraction must be finite and in [0, 1)."
        )
    if protected_max_increase is not None and (
        not math.isfinite(protected_max_increase) or protected_max_increase < 0.0
    ):
        raise ValueError(
            "ipopt_h_type_protected_component_max_normalized_increase must be finite "
            "and non-negative."
        )
    persistent_gamma_p = float(persistent_filter_gamma_p)
    persistent_gamma_theta = float(persistent_filter_gamma_theta)
    persistent_theta_max_factor = float(persistent_filter_theta_max_factor)
    if not math.isfinite(persistent_gamma_p) or persistent_gamma_p < 0.0:
        raise ValueError("persistent_filter_gamma_p must be finite and non-negative.")
    if not math.isfinite(persistent_gamma_theta) or persistent_gamma_theta < 0.0:
        raise ValueError("persistent_filter_gamma_theta must be finite and non-negative.")
    if not math.isfinite(persistent_theta_max_factor) or persistent_theta_max_factor <= 0.0:
        raise ValueError("persistent_filter_theta_max_factor must be finite and positive.")
    strict_barrier_names = tuple(str(name) for name in strict_barrier_update_components)
    strict_barrier_threshold = float(strict_barrier_update_threshold)
    if not math.isfinite(strict_barrier_threshold) or strict_barrier_threshold < 0.0:
        raise ValueError("strict_barrier_update_threshold must be finite and non-negative.")
    if center_metric_policy not in {
        "raw_l2",
        "component_max",
        "amount_weighted_kkt_max",
    }:
        raise ValueError(
            "center_metric_policy must be raw_l2, component_max, or "
            "amount_weighted_kkt_max."
        )
    residual_tol = float(residual_worsening_tolerance)
    if not math.isfinite(residual_tol) or residual_tol < 0.0:
        raise ValueError("residual_worsening_tolerance must be finite and non-negative.")
    enforce_trial_condensate_capacity = True

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    cond_source = _as_vector(condensate_standard_source, "condensate_standard_source")
    q = _as_vector(state.ln_nk, "state.ln_nk")
    r = _as_vector(state.ln_mk, "state.ln_mk")
    lam = _as_vector(state.element_potential, "state.element_potential")
    rho = _as_vector(state.rho, "state.rho")
    qtot = float(state.ln_ntot)
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    outer_records: list[Mapping[str, Any]] = []
    inner_count = 0
    stopped_reason = "max_outer_iterations_reached"
    reached_final = False
    converged_final = False
    persistent_filter_entries: tuple[IpoptFilterEntry, ...] = ()
    persistent_filter_reference_theta: float | None = None

    for outer_index in range(int(max_outer_iterations)):
        nu = float(np.exp(eps))
        center_threshold = float(center_tolerance_multiplier * nu)
        inner_records: list[Mapping[str, Any]] = []
        for inner_index in range(int(max_inner_iterations)):
            current_residual_l2, current_components = _residual_norm(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                external_condensate_budget=external_budget,
                gas_stationarity_source=gas_source,
                condensate_standard_source=cond_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                epsilon=eps,
            )
            current_p = _p_merit(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                external_condensate_budget=external_budget,
                gas_stationarity_source=gas_source,
                condensate_standard_source=cond_source,
                q=q,
                r=r,
                qtot=qtot,
                qtot_reference=qtot,
                epsilon=eps,
                equality_penalty_weight=equality_penalty_weight,
                total_density_penalty_weight=total_density_penalty_weight,
            )
            current_center_metric = _center_metric_from_components(
                current_residual_l2,
                current_components,
                center_metric_policy=center_metric_policy,
                center_component_weights=center_component_weights,
                center_component_scales=center_component_scales,
            )
            if trial_acceptance_policy == "ipopt_persistent_h_type":
                weights = ipopt_h_type_component_weights or filter_component_weights
                if weights is None:
                    raise ValueError(
                        "ipopt_h_type_component_weights or filter_component_weights "
                        "must be provided for ipopt_persistent_h_type."
                    )
                current_theta = _weighted_filter_theta(current_components, weights)
                if persistent_filter_reference_theta is None:
                    persistent_filter_reference_theta = current_theta
                if not persistent_filter_entries:
                    persistent_filter_entries = augment_persistent_filter(
                        persistent_filter_entries,
                        p_merit=float(current_p["total_merit"]),
                        theta=current_theta,
                        iteration=-1,
                        gamma_p=persistent_gamma_p,
                        gamma_theta=persistent_gamma_theta,
                    )
            if current_center_metric <= center_threshold:
                inner_records.append(
                    {
                        "inner_index": inner_index,
                        "status": "center_threshold_satisfied_before_step",
                        "residual_l2": current_residual_l2,
                        "center_metric": current_center_metric,
                        "center_metric_policy": center_metric_policy,
                        "center_component_weights": None
                        if center_component_weights is None
                        else dict(center_component_weights),
                        "center_component_scales": None
                        if center_component_scales is None
                        else dict(center_component_scales),
                        "center_threshold": center_threshold,
                        "p_merit": current_p["total_merit"],
                    }
                )
                break

            current_state = build_pdipm_rgie_condensate_state(
                ln_nk=q,
                ln_mk=r,
                element_potential=lam,
                ln_ntot=qtot,
                rho=rho,
                eta=np.exp(rho),
                field_provenance=state.field_provenance,
            )
            step = solve_pdipm_rgie_algorithm_v11_reduced_step(
                explicit_opt_in=True,
                state=current_state,
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                external_condensate_budget=external_budget,
                gas_stationarity_source=gas_source,
                condensate_standard_source=cond_source,
                epsilon=eps,
                alpha_candidates=(1.0,),
                max_abs_delta_q=max_abs_delta_q,
                max_abs_delta_r=max_abs_delta_r,
                max_abs_delta_rho=max_abs_delta_rho,
                max_abs_delta_lambda=max_abs_delta_lambda,
            )
            step_dict = step.as_dict()
            delta_q = np.asarray(step_dict["delta_q"], dtype=np.float64)
            delta_r = np.asarray(step_dict["delta_r"], dtype=np.float64)
            delta_lambda = np.asarray(step_dict["delta_lambda"], dtype=np.float64)
            delta_rho = np.asarray(step_dict["delta_rho"], dtype=np.float64)
            delta_qtot = float(step_dict["delta_qtot"])
            direction_candidates = _direction_candidates(
                direction_policy=direction_policy,
                algorithm_fraction_grid=algorithm_fraction_grid,
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                external_condensate_budget=external_budget,
                gas_stationarity_source=gas_source,
                q=q,
                r=r,
                lam=lam,
                rho=rho,
                qtot=qtot,
                delta_q=delta_q,
                delta_r=delta_r,
                delta_lambda=delta_lambda,
                delta_rho=delta_rho,
                delta_qtot=delta_qtot,
                max_abs_delta_q=max_abs_delta_q,
                max_abs_delta_r=max_abs_delta_r,
            )
            direction_records: list[dict[str, Any]] = []
            selected_direction: AlgorithmV11Direction | None = None
            selected_trial: Mapping[str, Any] | None = None
            selected_score: tuple[float, float] | None = None
            selected_selection = None
            for direction in direction_candidates:
                alpha_max = fraction_to_boundary_alpha(
                    delta_r=direction.delta_r,
                    delta_rho=direction.delta_rho,
                    gamma=fraction_to_boundary_gamma,
                )
                trials = _trial_rows(
                    alpha_grid=alpha_grid,
                    alpha_max=alpha_max,
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    external_condensate_budget=external_budget,
                    gas_stationarity_source=gas_source,
                    condensate_standard_source=cond_source,
                    q=q,
                    r=r,
                    lam=lam,
                    rho=rho,
                    qtot=qtot,
                    epsilon=eps,
                    delta_q=direction.delta_q,
                    delta_r=direction.delta_r,
                    delta_lambda=direction.delta_lambda,
                    delta_rho=direction.delta_rho,
                    delta_qtot=direction.delta_qtot,
                    equality_penalty_weight=equality_penalty_weight,
                    total_density_penalty_weight=total_density_penalty_weight,
                    enforce_condensate_capacity=enforce_trial_condensate_capacity,
                )
                selection = select_p_based_armijo_trial(
                    trials,
                    current_merit=float(current_p["total_merit"]),
                    directional_derivative=None,
                    c1=armijo_c1,
                    choose_largest_alpha=True,
                )
                filter_component_scales = (
                    current_components if filter_component_scale_policy == "current" else None
                )
                filter_selection = select_filter_restoration_trial(
                    trials,
                    current_merit=float(current_p["total_merit"]),
                    current_components=current_components,
                    component_weights=filter_component_weights,
                    component_scales=filter_component_scales,
                    choose_largest_alpha=True,
                )
                ipopt_h_type_selection = None
                ipopt_h_type_trial = None
                persistent_filter_report = None
                persistent_filter_theta = None
                persistent_filter_theta_max = None
                persistent_filter_entry_count_before = len(persistent_filter_entries)
                if trial_acceptance_policy in {
                    "ipopt_fastchem4_h_type",
                    "ipopt_persistent_h_type",
                }:
                    weights = ipopt_h_type_component_weights or filter_component_weights
                    if weights is None:
                        raise ValueError(
                            "ipopt_h_type_component_weights or filter_component_weights "
                            "must be provided for ipopt h-type trial policies."
                        )
                    ipopt_h_type_selection = select_ipopt_h_type_filter_trial(
                        trials,
                        current_components=current_components,
                        current_p_merit=float(current_p["total_merit"]),
                        component_weights=weights,
                        theta_reduction_fraction=h_type_theta_fraction,
                        protected_components=ipopt_h_type_protected_components,
                        protected_component_max_normalized_increase=protected_max_increase,
                        choose_largest_alpha=True,
                    )
                    ipopt_h_type_trial = (
                        trials[ipopt_h_type_selection.selected_index]
                        if ipopt_h_type_selection.selected
                        and ipopt_h_type_selection.selected_index is not None
                        else None
                    )
                    if (
                        trial_acceptance_policy == "ipopt_persistent_h_type"
                        and ipopt_h_type_trial is not None
                    ):
                        persistent_filter_theta = _weighted_filter_theta(
                            ipopt_h_type_trial["residual_components"],
                            weights,
                        )
                        reference_theta = (
                            1.0
                            if persistent_filter_reference_theta is None
                            else max(1.0, float(persistent_filter_reference_theta))
                        )
                        persistent_filter_theta_max = max(
                            1.0,
                            persistent_theta_max_factor * reference_theta,
                        )
                        persistent_filter_report = is_acceptable_to_persistent_filter(
                            p_merit=float(ipopt_h_type_trial["p_merit"]),
                            theta=persistent_filter_theta,
                            entries=persistent_filter_entries,
                            gamma_p=persistent_gamma_p,
                            gamma_theta=persistent_gamma_theta,
                            theta_max=persistent_filter_theta_max,
                        )
                        if not persistent_filter_report.acceptable:
                            ipopt_h_type_trial = None
                candidate_trial = (
                    trials[selection.selected_index]
                    if selection.selected and selection.selected_index is not None
                    else None
                )
                filter_trial = (
                    trials[filter_selection.selected_index]
                    if filter_selection.selected and filter_selection.selected_index is not None
                    else None
                )
                finite_trials = [trial for trial in trials if trial["all_finite"]]
                best_residual_trial = (
                    min(finite_trials, key=lambda trial: trial["residual_l2"])
                    if finite_trials
                    else None
                )
                direction_record = {
                    "direction_kind": direction.direction_kind,
                    "fraction_to_boundary_alpha": alpha_max,
                    "p_armijo_selection": selection.as_dict(),
                    "filter_selection": filter_selection.as_dict(),
                    "ipopt_h_type_selection": None
                    if ipopt_h_type_selection is None
                    else ipopt_h_type_selection.as_dict(),
                    "selected_trial": candidate_trial,
                    "best_residual_trial": best_residual_trial,
                    "filter_trial": filter_trial,
                    "ipopt_h_type_trial": ipopt_h_type_trial,
                    "persistent_filter_entry_count_before": persistent_filter_entry_count_before,
                    "persistent_filter_theta": persistent_filter_theta,
                    "persistent_filter_theta_max": persistent_filter_theta_max,
                    "persistent_filter_report": None
                    if persistent_filter_report is None
                    else persistent_filter_report.as_dict(),
                }
                direction_records.append(direction_record)
                if trial_acceptance_policy == "ipopt_fastchem4_h_type":
                    trial_for_score = ipopt_h_type_trial
                    if trial_for_score is None and filter_trial is not None:
                        trial_for_score = filter_trial
                    if trial_for_score is None and candidate_trial is not None:
                        trial_for_score = candidate_trial
                elif trial_acceptance_policy == "filter_restoration":
                    trial_for_score = filter_trial
                    if trial_for_score is None and candidate_trial is not None:
                        trial_for_score = candidate_trial
                elif trial_acceptance_policy == "ipopt_persistent_h_type":
                    trial_for_score = ipopt_h_type_trial
                else:
                    trial_for_score = (
                        candidate_trial if candidate_trial is not None else best_residual_trial
                    )
                if trial_for_score is None:
                    continue
                if require_residual_nonworsening and not (
                    float(trial_for_score["residual_l2"])
                    <= current_residual_l2 * (1.0 + residual_tol)
                ):
                    continue
                score = (
                    float(trial_for_score["residual_l2"]),
                    float(trial_for_score["p_merit"]),
                )
                if selected_score is None or score < selected_score:
                    selected_score = score
                    selected_direction = direction
                    selected_trial = trial_for_score
                    selected_selection = selection
            record = {
                "inner_index": inner_index,
                "status": "p_armijo_selected" if selected_trial is not None else "no_p_armijo_trial",
                "epsilon": eps,
                "nu": nu,
                "residual_l2_before": current_residual_l2,
                "residual_components_before": current_components,
                "center_metric_before": current_center_metric,
                "center_metric_policy": center_metric_policy,
                "center_component_weights": None
                if center_component_weights is None
                else dict(center_component_weights),
                "center_component_scales": None
                if center_component_scales is None
                else dict(center_component_scales),
                "p_merit_before": current_p["total_merit"],
                "center_threshold": center_threshold,
                "direction_policy": direction_policy,
                "trial_acceptance_policy": trial_acceptance_policy,
                "filter_component_weights": None
                if filter_component_weights is None
                else dict(filter_component_weights),
                "filter_component_scale_policy": filter_component_scale_policy,
                "ipopt_h_type_component_weights": None
                if ipopt_h_type_component_weights is None
                else dict(ipopt_h_type_component_weights),
                "ipopt_h_type_theta_reduction_fraction": h_type_theta_fraction,
                "ipopt_h_type_protected_components": tuple(
                    str(name) for name in ipopt_h_type_protected_components
                ),
                "ipopt_h_type_protected_component_max_normalized_increase": protected_max_increase,
                "persistent_filter_gamma_p": persistent_gamma_p,
                "persistent_filter_gamma_theta": persistent_gamma_theta,
                "persistent_filter_theta_max_factor": persistent_theta_max_factor,
                "persistent_filter_entry_count_before": len(persistent_filter_entries),
                "strict_barrier_update_components": strict_barrier_names,
                "strict_barrier_update_threshold": strict_barrier_threshold,
                "enable_native_soft_restoration_fallback": bool(
                    enable_native_soft_restoration_fallback
                ),
                "soft_restoration_component_weights": None
                if soft_restoration_component_weights is None
                else dict(soft_restoration_component_weights),
                "soft_restoration_proximity_weight": soft_proximity_weight,
                "soft_restoration_max_proximity": soft_max_proximity,
                "enable_dedicated_restoration_filter_phase": bool(
                    enable_dedicated_restoration_filter_phase
                ),
                "dedicated_restoration_component_weights": None
                if dedicated_restoration_component_weights is None
                else dict(dedicated_restoration_component_weights),
                "dedicated_restoration_max_proximity": dedicated_max_proximity,
                "require_residual_nonworsening": require_residual_nonworsening,
                "residual_worsening_tolerance": residual_tol,
                "direction_records": direction_records,
                "selected_direction_kind": None
                if selected_direction is None
                else selected_direction.direction_kind,
                "p_armijo_selection": None
                if selected_selection is None
                else selected_selection.as_dict(),
                "selected_trial": selected_trial,
                "soft_restoration_fallback_selection": None,
                "soft_restoration_fallback_trial": None,
            }
            if selected_trial is None and enable_native_soft_restoration_fallback:
                restoration_direction = build_linear_budget_total_density_restoration_direction(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    external_condensate_budget=external_budget,
                    q=q,
                    r=r,
                    lambda_size=lam.shape[0],
                    rho_size=rho.shape[0],
                    qtot=qtot,
                )
                alpha_max = fraction_to_boundary_alpha(
                    delta_r=restoration_direction.delta_r,
                    delta_rho=restoration_direction.delta_rho,
                    gamma=fraction_to_boundary_gamma,
                )
                restoration_trials = _trial_rows(
                    alpha_grid=alpha_grid,
                    alpha_max=alpha_max,
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    external_condensate_budget=external_budget,
                    gas_stationarity_source=gas_source,
                    condensate_standard_source=cond_source,
                    q=q,
                    r=r,
                    lam=lam,
                    rho=rho,
                    qtot=qtot,
                    epsilon=eps,
                    delta_q=restoration_direction.delta_q,
                    delta_r=restoration_direction.delta_r,
                    delta_lambda=restoration_direction.delta_lambda,
                    delta_rho=restoration_direction.delta_rho,
                    delta_qtot=restoration_direction.delta_qtot,
                    equality_penalty_weight=equality_penalty_weight,
                    total_density_penalty_weight=total_density_penalty_weight,
                    enforce_condensate_capacity=enforce_trial_condensate_capacity,
                )
                proximity_norm = math.sqrt(
                    float(np.linalg.norm(restoration_direction.delta_q)) ** 2
                    + float(np.linalg.norm(restoration_direction.delta_r)) ** 2
                    + float(restoration_direction.delta_qtot) ** 2
                )
                restoration_trials = [
                    {**trial, "proximity": abs(float(trial["alpha"])) * proximity_norm}
                    for trial in restoration_trials
                ]
                dedicated_restoration_filter_selection = None
                if enable_dedicated_restoration_filter_phase:
                    dedicated_weights = (
                        dedicated_restoration_component_weights
                        or soft_restoration_component_weights
                    )
                    if dedicated_weights is None:
                        raise ValueError(
                            "dedicated_restoration_component_weights or "
                            "soft_restoration_component_weights must be provided when "
                            "enable_dedicated_restoration_filter_phase is true."
                        )
                    restoration_trial, dedicated_restoration_filter_selection = (
                        _select_dedicated_restoration_trial(
                            restoration_trials,
                            current_components=current_components,
                            component_weights=dedicated_weights,
                            max_proximity=dedicated_max_proximity,
                        )
                    )
                    restoration_selection = select_soft_restoration_trial(
                        restoration_trials,
                        current_components=current_components,
                        component_weights=soft_restoration_component_weights,
                        component_scales=filter_component_scales,
                        proximity_weight=soft_proximity_weight,
                        max_proximity=soft_max_proximity,
                    )
                else:
                    restoration_selection = select_soft_restoration_trial(
                        restoration_trials,
                        current_components=current_components,
                        component_weights=soft_restoration_component_weights,
                        component_scales=filter_component_scales,
                        proximity_weight=soft_proximity_weight,
                        max_proximity=soft_max_proximity,
                    )
                    restoration_trial = (
                        restoration_trials[restoration_selection.selected_index]
                        if restoration_selection.selected
                        and restoration_selection.selected_index is not None
                        else None
                    )
                restoration_persistent_filter_report = None
                restoration_persistent_filter_theta = None
                restoration_persistent_filter_theta_max = None
                if (
                    trial_acceptance_policy == "ipopt_persistent_h_type"
                    and not enable_dedicated_restoration_filter_phase
                    and restoration_trial is not None
                ):
                    weights = (
                        soft_restoration_component_weights
                        or ipopt_h_type_component_weights
                        or filter_component_weights
                    )
                    if weights is None:
                        raise ValueError(
                            "soft_restoration_component_weights, "
                            "ipopt_h_type_component_weights, or filter_component_weights "
                            "must be provided for persistent restoration fallback."
                        )
                    restoration_persistent_filter_theta = _weighted_filter_theta(
                        restoration_trial["residual_components"],
                        weights,
                    )
                    reference_theta = (
                        1.0
                        if persistent_filter_reference_theta is None
                        else max(1.0, float(persistent_filter_reference_theta))
                    )
                    restoration_persistent_filter_theta_max = max(
                        1.0,
                        persistent_theta_max_factor * reference_theta,
                    )
                    restoration_persistent_filter_report = is_acceptable_to_persistent_filter(
                        p_merit=float(restoration_trial["p_merit"]),
                        theta=restoration_persistent_filter_theta,
                        entries=persistent_filter_entries,
                        gamma_p=persistent_gamma_p,
                        gamma_theta=persistent_gamma_theta,
                        theta_max=restoration_persistent_filter_theta_max,
                    )
                    if not restoration_persistent_filter_report.acceptable:
                        restoration_trial = None
                record["soft_restoration_fallback_selection"] = restoration_selection.as_dict()
                record["soft_restoration_fallback_trial"] = restoration_trial
                record["dedicated_restoration_filter_selection"] = (
                    dedicated_restoration_filter_selection
                )
                record["soft_restoration_persistent_filter_theta"] = (
                    restoration_persistent_filter_theta
                )
                record["soft_restoration_persistent_filter_theta_max"] = (
                    restoration_persistent_filter_theta_max
                )
                record["soft_restoration_persistent_filter_report"] = (
                    None
                    if restoration_persistent_filter_report is None
                    else restoration_persistent_filter_report.as_dict()
                )
                if restoration_trial is not None:
                    selected_direction = restoration_direction
                    selected_trial = restoration_trial
                    selected_selection = None
                    record["status"] = (
                        "dedicated_restoration_filter_selected"
                        if enable_dedicated_restoration_filter_phase
                        else "soft_restoration_selected"
                    )
                    record["selected_direction_kind"] = restoration_direction.direction_kind
                    record["p_armijo_selection"] = None
                    record["selected_trial"] = selected_trial
            inner_records.append(record)
            inner_count += 1
            if selected_trial is None:
                stopped_reason = "no_p_armijo_trial"
                break
            if (
                trial_acceptance_policy == "ipopt_persistent_h_type"
                and record.get("status") != "dedicated_restoration_filter_selected"
            ):
                weights = ipopt_h_type_component_weights or filter_component_weights
                assert weights is not None
                accepted_theta = _weighted_filter_theta(
                    selected_trial["residual_components"],
                    weights,
                )
                persistent_filter_entries = augment_persistent_filter(
                    persistent_filter_entries,
                    p_merit=float(selected_trial["p_merit"]),
                    theta=accepted_theta,
                    iteration=inner_count,
                    gamma_p=persistent_gamma_p,
                    gamma_theta=persistent_gamma_theta,
                )
                record["persistent_filter_theta_after_acceptance"] = accepted_theta
                record["persistent_filter_entry_count_after_acceptance"] = len(
                    persistent_filter_entries
                )
            alpha = float(selected_trial["alpha"])
            assert selected_direction is not None
            q = q + alpha * selected_direction.delta_q
            r = r + alpha * selected_direction.delta_r
            if enforce_trial_condensate_capacity:
                r, capacity_cap_count = _apply_condensate_capacity_cap(
                    r=r,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                )
                record["accepted_condensate_capacity_cap_count"] = capacity_cap_count
            lam = lam + alpha * selected_direction.delta_lambda
            rho = rho + alpha * selected_direction.delta_rho
            qtot = float(qtot + alpha * selected_direction.delta_qtot)
        final_inner = inner_records[-1] if inner_records else {}
        residual_after_outer, _components_after_outer = _residual_norm(
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=target,
            external_condensate_budget=external_budget,
            gas_stationarity_source=gas_source,
            condensate_standard_source=cond_source,
            q=q,
            r=r,
            lam=lam,
            rho=rho,
            qtot=qtot,
            epsilon=eps,
        )
        center_metric_after_outer = _center_metric_from_components(
            residual_after_outer,
            _components_after_outer,
            center_metric_policy=center_metric_policy,
            center_component_weights=center_component_weights,
            center_component_scales=center_component_scales,
        )
        centered = center_metric_after_outer <= center_threshold
        strict_barrier_update_allowed = True
        if strict_barrier_names:
            strict_barrier_update_allowed = _strict_barrier_components_met(
                _components_after_outer,
                strict_barrier_names,
                strict_barrier_threshold,
            )
        epsilon_after_outer = eps
        nu_after_outer = nu
        barrier_update_reason = "not_evaluated"
        barrier_updated = False
        persistent_filter_reset_on_barrier_update = False
        if eps <= eps_final:
            reached_final = True
            converged_final = centered and strict_barrier_update_allowed
            barrier_update_reason = "already_at_final_barrier"
            stopped_reason = "final_barrier_centered" if centered else "final_barrier_not_centered"
            if centered and not strict_barrier_update_allowed:
                stopped_reason = "final_barrier_strict_components_not_met"
        elif not centered:
            barrier_update_reason = "current_barrier_not_centered"
            stopped_reason = (
                "no_p_armijo_trial"
                if final_inner.get("status") == "no_p_armijo_trial"
                else "current_barrier_not_centered"
            )
        elif not strict_barrier_update_allowed:
            barrier_update_reason = "strict_barrier_update_blocked"
            stopped_reason = "strict_barrier_update_blocked"
        else:
            epsilon_after_outer = _next_barrier_epsilon(
                epsilon=eps,
                final_epsilon=eps_final,
                tau=tau_value,
                barrier_schedule_policy=schedule_policy,
                ipopt_mu_linear_decrease_factor=ipopt_linear_factor,
                ipopt_mu_superlinear_decrease_power=ipopt_superlinear_power,
                ipopt_enable_superlinear_decrease=ipopt_superlinear_enabled,
            )
            nu_after_outer = float(np.exp(epsilon_after_outer))
            barrier_updated = epsilon_after_outer < eps
            barrier_update_reason = "barrier_decreased" if barrier_updated else "barrier_unchanged"
            if barrier_updated and persistent_filter_entries:
                persistent_filter_entries = ()
                persistent_filter_reference_theta = None
                persistent_filter_reset_on_barrier_update = True
        outer_records.append(
            {
                "outer_index": outer_index,
                "epsilon": eps,
                "nu": nu,
                "epsilon_after_outer": epsilon_after_outer,
                "nu_after_outer": nu_after_outer,
                "barrier_schedule_policy": schedule_policy,
                "ipopt_mu_linear_decrease_factor": ipopt_linear_factor,
                "ipopt_mu_superlinear_decrease_power": ipopt_superlinear_power,
                "ipopt_enable_superlinear_decrease": ipopt_superlinear_enabled,
                "center_threshold": center_threshold,
                "inner_records": inner_records,
                "residual_l2_after_outer": residual_after_outer,
                "residual_components_after_outer": _components_after_outer,
                "center_metric_after_outer": center_metric_after_outer,
                "center_metric_policy": center_metric_policy,
                "center_component_weights": None
                if center_component_weights is None
                else dict(center_component_weights),
                "center_component_scales": None
                if center_component_scales is None
                else dict(center_component_scales),
                "centered_at_current_barrier": centered,
                "barrier_update_allowed": centered and strict_barrier_update_allowed,
                "barrier_updated": barrier_updated,
                "barrier_update_reason": barrier_update_reason,
                "strict_barrier_update_allowed": strict_barrier_update_allowed,
                "strict_barrier_update_components": strict_barrier_names,
                "strict_barrier_update_threshold": strict_barrier_threshold,
                "strict_barrier_components_after_outer": {
                    name: _components_after_outer.get(name, math.inf)
                    for name in strict_barrier_names
                },
                "persistent_filter_entry_count_after_outer": len(persistent_filter_entries),
                "persistent_filter_reset_on_barrier_update": (
                    persistent_filter_reset_on_barrier_update
                ),
            }
        )
        if eps <= eps_final:
            break
        if not centered:
            break
        if not strict_barrier_update_allowed:
            break
        eps = epsilon_after_outer

    final_residual, _final_components = _residual_norm(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        external_condensate_budget=external_budget,
        gas_stationarity_source=gas_source,
        condensate_standard_source=cond_source,
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        qtot=qtot,
        epsilon=eps,
    )
    final_p = _p_merit(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        external_condensate_budget=external_budget,
        gas_stationarity_source=gas_source,
        condensate_standard_source=cond_source,
        q=q,
        r=r,
        qtot=qtot,
        qtot_reference=qtot,
        epsilon=eps,
        equality_penalty_weight=equality_penalty_weight,
        total_density_penalty_weight=total_density_penalty_weight,
    )
    if eps <= eps_final:
        reached_final = True
        final_center_threshold = float(center_tolerance_multiplier * np.exp(eps_final))
        final_center_metric = _center_metric_from_components(
            final_residual,
            _final_components,
            center_metric_policy=center_metric_policy,
            center_component_weights=center_component_weights,
            center_component_scales=center_component_scales,
        )
        final_strict_components_met = True
        if strict_barrier_names:
            final_strict_components_met = _strict_barrier_components_met(
                _final_components,
                strict_barrier_names,
                strict_barrier_threshold,
            )
        converged_final = (
            final_center_metric <= final_center_threshold and final_strict_components_met
        )
    final_state = build_pdipm_rgie_condensate_state(
        ln_nk=q,
        ln_mk=r,
        element_potential=lam,
        ln_ntot=qtot,
        rho=rho,
        eta=np.exp(rho),
        field_provenance=state.field_provenance,
    )
    return AlgorithmV11ContinuationReport(
        report_schema="exogibbs_algorithm_v11_pdipm_continuation_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        outer_iteration_count=len(outer_records),
        inner_iteration_count=inner_count,
        reached_final_barrier=reached_final,
        converged_at_final_barrier=converged_final,
        stopped_reason=stopped_reason,
        initial_epsilon=float(initial_epsilon),
        final_epsilon_target=eps_final,
        final_epsilon=eps,
        tau=tau_value,
        barrier_schedule_policy=schedule_policy,
        ipopt_mu_linear_decrease_factor=ipopt_linear_factor,
        ipopt_mu_superlinear_decrease_power=ipopt_superlinear_power,
        ipopt_enable_superlinear_decrease=ipopt_superlinear_enabled,
        center_tolerance_multiplier=float(center_tolerance_multiplier),
        final_residual_l2=final_residual,
        final_p_merit=float(final_p["total_merit"]),
        final_state=final_state,
        outer_records=tuple(outer_records),
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "AlgorithmV11ContinuationReport",
    "fraction_to_boundary_alpha",
    "run_algorithm_v11_pdipm_continuation",
)
