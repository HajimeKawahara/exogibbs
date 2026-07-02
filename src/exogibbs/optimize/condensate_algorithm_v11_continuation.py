"""Diagnostic algorithm-v1.1 PD-IPM outer/inner continuation driver.

This module is explicit-import only. It uses the existing diagnostic reduced
step as an inner Newton direction provider, adds P-based Armijo trial
selection, and separates fixed-barrier inner iterations from barrier-parameter
outer continuation. It does not change production behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.optimize.condensate_algorithm_v11_direction import (
    AlgorithmV11Direction,
    blend_algorithm_v11_directions,
    build_active_condensate_budget_correction_direction,
    build_linear_budget_total_density_amount_gas_condensate_direction,
    build_linear_budget_total_density_amount_gas_direction,
    build_linear_budget_total_density_restoration_direction,
    build_log_complementarity_centering_direction,
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
    continuation_mode: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    outer_iteration_count: int
    inner_iteration_count: int
    filter_accept_count: int
    restoration_count: int
    barrier_update_count: int
    tiny_step_count: int
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
    ipopt_allow_fast_monotone_decrease: bool
    center_tolerance_multiplier: float
    final_residual_l2: float
    final_p_merit: float
    final_state: PdipmRgieCondensateState
    outer_records: tuple[Mapping[str, Any], ...]
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_schema": self.report_schema,
            "continuation_mode": self.continuation_mode,
            "diagnostic_only": self.diagnostic_only,
            "default_off": self.default_off,
            "explicit_opt_in": self.explicit_opt_in,
            "production_behavior_change": self.production_behavior_change,
            "production_return_signature_change": self.production_return_signature_change,
            "preset_default_wiring_change": self.preset_default_wiring_change,
            "outer_iteration_count": self.outer_iteration_count,
            "inner_iteration_count": self.inner_iteration_count,
            "filter_accept_count": self.filter_accept_count,
            "restoration_count": self.restoration_count,
            "barrier_update_count": self.barrier_update_count,
            "tiny_step_count": self.tiny_step_count,
            "reached_final_barrier": self.reached_final_barrier,
            "converged_at_final_barrier": self.converged_at_final_barrier,
            "stopped_reason": self.stopped_reason,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon_target": self.final_epsilon_target,
            "final_epsilon": self.final_epsilon,
            "tau": self.tau,
            "barrier_schedule_policy": self.barrier_schedule_policy,
            "ipopt_mu_linear_decrease_factor": self.ipopt_mu_linear_decrease_factor,
            "ipopt_mu_superlinear_decrease_power": self.ipopt_mu_superlinear_decrease_power,
            "ipopt_enable_superlinear_decrease": self.ipopt_enable_superlinear_decrease,
            "ipopt_allow_fast_monotone_decrease": self.ipopt_allow_fast_monotone_decrease,
            "center_tolerance_multiplier": self.center_tolerance_multiplier,
            "final_residual_l2": self.final_residual_l2,
            "final_p_merit": self.final_p_merit,
            "final_state": self.final_state.as_dict(),
            "outer_records": self.outer_records,
            "fastchem4_trace_public_runtime_constructor_inputs_used": (
                self.fastchem4_trace_public_runtime_constructor_inputs_used
            ),
        }


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
    target = np.asarray(element_inventory_target, dtype=np.float64)
    budget = np.asarray(residuals["budget"], dtype=np.float64)
    positive_target = target[target > 0.0]
    target_scale = float(np.max(positive_target)) if positive_target.size else 1.0
    floor = max(float(np.finfo(np.float64).tiny), 1.0e-300 * target_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_budget = budget / np.maximum(np.abs(target), floor)
    relative_budget = np.where(target > 0.0, relative_budget, 0.0)
    finite_relative_budget = relative_budget[np.isfinite(relative_budget)]
    components["relative_budget_max"] = (
        float(np.max(np.abs(finite_relative_budget)))
        if finite_relative_budget.size
        else math.inf
    )
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


def _protected_component_acceptance_report(
    *,
    current_components: Mapping[str, Any],
    trial_components: Mapping[str, Any],
    component_scales: Mapping[str, float],
    protected_components: Sequence[str],
    max_normalized_increase: float | None,
) -> dict[str, Any]:
    if max_normalized_increase is None or not protected_components:
        return {
            "report_schema": "exogibbs_protected_component_acceptance_report_v1",
            "accepted": True,
            "protected_components": tuple(str(name) for name in protected_components),
            "max_normalized_increase": None,
            "blocking_components": (),
            "diagnostic_only": True,
            "production_behavior_change": False,
        }
    limit = float(max_normalized_increase)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("max_normalized_increase must be finite and non-negative.")
    blocking: list[dict[str, Any]] = []
    for name in protected_components:
        key = str(name)
        current = float(current_components.get(key, math.nan))
        trial = float(trial_components.get(key, math.nan))
        scale = float(component_scales.get(key, math.nan))
        if not (math.isfinite(current) and math.isfinite(trial) and math.isfinite(scale)):
            blocking.append(
                {
                    "component": key,
                    "current": current,
                    "trial": trial,
                    "scale": scale,
                    "normalized_increase": math.inf,
                }
            )
            continue
        if scale <= 0.0:
            blocking.append(
                {
                    "component": key,
                    "current": current,
                    "trial": trial,
                    "scale": scale,
                    "normalized_increase": math.inf,
                }
            )
            continue
        normalized_increase = (trial - current) / scale
        if normalized_increase > limit:
            blocking.append(
                {
                    "component": key,
                    "current": current,
                    "trial": trial,
                    "scale": scale,
                    "normalized_increase": float(normalized_increase),
                }
            )
    return {
        "report_schema": "exogibbs_protected_component_acceptance_report_v1",
        "accepted": not blocking,
        "protected_components": tuple(str(name) for name in protected_components),
        "max_normalized_increase": limit,
        "blocking_components": tuple(blocking),
        "diagnostic_only": True,
        "production_behavior_change": False,
    }


def _inner_status_for_accepted_source(source: str | None) -> str:
    if source == "ipopt_h_filter":
        return "ipopt_h_filter_selected"
    if source == "ipopt_persistent_filter_f_type":
        return "ipopt_persistent_filter_f_type_selected"
    if source == "filter_restoration":
        return "filter_restoration_selected"
    if source == "best_residual":
        return "best_residual_selected"
    return "p_armijo_selected"


def _line_search_failure_summary(
    *,
    trial_acceptance_policy: str,
    direction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    p_selected = 0
    filter_selected = 0
    h_selected = 0
    h_accepted = 0
    persistent_blocked = 0
    f_type_blocked = 0
    finite_trials = 0
    for direction in direction_records:
        p_report = direction.get("p_armijo_selection")
        filter_report = direction.get("filter_selection")
        h_report = direction.get("ipopt_h_type_selection")
        persistent_report = direction.get("persistent_filter_report")
        f_type_report = direction.get("persistent_filter_f_type_report")
        if isinstance(p_report, Mapping):
            finite_trials += int(p_report.get("finite_trial_count", 0))
            if bool(p_report.get("selected", False)):
                p_selected += 1
        if isinstance(filter_report, Mapping) and bool(filter_report.get("selected", False)):
            filter_selected += 1
        if isinstance(h_report, Mapping):
            h_accepted += int(h_report.get("accepted_trial_count", 0))
            if bool(h_report.get("selected", False)):
                h_selected += 1
        if isinstance(persistent_report, Mapping) and not bool(
            persistent_report.get("acceptable", True)
        ):
            persistent_blocked += 1
        if isinstance(f_type_report, Mapping) and not bool(
            f_type_report.get("acceptable", True)
        ):
            f_type_blocked += 1

    if trial_acceptance_policy == "ipopt_persistent_h_type":
        if h_selected > 0 and persistent_blocked > 0:
            status = "ipopt_persistent_filter_rejected"
            reason = "An h-type candidate existed, but persistent filter memory rejected it."
        elif p_selected > 0 and h_accepted == 0:
            status = "ipopt_h_filter_rejected"
            reason = (
                "P-Armijo produced a merit-improving candidate, but the Ipopt-style "
                "h-filter did not accept a constraint-violation step."
            )
        elif f_type_blocked > 0:
            status = "ipopt_persistent_filter_f_type_rejected"
            reason = "A P-Armijo f-type candidate was blocked by persistent filter memory."
        elif finite_trials == 0:
            status = "no_finite_trial"
            reason = "No finite line-search trials were available."
        else:
            status = "no_acceptable_ipopt_filter_trial"
            reason = "No h-type, f-type, or restoration trial passed the active filter policy."
    elif p_selected == 0:
        status = "no_p_armijo_trial"
        reason = "No P-Armijo trial satisfied sufficient merit decrease."
    elif filter_selected == 0 and trial_acceptance_policy == "filter_restoration":
        status = "filter_restoration_rejected"
        reason = "A P-Armijo candidate existed, but the restoration filter rejected it."
    else:
        status = "no_accepted_trial"
        reason = "Finite trials existed, but none passed the active acceptance policy."
    return {
        "summary_schema": "exogibbs_line_search_failure_summary_v1",
        "status": status,
        "reason": reason,
        "trial_acceptance_policy": trial_acceptance_policy,
        "p_armijo_selected_direction_count": p_selected,
        "filter_selected_direction_count": filter_selected,
        "ipopt_h_selected_direction_count": h_selected,
        "ipopt_h_accepted_trial_count": h_accepted,
        "persistent_filter_blocked_direction_count": persistent_blocked,
        "persistent_filter_f_type_blocked_direction_count": f_type_blocked,
        "finite_trial_count": finite_trials,
        "diagnostic_only": True,
        "production_behavior_change": True,
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


def primal_dual_fraction_to_boundary_alphas(
    *,
    delta_r: Sequence[float],
    delta_rho: Sequence[float],
    gamma: float = 0.995,
) -> dict[str, float]:
    """Return separate Ipopt-style primal and dual step-length limits."""

    dr = _as_vector(delta_r, "delta_r")
    drho = _as_vector(delta_rho, "delta_rho")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0 or gamma_value >= 1.0:
        raise ValueError("gamma must be finite and in the interval (0, 1).")

    def limited_alpha(direction: np.ndarray) -> float:
        candidates = [1.0]
        negative = direction[direction < 0.0]
        if negative.size:
            candidates.append(float(np.min(-1.0 / negative)))
        return float(max(0.0, min(1.0, gamma_value * min(candidates))))

    alpha_primal = limited_alpha(dr)
    alpha_dual = limited_alpha(drho)
    return {
        "alpha_primal": alpha_primal,
        "alpha_dual": alpha_dual,
        "alpha_combined": min(alpha_primal, alpha_dual),
        "gamma": gamma_value,
    }


def fraction_to_boundary_blocker_report(
    *,
    r: Sequence[float],
    rho: Sequence[float],
    delta_r: Sequence[float],
    delta_rho: Sequence[float],
    gamma: float = 0.995,
    species_names: Sequence[str] | None = None,
    max_entries: int = 8,
) -> dict[str, Any]:
    """Return diagnostics for variables that limit scalar fraction-to-boundary."""

    r_array = _as_vector(r, "r")
    rho_array = _as_vector(rho, "rho")
    dr = _as_vector(delta_r, "delta_r")
    drho = _as_vector(delta_rho, "delta_rho")
    if r_array.shape != dr.shape or rho_array.shape != drho.shape:
        raise ValueError("current values and directions must have matching shapes.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0 or gamma_value >= 1.0:
        raise ValueError("gamma must be finite and in the interval (0, 1).")
    names: tuple[str, ...] | None = None
    if species_names is not None:
        names = tuple(str(name) for name in species_names)
        if len(names) != r_array.shape[0]:
            raise ValueError("species_names must match the condensate vector length.")
    blockers: list[dict[str, Any]] = []
    for group, current, direction in (
        ("r", r_array, dr),
        ("rho", rho_array, drho),
    ):
        for index, value in enumerate(direction):
            if not math.isfinite(float(value)) or float(value) >= 0.0:
                continue
            raw_alpha = float(-1.0 / float(value))
            if not math.isfinite(raw_alpha) or raw_alpha <= 0.0:
                continue
            row: dict[str, Any] = {
                "variable_group": group,
                "local_index": int(index),
                "current_log_value": float(current[index]),
                "direction": float(value),
                "raw_alpha": raw_alpha,
                "safety_alpha": float(min(1.0, gamma_value * raw_alpha)),
            }
            if names is not None:
                row["species_name"] = names[index]
            blockers.append(row)
    blockers.sort(key=lambda row: (float(row["safety_alpha"]), float(row["raw_alpha"])))
    limiting = blockers[0] if blockers else None
    return {
        "report_schema": "exogibbs_fraction_to_boundary_blocker_report_v1",
        "step_policy": "scalar_fraction_to_boundary",
        "safety": gamma_value,
        "limiting_variable_group": None if limiting is None else limiting["variable_group"],
        "limiting_local_index": None if limiting is None else limiting["local_index"],
        "limiting_species_name": None if limiting is None else limiting.get("species_name"),
        "limiting_raw_alpha": None if limiting is None else limiting["raw_alpha"],
        "limiting_safety_alpha": None if limiting is None else limiting["safety_alpha"],
        "blocker_count": len(blockers),
        "top_blockers": tuple(blockers[: int(max_entries)]),
        "diagnostic_only": True,
        "production_behavior_change": False,
    }


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
    alpha_dual: float | None = None,
) -> list[dict[str, Any]]:
    alpha_dual_value = None if alpha_dual is None else float(alpha_dual)
    if alpha_dual_value is not None and (
        not math.isfinite(alpha_dual_value) or alpha_dual_value < 0.0
    ):
        raise ValueError("alpha_dual must be finite and non-negative when provided.")
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
        dual_alpha = float(alpha) if alpha_dual_value is None else min(1.0, alpha_dual_value)
        q_trial = q + alpha * delta_q
        r_trial = r + alpha * delta_r
        capacity_cap_count = 0
        if enforce_condensate_capacity:
            r_trial, capacity_cap_count = _apply_condensate_capacity_cap(
                r=r_trial,
                formula_matrix_cond_active=formula_matrix_cond_active,
                element_inventory_target=element_inventory_target,
            )
        lam_trial = lam + dual_alpha * delta_lambda
        rho_trial = rho + dual_alpha * delta_rho
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
                "alpha_primal": float(alpha),
                "alpha_dual": float(dual_alpha),
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
    condensate_standard_source: np.ndarray,
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
    budget_row_scaling_policy: str,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_condensate_linearized":
        return [
            build_linear_budget_total_density_amount_gas_condensate_direction(
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
                target_direction=algorithm_direction,
                budget_weight=30.0,
                total_density_weight=30.0,
                amount_gas_weight=1.0,
                amount_condensate_weight=1.0,
                target_direction_weight=1.0e-2,
                target_direction_component_mode="condensate_dual_only",
                max_abs_delta_q=max_abs_delta_q,
                max_abs_delta_r=max_abs_delta_r,
                budget_row_scaling_policy=budget_row_scaling_policy,
            )
        ]
    if direction_policy == "joint_budget_amount_gas_condensate_linearized_no_prior":
        return [
            build_linear_budget_total_density_amount_gas_condensate_direction(
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
                target_direction=None,
                budget_weight=30.0,
                total_density_weight=30.0,
                amount_gas_weight=1.0,
                amount_condensate_weight=1.0,
                target_direction_weight=0.0,
                max_abs_delta_q=max_abs_delta_q,
                max_abs_delta_r=max_abs_delta_r,
                budget_row_scaling_policy=budget_row_scaling_policy,
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
            "joint_budget_amount_gas_condensate_prior_balanced, or "
            "joint_budget_amount_gas_condensate_linearized, or "
            "joint_budget_amount_gas_condensate_linearized_no_prior."
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
    ipopt_allow_fast_monotone_decrease: bool = False,
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
    continuation_mode: str = "legacy_policy",
    step_control_policy: str = "component_clip",
    fraction_to_boundary_safety: float = 0.995,
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
    ipopt_tiny_step_alpha_threshold: float = 1.0e-8,
    ipopt_tiny_step_consecutive_limit: int = 1,
    ipopt_tiny_step_switch_to_restoration: bool = True,
    strict_barrier_update_components: Sequence[str] = (),
    strict_barrier_update_threshold: float = 1.0e-6,
    center_metric_policy: str = "raw_l2",
    center_component_weights: Mapping[str, float] | None = None,
    center_component_scales: Mapping[str, float] | None = None,
    enable_native_soft_restoration_fallback: bool = False,
    enable_log_complementarity_centering_fallback: bool = False,
    soft_restoration_component_weights: Mapping[str, float] | None = None,
    soft_restoration_proximity_weight: float = 1.0e-2,
    soft_restoration_max_proximity: float | None = 10.0,
    enable_dedicated_restoration_filter_phase: bool = False,
    dedicated_restoration_component_weights: Mapping[str, float] | None = None,
    dedicated_restoration_max_proximity: float | None = 10.0,
    require_residual_nonworsening: bool = False,
    residual_worsening_tolerance: float = 0.0,
    budget_row_scaling_policy: str = "absolute",
    species_names: Sequence[str] | None = None,
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
    ipopt_fast_monotone_enabled = bool(ipopt_allow_fast_monotone_decrease)
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
    mode = str(continuation_mode)
    if mode not in {"legacy_policy", "pdipm_core", "pdipm_core_single_loop"}:
        raise ValueError(
            "continuation_mode must be legacy_policy, pdipm_core, "
            "or pdipm_core_single_loop."
        )
    pdipm_core_policy_mode = mode in {"pdipm_core", "pdipm_core_single_loop"}
    single_loop_mode = mode == "pdipm_core_single_loop"
    outer_iteration_limit = int(max_outer_iterations)
    inner_iteration_limit = int(max_inner_iterations)
    if single_loop_mode:
        outer_iteration_limit = outer_iteration_limit * inner_iteration_limit
        inner_iteration_limit = 1
    if pdipm_core_policy_mode:
        step_control_policy = "scalar_fraction_to_boundary"
        direction_policy = "algorithm_v11_reduced"
        trial_acceptance_policy = "ipopt_persistent_h_type"
        filter_component_scale_policy = "current"
        enable_native_soft_restoration_fallback = True
        enable_log_complementarity_centering_fallback = True
        enable_dedicated_restoration_filter_phase = True
        ipopt_tiny_step_switch_to_restoration = True
        require_residual_nonworsening = False
        if filter_component_weights is None:
            filter_component_weights = {
                "budget": 1.0,
                "total_density": 1.0,
                "amount_weighted_gas": 1.0,
                "amount_weighted_condensate": 1.0,
                "complementarity": 1.0,
            }
        if ipopt_h_type_component_weights is None:
            ipopt_h_type_component_weights = filter_component_weights
        if soft_restoration_component_weights is None:
            soft_restoration_component_weights = {
                "budget": 1.0,
                "total_density": 1.0,
                "amount_weighted_gas": 1.0,
                "amount_weighted_condensate": 1.0,
            }
        if dedicated_restoration_component_weights is None:
            dedicated_restoration_component_weights = soft_restoration_component_weights
        if not ipopt_h_type_protected_components:
            ipopt_h_type_protected_components = ("budget", "total_density")
    if step_control_policy not in {"component_clip", "scalar_fraction_to_boundary"}:
        raise ValueError(
            "step_control_policy must be component_clip or scalar_fraction_to_boundary."
        )
    fraction_safety = float(fraction_to_boundary_safety)
    if (
        not math.isfinite(fraction_safety)
        or fraction_safety <= 0.0
        or fraction_safety > 1.0
    ):
        raise ValueError("fraction_to_boundary_safety must be in the interval (0, 1].")
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
    tiny_step_alpha_threshold = float(ipopt_tiny_step_alpha_threshold)
    if not math.isfinite(tiny_step_alpha_threshold) or tiny_step_alpha_threshold < 0.0:
        raise ValueError("ipopt_tiny_step_alpha_threshold must be finite and non-negative.")
    tiny_step_consecutive_limit = int(ipopt_tiny_step_consecutive_limit)
    if tiny_step_consecutive_limit < 1:
        raise ValueError("ipopt_tiny_step_consecutive_limit must be positive.")
    tiny_step_switch_to_restoration = bool(ipopt_tiny_step_switch_to_restoration)
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
    if budget_row_scaling_policy not in {"absolute", "relative_target"}:
        raise ValueError("budget_row_scaling_policy must be absolute or relative_target.")
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
    active_species_names: tuple[str, ...] | None = None
    if species_names is not None:
        active_species_names = tuple(str(name) for name in species_names)
        if len(active_species_names) != r.shape[0]:
            raise ValueError("species_names must match active condensate length.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    outer_records: list[Mapping[str, Any]] = []
    inner_count = 0
    tiny_step_count = 0
    consecutive_tiny_step_count = 0
    stopped_reason = "max_outer_iterations_reached"
    reached_final = False
    converged_final = False
    persistent_filter_entries: tuple[IpoptFilterEntry, ...] = ()
    persistent_filter_reference_theta: float | None = None

    for outer_index in range(outer_iteration_limit):
        nu = float(np.exp(eps))
        center_threshold = float(center_tolerance_multiplier * nu)
        inner_records: list[Mapping[str, Any]] = []
        for inner_index in range(inner_iteration_limit):
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
                step_control_policy=step_control_policy,
                fraction_to_boundary_safety=fraction_safety,
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
                condensate_standard_source=cond_source,
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
                budget_row_scaling_policy=budget_row_scaling_policy,
            )
            direction_records: list[dict[str, Any]] = []
            selected_direction: AlgorithmV11Direction | None = None
            selected_trial: Mapping[str, Any] | None = None
            selected_score: tuple[float, float] | None = None
            selected_selection = None
            selected_acceptance_source: str | None = None
            for direction in direction_candidates:
                alpha_limits = primal_dual_fraction_to_boundary_alphas(
                    delta_r=direction.delta_r,
                    delta_rho=direction.delta_rho,
                    gamma=fraction_to_boundary_gamma,
                )
                alpha_max = (
                    alpha_limits["alpha_primal"]
                    if pdipm_core_policy_mode
                    else alpha_limits["alpha_combined"]
                )
                alpha_dual = alpha_limits["alpha_dual"] if pdipm_core_policy_mode else None
                alpha_blocker_report = fraction_to_boundary_blocker_report(
                    r=r,
                    rho=rho,
                    delta_r=direction.delta_r,
                    delta_rho=direction.delta_rho,
                    gamma=fraction_to_boundary_gamma,
                    species_names=active_species_names,
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
                    alpha_dual=alpha_dual,
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
                persistent_filter_f_type_report = None
                persistent_filter_f_type_theta = None
                persistent_filter_f_type_theta_max = None
                persistent_filter_f_type_protected_report = None
                persistent_filter_f_type_trial = None
                if (
                    trial_acceptance_policy == "ipopt_persistent_h_type"
                    and ipopt_h_type_trial is None
                    and candidate_trial is not None
                ):
                    weights = ipopt_h_type_component_weights or filter_component_weights
                    assert weights is not None
                    persistent_filter_f_type_theta = _weighted_filter_theta(
                        candidate_trial["residual_components"],
                        weights,
                    )
                    reference_theta = (
                        1.0
                        if persistent_filter_reference_theta is None
                        else max(1.0, float(persistent_filter_reference_theta))
                    )
                    persistent_filter_f_type_theta_max = max(
                        1.0,
                        persistent_theta_max_factor * reference_theta,
                    )
                    persistent_filter_f_type_report = is_acceptable_to_persistent_filter(
                        p_merit=float(candidate_trial["p_merit"]),
                        theta=persistent_filter_f_type_theta,
                        entries=persistent_filter_entries,
                        gamma_p=persistent_gamma_p,
                        gamma_theta=persistent_gamma_theta,
                        theta_max=persistent_filter_f_type_theta_max,
                    )
                    component_scales = (
                        ipopt_h_type_selection.component_scales
                        if ipopt_h_type_selection is not None
                        else {
                            str(name): max(abs(float(current_components.get(name, 0.0))), 1.0)
                            for name in weights
                        }
                    )
                    persistent_filter_f_type_protected_report = (
                        _protected_component_acceptance_report(
                            current_components=current_components,
                            trial_components=candidate_trial["residual_components"],
                            component_scales=component_scales,
                            protected_components=ipopt_h_type_protected_components,
                            max_normalized_increase=protected_max_increase,
                        )
                    )
                    if (
                        persistent_filter_f_type_report.acceptable
                        and persistent_filter_f_type_protected_report["accepted"]
                    ):
                        persistent_filter_f_type_trial = candidate_trial
                direction_record = {
                    "direction_kind": direction.direction_kind,
                    "fraction_to_boundary_alpha_primal": alpha_limits["alpha_primal"],
                    "fraction_to_boundary_alpha_dual": alpha_limits["alpha_dual"],
                    "fraction_to_boundary_alpha_combined": alpha_limits["alpha_combined"],
                    "fraction_to_boundary_alpha": alpha_max,
                    "fraction_to_boundary_blocker_report": alpha_blocker_report,
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
                    "persistent_filter_f_type_theta": persistent_filter_f_type_theta,
                    "persistent_filter_f_type_theta_max": persistent_filter_f_type_theta_max,
                    "persistent_filter_f_type_report": None
                    if persistent_filter_f_type_report is None
                    else persistent_filter_f_type_report.as_dict(),
                    "persistent_filter_f_type_protected_report": (
                        persistent_filter_f_type_protected_report
                    ),
                    "persistent_filter_f_type_trial": persistent_filter_f_type_trial,
                }
                direction_records.append(direction_record)
                acceptance_source: str | None = None
                if trial_acceptance_policy == "ipopt_fastchem4_h_type":
                    trial_for_score = ipopt_h_type_trial
                    acceptance_source = "ipopt_h_filter" if trial_for_score is not None else None
                    if trial_for_score is None and filter_trial is not None:
                        trial_for_score = filter_trial
                        acceptance_source = "filter_restoration"
                    if trial_for_score is None and candidate_trial is not None:
                        trial_for_score = candidate_trial
                        acceptance_source = "p_armijo"
                elif trial_acceptance_policy == "filter_restoration":
                    trial_for_score = filter_trial
                    acceptance_source = (
                        "filter_restoration" if trial_for_score is not None else None
                    )
                    if trial_for_score is None and candidate_trial is not None:
                        trial_for_score = candidate_trial
                        acceptance_source = "p_armijo"
                elif trial_acceptance_policy == "ipopt_persistent_h_type":
                    trial_for_score = ipopt_h_type_trial
                    acceptance_source = "ipopt_h_filter" if trial_for_score is not None else None
                    if trial_for_score is None and persistent_filter_f_type_trial is not None:
                        trial_for_score = persistent_filter_f_type_trial
                        acceptance_source = "ipopt_persistent_filter_f_type"
                else:
                    trial_for_score = (
                        candidate_trial if candidate_trial is not None else best_residual_trial
                    )
                    acceptance_source = (
                        "p_armijo" if candidate_trial is not None else "best_residual"
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
                    selected_acceptance_source = acceptance_source
            record = {
                "inner_index": inner_index,
                "status": (
                    _inner_status_for_accepted_source(selected_acceptance_source)
                    if selected_trial is not None
                    else _line_search_failure_summary(
                        trial_acceptance_policy=trial_acceptance_policy,
                        direction_records=direction_records,
                    )["status"]
                ),
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
                "continuation_mode": mode,
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
                "ipopt_tiny_step_alpha_threshold": tiny_step_alpha_threshold,
                "ipopt_tiny_step_consecutive_limit": tiny_step_consecutive_limit,
                "ipopt_tiny_step_switch_to_restoration": tiny_step_switch_to_restoration,
                "persistent_filter_entry_count_before": len(persistent_filter_entries),
                "strict_barrier_update_components": strict_barrier_names,
                "strict_barrier_update_threshold": strict_barrier_threshold,
                "enable_native_soft_restoration_fallback": bool(
                    enable_native_soft_restoration_fallback
                ),
                "enable_log_complementarity_centering_fallback": bool(
                    enable_log_complementarity_centering_fallback
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
                "step_control_policy": step_dict["step_control_policy"],
                "step_fraction_to_boundary_alpha": step_dict[
                    "fraction_to_boundary_alpha"
                ],
                "step_fraction_to_boundary_safety": step_dict[
                    "fraction_to_boundary_safety"
                ],
                "step_fraction_to_boundary_blocker_report": (
                    direction_records[0].get("fraction_to_boundary_blocker_report")
                    if direction_records
                    else None
                ),
                "direction_records": direction_records,
                "line_search_failure_summary": None
                if selected_trial is not None
                else _line_search_failure_summary(
                    trial_acceptance_policy=trial_acceptance_policy,
                    direction_records=direction_records,
                ),
                "selected_direction_kind": None
                if selected_direction is None
                else selected_direction.direction_kind,
                "selected_acceptance_source": selected_acceptance_source,
                "p_armijo_selection": None
                if selected_selection is None
                else selected_selection.as_dict(),
                "selected_trial": selected_trial,
                "log_complementarity_centering_selection": None,
                "log_complementarity_centering_trial": None,
                "soft_restoration_fallback_selection": None,
                "soft_restoration_fallback_trial": None,
            }
            tiny_step_detected = False
            tiny_step_forces_restoration = False
            tiny_step_alpha = None
            if selected_trial is not None:
                selected_alpha = float(selected_trial["alpha"])
                selected_alpha_primal = float(
                    selected_trial.get("alpha_primal", selected_alpha)
                )
                tiny_step_alpha = selected_alpha_primal
                tiny_step_detected = (
                    pdipm_core_policy_mode
                    and tiny_step_alpha_threshold > 0.0
                    and selected_alpha_primal <= tiny_step_alpha_threshold
                )
                if tiny_step_detected:
                    tiny_step_count += 1
                    consecutive_tiny_step_count += 1
                    tiny_step_forces_restoration = (
                        tiny_step_switch_to_restoration
                        and consecutive_tiny_step_count >= tiny_step_consecutive_limit
                    )
                    if tiny_step_forces_restoration:
                        selected_direction = None
                        selected_trial = None
                        selected_selection = None
                        selected_acceptance_source = None
                        record["status"] = "tiny_step_requires_restoration"
                        record["selected_direction_kind"] = None
                        record["selected_acceptance_source"] = None
                        record["p_armijo_selection"] = None
                        record["selected_trial"] = None
                else:
                    consecutive_tiny_step_count = 0
            record["tiny_step_detected"] = tiny_step_detected
            record["tiny_step_alpha"] = tiny_step_alpha
            record["tiny_step_forces_restoration"] = tiny_step_forces_restoration
            record["consecutive_tiny_step_count"] = consecutive_tiny_step_count
            if (
                selected_trial is None
                and enable_log_complementarity_centering_fallback
                and trial_acceptance_policy == "ipopt_persistent_h_type"
            ):
                centering_direction = build_log_complementarity_centering_direction(
                    q_size=q.shape[0],
                    r=r,
                    lam_size=lam.shape[0],
                    rho=rho,
                    epsilon=eps,
                    max_abs_delta_rho=max_abs_delta_rho,
                )
                alpha_limits = primal_dual_fraction_to_boundary_alphas(
                    delta_r=centering_direction.delta_r,
                    delta_rho=centering_direction.delta_rho,
                    gamma=fraction_to_boundary_gamma,
                )
                centering_trials = _trial_rows(
                    alpha_grid=alpha_grid,
                    alpha_max=alpha_limits["alpha_primal"]
                    if pdipm_core_policy_mode
                    else alpha_limits["alpha_combined"],
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
                    delta_q=centering_direction.delta_q,
                    delta_r=centering_direction.delta_r,
                    delta_lambda=centering_direction.delta_lambda,
                    delta_rho=centering_direction.delta_rho,
                    delta_qtot=centering_direction.delta_qtot,
                    equality_penalty_weight=equality_penalty_weight,
                    total_density_penalty_weight=total_density_penalty_weight,
                    enforce_condensate_capacity=enforce_trial_condensate_capacity,
                    alpha_dual=alpha_limits["alpha_dual"] if pdipm_core_policy_mode else None,
                )
                weights = ipopt_h_type_component_weights or filter_component_weights
                if weights is None:
                    raise ValueError(
                        "ipopt_h_type_component_weights or filter_component_weights "
                        "must be provided for complementarity centering fallback."
                    )
                centering_selection = select_ipopt_h_type_filter_trial(
                    centering_trials,
                    current_components=current_components,
                    current_p_merit=float(current_p["total_merit"]),
                    component_weights=weights,
                    theta_reduction_fraction=h_type_theta_fraction,
                    protected_components=ipopt_h_type_protected_components,
                    protected_component_max_normalized_increase=protected_max_increase,
                    choose_largest_alpha=True,
                )
                centering_trial = (
                    centering_trials[centering_selection.selected_index]
                    if centering_selection.selected
                    and centering_selection.selected_index is not None
                    else None
                )
                centering_persistent_report = None
                centering_persistent_theta = None
                centering_persistent_theta_max = None
                if centering_trial is not None:
                    centering_persistent_theta = _weighted_filter_theta(
                        centering_trial["residual_components"],
                        weights,
                    )
                    reference_theta = (
                        1.0
                        if persistent_filter_reference_theta is None
                        else max(1.0, float(persistent_filter_reference_theta))
                    )
                    centering_persistent_theta_max = max(
                        1.0,
                        persistent_theta_max_factor * reference_theta,
                    )
                    centering_persistent_report = is_acceptable_to_persistent_filter(
                        p_merit=float(centering_trial["p_merit"]),
                        theta=centering_persistent_theta,
                        entries=persistent_filter_entries,
                        gamma_p=persistent_gamma_p,
                        gamma_theta=persistent_gamma_theta,
                        theta_max=centering_persistent_theta_max,
                    )
                    if not centering_persistent_report.acceptable:
                        centering_trial = None
                record["log_complementarity_centering_selection"] = (
                    centering_selection.as_dict()
                )
                record["log_complementarity_centering_trial"] = centering_trial
                record["log_complementarity_centering_alpha_limits"] = alpha_limits
                record["log_complementarity_centering_persistent_filter_theta"] = (
                    centering_persistent_theta
                )
                record["log_complementarity_centering_persistent_filter_theta_max"] = (
                    centering_persistent_theta_max
                )
                record["log_complementarity_centering_persistent_filter_report"] = (
                    None
                    if centering_persistent_report is None
                    else centering_persistent_report.as_dict()
                )
                if centering_trial is not None:
                    consecutive_tiny_step_count = 0
                    selected_direction = centering_direction
                    selected_trial = centering_trial
                    selected_selection = None
                    selected_acceptance_source = "log_complementarity_centering"
                    record["status"] = "log_complementarity_centering_selected"
                    record["selected_direction_kind"] = centering_direction.direction_kind
                    record["selected_acceptance_source"] = selected_acceptance_source
                    record["p_armijo_selection"] = None
                    record["selected_trial"] = selected_trial
                    record["consecutive_tiny_step_count"] = consecutive_tiny_step_count
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
                    consecutive_tiny_step_count = 0
                    selected_direction = restoration_direction
                    selected_trial = restoration_trial
                    selected_selection = None
                    selected_acceptance_source = (
                        "dedicated_restoration_filter"
                        if enable_dedicated_restoration_filter_phase
                        else "soft_restoration"
                    )
                    record["status"] = (
                        "dedicated_restoration_filter_selected"
                        if enable_dedicated_restoration_filter_phase
                        else "soft_restoration_selected"
                    )
                    record["selected_direction_kind"] = restoration_direction.direction_kind
                    record["selected_acceptance_source"] = selected_acceptance_source
                    record["p_armijo_selection"] = None
                    record["selected_trial"] = selected_trial
                    record["consecutive_tiny_step_count"] = consecutive_tiny_step_count
            inner_records.append(record)
            inner_count += 1
            if selected_trial is None:
                if record.get("status") == "tiny_step_requires_restoration":
                    stopped_reason = "tiny_step_no_restoration_trial"
                else:
                    stopped_reason = str(record.get("status", "no_accepted_trial"))
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
            alpha_primal = float(selected_trial.get("alpha_primal", alpha))
            alpha_dual = float(selected_trial.get("alpha_dual", alpha))
            assert selected_direction is not None
            q = q + alpha_primal * selected_direction.delta_q
            r = r + alpha_primal * selected_direction.delta_r
            if enforce_trial_condensate_capacity:
                r, capacity_cap_count = _apply_condensate_capacity_cap(
                    r=r,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                )
                record["accepted_condensate_capacity_cap_count"] = capacity_cap_count
            lam = lam + alpha_dual * selected_direction.delta_lambda
            rho = rho + alpha_dual * selected_direction.delta_rho
            qtot = float(qtot + alpha_primal * selected_direction.delta_qtot)
            record["accepted_alpha_primal"] = alpha_primal
            record["accepted_alpha_dual"] = alpha_dual
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
        center_metric_ratio_after_outer = (
            center_metric_after_outer / center_threshold
            if center_threshold > 0.0
            else math.inf
        )
        center_metric_excess_after_outer = max(
            0.0,
            center_metric_after_outer - center_threshold,
        )
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
        barrier_update_step_count = 0
        fast_monotone_decrease_records: list[Mapping[str, Any]] = []
        persistent_filter_reset_on_barrier_update = False
        if eps <= eps_final:
            reached_final = True
            converged_final = centered and strict_barrier_update_allowed
            barrier_update_reason = "already_at_final_barrier"
            if centered:
                stopped_reason = "final_barrier_centered"
            elif single_loop_mode and final_inner.get("selected_trial") is not None:
                barrier_update_reason = "final_barrier_not_centered_single_loop_continue"
            else:
                stopped_reason = "final_barrier_not_centered"
            if centered and not strict_barrier_update_allowed:
                stopped_reason = "final_barrier_strict_components_not_met"
        elif not centered:
            barrier_update_reason = "current_barrier_not_centered"
            inner_status = str(final_inner.get("status", ""))
            if single_loop_mode and final_inner.get("selected_trial") is not None:
                barrier_update_reason = "current_barrier_not_centered_single_loop_continue"
            else:
                stopped_reason = (
                    inner_status
                    if inner_status
                    in {
                        "no_p_armijo_trial",
                        "no_finite_trial",
                        "no_accepted_trial",
                        "no_acceptable_ipopt_filter_trial",
                        "ipopt_h_filter_rejected",
                        "ipopt_persistent_filter_rejected",
                        "ipopt_persistent_filter_f_type_rejected",
                        "filter_restoration_rejected",
                        "tiny_step_requires_restoration",
                    }
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
            barrier_update_step_count = 1 if barrier_updated else 0
            barrier_update_reason = "barrier_decreased" if barrier_updated else "barrier_unchanged"
            if ipopt_fast_monotone_enabled and barrier_updated:
                fast_eps = epsilon_after_outer
                while fast_eps > eps_final:
                    fast_residual, fast_components = _residual_norm(
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
                        epsilon=fast_eps,
                    )
                    fast_center_metric = _center_metric_from_components(
                        fast_residual,
                        fast_components,
                        center_metric_policy=center_metric_policy,
                        center_component_weights=center_component_weights,
                        center_component_scales=center_component_scales,
                    )
                    fast_center_threshold = float(
                        center_tolerance_multiplier * np.exp(fast_eps)
                    )
                    fast_centered = fast_center_metric <= fast_center_threshold
                    fast_strict_allowed = True
                    if strict_barrier_names:
                        fast_strict_allowed = _strict_barrier_components_met(
                            fast_components,
                            strict_barrier_names,
                            strict_barrier_threshold,
                        )
                    if not fast_centered or not fast_strict_allowed:
                        fast_monotone_decrease_records.append(
                            {
                                "status": "candidate_barrier_not_ready",
                                "epsilon": fast_eps,
                                "nu": float(np.exp(fast_eps)),
                                "residual_l2": fast_residual,
                                "residual_components": fast_components,
                                "center_metric": fast_center_metric,
                                "center_threshold": fast_center_threshold,
                                "centered": fast_centered,
                                "strict_barrier_update_allowed": fast_strict_allowed,
                            }
                        )
                        break
                    next_fast_eps = _next_barrier_epsilon(
                        epsilon=fast_eps,
                        final_epsilon=eps_final,
                        tau=tau_value,
                        barrier_schedule_policy=schedule_policy,
                        ipopt_mu_linear_decrease_factor=ipopt_linear_factor,
                        ipopt_mu_superlinear_decrease_power=ipopt_superlinear_power,
                        ipopt_enable_superlinear_decrease=ipopt_superlinear_enabled,
                    )
                    if next_fast_eps >= fast_eps:
                        fast_monotone_decrease_records.append(
                            {
                                "status": "candidate_barrier_unchanged",
                                "epsilon": fast_eps,
                                "nu": float(np.exp(fast_eps)),
                                "residual_l2": fast_residual,
                                "residual_components": fast_components,
                                "center_metric": fast_center_metric,
                                "center_threshold": fast_center_threshold,
                                "centered": fast_centered,
                                "strict_barrier_update_allowed": fast_strict_allowed,
                            }
                        )
                        break
                    fast_monotone_decrease_records.append(
                        {
                            "status": "fast_barrier_decreased",
                            "epsilon": fast_eps,
                            "nu": float(np.exp(fast_eps)),
                            "epsilon_after": next_fast_eps,
                            "nu_after": float(np.exp(next_fast_eps)),
                            "residual_l2": fast_residual,
                            "residual_components": fast_components,
                            "center_metric": fast_center_metric,
                            "center_threshold": fast_center_threshold,
                            "centered": fast_centered,
                            "strict_barrier_update_allowed": fast_strict_allowed,
                        }
                    )
                    fast_eps = next_fast_eps
                    epsilon_after_outer = fast_eps
                    nu_after_outer = float(np.exp(epsilon_after_outer))
                    barrier_update_step_count += 1
                if barrier_update_step_count > 1:
                    barrier_update_reason = "fast_monotone_barrier_decreased"
            if barrier_updated and persistent_filter_entries:
                persistent_filter_entries = ()
                persistent_filter_reference_theta = None
                persistent_filter_reset_on_barrier_update = True
        outer_records.append(
            {
                "outer_index": outer_index,
                "continuation_mode": mode,
                "epsilon": eps,
                "nu": nu,
                "epsilon_after_outer": epsilon_after_outer,
                "nu_after_outer": nu_after_outer,
                "barrier_schedule_policy": schedule_policy,
                "ipopt_mu_linear_decrease_factor": ipopt_linear_factor,
                "ipopt_mu_superlinear_decrease_power": ipopt_superlinear_power,
                "ipopt_enable_superlinear_decrease": ipopt_superlinear_enabled,
                "ipopt_allow_fast_monotone_decrease": ipopt_fast_monotone_enabled,
                "center_threshold": center_threshold,
                "inner_records": inner_records,
                "residual_l2_after_outer": residual_after_outer,
                "residual_components_after_outer": _components_after_outer,
                "center_metric_after_outer": center_metric_after_outer,
                "center_metric_ratio_after_outer": center_metric_ratio_after_outer,
                "center_metric_excess_after_outer": center_metric_excess_after_outer,
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
                "barrier_update_step_count": barrier_update_step_count,
                "barrier_update_reason": barrier_update_reason,
                "fast_monotone_decrease_records": tuple(fast_monotone_decrease_records),
                "fast_monotone_decrease_count": sum(
                    1
                    for record in fast_monotone_decrease_records
                    if record.get("status") == "fast_barrier_decreased"
                ),
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
        if eps <= eps_final and not (
            single_loop_mode
            and not centered
            and final_inner.get("selected_trial") is not None
        ):
            break
        if not centered and not (
            single_loop_mode and final_inner.get("selected_trial") is not None
        ):
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
    filter_accept_count = 0
    restoration_count = 0
    for outer_record in outer_records:
        for inner_record in outer_record.get("inner_records", ()):
            status = str(inner_record.get("status", ""))
            if status in {
                "p_armijo_selected",
                "dedicated_restoration_filter_selected",
                "soft_restoration_selected",
            }:
                selected_trial = inner_record.get("selected_trial")
                if selected_trial is not None:
                    filter_accept_count += 1
            if status in {
                "dedicated_restoration_filter_selected",
                "soft_restoration_selected",
            }:
                restoration_count += 1
    barrier_update_count = sum(
        int(outer_record.get("barrier_update_step_count", 0))
        for outer_record in outer_records
    )
    return AlgorithmV11ContinuationReport(
        report_schema="exogibbs_algorithm_v11_pdipm_continuation_report_v1",
        continuation_mode=mode,
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        outer_iteration_count=len(outer_records),
        inner_iteration_count=inner_count,
        filter_accept_count=filter_accept_count,
        restoration_count=restoration_count,
        barrier_update_count=barrier_update_count,
        tiny_step_count=tiny_step_count,
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
        ipopt_allow_fast_monotone_decrease=ipopt_fast_monotone_enabled,
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
    "primal_dual_fraction_to_boundary_alphas",
    "run_algorithm_v11_pdipm_continuation",
)
