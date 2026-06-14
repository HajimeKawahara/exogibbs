"""Explicit diagnostic callsite wrapper for algorithm-v1.1 PD-IPM R-GIE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.condensates.thermo_valid_support import (
    ThermoValidSupportFilterReport,
    filter_thermo_valid_condensate_support,
)
from exogibbs.optimize.condensate_algorithm_v11_continuation import (
    AlgorithmV11ContinuationReport,
    run_algorithm_v11_pdipm_continuation,
)
from exogibbs.optimize.pdipm_rgie_cond import (
    PdipmRgieCondensateState,
    PdipmRgieReducedStepReport,
    build_pdipm_rgie_condensate_state,
    solve_pdipm_rgie_algorithm_v11_reduced_step,
)


def algorithm_v11_experimental_high_start_callsite_policy() -> dict[str, Any]:
    """Return the explicit experimental callsite continuation policy."""

    return {
        "initial_epsilon": math.log(1.0e-6),
        "final_epsilon": math.log(1.0e-12),
        "barrier_schedule_policy": "ipopt_like_monotone",
        "ipopt_mu_linear_decrease_factor": 0.2,
        "ipopt_mu_superlinear_decrease_power": 1.5,
        "ipopt_enable_superlinear_decrease": False,
        "max_outer_iterations": 14,
        "max_inner_iterations": 120,
        "center_tolerance_multiplier": 1.0e8,
        "center_metric_policy": "amount_weighted_kkt_max",
        "alpha_grid": (1.0, 0.5, 0.25, 0.125, 0.0625, 1.0e-2, 1.0e-3, 1.0e-4),
        "equality_penalty_weight": 1000.0,
        "total_density_penalty_weight": 1000.0,
        "direction_policy": "algorithm_v11_reduced",
        "require_residual_nonworsening": True,
    }


@dataclass(frozen=True)
class AlgorithmV11ThermoValidCallsiteReport:
    """Report for a thermo-valid algorithm-v1.1 diagnostic callsite."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    original_support_count: int
    filtered_support_count: int
    removed_support_count: int
    external_condensate_support_indices: tuple[int, ...]
    external_condensate_amounts: tuple[float, ...]
    external_condensate_budget: tuple[float, ...]
    filter_report: ThermoValidSupportFilterReport
    reduced_step_report: PdipmRgieReducedStepReport
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_report"] = self.filter_report.as_dict()
        payload["reduced_step_report"] = self.reduced_step_report.as_dict()
        return payload


@dataclass(frozen=True)
class AlgorithmV11ThermoValidContinuationCallsiteReport:
    """Report for a thermo-valid algorithm-v1.1 continuation callsite."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    original_support_count: int
    filtered_support_count: int
    removed_support_count: int
    external_condensate_support_indices: tuple[int, ...]
    external_condensate_amounts: tuple[float, ...]
    external_condensate_budget: tuple[float, ...]
    filter_report: ThermoValidSupportFilterReport
    continuation_report: AlgorithmV11ContinuationReport
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_report"] = self.filter_report.as_dict()
        payload["continuation_report"] = self.continuation_report.as_dict()
        return payload


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def run_algorithm_v11_thermo_valid_reduced_callsite(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    support_indices: Sequence[int],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    epsilon: float,
    external_condensate_budget: Sequence[float] | None = None,
    species_names: Sequence[str] | None = None,
    sentinel_abs_threshold: float = 1.0e10,
    alpha_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    qhat_regularization: float = 0.0,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_rho: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    require_budget_nonworsening: bool = False,
    field_provenance: Mapping[str, str] | None = None,
) -> AlgorithmV11ThermoValidCallsiteReport:
    """Filter thermo-invalid support and run one algorithm-v1.1 reduced step."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for algorithm-v1.1 callsites.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    if state.rho is None:
        raise ValueError("state.rho is required for algorithm-v1.1 callsites.")
    ln_mk = _as_vector(state.ln_mk, "state.ln_mk")
    rho = _as_vector(state.rho, "state.rho")
    eta = np.exp(rho)
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    filtered = filter_thermo_valid_condensate_support(
        explicit_opt_in=True,
        support_indices=support_indices,
        condensate_standard_source=condensate_standard_source,
        formula_matrix_cond_active=formula_matrix_cond_active,
        ln_mk=ln_mk,
        rho=rho,
        eta=eta,
        species_names=species_names,
        sentinel_abs_threshold=sentinel_abs_threshold,
        field_provenance=field_provenance or state.field_provenance,
    )
    active_matrix = np.asarray(formula_matrix_cond_active, dtype=np.float64)
    if active_matrix.ndim != 2 or active_matrix.shape[0] != target.shape[0]:
        raise ValueError("formula_matrix_cond_active row count must match element_inventory_target.")
    if active_matrix.shape[1] != ln_mk.shape[0]:
        raise ValueError("formula_matrix_cond_active column count must match state.ln_mk.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    removed_support_indices: tuple[int, ...] = ()
    removed_amounts = np.zeros((0,), dtype=np.float64)
    removed_budget = np.zeros_like(target, dtype=np.float64)
    if filtered.report.removed_local_indices:
        removed = np.asarray(filtered.report.removed_local_indices, dtype=np.int64)
        removed_support_indices = tuple(int(index) for index in filtered.report.removed_support_indices)
        removed_amounts = np.exp(ln_mk[removed])
        removed_budget = active_matrix[:, removed] @ removed_amounts
        external_budget = external_budget + removed_budget
    filtered_state = build_pdipm_rgie_condensate_state(
        ln_nk=state.ln_nk,
        ln_mk=filtered.ln_mk or (),
        element_potential=state.element_potential,
        ln_ntot=state.ln_ntot,
        rho=filtered.rho,
        eta=filtered.eta,
        field_provenance=state.field_provenance,
    )
    reduced_step = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=filtered_state,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=filtered.formula_matrix_cond_active or (),
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_budget,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=filtered.condensate_standard_source,
        epsilon=epsilon,
        alpha_candidates=alpha_candidates,
        qhat_regularization=qhat_regularization,
        max_abs_delta_q=max_abs_delta_q,
        max_abs_delta_r=max_abs_delta_r,
        max_abs_delta_rho=max_abs_delta_rho,
        max_abs_delta_lambda=max_abs_delta_lambda,
        require_budget_nonworsening=require_budget_nonworsening,
    )
    return AlgorithmV11ThermoValidCallsiteReport(
        report_schema="exogibbs_algorithm_v11_thermo_valid_callsite_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        original_support_count=filtered.report.original_support_count,
        filtered_support_count=filtered.report.filtered_support_count,
        removed_support_count=filtered.report.removed_support_count,
        external_condensate_support_indices=removed_support_indices,
        external_condensate_amounts=tuple(float(value) for value in removed_amounts),
        external_condensate_budget=tuple(float(value) for value in removed_budget),
        filter_report=filtered.report,
        reduced_step_report=reduced_step,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


def run_algorithm_v11_thermo_valid_continuation_callsite(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    support_indices: Sequence[int],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    initial_epsilon: float,
    final_epsilon: float,
    external_condensate_budget: Sequence[float] | None = None,
    species_names: Sequence[str] | None = None,
    sentinel_abs_threshold: float = 1.0e10,
    barrier_schedule_policy: str = "fixed_tau",
    ipopt_mu_linear_decrease_factor: float = 0.2,
    ipopt_mu_superlinear_decrease_power: float = 1.5,
    ipopt_enable_superlinear_decrease: bool = True,
    max_outer_iterations: int = 4,
    max_inner_iterations: int = 6,
    center_tolerance_multiplier: float = 10.0,
    center_metric_policy: str = "raw_l2",
    alpha_grid: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    equality_penalty_weight: float = 100.0,
    total_density_penalty_weight: float = 100.0,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_rho: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    direction_policy: str = "algorithm_v11_reduced",
    algorithm_fraction_grid: Sequence[float] = (
        0.0,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
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
    budget_row_scaling_policy: str = "absolute",
    field_provenance: Mapping[str, str] | None = None,
) -> AlgorithmV11ThermoValidContinuationCallsiteReport:
    """Filter thermo-invalid support and run algorithm-v1.1 continuation."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for algorithm-v1.1 callsites.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    if state.rho is None:
        raise ValueError("state.rho is required for algorithm-v1.1 callsites.")
    ln_mk = _as_vector(state.ln_mk, "state.ln_mk")
    rho = _as_vector(state.rho, "state.rho")
    eta = np.exp(rho)
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    filtered = filter_thermo_valid_condensate_support(
        explicit_opt_in=True,
        support_indices=support_indices,
        condensate_standard_source=condensate_standard_source,
        formula_matrix_cond_active=formula_matrix_cond_active,
        ln_mk=ln_mk,
        rho=rho,
        eta=eta,
        species_names=species_names,
        sentinel_abs_threshold=sentinel_abs_threshold,
        field_provenance=field_provenance or state.field_provenance,
    )
    active_matrix = np.asarray(formula_matrix_cond_active, dtype=np.float64)
    if active_matrix.ndim != 2 or active_matrix.shape[0] != target.shape[0]:
        raise ValueError("formula_matrix_cond_active row count must match element_inventory_target.")
    if active_matrix.shape[1] != ln_mk.shape[0]:
        raise ValueError("formula_matrix_cond_active column count must match state.ln_mk.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    removed_support_indices: tuple[int, ...] = ()
    removed_amounts = np.zeros((0,), dtype=np.float64)
    removed_budget = np.zeros_like(target, dtype=np.float64)
    if filtered.report.removed_local_indices:
        removed = np.asarray(filtered.report.removed_local_indices, dtype=np.int64)
        removed_support_indices = tuple(int(index) for index in filtered.report.removed_support_indices)
        removed_amounts = np.exp(ln_mk[removed])
        removed_budget = active_matrix[:, removed] @ removed_amounts
        external_budget = external_budget + removed_budget
    filtered_state = build_pdipm_rgie_condensate_state(
        ln_nk=state.ln_nk,
        ln_mk=filtered.ln_mk or (),
        element_potential=state.element_potential,
        ln_ntot=state.ln_ntot,
        rho=filtered.rho,
        eta=filtered.eta,
        field_provenance=state.field_provenance,
    )
    continuation = run_algorithm_v11_pdipm_continuation(
        explicit_opt_in=True,
        state=filtered_state,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=filtered.formula_matrix_cond_active or (),
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_budget,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=filtered.condensate_standard_source,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        barrier_schedule_policy=barrier_schedule_policy,
        ipopt_mu_linear_decrease_factor=ipopt_mu_linear_decrease_factor,
        ipopt_mu_superlinear_decrease_power=ipopt_mu_superlinear_decrease_power,
        ipopt_enable_superlinear_decrease=ipopt_enable_superlinear_decrease,
        max_outer_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations,
        center_tolerance_multiplier=center_tolerance_multiplier,
        center_metric_policy=center_metric_policy,
        alpha_grid=alpha_grid,
        equality_penalty_weight=equality_penalty_weight,
        total_density_penalty_weight=total_density_penalty_weight,
        max_abs_delta_q=max_abs_delta_q,
        max_abs_delta_r=max_abs_delta_r,
        max_abs_delta_rho=max_abs_delta_rho,
        max_abs_delta_lambda=max_abs_delta_lambda,
        direction_policy=direction_policy,
        algorithm_fraction_grid=algorithm_fraction_grid,
        trial_acceptance_policy=trial_acceptance_policy,
        filter_component_weights=filter_component_weights,
        filter_component_scale_policy=filter_component_scale_policy,
        ipopt_h_type_component_weights=ipopt_h_type_component_weights,
        ipopt_h_type_theta_reduction_fraction=ipopt_h_type_theta_reduction_fraction,
        ipopt_h_type_protected_components=ipopt_h_type_protected_components,
        ipopt_h_type_protected_component_max_normalized_increase=(
            ipopt_h_type_protected_component_max_normalized_increase
        ),
        persistent_filter_gamma_p=persistent_filter_gamma_p,
        persistent_filter_gamma_theta=persistent_filter_gamma_theta,
        persistent_filter_theta_max_factor=persistent_filter_theta_max_factor,
        strict_barrier_update_components=strict_barrier_update_components,
        strict_barrier_update_threshold=strict_barrier_update_threshold,
        center_component_weights=center_component_weights,
        center_component_scales=center_component_scales,
        enable_native_soft_restoration_fallback=enable_native_soft_restoration_fallback,
        soft_restoration_component_weights=soft_restoration_component_weights,
        soft_restoration_proximity_weight=soft_restoration_proximity_weight,
        soft_restoration_max_proximity=soft_restoration_max_proximity,
        enable_dedicated_restoration_filter_phase=enable_dedicated_restoration_filter_phase,
        dedicated_restoration_component_weights=dedicated_restoration_component_weights,
        dedicated_restoration_max_proximity=dedicated_restoration_max_proximity,
        require_residual_nonworsening=require_residual_nonworsening,
        residual_worsening_tolerance=residual_worsening_tolerance,
        budget_row_scaling_policy=budget_row_scaling_policy,
    )
    return AlgorithmV11ThermoValidContinuationCallsiteReport(
        report_schema="exogibbs_algorithm_v11_thermo_valid_continuation_callsite_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        original_support_count=filtered.report.original_support_count,
        filtered_support_count=filtered.report.filtered_support_count,
        removed_support_count=filtered.report.removed_support_count,
        external_condensate_support_indices=removed_support_indices,
        external_condensate_amounts=tuple(float(value) for value in removed_amounts),
        external_condensate_budget=tuple(float(value) for value in removed_budget),
        filter_report=filtered.report,
        continuation_report=continuation,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "AlgorithmV11ThermoValidContinuationCallsiteReport",
    "AlgorithmV11ThermoValidCallsiteReport",
    "algorithm_v11_experimental_high_start_callsite_policy",
    "run_algorithm_v11_thermo_valid_continuation_callsite",
    "run_algorithm_v11_thermo_valid_reduced_callsite",
)
