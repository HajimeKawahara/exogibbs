"""Explicit budget-centering seeded restricted-solver callsite adapter.

This module is diagnostic and production-adjacent only. It does not import
FastChem4, call pyfastchem, change production solver return signatures, or wire
any default path. It only forwards explicit native support and amount seeds to
the existing restricted-support condensate solver when explicitly requested.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ThermoState
from exogibbs.diagnostics.condensate_native_bundle import (
    validate_native_bundle_provenance,
)
from exogibbs.optimize.minimize_cond import (
    CondensateEquilibriumInit,
    CondensateRGIEReducedCouplingConfig,
    solve_restricted_support_condensate_layer,
)


LOWDIM_BASE_GAS_STEP_SCALE = 1.0 / 4445.0
LOWDIM_ALPHA_Q = 0.06951727728082234
LOWDIM_ALPHA_QTOT = 1.0
LOWDIM_ALPHA_M = 0.4786524590532266


@dataclass(frozen=True)
class BudgetCenteringSeededCallsiteReport:
    """Report returned by the explicit budget-centering seeded callsite."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    support_indices_shape_matches: bool
    support_amounts_init_shape_matches: bool
    finite_solver_inputs: bool
    budget_fraction_before_solver: float | None
    solver_called: bool
    solver_success: bool | None
    solver_status: int | None
    post_solver_budget_residual: float | None
    post_solver_kkt_residual_diagnostic: float | None
    post_solver_kkt_residual_log_variable_diagnostic: float | None
    post_solver_negative_budget_inf: float | None
    line_search_selection_policy: str
    support_size: int
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool
    field_provenance: Mapping[str, str]
    reduced_coupling_config_mode: str | None = None
    reduced_coupling_selected_mode: str | None = None
    reduced_coupling_selected_alpha_s: float | None = None
    gas_step_scale: float | None = None
    gas_step_direction_sign: float | None = None
    ntot_step_scale: float | None = None
    condensate_step_scale: float | None = None
    initial_residual_policy: str | None = None
    inactive_positive_count: int | None = None
    top_positive_inactive_indices: tuple[int, ...] = ()
    top_inactive_names: tuple[str, ...] = ()
    top_inactive_driving: tuple[float, ...] = ()
    row_resolved_stationarity_available: bool = False
    active_stationarity_row_names: tuple[str, ...] = ()
    active_stationarity_driving: tuple[float, ...] = ()
    active_stationarity_abs_top_names: tuple[str, ...] = ()
    active_stationarity_abs_top_values: tuple[float, ...] = ()
    complementarity_row_names: tuple[str, ...] = ()
    complementarity_values: tuple[float, ...] = ()
    complementarity_abs_top_names: tuple[str, ...] = ()
    complementarity_abs_top_values: tuple[float, ...] = ()
    gas_stationarity_values: tuple[float, ...] = ()
    gas_stationarity_abs_top_indices: tuple[int, ...] = ()
    gas_stationarity_abs_top_values: tuple[float, ...] = ()
    gas_stationarity_log_scaled_values: tuple[float, ...] = ()
    gas_stationarity_log_scaled_abs_top_indices: tuple[int, ...] = ()
    gas_stationarity_log_scaled_abs_top_values: tuple[float, ...] = ()
    gas_stationarity_abs_top_names: tuple[str, ...] = ()
    gas_stationarity_log_scaled_abs_top_names: tuple[str, ...] = ()
    gas_species_order: tuple[str, ...] = ()
    solver_n_iter: int | None = None
    solver_final_residual: float | None = None
    solver_final_step_size: float | None = None
    final_support_amounts: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_case_adaptive_lowdim_reduced_coupling_config(
    selected_variant: str,
    *,
    base_gas_step_scale: float = LOWDIM_BASE_GAS_STEP_SCALE,
) -> CondensateRGIEReducedCouplingConfig:
    """Build an explicit opt-in reduced-coupling config for M698 policy replay."""

    if base_gas_step_scale <= 0.0 or base_gas_step_scale > 1.0:
        raise ValueError("base_gas_step_scale must satisfy 0 < base_gas_step_scale <= 1.")
    if selected_variant == "baseline_current_reduced_callsite":
        return CondensateRGIEReducedCouplingConfig(
            gas_step_scale=base_gas_step_scale,
            gas_step_direction_sign=1.0,
            ntot_step_scale=None,
            condensate_step_scale=1.0,
            initial_residual_policy="computed_fresh",
        )
    if selected_variant == "lowdim_step_factor_candidate":
        return CondensateRGIEReducedCouplingConfig(
            gas_step_scale=base_gas_step_scale * LOWDIM_ALPHA_Q,
            gas_step_direction_sign=1.0,
            ntot_step_scale=base_gas_step_scale * LOWDIM_ALPHA_QTOT,
            condensate_step_scale=LOWDIM_ALPHA_M,
            initial_residual_policy="computed_fresh",
        )
    raise ValueError("selected_variant is not a supported case-adaptive policy variant.")


def _validate_support_inputs(
    support_indices: Sequence[int],
    support_amounts_init: Sequence[float],
    condensate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(support_indices, dtype=np.int64)
    amounts = np.asarray(support_amounts_init, dtype=np.float64)
    if indices.ndim != 1:
        raise ValueError("support_indices must be a one-dimensional vector.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts_init must be a one-dimensional vector.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts_init must have matching length.")
    if indices.shape[0] == 0:
        return indices, amounts
    if np.any(indices < 0) or np.any(indices >= int(condensate_count)):
        raise ValueError("support_indices contains an out-of-range condensate index.")
    if len(set(int(index) for index in indices.tolist())) != indices.shape[0]:
        raise ValueError("support_indices must not contain duplicates.")
    if not np.all(np.isfinite(amounts)):
        raise ValueError("support_amounts_init must contain finite values.")
    if np.any(amounts <= 0.0):
        raise ValueError("support_amounts_init must be positive for non-empty support.")
    return indices, amounts


def _budget_fraction(
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    support_indices: np.ndarray,
    support_amounts_init: np.ndarray,
) -> float | None:
    if support_indices.shape[0] == 0:
        return None
    ac = np.asarray(formula_matrix_cond, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    full = np.zeros((ac.shape[1],), dtype=np.float64)
    full[support_indices] = support_amounts_init
    burden = ac.dot(full)
    positive = np.abs(target) > 0.0
    if not np.any(positive):
        raise ValueError("element_inventory_target must contain a nonzero budget.")
    return float(np.max(np.abs(burden[positive]) / np.abs(target[positive])))


def run_explicit_budget_centering_seeded_callsite(
    *,
    explicit_opt_in: bool,
    state: ThermoState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    hvector_func: Callable[[Any], Any],
    hvector_cond_func: Callable[[Any], Any],
    condensate_species_order: Sequence[str],
    element_order: Sequence[str],
    gas_species_order: Sequence[str] | None = None,
    support_indices: Sequence[int],
    support_amounts_init: Sequence[float],
    initial_log_state_override: CondensateEquilibriumInit | None = None,
    max_budget_fraction: float = 1.0,
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 20,
    epsilon: float = -5.0,
    max_iter: int = 80,
    reduced_coupling_config: CondensateRGIEReducedCouplingConfig | None = None,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_name: str | None = None,
    line_search_charge_weight: float = 1.0,
    field_provenance: Mapping[str, str] | None = None,
) -> BudgetCenteringSeededCallsiteReport:
    """Run a restricted solver callsite with explicit budget-centered seeds."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for the seeded callsite.")
    if max_budget_fraction <= 0.0:
        raise ValueError("max_budget_fraction must be positive.")
    provenance = validate_native_bundle_provenance(field_provenance)
    condensate_count = len(tuple(condensate_species_order))
    indices, amounts = _validate_support_inputs(
        support_indices,
        support_amounts_init,
        condensate_count,
    )
    shape_matches = indices.shape[0] == amounts.shape[0]
    finite_inputs = bool(
        shape_matches
        and np.all(np.isfinite(indices))
        and np.all(np.isfinite(amounts))
        and (indices.shape[0] == 0 or np.all(amounts > 0.0))
    )
    budget_fraction = _budget_fraction(
        formula_matrix_cond,
        np.asarray(state.element_vector, dtype=np.float64).tolist(),
        indices,
        amounts,
    )
    if budget_fraction is not None and budget_fraction > float(max_budget_fraction):
        raise ValueError("support_amounts_init exceeds max_budget_fraction.")
    if indices.shape[0] == 0:
        return BudgetCenteringSeededCallsiteReport(
            report_schema="exogibbs_budget_centering_seeded_callsite_report_v1",
            diagnostic_only=True,
            default_off=True,
            explicit_opt_in=True,
            production_behavior_change=False,
            production_return_signature_change=False,
            preset_default_wiring_change=False,
            support_indices_shape_matches=True,
            support_amounts_init_shape_matches=True,
            finite_solver_inputs=True,
            budget_fraction_before_solver=budget_fraction,
            solver_called=False,
            solver_success=None,
            solver_status=None,
            post_solver_budget_residual=None,
            post_solver_kkt_residual_diagnostic=None,
            post_solver_kkt_residual_log_variable_diagnostic=None,
            post_solver_negative_budget_inf=None,
            line_search_selection_policy=line_search_selection_policy,
            support_size=0,
            support_indices=(),
            support_amounts_init=(),
            fastchem4_trace_values_used=False,
            fastchem4_public_values_used_as_constructor_inputs=False,
            field_provenance=provenance,
            gas_step_scale=(
                None
                if reduced_coupling_config is None
                else float(reduced_coupling_config.gas_step_scale)
            ),
            gas_step_direction_sign=(
                None
                if reduced_coupling_config is None
                else float(reduced_coupling_config.gas_step_direction_sign)
            ),
            ntot_step_scale=(
                None
                if reduced_coupling_config is None
                else (
                    float(reduced_coupling_config.gas_step_scale)
                    if reduced_coupling_config.ntot_step_scale is None
                    else float(reduced_coupling_config.ntot_step_scale)
                )
            ),
            condensate_step_scale=(
                None
                if reduced_coupling_config is None
                else float(reduced_coupling_config.condensate_step_scale)
            ),
            initial_residual_policy=(
                None
                if reduced_coupling_config is None
                else str(reduced_coupling_config.initial_residual_policy)
            ),
        )
    charge_row_index = None
    if line_search_charge_row_name is not None:
        try:
            charge_row_index = tuple(element_order).index(str(line_search_charge_row_name))
        except ValueError as exc:
            raise ValueError("line_search_charge_row_name must be present in element_order.") from exc
    result = solve_restricted_support_condensate_layer(
        state,
        jnp.asarray(formula_matrix, dtype=jnp.float64),
        jnp.asarray(formula_matrix_cond, dtype=jnp.float64),
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        support_indices=[int(index) for index in indices.tolist()],
        condensate_species=condensate_species_order,
        element_names=element_order,
        support_amounts_init=jnp.asarray(amounts, dtype=jnp.float64),
        initial_log_state_override=initial_log_state_override,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        epsilon=epsilon,
        max_iter=max_iter,
        reduced_coupling_config=reduced_coupling_config,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )
    budget_residual = float(result["feasibility_residual_inf"])
    kkt_residual = float(result["restricted_kkt_gap_inf"])
    log_variable_kkt_residual = float(result["restricted_kkt_gap_log_variable_inf"])
    negative_budget = float(result["negative_budget_inf"])
    diagnostics = result["diagnostics"]
    reduced_config_mode = (
        None
        if reduced_coupling_config is None
        else reduced_coupling_config.reduced_coupling_mode
    )
    if reduced_coupling_config is None:
        reduced_selected_mode = "current"
    elif reduced_coupling_config.reduced_coupling_mode == "capped_s_only_fixed_alpha":
        reduced_selected_mode = "capped_s_only"
    elif reduced_coupling_config.reduced_coupling_mode in (
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_weighted_mask",
    ):
        reduced_selected_mode = reduced_coupling_config.reduced_coupling_mode
    elif reduced_coupling_config.reduced_coupling_mode == "capped_s_only_conditional":
        reduced_selected_mode = None
    else:
        reduced_selected_mode = None
    reduced_selected_alpha = diagnostics.get("reduced_coupling_selected_alpha_s")
    finite_output = all(
        math.isfinite(value) for value in (budget_residual, kkt_residual, negative_budget)
    )
    full_driving = np.asarray(result["full_driving"], dtype=np.float64)
    active_driving = full_driving[indices] if indices.shape[0] else np.asarray([], dtype=np.float64)
    complementarity = np.asarray(result["complementarity"], dtype=np.float64)
    gas_stationarity = np.asarray(result["gas_stationarity"], dtype=np.float64)
    gas_stationarity_log_scaled = np.asarray(
        result["gas_stationarity_log_scaled"], dtype=np.float64
    )
    support_names = tuple(str(name) for name in result["support_names"])
    gas_names = (
        tuple(str(name) for name in gas_species_order)
        if gas_species_order is not None
        else tuple(str(index) for index in range(gas_stationarity.size))
    )
    top_active_positions = np.argsort(-np.abs(active_driving))[: min(10, active_driving.size)]
    top_complementarity_positions = np.argsort(-np.abs(complementarity))[
        : min(10, complementarity.size)
    ]
    top_gas_positions = np.argsort(-np.abs(gas_stationarity))[: min(10, gas_stationarity.size)]
    top_gas_log_positions = np.argsort(-np.abs(gas_stationarity_log_scaled))[
        : min(10, gas_stationarity_log_scaled.size)
    ]
    return BudgetCenteringSeededCallsiteReport(
        report_schema="exogibbs_budget_centering_seeded_callsite_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        support_indices_shape_matches=shape_matches,
        support_amounts_init_shape_matches=shape_matches,
        finite_solver_inputs=bool(finite_inputs and finite_output),
        budget_fraction_before_solver=budget_fraction,
        solver_called=True,
        solver_success=bool(result["solver_success"]),
        solver_status=int(result["solver_status"]),
        post_solver_budget_residual=budget_residual,
        post_solver_kkt_residual_diagnostic=kkt_residual,
        post_solver_kkt_residual_log_variable_diagnostic=log_variable_kkt_residual,
        post_solver_negative_budget_inf=negative_budget,
        line_search_selection_policy=line_search_selection_policy,
        support_size=int(indices.shape[0]),
        support_indices=tuple(int(index) for index in indices.tolist()),
        support_amounts_init=tuple(float(amount) for amount in amounts.tolist()),
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
        field_provenance=provenance,
        reduced_coupling_config_mode=(
            None if reduced_config_mode is None else str(reduced_config_mode)
        ),
        reduced_coupling_selected_mode=(
            None if reduced_selected_mode is None else str(reduced_selected_mode)
        ),
        reduced_coupling_selected_alpha_s=(
            None if reduced_selected_alpha is None else float(reduced_selected_alpha)
        ),
        gas_step_scale=(
            None
            if reduced_coupling_config is None
            else float(reduced_coupling_config.gas_step_scale)
        ),
        gas_step_direction_sign=(
            None
            if reduced_coupling_config is None
            else float(reduced_coupling_config.gas_step_direction_sign)
        ),
        ntot_step_scale=(
            None
            if reduced_coupling_config is None
            else (
                float(reduced_coupling_config.gas_step_scale)
                if reduced_coupling_config.ntot_step_scale is None
                else float(reduced_coupling_config.ntot_step_scale)
            )
        ),
        condensate_step_scale=(
            None
            if reduced_coupling_config is None
            else float(reduced_coupling_config.condensate_step_scale)
        ),
        initial_residual_policy=(
            None
            if reduced_coupling_config is None
            else str(reduced_coupling_config.initial_residual_policy)
        ),
        inactive_positive_count=int(result["inactive_positive_count"]),
        top_positive_inactive_indices=tuple(
            int(index) for index in result["top_positive_inactive_indices"]
        ),
            top_inactive_names=tuple(str(name) for name in result["top_inactive_names"]),
            top_inactive_driving=tuple(
                float(value) for value in result["top_inactive_driving"]
            ),
            row_resolved_stationarity_available=True,
            active_stationarity_row_names=support_names,
            active_stationarity_driving=tuple(float(value) for value in active_driving.tolist()),
            active_stationarity_abs_top_names=tuple(
                support_names[int(position)] for position in top_active_positions.tolist()
            ),
            active_stationarity_abs_top_values=tuple(
                float(abs(active_driving[int(position)]))
                for position in top_active_positions.tolist()
            ),
            complementarity_row_names=support_names,
            complementarity_values=tuple(float(value) for value in complementarity.tolist()),
            complementarity_abs_top_names=tuple(
                support_names[int(position)]
                for position in top_complementarity_positions.tolist()
            ),
            complementarity_abs_top_values=tuple(
                float(abs(complementarity[int(position)]))
                for position in top_complementarity_positions.tolist()
            ),
            gas_stationarity_values=tuple(float(value) for value in gas_stationarity.tolist()),
            gas_stationarity_abs_top_indices=tuple(
                int(position) for position in top_gas_positions.tolist()
            ),
            gas_stationarity_abs_top_values=tuple(
                float(abs(gas_stationarity[int(position)]))
                for position in top_gas_positions.tolist()
            ),
            gas_stationarity_log_scaled_values=tuple(
                float(value) for value in gas_stationarity_log_scaled.tolist()
            ),
            gas_stationarity_log_scaled_abs_top_indices=tuple(
                int(position) for position in top_gas_log_positions.tolist()
            ),
            gas_stationarity_log_scaled_abs_top_values=tuple(
                float(abs(gas_stationarity_log_scaled[int(position)]))
                for position in top_gas_log_positions.tolist()
            ),
            gas_stationarity_abs_top_names=tuple(
                gas_names[int(position)] for position in top_gas_positions.tolist()
            ),
            gas_stationarity_log_scaled_abs_top_names=tuple(
                gas_names[int(position)] for position in top_gas_log_positions.tolist()
            ),
            gas_species_order=gas_names,
            solver_n_iter=int(result["diagnostics"]["n_iter"]),
            solver_final_residual=float(result["diagnostics"]["final_residual"]),
            solver_final_step_size=float(result["diagnostics"]["final_step_size"]),
            final_support_amounts=tuple(
                float(value) for value in result["m_support"].tolist()
            ),
    )


__all__ = (
    "BudgetCenteringSeededCallsiteReport",
    "run_explicit_budget_centering_seeded_callsite",
)
