"""Explicit component-safe restricted solver callsite experiment.

This module is diagnostic and production-adjacent only. It does not import
FastChem4, call pyfastchem, change production solver return signatures, or wire
any default behavior. It connects a validated component-safe payload to the
existing explicit restricted-support callsite only when explicitly requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from exogibbs.api.chemistry import ThermoState
from exogibbs.diagnostics.condensate_budget_centering_seeded_callsite import (
    BudgetCenteringSeededCallsiteReport,
    run_explicit_budget_centering_seeded_callsite,
)
from exogibbs.diagnostics.condensate_component_safe_callsite_adapter import (
    ComponentSafeCallsiteInputs,
    build_component_safe_callsite_inputs,
)
from exogibbs.diagnostics.condensate_residual_balanced_direction import (
    ComponentSafePolicyPayload,
)
from exogibbs.optimize.minimize_cond import (
    CondensateEquilibriumInit,
    CondensateRGIEReducedCouplingConfig,
)


@dataclass(frozen=True)
class ComponentSafeRestrictedCallsiteExperimentReport:
    """Report for a component-safe payload at the restricted solver callsite."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    normal_default_path_unchanged: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    payload_kind: str
    selected_policy: str
    solver_call_policy: str
    adapter_inputs: ComponentSafeCallsiteInputs
    seeded_callsite_report: BudgetCenteringSeededCallsiteReport | None
    solver_called: bool
    solver_success: bool | None
    solver_status: int | None
    post_solver_budget_residual: float | None
    post_solver_kkt_residual_diagnostic: float | None
    post_solver_negative_budget_inf: float | None
    support_indices_shape_matches: bool
    support_amounts_init_shape_matches: bool
    finite_solver_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapter_inputs"] = self.adapter_inputs.as_dict()
        payload["seeded_callsite_report"] = (
            None
            if self.seeded_callsite_report is None
            else self.seeded_callsite_report.as_dict()
        )
        return payload


def run_component_safe_restricted_callsite_experiment(
    *,
    explicit_opt_in: bool,
    payload: ComponentSafePolicyPayload | Mapping[str, Any],
    state: ThermoState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    hvector_func: Callable[[Any], Any],
    hvector_cond_func: Callable[[Any], Any],
    condensate_species_order: Sequence[str],
    element_order: Sequence[str],
    gas_species_order: Sequence[str] | None = None,
    support_indices: Sequence[int] = (),
    support_amounts_init: Sequence[float] = (),
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
) -> ComponentSafeRestrictedCallsiteExperimentReport:
    """Run a component-safe payload through the explicit restricted callsite."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for component-safe restricted callsite.")
    adapter_inputs = build_component_safe_callsite_inputs(
        payload=payload,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        explicit_opt_in=True,
        field_provenance=field_provenance,
    )
    if adapter_inputs.solver_call_policy.startswith("skip_solver_"):
        return ComponentSafeRestrictedCallsiteExperimentReport(
            report_schema="exogibbs_component_safe_restricted_callsite_experiment_v1",
            diagnostic_only=True,
            default_off=True,
            explicit_opt_in=True,
            production_behavior_change=False,
            production_return_signature_change=False,
            preset_default_wiring_change=False,
            normal_default_path_unchanged=True,
            fastchem4_trace_public_runtime_constructor_inputs_used=False,
            case_id=adapter_inputs.case_id,
            payload_kind=adapter_inputs.payload_kind,
            selected_policy=adapter_inputs.selected_policy,
            solver_call_policy=adapter_inputs.solver_call_policy,
            adapter_inputs=adapter_inputs,
            seeded_callsite_report=None,
            solver_called=False,
            solver_success=None,
            solver_status=None,
            post_solver_budget_residual=None,
            post_solver_kkt_residual_diagnostic=None,
            post_solver_negative_budget_inf=None,
            support_indices_shape_matches=adapter_inputs.support_indices_shape_matches,
            support_amounts_init_shape_matches=adapter_inputs.support_amounts_init_shape_matches,
            finite_solver_inputs=adapter_inputs.finite_solver_inputs,
        )
    seeded_report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        state=state,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        condensate_species_order=condensate_species_order,
        element_order=element_order,
        gas_species_order=gas_species_order,
        support_indices=adapter_inputs.support_indices,
        support_amounts_init=adapter_inputs.support_amounts_init,
        initial_log_state_override=initial_log_state_override,
        max_budget_fraction=max_budget_fraction,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        epsilon=epsilon,
        max_iter=max_iter,
        reduced_coupling_config=reduced_coupling_config,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_name=line_search_charge_row_name,
        line_search_charge_weight=line_search_charge_weight,
        field_provenance=field_provenance,
    )
    return ComponentSafeRestrictedCallsiteExperimentReport(
        report_schema="exogibbs_component_safe_restricted_callsite_experiment_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        normal_default_path_unchanged=True,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=adapter_inputs.case_id,
        payload_kind=adapter_inputs.payload_kind,
        selected_policy=adapter_inputs.selected_policy,
        solver_call_policy=adapter_inputs.solver_call_policy,
        adapter_inputs=adapter_inputs,
        seeded_callsite_report=seeded_report,
        solver_called=seeded_report.solver_called,
        solver_success=seeded_report.solver_success,
        solver_status=seeded_report.solver_status,
        post_solver_budget_residual=seeded_report.post_solver_budget_residual,
        post_solver_kkt_residual_diagnostic=seeded_report.post_solver_kkt_residual_diagnostic,
        post_solver_negative_budget_inf=seeded_report.post_solver_negative_budget_inf,
        support_indices_shape_matches=(
            adapter_inputs.support_indices_shape_matches
            and seeded_report.support_indices_shape_matches
        ),
        support_amounts_init_shape_matches=(
            adapter_inputs.support_amounts_init_shape_matches
            and seeded_report.support_amounts_init_shape_matches
        ),
        finite_solver_inputs=adapter_inputs.finite_solver_inputs and seeded_report.finite_solver_inputs,
    )


__all__ = (
    "ComponentSafeRestrictedCallsiteExperimentReport",
    "run_component_safe_restricted_callsite_experiment",
)
