"""HEAD route lifecycle facade for condensate equilibrium."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from exogibbs.condensates.head_route_acceptance import TIGHT_RESIDUAL_STATUS
from exogibbs.optimize.condensate_algorithm_v11_callsite import (
    algorithm_v11_experimental_high_start_callsite_policy,
    run_algorithm_v11_thermo_valid_continuation_callsite,
)
from exogibbs.condensates.head_route_selector import (
    case_family_from_case_id,
    select_algorithm_v11_integrated_route,
)
from exogibbs.condensates.center_primary_fallback import (
    CenterPrimaryFallbackCandidate,
    CenterPrimaryFallbackReport,
    select_center_primary_fallback,
)
from exogibbs.condensates.continuation_input import (
    CondensateContinuationInput,
    build_condensate_continuation_input,
)
from exogibbs.condensates.electron_refresh import (
    ElectronRefreshReport,
    check_source_convention_safe_electron_refresh,
)
from exogibbs.condensates.frontier_refresh import (
    select_frontier_refresh_from_metrics,
)
from exogibbs.condensates.route_result import (
    CondensateHeadRouteResult,
    build_head_route_lifecycle_result,
)
from exogibbs.condensates.support_boundary import (
    CondensateSupportBoundary,
    build_condensate_support_boundary,
)


@dataclass(frozen=True)
class CondensateHeadRouteLifecycleReport:
    """Integrated lifecycle report for the condensate HEAD route."""

    report_schema: str
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    family: str
    support_boundary: CondensateSupportBoundary
    continuation_input: CondensateContinuationInput
    primary_summary: Mapping[str, Any]
    primary_execution_report: Mapping[str, Any] | None
    center_fallback_report: CenterPrimaryFallbackReport | None
    electron_refresh_report: ElectronRefreshReport | None
    frontier_refresh_report: Mapping[str, Any] | None
    route_selection_report: Mapping[str, Any]
    route_result: CondensateHeadRouteResult

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["support_boundary"] = self.support_boundary.as_dict()
        payload["continuation_input"] = self.continuation_input.as_dict()
        payload["primary_execution_report"] = (
            dict(self.primary_execution_report)
            if self.primary_execution_report is not None
            else None
        )
        payload["center_fallback_report"] = (
            self.center_fallback_report.as_dict()
            if self.center_fallback_report is not None
            else None
        )
        payload["electron_refresh_report"] = (
            self.electron_refresh_report.as_dict()
            if self.electron_refresh_report is not None
            else None
        )
        payload["route_result"] = self.route_result.as_dict()
        return payload


def _continuation_report_summary(report_payload: Mapping[str, Any]) -> dict[str, Any]:
    continuation = report_payload.get("continuation_report", {})
    if not isinstance(continuation, Mapping):
        continuation = {}
    return {
        "row_status": (
            "centered"
            if bool(continuation.get("converged_at_final_barrier", False))
            else "not_centered"
        ),
        "converged_at_final_barrier": bool(
            continuation.get("converged_at_final_barrier", False)
        ),
        "reached_final_barrier": bool(continuation.get("reached_final_barrier", False)),
        "stopped_reason": str(continuation.get("stopped_reason", "unknown")),
        "outer_iteration_count": int(continuation.get("outer_iteration_count", 0)),
        "inner_iteration_count": int(continuation.get("inner_iteration_count", 0)),
        "continuation_report": dict(continuation),
    }


def _final_outer_record(report_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    continuation = report_payload.get("continuation_report", {})
    if not isinstance(continuation, Mapping):
        return {}
    outer_records = continuation.get("outer_records", ())
    if not outer_records:
        return {}
    final_outer = outer_records[-1]
    return dict(final_outer) if isinstance(final_outer, Mapping) else {}


def _center_ratio_from_report(report_payload: Mapping[str, Any]) -> float:
    final_outer = _final_outer_record(report_payload)
    center_metric = float(final_outer.get("center_metric_after_outer", float("inf")))
    center_threshold = float(final_outer.get("center_threshold", 1.0))
    if center_threshold <= 0.0:
        return float("inf")
    return float(center_metric / center_threshold)


def _component_from_report(
    report_payload: Mapping[str, Any],
    component_name: str,
    default: float = float("inf"),
) -> float:
    final_outer = _final_outer_record(report_payload)
    components = final_outer.get("residual_components_after_outer", {})
    if not isinstance(components, Mapping):
        return float(default)
    return float(components.get(component_name, default))


def _final_components_from_summary(primary_summary: Mapping[str, Any]) -> Mapping[str, float]:
    continuation = primary_summary.get("continuation_report")
    if not isinstance(continuation, Mapping):
        return {}
    outer_records = continuation.get("outer_records", ())
    if not outer_records:
        return {}
    final_outer = outer_records[-1]
    if not isinstance(final_outer, Mapping):
        return {}
    components = final_outer.get("residual_components_after_outer", {})
    if not isinstance(components, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in components.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[str(key)] = number
    return out


def _guarded_primary_metric_status(
    *,
    primary_summary: Mapping[str, Any],
    route_selected: str,
    integrated_status: str,
    primary_acceptance_guard: str | None,
    max_budget: float,
    max_amount_weighted_gas: float,
    max_gas_stationarity: float,
    max_condensate_stationarity: float,
) -> tuple[str | None, dict[str, Any] | None]:
    if primary_acceptance_guard is None:
        return None, None
    if primary_acceptance_guard != "tight_weighted_components":
        raise ValueError(
            "primary_acceptance_guard must be None or 'tight_weighted_components'."
        )
    is_primary_route = str(route_selected) == "m4310_full_promoted_policy_route"
    if str(integrated_status) != "accepted" or not is_primary_route:
        return None, None
    components = _final_components_from_summary(primary_summary)
    limits = {
        "budget": float(max_budget),
        "amount_weighted_gas": float(max_amount_weighted_gas),
        "gas": float(max_gas_stationarity),
        "condensate": float(max_condensate_stationarity),
    }
    checks = {
        name: (
            name in components
            and math.isfinite(components[name])
            and components[name] <= limit
        )
        for name, limit in limits.items()
    }
    accepted = all(checks.values())
    report = {
        "guard_schema": "exogibbs_head_route_primary_acceptance_guard_v1",
        "guard_name": "tight_weighted_components",
        "accepted": bool(accepted),
        "component_limits": limits,
        "component_values": {name: components.get(name) for name in limits},
        "component_checks": checks,
        "metric_status": TIGHT_RESIDUAL_STATUS
        if accepted
        else "guarded_primary_tight_weighted_components_failed",
    }
    return str(report["metric_status"]), report


def _run_primary_continuation(
    *,
    continuation_input: CondensateContinuationInput,
    primary_continuation_policy: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = dict(
        algorithm_v11_experimental_high_start_callsite_policy()
    )
    if primary_continuation_policy is not None:
        policy.update(primary_continuation_policy)
    for metadata_key in ("candidate_name", "policy_name", "candidate_kind", "floor_value"):
        policy.pop(metadata_key, None)
    report = run_algorithm_v11_thermo_valid_continuation_callsite(
        explicit_opt_in=True,
        state=continuation_input.state,
        support_indices=continuation_input.support_indices,
        formula_matrix=continuation_input.formula_matrix,
        formula_matrix_cond_active=continuation_input.formula_matrix_cond_active,
        element_inventory_target=continuation_input.element_inventory_target,
        external_condensate_budget=continuation_input.external_condensate_budget,
        gas_stationarity_source=continuation_input.gas_stationarity_source,
        condensate_standard_source=continuation_input.condensate_standard_source,
        field_provenance=continuation_input.field_provenance,
        **policy,
    )
    payload = report.as_dict()
    return _continuation_report_summary(payload), payload


def _run_center_fallback_candidates(
    *,
    continuation_input: CondensateContinuationInput,
    center_fallback_continuation_policies: Sequence[Mapping[str, Any]],
) -> tuple[CenterPrimaryFallbackCandidate, ...]:
    candidates: list[CenterPrimaryFallbackCandidate] = []
    for index, policy in enumerate(center_fallback_continuation_policies):
        summary, payload = _run_primary_continuation(
            continuation_input=continuation_input,
            primary_continuation_policy=policy,
        )
        budget = _component_from_report(payload, "budget")
        candidates.append(
            CenterPrimaryFallbackCandidate(
                candidate_name=str(policy.get("candidate_name", f"center_fallback_{index}")),
                converged_at_final_barrier=bool(summary["converged_at_final_barrier"]),
                final_center_ratio=_center_ratio_from_report(payload),
                budget_ratio=float(budget),
                metadata={
                    "continuation_report": payload,
                    "stopped_reason": summary["stopped_reason"],
                },
            )
        )
    return tuple(candidates)


def _run_frontier_refresh_candidates(
    *,
    continuation_input: CondensateContinuationInput,
    frontier_refresh_continuation_policies: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for index, policy in enumerate(frontier_refresh_continuation_policies):
        summary, payload = _run_primary_continuation(
            continuation_input=continuation_input,
            primary_continuation_policy=policy,
        )
        candidates.append(
            {
                "policy_name": str(policy.get("policy_name", "adaptive_floor_frontier_repair")),
                "candidate_kind": str(policy.get("candidate_kind", "continuation_refresh_candidate")),
                "floor_value": policy.get("floor_value"),
                "solver_success": True,
                "reached_final_barrier": bool(summary["reached_final_barrier"]),
                "converged_at_final_barrier": bool(summary["converged_at_final_barrier"]),
                "budget": _component_from_report(payload, "budget"),
                "amount_weighted_gas": _component_from_report(payload, "amount_weighted_gas"),
                "complementarity": _component_from_report(payload, "complementarity"),
                "metadata": {
                    "continuation_report": payload,
                    "stopped_reason": summary["stopped_reason"],
                    "candidate_index": index,
                },
            }
        )
    return tuple(candidates)


def run_condensate_head_route_lifecycle(
    *,
    explicit_opt_in: bool,
    case_id: str,
    ln_nk: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    element_potential: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    primary_summary: Mapping[str, Any] | None = None,
    primary_continuation_policy: Mapping[str, Any] | None = None,
    center_fallback_candidates: Sequence[CenterPrimaryFallbackCandidate] = (),
    center_fallback_continuation_policies: Sequence[Mapping[str, Any]] = (),
    electron_refresh_enabled: bool = False,
    refresh_policy_summary: Mapping[str, Any] | None = None,
    frontier_refresh_candidate_metrics: Sequence[Mapping[str, Any]] = (),
    frontier_refresh_continuation_policies: Sequence[Mapping[str, Any]] = (),
    max_frontier_refresh_budget: float = 1.0e-8,
    max_frontier_refresh_amount_weighted_gas: float = 1.0,
    primary_acceptance_guard: str | None = None,
    primary_guard_max_budget: float = 1.0e-8,
    primary_guard_max_amount_weighted_gas: float = 1.0e-8,
    primary_guard_max_gas_stationarity: float = 1.0,
    primary_guard_max_condensate_stationarity: float = 10.0,
    metric_status: str | None = None,
    selected_route_override: str | None = None,
    initial_epsilon: float = -27.631021115928547,
    field_provenance: Mapping[str, str] | None = None,
) -> CondensateHeadRouteLifecycleReport:
    """Run the lightweight HEAD route lifecycle facade from explicit arrays."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for condensate HEAD route lifecycle.")
    family = case_family_from_case_id(case_id)
    support_boundary = build_condensate_support_boundary(
        explicit_opt_in=True,
        ln_nk=ln_nk,
        support_indices=support_indices,
        support_amounts=support_amounts,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        field_provenance=field_provenance,
    )
    continuation_input = build_condensate_continuation_input(
        explicit_opt_in=True,
        ln_nk=support_boundary.ln_nk,
        ln_mk=support_boundary.ln_mk,
        element_potential=element_potential,
        support_indices=support_boundary.support_indices,
        formula_matrix=support_boundary.formula_matrix,
        formula_matrix_cond_active=support_boundary.formula_matrix_cond_active,
        element_inventory_target=support_boundary.element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=condensate_standard_source,
        ln_ntot=support_boundary.ln_ntot,
        epsilon=initial_epsilon,
        field_provenance=field_provenance,
    )
    primary_execution_report: Mapping[str, Any] | None = None
    if primary_summary is None:
        primary, primary_execution_report = _run_primary_continuation(
            continuation_input=continuation_input,
            primary_continuation_policy=primary_continuation_policy,
        )
    else:
        primary = dict(
            primary_summary
            or {
            "row_status": "not_centered",
            "converged_at_final_barrier": False,
            "reason": "primary_continuation_not_run",
            }
        )
    fallback_candidates = tuple(center_fallback_candidates)
    if not fallback_candidates and center_fallback_continuation_policies:
        fallback_candidates = _run_center_fallback_candidates(
            continuation_input=continuation_input,
            center_fallback_continuation_policies=center_fallback_continuation_policies,
        )
    center_report = (
        select_center_primary_fallback(
            explicit_opt_in=True,
            primary_summary=primary,
            candidates=fallback_candidates,
        )
        if fallback_candidates
        else None
    )
    electron_report = (
        check_source_convention_safe_electron_refresh(
            explicit_opt_in=True,
            ln_nk=support_boundary.ln_nk,
            element_potential=element_potential,
            formula_matrix=support_boundary.formula_matrix,
            gas_stationarity_source=gas_stationarity_source,
        )
        if electron_refresh_enabled
        else None
    )
    frontier_metrics = tuple(frontier_refresh_candidate_metrics)
    if not frontier_metrics and frontier_refresh_continuation_policies:
        frontier_metrics = _run_frontier_refresh_candidates(
            continuation_input=continuation_input,
            frontier_refresh_continuation_policies=frontier_refresh_continuation_policies,
        )
    frontier_report = (
        select_frontier_refresh_from_metrics(
            explicit_opt_in=True,
            case_id=case_id,
            candidate_metrics=frontier_metrics,
            max_accepted_budget=max_frontier_refresh_budget,
            max_accepted_amount_weighted_gas=max_frontier_refresh_amount_weighted_gas,
        ).as_dict()
        if frontier_metrics
        else None
    )
    refresh_summary: Mapping[str, Any] | None = (
        dict(refresh_policy_summary) if refresh_policy_summary is not None else frontier_report
    )
    if refresh_summary is None and electron_report is not None and electron_report.accepted:
        refresh_summary = {
            "accepted": True,
            "selected_policy": "fastchem4_style_electron_refresh_route",
            "electron_refresh_report": electron_report.as_dict(),
        }
    route_selection = select_algorithm_v11_integrated_route(
        explicit_opt_in=True,
        case_id=case_id,
        primary_summary=primary,
        fallback_summary=center_report.as_dict() if center_report is not None else None,
        refresh_policy_summary=refresh_summary,
    )
    guarded_metric_status, primary_guard_report = _guarded_primary_metric_status(
        primary_summary=primary,
        route_selected=selected_route_override or route_selection.selected_route,
        integrated_status=route_selection.integrated_status,
        primary_acceptance_guard=primary_acceptance_guard,
        max_budget=primary_guard_max_budget,
        max_amount_weighted_gas=primary_guard_max_amount_weighted_gas,
        max_gas_stationarity=primary_guard_max_gas_stationarity,
        max_condensate_stationarity=primary_guard_max_condensate_stationarity,
    )
    route_result = build_head_route_lifecycle_result(
        explicit_opt_in=True,
        case_id=case_id,
        family=family,
        selected_route=selected_route_override or route_selection.selected_route,
        integrated_status=route_selection.integrated_status,
        metric_status=metric_status or guarded_metric_status,
        diagnostics={
            "support_boundary_budget_l2": support_boundary.budget_residual_l2,
            "gas_lambda_gauge_residual_l2": (
                continuation_input.gas_lambda_gauge_residual_l2
            ),
            "primary_acceptance_guard": primary_guard_report,
        },
    )
    return CondensateHeadRouteLifecycleReport(
        report_schema="exogibbs_condensate_head_route_lifecycle_report_v1",
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(case_id),
        family=family,
        support_boundary=support_boundary,
        continuation_input=continuation_input,
        primary_summary=primary,
        primary_execution_report=primary_execution_report,
        center_fallback_report=center_report,
        electron_refresh_report=electron_report,
        frontier_refresh_report=frontier_report,
        route_selection_report=route_selection.as_dict(),
        route_result=route_result,
    )


__all__ = (
    "CondensateHeadRouteLifecycleReport",
    "run_condensate_head_route_lifecycle",
)
