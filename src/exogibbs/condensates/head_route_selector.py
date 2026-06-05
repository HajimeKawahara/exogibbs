"""Explicit diagnostic route selector for algorithm-v1.1 condensate callsites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


DEFAULT_CENTER_FALLBACK_FAMILIES = ("lowT_strong_condensation_budget_stress",)


@dataclass(frozen=True)
class AlgorithmV11RouteSelectionReport:
    """Report for the explicit algorithm-v1.1 route selector."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    case_family: str
    selected_route: str
    integrated_status: str
    route_reason: str
    primary_centered: bool
    fallback_allowed: bool
    fallback_available: bool
    fallback_accepted: bool
    refresh_policy_available: bool
    refresh_policy_accepted: bool
    primary_summary: Mapping[str, Any]
    fallback_summary: Mapping[str, Any] | None
    refresh_policy_summary: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def case_family_from_case_id(case_id: str) -> str:
    """Return the case-family prefix before the first double underscore."""

    text = str(case_id)
    return text.split("__", 1)[0]


def _bool_field(payload: Mapping[str, Any], name: str) -> bool:
    return bool(payload.get(name, False))


def _primary_centered(primary_summary: Mapping[str, Any]) -> bool:
    if str(primary_summary.get("row_status", "")) == "centered":
        return True
    if _bool_field(primary_summary, "converged_at_final_barrier"):
        return True
    continuation = primary_summary.get("continuation_report")
    if isinstance(continuation, Mapping):
        return _bool_field(continuation, "converged_at_final_barrier")
    return False


def _fallback_accepted(fallback_summary: Mapping[str, Any] | None) -> bool:
    if fallback_summary is None:
        return False
    classification = str(fallback_summary.get("classification", ""))
    if classification.startswith("center_primary_budget_guard_accepts"):
        return True
    if _bool_field(fallback_summary, "converged_at_final_barrier"):
        return True
    best_row = fallback_summary.get("best_row")
    if isinstance(best_row, Mapping):
        advancement = best_row.get("advancement_summary")
        if isinstance(advancement, Mapping):
            return _bool_field(advancement, "converged_at_final_barrier")
    advancement = fallback_summary.get("advancement_summary")
    if isinstance(advancement, Mapping):
        return _bool_field(advancement, "converged_at_final_barrier")
    return False


def _refresh_policy_accepted(refresh_policy_summary: Mapping[str, Any] | None) -> bool:
    if refresh_policy_summary is None:
        return False
    return bool(refresh_policy_summary.get("accepted", False))


def select_algorithm_v11_integrated_route(
    *,
    explicit_opt_in: bool,
    case_id: str,
    primary_summary: Mapping[str, Any],
    fallback_summary: Mapping[str, Any] | None = None,
    refresh_policy_summary: Mapping[str, Any] | None = None,
    center_fallback_families: Sequence[str] = DEFAULT_CENTER_FALLBACK_FAMILIES,
) -> AlgorithmV11RouteSelectionReport:
    """Select the primary route or the center-primary fallback for one row.

    The selector is intentionally narrow: a centered primary continuation always
    wins, and center-primary fallback is allowed only for configured case
    families. It does not run solvers or construct inputs.
    """

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for algorithm-v1.1 route selection.")
    families = tuple(str(item) for item in center_fallback_families)
    if not families:
        raise ValueError("center_fallback_families must not be empty.")
    if len(set(families)) != len(families):
        raise ValueError("center_fallback_families must not contain duplicates.")
    family = case_family_from_case_id(case_id)
    primary_is_centered = _primary_centered(primary_summary)
    fallback_allowed = family in families
    fallback_available = fallback_summary is not None
    fallback_is_accepted = _fallback_accepted(fallback_summary)
    refresh_available = refresh_policy_summary is not None
    refresh_is_accepted = _refresh_policy_accepted(refresh_policy_summary)
    if primary_is_centered:
        selected_route = "m4310_full_promoted_policy_route"
        integrated_status = "accepted"
        reason = "The primary promoted policy route is already centered."
    elif fallback_allowed and fallback_available and fallback_is_accepted:
        selected_route = "m4326_center_primary_budget_guard_fallback"
        integrated_status = "accepted"
        reason = (
            "The primary route is not centered, and the configured case family "
            "has an accepted center-primary budget-guard fallback."
        )
    elif refresh_available and refresh_is_accepted:
        selected_route = str(refresh_policy_summary.get("selected_policy", "gas_boundary_refresh_policy"))
        integrated_status = "accepted"
        reason = "The reusable gas-boundary refresh policy has an accepted candidate."
    elif fallback_allowed and not fallback_available:
        selected_route = "m4326_center_primary_budget_guard_fallback_missing"
        integrated_status = "not_applicable"
        reason = "The configured fallback family has no fallback report."
    elif fallback_allowed:
        selected_route = "m4326_center_primary_budget_guard_fallback_rejected"
        integrated_status = "not_accepted"
        reason = "The configured fallback family has a fallback report, but it is not accepted."
    else:
        selected_route = "support_boundary_construction_required_before_selector"
        integrated_status = "not_applicable"
        reason = "No centered primary route or configured fallback route is available."
    return AlgorithmV11RouteSelectionReport(
        report_schema="exogibbs_algorithm_v11_route_selection_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(case_id),
        case_family=family,
        selected_route=selected_route,
        integrated_status=integrated_status,
        route_reason=reason,
        primary_centered=primary_is_centered,
        fallback_allowed=fallback_allowed,
        fallback_available=fallback_available,
        fallback_accepted=fallback_is_accepted,
        refresh_policy_available=refresh_available,
        refresh_policy_accepted=refresh_is_accepted,
        primary_summary=dict(primary_summary),
        fallback_summary=dict(fallback_summary) if fallback_summary is not None else None,
        refresh_policy_summary=(
            dict(refresh_policy_summary) if refresh_policy_summary is not None else None
        ),
    )


__all__ = (
    "AlgorithmV11RouteSelectionReport",
    "DEFAULT_CENTER_FALLBACK_FAMILIES",
    "case_family_from_case_id",
    "select_algorithm_v11_integrated_route",
)
