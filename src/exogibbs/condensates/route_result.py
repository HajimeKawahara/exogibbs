"""HEAD route lifecycle result normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from exogibbs.condensates.head_route_standard_gate import (
    BUDGET_TRADEOFF_STATUS,
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
    NOT_CONVERGED,
    RAW_GAS_CAVEAT_STATUS,
    TIGHT_RESIDUAL_STATUS,
    classify_head_route_standard_gate_row,
)


@dataclass(frozen=True)
class CondensateHeadRouteResult:
    """Normalized HEAD route lifecycle result."""

    result_schema: str
    case_id: str
    family: str
    selected_route: str
    integrated_status: str
    metric_status: str
    acceptance_tier: str
    standard_path_status: str
    converged: bool
    warning_messages: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_metric_status_from_selected_route(selected_route: str, integrated_status: str) -> str:
    """Infer the HEAD metric status from the selected route label."""

    if str(integrated_status) != "accepted":
        return "not_accepted"
    route = str(selected_route)
    if "budget_tradeoff" in route or "center_primary_budget_guard" in route:
        return BUDGET_TRADEOFF_STATUS
    if "electron" in route or "raw_gas" in route:
        return RAW_GAS_CAVEAT_STATUS
    return TIGHT_RESIDUAL_STATUS


def build_head_route_lifecycle_result(
    *,
    explicit_opt_in: bool,
    case_id: str,
    family: str,
    selected_route: str,
    integrated_status: str,
    metric_status: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> CondensateHeadRouteResult:
    """Normalize route selection into a standard lifecycle result."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for HEAD route lifecycle result.")
    metric = metric_status or infer_metric_status_from_selected_route(
        selected_route,
        integrated_status,
    )
    if str(integrated_status) == "accepted":
        gate = classify_head_route_standard_gate_row(
            condensate_enabled=True,
            case_id=case_id,
            family=family,
            selected_route=selected_route,
            metric_status=metric,
        )
        status = gate.standard_path_status
        tier = gate.acceptance_tier
        warnings = gate.warning_messages
    else:
        status = NOT_CONVERGED
        tier = "not_accepted"
        warnings = ("The HEAD route selector did not accept this lifecycle run.",)
    return CondensateHeadRouteResult(
        result_schema="exogibbs_condensate_head_route_result_v1",
        case_id=str(case_id),
        family=str(family),
        selected_route=str(selected_route),
        integrated_status=str(integrated_status),
        metric_status=str(metric),
        acceptance_tier=tier,
        standard_path_status=status,
        converged=status in {CONVERGED, CONVERGED_WITH_CAVEAT},
        warning_messages=warnings,
        diagnostics=dict(diagnostics or {}),
    )


__all__ = (
    "CondensateHeadRouteResult",
    "build_head_route_lifecycle_result",
    "infer_metric_status_from_selected_route",
)
