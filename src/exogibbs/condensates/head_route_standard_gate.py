"""HEAD route standard-path gate for condensate production entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from exogibbs.condensates.head_route_acceptance import (
    BUDGET_TRADEOFF_STATUS,
    RAW_GAS_CAVEAT_STATUS,
    TIGHT_RESIDUAL_STATUS,
    TIER_1,
    TIER_2,
    TIER_3,
    classify_head_route_acceptance_tier,
)


HEAD_ROUTE_STANDARD = "head_v1"
CONVERGED = "converged"
CONVERGED_WITH_CAVEAT = "converged_with_caveat"
NOT_CONVERGED = "not_converged"
UNSUPPORTED_INPUT = "unsupported_input"


@dataclass(frozen=True)
class HeadRouteStandardGateRow:
    """Standard-path classification for one HEAD route row."""

    report_schema: str
    condensate_enabled: bool
    route: str
    gas_only_default_path_protected: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    family: str
    selected_route: str
    metric_status: str
    acceptance_tier: str
    standard_path_status: str
    warning_messages: tuple[str, ...]
    result_metadata_required: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeadRouteStandardGateReport:
    """Aggregate standard-path gate report for HEAD route rows."""

    report_schema: str
    condensate_enabled: bool
    route: str
    gas_only_default_path_protected: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    row_count: int
    converged_count: int
    converged_with_caveat_count: int
    not_converged_count: int
    unsupported_input_count: int
    standard_path_rows: tuple[HeadRouteStandardGateRow, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["standard_path_rows"] = [row.as_dict() for row in self.standard_path_rows]
        return payload


def classify_head_route_standard_gate_row(
    *,
    condensate_enabled: bool,
    case_id: str,
    family: str,
    selected_route: str,
    metric_status: str,
    route: str = HEAD_ROUTE_STANDARD,
) -> HeadRouteStandardGateRow:
    """Classify one HEAD route row for the condensate standard path."""

    if not condensate_enabled:
        raise ValueError("condensate_enabled must be true for the condensate standard gate.")
    if route != HEAD_ROUTE_STANDARD:
        raise ValueError(f"Unsupported condensate route '{route}'. Expected '{HEAD_ROUTE_STANDARD}'.")
    tier = classify_head_route_acceptance_tier(
        explicit_opt_in=True,
        case_id=case_id,
        family=family,
        selected_route=selected_route,
        metric_status=metric_status,
    )
    if tier.acceptance_tier == TIER_1:
        status = CONVERGED
        warnings: tuple[str, ...] = ()
        reason = "The row has tight residual components and is standard-path converged."
    elif tier.acceptance_tier == TIER_2:
        status = CONVERGED_WITH_CAVEAT
        warnings = (
            "The row uses a budget-tradeoff accepted path.",
            "Keep budget residual diagnostics in the production result metadata.",
        )
        reason = "The row reaches the HEAD route but carries a budget-tradeoff caveat."
    elif tier.acceptance_tier == TIER_3:
        status = CONVERGED_WITH_CAVEAT
        warnings = (
            "The row carries a raw-gas residual frame caveat.",
            "Keep raw-gas and amount-weighted gas diagnostics in the production result metadata.",
        )
        reason = "The row reaches the HEAD route but carries a raw-gas caveat."
    else:
        status = NOT_CONVERGED
        warnings = ("The row has an unrecognized HEAD route acceptance tier.",)
        reason = "The row cannot be accepted by the HEAD route standard gate."
    return HeadRouteStandardGateRow(
        report_schema="exogibbs_head_route_standard_gate_row_v1",
        condensate_enabled=True,
        route=route,
        gas_only_default_path_protected=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(case_id),
        family=str(family),
        selected_route=str(selected_route),
        metric_status=str(metric_status),
        acceptance_tier=tier.acceptance_tier,
        standard_path_status=status,
        warning_messages=warnings,
        result_metadata_required=(
            "selected_route",
            "acceptance_tier",
            "budget_residual",
            "amount_weighted_gas_residual",
            "raw_gas_residual",
            "complementarity_residual",
            "final_barrier_reached",
        ),
        reason=reason,
    )


def build_head_route_standard_gate(
    *,
    condensate_enabled: bool,
    rows: Sequence[Mapping[str, Any]],
    route: str = HEAD_ROUTE_STANDARD,
) -> HeadRouteStandardGateReport:
    """Build the standard-path gate for HEAD route rows."""

    if not condensate_enabled:
        raise ValueError("condensate_enabled must be true for the condensate standard gate.")
    standard_rows = tuple(
        classify_head_route_standard_gate_row(
            condensate_enabled=True,
            route=route,
            case_id=str(row["case_id"]),
            family=str(row["family"]),
            selected_route=str(row["selected_route"]),
            metric_status=str(row["metric_status"]),
        )
        for row in rows
    )
    return HeadRouteStandardGateReport(
        report_schema="exogibbs_head_route_standard_gate_v1",
        condensate_enabled=True,
        route=route,
        gas_only_default_path_protected=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        row_count=len(standard_rows),
        converged_count=sum(1 for row in standard_rows if row.standard_path_status == CONVERGED),
        converged_with_caveat_count=sum(
            1 for row in standard_rows if row.standard_path_status == CONVERGED_WITH_CAVEAT
        ),
        not_converged_count=sum(1 for row in standard_rows if row.standard_path_status == NOT_CONVERGED),
        unsupported_input_count=sum(
            1 for row in standard_rows if row.standard_path_status == UNSUPPORTED_INPUT
        ),
        standard_path_rows=standard_rows,
    )


__all__ = (
    "BUDGET_TRADEOFF_STATUS",
    "CONVERGED",
    "CONVERGED_WITH_CAVEAT",
    "HEAD_ROUTE_STANDARD",
    "NOT_CONVERGED",
    "RAW_GAS_CAVEAT_STATUS",
    "TIGHT_RESIDUAL_STATUS",
    "TIER_1",
    "TIER_2",
    "TIER_3",
    "UNSUPPORTED_INPUT",
    "HeadRouteStandardGateReport",
    "HeadRouteStandardGateRow",
    "build_head_route_standard_gate",
    "classify_head_route_standard_gate_row",
)
