"""Explicit opt-in HEAD route acceptance-tier helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


TIER_1 = "tier_1_tight_residual_production_adjacent_candidate"
TIER_2 = "tier_2_budget_tradeoff_experimental_only"
TIER_3 = "tier_3_raw_gas_caveat_diagnostic_only"

TIGHT_RESIDUAL_STATUS = "tight_residual_components"
BUDGET_TRADEOFF_STATUS = "accepted_budget_tradeoff_components"
RAW_GAS_CAVEAT_STATUS = "barrier_centered_with_raw_gas_caveat"


@dataclass(frozen=True)
class HeadRouteAcceptanceTierReport:
    """Acceptance-tier report for one HEAD route row."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    gas_only_default_path_protected: bool
    condensate_legacy_default_preserved_as_acceptance_target: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    family: str
    selected_route: str
    metric_status: str
    acceptance_tier: str
    production_adjacent_opt_in_candidate: bool
    requires_further_diagnostic_before_production_gate: bool
    allowed_next_step: str
    forbidden_next_step: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeadRouteOptInHardeningScopeReport:
    """Scope report for tier-1 production-adjacent opt-in hardening."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    gas_only_default_path_protected: bool
    condensate_legacy_default_preserved_as_acceptance_target: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    row_count: int
    tier_1_candidate_count: int
    caveat_row_count: int
    tier_counts: Mapping[str, int]
    tier_1_rows: Sequence[HeadRouteAcceptanceTierReport]
    caveat_rows: Sequence[HeadRouteAcceptanceTierReport]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier_1_rows"] = [row.as_dict() for row in self.tier_1_rows]
        payload["caveat_rows"] = [row.as_dict() for row in self.caveat_rows]
        return payload


def classify_head_route_acceptance_tier(
    *,
    explicit_opt_in: bool,
    case_id: str,
    family: str,
    selected_route: str,
    metric_status: str,
) -> HeadRouteAcceptanceTierReport:
    """Classify one HEAD route row for explicit opt-in hardening."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for HEAD route acceptance-tier classification.")
    if not str(case_id):
        raise ValueError("case_id must not be empty.")
    if not str(family):
        raise ValueError("family must not be empty.")
    if not str(selected_route):
        raise ValueError("selected_route must not be empty.")
    status = str(metric_status)
    if status == TIGHT_RESIDUAL_STATUS:
        tier = TIER_1
        opt_in_candidate = True
        needs_diagnostic = False
        allowed_next_step = "production_adjacent_opt_in_hardening"
        forbidden_next_step = "default_on_production_wiring"
        reason = "The row has tight budget and amount-weighted gas residual components."
    elif status == BUDGET_TRADEOFF_STATUS:
        tier = TIER_2
        opt_in_candidate = False
        needs_diagnostic = True
        allowed_next_step = "explicit_opt_in_experimental_replay_only"
        forbidden_next_step = "production_acceptance_gate"
        reason = "The row is accepted only through a budget-tradeoff path."
    elif status == RAW_GAS_CAVEAT_STATUS:
        tier = TIER_3
        opt_in_candidate = False
        needs_diagnostic = True
        allowed_next_step = "diagnostic_frame_decomposition"
        forbidden_next_step = "production_acceptance_gate"
        reason = "The row has a raw-gas residual caveat despite final-barrier evidence."
    else:
        tier = TIER_3
        opt_in_candidate = False
        needs_diagnostic = True
        allowed_next_step = "diagnostic_frame_decomposition"
        forbidden_next_step = "production_acceptance_gate"
        reason = "The row has an unrecognized or caveat-bearing metric status."
    return HeadRouteAcceptanceTierReport(
        report_schema="exogibbs_head_route_acceptance_tier_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        gas_only_default_path_protected=True,
        condensate_legacy_default_preserved_as_acceptance_target=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(case_id),
        family=str(family),
        selected_route=str(selected_route),
        metric_status=status,
        acceptance_tier=tier,
        production_adjacent_opt_in_candidate=opt_in_candidate,
        requires_further_diagnostic_before_production_gate=needs_diagnostic,
        allowed_next_step=allowed_next_step,
        forbidden_next_step=forbidden_next_step,
        reason=reason,
    )


def build_head_route_opt_in_hardening_scope(
    *,
    explicit_opt_in: bool,
    rows: Sequence[Mapping[str, Any]],
) -> HeadRouteOptInHardeningScopeReport:
    """Build the tier-1 hardening scope without preserving broken condensate defaults."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for HEAD route opt-in hardening scope.")
    reports = tuple(
        classify_head_route_acceptance_tier(
            explicit_opt_in=True,
            case_id=str(row["case_id"]),
            family=str(row["family"]),
            selected_route=str(row["selected_route"]),
            metric_status=str(row["metric_status"]),
        )
        for row in rows
    )
    tier_counts: dict[str, int] = {}
    for report in reports:
        tier_counts[report.acceptance_tier] = tier_counts.get(report.acceptance_tier, 0) + 1
    tier_1_rows = tuple(row for row in reports if row.production_adjacent_opt_in_candidate)
    caveat_rows = tuple(row for row in reports if not row.production_adjacent_opt_in_candidate)
    return HeadRouteOptInHardeningScopeReport(
        report_schema="exogibbs_head_route_opt_in_hardening_scope_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        gas_only_default_path_protected=True,
        condensate_legacy_default_preserved_as_acceptance_target=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        row_count=len(reports),
        tier_1_candidate_count=len(tier_1_rows),
        caveat_row_count=len(caveat_rows),
        tier_counts=tier_counts,
        tier_1_rows=tier_1_rows,
        caveat_rows=caveat_rows,
    )


__all__ = (
    "BUDGET_TRADEOFF_STATUS",
    "RAW_GAS_CAVEAT_STATUS",
    "TIER_1",
    "TIER_2",
    "TIER_3",
    "TIGHT_RESIDUAL_STATUS",
    "HeadRouteAcceptanceTierReport",
    "HeadRouteOptInHardeningScopeReport",
    "build_head_route_opt_in_hardening_scope",
    "classify_head_route_acceptance_tier",
)
