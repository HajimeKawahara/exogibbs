"""Center-primary fallback selection for condensate HEAD route lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CenterPrimaryFallbackCandidate:
    """One center-primary fallback candidate."""

    candidate_name: str
    converged_at_final_barrier: bool
    final_center_ratio: float
    budget_ratio: float
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CenterPrimaryFallbackReport:
    """Selection report for center-primary fallback."""

    report_schema: str
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    primary_centered: bool
    attempted: bool
    accepted: bool
    classification: str
    selected_candidate_name: str | None
    max_budget_ratio: float
    max_center_ratio: float
    candidate_count: int
    candidates: tuple[CenterPrimaryFallbackCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.as_dict() for candidate in self.candidates]
        return payload


def build_center_primary_fallback_candidate(
    *,
    candidate_name: str,
    converged_at_final_barrier: bool,
    final_center_ratio: float,
    budget_ratio: float,
    metadata: Mapping[str, Any] | None = None,
) -> CenterPrimaryFallbackCandidate:
    """Build one explicit center-primary fallback candidate."""

    if not str(candidate_name):
        raise ValueError("candidate_name must not be empty.")
    center = float(final_center_ratio)
    budget = float(budget_ratio)
    if center < 0.0:
        raise ValueError("final_center_ratio must be non-negative.")
    if budget < 0.0:
        raise ValueError("budget_ratio must be non-negative.")
    return CenterPrimaryFallbackCandidate(
        candidate_name=str(candidate_name),
        converged_at_final_barrier=bool(converged_at_final_barrier),
        final_center_ratio=center,
        budget_ratio=budget,
        metadata=dict(metadata or {}),
    )


def select_center_primary_fallback(
    *,
    explicit_opt_in: bool,
    primary_summary: Mapping[str, Any],
    candidates: Sequence[CenterPrimaryFallbackCandidate],
    max_budget_ratio: float = 1.05,
    max_center_ratio: float = 1.0,
) -> CenterPrimaryFallbackReport:
    """Select the first center-primary fallback candidate accepted by guards."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for center-primary fallback.")
    budget_limit = float(max_budget_ratio)
    center_limit = float(max_center_ratio)
    if budget_limit < 0.0:
        raise ValueError("max_budget_ratio must be non-negative.")
    if center_limit < 0.0:
        raise ValueError("max_center_ratio must be non-negative.")
    primary_centered = bool(
        primary_summary.get("converged_at_final_barrier", False)
        or str(primary_summary.get("row_status", "")) == "centered"
    )
    ordered = tuple(candidates)
    if primary_centered:
        accepted: CenterPrimaryFallbackCandidate | None = None
        attempted = False
        classification = "center_primary_budget_guard_not_attempted_primary_centered"
    else:
        attempted = True
        accepted_rows = [
            candidate
            for candidate in ordered
            if candidate.converged_at_final_barrier
            and candidate.final_center_ratio <= center_limit
            and candidate.budget_ratio <= budget_limit
        ]
        accepted = accepted_rows[0] if accepted_rows else None
        classification = (
            "center_primary_budget_guard_accepts_with_budget_tradeoff"
            if accepted is not None
            else "center_primary_budget_guard_rejects_all_candidates"
        )
    return CenterPrimaryFallbackReport(
        report_schema="exogibbs_center_primary_fallback_report_v1",
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        primary_centered=primary_centered,
        attempted=attempted,
        accepted=accepted is not None,
        classification=classification,
        selected_candidate_name=accepted.candidate_name if accepted is not None else None,
        max_budget_ratio=budget_limit,
        max_center_ratio=center_limit,
        candidate_count=len(ordered),
        candidates=ordered,
    )


__all__ = (
    "CenterPrimaryFallbackCandidate",
    "CenterPrimaryFallbackReport",
    "build_center_primary_fallback_candidate",
    "select_center_primary_fallback",
)
