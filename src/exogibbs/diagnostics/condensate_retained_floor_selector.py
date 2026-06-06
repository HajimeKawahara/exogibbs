"""Guarded retained-floor candidate selection diagnostics.

This module selects between a no-floor retained-support candidate and an
absolute-floor retained-support candidate. It uses only caller-provided
diagnostic metrics and does not call FastChem4 or production solvers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetainedFloorCandidate:
    """Candidate metrics for guarded retained-floor selection."""

    label: str
    budget_residual: float
    kkt_residual: float
    budget_nonworse: bool
    kkt_improved: bool
    retained_amount_floor: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedRetainedFloorSelection:
    """Selected retained-floor candidate and guardrail diagnostics."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    selection_schema: str
    selected_candidate: RetainedFloorCandidate
    rejected_candidate: RetainedFloorCandidate
    selection_reason: str
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool
    fastchem4_runtime_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_candidate"] = self.selected_candidate.as_dict()
        payload["rejected_candidate"] = self.rejected_candidate.as_dict()
        return payload


def _validate_candidate(candidate: RetainedFloorCandidate, name: str) -> None:
    if not candidate.label:
        raise ValueError(f"{name}.label must be non-empty.")
    if not np.isfinite(candidate.budget_residual) or candidate.budget_residual < 0.0:
        raise ValueError(f"{name}.budget_residual must be finite and non-negative.")
    if not np.isfinite(candidate.kkt_residual) or candidate.kkt_residual < 0.0:
        raise ValueError(f"{name}.kkt_residual must be finite and non-negative.")
    if candidate.retained_amount_floor is not None and (
        candidate.retained_amount_floor <= 0.0
        or not np.isfinite(candidate.retained_amount_floor)
    ):
        raise ValueError(f"{name}.retained_amount_floor must be positive when provided.")


def select_guarded_retained_floor_candidate(
    *,
    no_floor_candidate: RetainedFloorCandidate,
    floor_candidate: RetainedFloorCandidate,
) -> GuardedRetainedFloorSelection:
    """Select the floor candidate only when it is budget-safe and KKT-lower."""

    _validate_candidate(no_floor_candidate, "no_floor_candidate")
    _validate_candidate(floor_candidate, "floor_candidate")
    if floor_candidate.budget_nonworse and (
        floor_candidate.kkt_residual < no_floor_candidate.kkt_residual
    ):
        selected = floor_candidate
        rejected = no_floor_candidate
        reason = "floor_budget_safe_and_kkt_lower_than_no_floor"
    elif no_floor_candidate.budget_nonworse:
        selected = no_floor_candidate
        rejected = floor_candidate
        reason = "no_floor_preserves_better_kkt_or_floor_not_safe"
    elif floor_candidate.budget_nonworse:
        selected = floor_candidate
        rejected = no_floor_candidate
        reason = "fallback_to_budget_safe_floor_candidate"
    else:
        selected = no_floor_candidate
        rejected = floor_candidate
        reason = "no_budget_safe_candidate_available"
    return GuardedRetainedFloorSelection(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        selection_schema="exogibbs_guarded_retained_floor_selection_v1",
        selected_candidate=selected,
        rejected_candidate=rejected,
        selection_reason=reason,
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
        fastchem4_runtime_values_used_as_constructor_inputs=False,
    )


__all__ = (
    "GuardedRetainedFloorSelection",
    "RetainedFloorCandidate",
    "select_guarded_retained_floor_candidate",
)
