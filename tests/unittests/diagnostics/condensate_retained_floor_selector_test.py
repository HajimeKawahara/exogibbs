"""Tests for guarded retained-floor selector diagnostics."""

from __future__ import annotations

import pytest

from exogibbs.diagnostics.condensate_retained_floor_selector import (
    RetainedFloorCandidate,
    select_guarded_retained_floor_candidate,
)


def candidate(
    *,
    label: str,
    budget: float,
    kkt: float,
    budget_nonworse: bool = True,
    kkt_improved: bool = True,
    floor: float | None = None,
):
    return RetainedFloorCandidate(
        label=label,
        budget_residual=budget,
        kkt_residual=kkt,
        budget_nonworse=budget_nonworse,
        kkt_improved=kkt_improved,
        retained_amount_floor=floor,
    )


def test_guarded_selector_selects_floor_when_budget_safe_and_kkt_lower():
    selected = select_guarded_retained_floor_candidate(
        no_floor_candidate=candidate(label="no_floor", budget=1.0, kkt=10.0),
        floor_candidate=candidate(label="floor", budget=0.9, kkt=9.0, floor=1.0e-14),
    )

    assert selected.diagnostic_only is True
    assert selected.default_off is True
    assert selected.production_behavior_change is False
    assert selected.selected_candidate.label == "floor"
    assert selected.rejected_candidate.label == "no_floor"
    assert selected.selection_reason == "floor_budget_safe_and_kkt_lower_than_no_floor"
    assert selected.fastchem4_trace_values_used is False


def test_guarded_selector_rejects_floor_when_kkt_is_worse():
    selected = select_guarded_retained_floor_candidate(
        no_floor_candidate=candidate(label="no_floor", budget=1.0, kkt=10.0),
        floor_candidate=candidate(label="floor", budget=0.9, kkt=11.0, floor=1.0e-14),
    )

    assert selected.selected_candidate.label == "no_floor"
    assert selected.selection_reason == "no_floor_preserves_better_kkt_or_floor_not_safe"


def test_guarded_selector_uses_budget_safe_fallback():
    selected = select_guarded_retained_floor_candidate(
        no_floor_candidate=candidate(
            label="no_floor",
            budget=1.0,
            kkt=10.0,
            budget_nonworse=False,
        ),
        floor_candidate=candidate(label="floor", budget=0.9, kkt=11.0, floor=1.0e-14),
    )

    assert selected.selected_candidate.label == "floor"
    assert selected.selection_reason == "fallback_to_budget_safe_floor_candidate"


def test_guarded_selector_rejects_invalid_candidate_metrics():
    with pytest.raises(ValueError, match="finite and non-negative"):
        select_guarded_retained_floor_candidate(
            no_floor_candidate=candidate(label="no_floor", budget=-1.0, kkt=10.0),
            floor_candidate=candidate(label="floor", budget=0.9, kkt=9.0, floor=1.0e-14),
        )
