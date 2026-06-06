"""Tests for guarded retained-floor policy gate diagnostics."""

from __future__ import annotations

import pytest

from exogibbs.diagnostics.condensate_guarded_retained_floor_policy import (
    GuardedRetainedFloorPolicyConfig,
    build_guarded_retained_floor_policy_gate_report,
    validate_guarded_retained_floor_policy_config,
)


def config():
    return GuardedRetainedFloorPolicyConfig(
        explicit_opt_in=True,
        direct_solve_threshold=30.0,
        retained_amount_update_factor=3.0,
        retained_amount_floor=1.0e-14,
        budget_tolerance=1.0e-8,
        require_kkt_improvement=True,
    )


def test_policy_config_requires_explicit_opt_in():
    bad = GuardedRetainedFloorPolicyConfig(
        explicit_opt_in=False,
        direct_solve_threshold=30.0,
        retained_amount_update_factor=3.0,
        retained_amount_floor=1.0e-14,
        budget_tolerance=1.0e-8,
        require_kkt_improvement=True,
    )

    with pytest.raises(ValueError, match="explicit_opt_in"):
        validate_guarded_retained_floor_policy_config(bad)


def test_policy_gate_passes_when_all_selected_cases_are_safe_and_improved():
    report = build_guarded_retained_floor_policy_gate_report(
        config=config(),
        selected_budget_nonworse=(True, True, True, True),
        selected_kkt_improved=(True, True, True, True),
        finite_candidate_pair=(True, True, True, True),
        selected_candidate_labels=("floor_1e-14", "floor_1e-14", "no_floor", "floor_1e-14"),
    )

    assert report.gate_passed is True
    assert report.case_count == 4
    assert report.budget_safe_count == 4
    assert report.kkt_improved_count == 4
    assert report.floor_selected_count == 3
    assert report.failed_reasons == ()
    assert report.production_behavior_change is False


def test_policy_gate_fails_when_kkt_improvement_is_required_but_missing():
    report = build_guarded_retained_floor_policy_gate_report(
        config=config(),
        selected_budget_nonworse=(True, True),
        selected_kkt_improved=(True, False),
        finite_candidate_pair=(True, True),
        selected_candidate_labels=("floor_1e-14", "no_floor"),
    )

    assert report.gate_passed is False
    assert "not_all_selected_cases_improve_kkt_diagnostic" in report.failed_reasons


def test_policy_gate_rejects_metric_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        build_guarded_retained_floor_policy_gate_report(
            config=config(),
            selected_budget_nonworse=(True,),
            selected_kkt_improved=(True, True),
            finite_candidate_pair=(True,),
            selected_candidate_labels=("no_floor",),
        )
