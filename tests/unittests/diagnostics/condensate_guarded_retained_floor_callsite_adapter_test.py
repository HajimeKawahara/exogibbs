"""Tests for guarded retained-floor callsite dry-run adapter."""

from __future__ import annotations

import pytest

from exogibbs.diagnostics.condensate_guarded_retained_floor_callsite_adapter import (
    GuardedRetainedFloorCallsiteCandidatePayload,
    build_guarded_retained_floor_callsite_payload,
)


def payload(label: str, kkt: float, floor: float | None):
    return GuardedRetainedFloorCallsiteCandidatePayload(
        label=label,
        support_indices=(1, 3),
        support_amounts_init=(1.0e-12, 2.0e-12),
        budget_residual=1.0e-9,
        kkt_residual=kkt,
        budget_nonworse=True,
        kkt_improved=True,
        retained_amount_floor=floor,
    )


def test_callsite_adapter_selects_floor_payload_when_guarded_selector_selects_floor():
    report = build_guarded_retained_floor_callsite_payload(
        explicit_opt_in=True,
        no_floor_payload=payload("no_floor", 10.0, None),
        floor_payload=payload("floor_1e-14", 9.0, 1.0e-14),
    )

    assert report.diagnostic_only is True
    assert report.default_off is True
    assert report.production_behavior_change is False
    assert report.selected_candidate_label == "floor_1e-14"
    assert report.support_indices == (1, 3)
    assert report.support_amounts_init == pytest.approx((1.0e-12, 2.0e-12))
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False


def test_callsite_adapter_rejects_missing_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit_opt_in"):
        build_guarded_retained_floor_callsite_payload(
            explicit_opt_in=False,
            no_floor_payload=payload("no_floor", 10.0, None),
            floor_payload=payload("floor_1e-14", 9.0, 1.0e-14),
        )


def test_callsite_adapter_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same length"):
        build_guarded_retained_floor_callsite_payload(
            explicit_opt_in=True,
            no_floor_payload=GuardedRetainedFloorCallsiteCandidatePayload(
                label="no_floor",
                support_indices=(1, 3),
                support_amounts_init=(1.0e-12,),
                budget_residual=1.0e-9,
                kkt_residual=10.0,
                budget_nonworse=True,
                kkt_improved=True,
                retained_amount_floor=None,
            ),
            floor_payload=payload("floor_1e-14", 9.0, 1.0e-14),
        )
