"""Tests for FC4-M4090 amount-weighted remaining residual decomposition."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = (
    ROOT / "results" / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition.json"
)
COMPACT_PATH = (
    ROOT
    / "results"
    / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition_compact.json"
)
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4090_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4090_artifacts_are_default_off_and_explicit() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4090"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4090_remaining_residual_classes_are_recorded() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["decision"] in {
        "AMOUNT_WEIGHTED_REPLAY_NEEDS_STATE_CAP_AND_REMAINING_COMPONENT_SPLIT",
        "AMOUNT_WEIGHTED_REPLAY_REMAINING_RESIDUAL_CLASSIFIED",
    }
    assert audit["next_default_action"] == "test_amount_weighted_policy_with_state_and_step_caps"
    assert isinstance(audit["remaining_residual_class_counts"], dict)
    for row in audit["rows"]:
        assert row["remaining_residual_class"] in {
            "state_overflow_risk_requires_step_cap",
            "trace_species_raw_gas_large_but_weighted_floor_small",
            "condensate_or_barrier_residual_still_large",
            "amount_weighted_gas_floor_dominates",
            "mixed_small_residual_floor",
        }
        assert "max_abs_q" in row["final_state_extremes"]
        assert row["final_components"]["budget"] >= 0.0
        assert row["final_gas_metric_frames"]["amount_weighted_gas_l2"] >= 0.0

