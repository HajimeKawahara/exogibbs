"""Tests for FC4-M4088 gas-stationarity metric-frame decomposition artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "results" / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition.json"
COMPACT_PATH = (
    ROOT / "results" / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition_compact.json"
)
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4088_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4088_artifacts_are_default_off_and_explicit() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4088"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4088_records_gas_metric_frames_for_all_cases() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["decision"] in {
        "GAS_STATIONARITY_STOP_IS_RAW_METRIC_FRAME_MISMATCH",
        "GAS_STATIONARITY_STOP_MIXED_RAW_AND_WEIGHTED_FAILURE",
        "GAS_STATIONARITY_STOP_REMAINS_WEIGHTED_GAS_FAILURE",
    }
    assert audit["next_default_target"] == "FC4-M4089"
    assert set(audit["gas_metric_frames"]) == {
        "raw_gas_l2",
        "amount_weighted_gas_l2",
        "sqrt_amount_weighted_gas_l2",
        "mole_fraction_weighted_gas_l2",
    }
    for row in audit["rows"]:
        assert row["iteration_count"] > 0
        assert row["stop_classification"] in {
            "raw_policy_stopped_but_amount_weighted_policy_has_candidate",
            "raw_and_amount_weighted_policies_both_stopped",
            "converged_before_trial",
            "converged_after_trial",
            "max_iterations_reached",
        }
        stop = row["stop_record"]
        frames = stop["baseline_gas_metric_frames"]
        assert frames["raw_gas_l2"] >= 0.0
        assert frames["amount_weighted_gas_l2"] >= 0.0
        assert frames["sqrt_amount_weighted_gas_l2"] >= 0.0
        assert frames["mole_fraction_weighted_gas_l2"] >= 0.0

