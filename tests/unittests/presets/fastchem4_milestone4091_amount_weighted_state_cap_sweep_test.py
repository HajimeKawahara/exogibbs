"""Tests for FC4-M4091 amount-weighted state cap sweep artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "results" / "fastchem4_milestone4091_amount_weighted_state_cap_sweep.json"
COMPACT_PATH = ROOT / "results" / "fastchem4_milestone4091_amount_weighted_state_cap_sweep_compact.json"
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4091_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4091_artifacts_are_default_off_and_explicit() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4091"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4091_policy_summaries_are_recorded() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["policy_count"] == 4
    assert audit["decision"] in {
        "STATE_CAP_SWEEP_FOUND_CONVERGED_POLICY",
        "STATE_CAP_SWEEP_REDUCES_STATE_EXTREMES_BUT_NOT_CONVERGED",
    }
    assert audit["best_policy"]["policy_name"].startswith("delta_q_cap_")
    assert len(audit["policy_summaries"]) == audit["policy_count"]
    assert len(audit["rows"]) == audit["case_count"] * audit["policy_count"]
    for row in audit["rows"]:
        assert row["max_abs_q"] < 1.0e4
        assert row["final_components"]["budget"] >= 0.0
        assert row["final_gas_metric_frames"]["amount_weighted_gas_l2"] >= 0.0
