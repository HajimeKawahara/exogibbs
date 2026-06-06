"""Tests for FC4-M4089 amount-weighted gas acceptance replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "results" / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay.json"
COMPACT_PATH = ROOT / "results" / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay_compact.json"
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4089_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4089_artifacts_are_default_off_and_explicit() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4089"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4089_amount_weighted_replay_records_progress_or_stop() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["decision"] in {
        "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_CONVERGED_ALL_CASES",
        "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_MOVES_BUT_NOT_CONVERGED",
        "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_NO_PROGRESS",
    }
    assert audit["next_default_target"] == "FC4-M4090"
    assert isinstance(audit["final_dominant_weighted_component_counts"], dict)
    for row in audit["rows"]:
        assert row["iteration_count"] > 0
        assert row["status"] in {
            "weighted_converged_before_trial",
            "weighted_converged_after_trial",
            "no_amount_weighted_accepted_trial",
            "max_iterations_reached",
        }
        assert row["final_dominant_weighted_component"] in {
            "amount_weighted_gas",
            "condensate",
            "budget",
            "complementarity",
            "total_density",
            "unavailable",
        }
        assert row["final_gas_metric_frames"]["amount_weighted_gas_l2"] >= 0.0

