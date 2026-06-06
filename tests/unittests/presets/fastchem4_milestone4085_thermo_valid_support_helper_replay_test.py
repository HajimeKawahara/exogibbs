"""Tests for FC4-M4085 thermo-valid support helper replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "results" / "fastchem4_milestone4085_thermo_valid_support_helper_replay.json"
COMPACT_PATH = ROOT / "results" / "fastchem4_milestone4085_thermo_valid_support_helper_replay_compact.json"
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4085_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4085_artifacts_are_default_off_and_validated() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4085"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4085_helper_replay_reproduces_m4084_rescue() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["accepted_case_count"] == audit["case_count"]
    assert audit["decision"] == "THERMO_VALID_SUPPORT_HELPER_REPRODUCES_M4084_RESCUE"
    assert audit["helper_module_path"] == (
        "src/exogibbs/diagnostics/condensate_thermo_valid_support.py"
    )
    for row in audit["rows"]:
        assert row["accepted_by_m4085_policy"] is True
        assert row["filter_report"]["removed_support_count"] > 0
        assert row["filter_report"]["filtered_support_count"] < row["filter_report"][
            "original_support_count"
        ]
        assert row["selected_candidate"]["relative_component_delta"] <= -1.0e-8

