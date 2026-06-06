"""Tests for FC4-M3986 to M3995 stationarity frame artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "examples"
    / "comparisons"
    / "fastchem4_milestone3986_to_3995_stationarity_frame_audit.py"
)
AUDIT = ROOT / "results" / "fastchem4_milestone3991_stationarity_frame_audit.json"
CLASSIFICATION = ROOT / "results" / "fastchem4_milestone3995_stationarity_frame_classification.json"
COMPACT = ROOT / "results" / "fastchem4_milestone3986_to_3995_stationarity_frame_compact.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stationarity_frame_script_runs() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)

    assert AUDIT.exists()
    assert CLASSIFICATION.exists()
    assert COMPACT.exists()


def test_stationarity_frame_records_potential_and_log_frames() -> None:
    audit = _load(AUDIT)

    assert audit["milestone"] == "FC4-M3991"
    assert audit["case_count"] == 4
    for row in audit["rows"]:
        assert row["gas_potential_frame_max_abs"] >= 0.0
        assert row["gas_log_variable_frame_max_abs"] >= 0.0
        assert row["log_variable_frame_kkt_residual"] >= 0.0
        assert len(row["top_gas_potential_frame_rows"]) > 0
        assert len(row["top_gas_log_variable_frame_rows"]) > 0
        assert "species" in row["top_gas_potential_frame_rows"][0]


def test_stationarity_frame_classification_and_guardrails() -> None:
    classification = _load(CLASSIFICATION)
    compact = _load(COMPACT)

    assert classification["milestone"] == "FC4-M3995"
    assert classification["next_default_target"] == "FC4-M3996"
    assert classification["classification"] in {
        "gas_stationarity_floor_was_potential_frame_diagnostic_artifact",
        "gas_stationarity_frame_mismatch_not_fully_explanatory",
    }
    assert compact["fix_summary"]["log_variable_frame_kkt_added"] is True
    assert compact["no_fastchem4_files_modified"] is True
    assert compact["no_production_behavior_change"] is True
    assert compact["no_production_return_signature_change"] is True
    assert compact["no_preset_default_wiring"] is True
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert compact["english_only_file_guard"]["english_only_file_guard_passed"] is True
