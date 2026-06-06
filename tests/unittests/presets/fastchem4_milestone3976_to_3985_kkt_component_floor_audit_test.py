"""Tests for FC4-M3976 to M3985 KKT component floor artifacts."""

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
    / "fastchem4_milestone3976_to_3985_kkt_component_floor_audit.py"
)
REPLAY = ROOT / "results" / "fastchem4_milestone3976_component_floor_replay.json"
AUDIT = ROOT / "results" / "fastchem4_milestone3981_kkt_component_floor_audit.json"
CLASSIFICATION = ROOT / "results" / "fastchem4_milestone3985_kkt_component_floor_classification.json"
COMPACT = ROOT / "results" / "fastchem4_milestone3976_to_3985_kkt_component_floor_compact.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_kkt_component_floor_script_runs() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)

    assert REPLAY.exists()
    assert AUDIT.exists()
    assert CLASSIFICATION.exists()
    assert COMPACT.exists()


def test_kkt_component_floor_records_dominant_component() -> None:
    audit = _load(AUDIT)

    assert audit["milestone"] == "FC4-M3981"
    assert audit["case_count"] == 4
    assert sum(audit["dominant_component_counts"].values()) == audit["case_count"]
    for row in audit["rows"]:
        maxima = row["component_maxima"]
        assert row["dominant_component"] in maxima
        assert maxima[row["dominant_component"]] >= 0.0
        assert len(row["top_active_stationarity_rows"]) > 0
        assert len(row["top_complementarity_rows"]) > 0
        assert len(row["top_gas_log_variable_stationarity_indices"]) > 0
        assert len(row["top_gas_potential_frame_indices"]) > 0
        assert row["active_condensate_potential_driving_max_abs"] >= 0.0


def test_kkt_component_floor_classification_and_guardrails() -> None:
    classification = _load(CLASSIFICATION)
    compact = _load(COMPACT)

    assert classification["milestone"] == "FC4-M3985"
    assert classification["next_default_target"] == "FC4-M3986"
    assert classification["classification"] in {
        "gas_stationarity_rows_explain_kkt_floor",
        "gas_log_variable_stationarity_rows_explain_kkt_floor",
        "complementarity_rows_explain_some_kkt_floor",
        "kkt_floor_not_explained_by_exported_components",
    }
    assert compact["milestone"] == "FC4-M3976_TO_M3985"
    assert compact["no_fastchem4_files_modified"] is True
    assert compact["no_production_behavior_change"] is True
    assert compact["no_production_return_signature_change"] is True
    assert compact["no_preset_default_wiring"] is True
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert compact["english_only_file_guard"]["english_only_file_guard_passed"] is True
