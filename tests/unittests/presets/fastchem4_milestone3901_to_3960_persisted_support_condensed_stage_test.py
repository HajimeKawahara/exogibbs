"""Tests for FC4-M3901 to M3960 persisted support condensed-stage artifacts."""

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
    / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage.py"
)
REPLAY = ROOT / "results" / "fastchem4_milestone3901_persisted_support_condensed_stage.json"
CLASSIFICATION = (
    ROOT / "results" / "fastchem4_milestone3960_persisted_support_next_process_classification.json"
)
COMPACT = (
    ROOT
    / "results"
    / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage_compact.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_persisted_support_condensed_stage_script_runs() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)

    assert REPLAY.exists()
    assert CLASSIFICATION.exists()
    assert COMPACT.exists()


def test_persisted_stage_is_safe_and_records_stage_comparison() -> None:
    replay = _load(REPLAY)

    assert replay["milestone"] == "FC4-M3901"
    assert replay["case_count"] == 4
    assert replay["finite_solver_input_count"] == replay["case_count"]
    assert replay["normal_default_path_unchanged_count"] == replay["case_count"]
    assert 0 < replay["stage1_budget_nonworse_count"] <= replay["case_count"]
    assert 0 < replay["stage1_kkt_nonworse_count"] <= replay["case_count"]
    assert replay["production_behavior_change"] is False
    assert replay["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    for row in replay["rows"]:
        assert row["stage0"]["solver_called"] is True
        assert row["stage1"]["solver_called"] is True
        assert row["stage1"]["finite_solver_inputs"] is True
        assert row["stage1"]["normal_default_path_unchanged"] is True
        assert isinstance(row["stage1_budget_nonworse"], bool)
        assert isinstance(row["stage1_kkt_nonworse"], bool)


def test_persisted_stage_classification_and_guardrails() -> None:
    classification = _load(CLASSIFICATION)
    compact = _load(COMPACT)

    assert classification["milestone"] == "FC4-M3960"
    assert classification["next_default_target"] == "FC4-M3961"
    assert classification["classification"] in {
        "persisted_support_stage_reaches_partial_solver_success",
        "persisted_support_stage_stabilizes_kkt_without_convergence",
        "persisted_support_stage_mixed_kkt_response",
        "persisted_support_stage_not_sufficient",
    }
    assert compact["milestone"] == "FC4-M3901_TO_M3960"
    assert compact["no_fastchem4_files_modified"] is True
    assert compact["no_production_behavior_change"] is True
    assert compact["no_production_return_signature_change"] is True
    assert compact["no_preset_default_wiring"] is True
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert compact["english_only_file_guard"]["english_only_file_guard_passed"] is True
