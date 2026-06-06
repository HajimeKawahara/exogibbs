"""Tests for FC4-M3821 to M3900 outer recompute tradeoff artifacts."""

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
    / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_and_itercap.py"
)
TRADEOFF = ROOT / "results" / "fastchem4_milestone3821_outer_recompute_tradeoff_study.json"
ITERCAP = ROOT / "results" / "fastchem4_milestone3861_outer_recompute_itercap_replay.json"
CLASSIFICATION = (
    ROOT / "results" / "fastchem4_milestone3900_outer_recompute_next_process_classification.json"
)
COMPACT = ROOT / "results" / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_compact.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_outer_recompute_tradeoff_script_runs() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)

    assert TRADEOFF.exists()
    assert ITERCAP.exists()
    assert CLASSIFICATION.exists()
    assert COMPACT.exists()


def test_tradeoff_study_separates_budget_inactive_from_kkt() -> None:
    tradeoff = _load(TRADEOFF)

    assert tradeoff["milestone"] == "FC4-M3821"
    assert tradeoff["case_count"] == 4
    assert tradeoff["budget_floor_reached_count"] == tradeoff["case_count"]
    assert tradeoff["inactive_zero_count"] == tradeoff["case_count"]
    assert 0 < tradeoff["kkt_worsening_count"] < tradeoff["case_count"]
    for row in tradeoff["rows"]:
        assert row["final_budget_floor_reached"] is True
        assert row["final_inactive_zero"] is True
        assert row["tradeoff_class"] in {
            "budget_inactive_and_kkt_improve",
            "budget_inactive_improve_but_kkt_worsens",
        }


def test_itercap_replay_is_safe_and_default_off() -> None:
    itercap = _load(ITERCAP)

    assert itercap["milestone"] == "FC4-M3861"
    assert itercap["case_count"] == 4
    assert itercap["max_iter"] == 40
    assert itercap["finite_solver_input_count"] == itercap["case_count"]
    assert itercap["normal_default_path_unchanged_count"] == itercap["case_count"]
    assert itercap["production_behavior_change"] is False
    assert itercap["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    for row in itercap["rows"]:
        assert row["solver_called"] is True
        assert row["finite_solver_inputs"] is True
        assert row["normal_default_path_unchanged"] is True


def test_tradeoff_compact_guardrails() -> None:
    classification = _load(CLASSIFICATION)
    compact = _load(COMPACT)

    assert classification["milestone"] == "FC4-M3900"
    assert classification["next_default_target"] == "FC4-M3901"
    assert compact["milestone"] == "FC4-M3821_TO_M3900"
    assert compact["no_fastchem4_files_modified"] is True
    assert compact["no_production_behavior_change"] is True
    assert compact["no_production_return_signature_change"] is True
    assert compact["no_preset_default_wiring"] is True
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert compact["english_only_file_guard"]["english_only_file_guard_passed"] is True
