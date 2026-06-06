"""Tests for FC4-M4087 algorithm-v1.1 multi-step convergence probe artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "results" / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe.json"
COMPACT_PATH = (
    ROOT / "results" / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe_compact.json"
)
GUARD_PATH = ROOT / "results" / "fastchem4_milestone4087_english_only_guard.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_m4087_artifacts_are_default_off_and_explicit() -> None:
    audit = _load_json(AUDIT_PATH)
    compact = _load_json(COMPACT_PATH)
    guard = _load_json(GUARD_PATH)

    assert audit["milestone"] == "FC4-M4087"
    assert audit["diagnostic_only"] is True
    assert audit["default_off"] is True
    assert audit["explicit_opt_in"] is True
    assert audit["production_behavior_change"] is False
    assert audit["production_return_signature_change"] is False
    assert audit["preset_default_wiring_change"] is False
    assert audit["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert compact["no_trace_values_as_constructor_inputs"] is True
    assert guard["english_only_file_guard_passed"] is True


def test_m4087_multistep_moves_but_does_not_converge() -> None:
    audit = _load_json(AUDIT_PATH)

    assert audit["case_count"] == 4
    assert audit["accepted_any_case_count"] == audit["case_count"]
    assert audit["converged_case_count"] == 0
    assert audit["decision"] == "ALGORITHM_V11_THERMO_VALID_MULTISTEP_MOVES_BUT_NOT_CONVERGED"
    assert audit["next_default_action"] == "decompose_algorithm_v11_multistep_nonconvergence"
    for row in audit["rows"]:
        assert row["accepted_iteration_count"] > 0
        assert row["converged"] is False
        assert row["status"] == "no_accepted_multistep_trial"
        assert row["final_components"]["budget"] <= max(
            row["initial_components"]["budget"],
            1.0e-8,
        )

