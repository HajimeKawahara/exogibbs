"""FC4-M3986 to M3995 gas stationarity frame audit."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.comparisons.fastchem4_milestone3321_to_3340_dynamic_budget_priority_guard_envelope import (  # noqa: E402
    SEMANTIC_LEDGER,
    _git_status,
    _load_json,
    _write_json,
)
from examples.comparisons.fastchem4_milestone3901_to_3960_persisted_support_condensed_stage import (  # noqa: E402
    build_persisted_replay,
)


RESULTS = ROOT / "results"
REPLAY_PATH = RESULTS / "fastchem4_milestone3986_stationarity_frame_replay.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone3991_stationarity_frame_audit.json"
CLASSIFICATION_PATH = RESULTS / "fastchem4_milestone3995_stationarity_frame_classification.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone3986_to_3995_stationarity_frame_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone3986_to_3995_stationarity_frame_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone3986_to_3995_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone3986_to_3995_stationarity_frame_audit.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone3986_to_3995_stationarity_frame_audit_test.py",
    REPLAY_PATH,
    AUDIT_PATH,
    CLASSIFICATION_PATH,
    COMPACT_PATH,
    COMPACT_MD_PATH,
    ENGLISH_GUARD_PATH,
    SEMANTIC_LEDGER,
]


def _english_only_guard(paths: Sequence[Path]) -> dict[str, Any]:
    scanned: list[str] = []
    violations: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        scanned.append(str(path.relative_to(ROOT)))
        if JAPANESE_OR_CJK_RE.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(ROOT)))
    return {
        "milestone": "FC4-M3986_TO_M3995",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _top(values: Sequence[float], names: Sequence[str], *, limit: int = 8) -> list[dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return []
    order = np.argsort(-np.abs(array))[: min(limit, array.size)]
    return [
        {
            "index": int(index),
            "species": str(names[int(index)]) if int(index) < len(names) else str(int(index)),
            "value": float(array[int(index)]),
            "abs_value": float(abs(array[int(index)])),
        }
        for index in order.tolist()
    ]


def _max_abs(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def build_frame_audit(replay: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    corrected_counts = {
        "gas_log_variable_stationarity": 0,
        "active_complementarity": 0,
        "budget": 0,
        "inactive_positive_count": 0,
    }
    for row in replay["rows"]:
        stage = row["stage1"]
        gas_names = stage.get("gas_stationarity_abs_top_names")
        all_gas_names = stage.get("gas_species_order", ())
        if not all_gas_names:
            all_gas_names = [str(index) for index in range(len(stage["gas_stationarity_values"]))]
        potential_values = stage["gas_stationarity_values"]
        log_values = stage["gas_stationarity_log_scaled_values"]
        potential_max = _max_abs(potential_values)
        log_max = _max_abs(log_values)
        complementarity_max = _max_abs(stage["complementarity_values"])
        budget = float(stage["post_solver_budget_residual"])
        inactive = float(stage["inactive_positive_count"])
        corrected_components = {
            "gas_log_variable_stationarity": log_max,
            "active_complementarity": complementarity_max,
            "budget": budget,
            "inactive_positive_count": inactive,
        }
        corrected_dominant = max(corrected_components, key=corrected_components.get)
        corrected_counts[corrected_dominant] += 1
        top_potential = _top(potential_values, all_gas_names)
        top_log = _top(log_values, all_gas_names)
        top_potential_log_scaled_values = [
            {
                "index": entry["index"],
                "species": entry["species"],
                "potential_abs_value": entry["abs_value"],
                "log_scaled_abs_value": float(abs(np.asarray(log_values)[entry["index"]])),
            }
            for entry in top_potential
        ]
        rows.append(
            {
                "case_id": row["case_id"],
                "support_count": row["support_count"],
                "solver_success": bool(stage["solver_success"]),
                "solver_n_iter": int(stage["solver_n_iter"]),
                "potential_frame_kkt_residual": float(
                    stage["post_solver_kkt_residual_diagnostic"]
                ),
                "log_variable_frame_kkt_residual": float(
                    stage["post_solver_kkt_residual_log_variable_diagnostic"]
                ),
                "gas_potential_frame_max_abs": potential_max,
                "gas_log_variable_frame_max_abs": log_max,
                "gas_log_to_potential_max_ratio": log_max / max(potential_max, 1.0e-300),
                "top_gas_potential_frame_rows": top_potential,
                "top_gas_log_variable_frame_rows": top_log,
                "top_potential_rows_after_log_scaling": top_potential_log_scaled_values,
                "corrected_component_maxima": corrected_components,
                "corrected_dominant_component": corrected_dominant,
                "legacy_top_gas_names": list(gas_names or ()),
            }
        )
    return {
        "milestone": "FC4-M3991",
        "audit_schema": "exogibbs_stationarity_frame_audit_v1",
        "case_count": len(rows),
        "rows": rows,
        "corrected_dominant_component_counts": corrected_counts,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def build_classification(audit: dict[str, Any]) -> dict[str, Any]:
    potential_large_log_small = 0
    for row in audit["rows"]:
        if (
            row["gas_potential_frame_max_abs"] > 1.0
            and row["gas_log_variable_frame_max_abs"] < 1.0e-6
        ):
            potential_large_log_small += 1
    if potential_large_log_small == audit["case_count"]:
        classification = "gas_stationarity_floor_was_potential_frame_diagnostic_artifact"
        next_action = "use_log_variable_frame_kkt_for_solver_stage_diagnostics"
    else:
        classification = "gas_stationarity_frame_mismatch_not_fully_explanatory"
        next_action = "inspect_species_mapped_gas_rows_and_pi_gauge"
    return {
        "milestone": "FC4-M3995",
        "classification_schema": "exogibbs_stationarity_frame_classification_v1",
        "classification": classification,
        "case_count": audit["case_count"],
        "potential_large_log_small_count": potential_large_log_small,
        "corrected_dominant_component_counts": audit["corrected_dominant_component_counts"],
        "next_default_target": "FC4-M3996",
        "next_default_action": next_action,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _update_semantic_ledger(classification: dict[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger["legacy_fastchem3_cond_status"] = "frozen_after_M401_stoploss"
    ledger["fastchem4_current_milestone"] = "FC4-M3986_TO_M3995"
    ledger["fastchem4_stationarity_frame_audit_status"] = classification["classification"]
    ledger["fastchem4_next_default_target"] = classification["next_default_target"]
    ledger["fastchem4_next_default_action"] = classification["next_default_action"]
    ledger["fastchem4_no_trace_values_as_constructor_inputs"] = True
    ledger["fastchem4_no_production_behavior_change"] = True
    ledger["fastchem4_no_preset_default_wiring"] = True
    _write_json(SEMANTIC_LEDGER, ledger)


def build_compact(
    audit: dict[str, Any],
    classification: dict[str, Any],
    english_guard: dict[str, Any],
) -> dict[str, Any]:
    return {
        "milestone": "FC4-M3986_TO_M3995",
        "campaign_type": "stationarity_frame_audit_and_fix",
        "git_status_start": _git_status(),
        "fastchem4_clone_status_start": _git_status("-C", "FastChem4"),
        "audit_summary": {
            "case_count": audit["case_count"],
            "corrected_dominant_component_counts": audit[
                "corrected_dominant_component_counts"
            ],
        },
        "classification_summary": {
            "classification": classification["classification"],
            "next_default_target": classification["next_default_target"],
            "next_default_action": classification["next_default_action"],
        },
        "fix_summary": {
            "legacy_potential_frame_preserved": True,
            "log_variable_frame_kkt_added": True,
            "gas_species_names_added_to_row_diagnostics": True,
        },
        "english_only_file_guard": english_guard,
        "no_fastchem4_files_modified": _git_status("-C", "FastChem4")
        == ["?? fastchem4_paper.pdf"],
        "no_production_behavior_change": True,
        "no_production_return_signature_change": True,
        "no_preset_default_wiring": True,
        "no_trace_values_as_constructor_inputs": True,
        "fastchem4_clone_status_end": _git_status("-C", "FastChem4"),
        "git_status_end": _git_status(),
    }


def _write_md(compact: dict[str, Any]) -> None:
    lines = [
        "# FC4-M3986 to M3995 Stationarity Frame Audit",
        "",
        f"- classification: {compact['classification_summary']['classification']}",
        f"- corrected dominant components: {compact['audit_summary']['corrected_dominant_component_counts']}",
        "- legacy potential-frame residuals were preserved as diagnostics",
        "- log-variable-frame KKT residuals were added for solver-stage diagnostics",
        f"- next target: {compact['classification_summary']['next_default_target']}",
        f"- next action: {compact['classification_summary']['next_default_action']}",
    ]
    COMPACT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    replay = build_persisted_replay()
    _write_json(REPLAY_PATH, replay)
    audit = build_frame_audit(replay)
    classification = build_classification(audit)
    _write_json(AUDIT_PATH, audit)
    _write_json(CLASSIFICATION_PATH, classification)
    _update_semantic_ledger(classification)
    english_guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, english_guard)
    compact = build_compact(audit, classification, english_guard)
    _write_json(COMPACT_PATH, compact)
    _write_md(compact)


if __name__ == "__main__":
    main()
