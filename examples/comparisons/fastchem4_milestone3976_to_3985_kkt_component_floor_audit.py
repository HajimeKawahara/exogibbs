"""FC4-M3976 to M3985 KKT component floor audit."""

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
COMPONENT_REPLAY_PATH = RESULTS / "fastchem4_milestone3976_component_floor_replay.json"
COMPONENT_AUDIT_PATH = RESULTS / "fastchem4_milestone3981_kkt_component_floor_audit.json"
CLASSIFICATION_PATH = RESULTS / "fastchem4_milestone3985_kkt_component_floor_classification.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone3976_to_3985_kkt_component_floor_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone3976_to_3985_kkt_component_floor_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone3976_to_3985_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone3976_to_3985_kkt_component_floor_audit.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone3976_to_3985_kkt_component_floor_audit_test.py",
    COMPONENT_REPLAY_PATH,
    COMPONENT_AUDIT_PATH,
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
        "milestone": "FC4-M3976_TO_M3985",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _abs_max(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _top_indices(values: Sequence[float], *, limit: int = 10) -> list[int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return []
    return [int(index) for index in np.argsort(-np.abs(array))[: min(limit, array.size)].tolist()]


def build_component_audit(replay: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in replay["rows"]:
        stage = row["stage1"]
        active_values = stage["active_stationarity_driving"]
        comp_values = stage["complementarity_values"]
        gas_values = stage.get(
            "gas_stationarity_log_scaled_values",
            stage["gas_stationarity_values"],
        )
        gas_potential_values = stage["gas_stationarity_values"]
        active_max = _abs_max(active_values)
        comp_max = _abs_max(comp_values)
        gas_max = _abs_max(gas_values)
        gas_potential_max = _abs_max(gas_potential_values)
        budget = float(stage["post_solver_budget_residual"])
        inactive = float(stage["inactive_positive_count"])
        components = {
            "active_complementarity": comp_max,
            "gas_log_variable_stationarity": gas_max,
            "budget": budget,
            "inactive_positive_count": inactive,
        }
        dominant_component = max(components, key=components.get)
        gas_top = _top_indices(gas_values)
        gas_potential_top = _top_indices(gas_potential_values)
        rows.append(
            {
                "case_id": row["case_id"],
                "support_count": row["support_count"],
                "solver_success": bool(stage["solver_success"]),
                "solver_n_iter": int(stage["solver_n_iter"]),
                "post_solver_kkt_residual_diagnostic": float(
                    stage["post_solver_kkt_residual_diagnostic"]
                ),
                "component_maxima": components,
                "active_condensate_potential_driving_max_abs": active_max,
                "gas_potential_frame_max_abs": gas_potential_max,
                "dominant_component": dominant_component,
                "dominant_component_to_kkt_ratio": components[dominant_component]
                / max(
                    float(
                        stage.get(
                            "post_solver_kkt_residual_log_variable_diagnostic",
                            stage["post_solver_kkt_residual_diagnostic"],
                        )
                    ),
                    1.0e-300,
                ),
                "top_active_stationarity_rows": list(
                    zip(
                        stage["active_stationarity_abs_top_names"],
                        stage["active_stationarity_abs_top_values"],
                    )
                )[:8],
                "top_complementarity_rows": list(
                    zip(
                        stage["complementarity_abs_top_names"],
                        stage["complementarity_abs_top_values"],
                    )
                )[:8],
                "top_gas_log_variable_stationarity_indices": gas_top[:8],
                "top_gas_log_variable_stationarity_abs_values": [
                    float(abs(np.asarray(gas_values, dtype=np.float64)[index]))
                    for index in gas_top[:8]
                ],
                "top_gas_potential_frame_indices": gas_potential_top[:8],
                "top_gas_potential_frame_abs_values": [
                    float(abs(np.asarray(gas_potential_values, dtype=np.float64)[index]))
                    for index in gas_potential_top[:8]
                ],
            }
        )
    return {
        "milestone": "FC4-M3981",
        "component_floor_schema": "exogibbs_kkt_component_floor_audit_v1",
        "case_count": len(rows),
        "dominant_component_counts": {
            name: sum(row["dominant_component"] == name for row in rows)
            for name in (
                "active_complementarity",
                "gas_log_variable_stationarity",
                "budget",
                "inactive_positive_count",
            )
        },
        "rows": rows,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def build_classification(audit: dict[str, Any]) -> dict[str, Any]:
    counts = audit["dominant_component_counts"]
    if counts["gas_log_variable_stationarity"] == audit["case_count"]:
        classification = "gas_log_variable_stationarity_rows_explain_kkt_floor"
        next_action = "map_top_gas_log_variable_stationarity_rows_to_species"
    elif counts["active_complementarity"] > 0:
        classification = "complementarity_rows_explain_some_kkt_floor"
        next_action = "design_activity_correction_or_barrier_centering_stage"
    else:
        classification = "kkt_floor_not_explained_by_exported_components"
        next_action = "audit_unexported_solver_residual_components"
    return {
        "milestone": "FC4-M3985",
        "classification_schema": "exogibbs_kkt_component_floor_classification_v1",
        "classification": classification,
        "case_count": audit["case_count"],
        "dominant_component_counts": counts,
        "next_default_target": "FC4-M3986",
        "next_default_action": next_action,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _update_semantic_ledger(classification: dict[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger["legacy_fastchem3_cond_status"] = "frozen_after_M401_stoploss"
    ledger["fastchem4_current_milestone"] = "FC4-M3976_TO_M3985"
    ledger["fastchem4_kkt_component_floor_status"] = classification["classification"]
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
        "milestone": "FC4-M3976_TO_M3985",
        "campaign_type": "kkt_component_floor_audit",
        "git_status_start": _git_status(),
        "fastchem4_clone_status_start": _git_status("-C", "FastChem4"),
        "component_floor_summary": {
            "case_count": audit["case_count"],
            "dominant_component_counts": audit["dominant_component_counts"],
        },
        "classification_summary": {
            "classification": classification["classification"],
            "next_default_target": classification["next_default_target"],
            "next_default_action": classification["next_default_action"],
        },
        "english_only_file_guard": english_guard,
        "no_fastchem4_files_modified": _git_status("-C", "FastChem4") == ["?? fastchem4_paper.pdf"],
        "no_production_behavior_change": True,
        "no_production_return_signature_change": True,
        "no_preset_default_wiring": True,
        "no_trace_values_as_constructor_inputs": True,
        "git_status_end": _git_status(),
        "fastchem4_clone_status_end": _git_status("-C", "FastChem4"),
    }


def _write_markdown(compact: dict[str, Any]) -> None:
    summary = compact["component_floor_summary"]
    classification = compact["classification_summary"]
    COMPACT_MD_PATH.write_text(
        "\n".join(
            [
                "# FC4-M3976 to M3985 KKT component floor audit",
                "",
                "## Summary",
                "",
                f"- Case count: {summary['case_count']}.",
                f"- Dominant component counts: `{summary['dominant_component_counts']}`.",
                f"- Classification: `{classification['classification']}`.",
                f"- Next action: `{classification['next_default_action']}`.",
                "",
                "## Guardrails",
                "",
                "- This remains diagnostic-only and explicit opt-in.",
                "- No production solver behavior changed.",
                "- No production return signature changed.",
                "- No preset/default wiring changed.",
                "- No FastChem4 trace/public/runtime values were used as constructor inputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    replay = build_persisted_replay()
    _write_json(COMPONENT_REPLAY_PATH, replay)
    audit = build_component_audit(replay)
    _write_json(COMPONENT_AUDIT_PATH, audit)
    classification = build_classification(audit)
    _write_json(CLASSIFICATION_PATH, classification)
    _update_semantic_ledger(classification)
    english_guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, english_guard)
    compact = build_compact(audit, classification, english_guard)
    _write_json(COMPACT_PATH, compact)
    _write_markdown(compact)


if __name__ == "__main__":
    main()
