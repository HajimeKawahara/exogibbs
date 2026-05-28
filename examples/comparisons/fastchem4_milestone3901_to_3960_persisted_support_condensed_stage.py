"""FC4-M3901 to M3960 persisted support condensed-stage replay."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.comparisons.fastchem4_milestone106_to_112_public_output_aligned_native_probe_records import (  # noqa: E402
    _cond_hvector,
    _gas_hvector,
    _state_for_record,
)
from examples.comparisons.fastchem4_milestone137_to_144_gas_logdensity_boundary_solver import (  # noqa: E402
    _ideal_density_budget,
    _mass_action_constants,
)
from examples.comparisons.fastchem4_milestone1541_to_1550_empty_support_budget_path_real_callsite_smoke import (  # noqa: E402
    _payload,
)
from examples.comparisons.fastchem4_milestone3321_to_3340_dynamic_budget_priority_guard_envelope import (  # noqa: E402
    SEMANTIC_LEDGER,
    _case_lookup,
    _git_status,
    _load_json,
    _write_json,
)
from examples.comparisons.fastchem4_milestone3761_to_3820_outer_activity_recompute_loop import (  # noqa: E402
    LOOP_PATH as M3781_LOOP_PATH,
    MIN_AMOUNT,
)
from exogibbs.diagnostics.condensate_component_safe_restricted_callsite import (  # noqa: E402
    run_component_safe_restricted_callsite_experiment,
)
from exogibbs.diagnostics.condensate_hardened_gas_logdensity_refresh import (  # noqa: E402
    build_hardened_gas_logdensity_refresh_init,
)


RESULTS = ROOT / "results"
PERSISTED_PATH = RESULTS / "fastchem4_milestone3901_persisted_support_condensed_stage.json"
CLASSIFICATION_PATH = RESULTS / "fastchem4_milestone3960_persisted_support_next_process_classification.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone3901_to_3960_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
STAGE_MAX_ITER = 40
GAS_MAX_ITER = 12

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage_test.py",
    PERSISTED_PATH,
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
        "milestone": "FC4-M3901_TO_M3960",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _support_amounts_from_iteration(iteration: dict[str, Any]) -> tuple[float, ...]:
    amounts = tuple(float(value) for value in iteration["support_amounts_init"])
    return tuple(max(amount, MIN_AMOUNT) for amount in amounts)


def _budget_safe_amounts(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    ac = np.asarray(formula_matrix_cond, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    amounts = np.asarray(support_amounts, dtype=np.float64)
    full = np.zeros(ac.shape[1], dtype=np.float64)
    full[np.asarray(support_indices, dtype=int)] = amounts
    burden = ac @ full
    positive = np.abs(target) > 0.0
    if not np.any(positive):
        return tuple(float(value) for value in amounts), 1.0
    fraction = float(np.max(np.abs(burden[positive]) / np.maximum(np.abs(target[positive]), 1.0e-300)))
    if fraction <= 1.0:
        return tuple(float(value) for value in amounts), 1.0
    scale = 0.999 / fraction
    return tuple(float(max(value * scale, MIN_AMOUNT)) for value in amounts), float(scale)


def _depleted_budget(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
) -> tuple[float, ...]:
    ac = np.asarray(formula_matrix_cond, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    full = np.zeros(ac.shape[1], dtype=np.float64)
    full[np.asarray(support_indices, dtype=int)] = np.asarray(support_amounts, dtype=np.float64)
    depleted = target - ac @ full
    depleted[np.abs(depleted) < 1.0e-300] = 0.0
    return tuple(float(value) for value in depleted)


def _run_stage(
    *,
    case_id: str,
    stage_label: str,
    support_indices: tuple[int, ...],
    support_amounts: tuple[float, ...],
) -> dict[str, Any]:
    case = _case_lookup()[case_id]
    contract = case["contract"]
    record = case["record"]
    safe_amounts, amount_scale = _budget_safe_amounts(
        formula_matrix_cond=contract.formula_matrix_cond,
        element_inventory_target=contract.element_inventory_target,
        support_indices=support_indices,
        support_amounts=support_amounts,
    )
    init, refresh = build_hardened_gas_logdensity_refresh_init(
        explicit_opt_in=True,
        formula_matrix=contract.formula_matrix,
        source_element_budget=_ideal_density_budget(
            contract,
            float(record["temperature"]),
            float(record["pressure"]),
        ),
        native_element_budget=_depleted_budget(
            formula_matrix_cond=contract.formula_matrix_cond,
            element_inventory_target=contract.element_inventory_target,
            support_indices=support_indices,
            support_amounts=safe_amounts,
        ),
        mass_action_constants=_mass_action_constants(contract, float(record["temperature"])),
        support_indices=support_indices,
        ln_mk=tuple(float(np.log(max(value, MIN_AMOUNT))) for value in safe_amounts),
        field_provenance={
            "formula_matrix": "exogibbs_native_static_contract",
            "source_element_budget": "exogibbs_native_ideal_density_budget",
            "native_element_budget": "exogibbs_native_persisted_support_stage",
            "mass_action_constants": "exogibbs_native_thermochemistry",
            "support_indices": "exogibbs_native_persisted_support_stage",
            "ln_mk": "exogibbs_native_persisted_support_stage",
        },
    )
    report = run_component_safe_restricted_callsite_experiment(
        explicit_opt_in=True,
        payload=_payload(case_id, len(support_indices)),
        state=_state_for_record(record, contract),
        formula_matrix=contract.formula_matrix,
        formula_matrix_cond=contract.formula_matrix_cond,
        hvector_func=_gas_hvector(contract),
        hvector_cond_func=_cond_hvector(contract),
        condensate_species_order=contract.condensate_species_order,
        element_order=contract.element_order,
        support_indices=support_indices,
        support_amounts_init=safe_amounts,
        initial_log_state_override=init,
        max_budget_fraction=1.0,
        gas_max_iter=GAS_MAX_ITER,
        max_iter=STAGE_MAX_ITER,
        line_search_selection_policy="first_monotone_with_best_finite_fallback",
        line_search_charge_row_name="e-",
        line_search_charge_weight=1.0,
        field_provenance={
            "formula_matrix": "exogibbs_native_static_contract",
            "formula_matrix_cond": "exogibbs_native_static_contract",
            "element_inventory_target": "exogibbs_native_abundance_contract",
            "hvector_cond": "exogibbs_native_thermochemistry_static_contract",
            "support_indices": "exogibbs_native_persisted_support_stage",
            "support_amounts_init": "exogibbs_native_persisted_support_stage",
            "initial_log_state_override": "exogibbs_native_hardened_gas_logdensity_refresh",
        },
    )
    data = report.as_dict()
    seeded = data["seeded_callsite_report"]
    final_amounts = tuple(float(value) for value in seeded["final_support_amounts"])
    if len(final_amounts) != len(support_indices):
        final_amounts = safe_amounts
    return {
        "stage_label": stage_label,
        "support_count": len(support_indices),
        "amount_scale_applied": amount_scale,
        "gas_refresh_success": bool(refresh.gas_solver_success),
        "solver_called": data["solver_called"],
        "solver_success": data["solver_success"],
        "solver_status": data["solver_status"],
        "solver_n_iter": seeded["solver_n_iter"],
        "solver_final_residual": seeded["solver_final_residual"],
        "post_solver_budget_residual": seeded["post_solver_budget_residual"],
        "post_solver_kkt_residual_diagnostic": seeded["post_solver_kkt_residual_diagnostic"],
        "inactive_positive_count": seeded["inactive_positive_count"],
        "finite_solver_inputs": data["finite_solver_inputs"],
        "normal_default_path_unchanged": data["normal_default_path_unchanged"],
        "production_behavior_change": data["production_behavior_change"],
        "production_return_signature_change": data["production_return_signature_change"],
        "preset_default_wiring_change": data["preset_default_wiring_change"],
        "fastchem4_trace_public_runtime_constructor_inputs_used": data[
            "fastchem4_trace_public_runtime_constructor_inputs_used"
        ],
        "final_support_amounts": list(final_amounts),
    }


def build_persisted_replay() -> dict[str, Any]:
    loop = _load_json(M3781_LOOP_PATH)
    rows: list[dict[str, Any]] = []
    for case_row in loop["rows"]:
        case_id = str(case_row["case_id"])
        final_iteration = case_row["iterations"][-1]
        support_indices = tuple(int(index) for index in final_iteration["support_indices"])
        support_amounts = _support_amounts_from_iteration(final_iteration)
        stage0 = _run_stage(
            case_id=case_id,
            stage_label="stage0_replay_from_outer_recompute_final_support",
            support_indices=support_indices,
            support_amounts=support_amounts,
        )
        stage1 = _run_stage(
            case_id=case_id,
            stage_label="stage1_persisted_final_support_amounts",
            support_indices=support_indices,
            support_amounts=tuple(float(value) for value in stage0["final_support_amounts"]),
        )
        kkt0 = float(stage0["post_solver_kkt_residual_diagnostic"])
        kkt1 = float(stage1["post_solver_kkt_residual_diagnostic"])
        budget0 = float(stage0["post_solver_budget_residual"])
        budget1 = float(stage1["post_solver_budget_residual"])
        rows.append(
            {
                "case_id": case_id,
                "support_count": len(support_indices),
                "stage0": stage0,
                "stage1": stage1,
                "stage1_kkt_ratio_to_stage0": kkt1 / max(kkt0, 1.0e-300),
                "stage1_budget_ratio_to_stage0": budget1 / max(budget0, 1.0e-300),
                "stage1_kkt_nonworse": kkt1 <= kkt0,
                "stage1_budget_nonworse": budget1 <= max(budget0, 1.0e-12),
                "stage1_success": bool(stage1["solver_success"]),
            }
        )
    return {
        "milestone": "FC4-M3901",
        "persisted_stage_schema": "exogibbs_persisted_support_condensed_stage_replay_v1",
        "stage_max_iter": STAGE_MAX_ITER,
        "case_count": len(rows),
        "stage1_success_count": sum(row["stage1_success"] for row in rows),
        "stage1_kkt_nonworse_count": sum(row["stage1_kkt_nonworse"] for row in rows),
        "stage1_budget_nonworse_count": sum(row["stage1_budget_nonworse"] for row in rows),
        "finite_solver_input_count": sum(row["stage1"]["finite_solver_inputs"] for row in rows),
        "normal_default_path_unchanged_count": sum(row["stage1"]["normal_default_path_unchanged"] for row in rows),
        "rows": rows,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def build_classification(replay: dict[str, Any]) -> dict[str, Any]:
    if replay["stage1_success_count"] > 0:
        classification = "persisted_support_stage_reaches_partial_solver_success"
        next_action = "broaden_persisted_support_stage_to_curated_cases"
    elif replay["stage1_kkt_nonworse_count"] == replay["case_count"]:
        classification = "persisted_support_stage_stabilizes_kkt_without_convergence"
        next_action = "add_outer_loop_stage_acceptance_and_repeat_policy"
    elif replay["stage1_kkt_nonworse_count"] > 0:
        classification = "persisted_support_stage_mixed_kkt_response"
        next_action = "decompose_persisted_stage_by_amount_change_and_stationarity_rows"
    else:
        classification = "persisted_support_stage_not_sufficient"
        next_action = "audit_condensed_amount_update_direction_and_activity_correction_coupling"
    return {
        "milestone": "FC4-M3960",
        "classification_schema": "exogibbs_persisted_support_next_process_classification_v1",
        "classification": classification,
        "stage1_success_count": replay["stage1_success_count"],
        "stage1_kkt_nonworse_count": replay["stage1_kkt_nonworse_count"],
        "stage1_budget_nonworse_count": replay["stage1_budget_nonworse_count"],
        "next_default_target": "FC4-M3961",
        "next_default_action": next_action,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _update_semantic_ledger(classification: dict[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger["legacy_fastchem3_cond_status"] = "frozen_after_M401_stoploss"
    ledger["fastchem4_current_milestone"] = "FC4-M3901_TO_M3960"
    ledger["fastchem4_persisted_support_condensed_stage_status"] = classification["classification"]
    ledger["fastchem4_next_default_target"] = classification["next_default_target"]
    ledger["fastchem4_next_default_action"] = classification["next_default_action"]
    ledger["fastchem4_no_trace_values_as_constructor_inputs"] = True
    ledger["fastchem4_no_production_behavior_change"] = True
    ledger["fastchem4_no_preset_default_wiring"] = True
    _write_json(SEMANTIC_LEDGER, ledger)


def build_compact(
    replay: dict[str, Any],
    classification: dict[str, Any],
    english_guard: dict[str, Any],
) -> dict[str, Any]:
    return {
        "milestone": "FC4-M3901_TO_M3960",
        "campaign_type": "persisted_support_condensed_stage",
        "git_status_start": _git_status(),
        "fastchem4_clone_status_start": _git_status("-C", "FastChem4"),
        "persisted_stage_summary": {
            "case_count": replay["case_count"],
            "stage1_success_count": replay["stage1_success_count"],
            "stage1_kkt_nonworse_count": replay["stage1_kkt_nonworse_count"],
            "stage1_budget_nonworse_count": replay["stage1_budget_nonworse_count"],
            "finite_solver_input_count": replay["finite_solver_input_count"],
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
    summary = compact["persisted_stage_summary"]
    classification = compact["classification_summary"]
    COMPACT_MD_PATH.write_text(
        "\n".join(
            [
                "# FC4-M3901 to M3960 persisted support condensed-stage replay",
                "",
                "## Summary",
                "",
                f"- Stage-1 solver success: {summary['stage1_success_count']}/{summary['case_count']}.",
                f"- Stage-1 KKT nonworse: {summary['stage1_kkt_nonworse_count']}/{summary['case_count']}.",
                f"- Stage-1 budget nonworse: {summary['stage1_budget_nonworse_count']}/{summary['case_count']}.",
                f"- Finite solver inputs: {summary['finite_solver_input_count']}/{summary['case_count']}.",
                f"- Classification: `{classification['classification']}`.",
                f"- Next action: `{classification['next_default_action']}`.",
                "",
                "## Guardrails",
                "",
                "- This remains explicit opt-in and diagnostic-only.",
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
    _write_json(PERSISTED_PATH, replay)
    classification = build_classification(replay)
    _write_json(CLASSIFICATION_PATH, classification)
    _update_semantic_ledger(classification)
    english_guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, english_guard)
    compact = build_compact(replay, classification, english_guard)
    _write_json(COMPACT_PATH, compact)
    _write_markdown(compact)


if __name__ == "__main__":
    main()
