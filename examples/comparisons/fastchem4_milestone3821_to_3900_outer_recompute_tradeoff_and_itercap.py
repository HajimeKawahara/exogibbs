"""FC4-M3821 to M3900 outer recompute tradeoff and iteration-cap study."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.comparisons.fastchem4_milestone137_to_144_gas_logdensity_boundary_solver import (  # noqa: E402
    _ideal_density_budget,
    _mass_action_constants,
)
from examples.comparisons.fastchem4_milestone106_to_112_public_output_aligned_native_probe_records import (  # noqa: E402
    _cond_hvector,
    _gas_hvector,
    _state_for_record,
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
TRADEOFF_PATH = RESULTS / "fastchem4_milestone3821_outer_recompute_tradeoff_study.json"
ITERCAP_PATH = RESULTS / "fastchem4_milestone3861_outer_recompute_itercap_replay.json"
CLASSIFICATION_PATH = RESULTS / "fastchem4_milestone3900_outer_recompute_next_process_classification.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone3821_to_3900_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
EXTENDED_MAX_ITER = 40
EXTENDED_GAS_MAX_ITER = 12
SUPPORT_AMOUNT_FALLBACK = 1.0e-12

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_and_itercap.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_and_itercap_test.py",
    TRADEOFF_PATH,
    ITERCAP_PATH,
    CLASSIFICATION_PATH,
    COMPACT_PATH,
    COMPACT_MD_PATH,
    ENGLISH_GUARD_PATH,
    SEMANTIC_LEDGER,
]


def _english_only_guard(paths: list[Path]) -> dict[str, Any]:
    scanned: list[str] = []
    violations: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        scanned.append(str(path.relative_to(ROOT)))
        if JAPANESE_OR_CJK_RE.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(ROOT)))
    return {
        "milestone": "FC4-M3821_TO_M3900",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _composite_merit(
    *,
    budget: float,
    kkt: float,
    inactive: int,
    baseline_budget: float,
    baseline_kkt: float,
    baseline_inactive: int,
) -> float:
    budget_scale = max(abs(baseline_budget), 1.0e-300)
    kkt_scale = max(abs(baseline_kkt), 1.0e-300)
    inactive_scale = max(float(baseline_inactive), 1.0)
    return float(
        np.log1p(max(budget, 0.0) / budget_scale)
        + 0.1 * np.log1p(max(kkt, 0.0) / kkt_scale)
        + 0.01 * (max(float(inactive), 0.0) / inactive_scale)
    )


def build_tradeoff() -> dict[str, Any]:
    loop = _load_json(M3781_LOOP_PATH)
    rows: list[dict[str, Any]] = []
    for case_row in loop["rows"]:
        baseline_budget = float(case_row["baseline_post_solver_budget_residual"])
        baseline_kkt = float(case_row["baseline_post_solver_kkt_residual"])
        baseline_inactive = int(case_row["baseline_inactive_positive_count"])
        iteration_rows = []
        for iteration in case_row["iterations"]:
            budget = float(iteration["post_solver_budget_residual"])
            kkt = float(iteration["post_solver_kkt_residual_diagnostic"])
            inactive = int(iteration["inactive_positive_count"])
            iteration_rows.append(
                {
                    "iteration_index": iteration["iteration_index"],
                    "support_count": iteration["support_count"],
                    "newly_active_count": iteration["newly_active_count"],
                    "budget_ratio_to_baseline": budget / max(baseline_budget, 1.0e-300),
                    "kkt_ratio_to_baseline": kkt / max(baseline_kkt, 1.0e-300),
                    "inactive_ratio_to_baseline": inactive / max(float(baseline_inactive), 1.0),
                    "solver_final_residual": iteration["solver_final_residual"],
                    "solver_n_iter": iteration["solver_n_iter"],
                    "solver_status": iteration["solver_status"],
                    "composite_merit": _composite_merit(
                        budget=budget,
                        kkt=kkt,
                        inactive=inactive,
                        baseline_budget=baseline_budget,
                        baseline_kkt=baseline_kkt,
                        baseline_inactive=baseline_inactive,
                    ),
                }
            )
        best_by_merit = min(iteration_rows, key=lambda item: item["composite_merit"])
        final = iteration_rows[-1]
        rows.append(
            {
                "case_id": case_row["case_id"],
                "baseline_budget": baseline_budget,
                "baseline_kkt": baseline_kkt,
                "baseline_inactive": baseline_inactive,
                "iteration_rows": iteration_rows,
                "best_iteration_by_composite_merit": best_by_merit["iteration_index"],
                "final_iteration_index": final["iteration_index"],
                "final_budget_floor_reached": final["budget_ratio_to_baseline"] < 1.0e-6,
                "final_inactive_zero": final["inactive_ratio_to_baseline"] == 0.0,
                "final_kkt_nonworse": final["kkt_ratio_to_baseline"] <= 1.0,
                "tradeoff_class": (
                    "budget_inactive_and_kkt_improve"
                    if final["kkt_ratio_to_baseline"] <= 1.0
                    else "budget_inactive_improve_but_kkt_worsens"
                ),
            }
        )
    return {
        "milestone": "FC4-M3821",
        "tradeoff_schema": "exogibbs_outer_recompute_tradeoff_study_v1",
        "source_artifact": str(M3781_LOOP_PATH.relative_to(ROOT)),
        "case_count": len(rows),
        "budget_floor_reached_count": sum(row["final_budget_floor_reached"] for row in rows),
        "inactive_zero_count": sum(row["final_inactive_zero"] for row in rows),
        "kkt_nonworse_count": sum(row["final_kkt_nonworse"] for row in rows),
        "kkt_worsening_count": sum(not row["final_kkt_nonworse"] for row in rows),
        "solver_iteration_cap_hit_count": sum(
            any(iteration["solver_n_iter"] == 8 for iteration in row["iteration_rows"])
            for row in rows
        ),
        "rows": rows,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _run_extended_case(case_row: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case_row["case_id"])
    case = _case_lookup()[case_id]
    contract = case["contract"]
    record = case["record"]
    final_iteration = case_row["iterations"][-1]
    support_indices = tuple(int(index) for index in final_iteration["support_indices"])
    support_amounts = tuple(float(value) for value in final_iteration["support_amounts_init"])
    if not support_amounts:
        support_amounts = tuple(SUPPORT_AMOUNT_FALLBACK for _ in support_indices)
    full_support_amounts = np.zeros(len(contract.condensate_species_order), dtype=np.float64)
    full_support_amounts[np.asarray(support_indices, dtype=int)] = np.asarray(
        support_amounts,
        dtype=np.float64,
    )
    depleted_budget = np.asarray(contract.element_inventory_target, dtype=np.float64) - (
        np.asarray(contract.formula_matrix_cond, dtype=np.float64) @ full_support_amounts
    )
    depleted_budget[np.abs(depleted_budget) < 1.0e-300] = 0.0
    init, refresh = build_hardened_gas_logdensity_refresh_init(
        explicit_opt_in=True,
        formula_matrix=contract.formula_matrix,
        source_element_budget=_ideal_density_budget(
            contract,
            float(record["temperature"]),
            float(record["pressure"]),
        ),
        native_element_budget=tuple(float(value) for value in depleted_budget),
        mass_action_constants=_mass_action_constants(contract, float(record["temperature"])),
        support_indices=support_indices,
        ln_mk=tuple(float(np.log(max(value, MIN_AMOUNT))) for value in support_amounts),
        field_provenance={
            "formula_matrix": "exogibbs_native_static_contract",
            "source_element_budget": "exogibbs_native_ideal_density_budget",
            "native_element_budget": "exogibbs_native_itercap_replay",
            "mass_action_constants": "exogibbs_native_thermochemistry",
            "support_indices": "exogibbs_native_itercap_replay",
            "ln_mk": "exogibbs_native_itercap_replay",
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
        support_amounts_init=support_amounts,
        initial_log_state_override=init,
        max_budget_fraction=1.0,
        gas_max_iter=EXTENDED_GAS_MAX_ITER,
        max_iter=EXTENDED_MAX_ITER,
        line_search_selection_policy="first_monotone_with_best_finite_fallback",
        line_search_charge_row_name="e-",
        line_search_charge_weight=1.0,
        field_provenance={
            "formula_matrix": "exogibbs_native_static_contract",
            "formula_matrix_cond": "exogibbs_native_static_contract",
            "element_inventory_target": "exogibbs_native_abundance_contract",
            "hvector_cond": "exogibbs_native_thermochemistry_static_contract",
            "support_indices": "exogibbs_native_itercap_replay",
            "support_amounts_init": "exogibbs_native_itercap_replay",
            "initial_log_state_override": "exogibbs_native_hardened_gas_logdensity_refresh",
        },
    )
    data = report.as_dict()
    seeded = data["seeded_callsite_report"]
    return {
        "case_id": case_id,
        "support_count": len(support_indices),
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
    }


def build_itercap_replay() -> dict[str, Any]:
    loop = _load_json(M3781_LOOP_PATH)
    rows = [_run_extended_case(row) for row in loop["rows"]]
    return {
        "milestone": "FC4-M3861",
        "itercap_replay_schema": "exogibbs_outer_recompute_extended_itercap_replay_v1",
        "max_iter": EXTENDED_MAX_ITER,
        "case_count": len(rows),
        "solver_success_count": sum(row["solver_success"] is True for row in rows),
        "finite_solver_input_count": sum(row["finite_solver_inputs"] for row in rows),
        "normal_default_path_unchanged_count": sum(row["normal_default_path_unchanged"] for row in rows),
        "iteration_cap_hit_count": sum(row["solver_n_iter"] == EXTENDED_MAX_ITER for row in rows),
        "rows": rows,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def build_classification(tradeoff: dict[str, Any], itercap: dict[str, Any]) -> dict[str, Any]:
    if itercap["solver_success_count"] > 0:
        classification = "extended_itercap_reaches_partial_solver_success"
        next_action = "replay_outer_recompute_with_persisted_final_support_state"
    elif itercap["iteration_cap_hit_count"] == itercap["case_count"]:
        classification = "kkt_tradeoff_not_resolved_by_iteration_cap_only"
        next_action = "persist_final_support_state_and_add_condensed_update_stage"
    else:
        classification = "kkt_tradeoff_requires_component_specific_decomposition"
        next_action = "decompose_kkt_floor_by_stationarity_and_support_rows"
    return {
        "milestone": "FC4-M3900",
        "classification_schema": "exogibbs_outer_recompute_next_process_classification_v1",
        "classification": classification,
        "tradeoff_kkt_worsening_count": tradeoff["kkt_worsening_count"],
        "itercap_solver_success_count": itercap["solver_success_count"],
        "itercap_iteration_cap_hit_count": itercap["iteration_cap_hit_count"],
        "next_default_target": "FC4-M3901",
        "next_default_action": next_action,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _update_semantic_ledger(classification: dict[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger["legacy_fastchem3_cond_status"] = "frozen_after_M401_stoploss"
    ledger["fastchem4_current_milestone"] = "FC4-M3821_TO_M3900"
    ledger["fastchem4_outer_recompute_tradeoff_status"] = classification["classification"]
    ledger["fastchem4_next_default_target"] = classification["next_default_target"]
    ledger["fastchem4_next_default_action"] = classification["next_default_action"]
    ledger["fastchem4_no_trace_values_as_constructor_inputs"] = True
    ledger["fastchem4_no_production_behavior_change"] = True
    ledger["fastchem4_no_preset_default_wiring"] = True
    _write_json(SEMANTIC_LEDGER, ledger)


def build_compact(
    tradeoff: dict[str, Any],
    itercap: dict[str, Any],
    classification: dict[str, Any],
    english_guard: dict[str, Any],
) -> dict[str, Any]:
    return {
        "milestone": "FC4-M3821_TO_M3900",
        "campaign_type": "outer_recompute_tradeoff_and_itercap",
        "git_status_start": _git_status(),
        "fastchem4_clone_status_start": _git_status("-C", "FastChem4"),
        "tradeoff_summary": {
            "case_count": tradeoff["case_count"],
            "budget_floor_reached_count": tradeoff["budget_floor_reached_count"],
            "inactive_zero_count": tradeoff["inactive_zero_count"],
            "kkt_nonworse_count": tradeoff["kkt_nonworse_count"],
            "kkt_worsening_count": tradeoff["kkt_worsening_count"],
        },
        "itercap_summary": {
            "max_iter": itercap["max_iter"],
            "solver_success_count": itercap["solver_success_count"],
            "iteration_cap_hit_count": itercap["iteration_cap_hit_count"],
            "finite_solver_input_count": itercap["finite_solver_input_count"],
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
    tradeoff = compact["tradeoff_summary"]
    itercap = compact["itercap_summary"]
    classification = compact["classification_summary"]
    COMPACT_MD_PATH.write_text(
        "\n".join(
            [
                "# FC4-M3821 to M3900 outer recompute tradeoff and iteration-cap study",
                "",
                "## Summary",
                "",
                f"- Budget floor reached: {tradeoff['budget_floor_reached_count']}/{tradeoff['case_count']}.",
                f"- Inactive zero reached: {tradeoff['inactive_zero_count']}/{tradeoff['case_count']}.",
                f"- KKT nonworse cases: {tradeoff['kkt_nonworse_count']}/{tradeoff['case_count']}.",
                f"- Extended max_iter: {itercap['max_iter']}.",
                f"- Extended solver success: {itercap['solver_success_count']}/{tradeoff['case_count']}.",
                f"- Extended iteration cap hits: {itercap['iteration_cap_hit_count']}/{tradeoff['case_count']}.",
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
    tradeoff = build_tradeoff()
    _write_json(TRADEOFF_PATH, tradeoff)
    itercap = build_itercap_replay()
    _write_json(ITERCAP_PATH, itercap)
    classification = build_classification(tradeoff, itercap)
    _write_json(CLASSIFICATION_PATH, classification)
    _update_semantic_ledger(classification)
    english_guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, english_guard)
    compact = build_compact(tradeoff, itercap, classification, english_guard)
    _write_json(COMPACT_PATH, compact)
    _write_markdown(compact)


if __name__ == "__main__":
    main()
