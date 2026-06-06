"""FC4-M3761 to M3820 FastChem4-style outer activity recompute loop."""

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
from examples.comparisons.fastchem4_milestone2181_to_2200_multi_iteration_outer_lifecycle_refresh_loop import (  # noqa: E402
    _run_limited_callsite,
)
from examples.comparisons.fastchem4_milestone3321_to_3340_dynamic_budget_priority_guard_envelope import (  # noqa: E402
    M2581_LIVE,
    SEMANTIC_LEDGER,
    _case_arrays,
    _case_lookup,
    _git_status,
    _last_budget_safe_state,
    _load_json,
    _write_json,
)
from examples.comparisons.fastchem4_milestone3371_to_3400_driver_aligned_support_refresh import (  # noqa: E402
    _source_row_by_case,
)
from examples.comparisons.fastchem4_milestone3541_to_3580_activity_threshold_support_lifecycle import (  # noqa: E402
    DIRECT_SOLVE_THRESHOLD,
    MAX_SEED_AMOUNT,
    MIN_SEED_AMOUNT,
)
from exogibbs.diagnostics.condensate_fastchem4_support_lifecycle import (  # noqa: E402
    build_fastchem4_informed_support_lifecycle_plan,
)
from exogibbs.diagnostics.condensate_hardened_gas_logdensity_refresh import (  # noqa: E402
    build_hardened_gas_logdensity_refresh_init,
)


RESULTS = ROOT / "results"
DESIGN_PATH = RESULTS / "fastchem4_milestone3761_outer_activity_recompute_design.json"
LOOP_PATH = RESULTS / "fastchem4_milestone3781_outer_activity_recompute_loop.json"
CLASSIFICATION_PATH = RESULTS / "fastchem4_milestone3820_outer_activity_recompute_classification.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone3761_to_3820_outer_activity_recompute_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone3761_to_3820_outer_activity_recompute_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone3761_to_3820_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
ACTIVE_THRESHOLD = 0.0
SEED_FRACTION = 1.0e-5
MAX_OUTER_ITERATIONS = 2
MIN_AMOUNT = 1.0e-300
NATIVE_BUDGET_SAFETY = 0.99

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone3761_to_3820_outer_activity_recompute_loop.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone3761_to_3820_outer_activity_recompute_loop_test.py",
    DESIGN_PATH,
    LOOP_PATH,
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
        "milestone": "FC4-M3761_TO_M3820",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _full_amount_vector(ncond: int, indices: tuple[int, ...], amounts: tuple[float, ...]) -> np.ndarray:
    full = np.zeros(ncond, dtype=np.float64)
    if indices:
        full[np.asarray(indices, dtype=np.int64)] = np.asarray(amounts, dtype=np.float64)
    return full


def _budget_fraction(ac: np.ndarray, target: np.ndarray, amounts_full: np.ndarray) -> float | None:
    burden = ac @ amounts_full
    positive = np.abs(target) > 0.0
    if not np.any(positive):
        return None
    return float(np.max(np.abs(burden[positive]) / np.abs(target[positive])))


def _scale_to_budget_if_needed(
    *,
    ac: np.ndarray,
    target: np.ndarray,
    indices: tuple[int, ...],
    amounts: tuple[float, ...],
) -> tuple[tuple[float, ...], float, float | None]:
    full = _full_amount_vector(ac.shape[1], indices, amounts)
    fraction = _budget_fraction(ac, target, full)
    if fraction is None or fraction <= NATIVE_BUDGET_SAFETY:
        return amounts, 1.0, fraction
    scale = NATIVE_BUDGET_SAFETY / fraction
    return tuple(float(value * scale) for value in amounts), float(scale), fraction


def _activity_like_values(
    *,
    ncond: int,
    current_support: tuple[int, ...],
    top_inactive_indices: tuple[int, ...],
    top_inactive_driving: tuple[float, ...],
) -> tuple[float, ...]:
    values = np.full(ncond, -1.0, dtype=np.float64)
    for index in current_support:
        values[int(index)] = 0.0
    for index in top_inactive_indices:
        values[int(index)] = 1.0
    for index, driving in zip(top_inactive_indices, top_inactive_driving):
        values[int(index)] = max(1.0, float(driving))
    return tuple(float(value) for value in values)


def _plan_from_activity(
    *,
    contract: Any,
    element_inventory_target: tuple[float, ...],
    current_support: tuple[int, ...],
    top_inactive_indices: tuple[int, ...],
    top_inactive_driving: tuple[float, ...],
) -> Any:
    activity_like = _activity_like_values(
        ncond=len(contract.condensate_species_order),
        current_support=current_support,
        top_inactive_indices=top_inactive_indices,
        top_inactive_driving=top_inactive_driving,
    )
    return build_fastchem4_informed_support_lifecycle_plan(
        formula_matrix_cond=contract.formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=contract.condensate_species_order,
        activity_like_values=activity_like,
        active_threshold=ACTIVE_THRESHOLD,
        direct_solve_threshold=DIRECT_SOLVE_THRESHOLD,
        seed_fraction=SEED_FRACTION,
        max_seed_amount=MAX_SEED_AMOUNT,
        min_seed_amount=MIN_AMOUNT,
        field_provenance={
            "formula_matrix_cond": "exogibbs_native",
            "element_inventory_target": "exogibbs_native",
            "activity_like_values": "exogibbs_native_post_solver_activity_proxy",
        },
    )


def _merge_support_amounts(
    *,
    plan: Any,
    previous_amount_by_index: dict[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...], int]:
    support_indices = tuple(int(index) for index in plan.active_support_indices)
    seed_by_index = {
        int(index): float(amount)
        for index, amount in zip(plan.active_support_indices, plan.support_amounts_init)
    }
    new_count = 0
    amounts: list[float] = []
    for index in support_indices:
        if index in previous_amount_by_index:
            amounts.append(max(float(previous_amount_by_index[index]), MIN_AMOUNT))
        else:
            amounts.append(max(float(seed_by_index[index]), MIN_AMOUNT))
            new_count += 1
    return support_indices, tuple(amounts), new_count


def build_design() -> dict[str, Any]:
    return {
        "milestone": "FC4-M3761",
        "design_schema": "exogibbs_outer_activity_recompute_loop_design_v1",
        "fastchem4_source_basis": {
            "outer_order": "gas solve, activity calculation, active condensate selection, condensed update, updatePhi, gas solve, total density refresh, activity recomputation",
            "activity_recompute_proxy": "post-solver top inactive drivers are used as an ExoGibbs-native activity proxy for the next diagnostic iteration",
            "support_memory": "previous active support is retained while newly active condensates are added from recomputed activity proxy",
            "gas_refresh": "each iteration refreshes gas log-density against the depleted native budget before restricted callsite replay",
        },
        "max_outer_iterations": MAX_OUTER_ITERATIONS,
        "seed_fraction": SEED_FRACTION,
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _initial_top_inactive(source_row: dict[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(int(index) for index in source_row["top_positive_inactive_indices"]),
        tuple(float(value) for value in source_row["top_inactive_driving"]),
    )


def _run_iteration(
    *,
    case_id: str,
    case: dict[str, Any],
    element_inventory_target: tuple[float, ...],
    current_support: tuple[int, ...],
    previous_amount_by_index: dict[int, float],
    top_inactive_indices: tuple[int, ...],
    top_inactive_driving: tuple[float, ...],
    iteration_index: int,
    baseline_budget: float,
    baseline_kkt: float,
    baseline_inactive: int,
) -> tuple[dict[str, Any], dict[int, float], tuple[int, ...], tuple[float, ...]]:
    contract = case["contract"]
    record = case["record"]
    ac = np.asarray(contract.formula_matrix_cond, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    plan = _plan_from_activity(
        contract=contract,
        element_inventory_target=element_inventory_target,
        current_support=current_support,
        top_inactive_indices=top_inactive_indices,
        top_inactive_driving=top_inactive_driving,
    )
    support_indices, support_amounts, newly_active_count = _merge_support_amounts(
        plan=plan,
        previous_amount_by_index=previous_amount_by_index,
    )
    support_amounts, budget_scale, budget_fraction_before_scale = _scale_to_budget_if_needed(
        ac=ac,
        target=target,
        indices=support_indices,
        amounts=support_amounts,
    )
    full_amounts = _full_amount_vector(ac.shape[1], support_indices, support_amounts)
    depleted = target - ac @ full_amounts
    depleted[np.abs(depleted) < 1.0e-300] = 0.0
    init, refresh = build_hardened_gas_logdensity_refresh_init(
        explicit_opt_in=True,
        formula_matrix=contract.formula_matrix,
        source_element_budget=_ideal_density_budget(
            contract,
            float(record["temperature"]),
            float(record["pressure"]),
        ),
        native_element_budget=tuple(float(value) for value in depleted),
        mass_action_constants=_mass_action_constants(contract, float(record["temperature"])),
        support_indices=support_indices,
        ln_mk=tuple(float(np.log(max(value, MIN_AMOUNT))) for value in support_amounts),
        field_provenance={
            "formula_matrix": "exogibbs_native_static_contract",
            "source_element_budget": "exogibbs_native_ideal_density_budget",
            "native_element_budget": "exogibbs_native_outer_activity_recompute_loop",
            "mass_action_constants": "exogibbs_native_thermochemistry",
            "support_indices": "exogibbs_native_outer_activity_recompute_loop",
            "ln_mk": "exogibbs_native_outer_activity_recompute_loop",
        },
    )
    callsite = _run_limited_callsite(
        contract=contract,
        record=record,
        case_id=case_id,
        support_indices=support_indices,
        support_amounts=support_amounts,
        initial_log_state_override=init,
    )
    seeded = callsite["seeded_callsite_report"]
    final_amount_by_index = {
        int(index): max(float(amount), MIN_AMOUNT)
        for index, amount in zip(support_indices, seeded["final_support_amounts"])
    }
    post_budget = float(seeded["post_solver_budget_residual"])
    post_kkt = float(seeded["post_solver_kkt_residual_diagnostic"])
    inactive_count = int(seeded["inactive_positive_count"])
    next_top_inactive = tuple(int(index) for index in seeded["top_positive_inactive_indices"])
    next_top_driving = tuple(float(value) for value in seeded["top_inactive_driving"])
    row = {
        "iteration_index": iteration_index,
        "support_count": len(support_indices),
        "support_indices": list(support_indices),
        "support_amounts_init": list(support_amounts),
        "newly_active_count": newly_active_count,
        "direct_solve_support_count": len(plan.direct_solve_support_indices),
        "retained_support_count": len(plan.retained_support_indices),
        "budget_fraction_before_scale": budget_fraction_before_scale,
        "budget_scale_applied": budget_scale,
        "gas_refresh_success": bool(refresh.gas_solver_success),
        "support_indices_shape_matches": bool(callsite["support_indices_shape_matches"]),
        "support_amounts_init_shape_matches": bool(callsite["support_amounts_init_shape_matches"]),
        "finite_solver_inputs": bool(callsite["finite_solver_inputs"]),
        "solver_called": bool(callsite["solver_called"]),
        "solver_success": callsite["solver_success"],
        "solver_status": callsite["solver_status"],
        "solver_n_iter": seeded["solver_n_iter"],
        "solver_final_residual": seeded["solver_final_residual"],
        "post_solver_budget_residual": post_budget,
        "post_solver_budget_delta_to_baseline": post_budget - baseline_budget,
        "post_solver_kkt_residual_diagnostic": post_kkt,
        "post_solver_kkt_delta_to_baseline": post_kkt - baseline_kkt,
        "inactive_positive_count": inactive_count,
        "inactive_positive_count_delta_to_baseline": inactive_count - baseline_inactive,
        "top_positive_inactive_indices": list(next_top_inactive),
        "top_inactive_names": list(seeded["top_inactive_names"]),
        "top_inactive_driving": list(next_top_driving),
        "normal_default_path_unchanged": bool(callsite["normal_default_path_unchanged"]),
        "production_behavior_change": bool(callsite["production_behavior_change"]),
        "production_return_signature_change": bool(callsite["production_return_signature_change"]),
        "preset_default_wiring_change": bool(callsite["preset_default_wiring_change"]),
        "fastchem4_trace_public_runtime_constructor_inputs_used": bool(
            callsite["fastchem4_trace_public_runtime_constructor_inputs_used"]
        ),
    }
    return row, final_amount_by_index, next_top_inactive, next_top_driving


def _case_loop(live_row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    case_id = str(live_row["case_id"])
    case = _case_lookup()[case_id]
    budget_state = _last_budget_safe_state(live_row)
    arrays = _case_arrays(case, budget_state)
    element_inventory_target = tuple(float(value) for value in arrays["element_inventory_target"])
    current_support = tuple(int(index) for index in budget_state["support_indices"])
    previous_amount_by_index = {
        int(index): float(amount)
        for index, amount in zip(
            current_support,
            budget_state["support_amounts_init"],
        )
    }
    top_inactive_indices, top_inactive_driving = _initial_top_inactive(source_row)
    baseline_budget = float(source_row["post_solver_budget_residual"])
    baseline_kkt = float(source_row["post_solver_kkt_residual_diagnostic"])
    baseline_inactive = int(source_row["inactive_positive_count"])
    iterations: list[dict[str, Any]] = []
    for iteration_index in range(MAX_OUTER_ITERATIONS):
        row, previous_amount_by_index, top_inactive_indices, top_inactive_driving = _run_iteration(
            case_id=case_id,
            case=case,
            element_inventory_target=element_inventory_target,
            current_support=tuple(sorted(previous_amount_by_index)),
            previous_amount_by_index=previous_amount_by_index,
            top_inactive_indices=top_inactive_indices,
            top_inactive_driving=top_inactive_driving,
            iteration_index=iteration_index,
            baseline_budget=baseline_budget,
            baseline_kkt=baseline_kkt,
            baseline_inactive=baseline_inactive,
        )
        iterations.append(row)
    final = iterations[-1]
    best_budget = min(float(row["post_solver_budget_residual"]) for row in iterations)
    best_kkt = min(float(row["post_solver_kkt_residual_diagnostic"]) for row in iterations)
    best_inactive = min(int(row["inactive_positive_count"]) for row in iterations)
    return {
        "case_id": case_id,
        "temperature": live_row["temperature"],
        "pressure": live_row["pressure"],
        "baseline_post_solver_budget_residual": baseline_budget,
        "baseline_post_solver_kkt_residual": baseline_kkt,
        "baseline_inactive_positive_count": baseline_inactive,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "final_budget_improved_vs_baseline": final["post_solver_budget_delta_to_baseline"] < 0.0,
        "final_kkt_improved_vs_baseline": final["post_solver_kkt_delta_to_baseline"] < 0.0,
        "final_inactive_reduced_vs_baseline": final[
            "inactive_positive_count_delta_to_baseline"
        ]
        < 0,
        "best_budget_residual": best_budget,
        "best_kkt_residual": best_kkt,
        "best_inactive_positive_count": best_inactive,
        "solver_success_any_iteration": any(row["solver_success"] is True for row in iterations),
        "all_iterations_finite": all(row["finite_solver_inputs"] for row in iterations),
        "normal_default_path_unchanged": all(
            row["normal_default_path_unchanged"] for row in iterations
        ),
    }


def build_loop() -> dict[str, Any]:
    live = _load_json(M2581_LIVE)
    source_by_case = _source_row_by_case()
    rows = [_case_loop(row, source_by_case[str(row["case_id"])]) for row in live["rows"]]
    return {
        "milestone": "FC4-M3781",
        "loop_schema": "exogibbs_outer_activity_recompute_loop_v1",
        "case_count": len(rows),
        "max_outer_iterations": MAX_OUTER_ITERATIONS,
        "solver_success_case_count": sum(row["solver_success_any_iteration"] for row in rows),
        "all_iterations_finite_case_count": sum(row["all_iterations_finite"] for row in rows),
        "final_budget_improved_case_count": sum(row["final_budget_improved_vs_baseline"] for row in rows),
        "final_kkt_improved_case_count": sum(row["final_kkt_improved_vs_baseline"] for row in rows),
        "final_inactive_reduced_case_count": sum(row["final_inactive_reduced_vs_baseline"] for row in rows),
        "normal_default_path_unchanged_case_count": sum(
            row["normal_default_path_unchanged"] for row in rows
        ),
        "rows": rows,
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def build_classification(loop: dict[str, Any]) -> dict[str, Any]:
    case_count = loop["case_count"]
    if loop["solver_success_case_count"] > 0:
        classification = "outer_activity_recompute_loop_reaches_partial_solver_success"
        next_action = "decompose_outer_loop_success_cases"
    elif (
        loop["final_budget_improved_case_count"] == case_count
        and loop["final_inactive_reduced_case_count"] == case_count
        and loop["final_kkt_improved_case_count"] == 0
    ):
        classification = "outer_activity_recompute_reduces_budget_and_inactive_but_kkt_still_dominates"
        next_action = "add_fastchem4_style_condensed_update_before_recompute"
    elif loop["all_iterations_finite_case_count"] == case_count:
        classification = "outer_activity_recompute_safe_but_mixed_residual_response"
        next_action = "decompose_outer_activity_recompute_residual_tradeoffs"
    else:
        classification = "outer_activity_recompute_has_input_safety_gap"
        next_action = "harden_outer_activity_recompute_inputs"
    return {
        "milestone": "FC4-M3820",
        "classification_schema": "exogibbs_outer_activity_recompute_classification_v1",
        "classification": classification,
        "solver_success_case_count": loop["solver_success_case_count"],
        "final_budget_improved_case_count": loop["final_budget_improved_case_count"],
        "final_kkt_improved_case_count": loop["final_kkt_improved_case_count"],
        "final_inactive_reduced_case_count": loop["final_inactive_reduced_case_count"],
        "next_default_target": "FC4-M3821",
        "next_default_action": next_action,
        "diagnostic_only": True,
        "default_off": True,
        "production_behavior_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _update_semantic_ledger(classification: dict[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger["legacy_fastchem3_cond_status"] = "frozen_after_M401_stoploss"
    ledger["fastchem4_current_milestone"] = "FC4-M3761_TO_M3820"
    ledger["fastchem4_outer_activity_recompute_loop_status"] = classification["classification"]
    ledger["fastchem4_next_default_target"] = classification["next_default_target"]
    ledger["fastchem4_next_default_action"] = classification["next_default_action"]
    ledger["fastchem4_no_trace_values_as_constructor_inputs"] = True
    ledger["fastchem4_no_production_behavior_change"] = True
    ledger["fastchem4_no_preset_default_wiring"] = True
    _write_json(SEMANTIC_LEDGER, ledger)


def build_compact(
    design: dict[str, Any],
    loop: dict[str, Any],
    classification: dict[str, Any],
    english_guard: dict[str, Any],
) -> dict[str, Any]:
    return {
        "milestone": "FC4-M3761_TO_M3820",
        "campaign_type": "outer_activity_recompute_loop",
        "git_status_start": _git_status(),
        "fastchem4_clone_status_start": _git_status("-C", "FastChem4"),
        "design_summary": {
            "max_outer_iterations": design["max_outer_iterations"],
            "seed_fraction": design["seed_fraction"],
        },
        "loop_summary": {
            "case_count": loop["case_count"],
            "solver_success_case_count": loop["solver_success_case_count"],
            "all_iterations_finite_case_count": loop["all_iterations_finite_case_count"],
            "final_budget_improved_case_count": loop["final_budget_improved_case_count"],
            "final_kkt_improved_case_count": loop["final_kkt_improved_case_count"],
            "final_inactive_reduced_case_count": loop["final_inactive_reduced_case_count"],
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
    loop = compact["loop_summary"]
    classification = compact["classification_summary"]
    COMPACT_MD_PATH.write_text(
        "\n".join(
            [
                "# FC4-M3761 to M3820 outer activity recompute loop",
                "",
                "## Summary",
                "",
                f"- Solver success cases: {loop['solver_success_case_count']}/{loop['case_count']}.",
                f"- Finite all-iteration cases: {loop['all_iterations_finite_case_count']}/{loop['case_count']}.",
                f"- Final budget improvement cases: {loop['final_budget_improved_case_count']}/{loop['case_count']}.",
                f"- Final KKT improvement cases: {loop['final_kkt_improved_case_count']}/{loop['case_count']}.",
                f"- Final inactive reduction cases: {loop['final_inactive_reduced_case_count']}/{loop['case_count']}.",
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
    design = build_design()
    _write_json(DESIGN_PATH, design)
    loop = build_loop()
    _write_json(LOOP_PATH, loop)
    classification = build_classification(loop)
    _write_json(CLASSIFICATION_PATH, classification)
    _update_semantic_ledger(classification)
    english_guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, english_guard)
    compact = build_compact(design, loop, classification, english_guard)
    _write_json(COMPACT_PATH, compact)
    _write_markdown(compact)


if __name__ == "__main__":
    main()
