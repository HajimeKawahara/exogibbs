"""FC4-M4090 remaining residual decomposition after amount-weighted gas replay."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.comparisons.fastchem4_milestone4083_algorithm_v11_globalization_merit_audit import (  # noqa: E402
    RESTORATION_REPLAY,
    _case_inputs,
    _component_norms,
    _residuals_at,
)
from examples.comparisons.fastchem4_milestone2441_to_2460_budget_null_projected_outer_lifecycle_expansion import (  # noqa: E402
    _case_lookup,
)
from examples.comparisons.fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe import (  # noqa: E402
    MAX_ITERATIONS,
    THERMO_SENTINEL_ABS_THRESHOLD,
    _filtered_inputs_from_callsite,
)
from examples.comparisons.fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition import (  # noqa: E402
    _candidate_records,
    _gas_metric_frames,
)
from examples.comparisons.fastchem4_milestone4089_amount_weighted_gas_acceptance_replay import (  # noqa: E402
    _select_amount_weighted_candidate,
    _weighted_converged,
)
from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (  # noqa: E402
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state  # noqa: E402


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition.json"
COMPACT_PATH = (
    RESULTS / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition_compact.json"
)
COMPACT_MD_PATH = (
    RESULTS / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition_compact.md"
)
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4090_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

CAMPAIGN_FILES = [
    ROOT
    / "examples"
    / "comparisons"
    / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition.py",
    ROOT
    / "tests"
    / "unittests"
    / "presets"
    / "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition_test.py",
    AUDIT_PATH,
    COMPACT_PATH,
    COMPACT_MD_PATH,
    ENGLISH_GUARD_PATH,
    SEMANTIC_LEDGER,
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_status(*args: str) -> list[str]:
    command = ["git", *args, "status", "--short"] if args else ["git", "status", "--short"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.splitlines()


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
        "milestone": "FC4-M4090",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _state_extremes(
    *,
    q: np.ndarray,
    r_full: np.ndarray,
    rho_full: np.ndarray,
    gas_species_order: Sequence[str],
    condensate_species_order: Sequence[str],
) -> dict[str, Any]:
    q_max_index = int(np.argmax(q)) if q.size else -1
    q_min_index = int(np.argmin(q)) if q.size else -1
    r_max_index = int(np.argmax(r_full)) if r_full.size else -1
    rho_max_index = int(np.argmax(rho_full)) if rho_full.size else -1
    return {
        "max_q": float(q[q_max_index]) if q_max_index >= 0 else 0.0,
        "max_q_species": str(gas_species_order[q_max_index]) if q_max_index >= 0 else None,
        "min_q": float(q[q_min_index]) if q_min_index >= 0 else 0.0,
        "min_q_species": str(gas_species_order[q_min_index]) if q_min_index >= 0 else None,
        "max_abs_q": float(np.max(np.abs(q))) if q.size else 0.0,
        "max_r": float(r_full[r_max_index]) if r_max_index >= 0 else 0.0,
        "max_r_species": str(condensate_species_order[r_max_index]) if r_max_index >= 0 else None,
        "max_rho": float(rho_full[rho_max_index]) if rho_max_index >= 0 else 0.0,
        "max_rho_species": str(condensate_species_order[rho_max_index]) if rho_max_index >= 0 else None,
        "q_overflow_risk": bool(np.max(q) > 650.0) if q.size else False,
        "rho_overflow_risk": bool(np.max(rho_full) > 650.0) if rho_full.size else False,
    }


def _classify_remaining(row: Mapping[str, Any]) -> str:
    final_components = row["final_components"]
    frames = row["final_gas_metric_frames"]
    extremes = row["final_state_extremes"]
    if extremes["q_overflow_risk"] or extremes["rho_overflow_risk"]:
        return "state_overflow_risk_requires_step_cap"
    if (
        frames["raw_gas_l2"] > 1000.0
        and frames["amount_weighted_gas_l2"] < 1.0
        and final_components["condensate"] < 1.0
        and final_components["budget"] < 1.0e-3
    ):
        return "trace_species_raw_gas_large_but_weighted_floor_small"
    if final_components["condensate"] > 1.0 or final_components["complementarity"] > 1.0:
        return "condensate_or_barrier_residual_still_large"
    if frames["amount_weighted_gas_l2"] >= max(final_components["condensate"], final_components["budget"], 1.0e-6):
        return "amount_weighted_gas_floor_dominates"
    return "mixed_small_residual_floor"


def _probe_case(row: Mapping[str, Any]) -> dict[str, Any]:
    base_inputs = _case_inputs(row)
    contract = _case_lookup()[str(row["case_id"])]["contract"]
    gas_species_order = tuple(str(value) for value in contract.gas_species_order)
    condensate_species_order = tuple(str(value) for value in contract.condensate_species_order)
    support_indices = tuple(int(value) for value in row["correction_report"]["selected_condensate_indices"])
    base_inputs = {**base_inputs, "gas_species_order": gas_species_order}
    q = np.array(base_inputs["q"], dtype=np.float64)
    r_full = np.array(base_inputs["r"], dtype=np.float64)
    lam = np.array(base_inputs["lam"], dtype=np.float64)
    rho_full = np.array(base_inputs["rho"], dtype=np.float64)
    qtot = float(base_inputs["qtot"])
    accepted = 0
    status = "max_iterations_reached"
    last_selected: dict[str, Any] | None = None

    for _iteration in range(MAX_ITERATIONS):
        state = build_pdipm_rgie_condensate_state(
            ln_nk=q,
            ln_mk=r_full,
            element_potential=lam,
            ln_ntot=qtot,
            rho=rho_full,
            eta=np.exp(np.clip(rho_full, -745.0, 700.0)),
            field_provenance=base_inputs["state"].field_provenance,
        )
        inputs = {
            **base_inputs,
            "q": q,
            "r": r_full,
            "lam": lam,
            "rho": rho_full,
            "qtot": qtot,
            "state": state,
        }
        callsite = run_algorithm_v11_thermo_valid_reduced_callsite(
            explicit_opt_in=True,
            state=state,
            support_indices=support_indices,
            formula_matrix=inputs["formula_matrix"],
            formula_matrix_cond_active=inputs["formula_matrix_cond_active"],
            element_inventory_target=inputs["target"],
            gas_stationarity_source=inputs["g"],
            condensate_standard_source=inputs["c"],
            epsilon=inputs["epsilon"],
            species_names=condensate_species_order,
            sentinel_abs_threshold=THERMO_SENTINEL_ABS_THRESHOLD,
            alpha_candidates=(1.0,),
            max_abs_delta_q=1.0e300,
            max_abs_delta_r=1.0e300,
            max_abs_delta_rho=1.0e300,
            max_abs_delta_lambda=1.0e300,
            field_provenance={
                "ln_mk": "exogibbs_native_budget_restoration_selector",
                "rho": "exogibbs_native_budget_restoration_selector",
                "eta": "exogibbs_native_budget_restoration_selector",
            },
        )
        filtered_inputs = {
            **_filtered_inputs_from_callsite(inputs, callsite),
            "gas_species_order": gas_species_order,
        }
        baseline_residuals = _residuals_at(
            filtered_inputs,
            filtered_inputs["q"],
            filtered_inputs["r"],
            filtered_inputs["lam"],
            filtered_inputs["rho"],
            filtered_inputs["qtot"],
        )
        baseline_components = _component_norms(baseline_residuals)
        baseline_gas_frames = _gas_metric_frames(
            filtered_inputs,
            filtered_inputs["q"],
            filtered_inputs["lam"],
        )
        if _weighted_converged(baseline_components, baseline_gas_frames):
            status = "weighted_converged_before_trial"
            break
        records = _candidate_records(
            filtered_inputs=filtered_inputs,
            reduced=callsite.reduced_step_report.as_dict(),
            baseline_components=baseline_components,
            baseline_gas_frames=baseline_gas_frames,
        )
        selected = _select_amount_weighted_candidate(records)
        last_selected = selected
        if not selected["accepted_by_amount_weighted_policy"]:
            status = "no_amount_weighted_accepted_trial"
            break
        alpha = float(selected["alpha"])
        reduced = callsite.reduced_step_report.as_dict()
        q = q + alpha * np.asarray(reduced["delta_q"], dtype=np.float64)
        lam = lam + alpha * np.asarray(reduced["delta_lambda"], dtype=np.float64)
        qtot = float(qtot + alpha * float(reduced["delta_qtot"]))
        valid_local = tuple(callsite.filter_report.valid_local_indices)
        r_full[list(valid_local)] = filtered_inputs["r"] + alpha * np.asarray(
            reduced["delta_r"],
            dtype=np.float64,
        )
        rho_full[list(valid_local)] = filtered_inputs["rho"] + alpha * np.asarray(
            reduced["delta_rho"],
            dtype=np.float64,
        )
        accepted += 1
        if _weighted_converged(selected["components"], selected["gas_metric_frames"]):
            status = "weighted_converged_after_trial"
            break

    final_state = build_pdipm_rgie_condensate_state(
        ln_nk=q,
        ln_mk=r_full,
        element_potential=lam,
        ln_ntot=qtot,
        rho=rho_full,
        eta=np.exp(np.clip(rho_full, -745.0, 700.0)),
        field_provenance=base_inputs["state"].field_provenance,
    )
    final_unfiltered_inputs = {
        **base_inputs,
        "q": q,
        "r": r_full,
        "lam": lam,
        "rho": rho_full,
        "qtot": qtot,
        "state": final_state,
    }
    final_callsite = run_algorithm_v11_thermo_valid_reduced_callsite(
        explicit_opt_in=True,
        state=final_state,
        support_indices=support_indices,
        formula_matrix=final_unfiltered_inputs["formula_matrix"],
        formula_matrix_cond_active=final_unfiltered_inputs["formula_matrix_cond_active"],
        element_inventory_target=final_unfiltered_inputs["target"],
        gas_stationarity_source=final_unfiltered_inputs["g"],
        condensate_standard_source=final_unfiltered_inputs["c"],
        epsilon=final_unfiltered_inputs["epsilon"],
        species_names=condensate_species_order,
        sentinel_abs_threshold=THERMO_SENTINEL_ABS_THRESHOLD,
        alpha_candidates=(1.0,),
        max_abs_delta_q=1.0e300,
        max_abs_delta_r=1.0e300,
        max_abs_delta_rho=1.0e300,
        max_abs_delta_lambda=1.0e300,
        field_provenance={
            "ln_mk": "exogibbs_native_budget_restoration_selector",
            "rho": "exogibbs_native_budget_restoration_selector",
            "eta": "exogibbs_native_budget_restoration_selector",
        },
    )
    final_inputs = {
        **_filtered_inputs_from_callsite(final_unfiltered_inputs, final_callsite),
        "gas_species_order": gas_species_order,
    }
    final_residuals = _residuals_at(
        final_inputs,
        final_inputs["q"],
        final_inputs["r"],
        final_inputs["lam"],
        final_inputs["rho"],
        final_inputs["qtot"],
    )
    final_components = _component_norms(final_residuals)
    final_gas_frames = _gas_metric_frames(final_inputs, final_inputs["q"], final_inputs["lam"])
    result = {
        "case_id": row["case_id"],
        "status": status,
        "accepted_iteration_count": accepted,
        "final_components": final_components,
        "final_gas_metric_frames": final_gas_frames,
        "final_state_extremes": _state_extremes(
            q=q,
            r_full=r_full,
            rho_full=rho_full,
            gas_species_order=gas_species_order,
            condensate_species_order=condensate_species_order,
        ),
        "last_selected_candidate_summary": {
            "alpha": last_selected.get("alpha") if last_selected else None,
            "amount_weighted_relative_component_delta": last_selected.get(
                "amount_weighted_relative_component_delta"
            )
            if last_selected
            else None,
            "raw_relative_component_delta": last_selected.get("raw_relative_component_delta")
            if last_selected
            else None,
            "accepted_by_amount_weighted_policy": last_selected.get(
                "accepted_by_amount_weighted_policy"
            )
            if last_selected
            else None,
        },
    }
    return {**result, "remaining_residual_class": _classify_remaining(result)}


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [_probe_case(row) for row in source["rows"]]
    class_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["remaining_residual_class"])
        class_counts[key] = class_counts.get(key, 0) + 1
    overflow_risk_count = sum(row["final_state_extremes"]["q_overflow_risk"] or row["final_state_extremes"]["rho_overflow_risk"] for row in rows)
    decision = (
        "AMOUNT_WEIGHTED_REPLAY_NEEDS_STATE_CAP_AND_REMAINING_COMPONENT_SPLIT"
        if overflow_risk_count
        else "AMOUNT_WEIGHTED_REPLAY_REMAINING_RESIDUAL_CLASSIFIED"
    )
    return {
        "milestone": "FC4-M4090",
        "audit_schema": "exogibbs_algorithm_v11_amount_weighted_remaining_residual_decomposition_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "case_count": len(rows),
        "remaining_residual_class_counts": class_counts,
        "state_overflow_risk_count": overflow_risk_count,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4091",
        "next_default_action": "test_amount_weighted_policy_with_state_and_step_caps",
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4090 Amount-Weighted Remaining Residual Decomposition",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- State overflow risk count: `{summary['state_overflow_risk_count']}`",
        f"- Remaining residual class counts: `{summary['remaining_residual_class_counts']}`",
        f"- Next target: `{summary['next_default_target']}`",
        f"- Next action: `{summary['next_default_action']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_ledger(audit: Mapping[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger.update(
        {
            "legacy_fastchem3_cond_status": "frozen_after_M401_stoploss",
            "fastchem4_current_milestone": "FC4-M4090",
            "fastchem4_algorithm_v11_amount_weighted_remaining_residual_status": audit[
                "decision"
            ],
            "fastchem4_algorithm_v11_amount_weighted_state_overflow_risk_count": audit[
                "state_overflow_risk_count"
            ],
            "fastchem4_no_trace_values_as_constructor_inputs": True,
            "fastchem4_no_production_behavior_change": True,
            "fastchem4_no_preset_default_wiring": True,
            "fastchem4_next_default_target": audit["next_default_target"],
            "fastchem4_next_default_action": audit["next_default_action"],
        }
    )
    _write_json(SEMANTIC_LEDGER, ledger)


def main() -> None:
    git_status_start = _git_status()
    fastchem4_status_start = _git_status("-C", "FastChem4")
    audit = build_audit()
    _update_ledger(audit)
    _write_json(AUDIT_PATH, audit)
    guard = _english_only_guard(CAMPAIGN_FILES)
    _write_json(ENGLISH_GUARD_PATH, guard)
    git_status_end = _git_status()
    fastchem4_status_end = _git_status("-C", "FastChem4")
    compact = {
        "milestone": "FC4-M4090",
        "campaign_type": "amount_weighted_remaining_residual_decomposition",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "remaining_residual_class_counts": audit["remaining_residual_class_counts"],
            "state_overflow_risk_count": audit["state_overflow_risk_count"],
            "next_default_target": audit["next_default_target"],
            "next_default_action": audit["next_default_action"],
        },
        "english_only_file_guard": guard,
        "no_fastchem4_files_modified": fastchem4_status_end == ["?? fastchem4_paper.pdf"],
        "no_production_behavior_change": True,
        "no_production_return_signature_change": True,
        "no_preset_default_wiring": True,
        "no_trace_values_as_constructor_inputs": True,
        "validation": {
            "json_validation": "pending_external_validation",
            "py_compile": "pending_external_validation",
            "pytest": "pending_external_validation",
            "english_only_file_guard": (
                "passed" if guard["english_only_file_guard_passed"] else "failed"
            ),
        },
    }
    _write_json(COMPACT_PATH, compact)
    _write_markdown(COMPACT_MD_PATH, compact)
    print(json.dumps(compact["audit_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
