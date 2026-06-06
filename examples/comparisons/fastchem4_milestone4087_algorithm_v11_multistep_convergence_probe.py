"""FC4-M4087 multi-step convergence probe for thermo-valid algorithm-v1.1 callsite."""

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
    ALPHA_GRID,
    COMPONENT_KEYS,
    MEANINGFUL_RELATIVE_COMPONENT_DELTA,
    RESTORATION_REPLAY,
    _case_inputs,
    _component_norms,
    _finite_residuals,
    _relative_component_score,
    _residuals_at,
)
from examples.comparisons.fastchem4_milestone2441_to_2460_budget_null_projected_outer_lifecycle_expansion import (  # noqa: E402
    _case_lookup,
)
from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (  # noqa: E402
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state  # noqa: E402


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4087_english_only_guard.json"

MAX_ITERATIONS = 40
THERMO_SENTINEL_ABS_THRESHOLD = 1.0e10
BUDGET_ABSOLUTE_NONWORSENING_TOLERANCE = 1.0e-8
CONVERGED_COMPONENT_THRESHOLD = 1.0e-6
MIN_MEANINGFUL_RELATIVE_DELTA = -1.0e-8
JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

CAMPAIGN_FILES = [
    ROOT
    / "examples"
    / "comparisons"
    / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe.py",
    ROOT
    / "tests"
    / "unittests"
    / "presets"
    / "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe_test.py",
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
        "milestone": "FC4-M4087",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _budget_tolerant_nonworsening(candidate: float, baseline: float) -> bool:
    return bool(candidate <= max(baseline, BUDGET_ABSOLUTE_NONWORSENING_TOLERANCE))


def _converged(components: Mapping[str, float]) -> bool:
    return all(float(components[key]) <= CONVERGED_COMPONENT_THRESHOLD for key in COMPONENT_KEYS)


def _filtered_inputs_from_callsite(inputs: Mapping[str, Any], callsite) -> dict[str, Any]:
    valid_local = tuple(callsite.filter_report.valid_local_indices)
    state = callsite.reduced_step_report.initial_state
    return {
        **inputs,
        "formula_matrix_cond_active": inputs["formula_matrix_cond_active"][:, valid_local],
        "c": inputs["c"][list(valid_local)],
        "r": np.asarray(state.ln_mk, dtype=np.float64),
        "rho": np.asarray(state.rho, dtype=np.float64),
    }


def _select_trial(
    *,
    filtered_inputs: Mapping[str, Any],
    reduced: Mapping[str, Any],
    baseline: Mapping[str, float],
) -> dict[str, Any]:
    delta_q = np.asarray(reduced["delta_q"], dtype=np.float64)
    delta_r = np.asarray(reduced["delta_r"], dtype=np.float64)
    delta_lambda = np.asarray(reduced["delta_lambda"], dtype=np.float64)
    delta_rho = np.asarray(reduced["delta_rho"], dtype=np.float64)
    delta_qtot = float(reduced["delta_qtot"])
    weights = {key: 1.0 for key in COMPONENT_KEYS}
    baseline_score = _relative_component_score(baseline, baseline, weights)
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        residuals = _residuals_at(
            filtered_inputs,
            filtered_inputs["q"] + float(alpha) * delta_q,
            filtered_inputs["r"] + float(alpha) * delta_r,
            filtered_inputs["lam"] + float(alpha) * delta_lambda,
            filtered_inputs["rho"] + float(alpha) * delta_rho,
            float(filtered_inputs["qtot"] + float(alpha) * delta_qtot),
        )
        components = _component_norms(residuals)
        score = _relative_component_score(components, baseline, weights)
        candidates.append(
            {
                "alpha": float(alpha),
                "finite": bool(_finite_residuals(residuals) and np.isfinite(score)),
                "components": components,
                "relative_component_score": float(score),
                "relative_component_delta": float(score - baseline_score),
                "budget_tolerant_nonworsening": _budget_tolerant_nonworsening(
                    components["budget"],
                    baseline["budget"],
                ),
            }
        )
    accepted = [
        candidate
        for candidate in candidates
        if candidate["finite"]
        and candidate["relative_component_delta"] <= MEANINGFUL_RELATIVE_COMPONENT_DELTA
        and candidate["budget_tolerant_nonworsening"]
    ]
    selected = min(
        accepted or candidates,
        key=lambda candidate: (
            not candidate["budget_tolerant_nonworsening"],
            candidate["relative_component_score"],
            candidate["alpha"],
        ),
    )
    return {
        "selected": selected,
        "accepted": bool(selected in accepted),
        "accepted_candidate_count": len(accepted),
        "delta_q": delta_q,
        "delta_r": delta_r,
        "delta_lambda": delta_lambda,
        "delta_rho": delta_rho,
        "delta_qtot": delta_qtot,
    }


def _probe_case(row: Mapping[str, Any]) -> dict[str, Any]:
    base_inputs = _case_inputs(row)
    support_indices = tuple(int(value) for value in row["correction_report"]["selected_condensate_indices"])
    contract = _case_lookup()[str(row["case_id"])]["contract"]
    q = np.array(base_inputs["q"], dtype=np.float64)
    r_full = np.array(base_inputs["r"], dtype=np.float64)
    lam = np.array(base_inputs["lam"], dtype=np.float64)
    rho_full = np.array(base_inputs["rho"], dtype=np.float64)
    qtot = float(base_inputs["qtot"])
    history: list[dict[str, Any]] = []
    status = "max_iterations_reached"

    for iteration in range(MAX_ITERATIONS):
        state = build_pdipm_rgie_condensate_state(
            ln_nk=q,
            ln_mk=r_full,
            element_potential=lam,
            ln_ntot=qtot,
            rho=rho_full,
            eta=np.exp(rho_full),
            field_provenance=base_inputs["state"].field_provenance,
        )
        inputs = {**base_inputs, "q": q, "r": r_full, "lam": lam, "rho": rho_full, "qtot": qtot, "state": state}
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
            species_names=contract.condensate_species_order,
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
        filtered_inputs = _filtered_inputs_from_callsite(inputs, callsite)
        baseline = _component_norms(
            _residuals_at(
                filtered_inputs,
                filtered_inputs["q"],
                filtered_inputs["r"],
                filtered_inputs["lam"],
                filtered_inputs["rho"],
                filtered_inputs["qtot"],
            )
        )
        if _converged(baseline):
            status = "converged_before_trial"
            history.append(
                {
                    "iteration": iteration,
                    "baseline_components": baseline,
                    "accepted": False,
                    "reason": "already_converged",
                }
            )
            break
        trial = _select_trial(
            filtered_inputs=filtered_inputs,
            reduced=callsite.reduced_step_report.as_dict(),
            baseline=baseline,
        )
        selected = trial["selected"]
        history.append(
            {
                "iteration": iteration,
                "baseline_components": baseline,
                "selected_candidate": selected,
                "accepted": trial["accepted"],
                "removed_support_count": callsite.removed_support_count,
                "filtered_support_count": callsite.filtered_support_count,
            }
        )
        if not trial["accepted"]:
            status = "no_accepted_multistep_trial"
            break
        alpha = float(selected["alpha"])
        q = q + alpha * trial["delta_q"]
        lam = lam + alpha * trial["delta_lambda"]
        qtot = float(qtot + alpha * trial["delta_qtot"])
        valid_local = tuple(callsite.filter_report.valid_local_indices)
        r_full[list(valid_local)] = filtered_inputs["r"] + alpha * trial["delta_r"]
        rho_full[list(valid_local)] = filtered_inputs["rho"] + alpha * trial["delta_rho"]
        if _converged(selected["components"]):
            status = "converged_after_trial"
            break

    initial_components = history[0]["baseline_components"] if history else {}
    final_components = (
        history[-1].get("selected_candidate", {}).get("components")
        or history[-1].get("baseline_components", {})
        if history
        else {}
    )
    accepted_iterations = sum(item.get("accepted", False) for item in history)
    final_score = (
        _relative_component_score(final_components, initial_components, {key: 1.0 for key in COMPONENT_KEYS})
        if initial_components and final_components
        else float("inf")
    )
    return {
        "case_id": row["case_id"],
        "status": status,
        "converged": status in {"converged_before_trial", "converged_after_trial"},
        "iteration_count": len(history),
        "accepted_iteration_count": accepted_iterations,
        "initial_components": initial_components,
        "final_components": final_components,
        "final_relative_score_vs_initial": float(final_score),
        "history": history,
    }


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [_probe_case(row) for row in source["rows"]]
    converged_count = sum(row["converged"] for row in rows)
    accepted_any_count = sum(row["accepted_iteration_count"] > 0 for row in rows)
    decision = (
        "ALGORITHM_V11_THERMO_VALID_MULTISTEP_CONVERGED_ALL_CASES"
        if converged_count == len(rows)
        else (
            "ALGORITHM_V11_THERMO_VALID_MULTISTEP_MOVES_BUT_NOT_CONVERGED"
            if accepted_any_count > 0
            else "ALGORITHM_V11_THERMO_VALID_MULTISTEP_NO_PROGRESS"
        )
    )
    return {
        "milestone": "FC4-M4087",
        "audit_schema": "exogibbs_algorithm_v11_multistep_convergence_probe_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "max_iterations": MAX_ITERATIONS,
        "converged_component_threshold": CONVERGED_COMPONENT_THRESHOLD,
        "case_count": len(rows),
        "converged_case_count": converged_count,
        "accepted_any_case_count": accepted_any_count,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4088",
        "next_default_action": (
            "decompose_algorithm_v11_multistep_nonconvergence"
            if converged_count < len(rows)
            else "compare_converged_multistep_outputs_to_public_fastchem4"
        ),
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4087 Algorithm-v1.1 Multi-Step Convergence Probe",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Converged case count: `{summary['converged_case_count']}`",
        f"- Accepted-any case count: `{summary['accepted_any_case_count']}`",
        f"- Next target: `{summary['next_default_target']}`",
        f"- Next action: `{summary['next_default_action']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_ledger(audit: Mapping[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger.update(
        {
            "legacy_fastchem3_cond_status": "frozen_after_M401_stoploss",
            "fastchem4_current_milestone": "FC4-M4087",
            "fastchem4_algorithm_v11_multistep_convergence_status": audit["decision"],
            "fastchem4_algorithm_v11_multistep_converged_case_count": audit[
                "converged_case_count"
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
        "milestone": "FC4-M4087",
        "campaign_type": "algorithm_v11_multistep_convergence_probe",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "converged_case_count": audit["converged_case_count"],
            "accepted_any_case_count": audit["accepted_any_case_count"],
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
