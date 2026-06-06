"""FC4-M4089 amount-weighted gas-stationarity acceptance replay."""

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
    COMPONENT_KEYS,
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
    _converged,
    _filtered_inputs_from_callsite,
)
from examples.comparisons.fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition import (  # noqa: E402
    _candidate_records,
    _gas_metric_frames,
)
from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (  # noqa: E402
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state  # noqa: E402


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4089_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
CONVERGED_WEIGHTED_COMPONENT_THRESHOLD = 1.0e-6

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay.py",
    ROOT
    / "tests"
    / "unittests"
    / "presets"
    / "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay_test.py",
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
        "milestone": "FC4-M4089",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _weighted_converged(
    components: Mapping[str, float],
    gas_frames: Mapping[str, float],
) -> bool:
    return bool(
        float(gas_frames["amount_weighted_gas_l2"]) <= CONVERGED_WEIGHTED_COMPONENT_THRESHOLD
        and float(components["condensate"]) <= CONVERGED_WEIGHTED_COMPONENT_THRESHOLD
        and float(components["budget"]) <= CONVERGED_WEIGHTED_COMPONENT_THRESHOLD
        and float(components["complementarity"]) <= CONVERGED_WEIGHTED_COMPONENT_THRESHOLD
        and float(components["total_density"]) <= CONVERGED_WEIGHTED_COMPONENT_THRESHOLD
    )


def _select_amount_weighted_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [record for record in records if record["accepted_by_amount_weighted_policy"]]
    return dict(
        min(
            accepted or records,
            key=lambda record: (
                not record["budget_tolerant_nonworsening"],
                record["amount_weighted_relative_component_delta"],
                record["alpha"],
            ),
        )
    )


def _dominant_weighted_component(
    components: Mapping[str, float],
    gas_frames: Mapping[str, float],
) -> str:
    values = {
        "amount_weighted_gas": float(gas_frames["amount_weighted_gas_l2"]),
        "condensate": float(components["condensate"]),
        "budget": float(components["budget"]),
        "complementarity": float(components["complementarity"]),
        "total_density": float(components["total_density"]),
    }
    return max(values, key=values.get)


def _probe_case(row: Mapping[str, Any]) -> dict[str, Any]:
    base_inputs = _case_inputs(row)
    contract = _case_lookup()[str(row["case_id"])]["contract"]
    support_indices = tuple(int(value) for value in row["correction_report"]["selected_condensate_indices"])
    base_inputs = {
        **base_inputs,
        "gas_species_order": tuple(str(value) for value in contract.gas_species_order),
    }
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
        filtered_inputs = {
            **_filtered_inputs_from_callsite(inputs, callsite),
            "gas_species_order": base_inputs["gas_species_order"],
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
            history.append(
                {
                    "iteration": iteration,
                    "baseline_components": baseline_components,
                    "baseline_gas_metric_frames": baseline_gas_frames,
                    "accepted": False,
                    "reason": "already_weighted_converged",
                }
            )
            break
        records = _candidate_records(
            filtered_inputs=filtered_inputs,
            reduced=callsite.reduced_step_report.as_dict(),
            baseline_components=baseline_components,
            baseline_gas_frames=baseline_gas_frames,
        )
        selected = _select_amount_weighted_candidate(records)
        accepted = bool(selected["accepted_by_amount_weighted_policy"])
        history.append(
            {
                "iteration": iteration,
                "baseline_components": baseline_components,
                "baseline_gas_metric_frames": baseline_gas_frames,
                "selected_candidate": selected,
                "accepted": accepted,
                "amount_weighted_accepted_candidate_count": sum(
                    1 for record in records if record["accepted_by_amount_weighted_policy"]
                ),
                "raw_accepted_candidate_count": sum(
                    1 for record in records if record["accepted_by_raw_m4087_policy"]
                ),
                "dominant_weighted_baseline_component": _dominant_weighted_component(
                    baseline_components,
                    baseline_gas_frames,
                ),
                "dominant_weighted_candidate_component": _dominant_weighted_component(
                    selected["components"],
                    selected["gas_metric_frames"],
                ),
            }
        )
        if not accepted:
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
        if _weighted_converged(selected["components"], selected["gas_metric_frames"]):
            status = "weighted_converged_after_trial"
            break

    initial_components = history[0]["baseline_components"] if history else {}
    initial_gas_frames = history[0]["baseline_gas_metric_frames"] if history else {}
    final_components = (
        history[-1].get("selected_candidate", {}).get("components")
        or history[-1].get("baseline_components", {})
        if history
        else {}
    )
    final_gas_frames = (
        history[-1].get("selected_candidate", {}).get("gas_metric_frames")
        or history[-1].get("baseline_gas_metric_frames", {})
        if history
        else {}
    )
    return {
        "case_id": row["case_id"],
        "status": status,
        "weighted_converged": status in {
            "weighted_converged_before_trial",
            "weighted_converged_after_trial",
        },
        "raw_converged": bool(final_components and _converged(final_components)),
        "iteration_count": len(history),
        "accepted_iteration_count": sum(item.get("accepted", False) for item in history),
        "initial_components": initial_components,
        "initial_gas_metric_frames": initial_gas_frames,
        "final_components": final_components,
        "final_gas_metric_frames": final_gas_frames,
        "final_dominant_weighted_component": (
            _dominant_weighted_component(final_components, final_gas_frames)
            if final_components and final_gas_frames
            else "unavailable"
        ),
        "history": history,
    }


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [_probe_case(row) for row in source["rows"]]
    weighted_converged_count = sum(row["weighted_converged"] for row in rows)
    raw_converged_count = sum(row["raw_converged"] for row in rows)
    accepted_any_count = sum(row["accepted_iteration_count"] > 0 for row in rows)
    decision = (
        "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_CONVERGED_ALL_CASES"
        if weighted_converged_count == len(rows)
        else (
            "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_MOVES_BUT_NOT_CONVERGED"
            if accepted_any_count > 0
            else "AMOUNT_WEIGHTED_GAS_ACCEPTANCE_NO_PROGRESS"
        )
    )
    dominant_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["final_dominant_weighted_component"])
        dominant_counts[key] = dominant_counts.get(key, 0) + 1
    return {
        "milestone": "FC4-M4089",
        "audit_schema": "exogibbs_algorithm_v11_amount_weighted_gas_acceptance_replay_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "case_count": len(rows),
        "weighted_converged_case_count": weighted_converged_count,
        "raw_converged_case_count": raw_converged_count,
        "accepted_any_case_count": accepted_any_count,
        "final_dominant_weighted_component_counts": dominant_counts,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4090",
        "next_default_action": (
            "decompose_amount_weighted_multistep_remaining_residual"
            if weighted_converged_count < len(rows)
            else "compare_algorithm_v11_weighted_converged_outputs"
        ),
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4089 Amount-Weighted Gas Acceptance Replay",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Weighted converged case count: `{summary['weighted_converged_case_count']}`",
        f"- Raw converged case count: `{summary['raw_converged_case_count']}`",
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
            "fastchem4_current_milestone": "FC4-M4089",
            "fastchem4_algorithm_v11_amount_weighted_gas_acceptance_status": audit[
                "decision"
            ],
            "fastchem4_algorithm_v11_amount_weighted_gas_weighted_converged_case_count": audit[
                "weighted_converged_case_count"
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
        "milestone": "FC4-M4089",
        "campaign_type": "amount_weighted_gas_acceptance_replay",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "weighted_converged_case_count": audit["weighted_converged_case_count"],
            "raw_converged_case_count": audit["raw_converged_case_count"],
            "accepted_any_case_count": audit["accepted_any_case_count"],
            "final_dominant_weighted_component_counts": audit[
                "final_dominant_weighted_component_counts"
            ],
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
