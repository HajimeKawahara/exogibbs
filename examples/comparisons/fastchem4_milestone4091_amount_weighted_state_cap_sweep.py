"""FC4-M4091 amount-weighted gas policy with log-state and step caps."""

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
    _dominant_weighted_component,
    _weighted_converged,
)
from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (  # noqa: E402
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state  # noqa: E402


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4091_amount_weighted_state_cap_sweep.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone4091_amount_weighted_state_cap_sweep_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone4091_amount_weighted_state_cap_sweep_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4091_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
POLICIES = (
    {"name": "delta_q_cap_2_state_abs_500", "delta_q_cap": 2.0, "state_abs_q_cap": 500.0},
    {"name": "delta_q_cap_1_state_abs_500", "delta_q_cap": 1.0, "state_abs_q_cap": 500.0},
    {"name": "delta_q_cap_0p5_state_abs_500", "delta_q_cap": 0.5, "state_abs_q_cap": 500.0},
    {"name": "delta_q_cap_0p25_state_abs_500", "delta_q_cap": 0.25, "state_abs_q_cap": 500.0},
)

CAMPAIGN_FILES = [
    ROOT / "examples" / "comparisons" / "fastchem4_milestone4091_amount_weighted_state_cap_sweep.py",
    ROOT / "tests" / "unittests" / "presets" / "fastchem4_milestone4091_amount_weighted_state_cap_sweep_test.py",
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
        "milestone": "FC4-M4091",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _select_capped_candidate(
    *,
    records: Sequence[Mapping[str, Any]],
    filtered_inputs: Mapping[str, Any],
    reduced: Mapping[str, Any],
    state_abs_q_cap: float,
) -> dict[str, Any]:
    delta_q = np.asarray(reduced["delta_q"], dtype=np.float64)
    capped_records: list[dict[str, Any]] = []
    for record in records:
        candidate = dict(record)
        candidate_q = filtered_inputs["q"] + float(record["alpha"]) * delta_q
        candidate["state_abs_q_cap_ok"] = bool(
            np.all(np.isfinite(candidate_q)) and np.max(np.abs(candidate_q)) <= state_abs_q_cap
        )
        capped_records.append(candidate)
    accepted = [
        record
        for record in capped_records
        if record["accepted_by_amount_weighted_policy"] and record["state_abs_q_cap_ok"]
    ]
    return dict(
        min(
            accepted or capped_records,
            key=lambda record: (
                not record["state_abs_q_cap_ok"],
                not record["budget_tolerant_nonworsening"],
                record["amount_weighted_relative_component_delta"],
                record["alpha"],
            ),
        )
    )


def _run_case_policy(row: Mapping[str, Any], policy: Mapping[str, float | str]) -> dict[str, Any]:
    base_inputs = _case_inputs(row)
    contract = _case_lookup()[str(row["case_id"])]["contract"]
    gas_species_order = tuple(str(value) for value in contract.gas_species_order)
    support_indices = tuple(int(value) for value in row["correction_report"]["selected_condensate_indices"])
    base_inputs = {**base_inputs, "gas_species_order": gas_species_order}
    q = np.array(base_inputs["q"], dtype=np.float64)
    r_full = np.array(base_inputs["r"], dtype=np.float64)
    lam = np.array(base_inputs["lam"], dtype=np.float64)
    rho_full = np.array(base_inputs["rho"], dtype=np.float64)
    qtot = float(base_inputs["qtot"])
    accepted = 0
    rejected_by_state_cap = 0
    status = "max_iterations_reached"

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
            species_names=contract.condensate_species_order,
            sentinel_abs_threshold=THERMO_SENTINEL_ABS_THRESHOLD,
            alpha_candidates=(1.0,),
            max_abs_delta_q=float(policy["delta_q_cap"]),
            max_abs_delta_r=2.0,
            max_abs_delta_rho=2.0,
            max_abs_delta_lambda=100.0,
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
        reduced = callsite.reduced_step_report.as_dict()
        records = _candidate_records(
            filtered_inputs=filtered_inputs,
            reduced=reduced,
            baseline_components=baseline_components,
            baseline_gas_frames=baseline_gas_frames,
        )
        selected = _select_capped_candidate(
            records=records,
            filtered_inputs=filtered_inputs,
            reduced=reduced,
            state_abs_q_cap=float(policy["state_abs_q_cap"]),
        )
        if not selected.get("state_abs_q_cap_ok", False):
            rejected_by_state_cap += 1
        if not (
            selected["accepted_by_amount_weighted_policy"]
            and selected.get("state_abs_q_cap_ok", False)
        ):
            status = "no_capped_amount_weighted_accepted_trial"
            break
        alpha = float(selected["alpha"])
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
        species_names=contract.condensate_species_order,
        sentinel_abs_threshold=THERMO_SENTINEL_ABS_THRESHOLD,
        alpha_candidates=(1.0,),
        max_abs_delta_q=float(policy["delta_q_cap"]),
        max_abs_delta_r=2.0,
        max_abs_delta_rho=2.0,
        max_abs_delta_lambda=100.0,
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
    final_components = _component_norms(
        _residuals_at(
            final_inputs,
            final_inputs["q"],
            final_inputs["r"],
            final_inputs["lam"],
            final_inputs["rho"],
            final_inputs["qtot"],
        )
    )
    final_gas_frames = _gas_metric_frames(final_inputs, final_inputs["q"], final_inputs["lam"])
    return {
        "case_id": row["case_id"],
        "policy_name": str(policy["name"]),
        "delta_q_cap": float(policy["delta_q_cap"]),
        "state_abs_q_cap": float(policy["state_abs_q_cap"]),
        "status": status,
        "weighted_converged": status in {
            "weighted_converged_before_trial",
            "weighted_converged_after_trial",
        },
        "accepted_iteration_count": accepted,
        "rejected_by_state_cap_count": rejected_by_state_cap,
        "final_components": final_components,
        "final_gas_metric_frames": final_gas_frames,
        "final_dominant_weighted_component": _dominant_weighted_component(
            final_components,
            final_gas_frames,
        ),
        "max_abs_q": float(np.max(np.abs(q))) if q.size else 0.0,
    }


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [
        _run_case_policy(row, policy)
        for policy in POLICIES
        for row in source["rows"]
    ]
    policy_summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_rows = [row for row in rows if row["policy_name"] == policy["name"]]
        converged = sum(row["weighted_converged"] for row in policy_rows)
        accepted_any = sum(row["accepted_iteration_count"] > 0 for row in policy_rows)
        mean_weighted_gas = float(
            np.mean([row["final_gas_metric_frames"]["amount_weighted_gas_l2"] for row in policy_rows])
        )
        mean_condensate = float(np.mean([row["final_components"]["condensate"] for row in policy_rows]))
        mean_budget = float(np.mean([row["final_components"]["budget"] for row in policy_rows]))
        policy_summaries.append(
            {
                "policy_name": policy["name"],
                "weighted_converged_case_count": converged,
                "accepted_any_case_count": accepted_any,
                "mean_final_amount_weighted_gas_l2": mean_weighted_gas,
                "mean_final_condensate_l2": mean_condensate,
                "mean_final_budget_l2": mean_budget,
                "max_final_abs_q": float(max(row["max_abs_q"] for row in policy_rows)),
            }
        )
    best_policy = min(
        policy_summaries,
        key=lambda row: (
            -row["weighted_converged_case_count"],
            row["mean_final_condensate_l2"] + row["mean_final_amount_weighted_gas_l2"],
            row["max_final_abs_q"],
        ),
    )
    decision = (
        "STATE_CAP_SWEEP_FOUND_CONVERGED_POLICY"
        if best_policy["weighted_converged_case_count"] == len(source["rows"])
        else "STATE_CAP_SWEEP_REDUCES_STATE_EXTREMES_BUT_NOT_CONVERGED"
    )
    return {
        "milestone": "FC4-M4091",
        "audit_schema": "exogibbs_algorithm_v11_amount_weighted_state_cap_sweep_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "case_count": len(source["rows"]),
        "policy_count": len(POLICIES),
        "policy_summaries": policy_summaries,
        "best_policy": best_policy,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4092",
        "next_default_action": "compare_capped_weighted_policy_against_condensate_barrier_floor",
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4091 Amount-Weighted State Cap Sweep",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Policy count: `{summary['policy_count']}`",
        f"- Best policy: `{summary['best_policy']['policy_name']}`",
        f"- Next target: `{summary['next_default_target']}`",
        f"- Next action: `{summary['next_default_action']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_ledger(audit: Mapping[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger.update(
        {
            "legacy_fastchem3_cond_status": "frozen_after_M401_stoploss",
            "fastchem4_current_milestone": "FC4-M4091",
            "fastchem4_algorithm_v11_amount_weighted_state_cap_status": audit["decision"],
            "fastchem4_algorithm_v11_amount_weighted_state_cap_best_policy": audit[
                "best_policy"
            ]["policy_name"],
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
        "milestone": "FC4-M4091",
        "campaign_type": "amount_weighted_state_cap_sweep",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "policy_count": audit["policy_count"],
            "best_policy": audit["best_policy"],
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
