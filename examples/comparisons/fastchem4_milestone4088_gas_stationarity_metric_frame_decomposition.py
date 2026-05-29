"""FC4-M4088 gas-stationarity metric-frame decomposition for algorithm-v1.1."""

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
from examples.comparisons.fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe import (  # noqa: E402
    BUDGET_ABSOLUTE_NONWORSENING_TOLERANCE,
    MAX_ITERATIONS,
    THERMO_SENTINEL_ABS_THRESHOLD,
    _budget_tolerant_nonworsening,
    _converged,
    _filtered_inputs_from_callsite,
)
from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (  # noqa: E402
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state  # noqa: E402


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition.json"
COMPACT_PATH = (
    RESULTS / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition_compact.json"
)
COMPACT_MD_PATH = (
    RESULTS / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition_compact.md"
)
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4088_english_only_guard.json"

JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
WEIGHTED_RELATIVE_DELTA_TOLERANCE = -1.0e-8

CAMPAIGN_FILES = [
    ROOT
    / "examples"
    / "comparisons"
    / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition.py",
    ROOT
    / "tests"
    / "unittests"
    / "presets"
    / "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition_test.py",
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
        "milestone": "FC4-M4088",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _gas_metric_frames(inputs: Mapping[str, Any], q: np.ndarray, lam: np.ndarray) -> dict[str, Any]:
    residuals = _residuals_at(
        inputs,
        q,
        inputs["r"],
        lam,
        inputs["rho"],
        inputs["qtot"],
    )
    gas = np.asarray(residuals["gas"], dtype=np.float64)
    n = np.exp(np.asarray(q, dtype=np.float64))
    ntot = float(np.sum(n))
    mole_fraction = n / max(ntot, 1.0e-300)
    species_names = tuple(str(value) for value in inputs["gas_species_order"])
    raw_abs = np.abs(gas)
    amount_abs = np.abs(n * gas)
    mole_fraction_abs = np.abs(mole_fraction * gas)
    raw_top = int(np.argmax(raw_abs)) if raw_abs.size else -1
    amount_top = int(np.argmax(amount_abs)) if amount_abs.size else -1
    mole_top = int(np.argmax(mole_fraction_abs)) if mole_fraction_abs.size else -1
    return {
        "raw_gas_l2": float(np.linalg.norm(gas)),
        "amount_weighted_gas_l2": float(np.linalg.norm(n * gas)),
        "sqrt_amount_weighted_gas_l2": float(np.linalg.norm(np.sqrt(n) * gas)),
        "mole_fraction_weighted_gas_l2": float(np.linalg.norm(mole_fraction * gas)),
        "raw_top_species": species_names[raw_top] if raw_top >= 0 else None,
        "amount_weighted_top_species": species_names[amount_top] if amount_top >= 0 else None,
        "mole_fraction_weighted_top_species": species_names[mole_top] if mole_top >= 0 else None,
        "raw_top_abs": float(raw_abs[raw_top]) if raw_top >= 0 else 0.0,
        "amount_weighted_top_abs": float(amount_abs[amount_top]) if amount_top >= 0 else 0.0,
        "mole_fraction_weighted_top_abs": float(mole_fraction_abs[mole_top]) if mole_top >= 0 else 0.0,
        "total_gas_amount": ntot,
    }


def _gas_metric_relative_delta(candidate: Mapping[str, float], baseline: Mapping[str, float], key: str) -> float:
    return float(candidate[key] / max(float(baseline[key]), 1.0) - 1.0)


def _weighted_component_score(
    *,
    components: Mapping[str, float],
    gas_frames: Mapping[str, float],
    baseline_components: Mapping[str, float],
    baseline_gas_frames: Mapping[str, float],
    gas_key: str,
) -> float:
    score = 0.0
    for key in COMPONENT_KEYS:
        if key == "gas":
            score += float(gas_frames[gas_key]) / max(float(baseline_gas_frames[gas_key]), 1.0)
        else:
            score += float(components[key]) / max(float(baseline_components[key]), 1.0)
    return float(score)


def _candidate_records(
    *,
    filtered_inputs: Mapping[str, Any],
    reduced: Mapping[str, Any],
    baseline_components: Mapping[str, float],
    baseline_gas_frames: Mapping[str, float],
) -> list[dict[str, Any]]:
    delta_q = np.asarray(reduced["delta_q"], dtype=np.float64)
    delta_r = np.asarray(reduced["delta_r"], dtype=np.float64)
    delta_lambda = np.asarray(reduced["delta_lambda"], dtype=np.float64)
    delta_rho = np.asarray(reduced["delta_rho"], dtype=np.float64)
    delta_qtot = float(reduced["delta_qtot"])
    baseline_raw_score = _relative_component_score(
        baseline_components,
        baseline_components,
        {key: 1.0 for key in COMPONENT_KEYS},
    )
    baseline_amount_score = _weighted_component_score(
        components=baseline_components,
        gas_frames=baseline_gas_frames,
        baseline_components=baseline_components,
        baseline_gas_frames=baseline_gas_frames,
        gas_key="amount_weighted_gas_l2",
    )
    records: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        alpha_float = float(alpha)
        q = filtered_inputs["q"] + alpha_float * delta_q
        r = filtered_inputs["r"] + alpha_float * delta_r
        lam = filtered_inputs["lam"] + alpha_float * delta_lambda
        rho = filtered_inputs["rho"] + alpha_float * delta_rho
        qtot = float(filtered_inputs["qtot"] + alpha_float * delta_qtot)
        residuals = _residuals_at(filtered_inputs, q, r, lam, rho, qtot)
        components = _component_norms(residuals)
        frame_inputs = {**filtered_inputs, "r": r, "rho": rho, "qtot": qtot}
        gas_frames = _gas_metric_frames(frame_inputs, q, lam)
        raw_score = _relative_component_score(
            components,
            baseline_components,
            {key: 1.0 for key in COMPONENT_KEYS},
        )
        amount_score = _weighted_component_score(
            components=components,
            gas_frames=gas_frames,
            baseline_components=baseline_components,
            baseline_gas_frames=baseline_gas_frames,
            gas_key="amount_weighted_gas_l2",
        )
        budget_ok = _budget_tolerant_nonworsening(
            components["budget"],
            baseline_components["budget"],
        )
        finite = bool(_finite_residuals(residuals) and np.isfinite(raw_score) and np.isfinite(amount_score))
        records.append(
            {
                "alpha": alpha_float,
                "finite": finite,
                "components": components,
                "gas_metric_frames": gas_frames,
                "raw_relative_component_delta": float(raw_score - baseline_raw_score),
                "amount_weighted_relative_component_delta": float(
                    amount_score - baseline_amount_score
                ),
                "raw_gas_relative_delta": _gas_metric_relative_delta(
                    gas_frames,
                    baseline_gas_frames,
                    "raw_gas_l2",
                ),
                "amount_weighted_gas_relative_delta": _gas_metric_relative_delta(
                    gas_frames,
                    baseline_gas_frames,
                    "amount_weighted_gas_l2",
                ),
                "sqrt_amount_weighted_gas_relative_delta": _gas_metric_relative_delta(
                    gas_frames,
                    baseline_gas_frames,
                    "sqrt_amount_weighted_gas_l2",
                ),
                "mole_fraction_weighted_gas_relative_delta": _gas_metric_relative_delta(
                    gas_frames,
                    baseline_gas_frames,
                    "mole_fraction_weighted_gas_l2",
                ),
                "budget_tolerant_nonworsening": budget_ok,
                "accepted_by_raw_m4087_policy": bool(
                    finite
                    and raw_score - baseline_raw_score <= MEANINGFUL_RELATIVE_COMPONENT_DELTA
                    and budget_ok
                ),
                "accepted_by_amount_weighted_policy": bool(
                    finite
                    and amount_score - baseline_amount_score <= WEIGHTED_RELATIVE_DELTA_TOLERANCE
                    and budget_ok
                ),
            }
        )
    return records


def _best_record(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    finite = [record for record in records if record["finite"]]
    return dict(min(finite or records, key=lambda record: float(record[key])))


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
    iteration_records: list[dict[str, Any]] = []
    stop_classification = "max_iterations_reached"

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
        if _converged(baseline_components):
            stop_classification = "converged_before_trial"
            iteration_records.append(
                {
                    "iteration": iteration,
                    "baseline_components": baseline_components,
                    "baseline_gas_metric_frames": baseline_gas_frames,
                    "accepted_by_raw_m4087_policy": False,
                    "accepted_by_amount_weighted_policy": False,
                    "reason": "already_converged",
                }
            )
            break
        candidate_records = _candidate_records(
            filtered_inputs=filtered_inputs,
            reduced=callsite.reduced_step_report.as_dict(),
            baseline_components=baseline_components,
            baseline_gas_frames=baseline_gas_frames,
        )
        raw_accepted = [
            record for record in candidate_records if record["accepted_by_raw_m4087_policy"]
        ]
        weighted_accepted = [
            record for record in candidate_records if record["accepted_by_amount_weighted_policy"]
        ]
        selected_raw = min(
            raw_accepted or candidate_records,
            key=lambda record: (
                not record["budget_tolerant_nonworsening"],
                record["raw_relative_component_delta"],
                record["alpha"],
            ),
        )
        selected_weighted = min(
            weighted_accepted or candidate_records,
            key=lambda record: (
                not record["budget_tolerant_nonworsening"],
                record["amount_weighted_relative_component_delta"],
                record["alpha"],
            ),
        )
        record = {
            "iteration": iteration,
            "baseline_components": baseline_components,
            "baseline_gas_metric_frames": baseline_gas_frames,
            "selected_raw_policy_candidate": selected_raw,
            "selected_amount_weighted_policy_candidate": selected_weighted,
            "best_raw_gas_candidate": _best_record(candidate_records, "raw_gas_relative_delta"),
            "best_amount_weighted_gas_candidate": _best_record(
                candidate_records,
                "amount_weighted_gas_relative_delta",
            ),
            "raw_accepted_candidate_count": len(raw_accepted),
            "amount_weighted_accepted_candidate_count": len(weighted_accepted),
            "accepted_by_raw_m4087_policy": bool(raw_accepted),
            "accepted_by_amount_weighted_policy": bool(weighted_accepted),
            "removed_support_count": callsite.removed_support_count,
            "filtered_support_count": callsite.filtered_support_count,
        }
        iteration_records.append(record)
        if raw_accepted:
            alpha = float(selected_raw["alpha"])
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
            if _converged(selected_raw["components"]):
                stop_classification = "converged_after_trial"
                break
            continue
        stop_classification = (
            "raw_policy_stopped_but_amount_weighted_policy_has_candidate"
            if weighted_accepted
            else "raw_and_amount_weighted_policies_both_stopped"
        )
        break

    stop_record = iteration_records[-1] if iteration_records else {}
    metric_mismatch_likely = bool(
        stop_classification == "raw_policy_stopped_but_amount_weighted_policy_has_candidate"
    )
    true_weighted_gas_deterioration_likely = bool(
        stop_classification == "raw_and_amount_weighted_policies_both_stopped"
        and stop_record.get("best_amount_weighted_gas_candidate", {}).get(
            "amount_weighted_gas_relative_delta",
            1.0,
        )
        > 0.0
    )
    return {
        "case_id": row["case_id"],
        "iteration_count": len(iteration_records),
        "stop_classification": stop_classification,
        "metric_frame_mismatch_likely": metric_mismatch_likely,
        "true_weighted_gas_deterioration_likely": true_weighted_gas_deterioration_likely,
        "raw_policy_accepted_iteration_count": sum(
            1 for record in iteration_records if record.get("accepted_by_raw_m4087_policy")
        ),
        "amount_weighted_policy_available_at_stop": bool(
            stop_record.get("accepted_by_amount_weighted_policy", False)
        ),
        "stop_record": stop_record,
        "history": iteration_records,
    }


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [_probe_case(row) for row in source["rows"]]
    mismatch_count = sum(row["metric_frame_mismatch_likely"] for row in rows)
    weighted_deterioration_count = sum(row["true_weighted_gas_deterioration_likely"] for row in rows)
    decision = (
        "GAS_STATIONARITY_STOP_IS_RAW_METRIC_FRAME_MISMATCH"
        if mismatch_count == len(rows)
        else (
            "GAS_STATIONARITY_STOP_MIXED_RAW_AND_WEIGHTED_FAILURE"
            if mismatch_count > 0
            else "GAS_STATIONARITY_STOP_REMAINS_WEIGHTED_GAS_FAILURE"
        )
    )
    return {
        "milestone": "FC4-M4088",
        "audit_schema": "exogibbs_algorithm_v11_gas_stationarity_metric_frame_decomposition_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "gas_metric_frames": [
            "raw_gas_l2",
            "amount_weighted_gas_l2",
            "sqrt_amount_weighted_gas_l2",
            "mole_fraction_weighted_gas_l2",
        ],
        "case_count": len(rows),
        "metric_frame_mismatch_likely_count": mismatch_count,
        "true_weighted_gas_deterioration_likely_count": weighted_deterioration_count,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4089",
        "next_default_action": (
            "test_amount_weighted_gas_stationarity_acceptance_policy"
            if mismatch_count > 0
            else "decompose_weighted_gas_stationarity_direction_failure"
        ),
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4088 Gas-Stationarity Metric-Frame Decomposition",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Metric-frame mismatch likely count: `{summary['metric_frame_mismatch_likely_count']}`",
        f"- Weighted gas deterioration likely count: `{summary['true_weighted_gas_deterioration_likely_count']}`",
        f"- Next target: `{summary['next_default_target']}`",
        f"- Next action: `{summary['next_default_action']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_ledger(audit: Mapping[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger.update(
        {
            "legacy_fastchem3_cond_status": "frozen_after_M401_stoploss",
            "fastchem4_current_milestone": "FC4-M4088",
            "fastchem4_algorithm_v11_gas_stationarity_metric_frame_status": audit[
                "decision"
            ],
            "fastchem4_algorithm_v11_metric_frame_mismatch_likely_count": audit[
                "metric_frame_mismatch_likely_count"
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
        "milestone": "FC4-M4088",
        "campaign_type": "gas_stationarity_metric_frame_decomposition",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "metric_frame_mismatch_likely_count": audit[
                "metric_frame_mismatch_likely_count"
            ],
            "true_weighted_gas_deterioration_likely_count": audit[
                "true_weighted_gas_deterioration_likely_count"
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
