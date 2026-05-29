"""FC4-M4085 replay of the thermo-valid support helper on curated cases."""

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
from exogibbs.diagnostics.condensate_thermo_valid_support import (  # noqa: E402
    filter_thermo_valid_condensate_support,
)
from exogibbs.optimize.pdipm_rgie_cond import (  # noqa: E402
    build_pdipm_rgie_condensate_state,
    solve_pdipm_rgie_algorithm_v11_reduced_step,
)


RESULTS = ROOT / "results"
SEMANTIC_LEDGER = RESULTS / "condensate_fastchem_semantic_levers.json"
AUDIT_PATH = RESULTS / "fastchem4_milestone4085_thermo_valid_support_helper_replay.json"
COMPACT_PATH = RESULTS / "fastchem4_milestone4085_thermo_valid_support_helper_replay_compact.json"
COMPACT_MD_PATH = RESULTS / "fastchem4_milestone4085_thermo_valid_support_helper_replay_compact.md"
ENGLISH_GUARD_PATH = RESULTS / "fastchem4_milestone4085_english_only_guard.json"

THERMO_SENTINEL_ABS_THRESHOLD = 1.0e10
BUDGET_ABSOLUTE_NONWORSENING_TOLERANCE = 1.0e-8
JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

CAMPAIGN_FILES = [
    ROOT / "src" / "exogibbs" / "diagnostics" / "condensate_thermo_valid_support.py",
    ROOT / "tests" / "unittests" / "diagnostics" / "condensate_thermo_valid_support_test.py",
    ROOT / "examples" / "comparisons" / "fastchem4_milestone4085_thermo_valid_support_helper_replay.py",
    ROOT
    / "tests"
    / "unittests"
    / "presets"
    / "fastchem4_milestone4085_thermo_valid_support_helper_replay_test.py",
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
        "milestone": "FC4-M4085",
        "english_only_file_guard_passed": not violations,
        "files_scanned": scanned,
        "remaining_violations": violations,
    }


def _budget_tolerant_nonworsening(candidate: float, baseline: float) -> bool:
    return bool(candidate <= max(baseline, BUDGET_ABSOLUTE_NONWORSENING_TOLERANCE))


def _replay_case(row: Mapping[str, Any]) -> dict[str, Any]:
    inputs = _case_inputs(row)
    support_indices = tuple(int(value) for value in row["correction_report"]["selected_condensate_indices"])
    contract = _case_lookup()[str(row["case_id"])]["contract"]
    filtered = filter_thermo_valid_condensate_support(
        explicit_opt_in=True,
        support_indices=support_indices,
        condensate_standard_source=inputs["c"],
        formula_matrix_cond_active=inputs["formula_matrix_cond_active"],
        ln_mk=inputs["r"],
        rho=inputs["rho"],
        eta=np.exp(inputs["rho"]),
        species_names=contract.condensate_species_order,
        sentinel_abs_threshold=THERMO_SENTINEL_ABS_THRESHOLD,
        field_provenance={
            "ln_mk": "exogibbs_native_budget_restoration_selector",
            "rho": "exogibbs_native_budget_restoration_selector",
            "eta": "exogibbs_native_budget_restoration_selector",
        },
    )
    ac = np.asarray(filtered.formula_matrix_cond_active, dtype=np.float64)
    c = np.asarray(filtered.condensate_standard_source, dtype=np.float64)
    r = np.asarray(filtered.ln_mk, dtype=np.float64)
    rho = np.asarray(filtered.rho, dtype=np.float64)
    state = build_pdipm_rgie_condensate_state(
        ln_nk=inputs["q"],
        ln_mk=r,
        element_potential=inputs["lam"],
        ln_ntot=inputs["qtot"],
        rho=rho,
        eta=np.exp(rho),
        field_provenance={
            "ln_nk": "exogibbs_native_budget_restoration_selector",
            "ln_mk": "exogibbs_native_budget_restoration_selector",
            "element_potential": "exogibbs_native_budget_restoration_selector",
            "rho": "exogibbs_native_budget_restoration_selector",
            "eta": "exogibbs_native_budget_restoration_selector",
        },
    )
    replay_inputs = {
        **inputs,
        "formula_matrix_cond_active": ac,
        "c": c,
        "r": r,
        "rho": rho,
        "state": state,
    }
    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=replay_inputs["formula_matrix"],
        formula_matrix_cond_active=ac,
        element_inventory_target=replay_inputs["target"],
        gas_stationarity_source=replay_inputs["g"],
        condensate_standard_source=c,
        epsilon=replay_inputs["epsilon"],
        alpha_candidates=(1.0,),
        max_abs_delta_q=1.0e300,
        max_abs_delta_r=1.0e300,
        max_abs_delta_rho=1.0e300,
        max_abs_delta_lambda=1.0e300,
        qhat_regularization=0.0,
    ).as_dict()
    delta_q = np.asarray(report["delta_q"], dtype=np.float64)
    delta_r = np.asarray(report["delta_r"], dtype=np.float64)
    delta_lambda = np.asarray(report["delta_lambda"], dtype=np.float64)
    delta_rho = np.asarray(report["delta_rho"], dtype=np.float64)
    delta_qtot = float(report["delta_qtot"])
    baseline = _component_norms(
        _residuals_at(
            replay_inputs,
            replay_inputs["q"],
            replay_inputs["r"],
            replay_inputs["lam"],
            replay_inputs["rho"],
            replay_inputs["qtot"],
        )
    )
    weights = {key: 1.0 for key in COMPONENT_KEYS}
    baseline_score = _relative_component_score(baseline, baseline, weights)
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        residuals = _residuals_at(
            replay_inputs,
            replay_inputs["q"] + float(alpha) * delta_q,
            replay_inputs["r"] + float(alpha) * delta_r,
            replay_inputs["lam"] + float(alpha) * delta_lambda,
            replay_inputs["rho"] + float(alpha) * delta_rho,
            float(replay_inputs["qtot"] + float(alpha) * delta_qtot),
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
        "case_id": row["case_id"],
        "filter_report": filtered.report.as_dict(),
        "baseline_components": baseline,
        "qhat_condition_estimate": report["qhat_condition_estimate"],
        "selected_candidate": selected,
        "accepted_candidate_count": len(accepted),
        "accepted_by_m4085_policy": bool(selected in accepted),
    }


def build_audit() -> dict[str, Any]:
    source = _load_json(RESTORATION_REPLAY)
    rows = [_replay_case(row) for row in source["rows"]]
    accepted_count = sum(row["accepted_by_m4085_policy"] for row in rows)
    decision = (
        "THERMO_VALID_SUPPORT_HELPER_REPRODUCES_M4084_RESCUE"
        if accepted_count == len(rows)
        else "THERMO_VALID_SUPPORT_HELPER_REPLAY_INCOMPLETE"
    )
    return {
        "milestone": "FC4-M4085",
        "audit_schema": "exogibbs_thermo_valid_support_helper_replay_v1",
        "diagnostic_only": True,
        "default_off": True,
        "explicit_opt_in": True,
        "production_behavior_change": False,
        "production_return_signature_change": False,
        "preset_default_wiring_change": False,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        "helper_module_path": "src/exogibbs/diagnostics/condensate_thermo_valid_support.py",
        "case_count": len(rows),
        "accepted_case_count": accepted_count,
        "rows": rows,
        "decision": decision,
        "next_default_target": "FC4-M4086",
        "next_default_action": (
            "feed_thermo_valid_support_filter_into_algorithm_v11_callsite_policy"
            if accepted_count == len(rows)
            else "debug_thermo_valid_support_helper_replay"
        ),
    }


def _write_markdown(path: Path, compact: Mapping[str, Any]) -> None:
    summary = compact["audit_summary"]
    lines = [
        "# FC4-M4085 Thermo-Valid Support Helper Replay",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Accepted case count: `{summary['accepted_case_count']}`",
        f"- Helper module: `{summary['helper_module_path']}`",
        f"- Next target: `{summary['next_default_target']}`",
        f"- Next action: `{summary['next_default_action']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_ledger(audit: Mapping[str, Any]) -> None:
    ledger = _load_json(SEMANTIC_LEDGER)
    ledger.update(
        {
            "legacy_fastchem3_cond_status": "frozen_after_M401_stoploss",
            "fastchem4_current_milestone": "FC4-M4085",
            "fastchem4_thermo_valid_support_helper_status": audit["decision"],
            "fastchem4_thermo_valid_support_helper_module_path": audit["helper_module_path"],
            "fastchem4_thermo_valid_support_helper_accepted_case_count": audit[
                "accepted_case_count"
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
        "milestone": "FC4-M4085",
        "campaign_type": "thermo_valid_support_helper_replay",
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "fastchem4_clone_status_start": fastchem4_status_start,
        "fastchem4_clone_status_end": fastchem4_status_end,
        "audit_summary": {
            "decision": audit["decision"],
            "case_count": audit["case_count"],
            "accepted_case_count": audit["accepted_case_count"],
            "helper_module_path": audit["helper_module_path"],
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
