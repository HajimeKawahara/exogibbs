"""Diagnostic-only FastChem/ExoGibbs variable-mapping micro-parity audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
FASTCHEM_PYTHON = REPO_ROOT / "fastchem" / "python"
for path in (FASTCHEM_PYTHON, REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.common import current_timestamp_utc
from benchmarks.common import to_python
from exogibbs.api.chemistry import ThermoState
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup
from exogibbs.thermo.stoichiometry import contract_formula_matrix
from exogibbs.utils.fastchem_parity import build_aligned_abundance_vector

import audit_fastchem_downstream_staged_transplant as branch_map_audit
import audit_fastchem_exact_update_map_transplant as transplant
import audit_fastchem_reduced_reconstruction_parity as reduced_recon
import audit_fastchem_update_map_alignment_phase2 as phase2
import audit_fastchem_update_semantics_parity as update_parity
import audit_kl_exact_maxdensity_lifecycle as exact_lifecycle
import audit_parity_fixed_rgie_baseline_rebaseline as rebase


DEFAULT_CASES = ((0, 0.0), (45, -10.0), (90, -5.0))
DEFAULT_PROFILE_NPZ = REPO_ROOT / "documents" / "ipynb" / "pipm" / "rgie" / "vmr_fastchem_prof.npz"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "fastchem_exogibbs_variable_mapping_audit.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "results" / "fastchem_exogibbs_variable_mapping_audit.md"
DEFAULT_BRANCH_MAP_JSON = REPO_ROOT / "results" / "exogibbs_fastchem_experimental_branch_map.json"
DEFAULT_BRANCH_MAP_MD = REPO_ROOT / "results" / "exogibbs_fastchem_experimental_branch_map.md"

TARGET_STAGE_SPECS = (
    ("post_gas_only_activity_maxdensity_scan", "A post gas-only activity/maxDensity scan"),
    ("post_selectActiveCondensates_reset", "B post selectActiveCondensates reset"),
    ("post_calculate_entry_seeding", "C post calculate() entry seeding"),
    ("after_first_correctValues_update", "D post first correctValues / correctValuesFull update"),
    ("post_final_removal", "E post final removal"),
)
EXAMPLE_SPECIES = ("MgCO3", "NaOH", "Al(s)", "AlClO", "AlF3", "Na3AlF6", "KAlSi3O8", "Al2O3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=[f"{layer}:{epsilon}" for layer, epsilon in DEFAULT_CASES])
    parser.add_argument("--profile-npz", type=Path, default=DEFAULT_PROFILE_NPZ)
    parser.add_argument("--least-squares-max-nfev", type=int, default=20)
    parser.add_argument("--gas-max-iter", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--update-branch-map", action="store_true", default=True)
    parser.add_argument("--branch-map-json", type=Path, default=DEFAULT_BRANCH_MAP_JSON)
    parser.add_argument("--branch-map-md", type=Path, default=DEFAULT_BRANCH_MAP_MD)
    return parser


def parse_cases(values: Sequence[str]) -> list[tuple[int, float]]:
    return transplant.parse_cases(values)


def first_records_for_stage(records: Sequence[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    if stage == "after_first_correctValues_update":
        out = update_parity.records_by_stage(records, "after_first_correctValues_update", record_type="condensate")
        if not out:
            out = update_parity.records_by_stage(records, "after_first_correctValuesFull_update", record_type="condensate")
        return update_parity.first_record_per_condensate(out)
    return update_parity.first_record_per_condensate(
        update_parity.records_by_stage(records, stage, record_type="condensate")
    )


def safe_log_ratio(left: float, right: float) -> float | None:
    if not np.isfinite(left) or not np.isfinite(right) or left <= 0.0 or right <= 0.0:
        return None
    return float(math.log(left / right))


def update_factor(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0.0 or after <= 0.0:
        return None
    return float(after / before)


def stage_aligned_rows(
    *,
    fastchem_records: Sequence[dict[str, Any]],
    exogibbs_by_stage: dict[str, dict[int, dict[str, Any]]],
    cond_species: Sequence[str],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for stage, label in TARGET_STAGE_SPECS:
        fc_stage = first_records_for_stage(fastchem_records, stage)
        if stage == "after_first_correctValues_update" and not fc_stage:
            fc_stage = first_records_for_stage(fastchem_records, "after_first_correctValuesFull_update")
        rows = []
        for fc in fc_stage:
            idx = int(fc["condensate_index"])
            eg = exogibbs_by_stage.get(stage, {}).get(idx, {})
            fc_update = update_factor(float(fc.get("number_density_before", 0.0)), float(fc.get("number_density_after", 0.0)))
            eg_update = eg.get("update_factor")
            fc_n = float(fc.get("number_density_after", 0.0))
            eg_n = eg.get("number_density_exogibbs")
            fc_lam = float(fc.get("activity_correction_after", 0.0))
            eg_lam = eg.get("lambda_exogibbs")
            value_score = 0.0
            for ratio in (safe_log_ratio(fc_n, eg_n if eg_n is not None else 0.0), safe_log_ratio(fc_lam, eg_lam if eg_lam is not None else 0.0)):
                value_score += 0.0 if ratio is None else abs(ratio)
            if fc_update is not None and eg_update is not None:
                ratio = safe_log_ratio(fc_update, eg_update)
                value_score += 0.0 if ratio is None else abs(ratio)
            rows.append(
                {
                    "stage": stage,
                    "stage_label": label,
                    "condensate_index": idx,
                    "condensate": str(fc.get("condensate") or cond_species[idx]),
                    "log_activity_c": float(fc.get("log_activity", float("nan"))),
                    "maxDensity_c": float(fc.get("maxDensity", float("nan"))),
                    "number_density_fastchem": fc_n,
                    "number_density_exogibbs": eg_n,
                    "activity_correction_fastchem": fc_lam,
                    "lambda_exogibbs": eg_lam,
                    "chi_exogibbs": eg.get("chi_exogibbs"),
                    "update_factor_fastchem": fc_update,
                    "update_factor_exogibbs": eg_update,
                    "cap_fired_fastchem": bool(fc.get("cap_fired", False)),
                    "cap_fired_exogibbs": bool(eg.get("cap_fired_exogibbs", False)),
                    "newly_active": bool(fc.get("newly_active", False)) or bool(eg.get("newly_active", False)),
                    "removed": bool(fc.get("removed", False)) or bool(eg.get("removed", False)),
                    "stage_value_mismatch_score": float(value_score),
                }
            )
        rows.sort(key=lambda row: row["stage_value_mismatch_score"], reverse=True)
        out[stage] = rows[:top_k]
    return out


def mapping_diagnostic_bookkeeping(pairs: Sequence[dict[str, float]]) -> dict[str, Any]:
    scores = {"activity_correction~lambda": [], "activity_correction~1/lambda": [], "activity_correction~exp(-chi)": []}
    for pair in pairs:
        fc = float(pair["activity_correction_fastchem"])
        lam = float(pair["lambda_exogibbs"])
        chi = pair.get("chi_exogibbs")
        direct = safe_log_ratio(fc, lam)
        inverse = safe_log_ratio(fc, 1.0 / lam if lam > 0.0 else 0.0)
        exp_minus_chi = safe_log_ratio(fc, math.exp(-float(chi))) if chi is not None and np.isfinite(float(chi)) else None
        if direct is not None:
            scores["activity_correction~lambda"].append(abs(direct))
        if inverse is not None:
            scores["activity_correction~1/lambda"].append(abs(inverse))
        if exp_minus_chi is not None:
            scores["activity_correction~exp(-chi)"].append(abs(exp_minus_chi))
    mean_scores = {key: float(np.mean(vals)) if vals else float("inf") for key, vals in scores.items()}
    best = min(mean_scores, key=mean_scores.get)
    return {
        "mean_abs_log_scores": mean_scores,
        "best_mapping": best,
        "pair_count": int(len(pairs)),
        "inference": "FastChem activity_correction best matches ExoGibbs lambda directly."
        if best == "activity_correction~lambda"
        else "FastChem activity_correction appears inverted relative to ExoGibbs lambda/chi.",
    }


def maxdensity_formula_scores(
    *,
    fastchem_seed_records: Sequence[dict[str, Any]],
    candidate_indices: Sequence[int],
    formula_matrix_cond_np: np.ndarray,
    element_vector: np.ndarray,
    atomic_density: np.ndarray,
) -> dict[str, Any]:
    fc_by_idx = {int(row["condensate_index"]): float(row["maxDensity"]) for row in fastchem_seed_records}
    rows = []
    score_budget = []
    score_atomic = []
    for idx in candidate_indices:
        if int(idx) not in fc_by_idx:
            continue
        stoich = formula_matrix_cond_np[:, int(idx)]
        used = stoich > 0.0
        if not np.any(used):
            continue
        budget = float(np.min(element_vector[used] / stoich[used]))
        atomic = float(np.min(atomic_density[used] / stoich[used]))
        fc = max(float(fc_by_idx[int(idx)]), 1.0e-300)
        sb = abs(math.log(max(budget, 1.0e-300) / fc))
        sa = abs(math.log(max(atomic, 1.0e-300) / fc))
        score_budget.append(sb)
        score_atomic.append(sa)
        rows.append(
            {
                "condensate_index": int(idx),
                "fastchem_maxDensity": fc,
                "budget_formula_min_epsilon_total_over_stoich": budget,
                "current_atomic_density_over_stoich": atomic,
                "abs_log_budget_formula_ratio": float(sb),
                "abs_log_atomic_formula_ratio": float(sa),
            }
        )
    mean_budget = float(np.mean(score_budget)) if score_budget else float("inf")
    mean_atomic = float(np.mean(score_atomic)) if score_atomic else float("inf")
    if not score_budget:
        interpretation = "not observable for this trace"
    elif mean_budget < mean_atomic * 0.8:
        interpretation = "A: min_j(epsilon_j * total_element_density / stoich_j)"
    elif mean_atomic < mean_budget * 0.8:
        interpretation = "B: min_j(current_atomic_density_j / stoich_j)"
    else:
        interpretation = "C: formulas are effectively equivalent after normalization for these traces"
    return {
        "mean_abs_log_budget_formula": mean_budget,
        "mean_abs_log_current_atomic_formula": mean_atomic,
        "supported_interpretation": interpretation,
        "top_formula_rows": sorted(rows, key=lambda row: min(row["abs_log_budget_formula_ratio"], row["abs_log_atomic_formula_ratio"]), reverse=True)[:12],
    }


def _exogibbs_stage_snapshots(
    *,
    exact: exact_lifecycle.LifecycleSolve,
    phase2_trace: dict[str, Any],
    formula_matrix_cond_np: np.ndarray,
) -> dict[str, dict[int, dict[str, Any]]]:
    candidate_indices = [int(i) for i in exact.candidate_indices]
    out: dict[str, dict[int, dict[str, Any]]] = {stage: {} for stage, _label in TARGET_STAGE_SPECS}
    for local, idx in enumerate(candidate_indices):
        max_density = float(exact.max_density[local]) if local < len(exact.max_density) else None
        initial_log_activity = float(exact.log_activity_initial[idx]) if idx < len(exact.log_activity_initial) else None
        out["post_gas_only_activity_maxdensity_scan"][idx] = {
            "log_activity_exogibbs": initial_log_activity,
            "maxDensity_exogibbs": max_density,
            "number_density_exogibbs": 0.0,
            "lambda_exogibbs": 0.0,
            "chi_exogibbs": None,
        }
        out["post_selectActiveCondensates_reset"][idx] = {
            "log_activity_exogibbs": initial_log_activity,
            "maxDensity_exogibbs": max_density,
            "number_density_exogibbs": 0.0,
            "lambda_exogibbs": 0.0,
            "chi_exogibbs": None,
            "newly_active": idx in set(int(i) for i in exact.newly_active_indices),
        }
        lam_seed = float(exact.seed_lambda[local])
        out["post_calculate_entry_seeding"][idx] = {
            "log_activity_exogibbs": initial_log_activity,
            "maxDensity_exogibbs": max_density,
            "number_density_exogibbs": float(exact.seed_n[local]),
            "lambda_exogibbs": lam_seed,
            "chi_exogibbs": math.log(lam_seed) if lam_seed > 0.0 else None,
            "newly_active": idx in set(int(i) for i in exact.newly_active_indices),
        }
    first_rows = [row for row in phase2_trace.get("update_rows", []) if row.get("stage") == "first"]
    for row in first_rows:
        idx = int(row["condensate_index"])
        lam = float(row["lambda_after"])
        out["after_first_correctValues_update"][idx] = {
            "number_density_exogibbs": float(row["n_after"]),
            "lambda_exogibbs": lam,
            "chi_exogibbs": math.log(lam) if lam > 0.0 else None,
            "update_factor": float(row["update_factor"]),
            "cap_fired_exogibbs": bool(row.get("maxDensity_cap_fired", False)),
        }
    final_log_activity = phase2_trace.get("final_log_activity", [])
    final_n = phase2_trace.get("final_n", [])
    final_lambda = phase2_trace.get("final_lambda", [])
    removed = set(int(i) for i in phase2_trace.get("removed_indices", []))
    for local, idx in enumerate(candidate_indices):
        lam = float(final_lambda[local]) if local < len(final_lambda) else None
        out["post_final_removal"][idx] = {
            "log_activity_exogibbs": float(final_log_activity[local]) if local < len(final_log_activity) else None,
            "number_density_exogibbs": float(final_n[local]) if local < len(final_n) else None,
            "lambda_exogibbs": lam,
            "chi_exogibbs": math.log(lam) if lam is not None and lam > 0.0 else None,
            "removed": idx in removed,
            "d_elem_exogibbs": (formula_matrix_cond_np[:, idx] * float(final_n[local])).tolist() if local < len(final_n) else None,
        }
    return out


def _interesting_examples(rows_by_stage: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for stage, stage_rows in rows_by_stage.items():
        for row in stage_rows:
            name = str(row["condensate"])
            if any(token in name for token in EXAMPLE_SPECIES):
                rows.append({**row, "stage": stage})
    return rows


def _case_payload(
    *,
    layer: int,
    epsilon: float,
    temperature: float,
    pressure: float,
    state: ThermoState,
    gas_only: dict[str, Any],
    formula_matrix_gas: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    formula_matrix_cond_np: np.ndarray,
    hvector_gas: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    cond_species: Sequence[str],
    element_vector: np.ndarray,
    top_k: int,
    max_nfev: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _fc, _fc_output, records = transplant._run_instrumented_fastchem_variant(temperature, pressure)
    fastchem_split = reduced_recon.fastchem_split_from_trace(records)
    exact = exact_lifecycle._run_exact_lifecycle(
        variant=exact_lifecycle.VARIANTS["exact_maxdensity_lifecycle"],
        state=state,
        gas_only=gas_only,
        formula_matrix_gas=formula_matrix_gas,
        formula_matrix_cond=formula_matrix_cond,
        hvector_gas=hvector_gas,
        hvector_cond=hvector_cond,
        epsilon=epsilon,
        max_nfev=max_nfev,
        max_iter=3,
    )
    phase2_solve = phase2._phase2_update(
        variant=phase2.PHASE2_VARIANTS["update_map_alignment_phase2_reduced"],
        exact=exact,
        fastchem_split=fastchem_split,
        formula_matrix_gas=formula_matrix_gas,
        formula_matrix_cond=formula_matrix_cond,
        hvector_gas=hvector_gas,
        hvector_cond=hvector_cond,
        state=state,
        epsilon=epsilon,
    )
    exogibbs_by_stage = _exogibbs_stage_snapshots(
        exact=exact,
        phase2_trace=phase2_solve["trace"],
        formula_matrix_cond_np=formula_matrix_cond_np,
    )
    rows_by_stage = stage_aligned_rows(
        fastchem_records=records,
        exogibbs_by_stage=exogibbs_by_stage,
        cond_species=cond_species,
        top_k=top_k,
    )
    mapping_pairs = []
    for stage in ("post_calculate_entry_seeding", "after_first_correctValues_update"):
        for row in rows_by_stage.get(stage, []):
            if row["activity_correction_fastchem"] > 0.0 and row["lambda_exogibbs"] and row["lambda_exogibbs"] > 0.0:
                mapping_pairs.append(row)
    mapping = mapping_diagnostic_bookkeeping(mapping_pairs)
    seed_records = first_records_for_stage(records, "post_calculate_entry_seeding")
    atomic_density = np.asarray(
        jax.device_get(formula_matrix_gas @ jnp.exp(gas_only["ln_nk"])),
        dtype=np.float64,
    )
    maxdensity = maxdensity_formula_scores(
        fastchem_seed_records=seed_records,
        candidate_indices=exact.candidate_indices,
        formula_matrix_cond_np=formula_matrix_cond_np,
        element_vector=element_vector,
        atomic_density=atomic_density,
    )
    stage_scores = {
        stage: float(np.mean([row["stage_value_mismatch_score"] for row in rows])) if rows else 0.0
        for stage, rows in rows_by_stage.items()
    }
    top_stage = max(stage_scores, key=stage_scores.get) if stage_scores else "none"
    cap_score = sum(1 for row in rows_by_stage.get("after_first_correctValues_update", []) if row["cap_fired_fastchem"] != row["cap_fired_exogibbs"])
    split_j = fastchem_split.get("jac", [])
    split_r = fastchem_split.get("rem", [])
    split_overlap = reduced_recon.overlap(split_j, split_j) if split_j or split_r else {"jaccard": 1.0}
    if mapping["best_mapping"] != "activity_correction~lambda" and mapping["pair_count"] > 0:
        dominant = "variable mapping/sign mismatch"
        decision = "fix variable mapping first before any more semantic transplants"
    elif (
        maxdensity["supported_interpretation"].startswith("A") is False
        and maxdensity["supported_interpretation"].startswith("C") is False
        and maxdensity["supported_interpretation"].startswith("not observable") is False
    ):
        dominant = "maxDensity formula mismatch"
        decision = "fix maxDensity formula semantics first"
    elif cap_score > max(1, len(rows_by_stage.get("after_first_correctValues_update", [])) // 4):
        dominant = "cap timing mismatch"
        decision = "continue reduced/full reconstruction alignment next"
    else:
        dominant = "reduced/full split or reconstruction mismatch"
        decision = "continue reduced/full reconstruction alignment next"
    return {
        "layer_index": int(layer),
        "epsilon": float(epsilon),
        "temperature_K": float(temperature),
        "pressure_bar": float(pressure),
        "stage_aligned_side_by_side": rows_by_stage,
        "interesting_species_examples": _interesting_examples(rows_by_stage),
        "variable_mapping_diagnostic": mapping,
        "maxDensity_formula_diagnostic": maxdensity,
        "dominant_mismatch": dominant,
        "top_mismatch_stage": top_stage,
        "stage_mismatch_scores": stage_scores,
        "cap_timing_mismatch_count": int(cap_score),
        "reduced_full_split_diagnostic": {
            "fastchem_jac_count": len(fastchem_split.get("jac", [])),
            "fastchem_rem_count": len(fastchem_split.get("rem", [])),
            "self_overlap_smoke": split_overlap,
        },
        "case_decision": decision,
    }, {
        "layer_index": int(layer),
        "epsilon": float(epsilon),
        "fastchem_records": records,
        "exogibbs_by_stage": exogibbs_by_stage,
        "phase2_trace": phase2_solve["trace"],
        "candidate_indices": exact.candidate_indices,
    }


def _decision(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["dominant_mismatch"]] = counts.get(case["dominant_mismatch"], 0) + 1
    dominant = max(counts, key=counts.get) if counts else "unknown"
    if dominant == "variable mapping/sign mismatch":
        message = "fix variable mapping first before any more semantic transplants"
        next_target = "activity_correction/lambda sign and normalization"
    elif dominant == "maxDensity formula mismatch":
        message = "fix maxDensity formula semantics first"
        next_target = "literal maxDensity formula semantics"
    elif dominant == "reduced/full split or reconstruction mismatch" or dominant == "cap timing mismatch":
        message = "continue reduced/full reconstruction alignment next"
        next_target = (
            "post-update maxDensity cap timing inside reduced/full reconstruction"
            if dominant == "cap timing mismatch"
            else "reduced/full reconstruction alignment"
        )
    else:
        message = "move on to fixed_by_condensation / phi after mapping fix"
        next_target = "fixed_by_condensation / phi semantics"
    mapping_modes = [
        case["variable_mapping_diagnostic"]["best_mapping"]
        for case in cases
        if case["variable_mapping_diagnostic"]["pair_count"] > 0
    ]
    maxdensity_modes = [
        case["maxDensity_formula_diagnostic"]["supported_interpretation"]
        for case in cases
        if not case["maxDensity_formula_diagnostic"]["supported_interpretation"].startswith("not observable")
    ]
    stage_counts: dict[str, int] = {}
    for case in cases:
        stage_counts[case["top_mismatch_stage"]] = stage_counts.get(case["top_mismatch_stage"], 0) + 1
    return {
        "message": message,
        "dominant_mismatch_counts": counts,
        "best_mapping_by_case": mapping_modes,
        "maxDensity_interpretation_by_case": maxdensity_modes,
        "top_stage_counts": stage_counts,
        "inferred_variable_mapping": max(set(mapping_modes), key=mapping_modes.count) if mapping_modes else None,
        "literal_maxDensity_semantics_supported": max(set(maxdensity_modes), key=maxdensity_modes.count) if maxdensity_modes else None,
        "next_exact_transplant_target": next_target,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FastChem <-> ExoGibbs Variable-Mapping Micro-Parity Audit",
        "",
        f"Decision: **{payload['decision_summary']['message']}**",
        "",
        f"Inferred mapping: `{payload['decision_summary']['inferred_variable_mapping']}`",
        f"maxDensity interpretation: `{payload['decision_summary']['literal_maxDensity_semantics_supported']}`",
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## Layer {case['layer_index']} epsilon {case['epsilon']}",
                "",
                f"- Dominant mismatch: {case['dominant_mismatch']}",
                f"- Top mismatch stage: {case['top_mismatch_stage']}",
                f"- Mapping: {case['variable_mapping_diagnostic']['best_mapping']}",
                f"- maxDensity: {case['maxDensity_formula_diagnostic']['supported_interpretation']}",
                "",
            ]
        )
        for stage, label in TARGET_STAGE_SPECS:
            rows = case["stage_aligned_side_by_side"].get(stage, [])[:8]
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
                ]
            )
            for row in rows:
                lines.append(
                    f"| {row['condensate']} | {row['condensate_index']} | {row['log_activity_c']:.3g} | "
                    f"{row['maxDensity_c']:.3g} | {row['number_density_fastchem']:.3g} | "
                    f"{row['number_density_exogibbs'] if row['number_density_exogibbs'] is not None else 'NA'} | "
                    f"{row['activity_correction_fastchem']:.3g} | {row['lambda_exogibbs'] if row['lambda_exogibbs'] is not None else 'NA'} | "
                    f"{row['update_factor_fastchem'] if row['update_factor_fastchem'] is not None else 'NA'} | "
                    f"{row['update_factor_exogibbs'] if row['update_factor_exogibbs'] is not None else 'NA'} | "
                    f"{row['cap_fired_fastchem']} | {row['cap_fired_exogibbs']} | {row['newly_active']} | {row['removed']} |"
                )
            lines.append("")
        examples = case["interesting_species_examples"][:12]
        if examples:
            lines.extend(["### Candidate Species Examples", ""])
            for row in examples:
                lines.append(
                    f"- {row['condensate']} at {row['stage']}: n_FC={row['number_density_fastchem']:.3g}, "
                    f"n_EG={row['number_density_exogibbs']}, lambda_EG={row['lambda_exogibbs']}"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_branch_map(path_json: Path, path_md: Path, decision: dict[str, Any]) -> None:
    if path_json.exists():
        branch_map = json.loads(path_json.read_text(encoding="utf-8"))
    else:
        branch_map = branch_map_audit.generate_branch_map()
    branch_map["current_confirmed_mapping_assumptions"] = {
        "activity_correction_mapping": decision["inferred_variable_mapping"],
        "maxDensity_semantics": decision["literal_maxDensity_semantics_supported"],
    }
    branch_map["current_unconfirmed_mapping_assumptions"] = [
        "Whether chi should remain log(lambda) in every actual reduced/full branch.",
        "Whether cap timing differences appear only after unclipped exact correctValues reconstruction.",
    ]
    branch_map["next_exact_transplant_target"] = decision["next_exact_transplant_target"]
    path_json.write_text(json.dumps(to_python(branch_map), indent=2), encoding="utf-8")
    md = branch_map_audit.branch_map_to_markdown(to_python(branch_map))
    md += "\n## Current Variable-Mapping Assumptions\n\n"
    md += f"- Confirmed activity_correction mapping: `{decision['inferred_variable_mapping']}`\n"
    md += f"- Confirmed maxDensity semantics: `{decision['literal_maxDensity_semantics_supported']}`\n"
    md += "- Unconfirmed: whether chi should remain log(lambda) in every actual reduced/full branch.\n"
    md += "- Unconfirmed: whether cap timing differences appear only after unclipped exact correctValues reconstruction.\n"
    md += f"- Next exact transplant target: `{decision['next_exact_transplant_target']}`\n"
    path_md.write_text(md, encoding="utf-8")


def _print_table(cases: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    print(f"{'layer':>5} {'eps':>6} {'mapping':>31} {'maxDensity':>18} {'top_stage':>36} dominant")
    for case in cases:
        print(
            f"{case['layer_index']:5d} {case['epsilon']:6.1f} "
            f"{case['variable_mapping_diagnostic']['best_mapping'][:31]:>31} "
            f"{case['maxDensity_formula_diagnostic']['supported_interpretation'][:18]:>18} "
            f"{case['top_mismatch_stage'][:36]:>36} {case['dominant_mismatch']}"
        )
    print(f"decision: {decision['message']}")
    print(f"inferred_mapping: {decision['inferred_variable_mapping']}")
    print(f"maxDensity: {decision['literal_maxDensity_semantics_supported']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    case_specs = parse_cases(args.cases)
    temperatures_all, pressures_all = rebase._load_profile(args.profile_npz)
    gas_setup = gas_chemsetup(silent=True)
    cond_setup = cond_chemsetup(silent=True)
    aligned = build_aligned_abundance_vector(
        gas_setup.elements,
        source="fastchem_asplund_2020",
        element_file=rebase.FASTCHEM_TREE_ELEMENT_FILE,
        normalize=True,
    )
    element_vector_full = jnp.asarray(aligned.vector, dtype=jnp.float64)
    formula_matrix_gas, formula_matrix_cond, element_mask = contract_formula_matrix(
        gas_setup.formula_matrix,
        cond_setup.formula_matrix,
    )
    formula_matrix_gas = jnp.asarray(formula_matrix_gas, dtype=jnp.float64)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    formula_matrix_cond_np = np.asarray(jax.device_get(formula_matrix_cond), dtype=np.float64)
    element_vector = element_vector_full[element_mask]
    cond_species = list(cond_setup.species)
    cases = []
    traces = []
    with jax.default_device(jax.devices("cpu")[0]):
        for layer, epsilon in case_specs:
            T = float(temperatures_all[layer])
            P = float(pressures_all[layer])
            state = ThermoState(
                temperature=jnp.asarray(T, dtype=jnp.float64),
                ln_normalized_pressure=jnp.log(jnp.asarray(P, dtype=jnp.float64)),
                element_vector=element_vector,
            )
            hgas = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            hcond = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_only = rebase._gas_only(gas_setup, T, P, element_vector_full, max_iter=args.gas_max_iter)
            case, trace = _case_payload(
                layer=layer,
                epsilon=epsilon,
                temperature=T,
                pressure=P,
                state=state,
                gas_only=gas_only,
                formula_matrix_gas=formula_matrix_gas,
                formula_matrix_cond=formula_matrix_cond,
                formula_matrix_cond_np=formula_matrix_cond_np,
                hvector_gas=hgas,
                hvector_cond=hcond,
                cond_species=cond_species,
                element_vector=np.asarray(jax.device_get(element_vector), dtype=np.float64),
                top_k=args.top_k,
                max_nfev=args.least_squares_max_nfev,
            )
            cases.append(case)
            traces.append(trace)
    decision = _decision(cases)
    payload = {
        "timestamp_utc": current_timestamp_utc(),
        "diagnostic_only": True,
        "production_defaults_unchanged": True,
        "support_handling_paths_modified": False,
        "rgie_control_branch_untouched": True,
        "execution": {"platform": "cpu", "sequential": True, "runtime_seconds": time.perf_counter() - start},
        "target_stages": [{"stage": stage, "label": label} for stage, label in TARGET_STAGE_SPECS],
        "cases": cases,
        "decision_summary": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(payload), indent=2), encoding="utf-8")
    _write_markdown(args.md_output, to_python(payload))
    if args.update_branch_map:
        _update_branch_map(args.branch_map_json, args.branch_map_md, decision)
    _print_table(cases, decision)
    print(f"summary_json: {args.output}")
    print(f"summary_md: {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
