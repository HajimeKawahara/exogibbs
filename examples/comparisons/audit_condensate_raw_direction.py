"""Audit raw condensate directions for patched ExoGibbs condensate PIPM solvers.

This diagnostic keeps production behavior unchanged. It focuses on whether the
raw condensate direction is already degenerate before clipping, especially at
cold start, and whether raw or scalar-rescaled condensate directions can beat
the current clipped baseline on fresh residual.

Example:

    PYTHONPATH=src python examples/comparisons/audit_condensate_raw_direction.py --platform cpu
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.common import current_timestamp_utc
from benchmarks.common import device_for_platform
from benchmarks.common import to_python
from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.pipm_gie_cond import _compute_iteration_step_metrics as compute_gie_step_metrics
from exogibbs.optimize.pipm_gie_cond import _compute_residuals as compute_gie_residuals
from exogibbs.optimize.pipm_gie_cond import _contains_invalid_numbers as gie_contains_invalid_numbers
from exogibbs.optimize.pipm_gie_cond import _evaluate_trial_step as evaluate_gie_trial_step
from exogibbs.optimize.pipm_gie_cond import _recompute_pi_for_residual as recompute_gie_pi
from exogibbs.optimize.pipm_gie_cond import _update_all_with_metrics as gie_update_all_with_metrics
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_MODE
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_STRENGTH
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REDUCED_SOLVER
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics as compute_rgie_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals as compute_rgie_residuals
from exogibbs.optimize.pipm_rgie_cond import _contains_invalid_numbers as rgie_contains_invalid_numbers
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step as evaluate_rgie_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual as recompute_rgie_pi
from exogibbs.optimize.pipm_rgie_cond import _update_all_with_metrics as rgie_update_all_with_metrics
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_M0_VALUES = (1.0e-30, 1.0e-20, 1.0e-10, 1.0e-6)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_VARIANTS = (
    "current_component_clip_0p1",
    "cond_block_scalar_rescale_0p1",
    "cond_block_scalar_rescale_0p5",
    "raw_no_clip",
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "condensate_raw_direction_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "condensate_raw_direction_traces.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYER_INDICES))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--m0-values", type=float, nargs="+", default=list(DEFAULT_M0_VALUES))
    parser.add_argument("--lambda-trials", type=float, nargs="+", default=list(DEFAULT_LAMBDA_TRIALS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_OUTPUT)
    parser.add_argument(
        "--include-post-state",
        action="store_true",
        help="Also evaluate one post-first-iteration baseline state per case.",
    )
    parser.add_argument(
        "--include-trajectory",
        action="store_true",
        help="Run a tiny trajectory ablation. Disabled by default because one-step diagnostics are primary.",
    )
    parser.add_argument("--trajectory-max-iter", type=int, default=3)
    parser.add_argument("--gas-max-iter", type=int, default=1000)
    return parser


def _load_sparse_support_case(layer_index: int) -> dict[str, Any]:
    path = REPO_ROOT / "results" / f"sparse_layer{layer_index}_top1.json"
    payload = json.loads(path.read_text())
    layer_result = payload["results"][0]
    support_cfg = layer_result["configs"]["sparse_lp20_top1"]
    return {
        "temperature_K": float(payload["benchmark"]["temperature_K"]),
        "pressure_bar": float(layer_result["pressure_bar"]),
        "support_names": list(support_cfg["final_support_names"]),
        "support_source_json": str(path),
    }


def _build_profile_states(element_vector: jax.Array, layer_indices: list[int]) -> list[dict[str, Any]]:
    states = []
    for layer_index in layer_indices:
        support_case = _load_sparse_support_case(layer_index)
        state = ThermoState(
            temperature=jnp.asarray(support_case["temperature_K"], dtype=jnp.float64),
            ln_normalized_pressure=jnp.log(jnp.asarray(support_case["pressure_bar"], dtype=jnp.float64)),
            element_vector=jnp.asarray(element_vector, dtype=jnp.float64),
        )
        states.append(
            {
                "layer_index": int(layer_index),
                "temperature_K": support_case["temperature_K"],
                "pressure_bar": support_case["pressure_bar"],
                "support_names": support_case["support_names"],
                "support_source_json": support_case["support_source_json"],
                "state": state,
            }
        )
    return states


def _build_shared_gas_init(
    state: ThermoState,
    gas_setup: Any,
    gas_hvector: jax.Array,
    gas_max_iter: int,
) -> dict[str, Any]:
    ln_nk_init0 = jnp.zeros((gas_setup.formula_matrix.shape[1],), dtype=jnp.float64)
    ln_ntot_init0 = jnp.asarray(0.0, dtype=jnp.float64)
    ln_nk_gas, ln_ntot_gas, gas_n_iter, gas_final_residual = minimize_gibbs_core(
        state,
        ln_nk_init0,
        ln_ntot_init0,
        gas_setup.formula_matrix,
        lambda _temperature: gas_hvector,
        epsilon_crit=1.0e-12,
        max_iter=gas_max_iter,
    )
    return {
        "ln_nk": jnp.asarray(ln_nk_gas, dtype=jnp.float64),
        "ln_ntot": jnp.asarray(ln_ntot_gas, dtype=jnp.float64),
        "gas_init_n_iter": int(gas_n_iter),
        "gas_init_final_residual": float(gas_final_residual),
    }


def _safe_cosine(a: jax.Array, b: jax.Array) -> float | None:
    a = jnp.ravel(jnp.asarray(a, dtype=jnp.float64))
    b = jnp.ravel(jnp.asarray(b, dtype=jnp.float64))
    denom = float(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    if not math.isfinite(denom) or denom <= 1.0e-300:
        return None
    cosine = float(jnp.dot(a, b) / denom)
    return max(-1.0, min(1.0, cosine))


def _safe_angle_degrees(a: jax.Array, b: jax.Array) -> float | None:
    cosine = _safe_cosine(a, b)
    if cosine is None:
        return None
    return float(math.degrees(math.acos(cosine)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _fraction(mask: list[bool]) -> float | None:
    if not mask:
        return None
    return float(sum(1 for value in mask if value) / len(mask))


def _summary_triplet(array: jax.Array) -> dict[str, float]:
    arr = jnp.ravel(jnp.asarray(array, dtype=jnp.float64))
    return {
        "min": float(jnp.min(arr)),
        "median": float(jnp.median(arr)),
        "max": float(jnp.max(arr)),
    }


def _sign_histogram(array: jax.Array, near_zero_tol: float = 1.0e-12) -> dict[str, float]:
    arr = jnp.ravel(jnp.asarray(array, dtype=jnp.float64))
    near_zero = jnp.abs(arr) <= near_zero_tol
    positive = arr > near_zero_tol
    negative = arr < -near_zero_tol
    size = float(arr.shape[0]) if arr.shape[0] else 1.0
    return {
        "positive_fraction": float(jnp.sum(positive) / size),
        "near_zero_fraction": float(jnp.sum(near_zero) / size),
        "negative_fraction": float(jnp.sum(negative) / size),
    }


def _variant_direction(raw_delta_ln_mk: jax.Array, variant_name: str) -> dict[str, Any]:
    raw = jnp.asarray(raw_delta_ln_mk, dtype=jnp.float64)
    max_abs_raw = float(jnp.max(jnp.abs(raw)))
    if variant_name == "current_component_clip_0p1":
        delta = jnp.clip(raw, -0.1, 0.1)
        alpha = None
    elif variant_name == "cond_block_scalar_rescale_0p1":
        alpha = 1.0 if max_abs_raw <= 0.1 else 0.1 / max_abs_raw
        delta = raw * alpha
    elif variant_name == "cond_block_scalar_rescale_0p5":
        alpha = 1.0 if max_abs_raw <= 0.5 else 0.5 / max_abs_raw
        delta = raw * alpha
    elif variant_name == "raw_no_clip":
        delta = raw
        alpha = None
    else:
        raise ValueError(f"Unknown variant: {variant_name}")
    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta,
        "scalar_alpha": alpha,
        "max_abs_variant_delta_ln_mk": float(jnp.max(jnp.abs(delta))),
        "cosine_raw_vs_variant": _safe_cosine(raw, delta),
        "angle_degrees_raw_vs_variant": _safe_angle_degrees(raw, delta),
        "variant_cond_direction_norm": float(jnp.linalg.norm(delta)),
    }


def _solver_bundle(name: str) -> dict[str, Any]:
    if name == "RGIE":
        extra = {
            "reduced_solver": DEFAULT_REDUCED_SOLVER,
            "regularization_mode": DEFAULT_REGULARIZATION_MODE,
            "regularization_strength": DEFAULT_REGULARIZATION_STRENGTH,
        }
        return {
            "name": name,
            "compute_step_metrics": compute_rgie_step_metrics,
            "evaluate_trial_step": evaluate_rgie_trial_step,
            "recompute_pi": recompute_rgie_pi,
            "compute_residuals": compute_rgie_residuals,
            "contains_invalid_numbers": rgie_contains_invalid_numbers,
            "update_all_with_metrics": rgie_update_all_with_metrics,
            "extra_kwargs": extra,
        }
    if name == "GIE":
        return {
            "name": name,
            "compute_step_metrics": compute_gie_step_metrics,
            "evaluate_trial_step": evaluate_gie_trial_step,
            "recompute_pi": recompute_gie_pi,
            "compute_residuals": compute_gie_residuals,
            "contains_invalid_numbers": gie_contains_invalid_numbers,
            "update_all_with_metrics": gie_update_all_with_metrics,
            "extra_kwargs": {},
        }
    raise ValueError(f"Unknown solver bundle: {name}")


def _compute_current_state_metrics(
    bundle: dict[str, Any],
    *,
    state: ThermoState,
    ln_nk: jax.Array,
    ln_mk: jax.Array,
    ln_ntot: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
) -> dict[str, Any]:
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    An = formula_matrix @ nk
    Am = formula_matrix_cond @ mk
    invalid_numbers_detected = bool(
        bundle["contains_invalid_numbers"](ln_nk, ln_mk, ln_ntot, nk, mk, ntot, gk, An, Am)
    )
    if invalid_numbers_detected:
        residual = jnp.asarray(jnp.inf, dtype=jnp.float64)
        pi_vector = jnp.full_like(b, jnp.nan)
    else:
        pi_vector = bundle["recompute_pi"](
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            **bundle["extra_kwargs"],
        )
        residual = bundle["compute_residuals"](
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            jnp.exp(epsilon),
            An,
            Am,
            pi_vector,
        )
        if not bool(jnp.isfinite(residual)):
            residual = jnp.asarray(jnp.inf, dtype=jnp.float64)
    return {
        "gk": gk,
        "An": An,
        "Am": Am,
        "nk": nk,
        "mk": mk,
        "ntot": ntot,
        "pi_vector": pi_vector,
        "residual": residual,
    }


def _rgie_raw_anatomy(
    step_metrics: dict[str, Any],
    ln_mk: jax.Array,
    formula_matrix_cond: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
) -> dict[str, Any]:
    factor = jnp.exp(jnp.asarray(ln_mk) - epsilon)
    correction = factor * (formula_matrix_cond.T @ step_metrics["pi_vector"] - hvector_cond)
    raw_delta_ln_mk = 1.0 + correction
    ones = jnp.ones_like(raw_delta_ln_mk)
    return {
        "factor": to_python(jax.device_get(factor)),
        "correction": to_python(jax.device_get(correction)),
        "raw_delta_ln_mk": to_python(jax.device_get(raw_delta_ln_mk)),
        "factor_stats": _summary_triplet(factor),
        "correction_stats": _summary_triplet(correction),
        "raw_stats": _summary_triplet(raw_delta_ln_mk),
        "fraction_abs_correction_lt_1e-12": float(jnp.mean((jnp.abs(correction) < 1.0e-12).astype(jnp.float64))),
        "fraction_abs_correction_lt_1e-9": float(jnp.mean((jnp.abs(correction) < 1.0e-9).astype(jnp.float64))),
        "fraction_abs_correction_lt_1e-6": float(jnp.mean((jnp.abs(correction) < 1.0e-6).astype(jnp.float64))),
        "fraction_raw_in_0p99_1p01": float(
            jnp.mean(((raw_delta_ln_mk >= 0.99) & (raw_delta_ln_mk <= 1.01)).astype(jnp.float64))
        ),
        "fraction_raw_in_0p9_1p1": float(
            jnp.mean(((raw_delta_ln_mk >= 0.9) & (raw_delta_ln_mk <= 1.1)).astype(jnp.float64))
        ),
        "cosine_raw_vs_ones": _safe_cosine(raw_delta_ln_mk, ones),
        "raw_cond_direction_norm": float(jnp.linalg.norm(raw_delta_ln_mk)),
        "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(raw_delta_ln_mk))),
        "sign_histogram": _sign_histogram(raw_delta_ln_mk),
    }


def _generic_raw_summary(raw_delta_ln_mk: jax.Array) -> dict[str, Any]:
    ones = jnp.ones_like(raw_delta_ln_mk)
    return {
        "raw_stats": _summary_triplet(raw_delta_ln_mk),
        "fraction_raw_in_0p99_1p01": float(
            jnp.mean(((raw_delta_ln_mk >= 0.99) & (raw_delta_ln_mk <= 1.01)).astype(jnp.float64))
        ),
        "fraction_raw_in_0p9_1p1": float(
            jnp.mean(((raw_delta_ln_mk >= 0.9) & (raw_delta_ln_mk <= 1.1)).astype(jnp.float64))
        ),
        "cosine_raw_vs_ones": _safe_cosine(raw_delta_ln_mk, ones),
        "raw_cond_direction_norm": float(jnp.linalg.norm(raw_delta_ln_mk)),
        "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(raw_delta_ln_mk))),
        "sign_histogram": _sign_histogram(raw_delta_ln_mk),
    }


def _evaluate_variant_lambda_grid(
    bundle: dict[str, Any],
    *,
    state: ThermoState,
    ln_nk: jax.Array,
    ln_mk: jax.Array,
    ln_ntot: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
    current_residual: float,
    delta_ln_nk: jax.Array,
    delta_ln_mk: jax.Array,
    delta_ln_ntot: float,
    lambda_trials: list[float],
) -> dict[str, Any]:
    trials = []
    best = None
    any_improve = False
    for lam in lambda_trials:
        trial = bundle["evaluate_trial_step"](
            ln_nk,
            ln_mk,
            ln_ntot,
            jnp.asarray(lam, dtype=jnp.float64),
            delta_ln_nk,
            delta_ln_mk,
            delta_ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
            **bundle["extra_kwargs"],
        )
        fresh_residual = float(trial["fresh_residual"])
        valid_trial = bool(trial["all_finite"]) and math.isfinite(fresh_residual)
        sk_feasible = bool(jnp.all(2.0 * trial["ln_mk"] - epsilon <= LOG_S_MAX + 1.0e-12))
        record = {
            "lam": float(lam),
            "fresh_residual": fresh_residual,
            "valid_trial": valid_trial,
            "sk_feasible": sk_feasible,
            "max_abs_trial_delta_ln_mk": float(abs(lam) * jnp.max(jnp.abs(delta_ln_mk))),
        }
        trials.append(record)
        if valid_trial and (best is None or record["fresh_residual"] < best["fresh_residual"]):
            best = record
        if valid_trial and record["fresh_residual"] <= float(current_residual) + 1.0e-15:
            any_improve = True

    valid_count = sum(1 for record in trials if record["valid_trial"])
    invalid_count = len(trials) - valid_count
    return {
        "lambda_trials": trials,
        "valid_trial_count": int(valid_count),
        "invalid_trial_count": int(invalid_count),
        "best_lambda": None if best is None else best["lam"],
        "best_fresh_residual": None if best is None else best["fresh_residual"],
        "any_trial_improves_residual": bool(any_improve),
        "best_trial_sk_feasible": None if best is None else bool(best["sk_feasible"]),
        "max_abs_trial_delta_ln_mk": 0.0 if best is None else best["max_abs_trial_delta_ln_mk"],
    }


def _build_post_first_iteration_state(
    bundle: dict[str, Any],
    *,
    state: ThermoState,
    ln_nk: jax.Array,
    ln_mk: jax.Array,
    ln_ntot: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
) -> dict[str, Any]:
    current = _compute_current_state_metrics(
        bundle,
        state=state,
        ln_nk=ln_nk,
        ln_mk=ln_mk,
        ln_ntot=ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        b=b,
        hvector=hvector,
        hvector_cond=hvector_cond,
        epsilon=epsilon,
    )
    result = bundle["update_all_with_metrics"](
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        state.temperature,
        state.ln_normalized_pressure,
        hvector,
        hvector_cond,
        current["gk"],
        current["An"],
        current["Am"],
        current["residual"],
        epsilon,
        0,
        False,
        **bundle["extra_kwargs"],
    )
    post_ln_nk = result[0]
    post_ln_mk = result[1]
    post_ln_ntot = result[2]
    post_step_metrics = bundle["compute_step_metrics"](
        post_ln_nk,
        post_ln_mk,
        post_ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        result[3],
        hvector_cond,
        epsilon,
        **bundle["extra_kwargs"],
    )
    if bundle["name"] == "RGIE":
        post_raw_anatomy = _rgie_raw_anatomy(
            post_step_metrics,
            post_ln_mk,
            formula_matrix_cond,
            hvector_cond,
            epsilon,
        )
    else:
        post_raw_anatomy = _generic_raw_summary(jnp.asarray(post_step_metrics["raw_delta_ln_mk"]))
    return {
        "ln_nk": result[0],
        "ln_mk": result[1],
        "ln_ntot": result[2],
        "residual": float(result[6]),
        "metrics": result[8],
        "raw_anatomy": post_raw_anatomy,
    }


def _run_tiny_trajectory(
    bundle: dict[str, Any],
    *,
    state: ThermoState,
    ln_nk_init: jax.Array,
    ln_mk_init: jax.Array,
    ln_ntot_init: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
    max_iter: int,
) -> list[dict[str, Any]]:
    variants = ("current_component_clip_0p1", "cond_block_scalar_rescale_0p1", "raw_no_clip")
    runs = []
    for variant_name in variants:
        ln_nk = jnp.asarray(ln_nk_init)
        ln_mk = jnp.asarray(ln_mk_init)
        ln_ntot = jnp.asarray(ln_ntot_init)
        current = _compute_current_state_metrics(
            bundle,
            state=state,
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            b=b,
            hvector=hvector,
            hvector_cond=hvector_cond,
            epsilon=epsilon,
        )
        history = []
        for iter_count in range(max_iter):
            step_metrics = bundle["compute_step_metrics"](
                ln_nk,
                ln_mk,
                ln_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                current["gk"],
                hvector_cond,
                epsilon,
                **bundle["extra_kwargs"],
            )
            variant = _variant_direction(step_metrics["raw_delta_ln_mk"], variant_name)
            evals = _evaluate_variant_lambda_grid(
                bundle,
                state=state,
                ln_nk=ln_nk,
                ln_mk=ln_mk,
                ln_ntot=ln_ntot,
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                b=b,
                hvector=hvector,
                hvector_cond=hvector_cond,
                epsilon=epsilon,
                current_residual=current["residual"],
                delta_ln_nk=step_metrics["delta_ln_nk"],
                delta_ln_mk=variant["delta_ln_mk"],
                delta_ln_ntot=step_metrics["delta_ln_ntot"],
                lambda_trials=[1.0, 0.5, 0.2, 0.1, 0.05],
            )
            history.append(
                {
                    "iter": iter_count,
                    "residual_before": float(current["residual"]),
                    "best_fresh_residual": evals["best_fresh_residual"],
                    "any_trial_improves_residual": evals["any_trial_improves_residual"],
                }
            )
            break
        runs.append(
            {
                "variant": variant_name,
                "history": history,
            }
        )
    return runs


def _print_terminal_table(summary_rows: list[dict[str, Any]]) -> None:
    headers = ("solver", "m0", "cases", "degenerate", "raw>clip", "scalar>clip", "raw_cos")
    rows = [headers]
    for row in summary_rows:
        rows.append(
            (
                row["solver"],
                f"{row['m0']:.0e}",
                str(row["n_cases"]),
                "-" if row["startup_degenerate_fraction"] is None else f"{row['startup_degenerate_fraction']:.2f}",
                "-" if row["raw_no_clip_beats_clip_fraction"] is None else f"{row['raw_no_clip_beats_clip_fraction']:.2f}",
                "-" if row["scalar_beats_clip_fraction"] is None else f"{row['scalar_beats_clip_fraction']:.2f}",
                "-" if row["mean_raw_vs_ones_cosine"] is None else f"{row['mean_raw_vs_ones_cosine']:.3f}",
            )
        )
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    for idx, row in enumerate(rows):
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def _decision_summary(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    startup_degenerate = []
    raw_beats_clip = []
    scalar_beats_clip = []
    raw_cross_solver_cosines = []
    for case in case_records:
        rgie_anatomy = case["RGIE"]["raw_anatomy"]
        startup_degenerate.append(rgie_anatomy["fraction_raw_in_0p99_1p01"] >= 0.8)
        for solver_name in ("RGIE", "GIE"):
            variants = case[solver_name]["variant_trials"]
            baseline = variants["current_component_clip_0p1"]["best_fresh_residual"]
            raw_best = variants["raw_no_clip"]["best_fresh_residual"]
            scalar_best = variants["cond_block_scalar_rescale_0p1"]["best_fresh_residual"]
            if baseline is not None and raw_best is not None:
                raw_beats_clip.append(raw_best < baseline)
            if baseline is not None and scalar_best is not None:
                scalar_beats_clip.append(scalar_best < baseline)
        cosine = case["cross_solver"]["raw_cond_cosine"]
        if cosine is not None:
            raw_cross_solver_cosines.append(cosine)

    startup_degenerate_fraction = _fraction(startup_degenerate)
    raw_beats_clip_fraction = _fraction(raw_beats_clip)
    scalar_beats_clip_fraction = _fraction(scalar_beats_clip)
    raw_cross_solver_high = _fraction([cosine >= 0.999 for cosine in raw_cross_solver_cosines])

    messages = []
    if startup_degenerate_fraction is not None and startup_degenerate_fraction >= 0.8:
        messages.append("raw condensate direction is startup-degenerate before clipping")
    if (
        raw_beats_clip_fraction is not None
        and scalar_beats_clip_fraction is not None
        and max(raw_beats_clip_fraction, scalar_beats_clip_fraction) >= 0.3
    ):
        messages.append("clipping still destroys useful raw direction information")
    if (
        raw_beats_clip_fraction is not None
        and scalar_beats_clip_fraction is not None
        and max(raw_beats_clip_fraction, scalar_beats_clip_fraction) < 0.1
    ):
        messages.append("the main bottleneck is raw direction construction, not clipping")
    if raw_cross_solver_high is not None and raw_cross_solver_high >= 0.8:
        messages.append("full-vs-reduced is not the primary distinction at the raw condensate direction level")
    return {
        "startup_degenerate_fraction": startup_degenerate_fraction,
        "raw_no_clip_beats_clip_fraction": raw_beats_clip_fraction,
        "scalar_beats_clip_fraction": scalar_beats_clip_fraction,
        "raw_cross_solver_high_cosine_fraction": raw_cross_solver_high,
        "messages": messages,
        "recommended_next_move": (
            "change clipping policy"
            if scalar_beats_clip_fraction is not None and scalar_beats_clip_fraction >= 0.3
            else "inspect raw direction construction itself"
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    backend, device = device_for_platform(args.platform)
    lambda_trials = [float(value) for value in args.lambda_trials]
    with jax.default_device(device):
        gas_setup = gas_chemsetup(silent=True)
        cond_setup = cond_chemsetup(silent=True)
        profile_states = _build_profile_states(gas_setup.element_vector_reference, args.layers)
        case_records = []
        trace_records = []

        for layer_meta in profile_states:
            state = layer_meta["state"]
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
            support_indices = jnp.asarray(
                [cond_setup.species.index(name) for name in layer_meta["support_names"]],
                dtype=jnp.int32,
            )
            formula_matrix = gas_setup.formula_matrix
            formula_matrix_cond = cond_setup.formula_matrix[:, support_indices]
            hvector_cond = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
            b = jnp.asarray(state.element_vector)

            for epsilon in args.epsilons:
                for m0 in args.m0_values:
                    ln_nk = gas_init["ln_nk"]
                    ln_ntot = gas_init["ln_ntot"]
                    ln_mk = jnp.full((formula_matrix_cond.shape[1],), jnp.log(m0), dtype=jnp.float64)
                    solver_payload = {}
                    for solver_name in ("RGIE", "GIE"):
                        bundle = _solver_bundle(solver_name)
                        current = _compute_current_state_metrics(
                            bundle,
                            state=state,
                            ln_nk=ln_nk,
                            ln_mk=ln_mk,
                            ln_ntot=ln_ntot,
                            formula_matrix=formula_matrix,
                            formula_matrix_cond=formula_matrix_cond,
                            b=b,
                            hvector=gas_hvector,
                            hvector_cond=hvector_cond,
                            epsilon=epsilon,
                        )
                        step_metrics = bundle["compute_step_metrics"](
                            ln_nk,
                            ln_mk,
                            ln_ntot,
                            formula_matrix,
                            formula_matrix_cond,
                            b,
                            current["gk"],
                            hvector_cond,
                            epsilon,
                            **bundle["extra_kwargs"],
                        )
                        raw_delta_ln_mk = jnp.asarray(step_metrics["raw_delta_ln_mk"], dtype=jnp.float64)
                        if solver_name == "RGIE":
                            raw_anatomy = _rgie_raw_anatomy(
                                step_metrics,
                                ln_mk,
                                formula_matrix_cond,
                                hvector_cond,
                                epsilon,
                            )
                        else:
                            raw_anatomy = _generic_raw_summary(raw_delta_ln_mk)
                        variant_trials = {}
                        for variant_name in DEFAULT_VARIANTS:
                            variant = _variant_direction(raw_delta_ln_mk, variant_name)
                            trials = _evaluate_variant_lambda_grid(
                                bundle,
                                state=state,
                                ln_nk=ln_nk,
                                ln_mk=ln_mk,
                                ln_ntot=ln_ntot,
                                formula_matrix=formula_matrix,
                                formula_matrix_cond=formula_matrix_cond,
                                b=b,
                                hvector=gas_hvector,
                                hvector_cond=hvector_cond,
                                epsilon=epsilon,
                                current_residual=current["residual"],
                                delta_ln_nk=step_metrics["delta_ln_nk"],
                                delta_ln_mk=variant["delta_ln_mk"],
                                delta_ln_ntot=step_metrics["delta_ln_ntot"],
                                lambda_trials=lambda_trials,
                            )
                            variant_trials[variant_name] = {
                                **variant,
                                **trials,
                            }
                        post_state = None
                        if args.include_post_state and m0 == min(args.m0_values):
                            post_state = _build_post_first_iteration_state(
                                bundle,
                                state=state,
                                ln_nk=ln_nk,
                                ln_mk=ln_mk,
                                ln_ntot=ln_ntot,
                                formula_matrix=formula_matrix,
                                formula_matrix_cond=formula_matrix_cond,
                                b=b,
                                hvector=gas_hvector,
                                hvector_cond=hvector_cond,
                                epsilon=epsilon,
                            )
                        tiny_trajectory = None
                        if args.include_trajectory and m0 == min(args.m0_values):
                            tiny_trajectory = _run_tiny_trajectory(
                                bundle,
                                state=state,
                                ln_nk_init=ln_nk,
                                ln_mk_init=ln_mk,
                                ln_ntot_init=ln_ntot,
                                formula_matrix=formula_matrix,
                                formula_matrix_cond=formula_matrix_cond,
                                b=b,
                                hvector=gas_hvector,
                                hvector_cond=hvector_cond,
                                epsilon=epsilon,
                                max_iter=args.trajectory_max_iter,
                            )
                        solver_payload[solver_name] = {
                            "current_residual": float(current["residual"]),
                            "delta_ln_nk": to_python(jax.device_get(step_metrics["delta_ln_nk"])),
                            "delta_ln_ntot": float(step_metrics["delta_ln_ntot"]),
                            "raw_delta_ln_mk": to_python(jax.device_get(raw_delta_ln_mk)),
                            "raw_anatomy": raw_anatomy,
                            "variant_trials": variant_trials,
                            "post_first_iteration_state": None if post_state is None else {
                                "residual": post_state["residual"],
                                "lam_selected": float(post_state["metrics"]["lam_selected"]),
                                "line_search_accept_kind": post_state["metrics"]["line_search_accept_kind"],
                                "raw_anatomy": post_state["raw_anatomy"],
                            },
                            "tiny_trajectory": tiny_trajectory,
                        }

                    rgie_raw = jnp.asarray(solver_payload["RGIE"]["raw_delta_ln_mk"], dtype=jnp.float64)
                    gie_raw = jnp.asarray(solver_payload["GIE"]["raw_delta_ln_mk"], dtype=jnp.float64)
                    rgie_gas = jnp.asarray(solver_payload["RGIE"]["delta_ln_nk"], dtype=jnp.float64)
                    gie_gas = jnp.asarray(solver_payload["GIE"]["delta_ln_nk"], dtype=jnp.float64)
                    cross_solver = {
                        "raw_cond_cosine": _safe_cosine(rgie_raw, gie_raw),
                        "raw_cond_angle_degrees": _safe_angle_degrees(rgie_raw, gie_raw),
                        "max_abs_raw_cond_diff": float(jnp.max(jnp.abs(rgie_raw - gie_raw))),
                        "gas_cosine": _safe_cosine(rgie_gas, gie_gas),
                        "gas_angle_degrees": _safe_angle_degrees(rgie_gas, gie_gas),
                        "max_abs_gas_diff": float(jnp.max(jnp.abs(rgie_gas - gie_gas))),
                    }

                    for solver_name in ("RGIE", "GIE"):
                        variants = solver_payload[solver_name]["variant_trials"]
                        baseline = variants["current_component_clip_0p1"]["best_fresh_residual"]
                        raw_best = variants["raw_no_clip"]["best_fresh_residual"]
                        scalar_candidates = [
                            value
                            for value in (
                                variants["cond_block_scalar_rescale_0p1"]["best_fresh_residual"],
                                variants["cond_block_scalar_rescale_0p5"]["best_fresh_residual"],
                            )
                            if value is not None
                        ]
                        scalar_best = None if not scalar_candidates else min(scalar_candidates)
                        variants["raw_no_clip"]["beats_current_component_clip_0p1"] = (
                            raw_best is not None and baseline is not None and raw_best < baseline
                        )
                        variants["cond_block_scalar_rescale_0p1"]["scalar_rescale_beats_current_component_clip_0p1"] = (
                            scalar_best is not None and baseline is not None and scalar_best < baseline
                        )
                        variants["cond_block_scalar_rescale_0p5"]["scalar_rescale_beats_current_component_clip_0p1"] = (
                            scalar_best is not None and baseline is not None and scalar_best < baseline
                        )

                    case_record = {
                        "layer_index": layer_meta["layer_index"],
                        "temperature_K": layer_meta["temperature_K"],
                        "pressure_bar": layer_meta["pressure_bar"],
                        "support_names": list(layer_meta["support_names"]),
                        "epsilon": float(epsilon),
                        "m0": float(m0),
                        "gas_init_n_iter": gas_init["gas_init_n_iter"],
                        "gas_init_final_residual": gas_init["gas_init_final_residual"],
                        "RGIE": solver_payload["RGIE"],
                        "GIE": solver_payload["GIE"],
                        "cross_solver": cross_solver,
                    }
                    case_records.append(case_record)
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "m0": float(m0),
                            "RGIE_variant_trials": solver_payload["RGIE"]["variant_trials"],
                            "GIE_variant_trials": solver_payload["GIE"]["variant_trials"],
                        }
                    )

        summary_rows = []
        for solver_name in ("RGIE", "GIE"):
            for m0 in args.m0_values:
                matching = [record for record in case_records if record["m0"] == float(m0)]
                startup_deg = []
                raw_beats = []
                scalar_beats = []
                raw_vs_ones = []
                for record in matching:
                    anatomy = record[solver_name]["raw_anatomy"]
                    startup_deg.append(anatomy["fraction_raw_in_0p99_1p01"] >= 0.8)
                    raw_beats.append(
                        record[solver_name]["variant_trials"]["raw_no_clip"]["beats_current_component_clip_0p1"]
                    )
                    scalar_beats.append(
                        record[solver_name]["variant_trials"]["cond_block_scalar_rescale_0p1"][
                            "scalar_rescale_beats_current_component_clip_0p1"
                        ]
                    )
                    raw_vs_ones.append(anatomy["cosine_raw_vs_ones"])
                finite_cosines = [value for value in raw_vs_ones if value is not None]
                summary_rows.append(
                    {
                        "solver": solver_name,
                        "m0": float(m0),
                        "n_cases": len(matching),
                        "startup_degenerate_fraction": _fraction(startup_deg),
                        "raw_no_clip_beats_clip_fraction": _fraction(raw_beats),
                        "scalar_beats_clip_fraction": _fraction(scalar_beats),
                        "mean_raw_vs_ones_cosine": _mean(finite_cosines),
                    }
                )

        decision = _decision_summary(case_records)
        audit_payload = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": list(args.layers),
            "epsilons": list(args.epsilons),
            "m0_values": list(args.m0_values),
            "lambda_trials": lambda_trials,
            "include_post_state": bool(args.include_post_state),
            "include_trajectory": bool(args.include_trajectory),
            "cases": case_records,
            "summary_rows": summary_rows,
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "variant_trial_traces": trace_records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(audit_payload), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2))

    _print_terminal_table(summary_rows)
    print()
    for message in decision["messages"]:
        print(f"decision: {message}")
    print(f"next_move: {decision['recommended_next_move']}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
