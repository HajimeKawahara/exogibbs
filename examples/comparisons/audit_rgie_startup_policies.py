"""Audit RGIE startup policies through raw-direction anatomy and short trajectories.

This diagnostic is RGIE-only. It keeps the current RGIE raw direction formula,
componentwise clipping, and residual-based line search unchanged for the
baseline trajectory path while varying only the diagnostic startup policy used
to construct the initial supported condensate amounts.

Example:

    PYTHONPATH=src python examples/comparisons/audit_rgie_startup_policies.py --platform cpu
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
config.update("jax_disable_jit", True)

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
from exogibbs.optimize.minimize import solve_gibbs_iteration_equations
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_MODE
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_STRENGTH
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REDUCED_SOLVER
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_iterations
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_startup_policy_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_startup_policy_traces.json"
DEFAULT_TINY_FALLBACK = 1.0e-30
DEFAULT_SHORT_MAX_ITER = 7

POLICIES = (
    {
        "name": "absolute_uniform_m0_1e-30",
        "startup_policy": "absolute_uniform_m0",
        "kwargs": {"m0": 1.0e-30},
        "family": "absolute_uniform",
    },
    {
        "name": "absolute_uniform_m0_1e-20",
        "startup_policy": "absolute_uniform_m0",
        "kwargs": {"m0": 1.0e-20},
        "family": "absolute_uniform",
    },
    {
        "name": "absolute_uniform_m0_1e-10",
        "startup_policy": "absolute_uniform_m0",
        "kwargs": {"m0": 1.0e-10},
        "family": "absolute_uniform",
    },
    {
        "name": "absolute_uniform_m0_1e-6",
        "startup_policy": "absolute_uniform_m0",
        "kwargs": {"m0": 1.0e-6},
        "family": "absolute_uniform",
    },
    {
        "name": "ratio_uniform_r0_1e-12",
        "startup_policy": "ratio_uniform_r0",
        "kwargs": {"r0": 1.0e-12},
        "family": "ratio_uniform",
    },
    {
        "name": "ratio_uniform_r0_1e-9",
        "startup_policy": "ratio_uniform_r0",
        "kwargs": {"r0": 1.0e-9},
        "family": "ratio_uniform",
    },
    {
        "name": "ratio_uniform_r0_1e-6",
        "startup_policy": "ratio_uniform_r0",
        "kwargs": {"r0": 1.0e-6},
        "family": "ratio_uniform",
    },
    {
        "name": "ratio_uniform_r0_1e-3",
        "startup_policy": "ratio_uniform_r0",
        "kwargs": {"r0": 1.0e-3},
        "family": "ratio_uniform",
    },
    {
        "name": "ratio_positive_driving_r0_1e-6",
        "startup_policy": "ratio_positive_driving_r0",
        "kwargs": {"r0": 1.0e-6},
        "family": "ratio_selective",
    },
    {
        "name": "ratio_topk_positive_driving_r0_1e-6_topk3",
        "startup_policy": "ratio_topk_positive_driving_r0",
        "kwargs": {"r0": 1.0e-6, "top_k": 3},
        "family": "ratio_selective",
    },
    {
        "name": "ratio_topk_positive_driving_r0_1e-6_topk10",
        "startup_policy": "ratio_topk_positive_driving_r0",
        "kwargs": {"r0": 1.0e-6, "top_k": 10},
        "family": "ratio_selective",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYER_INDICES))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--lambda-trials", type=float, nargs="+", default=list(DEFAULT_LAMBDA_TRIALS))
    parser.add_argument("--gas-max-iter", type=int, default=1000)
    parser.add_argument("--short-max-iter", type=int, default=DEFAULT_SHORT_MAX_ITER)
    parser.add_argument("--tiny-fallback", type=float, default=DEFAULT_TINY_FALLBACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_OUTPUT)
    return parser


def _load_sparse_support_case(layer_index: int) -> dict[str, Any]:
    path = REPO_ROOT / "results" / f"sparse_layer{layer_index}_top1.json"
    if path.exists():
        payload = json.loads(path.read_text())
        layer_result = payload["results"][0]
        support_cfg = layer_result["configs"]["sparse_lp20_top1"]
        return {
            "temperature_K": float(payload["benchmark"]["temperature_K"]),
            "pressure_bar": float(layer_result["pressure_bar"]),
            "support_names": list(support_cfg["final_support_names"]),
            "support_source_json": str(path),
        }

    fallback_path = REPO_ROOT / "results" / "condensate_pipm_line_search_convergence_audit.json"
    if fallback_path.exists():
        payload = json.loads(fallback_path.read_text())
        for run in payload["runs"]:
            if int(run["layer_index"]) == int(layer_index):
                return {
                    "temperature_K": float(run["temperature_K"]),
                    "pressure_bar": float(run["pressure_bar"]),
                    "support_names": list(run["support_names"]),
                    "support_source_json": str(fallback_path),
                }

    raise FileNotFoundError(
        "Could not find support metadata for layer "
        f"{layer_index}. Looked for {path} and {fallback_path}."
    )


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


def _mean(values: list[float]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def _median(values: list[float]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(statistics.median(finite_values))


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


def _compute_gas_driving(
    state: ThermoState,
    ln_nk: jax.Array,
    ln_ntot: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
) -> dict[str, Any]:
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    An = formula_matrix @ nk
    pi_gas, delta_ln_ntot_gas = solve_gibbs_iteration_equations(
        nk,
        ntot,
        formula_matrix,
        b,
        gk,
        An,
    )
    driving = formula_matrix_cond.T @ pi_gas - hvector_cond
    return {
        "gk": gk,
        "pi_gas": pi_gas,
        "delta_ln_ntot_gas": delta_ln_ntot_gas,
        "driving": driving,
    }


def _variant_direction(raw_delta_ln_mk: jax.Array, variant_name: str) -> jax.Array:
    raw = jnp.asarray(raw_delta_ln_mk, dtype=jnp.float64)
    if variant_name == "current_clipped_baseline":
        return jnp.clip(raw, -0.1, 0.1)
    if variant_name == "scalar_rescale_0p1":
        max_abs = float(jnp.max(jnp.abs(raw)))
        alpha = 1.0 if max_abs <= 0.1 else 0.1 / max_abs
        return raw * alpha
    if variant_name == "raw_no_clip":
        return raw
    raise ValueError(f"Unknown variant_name: {variant_name}")


def _evaluate_lambda_grid(
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
    delta_ln_nk: jax.Array,
    delta_ln_mk: jax.Array,
    delta_ln_ntot: jax.Array,
    lambda_trials: list[float],
) -> dict[str, Any]:
    trials = []
    for lam in lambda_trials:
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            lam,
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
            reduced_solver=DEFAULT_REDUCED_SOLVER,
            regularization_mode=DEFAULT_REGULARIZATION_MODE,
            regularization_strength=DEFAULT_REGULARIZATION_STRENGTH,
        )
        fresh_residual = float(trial["fresh_residual"])
        finite = bool(jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"])
        trials.append(
            {
                "lambda_trial": float(lam),
                "fresh_residual": fresh_residual,
                "all_finite": finite,
            }
        )

    finite_residuals = [trial["fresh_residual"] for trial in trials if trial["all_finite"]]
    best_fresh_residual = None if not finite_residuals else min(finite_residuals)
    best_lambda = None
    if best_fresh_residual is not None:
        for trial in trials:
            if trial["all_finite"] and abs(trial["fresh_residual"] - best_fresh_residual) <= 1.0e-15:
                best_lambda = trial["lambda_trial"]
                break
    return {
        "best_fresh_residual": best_fresh_residual,
        "best_lambda": best_lambda,
        "trials": trials,
    }


def _compute_current_residual(
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
) -> float:
    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ nk
    Am = formula_matrix_cond @ mk
    pi = _recompute_pi_for_residual(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=DEFAULT_REDUCED_SOLVER,
        regularization_mode=DEFAULT_REGULARIZATION_MODE,
        regularization_strength=DEFAULT_REGULARIZATION_STRENGTH,
    )
    residual = _compute_residuals(
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
        pi,
    )
    return float(residual)


def _raw_anatomy(
    step_metrics: dict[str, Any],
    ln_mk: jax.Array,
    formula_matrix_cond: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
) -> dict[str, Any]:
    factor = jnp.exp(jnp.asarray(ln_mk, dtype=jnp.float64) - float(epsilon))
    driving = formula_matrix_cond.T @ jnp.asarray(step_metrics["pi_vector"], dtype=jnp.float64) - jnp.asarray(
        hvector_cond, dtype=jnp.float64
    )
    correction = factor * driving
    raw_delta_ln_mk = jnp.asarray(step_metrics["raw_delta_ln_mk"], dtype=jnp.float64)
    ones = jnp.ones_like(raw_delta_ln_mk)
    return {
        "factor_stats": _summary_triplet(factor),
        "correction_stats": _summary_triplet(correction),
        "raw_delta_ln_mk_stats": _summary_triplet(raw_delta_ln_mk),
        "fraction_abs_correction_lt_1e-12": float(jnp.mean(jnp.abs(correction) < 1.0e-12)),
        "fraction_abs_correction_lt_1e-9": float(jnp.mean(jnp.abs(correction) < 1.0e-9)),
        "fraction_abs_correction_lt_1e-6": float(jnp.mean(jnp.abs(correction) < 1.0e-6)),
        "fraction_raw_in_0p99_1p01": float(jnp.mean((raw_delta_ln_mk >= 0.99) & (raw_delta_ln_mk <= 1.01))),
        "fraction_raw_in_0p9_1p1": float(jnp.mean((raw_delta_ln_mk >= 0.9) & (raw_delta_ln_mk <= 1.1))),
        "cosine_raw_vs_ones": _safe_cosine(raw_delta_ln_mk, ones),
        "raw_cond_direction_norm": float(jnp.linalg.norm(raw_delta_ln_mk)),
        "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(raw_delta_ln_mk))),
    }


def _build_run_summary(trace: dict[str, Any], initial_raw_anatomy: dict[str, Any]) -> dict[str, Any]:
    history = trace["history"]
    monotone = [
        step["residual_after"] <= step["residual_before"] + 1.0e-15
        for step in history
        if math.isfinite(step["residual_before"]) and math.isfinite(step["residual_after"])
    ]
    accept_best = [step["line_search_accept_kind"] == "best_finite_fallback" for step in history]
    accept_zero = [step["line_search_accept_kind"] == "zero_step" for step in history]
    lam_reduced = [step["lam_selected"] < step["lam_heuristic"] - 1.0e-15 for step in history]
    return {
        "converged": bool(trace["converged"]),
        "n_iter": int(trace["n_iter"]),
        "final_residual": float(trace["final_residual"]),
        "hit_max_iter": bool(trace["hit_max_iter"]),
        "fraction_monotone_residual_decrease": _fraction(monotone),
        "fraction_accept_best_finite_fallback": _fraction(accept_best),
        "fraction_accept_zero_step": _fraction(accept_zero),
        "fraction_lam_selected_lt_lam_heuristic": _fraction(lam_reduced),
        "initial_raw_anatomy": initial_raw_anatomy,
    }


def _print_terminal_table(summary_rows: list[dict[str, Any]]) -> None:
    headers = (
        "policy",
        "deg(0.99-1.01)",
        "cos(raw,1)",
        "clip_best",
        "raw<clip",
        "scalar<clip",
        "conv",
        "mean_iter",
        "mean_final_res",
    )
    print(" ".join(f"{header:>18}" for header in headers))
    for row in summary_rows:
        def _fmt(value: Any, precision: int = 3) -> str:
            if value is None:
                return "-"
            if isinstance(value, str):
                return value
            if isinstance(value, (bool, int)):
                return str(value)
            if abs(float(value)) >= 1.0e3 or (0.0 < abs(float(value)) < 1.0e-3):
                return f"{float(value):.{precision}e}"
            return f"{float(value):.{precision}f}"

        print(
            f"{row['policy_name']:>18} "
            f"{_fmt(row['mean_fraction_raw_in_0p99_1p01']):>18} "
            f"{_fmt(row['mean_cosine_raw_vs_ones']):>18} "
            f"{_fmt(row['mean_clip_best_fresh_residual']):>18} "
            f"{_fmt(row['fraction_raw_beats_clip']):>18} "
            f"{_fmt(row['fraction_scalar_beats_clip']):>18} "
            f"{_fmt(row['convergence_rate']):>18} "
            f"{_fmt(row['mean_n_iter']):>18} "
            f"{_fmt(row['mean_final_residual']):>18}"
        )


def _decision_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["policy_name"]: row for row in summary_rows}
    baseline = by_name["absolute_uniform_m0_1e-30"]
    ratio_rows = [row for row in summary_rows if row["family"].startswith("ratio")]
    ratio_uniform_rows = [row for row in summary_rows if row["family"] == "ratio_uniform"]
    ratio_selective_rows = [row for row in summary_rows if row["family"] == "ratio_selective"]

    best_ratio = min(
        ratio_rows,
        key=lambda row: (
            -1.0 if row["convergence_rate"] is None else -row["convergence_rate"],
            math.inf if row["mean_final_residual"] is None else row["mean_final_residual"],
            math.inf if row["mean_fraction_raw_in_0p99_1p01"] is None else row["mean_fraction_raw_in_0p99_1p01"],
        ),
    )
    best_ratio_uniform = min(
        ratio_uniform_rows,
        key=lambda row: (
            -1.0 if row["convergence_rate"] is None else -row["convergence_rate"],
            math.inf if row["mean_final_residual"] is None else row["mean_final_residual"],
        ),
    )
    best_ratio_selective = min(
        ratio_selective_rows,
        key=lambda row: (
            -1.0 if row["convergence_rate"] is None else -row["convergence_rate"],
            math.inf if row["mean_final_residual"] is None else row["mean_final_residual"],
        ),
    )

    deg_reduction = None
    if (
        baseline["mean_fraction_raw_in_0p99_1p01"] is not None
        and best_ratio["mean_fraction_raw_in_0p99_1p01"] is not None
    ):
        deg_reduction = baseline["mean_fraction_raw_in_0p99_1p01"] - best_ratio["mean_fraction_raw_in_0p99_1p01"]

    residual_improvement = None
    if baseline["mean_final_residual"] is not None and best_ratio["mean_final_residual"] is not None:
        residual_improvement = baseline["mean_final_residual"] / max(best_ratio["mean_final_residual"], 1.0e-300)

    convergence_gain = None
    if baseline["convergence_rate"] is not None and best_ratio["convergence_rate"] is not None:
        convergence_gain = best_ratio["convergence_rate"] - baseline["convergence_rate"]

    selective_better = False
    if (
        best_ratio_selective["mean_final_residual"] is not None
        and best_ratio_uniform["mean_final_residual"] is not None
    ):
        selective_better = (
            best_ratio_selective["mean_final_residual"] <= 0.9 * best_ratio_uniform["mean_final_residual"]
            or (
                best_ratio_selective["convergence_rate"] is not None
                and best_ratio_uniform["convergence_rate"] is not None
                and best_ratio_selective["convergence_rate"] >= best_ratio_uniform["convergence_rate"] + 0.1
            )
        )

    ratio_is_real_lever = (
        deg_reduction is not None
        and deg_reduction >= 0.2
        and (
            (residual_improvement is not None and residual_improvement >= 1.25)
            or (convergence_gain is not None and convergence_gain >= 0.2)
        )
    )
    startup_alone_not_enough = (
        deg_reduction is not None
        and deg_reduction >= 0.2
        and not ratio_is_real_lever
    )

    messages = []
    if ratio_is_real_lever:
        messages.append("RGIE startup policy is a real lever; continue improving RGIE before switching solver family")
    if selective_better:
        messages.append("initial support-aware condensate seeding is promising for RGIE")
    if startup_alone_not_enough:
        messages.append("startup policy alone does not rescue RGIE; inspect raw direction construction itself")
    if best_ratio["family"].startswith("ratio") and not best_ratio["policy_name"].startswith("absolute"):
        messages.append("absolute m0 matters much less than r0 = m0 / nu in this audit")

    next_move = (
        "promote an RGIE startup policy"
        if ratio_is_real_lever
        else "leave startup alone and inspect the raw RGIE condensate update formula itself"
    )
    return {
        "best_ratio_policy": best_ratio["policy_name"],
        "best_ratio_uniform_policy": best_ratio_uniform["policy_name"],
        "best_ratio_selective_policy": best_ratio_selective["policy_name"],
        "degeneracy_reduction_vs_baseline": deg_reduction,
        "final_residual_improvement_factor_vs_baseline": residual_improvement,
        "convergence_gain_vs_baseline": convergence_gain,
        "messages": messages,
        "next_move": next_move,
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
            formula_matrix = jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64)
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
            support_indices = jnp.asarray(
                [cond_setup.species.index(name) for name in layer_meta["support_names"]],
                dtype=jnp.int32,
            )
            formula_matrix_cond = jnp.asarray(cond_setup.formula_matrix[:, support_indices], dtype=jnp.float64)
            hvector_cond = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
            b = jnp.asarray(state.element_vector, dtype=jnp.float64)
            gas_driving_payload = _compute_gas_driving(
                state,
                gas_init["ln_nk"],
                gas_init["ln_ntot"],
                formula_matrix,
                formula_matrix_cond,
                b,
                gas_hvector,
                hvector_cond,
            )

            for epsilon in args.epsilons:
                for policy in POLICIES:
                    ln_mk = build_rgie_condensate_init_from_policy(
                        epsilon=epsilon,
                        support_indices=support_indices,
                        startup_policy=policy["startup_policy"],
                        driving=gas_driving_payload["driving"],
                        tiny_fallback=args.tiny_fallback,
                        dtype=jnp.float64,
                        **policy["kwargs"],
                    )
                    ln_nk = gas_init["ln_nk"]
                    ln_ntot = gas_init["ln_ntot"]
                    current_residual = _compute_current_residual(
                        state,
                        ln_nk,
                        ln_mk,
                        ln_ntot,
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        gas_hvector,
                        hvector_cond,
                        epsilon,
                    )
                    gk = _compute_gk(
                        state.temperature,
                        ln_nk,
                        ln_ntot,
                        gas_hvector,
                        state.ln_normalized_pressure,
                    )
                    step_metrics = _compute_iteration_step_metrics(
                        ln_nk,
                        ln_mk,
                        ln_ntot,
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        gk,
                        hvector_cond,
                        epsilon,
                        reduced_solver=DEFAULT_REDUCED_SOLVER,
                        regularization_mode=DEFAULT_REGULARIZATION_MODE,
                        regularization_strength=DEFAULT_REGULARIZATION_STRENGTH,
                    )
                    raw_anatomy = _raw_anatomy(
                        step_metrics,
                        ln_mk,
                        formula_matrix_cond,
                        hvector_cond,
                        epsilon,
                    )

                    baseline_trials = _evaluate_lambda_grid(
                        state,
                        ln_nk,
                        ln_mk,
                        ln_ntot,
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        gas_hvector,
                        hvector_cond,
                        epsilon,
                        step_metrics["delta_ln_nk"],
                        _variant_direction(step_metrics["raw_delta_ln_mk"], "current_clipped_baseline"),
                        step_metrics["delta_ln_ntot"],
                        lambda_trials,
                    )
                    scalar_trials = _evaluate_lambda_grid(
                        state,
                        ln_nk,
                        ln_mk,
                        ln_ntot,
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        gas_hvector,
                        hvector_cond,
                        epsilon,
                        step_metrics["delta_ln_nk"],
                        _variant_direction(step_metrics["raw_delta_ln_mk"], "scalar_rescale_0p1"),
                        step_metrics["delta_ln_ntot"],
                        lambda_trials,
                    )
                    raw_trials = _evaluate_lambda_grid(
                        state,
                        ln_nk,
                        ln_mk,
                        ln_ntot,
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        gas_hvector,
                        hvector_cond,
                        epsilon,
                        step_metrics["delta_ln_nk"],
                        _variant_direction(step_metrics["raw_delta_ln_mk"], "raw_no_clip"),
                        step_metrics["delta_ln_ntot"],
                        lambda_trials,
                    )

                    trajectory_trace = trace_minimize_gibbs_cond_iterations(
                        state,
                        ln_nk_init=ln_nk,
                        ln_mk_init=ln_mk,
                        ln_ntot_init=ln_ntot,
                        formula_matrix=formula_matrix,
                        formula_matrix_cond=formula_matrix_cond,
                        hvector_func=lambda _temperature, arr=gas_hvector: arr,
                        hvector_cond_func=lambda _temperature, arr=hvector_cond: arr,
                        epsilon=epsilon,
                        residual_crit=float(jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))),
                        max_iter=args.short_max_iter,
                        reduced_solver=DEFAULT_REDUCED_SOLVER,
                        regularization_mode=DEFAULT_REGULARIZATION_MODE,
                        regularization_strength=DEFAULT_REGULARIZATION_STRENGTH,
                    )
                    run_summary = _build_run_summary(trajectory_trace, raw_anatomy)

                    case_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "temperature_K": layer_meta["temperature_K"],
                            "pressure_bar": layer_meta["pressure_bar"],
                            "support_names": list(layer_meta["support_names"]),
                            "epsilon": float(epsilon),
                            "policy_name": policy["name"],
                            "policy_family": policy["family"],
                            "startup_policy": policy["startup_policy"],
                            "policy_kwargs": dict(policy["kwargs"]),
                            "gas_init_n_iter": gas_init["gas_init_n_iter"],
                            "gas_init_final_residual": gas_init["gas_init_final_residual"],
                            "current_residual": current_residual,
                            "gas_driving_stats": _summary_triplet(gas_driving_payload["driving"]),
                            "positive_driving_fraction": float(jnp.mean(gas_driving_payload["driving"] > 0.0)),
                            "initial_ln_mk": to_python(jax.device_get(ln_mk)),
                            "initial_r0": to_python(jax.device_get(jnp.exp(ln_mk - float(epsilon)))),
                            "raw_anatomy_iter0": raw_anatomy,
                            "one_step_fresh_residuals": {
                                "current_clipped_baseline_best": baseline_trials["best_fresh_residual"],
                                "scalar_rescale_0p1_best": scalar_trials["best_fresh_residual"],
                                "raw_no_clip_best": raw_trials["best_fresh_residual"],
                                "raw_no_clip_beats_current_clipped_baseline": (
                                    raw_trials["best_fresh_residual"] is not None
                                    and baseline_trials["best_fresh_residual"] is not None
                                    and raw_trials["best_fresh_residual"] < baseline_trials["best_fresh_residual"]
                                ),
                                "scalar_rescale_beats_current_clipped_baseline": (
                                    scalar_trials["best_fresh_residual"] is not None
                                    and baseline_trials["best_fresh_residual"] is not None
                                    and scalar_trials["best_fresh_residual"] < baseline_trials["best_fresh_residual"]
                                ),
                            },
                            "short_run": run_summary,
                        }
                    )
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "policy_name": policy["name"],
                            "raw_anatomy_iter0": raw_anatomy,
                            "lambda_grid_trials": {
                                "current_clipped_baseline": baseline_trials,
                                "scalar_rescale_0p1": scalar_trials,
                                "raw_no_clip": raw_trials,
                            },
                            "trajectory_trace": trajectory_trace,
                        }
                    )

        summary_rows = []
        for policy in POLICIES:
            matching = [record for record in case_records if record["policy_name"] == policy["name"]]
            summary_rows.append(
                {
                    "policy_name": policy["name"],
                    "family": policy["family"],
                    "n_cases": len(matching),
                    "mean_fraction_raw_in_0p99_1p01": _mean(
                        [record["raw_anatomy_iter0"]["fraction_raw_in_0p99_1p01"] for record in matching]
                    ),
                    "mean_cosine_raw_vs_ones": _mean(
                        [
                            value
                            for value in (record["raw_anatomy_iter0"]["cosine_raw_vs_ones"] for record in matching)
                            if value is not None
                        ]
                    ),
                    "mean_clip_best_fresh_residual": _mean(
                        [record["one_step_fresh_residuals"]["current_clipped_baseline_best"] for record in matching]
                    ),
                    "fraction_raw_beats_clip": _fraction(
                        [
                            record["one_step_fresh_residuals"]["raw_no_clip_beats_current_clipped_baseline"]
                            for record in matching
                        ]
                    ),
                    "fraction_scalar_beats_clip": _fraction(
                        [
                            record["one_step_fresh_residuals"]["scalar_rescale_beats_current_clipped_baseline"]
                            for record in matching
                        ]
                    ),
                    "convergence_rate": _fraction([record["short_run"]["converged"] for record in matching]),
                    "mean_n_iter": _mean([record["short_run"]["n_iter"] for record in matching]),
                    "mean_final_residual": _mean([record["short_run"]["final_residual"] for record in matching]),
                }
            )

        decision = _decision_summary(summary_rows)
        audit_payload = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": list(args.layers),
            "epsilons": list(args.epsilons),
            "lambda_trials": lambda_trials,
            "short_max_iter": int(args.short_max_iter),
            "tiny_fallback": float(args.tiny_fallback),
            "policies": [dict(policy) for policy in POLICIES],
            "cases": case_records,
            "summary_rows": summary_rows,
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "lambda_trials": lambda_trials,
            "traces": trace_records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(audit_payload), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2))

    _print_terminal_table(summary_rows)
    print()
    for message in decision["messages"]:
        print(f"decision: {message}")
    print(f"next_move: {decision['next_move']}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
