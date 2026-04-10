"""Refine RGIE startup policy calibration around the ratio-based sweet spot.

This audit keeps the current RGIE raw direction formula, clipping, and
residual-based line search unchanged. It compares a narrow band of ratio-based
initializations around ``r0 = 1e-3`` and a small floor-protected hot-start
profile experiment through the structured RGIE API.
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
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import CondensateRGIEStartupConfig
from exogibbs.optimize.minimize_cond import minimize_gibbs_cond_profile
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_iterations
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_startup_policy_refinement_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_startup_policy_refinement_traces.json"
DEFAULT_SHORT_MAX_ITER = 7
DEFAULT_LEGACY_M0 = 1.0e-30

MAIN_POLICIES = (
    {"name": "legacy_absolute_m0_1e-30", "kind": "legacy", "m0": 1.0e-30},
    {"name": "ratio_uniform_r0_1e-4", "kind": "ratio_uniform", "r0": 1.0e-4},
    {"name": "ratio_uniform_r0_3e-4", "kind": "ratio_uniform", "r0": 3.0e-4},
    {"name": "ratio_uniform_r0_1e-3", "kind": "ratio_uniform", "r0": 1.0e-3},
    {"name": "ratio_uniform_r0_3e-3", "kind": "ratio_uniform", "r0": 3.0e-3},
    {"name": "ratio_uniform_r0_1e-2", "kind": "ratio_uniform", "r0": 1.0e-2},
)

PROFILE_POLICIES = (
    {
        "name": "legacy_absolute_m0_1e-30",
        "startup_config": None,
    },
    {
        "name": "ratio_uniform_r0_1e-3",
        "startup_config": CondensateRGIEStartupConfig(policy="ratio_uniform_r0", r0=1.0e-3),
    },
    {
        "name": "warm_previous_with_ratio_floor_r0_1e-4",
        "startup_config": CondensateRGIEStartupConfig(
            policy="warm_previous_with_ratio_floor",
            r0=1.0e-4,
        ),
    },
    {
        "name": "warm_previous_with_ratio_floor_r0_1e-3",
        "startup_config": CondensateRGIEStartupConfig(
            policy="warm_previous_with_ratio_floor",
            r0=1.0e-3,
        ),
    },
    {
        "name": "warm_previous_with_ratio_floor_r0_1e-2",
        "startup_config": CondensateRGIEStartupConfig(
            policy="warm_previous_with_ratio_floor",
            r0=1.0e-2,
        ),
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_OUTPUT)
    return parser


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


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


def _safe_cosine(a: jax.Array, b: jax.Array) -> float | None:
    a = jnp.ravel(jnp.asarray(a, dtype=jnp.float64))
    b = jnp.ravel(jnp.asarray(b, dtype=jnp.float64))
    denom = float(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    if not math.isfinite(denom) or denom <= 1.0e-300:
        return None
    cosine = float(jnp.dot(a, b) / denom)
    return max(-1.0, min(1.0, cosine))


def _load_support_case(layer_index: int) -> dict[str, Any]:
    sparse_path = REPO_ROOT / "results" / f"sparse_layer{layer_index}_top1.json"
    if sparse_path.exists():
        payload = json.loads(sparse_path.read_text())
        layer_result = payload["results"][0]
        support_cfg = layer_result["configs"]["sparse_lp20_top1"]
        return {
            "temperature_K": float(payload["benchmark"]["temperature_K"]),
            "pressure_bar": float(layer_result["pressure_bar"]),
            "support_names": list(support_cfg["final_support_names"]),
            "support_source_json": str(sparse_path),
        }

    fallback_path = REPO_ROOT / "results" / "condensate_pipm_line_search_convergence_audit.json"
    payload = json.loads(fallback_path.read_text())
    for run in payload["runs"]:
        if int(run["layer_index"]) == int(layer_index):
            return {
                "temperature_K": float(run["temperature_K"]),
                "pressure_bar": float(run["pressure_bar"]),
                "support_names": list(run["support_names"]),
                "support_source_json": str(fallback_path),
            }
    raise FileNotFoundError(f"Missing support metadata for layer {layer_index}.")


def _build_profile_states(element_vector: jax.Array, layer_indices: list[int]) -> list[dict[str, Any]]:
    states = []
    for layer_index in layer_indices:
        support_case = _load_support_case(layer_index)
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


def _build_shared_gas_init(state: ThermoState, gas_setup: Any, gas_hvector: jax.Array, gas_max_iter: int) -> dict[str, Any]:
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


def _build_ln_mk(policy: dict[str, Any], epsilon: float, n_cond: int) -> jax.Array:
    if policy["kind"] == "legacy":
        return jnp.full((n_cond,), jnp.log(policy["m0"]), dtype=jnp.float64)
    return jnp.full((n_cond,), epsilon + jnp.log(policy["r0"]), dtype=jnp.float64)


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
        )
        finite = bool(jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"])
        trials.append(
            {
                "lambda_trial": float(lam),
                "fresh_residual": float(trial["fresh_residual"]),
                "all_finite": finite,
            }
        )
    finite_residuals = [trial["fresh_residual"] for trial in trials if trial["all_finite"]]
    return {
        "best_fresh_residual": None if not finite_residuals else min(finite_residuals),
        "trials": trials,
    }


def _raw_metrics(step_metrics: dict[str, Any], ln_mk: jax.Array, epsilon: float) -> dict[str, Any]:
    raw = jnp.asarray(step_metrics["raw_delta_ln_mk"], dtype=jnp.float64)
    factor = jnp.exp(jnp.asarray(ln_mk, dtype=jnp.float64) - epsilon)
    return {
        "fraction_raw_in_0p99_1p01": float(jnp.mean((raw >= 0.99) & (raw <= 1.01))),
        "cosine_raw_vs_ones": _safe_cosine(raw, jnp.ones_like(raw)),
        "factor_stats": _summary_triplet(factor),
    }


def _run_summary(trace: dict[str, Any], raw_metrics: dict[str, Any], raw_no_clip_beats_clip: bool) -> dict[str, Any]:
    history = trace["history"]
    monotone = [
        step["residual_after"] <= step["residual_before"] + 1.0e-15
        for step in history
        if math.isfinite(step["residual_before"]) and math.isfinite(step["residual_after"])
    ]
    best_finite = [step["line_search_accept_kind"] == "best_finite_fallback" for step in history]
    zero_step = [step["line_search_accept_kind"] == "zero_step" for step in history]
    lam_reduced = [step["lam_selected"] < step["lam_heuristic"] - 1.0e-15 for step in history]
    return {
        "initial_raw_metrics": raw_metrics,
        "converged": bool(trace["converged"]),
        "n_iter": int(trace["n_iter"]),
        "final_residual": float(trace["final_residual"]),
        "hit_max_iter": bool(trace["hit_max_iter"]),
        "fraction_monotone_residual_decrease": _fraction(monotone),
        "fraction_accept_best_finite_fallback": _fraction(best_finite),
        "fraction_accept_zero_step": _fraction(zero_step),
        "fraction_lam_selected_lt_lam_heuristic": _fraction(lam_reduced),
        "raw_no_clip_beats_clipped_baseline_first_step": bool(raw_no_clip_beats_clip),
    }


def _print_terminal_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "policy",
        "deg(0.99-1.01)",
        "cos(raw,1)",
        "factor_med",
        "raw<clip",
        "conv",
        "mean_iter",
        "mean_final_res",
    )
    print(" ".join(f"{header:>18}" for header in headers))
    for row in rows:
        def _fmt(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, str):
                return value
            if abs(float(value)) >= 1.0e3 or (0.0 < abs(float(value)) < 1.0e-3):
                return f"{float(value):.3e}"
            return f"{float(value):.3f}"

        print(
            f"{row['policy_name']:>18} "
            f"{_fmt(row['mean_deg_fraction']):>18} "
            f"{_fmt(row['mean_cos_raw_vs_ones']):>18} "
            f"{_fmt(row['mean_factor_median']):>18} "
            f"{_fmt(row['fraction_raw_beats_clip']):>18} "
            f"{_fmt(row['convergence_rate']):>18} "
            f"{_fmt(row['mean_n_iter']):>18} "
            f"{_fmt(row['mean_final_residual']):>18}"
        )


def _decision_summary(main_rows: list[dict[str, Any]], profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["policy_name"]: row for row in main_rows}
    sweet_candidates = [
        by_name["ratio_uniform_r0_3e-4"],
        by_name["ratio_uniform_r0_1e-3"],
        by_name["ratio_uniform_r0_3e-3"],
    ]
    best_main = min(
        main_rows,
        key=lambda row: (
            math.inf if row["mean_final_residual"] is None else row["mean_final_residual"],
            -1.0 if row["fraction_raw_beats_clip"] is None else -row["fraction_raw_beats_clip"],
        ),
    )
    best_sweet = min(
        sweet_candidates,
        key=lambda row: math.inf if row["mean_final_residual"] is None else row["mean_final_residual"],
    )
    stable_sweet_spot = best_sweet["policy_name"] == "ratio_uniform_r0_1e-3"
    if best_sweet["mean_final_residual"] is not None:
        for neighbor_name in ("ratio_uniform_r0_3e-4", "ratio_uniform_r0_3e-3"):
            neighbor = by_name[neighbor_name]
            if neighbor["mean_final_residual"] is not None:
                stable_sweet_spot = stable_sweet_spot and (
                    best_sweet["mean_final_residual"] <= 0.95 * neighbor["mean_final_residual"]
                )

    profile_by_name = {row["policy_name"]: row for row in profile_rows}
    warm_beats_uniform = False
    if (
        "warm_previous_with_ratio_floor_r0_1e-3" in profile_by_name
        and "ratio_uniform_r0_1e-3" in profile_by_name
    ):
        warm = profile_by_name["warm_previous_with_ratio_floor_r0_1e-3"]
        uniform = profile_by_name["ratio_uniform_r0_1e-3"]
        if warm["mean_final_residual"] is not None and uniform["mean_final_residual"] is not None:
            warm_beats_uniform = warm["mean_final_residual"] <= 0.95 * uniform["mean_final_residual"]

    all_fail_converge = all((row["convergence_rate"] or 0.0) == 0.0 for row in main_rows)
    messages = []
    if stable_sweet_spot:
        messages.append("promote ratio-based RGIE startup policy; r0 has a stable sweet spot")
    if warm_beats_uniform:
        messages.append("use floor-protected hot starts for RGIE profile solves")
    if all_fail_converge:
        messages.append("startup policy is now good enough; inspect the raw RGIE condensate update formula next")
    messages.append("selective/support-aware seeding was not re-promoted here; prior audit did not beat uniform ratio seeding")

    next_move = (
        "startup is now mostly solved; inspect the raw RGIE condensate update formula"
        if all_fail_converge
        else "promote startup policy into the RGIE path"
    )
    return {
        "best_main_policy": best_main["policy_name"],
        "stable_sweet_spot": stable_sweet_spot,
        "best_sweet_spot_policy": best_sweet["policy_name"],
        "warm_previous_with_ratio_floor_beats_uniform_profile": warm_beats_uniform,
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
        layer_states = _build_profile_states(gas_setup.element_vector_reference, args.layers)

        case_records = []
        trace_records = []
        gas_init_by_layer = {}

        for layer_meta in layer_states:
            state = layer_meta["state"]
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
            gas_init_by_layer[layer_meta["layer_index"]] = gas_init
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            support_indices = jnp.asarray(
                [cond_setup.species.index(name) for name in layer_meta["support_names"]],
                dtype=jnp.int32,
            )
            formula_matrix = jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64)
            formula_matrix_cond = jnp.asarray(cond_setup.formula_matrix[:, support_indices], dtype=jnp.float64)
            hvector_cond = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
            b = jnp.asarray(state.element_vector, dtype=jnp.float64)

            for epsilon in args.epsilons:
                for policy in MAIN_POLICIES:
                    ln_nk = gas_init["ln_nk"]
                    ln_ntot = gas_init["ln_ntot"]
                    ln_mk = _build_ln_mk(policy, epsilon, formula_matrix_cond.shape[1])
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
                    )
                    raw_metrics = _raw_metrics(step_metrics, ln_mk, epsilon)
                    clipped_trials = _evaluate_lambda_grid(
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
                        jnp.clip(step_metrics["raw_delta_ln_mk"], -0.1, 0.1),
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
                        step_metrics["raw_delta_ln_mk"],
                        step_metrics["delta_ln_ntot"],
                        lambda_trials,
                    )
                    raw_beats_clip = (
                        raw_trials["best_fresh_residual"] is not None
                        and clipped_trials["best_fresh_residual"] is not None
                        and raw_trials["best_fresh_residual"] < clipped_trials["best_fresh_residual"]
                    )
                    trace = trace_minimize_gibbs_cond_iterations(
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
                    )
                    run_summary = _run_summary(trace, raw_metrics, raw_beats_clip)
                    case_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "policy_name": policy["name"],
                            "raw_metrics": raw_metrics,
                            "run_summary": run_summary,
                        }
                    )
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "policy_name": policy["name"],
                            "lambda_grid": {
                                "clipped": clipped_trials,
                                "raw_no_clip": raw_trials,
                            },
                            "trace": trace,
                        }
                    )

        main_rows = []
        for policy in MAIN_POLICIES:
            matching = [record for record in case_records if record["policy_name"] == policy["name"]]
            main_rows.append(
                {
                    "policy_name": policy["name"],
                    "mean_deg_fraction": _mean(
                        [record["raw_metrics"]["fraction_raw_in_0p99_1p01"] for record in matching]
                    ),
                    "mean_cos_raw_vs_ones": _mean(
                        [record["raw_metrics"]["cosine_raw_vs_ones"] for record in matching]
                    ),
                    "mean_factor_median": _mean(
                        [record["raw_metrics"]["factor_stats"]["median"] for record in matching]
                    ),
                    "fraction_raw_beats_clip": _fraction(
                        [record["run_summary"]["raw_no_clip_beats_clipped_baseline_first_step"] for record in matching]
                    ),
                    "convergence_rate": _fraction([record["run_summary"]["converged"] for record in matching]),
                    "mean_n_iter": _mean([record["run_summary"]["n_iter"] for record in matching]),
                    "mean_final_residual": _mean([record["run_summary"]["final_residual"] for record in matching]),
                }
            )

        fixed_support_layer = max(args.layers)
        support_case = next(item for item in layer_states if item["layer_index"] == fixed_support_layer)
        profile_support_indices = jnp.asarray(
            [cond_setup.species.index(name) for name in support_case["support_names"]],
            dtype=jnp.int32,
        )
        profile_formula_matrix_cond = jnp.asarray(
            cond_setup.formula_matrix[:, profile_support_indices],
            dtype=jnp.float64,
        )
        profile_cond_hvector = lambda temperature: jnp.asarray(
            cond_setup.hvector_func(temperature)[profile_support_indices],
            dtype=jnp.float64,
        )
        profile_init = CondensateEquilibriumInit(
            ln_nk=jnp.stack([gas_init_by_layer[item["layer_index"]]["ln_nk"] for item in layer_states], axis=0),
            ln_mk=jnp.full(
                (len(layer_states), profile_formula_matrix_cond.shape[1]),
                jnp.log(DEFAULT_LEGACY_M0),
                dtype=jnp.float64,
            ),
            ln_ntot=jnp.asarray([gas_init_by_layer[item["layer_index"]]["ln_ntot"] for item in layer_states], dtype=jnp.float64),
        )

        profile_rows = []
        profile_traces = []
        for policy in PROFILE_POLICIES:
            result = minimize_gibbs_cond_profile(
                temperatures=jnp.asarray([item["temperature_K"] for item in layer_states], dtype=jnp.float64),
                ln_normalized_pressures=jnp.log(
                    jnp.asarray([item["pressure_bar"] for item in layer_states], dtype=jnp.float64)
                ),
                element_vector=jnp.asarray(gas_setup.element_vector_reference, dtype=jnp.float64),
                init=profile_init,
                formula_matrix=jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64),
                formula_matrix_cond=profile_formula_matrix_cond,
                hvector_func=lambda temperature: jnp.asarray(gas_setup.hvector_func(temperature), dtype=jnp.float64),
                hvector_cond_func=profile_cond_hvector,
                epsilon_start=0.0,
                epsilon_crit=-5.0,
                n_step=1,
                max_iter=args.short_max_iter,
                method="scan_hot_from_bottom",
                startup_config=policy["startup_config"],
            )
            final_residual = to_python(jax.device_get(result.diagnostics.final_residual))
            converged = to_python(jax.device_get(result.diagnostics.converged))
            profile_rows.append(
                {
                    "policy_name": policy["name"],
                    "mean_final_residual": _mean([float(value) for value in final_residual]),
                    "convergence_rate": _fraction([bool(value) for value in converged]),
                    "layer_final_residuals": final_residual,
                    "layer_converged": converged,
                }
            )
            profile_traces.append(
                {
                    "policy_name": policy["name"],
                    "layer_final_residuals": final_residual,
                    "layer_converged": converged,
                    "fixed_support_layer": fixed_support_layer,
                    "fixed_support_names": list(support_case["support_names"]),
                }
            )

        decision = _decision_summary(main_rows, profile_rows)
        audit_payload = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": list(args.layers),
            "epsilons": list(args.epsilons),
            "lambda_trials": lambda_trials,
            "main_policies": [dict(policy) for policy in MAIN_POLICIES],
            "main_rows": main_rows,
            "main_cases": case_records,
            "profile_experiment": {
                "fixed_support_layer": fixed_support_layer,
                "fixed_support_names": list(support_case["support_names"]),
                "epsilon_start": 0.0,
                "epsilon_crit": -5.0,
                "n_step": 1,
                "rows": profile_rows,
            },
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "main_traces": trace_records,
            "profile_traces": profile_traces,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(audit_payload), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2))

    _print_terminal_table(main_rows)
    print()
    for row in profile_rows:
        print(
            "profile:"
            f" policy={row['policy_name']} mean_final_residual={row['mean_final_residual']:.3e}"
            f" convergence_rate={0.0 if row['convergence_rate'] is None else row['convergence_rate']:.3f}"
        )
    print()
    for message in decision["messages"]:
        print(f"decision: {message}")
    print(f"next_move: {decision['next_move']}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
