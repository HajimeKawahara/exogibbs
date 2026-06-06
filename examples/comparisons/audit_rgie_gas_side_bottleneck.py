"""Diagnostic-only RGIE gas-side bottleneck audit."""

from __future__ import annotations

import argparse
import json
import math
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
from exogibbs.optimize.pipm_rgie_cond import _compute_frozen_condensate_gas_direction_reference
from exogibbs.optimize.pipm_rgie_cond import _compute_gas_limiter_species_diagnostics
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import build_rgie_gas_direction_variant
from exogibbs.optimize.pipm_rgie_cond import compute_rgie_lam1_gas_ignore_trace_diagnostics
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_gas_side_bottleneck_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_gas_side_bottleneck_traces.json"

STARTUP_POLICIES = (
    {"name": "legacy_absolute_m0_1e-30", "policy": "absolute_uniform_m0", "kwargs": {"m0": 1.0e-30}},
    {"name": "ratio_uniform_r0_3e-3", "policy": "ratio_uniform_r0", "kwargs": {"r0": 3.0e-3}},
    {"name": "ratio_uniform_r0_1e-2", "policy": "ratio_uniform_r0", "kwargs": {"r0": 1.0e-2}},
)

GAS_VARIANTS = (
    "current_full_direction",
    "frozen_condensate_gas_only_reference",
    "no_common_ntot_shift",
    "partial_ntot_shift_0p25",
    "partial_ntot_shift_0p5",
    "gas_only_with_current_condensate_block",
)

TRACE_FLOORS = (1.0e-30, 1.0e-25, 1.0e-20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYER_INDICES))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--lambda-trials", type=float, nargs="+", default=list(DEFAULT_LAMBDA_TRIALS))
    parser.add_argument("--gas-max-iter", type=int, default=1000)
    parser.add_argument("--trajectory-max-iter", type=int, default=5)
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
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    An = formula_matrix @ nk
    Am = formula_matrix_cond @ mk
    pi = _recompute_pi_for_residual(
        nk, mk, ntot, formula_matrix, formula_matrix_cond, b, gk, hvector_cond, epsilon
    )
    residual = _compute_residuals(
        nk, mk, ntot, formula_matrix, formula_matrix_cond, b, gk, hvector_cond, jnp.exp(epsilon), An, Am, pi
    )
    return float(residual)


def _evaluate_gas_variant_lambda_grid(
    state: ThermoState,
    *,
    ln_nk: jax.Array,
    ln_mk: jax.Array,
    ln_ntot: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
    gas_variant: dict[str, Any],
    delta_ln_mk_current: jax.Array,
    lambda_trials: list[float],
    current_residual: float,
    production_lam: float,
) -> dict[str, Any]:
    trials = []
    for lam in lambda_trials:
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            lam,
            gas_variant["delta_ln_nk"],
            delta_ln_mk_current,
            gas_variant["delta_ln_ntot"],
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
        )
        valid = bool(trial["all_finite"]) and math.isfinite(float(trial["fresh_residual"]))
        trials.append(
            {
                "lambda_trial": float(lam),
                "fresh_residual": float(trial["fresh_residual"]),
                "valid_trial": valid,
            }
        )
    valid_trials = [trial for trial in trials if trial["valid_trial"]]
    best_trial = None if not valid_trials else min(valid_trials, key=lambda trial: trial["fresh_residual"])
    return {
        "valid_trial_count": len(valid_trials),
        "invalid_trial_count": len(trials) - len(valid_trials),
        "best_lambda": None if best_trial is None else best_trial["lambda_trial"],
        "best_fresh_residual": None if best_trial is None else best_trial["fresh_residual"],
        "any_trial_improves_on_current_residual": any(
            trial["valid_trial"] and trial["fresh_residual"] <= current_residual + 1.0e-15
            for trial in trials
        ),
        "best_lambda_exceeds_current_production_heuristic_lambda": (
            best_trial is not None and best_trial["lambda_trial"] > production_lam + 1.0e-15
        ),
        "trials": trials,
    }


def _run_simple_gas_variant_trajectory(
    state: ThermoState,
    *,
    ln_nk_init: jax.Array,
    ln_mk_init: jax.Array,
    ln_ntot_init: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond: jax.Array,
    epsilon: float,
    variant_name: str,
    max_iter: int,
) -> dict[str, Any]:
    ln_nk = jnp.asarray(ln_nk_init)
    ln_mk = jnp.asarray(ln_mk_init)
    ln_ntot = jnp.asarray(ln_ntot_init)
    residual = _compute_current_residual(
        state, ln_nk, ln_mk, ln_ntot, formula_matrix, formula_matrix_cond, b, hvector, hvector_cond, epsilon
    )
    history = []
    residual_crit = float(math.exp(float(epsilon)))

    for iter_count in range(max_iter):
        if residual <= residual_crit:
            break
        gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
        step_metrics = _compute_iteration_step_metrics(
            ln_nk, ln_mk, ln_ntot, formula_matrix, formula_matrix_cond, b, gk, hvector_cond, epsilon
        )
        gas_ref = _compute_frozen_condensate_gas_direction_reference(
            jnp.exp(ln_nk), jnp.exp(ln_mk), jnp.exp(ln_ntot), formula_matrix, formula_matrix_cond, b, gk
        )
        gas_variant = build_rgie_gas_direction_variant(
            variant_name,
            delta_ln_nk_current=step_metrics["delta_ln_nk"],
            delta_ln_ntot_current=step_metrics["delta_ln_ntot"],
            delta_ln_nk_ref=gas_ref["delta_ln_nk_ref"],
            delta_ln_ntot_ref=gas_ref["delta_ln_ntot_ref"],
        )
        lam1_gas_variant = stepsize_cea_gas(
            gas_variant["delta_ln_nk"], gas_variant["delta_ln_ntot"], ln_nk, ln_ntot
        )
        lam_heuristic = float(min(1.0, float(lam1_gas_variant), float(step_metrics["lam1_cond"]), float(step_metrics["lam2_cond"])))
        trial_result = None
        for lam in (lam_heuristic, lam_heuristic * 0.5, lam_heuristic * 0.2, lam_heuristic * 0.1, 0.0):
            trial = _evaluate_trial_step(
                ln_nk,
                ln_mk,
                ln_ntot,
                lam,
                gas_variant["delta_ln_nk"],
                step_metrics["delta_ln_mk"],
                gas_variant["delta_ln_ntot"],
                formula_matrix,
                formula_matrix_cond,
                b,
                state.temperature,
                state.ln_normalized_pressure,
                hvector,
                hvector_cond,
                epsilon,
            )
            if bool(trial["all_finite"]) and math.isfinite(float(trial["fresh_residual"])):
                trial_result = trial
                break
        if trial_result is None:
            break
        history.append(
            {
                "iter": iter_count,
                "lam_heuristic": lam_heuristic,
                "lam_selected": float(trial_result["lam"]),
                "residual_before": residual,
                "residual_after": float(trial_result["fresh_residual"]),
                "line_search_accept_kind": "monotone" if float(trial_result["lam"]) > 0.0 else "zero_step",
            }
        )
        ln_nk = trial_result["ln_nk"]
        ln_mk = trial_result["ln_mk"]
        ln_ntot = trial_result["ln_ntot"]
        residual = float(trial_result["fresh_residual"])
        if float(trial_result["lam"]) <= 1.0e-14:
            break

    return {
        "variant_name": variant_name,
        "converged": residual <= residual_crit,
        "n_iter": len(history),
        "final_residual": residual,
        "hit_max_iter": len(history) >= max_iter and residual > residual_crit,
        "fraction_monotone_residual_decrease": _fraction([
            step["residual_after"] <= step["residual_before"] + 1.0e-15 for step in history
        ]),
        "fraction_accept_best_finite_fallback": 0.0,
        "fraction_accept_zero_step": _fraction([step["line_search_accept_kind"] == "zero_step" for step in history]),
        "history": history,
    }


def _print_terminal_table(rows: list[dict[str, Any]]) -> None:
    headers = ("startup", "trace<1e-30", "trace<1e-25", "trace<1e-20", "ntot culprit", "ref>full")
    print(" ".join(f"{header:>18}" for header in headers))
    for row in rows:
        def _fmt(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, str):
                return value
            return f"{float(value):.3f}"
        print(
            f"{row['startup_policy_name']:>18} "
            f"{_fmt(row['fraction_top_trace_vmr_lt_1e_30']):>18} "
            f"{_fmt(row['fraction_top_trace_vmr_lt_1e_25']):>18} "
            f"{_fmt(row['fraction_top_trace_vmr_lt_1e_20']):>18} "
            f"{_fmt(row['fraction_partial_ntot_or_no_ntot_beats_full']):>18} "
            f"{_fmt(row['fraction_ref_beats_full']):>18}"
        )


def _decision_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace30 = max((row["fraction_top_trace_vmr_lt_1e_30"] or 0.0) for row in summary_rows)
    ntot_better = max((row["fraction_partial_ntot_or_no_ntot_beats_full"] or 0.0) for row in summary_rows)
    ref_better = max((row["fraction_ref_beats_full"] or 0.0) for row in summary_rows)
    messages = []
    if trace30 >= 0.7:
        messages.append("lam1_gas is dominated by trace-species control")
    if ntot_better >= 0.5:
        messages.append("delta_ln_ntot coupling is a major RGIE bottleneck")
    if ref_better >= 0.5:
        messages.append("current coupled RGIE gas direction is the main remaining bottleneck")
    if not messages:
        messages.append("gas-side globalization is not the main issue; inspect another nonlinear layer")
    next_move = (
        "modify gas/global RGIE direction logic"
        if any(
            msg in messages
            for msg in (
                "lam1_gas is dominated by trace-species control",
                "delta_ln_ntot coupling is a major RGIE bottleneck",
                "current coupled RGIE gas direction is the main remaining bottleneck",
            )
        )
        else "keep RGIE gas/global logic and inspect another bottleneck"
    )
    return {"messages": messages, "next_move": next_move}


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
            support_indices = jnp.asarray([cond_setup.species.index(name) for name in layer_meta["support_names"]], dtype=jnp.int32)
            formula_matrix = jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64)
            formula_matrix_cond = jnp.asarray(cond_setup.formula_matrix[:, support_indices], dtype=jnp.float64)
            hvector_cond = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
            b = jnp.asarray(state.element_vector, dtype=jnp.float64)

            for epsilon in args.epsilons:
                for startup in STARTUP_POLICIES:
                    ln_nk = gas_init["ln_nk"]
                    ln_ntot = gas_init["ln_ntot"]
                    ln_mk = build_rgie_condensate_init_from_policy(
                        epsilon=epsilon,
                        support_indices=support_indices,
                        startup_policy=startup["policy"],
                        dtype=jnp.float64,
                        **startup["kwargs"],
                    )
                    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, gas_hvector, state.ln_normalized_pressure)
                    step_metrics = _compute_iteration_step_metrics(
                        ln_nk, ln_mk, ln_ntot, formula_matrix, formula_matrix_cond, b, gk, hvector_cond, epsilon
                    )
                    gas_limiter = _compute_gas_limiter_species_diagnostics(
                        ln_nk, ln_ntot, step_metrics["delta_ln_nk"], step_metrics["delta_ln_ntot"], step_metrics["lam1_gas"],
                        species_names=gas_setup.species, top_k=10
                    )
                    gas_ref = _compute_frozen_condensate_gas_direction_reference(
                        jnp.exp(ln_nk), jnp.exp(ln_mk), jnp.exp(ln_ntot), formula_matrix, formula_matrix_cond, b, gk
                    )
                    current_residual = _compute_current_residual(
                        state, ln_nk, ln_mk, ln_ntot, formula_matrix, formula_matrix_cond, b, gas_hvector, hvector_cond, epsilon
                    )

                    variant_results = {}
                    for variant_name in GAS_VARIANTS:
                        variant = build_rgie_gas_direction_variant(
                            variant_name,
                            delta_ln_nk_current=step_metrics["delta_ln_nk"],
                            delta_ln_ntot_current=step_metrics["delta_ln_ntot"],
                            delta_ln_nk_ref=gas_ref["delta_ln_nk_ref"],
                            delta_ln_ntot_ref=gas_ref["delta_ln_ntot_ref"],
                        )
                        lam1_variant = stepsize_cea_gas(variant["delta_ln_nk"], variant["delta_ln_ntot"], ln_nk, ln_ntot)
                        variant_results[variant_name] = {
                            "gas_direction_norm": float(jnp.linalg.norm(variant["delta_ln_nk"])),
                            "cosine_vs_current": _safe_cosine(variant["delta_ln_nk"], step_metrics["delta_ln_nk"]),
                            "angle_vs_current": None
                            if _safe_cosine(variant["delta_ln_nk"], step_metrics["delta_ln_nk"]) is None
                            else float(math.degrees(math.acos(_safe_cosine(variant["delta_ln_nk"], step_metrics["delta_ln_nk"])))),
                            "max_abs_delta_ln_nk_diff_vs_current": float(jnp.max(jnp.abs(variant["delta_ln_nk"] - step_metrics["delta_ln_nk"]))),
                            "delta_ln_ntot_diff_vs_current": float(jnp.abs(variant["delta_ln_ntot"] - step_metrics["delta_ln_ntot"])),
                            "lam1_gas_variant": float(lam1_variant),
                            "fresh_residual_audit": _evaluate_gas_variant_lambda_grid(
                                state,
                                ln_nk=ln_nk,
                                ln_mk=ln_mk,
                                ln_ntot=ln_ntot,
                                formula_matrix=formula_matrix,
                                formula_matrix_cond=formula_matrix_cond,
                                b=b,
                                hvector=gas_hvector,
                                hvector_cond=hvector_cond,
                                epsilon=epsilon,
                                gas_variant=variant,
                                delta_ln_mk_current=step_metrics["delta_ln_mk"],
                                lambda_trials=lambda_trials,
                                current_residual=current_residual,
                                production_lam=float(step_metrics["lam"]),
                            ),
                        }

                    trace_sensitivity = {
                        f"{floor:.0e}": compute_rgie_lam1_gas_ignore_trace_diagnostics(
                            ln_nk=ln_nk,
                            ln_ntot=ln_ntot,
                            delta_ln_nk=step_metrics["delta_ln_nk"],
                            delta_ln_ntot=step_metrics["delta_ln_ntot"],
                            vmr_floor=floor,
                        )
                        for floor in TRACE_FLOORS
                    }
                    for key, diag in trace_sensitivity.items():
                        variant = build_rgie_gas_direction_variant(
                            "current_full_direction",
                            delta_ln_nk_current=step_metrics["delta_ln_nk"],
                            delta_ln_ntot_current=step_metrics["delta_ln_ntot"],
                            delta_ln_nk_ref=gas_ref["delta_ln_nk_ref"],
                            delta_ln_ntot_ref=gas_ref["delta_ln_ntot_ref"],
                        )
                        diag["best_fresh_residual_ignore_trace_cap"] = _evaluate_gas_variant_lambda_grid(
                            state,
                            ln_nk=ln_nk,
                            ln_mk=ln_mk,
                            ln_ntot=ln_ntot,
                            formula_matrix=formula_matrix,
                            formula_matrix_cond=formula_matrix_cond,
                            b=b,
                            hvector=gas_hvector,
                            hvector_cond=hvector_cond,
                            epsilon=epsilon,
                            gas_variant=variant,
                            delta_ln_mk_current=step_metrics["delta_ln_mk"],
                            lambda_trials=[lam for lam in lambda_trials if lam <= diag["lam1_gas_ignore_trace"] + 1.0e-15],
                            current_residual=current_residual,
                            production_lam=float(step_metrics["lam"]),
                        )["best_fresh_residual"]

                    top_species = gas_limiter["top_species"]
                    case_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "lam1_gas": float(step_metrics["lam1_gas"]),
                            "gas_limiter": gas_limiter,
                            "fraction_top_species_vmr_lt_1e_30": _fraction([species["vmr"] < 1.0e-30 for species in top_species]),
                            "fraction_top_species_vmr_lt_1e_25": _fraction([species["vmr"] < 1.0e-25 for species in top_species]),
                            "fraction_top_species_vmr_lt_1e_20": _fraction([species["vmr"] < 1.0e-20 for species in top_species]),
                            "trace_guard_dominates": bool(
                                sum(1 for species in top_species if species["trace_guard_active"]) >= max(1, len(top_species) // 2)
                            ),
                            "variant_results": variant_results,
                            "trace_sensitivity": trace_sensitivity,
                        }
                    )
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "gas_limiter": gas_limiter,
                            "variant_results": variant_results,
                            "trace_sensitivity": trace_sensitivity,
                        }
                    )

        summary_rows = []
        for startup in STARTUP_POLICIES:
            matching = [case for case in case_records if case["startup_policy_name"] == startup["name"]]
            summary_rows.append(
                {
                    "startup_policy_name": startup["name"],
                    "fraction_top_trace_vmr_lt_1e_30": _mean([case["fraction_top_species_vmr_lt_1e_30"] for case in matching]),
                    "fraction_top_trace_vmr_lt_1e_25": _mean([case["fraction_top_species_vmr_lt_1e_25"] for case in matching]),
                    "fraction_top_trace_vmr_lt_1e_20": _mean([case["fraction_top_species_vmr_lt_1e_20"] for case in matching]),
                    "fraction_trace_guard_dominates": _fraction([case["trace_guard_dominates"] for case in matching]),
                    "fraction_partial_ntot_or_no_ntot_beats_full": _fraction([
                        any(
                            case["variant_results"][name]["fresh_residual_audit"]["best_fresh_residual"] is not None
                            and case["variant_results"]["current_full_direction"]["fresh_residual_audit"]["best_fresh_residual"] is not None
                            and case["variant_results"][name]["fresh_residual_audit"]["best_fresh_residual"]
                            < case["variant_results"]["current_full_direction"]["fresh_residual_audit"]["best_fresh_residual"]
                            for name in ("no_common_ntot_shift", "partial_ntot_shift_0p25", "partial_ntot_shift_0p5")
                        )
                        for case in matching
                    ]),
                    "fraction_ref_beats_full": _fraction([
                        case["variant_results"]["frozen_condensate_gas_only_reference"]["fresh_residual_audit"]["best_fresh_residual"] is not None
                        and case["variant_results"]["current_full_direction"]["fresh_residual_audit"]["best_fresh_residual"] is not None
                        and case["variant_results"]["frozen_condensate_gas_only_reference"]["fresh_residual_audit"]["best_fresh_residual"]
                        < case["variant_results"]["current_full_direction"]["fresh_residual_audit"]["best_fresh_residual"]
                        for case in matching
                    ]),
                }
            )

        trajectory_runs = []
        focus_case_layer = 45
        focus_case_epsilon = -5.0
        layer_meta = next(item for item in profile_states if item["layer_index"] == focus_case_layer)
        state = layer_meta["state"]
        gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
        cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
        support_indices = jnp.asarray([cond_setup.species.index(name) for name in layer_meta["support_names"]], dtype=jnp.int32)
        formula_matrix = jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64)
        formula_matrix_cond = jnp.asarray(cond_setup.formula_matrix[:, support_indices], dtype=jnp.float64)
        hvector_cond = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
        b = jnp.asarray(state.element_vector, dtype=jnp.float64)
        gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
        for startup in STARTUP_POLICIES:
            ln_mk = build_rgie_condensate_init_from_policy(
                epsilon=focus_case_epsilon, support_indices=support_indices, startup_policy=startup["policy"], dtype=jnp.float64, **startup["kwargs"]
            )
            for variant_name in ("current_full_direction", "frozen_condensate_gas_only_reference", "partial_ntot_shift_0p5"):
                trajectory_runs.append(
                    {
                        "startup_policy_name": startup["name"],
                        **_run_simple_gas_variant_trajectory(
                            state,
                            ln_nk_init=gas_init["ln_nk"],
                            ln_mk_init=ln_mk,
                            ln_ntot_init=gas_init["ln_ntot"],
                            formula_matrix=formula_matrix,
                            formula_matrix_cond=formula_matrix_cond,
                            b=b,
                            hvector=gas_hvector,
                            hvector_cond=hvector_cond,
                            epsilon=focus_case_epsilon,
                            variant_name=variant_name,
                            max_iter=args.trajectory_max_iter,
                        ),
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
            "startup_policies": [dict(policy) for policy in STARTUP_POLICIES],
            "gas_variants": list(GAS_VARIANTS),
            "trace_floors": list(TRACE_FLOORS),
            "cases": case_records,
            "summary_rows": summary_rows,
            "trajectory_runs": trajectory_runs,
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "traces": trace_records,
            "trajectory_runs": trajectory_runs,
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
