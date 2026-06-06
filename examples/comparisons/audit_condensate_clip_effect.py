"""Audit the effect of condensate-direction clipping for patched PIPM solvers.

This diagnostic keeps production defaults unchanged. It compares the patched
reduced RGIE path and patched full GIE path on the same frozen-support initial
states, varying only the condensate-direction treatment.

Example:

    PYTHONPATH=src python examples/comparisons/audit_condensate_clip_effect.py --platform cpu
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
from exogibbs.optimize.pipm_gie_cond import _choose_lambda_by_residual_backtracking as choose_gie_lambda
from exogibbs.optimize.pipm_gie_cond import _compute_iteration_step_metrics as compute_gie_step_metrics
from exogibbs.optimize.pipm_gie_cond import _compute_residuals as compute_gie_residuals
from exogibbs.optimize.pipm_gie_cond import _contains_invalid_numbers as gie_contains_invalid_numbers
from exogibbs.optimize.pipm_gie_cond import _evaluate_trial_step as evaluate_gie_trial_step
from exogibbs.optimize.pipm_gie_cond import _recompute_pi_for_residual as recompute_gie_pi
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_MODE
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REGULARIZATION_STRENGTH
from exogibbs.optimize.pipm_rgie_cond import DEFAULT_REDUCED_SOLVER
from exogibbs.optimize.pipm_rgie_cond import _choose_lambda_by_residual_backtracking as choose_rgie_lambda
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics as compute_rgie_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals as compute_rgie_residuals
from exogibbs.optimize.pipm_rgie_cond import _contains_invalid_numbers as rgie_contains_invalid_numbers
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step as evaluate_rgie_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual as recompute_rgie_pi
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.optimize.stepsize import stepsize_cond_heurstic
from exogibbs.optimize.stepsize import stepsize_sk
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_TRAJECTORY_VARIANTS = (
    "current_component_clip_0p1",
    "component_clip_0p5",
    "cond_block_scalar_rescale_0p1",
    "cond_block_scalar_rescale_0p5",
    "no_clip",
)
DEFAULT_INITIAL_AMOUNT = 1.0e-30
DEFAULT_OUTPUT = REPO_ROOT / "results" / "condensate_clip_effect_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "condensate_clip_effect_traces.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        default="cpu",
        choices=("cpu", "gpu"),
        help="JAX platform selection. CPU is recommended for this sequential audit.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYER_INDICES),
        help="Layer indices to audit. Defaults to the hard-layer set used in the line-search audit.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="Barrier epsilons to audit. Defaults to 0, -5, -10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the summary JSON report.",
    )
    parser.add_argument(
        "--traces-output",
        type=Path,
        default=DEFAULT_TRACES_OUTPUT,
        help="Path for the per-iteration traces JSON report.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=8,
        help="Maximum diagnostic condensate iterations per trajectory run.",
    )
    parser.add_argument(
        "--gas-max-iter",
        type=int,
        default=1000,
        help="Maximum gas-equilibrium iterations used to build shared initial states.",
    )
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
        "support_size": int(support_cfg["final_support_size"]),
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
                "support_size": support_case["support_size"],
                "support_source_json": support_case["support_source_json"],
                "state": state,
            }
        )
    return states


def _build_shared_condensate_init(
    state: ThermoState,
    gas_setup: Any,
    cond_setup: Any,
    support_names: list[str],
    gas_hvector: jax.Array,
    cond_hvector_full: jax.Array,
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
    support_indices = jnp.asarray(
        [cond_setup.species.index(name) for name in support_names],
        dtype=jnp.int32,
    )
    formula_matrix_cond = cond_setup.formula_matrix[:, support_indices]
    gas_hvector_subset = jnp.asarray(gas_hvector, dtype=jnp.float64)
    cond_hvector_subset = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
    return {
        "ln_nk": jnp.asarray(ln_nk_gas, dtype=jnp.float64),
        "ln_mk": jnp.full((len(support_names),), jnp.log(DEFAULT_INITIAL_AMOUNT), dtype=jnp.float64),
        "ln_ntot": jnp.asarray(ln_ntot_gas, dtype=jnp.float64),
        "formula_matrix_cond": formula_matrix_cond,
        "support_indices": support_indices,
        "support_names": list(support_names),
        "gas_hvector_func": lambda _temperature: gas_hvector_subset,
        "hvector_cond_func": lambda _temperature: cond_hvector_subset,
        "gas_init_n_iter": int(gas_n_iter),
        "gas_init_final_residual": float(gas_final_residual),
    }


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


def _safe_float(value: Any) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _safe_cosine(a: jax.Array, b: jax.Array) -> float | None:
    a = jnp.ravel(jnp.asarray(a, dtype=jnp.float64))
    b = jnp.ravel(jnp.asarray(b, dtype=jnp.float64))
    denom = float(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    if not math.isfinite(denom) or denom <= 1.0e-300:
        return None
    cosine = float(jnp.dot(a, b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return cosine


def _safe_angle_degrees(a: jax.Array, b: jax.Array) -> float | None:
    cosine = _safe_cosine(a, b)
    if cosine is None:
        return None
    return float(math.degrees(math.acos(cosine)))


def _variant_direction(raw_delta_ln_mk: jax.Array, variant_name: str) -> dict[str, Any]:
    raw = jnp.asarray(raw_delta_ln_mk, dtype=jnp.float64)
    n_cond = int(raw.shape[0])
    max_abs_raw = float(jnp.max(jnp.abs(raw))) if n_cond else 0.0

    if variant_name == "current_component_clip_0p1":
        limit = 0.1
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
        alpha = None
    elif variant_name == "component_clip_0p5":
        limit = 0.5
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
        alpha = None
    elif variant_name == "cond_block_scalar_rescale_0p1":
        limit = 0.1
        alpha = 1.0 if max_abs_raw <= limit else limit / max_abs_raw
        delta = raw * alpha
        saturated = None
    elif variant_name == "cond_block_scalar_rescale_0p5":
        limit = 0.5
        alpha = 1.0 if max_abs_raw <= limit else limit / max_abs_raw
        delta = raw * alpha
        saturated = None
    elif variant_name == "no_clip":
        delta = raw
        saturated = None
        alpha = None
    else:
        raise ValueError(f"Unknown condensate direction variant: {variant_name}")

    if saturated is None:
        n_saturated = None
        saturated_fraction = None
    else:
        n_saturated = int(jnp.sum(saturated))
        saturated_fraction = float(n_saturated / n_cond) if n_cond else 0.0

    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta,
        "n_cond": n_cond,
        "n_saturated_components": n_saturated,
        "saturated_fraction": saturated_fraction,
        "scalar_alpha": alpha,
        "max_abs_raw_delta_ln_mk": max_abs_raw,
        "max_abs_variant_delta_ln_mk": float(jnp.max(jnp.abs(delta))) if n_cond else 0.0,
        "raw_cond_direction_norm": float(jnp.linalg.norm(raw)),
        "variant_cond_direction_norm": float(jnp.linalg.norm(delta)),
        "cosine_raw_vs_variant": _safe_cosine(raw, delta),
        "angle_degrees_raw_vs_variant": _safe_angle_degrees(raw, delta),
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
            "choose_lambda": choose_rgie_lambda,
            "recompute_pi": recompute_rgie_pi,
            "compute_residuals": compute_rgie_residuals,
            "contains_invalid_numbers": rgie_contains_invalid_numbers,
            "extra_kwargs": extra,
        }
    if name == "GIE":
        return {
            "name": name,
            "compute_step_metrics": compute_gie_step_metrics,
            "evaluate_trial_step": evaluate_gie_trial_step,
            "choose_lambda": choose_gie_lambda,
            "recompute_pi": recompute_gie_pi,
            "compute_residuals": compute_gie_residuals,
            "contains_invalid_numbers": gie_contains_invalid_numbers,
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
        "residual": residual,
        "invalid_numbers_detected": invalid_numbers_detected,
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
    variant_delta_ln_mk: jax.Array,
    delta_ln_ntot: float,
    lambda_trials: tuple[float, ...],
) -> dict[str, Any]:
    trials = []
    best_record = None
    any_improves = False
    finite_current = math.isfinite(float(current_residual))

    for lambda_trial in lambda_trials:
        trial = bundle["evaluate_trial_step"](
            ln_nk,
            ln_mk,
            ln_ntot,
            jnp.asarray(lambda_trial, dtype=jnp.float64),
            delta_ln_nk,
            variant_delta_ln_mk,
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
        monotone = valid_trial and (
            fresh_residual <= float(current_residual) + 1.0e-15 if finite_current else True
        )
        any_improves = any_improves or monotone
        record = {
            "lam": float(trial["lam"]),
            "fresh_residual": fresh_residual,
            "valid_trial": valid_trial,
            "sk_feasible": sk_feasible,
            "monotone_vs_current": monotone,
            "invalid_numbers_detected": not valid_trial,
        }
        trials.append(record)
        if valid_trial and (
            best_record is None or record["fresh_residual"] < best_record["fresh_residual"]
        ):
            best_record = record

    valid_trial_count = sum(1 for trial in trials if trial["valid_trial"])
    invalid_trial_count = len(trials) - valid_trial_count
    return {
        "lambda_trials": trials,
        "valid_trial_count": int(valid_trial_count),
        "invalid_trial_count": int(invalid_trial_count),
        "best_lambda": None if best_record is None else best_record["lam"],
        "best_fresh_residual": None if best_record is None else best_record["fresh_residual"],
        "any_trial_improves_residual": bool(any_improves),
        "best_trial_sk_feasible": None if best_record is None else bool(best_record["sk_feasible"]),
    }


def _accept_kind_from_code(code: int) -> str:
    if code == 0:
        return "monotone"
    if code == 1:
        return "best_finite_fallback"
    return "zero_step"


def _run_variant_trajectory(
    bundle: dict[str, Any],
    *,
    state: ThermoState,
    init_state: dict[str, Any],
    formula_matrix: jax.Array,
    epsilon: float,
    max_iter: int,
) -> dict[str, Any]:
    formula_matrix_cond = init_state["formula_matrix_cond"]
    hvector = init_state["gas_hvector_func"](state.temperature)
    hvector_cond = init_state["hvector_cond_func"](state.temperature)
    b = jnp.asarray(state.element_vector)

    runs = []
    for variant_name in DEFAULT_TRAJECTORY_VARIANTS:
        ln_nk = jnp.asarray(init_state["ln_nk"])
        ln_mk = jnp.asarray(init_state["ln_mk"])
        ln_ntot = jnp.asarray(init_state["ln_ntot"])
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
        residual = current["residual"]
        history = []
        residual_crit = float(math.exp(float(epsilon)))

        for iter_count in range(max_iter):
            if float(residual) <= residual_crit:
                break

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
            lam1_gas = stepsize_cea_gas(
                step_metrics["delta_ln_nk"],
                step_metrics["delta_ln_ntot"],
                ln_nk,
                ln_ntot,
            )
            lam1_cond = stepsize_cond_heurstic(variant["delta_ln_mk"])
            lam2_cond = stepsize_sk(variant["delta_ln_mk"], ln_mk, epsilon)
            lam_heuristic = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
            lam_heuristic = jnp.clip(lam_heuristic, 0.0, 1.0)

            line_search = bundle["choose_lambda"](
                ln_nk=ln_nk,
                ln_mk=ln_mk,
                ln_ntot=ln_ntot,
                current_gk=current["gk"],
                current_An=current["An"],
                current_Am=current["Am"],
                current_residual=residual,
                lam_init=lam_heuristic,
                delta_ln_nk=step_metrics["delta_ln_nk"],
                delta_ln_mk=variant["delta_ln_mk"],
                delta_ln_ntot=step_metrics["delta_ln_ntot"],
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                b=b,
                temperature=state.temperature,
                ln_normalized_pressure=state.ln_normalized_pressure,
                hvector=hvector,
                hvector_cond=hvector_cond,
                epsilon=epsilon,
                **bundle["extra_kwargs"],
            )
            accept_kind = _accept_kind_from_code(int(line_search["accept_code"]))
            lam_selected = float(line_search["lam"])
            residual_before = float(residual)
            residual_after = float(line_search["fresh_residual"])
            history.append(
                {
                    "iter": iter_count,
                    "variant": variant_name,
                    "lam_heuristic": float(lam_heuristic),
                    "lam_selected": lam_selected,
                    "lam1_gas": float(lam1_gas),
                    "lam1_cond": float(lam1_cond),
                    "lam2_cond": float(lam2_cond),
                    "n_backtracks": int(line_search["n_backtracks"]),
                    "line_search_accept_kind": accept_kind,
                    "residual_before": residual_before,
                    "residual_after": residual_after,
                    "max_abs_raw_delta_ln_mk": variant["max_abs_raw_delta_ln_mk"],
                    "max_abs_variant_delta_ln_mk": variant["max_abs_variant_delta_ln_mk"],
                    "cosine_raw_vs_variant": variant["cosine_raw_vs_variant"],
                    "angle_degrees_raw_vs_variant": variant["angle_degrees_raw_vs_variant"],
                }
            )

            ln_nk = line_search["ln_nk"]
            ln_mk = line_search["ln_mk"]
            ln_ntot = line_search["ln_ntot"]
            current = {
                "gk": line_search["gk"],
                "An": line_search["An"],
                "Am": line_search["Am"],
                "residual": line_search["fresh_residual"],
            }
            residual = line_search["fresh_residual"]
            if lam_selected <= 1.0e-14:
                break

        invalid_numbers_detected = bool(
            bundle["contains_invalid_numbers"](ln_nk, ln_mk, ln_ntot, residual)
        )
        monotone_fraction = _fraction(
            [
                math.isfinite(step["residual_before"])
                and step["residual_after"] <= step["residual_before"] + 1.0e-15
                for step in history
            ]
        )
        best_fallback_fraction = _fraction(
            [step["line_search_accept_kind"] == "best_finite_fallback" for step in history]
        )
        zero_step_fraction = _fraction(
            [step["line_search_accept_kind"] == "zero_step" for step in history]
        )
        lam_reduced_fraction = _fraction(
            [step["lam_selected"] < step["lam_heuristic"] - 1.0e-15 for step in history]
        )
        runs.append(
            {
                "solver": bundle["name"],
                "variant": variant_name,
                "converged": bool(float(residual) <= residual_crit),
                "n_iter": len(history),
                "final_residual": float(residual),
                "hit_max_iter": bool(len(history) >= max_iter and float(residual) > residual_crit),
                "invalid_numbers_detected": invalid_numbers_detected,
                "fraction_monotone_residual_decrease": monotone_fraction,
                "fraction_best_finite_fallback": best_fallback_fraction,
                "fraction_zero_step": zero_step_fraction,
                "fraction_lam_selected_lt_lam_heuristic": lam_reduced_fraction,
                "history": history,
            }
        )
    return {
        "runs": runs,
    }


def _summarize_trajectory_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for solver_name in ("RGIE", "GIE"):
        for variant_name in DEFAULT_TRAJECTORY_VARIANTS:
            matching = [
                run for run in runs if run["solver"] == solver_name and run["variant"] == variant_name
            ]
            summaries.append(
                {
                    "solver": solver_name,
                    "variant": variant_name,
                    "convergence_rate": _fraction([run["converged"] for run in matching]),
                    "mean_n_iter": _mean([run["n_iter"] for run in matching]),
                    "median_n_iter": _median([run["n_iter"] for run in matching]),
                    "mean_final_residual": _mean([run["final_residual"] for run in matching]),
                    "median_final_residual": _median([run["final_residual"] for run in matching]),
                    "mean_fraction_monotone_residual_decrease": _mean(
                        [
                            run["fraction_monotone_residual_decrease"]
                            for run in matching
                            if run["fraction_monotone_residual_decrease"] is not None
                        ]
                    ),
                    "mean_fraction_best_finite_fallback": _mean(
                        [
                            run["fraction_best_finite_fallback"]
                            for run in matching
                            if run["fraction_best_finite_fallback"] is not None
                        ]
                    ),
                    "mean_fraction_zero_step": _mean(
                        [run["fraction_zero_step"] for run in matching if run["fraction_zero_step"] is not None]
                    ),
                    "mean_fraction_lam_selected_lt_lam_heuristic": _mean(
                        [
                            run["fraction_lam_selected_lt_lam_heuristic"]
                            for run in matching
                            if run["fraction_lam_selected_lt_lam_heuristic"] is not None
                        ]
                    ),
                }
            )
    return summaries


def _print_terminal_table(trajectory_summaries: list[dict[str, Any]]) -> None:
    headers = (
        "solver",
        "variant",
        "conv",
        "mean_iter",
        "median_iter",
        "mean_final_res",
        "lam<heur",
        "mono",
        "best_fb",
        "zero",
    )
    rows = [headers]
    for summary in trajectory_summaries:
        rows.append(
            (
                summary["solver"],
                summary["variant"],
                "-" if summary["convergence_rate"] is None else f"{summary['convergence_rate']:.2f}",
                "-" if summary["mean_n_iter"] is None else f"{summary['mean_n_iter']:.2f}",
                "-" if summary["median_n_iter"] is None else f"{summary['median_n_iter']:.2f}",
                "-"
                if summary["mean_final_residual"] is None
                else f"{summary['mean_final_residual']:.3e}",
                "-"
                if summary["mean_fraction_lam_selected_lt_lam_heuristic"] is None
                else f"{summary['mean_fraction_lam_selected_lt_lam_heuristic']:.2f}",
                "-"
                if summary["mean_fraction_monotone_residual_decrease"] is None
                else f"{summary['mean_fraction_monotone_residual_decrease']:.2f}",
                "-"
                if summary["mean_fraction_best_finite_fallback"] is None
                else f"{summary['mean_fraction_best_finite_fallback']:.2f}",
                "-"
                if summary["mean_fraction_zero_step"] is None
                else f"{summary['mean_fraction_zero_step']:.2f}",
            )
        )
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    for idx, row in enumerate(rows):
        line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def _decision_summary(
    *,
    one_step_records: list[dict[str, Any]],
    trajectory_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    by_solver_variant = {
        (summary["solver"], summary["variant"]): summary
        for summary in trajectory_summaries
    }

    scalar_beats_component = False
    for solver_name in ("RGIE", "GIE"):
        component_summary = by_solver_variant.get((solver_name, "current_component_clip_0p1"))
        scalar_summary = by_solver_variant.get((solver_name, "cond_block_scalar_rescale_0p1"))
        no_clip_summary = by_solver_variant.get((solver_name, "no_clip"))
        if (
            component_summary is not None
            and scalar_summary is not None
            and component_summary["mean_final_residual"] is not None
            and scalar_summary["mean_final_residual"] is not None
            and scalar_summary["mean_final_residual"] < component_summary["mean_final_residual"]
        ):
            scalar_beats_component = True
        if (
            component_summary is not None
            and no_clip_summary is not None
            and component_summary["mean_final_residual"] is not None
            and no_clip_summary["mean_final_residual"] is not None
            and no_clip_summary["mean_final_residual"] < component_summary["mean_final_residual"]
        ):
            scalar_beats_component = True

    raw_vs_variant_masking = False
    no_clip_failure_with_scalar_help = False
    all_variants_similar = True
    for record in one_step_records:
        raw_cosine = record["cross_solver_masking"]["raw_cosine"]
        for variant_name, cross in record["cross_solver_masking"]["variants"].items():
            clipped_cosine = cross["variant_cosine"]
            if raw_cosine is not None and clipped_cosine is not None and clipped_cosine - raw_cosine > 0.1:
                raw_vs_variant_masking = True
        gie_no_clip = record["trajectory_snapshot"]["GIE"]["no_clip"]
        gie_scalar = record["trajectory_snapshot"]["GIE"]["cond_block_scalar_rescale_0p1"]
        gie_component = record["trajectory_snapshot"]["GIE"]["current_component_clip_0p1"]
        if (
            gie_no_clip["invalid_numbers_detected"]
            and not gie_scalar["invalid_numbers_detected"]
            and gie_scalar["final_residual"] < gie_component["final_residual"]
        ):
            no_clip_failure_with_scalar_help = True
    baseline_values = []
    for summary in trajectory_summaries:
        if summary["mean_final_residual"] is not None:
            baseline_values.append(summary["mean_final_residual"])
    if baseline_values:
        spread = max(baseline_values) / max(min(baseline_values), 1.0e-300)
        all_variants_similar = spread < 1.2

    if scalar_beats_component:
        primary = "componentwise clipping is a likely bottleneck"
    elif raw_vs_variant_masking:
        primary = "clipping is masking full-vs-reduced direction differences"
    elif no_clip_failure_with_scalar_help:
        primary = "step magnitude control is needed, but componentwise clipping is unnecessarily destructive"
    else:
        primary = "direction quality is the main bottleneck, not clipping"

    if no_clip_failure_with_scalar_help and primary != "step magnitude control is needed, but componentwise clipping is unnecessarily destructive":
        secondary = "step magnitude control is needed, but componentwise clipping is unnecessarily destructive"
    elif all_variants_similar and primary != "direction quality is the main bottleneck, not clipping":
        secondary = "direction quality is the main bottleneck, not clipping"
    else:
        secondary = None

    return {
        "primary": primary,
        "secondary": secondary,
        "componentwise_clipping_likely_bottleneck": scalar_beats_component,
        "clipping_masks_full_vs_reduced_differences": raw_vs_variant_masking,
        "all_variants_similar": all_variants_similar,
        "no_clip_fails_but_scalar_rescale_helps": no_clip_failure_with_scalar_help,
    }


def main() -> int:
    args = build_parser().parse_args()
    backend, device = device_for_platform(args.platform)
    with jax.default_device(device):
        gas_setup = gas_chemsetup(silent=True)
        cond_setup = cond_chemsetup(silent=True)
        profile_states = _build_profile_states(gas_setup.element_vector_reference, args.layers)
        one_step_records = []
        trajectory_runs = []
        for layer_meta in profile_states:
            state = layer_meta["state"]
            gas_hvector = gas_setup.hvector_func(state.temperature)
            cond_hvector_full = cond_setup.hvector_func(state.temperature)
            init_state = _build_shared_condensate_init(
                state,
                gas_setup,
                cond_setup,
                layer_meta["support_names"],
                gas_hvector,
                cond_hvector_full,
                args.gas_max_iter,
            )
            formula_matrix = gas_setup.formula_matrix
            formula_matrix_cond = init_state["formula_matrix_cond"]
            hvector = init_state["gas_hvector_func"](state.temperature)
            hvector_cond = init_state["hvector_cond_func"](state.temperature)
            b = jnp.asarray(state.element_vector)

            for epsilon in args.epsilons:
                solver_case_data = {}
                trajectory_snapshot: dict[str, dict[str, Any]] = {}
                for solver_name in ("RGIE", "GIE"):
                    bundle = _solver_bundle(solver_name)
                    current = _compute_current_state_metrics(
                        bundle,
                        state=state,
                        ln_nk=init_state["ln_nk"],
                        ln_mk=init_state["ln_mk"],
                        ln_ntot=init_state["ln_ntot"],
                        formula_matrix=formula_matrix,
                        formula_matrix_cond=formula_matrix_cond,
                        b=b,
                        hvector=hvector,
                        hvector_cond=hvector_cond,
                        epsilon=epsilon,
                    )
                    step_metrics = bundle["compute_step_metrics"](
                        init_state["ln_nk"],
                        init_state["ln_mk"],
                        init_state["ln_ntot"],
                        formula_matrix,
                        formula_matrix_cond,
                        b,
                        current["gk"],
                        hvector_cond,
                        epsilon,
                        **bundle["extra_kwargs"],
                    )
                    variants = {}
                    raw_delta_ln_mk = step_metrics["raw_delta_ln_mk"]
                    for variant_name in DEFAULT_TRAJECTORY_VARIANTS:
                        variant = _variant_direction(raw_delta_ln_mk, variant_name)
                        variant_eval = _evaluate_variant_lambda_grid(
                            bundle,
                            state=state,
                            ln_nk=init_state["ln_nk"],
                            ln_mk=init_state["ln_mk"],
                            ln_ntot=init_state["ln_ntot"],
                            formula_matrix=formula_matrix,
                            formula_matrix_cond=formula_matrix_cond,
                            b=b,
                            hvector=hvector,
                            hvector_cond=hvector_cond,
                            epsilon=epsilon,
                            current_residual=current["residual"],
                            delta_ln_nk=step_metrics["delta_ln_nk"],
                            variant_delta_ln_mk=variant["delta_ln_mk"],
                            delta_ln_ntot=step_metrics["delta_ln_ntot"],
                            lambda_trials=DEFAULT_LAMBDA_TRIALS,
                        )
                        variants[variant_name] = {
                            **variant,
                            **variant_eval,
                        }
                    solver_case_data[solver_name] = {
                        "current_residual": float(current["residual"]),
                        "delta_ln_ntot": float(step_metrics["delta_ln_ntot"]),
                        "max_abs_delta_ln_nk": float(step_metrics["max_abs_delta_ln_nk"]),
                        "raw_delta_ln_mk": to_python(jax.device_get(raw_delta_ln_mk)),
                        "variants": variants,
                    }

                    trajectory = _run_variant_trajectory(
                        bundle,
                        state=state,
                        init_state=init_state,
                        formula_matrix=formula_matrix,
                        epsilon=epsilon,
                        max_iter=args.max_iter,
                    )
                    for run in trajectory["runs"]:
                        run_record = {
                            "layer_index": layer_meta["layer_index"],
                            "temperature_K": layer_meta["temperature_K"],
                            "pressure_bar": layer_meta["pressure_bar"],
                            "support_names": list(init_state["support_names"]),
                            "epsilon": float(epsilon),
                            **run,
                        }
                        trajectory_runs.append(run_record)
                        trajectory_snapshot.setdefault(solver_name, {})[run["variant"]] = {
                            key: value for key, value in run_record.items() if key != "history"
                        }

                raw_rgie = jnp.asarray(solver_case_data["RGIE"]["raw_delta_ln_mk"], dtype=jnp.float64)
                raw_gie = jnp.asarray(solver_case_data["GIE"]["raw_delta_ln_mk"], dtype=jnp.float64)
                cross_solver = {
                    "raw_cosine": _safe_cosine(raw_rgie, raw_gie),
                    "raw_angle_degrees": _safe_angle_degrees(raw_rgie, raw_gie),
                    "max_abs_raw_delta_ln_mk_diff": float(jnp.max(jnp.abs(raw_rgie - raw_gie))),
                    "variants": {},
                }
                for variant_name in DEFAULT_TRAJECTORY_VARIANTS:
                    rgie_variant = jnp.asarray(
                        solver_case_data["RGIE"]["variants"][variant_name]["delta_ln_mk"],
                        dtype=jnp.float64,
                    )
                    gie_variant = jnp.asarray(
                        solver_case_data["GIE"]["variants"][variant_name]["delta_ln_mk"],
                        dtype=jnp.float64,
                    )
                    cross_solver["variants"][variant_name] = {
                        "variant_cosine": _safe_cosine(rgie_variant, gie_variant),
                        "variant_angle_degrees": _safe_angle_degrees(rgie_variant, gie_variant),
                        "max_abs_variant_delta_ln_mk_diff": float(
                            jnp.max(jnp.abs(rgie_variant - gie_variant))
                        ),
                    }
                    solver_case_data["RGIE"]["variants"][variant_name]["delta_ln_mk"] = to_python(
                        jax.device_get(rgie_variant)
                    )
                    solver_case_data["GIE"]["variants"][variant_name]["delta_ln_mk"] = to_python(
                        jax.device_get(gie_variant)
                    )

                one_step_records.append(
                    {
                        "layer_index": layer_meta["layer_index"],
                        "temperature_K": layer_meta["temperature_K"],
                        "pressure_bar": layer_meta["pressure_bar"],
                        "support_names": list(init_state["support_names"]),
                        "epsilon": float(epsilon),
                        "RGIE": solver_case_data["RGIE"],
                        "GIE": solver_case_data["GIE"],
                        "cross_solver_masking": cross_solver,
                        "trajectory_snapshot": trajectory_snapshot,
                    }
                )

        trajectory_summaries = _summarize_trajectory_runs(trajectory_runs)
        decision = _decision_summary(
            one_step_records=one_step_records,
            trajectory_summaries=trajectory_summaries,
        )

        audit_report = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": list(args.layers),
            "epsilons": list(args.epsilons),
            "lambda_trials": list(DEFAULT_LAMBDA_TRIALS),
            "trajectory_variants": list(DEFAULT_TRAJECTORY_VARIANTS),
            "max_iter": int(args.max_iter),
            "gas_max_iter": int(args.gas_max_iter),
            "one_step_cases": one_step_records,
            "trajectory_summaries": trajectory_summaries,
            "decision": decision,
        }
        traces_report = {
            "timestamp_utc": audit_report["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "trajectory_runs": trajectory_runs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(audit_report), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_report), indent=2))

    _print_terminal_table(trajectory_summaries)
    print()
    print(f"decision: {decision['primary']}")
    if decision["secondary"] is not None:
        print(f"secondary: {decision['secondary']}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
