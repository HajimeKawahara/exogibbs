"""Audit the raw RGIE condensate update formula without changing production behavior."""

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
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_direction_variant
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import diagnose_rgie_raw_condensate_update_block
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_raw_update_formula_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_raw_update_formula_traces.json"

STARTUP_POLICIES = (
    {
        "name": "legacy_absolute_m0_1e-30",
        "policy": "absolute_uniform_m0",
        "kwargs": {"m0": 1.0e-30},
    },
    {
        "name": "ratio_uniform_r0_3e-3",
        "policy": "ratio_uniform_r0",
        "kwargs": {"r0": 3.0e-3},
    },
    {
        "name": "ratio_uniform_r0_1e-2",
        "policy": "ratio_uniform_r0",
        "kwargs": {"r0": 1.0e-2},
    },
)

VARIANTS = (
    "production_clipped_current",
    "raw_current_no_clip",
    "correction_only_no_clip",
    "gas_only",
    "correction_only_scalar_rescale_0p1",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYER_INDICES))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--lambda-trials", type=float, nargs="+", default=list(DEFAULT_LAMBDA_TRIALS))
    parser.add_argument("--gas-max-iter", type=int, default=1000)
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


def _sign_agreement_fraction(a: jax.Array, b: jax.Array) -> float:
    a = jnp.ravel(jnp.asarray(a, dtype=jnp.float64))
    b = jnp.ravel(jnp.asarray(b, dtype=jnp.float64))
    same = jnp.sign(a) == jnp.sign(b)
    return float(jnp.mean(same.astype(jnp.float64)))


def _raw_update_metrics(raw_update: dict[str, Any]) -> dict[str, Any]:
    factor = jnp.asarray(raw_update["factor"], dtype=jnp.float64)
    driving = jnp.asarray(raw_update["driving"], dtype=jnp.float64)
    correction = jnp.asarray(raw_update["correction"], dtype=jnp.float64)
    raw_current = jnp.asarray(raw_update["raw_delta_ln_mk_current"], dtype=jnp.float64)
    ones = jnp.ones_like(raw_current)
    return {
        "nu": float(raw_update["nu"]),
        "raw_identity_max_abs_diff": float(raw_update["raw_identity_max_abs_diff"]),
        "factor_stats": _summary_triplet(factor),
        "driving_stats": _summary_triplet(driving),
        "correction_stats": _summary_triplet(correction),
        "raw_delta_ln_m_current_stats": _summary_triplet(raw_current),
        "fraction_abs_correction_lt_1e_minus_3": float(jnp.mean((jnp.abs(correction) < 1.0e-3).astype(jnp.float64))),
        "fraction_abs_correction_lt_1e_minus_1": float(jnp.mean((jnp.abs(correction) < 1.0e-1).astype(jnp.float64))),
        "fraction_abs_correction_lt_1": float(jnp.mean((jnp.abs(correction) < 1.0).astype(jnp.float64))),
        "fraction_raw_in_0p9_1p1": float(jnp.mean(((raw_current >= 0.9) & (raw_current <= 1.1)).astype(jnp.float64))),
        "fraction_raw_gt_0": float(jnp.mean((raw_current > 0.0).astype(jnp.float64))),
        "fraction_raw_lt_0": float(jnp.mean((raw_current < 0.0).astype(jnp.float64))),
        "cosine_raw_vs_ones": _safe_cosine(raw_current, ones),
        "cosine_correction_vs_ones": _safe_cosine(correction, ones),
        "sign_agreement_fraction": _sign_agreement_fraction(correction, raw_current),
        "norm_raw": float(jnp.linalg.norm(raw_current)),
        "norm_correction": float(jnp.linalg.norm(correction)),
        "norm_ones": float(jnp.linalg.norm(ones)),
        "correction_over_ones_norm_ratio": float(
            jnp.linalg.norm(correction) / jnp.maximum(jnp.linalg.norm(ones), 1.0e-300)
        ),
        "max_abs_raw_minus_correction": float(jnp.max(jnp.abs(raw_current - correction))),
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


def _evaluate_variant_lambda_grid(
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
    delta_ln_ntot: jax.Array,
    variant: dict[str, Any],
    lambda_trials: list[float],
    current_residual: float,
) -> dict[str, Any]:
    delta_ln_mk = jnp.asarray(variant["delta_ln_mk"], dtype=jnp.float64)
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
        trial_ln_mk = jnp.asarray(trial["ln_mk"], dtype=jnp.float64)
        sk_feasible = bool(jnp.all(LOG_S_MAX + epsilon - 2.0 * trial_ln_mk >= 0.0))
        all_finite = bool(jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"])
        trials.append(
            {
                "lambda_trial": float(lam),
                "fresh_residual": float(trial["fresh_residual"]),
                "all_finite": all_finite,
                "sk_feasible": sk_feasible,
                "max_abs_trial_delta_ln_m": float(jnp.max(jnp.abs(float(lam) * delta_ln_mk))),
            }
        )

    valid_trials = [trial for trial in trials if trial["all_finite"]]
    best_trial = None
    if valid_trials:
        best_trial = min(valid_trials, key=lambda trial: trial["fresh_residual"])
    return {
        "variant_name": variant["variant_name"],
        "valid_trial_count": len(valid_trials),
        "invalid_trial_count": len(trials) - len(valid_trials),
        "best_lambda": None if best_trial is None else best_trial["lambda_trial"],
        "best_fresh_residual": None if best_trial is None else best_trial["fresh_residual"],
        "any_trial_monotone_vs_current_residual": any(
            trial["all_finite"] and trial["fresh_residual"] <= current_residual
            for trial in trials
        ),
        "best_trial_sk_feasible": None if best_trial is None else best_trial["sk_feasible"],
        "max_abs_trial_delta_ln_m": float(
            max((trial["max_abs_trial_delta_ln_m"] for trial in trials), default=0.0)
        ),
        "trials": trials,
    }


def _print_terminal_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "startup",
        "deg[0.9,1.1]",
        "corr/1 norm",
        "gas>prod",
        "corr>prod",
        "raw>prod",
        "clip>prod",
        "best_prod",
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
            f"{row['startup_policy_name']:>18} "
            f"{_fmt(row['mean_fraction_raw_in_0p9_1p1']):>18} "
            f"{_fmt(row['mean_correction_over_ones_norm_ratio']):>18} "
            f"{_fmt(row['fraction_gas_only_beats_prod']):>18} "
            f"{_fmt(row['fraction_correction_only_beats_prod']):>18} "
            f"{_fmt(row['fraction_raw_no_clip_beats_prod']):>18} "
            f"{_fmt(row['fraction_correction_rescaled_beats_prod']):>18} "
            f"{_fmt(row['mean_best_production_residual']):>18}"
        )


def _decision_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gas_only_fraction = max((row["fraction_gas_only_beats_prod"] or 0.0) for row in summary_rows)
    correction_fraction = max((row["fraction_correction_only_beats_prod"] or 0.0) for row in summary_rows)
    raw_fraction = max((row["fraction_raw_no_clip_beats_prod"] or 0.0) for row in summary_rows)
    rescaled_fraction = max((row["fraction_correction_rescaled_beats_prod"] or 0.0) for row in summary_rows)
    mean_dom_ratio = _mean([row["mean_correction_over_ones_norm_ratio"] for row in summary_rows])
    mean_raw_cos = _mean([row["mean_cosine_raw_vs_ones"] for row in summary_rows])

    messages = []
    if gas_only_fraction >= 0.5:
        messages.append("condensate block is harmful; inspect raw update construction")
    if mean_dom_ratio is not None and mean_dom_ratio < 0.5 and mean_raw_cos is not None and mean_raw_cos >= 0.8:
        messages.append("barrier +1 term is too dominant")
    if raw_fraction >= 0.4 and (summary_rows[0]["fraction_gas_only_beats_prod"] or 0.0) < 0.5:
        messages.append("clipping is still the main issue even after startup improvement")
    if not messages:
        messages.append("raw RGIE condensate update looks reasonable; bottleneck is elsewhere")
    next_move = (
        "modify raw RGIE condensate update construction"
        if any(
            msg in messages
            for msg in (
                "condensate block is harmful; inspect raw update construction",
                "barrier +1 term is too dominant",
            )
        )
        else "keep the raw formula and inspect another bottleneck"
    )
    return {
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
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
            support_indices = jnp.asarray(
                [cond_setup.species.index(name) for name in layer_meta["support_names"]],
                dtype=jnp.int32,
            )
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
                    raw_update = diagnose_rgie_raw_condensate_update_block(
                        ln_mk=ln_mk,
                        epsilon=epsilon,
                        formula_matrix_cond=formula_matrix_cond,
                        pi_vector=step_metrics["pi_vector"],
                        hvector_cond=hvector_cond,
                    )
                    raw_metrics = _raw_update_metrics(raw_update)
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

                    variant_results = {}
                    for variant_name in VARIANTS:
                        variant = build_rgie_condensate_direction_variant(raw_update, variant_name)
                        variant_results[variant_name] = _evaluate_variant_lambda_grid(
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
                            step_metrics["delta_ln_ntot"],
                            variant,
                            lambda_trials,
                            current_residual,
                        )

                    production_best = variant_results["production_clipped_current"]["best_fresh_residual"]
                    gas_only_best = variant_results["gas_only"]["best_fresh_residual"]
                    correction_best = variant_results["correction_only_no_clip"]["best_fresh_residual"]
                    raw_best = variant_results["raw_current_no_clip"]["best_fresh_residual"]
                    correction_rescaled_best = variant_results["correction_only_scalar_rescale_0p1"]["best_fresh_residual"]

                    case_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "temperature_K": layer_meta["temperature_K"],
                            "pressure_bar": layer_meta["pressure_bar"],
                            "support_names": list(layer_meta["support_names"]),
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "gas_init_n_iter": gas_init["gas_init_n_iter"],
                            "gas_init_final_residual": gas_init["gas_init_final_residual"],
                            "current_residual": current_residual,
                            "raw_update_metrics": raw_metrics,
                            "variant_results": {
                                **variant_results,
                                "gas_only_beats_production_clipped_current": (
                                    gas_only_best is not None
                                    and production_best is not None
                                    and gas_only_best < production_best
                                ),
                                "correction_only_beats_production_clipped_current": (
                                    correction_best is not None
                                    and production_best is not None
                                    and correction_best < production_best
                                ),
                                "raw_current_no_clip_beats_production_clipped_current": (
                                    raw_best is not None
                                    and production_best is not None
                                    and raw_best < production_best
                                ),
                                "correction_only_scalar_rescale_0p1_beats_production_clipped_current": (
                                    correction_rescaled_best is not None
                                    and production_best is not None
                                    and correction_rescaled_best < production_best
                                ),
                            },
                        }
                    )
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "raw_update": {
                                key: to_python(jax.device_get(value))
                                for key, value in raw_update.items()
                            },
                            "variant_results": variant_results,
                        }
                    )

        summary_rows = []
        for startup in STARTUP_POLICIES:
            matching = [record for record in case_records if record["startup_policy_name"] == startup["name"]]
            summary_rows.append(
                {
                    "startup_policy_name": startup["name"],
                    "mean_fraction_raw_in_0p9_1p1": _mean(
                        [record["raw_update_metrics"]["fraction_raw_in_0p9_1p1"] for record in matching]
                    ),
                    "mean_cosine_raw_vs_ones": _mean(
                        [record["raw_update_metrics"]["cosine_raw_vs_ones"] for record in matching]
                    ),
                    "mean_correction_over_ones_norm_ratio": _mean(
                        [record["raw_update_metrics"]["correction_over_ones_norm_ratio"] for record in matching]
                    ),
                    "fraction_gas_only_beats_prod": _fraction(
                        [record["variant_results"]["gas_only_beats_production_clipped_current"] for record in matching]
                    ),
                    "fraction_correction_only_beats_prod": _fraction(
                        [record["variant_results"]["correction_only_beats_production_clipped_current"] for record in matching]
                    ),
                    "fraction_raw_no_clip_beats_prod": _fraction(
                        [record["variant_results"]["raw_current_no_clip_beats_production_clipped_current"] for record in matching]
                    ),
                    "fraction_correction_rescaled_beats_prod": _fraction(
                        [
                            record["variant_results"]["correction_only_scalar_rescale_0p1_beats_production_clipped_current"]
                            for record in matching
                        ]
                    ),
                    "mean_best_production_residual": _mean(
                        [record["variant_results"]["production_clipped_current"]["best_fresh_residual"] for record in matching]
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
            "variant_names": list(VARIANTS),
            "cases": case_records,
            "summary_rows": summary_rows,
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
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
