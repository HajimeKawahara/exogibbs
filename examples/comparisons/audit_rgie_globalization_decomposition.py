"""Diagnostic-only RGIE globalization decomposition audit."""

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
from exogibbs.optimize.pipm_rgie_cond import _choose_lambda_by_residual_backtracking
from exogibbs.optimize.pipm_rgie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _evaluate_trial_step
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_direction_transform_variant
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import compute_rgie_lambda_cap_policy
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_LAMBDA_TRIALS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_globalization_decomposition_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_globalization_decomposition_traces.json"

STARTUP_POLICIES = (
    {"name": "legacy_absolute_m0_1e-30", "policy": "absolute_uniform_m0", "kwargs": {"m0": 1.0e-30}},
    {"name": "ratio_uniform_r0_3e-3", "policy": "ratio_uniform_r0", "kwargs": {"r0": 3.0e-3}},
    {"name": "ratio_uniform_r0_1e-2", "policy": "ratio_uniform_r0", "kwargs": {"r0": 1.0e-2}},
)

TRANSFORM_POLICIES = (
    "current_component_clip_0p1",
    "component_clip_0p5",
    "scalar_rescale_inf_0p1",
    "scalar_rescale_inf_0p5",
    "raw_no_clip",
)

CAP_POLICIES = (
    "current_full_cap",
    "no_cond_cap",
    "no_sk_cap",
    "gas_only_cap",
    "no_heuristic_cap",
)

FOCUS_COMBOS = (
    ("current_component_clip_0p1", "current_full_cap"),
    ("scalar_rescale_inf_0p1", "current_full_cap"),
    ("scalar_rescale_inf_0p1", "no_cond_cap"),
    ("scalar_rescale_inf_0p1", "no_heuristic_cap"),
    ("raw_no_clip", "current_full_cap"),
    ("raw_no_clip", "no_heuristic_cap"),
)

TRAJECTORY_COMBOS = (
    ("current_component_clip_0p1", "current_full_cap"),
    ("scalar_rescale_inf_0p1", "current_full_cap"),
    ("scalar_rescale_inf_0p1", "no_cond_cap"),
    ("scalar_rescale_inf_0p1", "no_heuristic_cap"),
    ("raw_no_clip", "current_full_cap"),
)


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


def _evaluate_transform_cap_combo(
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
    step_metrics: dict[str, Any],
    transform_variant: dict[str, Any],
    cap_policy: dict[str, Any],
    lambda_trials: list[float],
    production_lam: float,
    current_residual: float,
) -> dict[str, Any]:
    trials = []
    lam_cap = float(cap_policy["lam_cap"])
    for lam in lambda_trials:
        if lam > lam_cap + 1.0e-15:
            continue
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            lam,
            step_metrics["delta_ln_nk"],
            transform_variant["delta_ln_mk"],
            step_metrics["delta_ln_ntot"],
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
        )
        ln_mk_trial = jnp.asarray(trial["ln_mk"], dtype=jnp.float64)
        sk_feasible = bool(jnp.all(LOG_S_MAX + epsilon - 2.0 * ln_mk_trial >= 0.0))
        valid = bool(trial["all_finite"]) and math.isfinite(float(trial["fresh_residual"]))
        trials.append(
            {
                "lambda_trial": float(lam),
                "fresh_residual": float(trial["fresh_residual"]),
                "valid_trial": valid,
                "sk_feasible": sk_feasible,
                "max_abs_trial_delta_ln_m": float(jnp.max(jnp.abs(float(lam) * transform_variant["delta_ln_mk"]))),
            }
        )

    valid_trials = [trial for trial in trials if trial["valid_trial"]]
    best_trial = None if not valid_trials else min(valid_trials, key=lambda trial: trial["fresh_residual"])
    return {
        "transform_policy": transform_variant["variant_name"],
        "cap_policy": cap_policy["policy_name"],
        "lam_cap": lam_cap,
        "valid_trial_count": len(valid_trials),
        "invalid_trial_count": len(trials) - len(valid_trials),
        "sk_infeasible_trial_count": sum(1 for trial in trials if not trial["sk_feasible"]),
        "best_lambda": None if best_trial is None else best_trial["lambda_trial"],
        "best_fresh_residual": None if best_trial is None else best_trial["fresh_residual"],
        "any_trial_improves_on_current_residual": any(
            trial["valid_trial"] and trial["fresh_residual"] <= current_residual + 1.0e-15
            for trial in trials
        ),
        "best_lambda_exceeds_current_production_heuristic_lambda": (
            best_trial is not None and best_trial["lambda_trial"] > production_lam + 1.0e-15
        ),
        "production_active_limiter": cap_policy["production_limiting_name"],
        "lam1_gas": float(cap_policy["lam1_gas"]),
        "lam1_cond": float(cap_policy["lam1_cond"]),
        "lam2_cond": float(cap_policy["lam2_cond"]),
        "max_abs_trial_delta_ln_m": 0.0
        if not trials
        else max(trial["max_abs_trial_delta_ln_m"] for trial in trials),
        "cosine_raw_delta_ln_m_vs_transformed": None
        if jnp.isnan(transform_variant["cosine_raw_vs_variant"])
        else float(transform_variant["cosine_raw_vs_variant"]),
        "saturated_fraction": None
        if transform_variant["saturated_fraction"] is None
        else float(transform_variant["saturated_fraction"]),
        "trials": trials,
    }


def _accept_kind_from_code(code: int) -> str:
    if code == 0:
        return "monotone"
    if code == 1:
        return "best_finite_fallback"
    return "zero_step"


def _run_custom_trajectory(
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
    transform_policy_name: str,
    cap_policy_name: str,
    max_iter: int,
) -> dict[str, Any]:
    ln_nk = jnp.asarray(ln_nk_init)
    ln_mk = jnp.asarray(ln_mk_init)
    ln_ntot = jnp.asarray(ln_ntot_init)
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    An = formula_matrix @ jnp.exp(ln_nk)
    Am = formula_matrix_cond @ jnp.exp(ln_mk)
    residual = _compute_current_residual(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
    )
    history = []
    residual_crit = float(math.exp(float(epsilon)))

    for iter_count in range(max_iter):
        if residual <= residual_crit:
            break
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
        transform_variant = build_rgie_condensate_direction_transform_variant(
            step_metrics["raw_delta_ln_mk"],
            transform_policy_name,
        )
        cap_policy = compute_rgie_lambda_cap_policy(
            cap_policy_name,
            lam1_gas=step_metrics["lam1_gas"],
            lam1_cond=jnp.asarray(
                step_metrics["lam1_cond"]
                if transform_policy_name == "current_component_clip_0p1"
                else __import__("exogibbs.optimize.stepsize", fromlist=["stepsize_cond_heurstic"]).stepsize_cond_heurstic(
                    transform_variant["delta_ln_mk"]
                )
            ),
            lam2_cond=jnp.asarray(
                step_metrics["lam2_cond"]
                if transform_policy_name == "current_component_clip_0p1"
                else __import__("exogibbs.optimize.stepsize", fromlist=["stepsize_sk"]).stepsize_sk(
                    transform_variant["delta_ln_mk"], ln_mk, epsilon
                )
            ),
        )
        line_search = _choose_lambda_by_residual_backtracking(
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            current_gk=gk,
            current_An=An,
            current_Am=Am,
            current_residual=residual,
            lam_init=cap_policy["lam_cap"],
            delta_ln_nk=step_metrics["delta_ln_nk"],
            delta_ln_mk=transform_variant["delta_ln_mk"],
            delta_ln_ntot=step_metrics["delta_ln_ntot"],
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            b=b,
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            hvector=hvector,
            hvector_cond=hvector_cond,
            epsilon=epsilon,
        )
        accept_kind = _accept_kind_from_code(int(line_search["accept_code"]))
        history.append(
            {
                "iter": iter_count,
                "lam_heuristic": float(cap_policy["lam_cap"]),
                "lam_selected": float(line_search["lam"]),
                "line_search_accept_kind": accept_kind,
                "residual_before": float(residual),
                "residual_after": float(line_search["fresh_residual"]),
            }
        )
        ln_nk = line_search["ln_nk"]
        ln_mk = line_search["ln_mk"]
        ln_ntot = line_search["ln_ntot"]
        gk = line_search["gk"]
        An = line_search["An"]
        Am = line_search["Am"]
        residual = float(line_search["fresh_residual"])
        if float(line_search["lam"]) <= 1.0e-14:
            break

    monotone = [
        step["residual_after"] <= step["residual_before"] + 1.0e-15
        for step in history
        if math.isfinite(step["residual_before"]) and math.isfinite(step["residual_after"])
    ]
    return {
        "transform_policy": transform_policy_name,
        "cap_policy": cap_policy_name,
        "converged": residual <= residual_crit,
        "n_iter": len(history),
        "final_residual": residual,
        "hit_max_iter": len(history) >= max_iter and residual > residual_crit,
        "fraction_monotone_residual_decrease": _fraction(monotone),
        "fraction_accept_best_finite_fallback": _fraction(
            [step["line_search_accept_kind"] == "best_finite_fallback" for step in history]
        ),
        "fraction_accept_zero_step": _fraction(
            [step["line_search_accept_kind"] == "zero_step" for step in history]
        ),
        "fraction_lam_selected_lt_lam_heuristic": _fraction(
            [step["lam_selected"] < step["lam_heuristic"] - 1.0e-15 for step in history]
        ),
        "history": history,
    }


def _print_terminal_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "combo",
        "raw<prod",
        "scalar<prod",
        "nocond<base",
        "noheur<base",
        "best_res",
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
            f"{row['label']:>18} "
            f"{_fmt(row['fraction_raw_no_clip_beats_prod']):>18} "
            f"{_fmt(row['fraction_scalar_beats_prod']):>18} "
            f"{_fmt(row['fraction_no_cond_cap_beats_scalar_full']):>18} "
            f"{_fmt(row['fraction_no_heuristic_cap_beats_scalar_full']):>18} "
            f"{_fmt(row['mean_best_production_residual']):>18}"
        )


def _decision_summary(case_records: list[dict[str, Any]], focus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prod = [
        case["pair_results"][("current_component_clip_0p1", "current_full_cap")]
        for case in case_records
    ]
    scalar_full = [
        case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]
        for case in case_records
    ]
    scalar_no_cond = [
        case["pair_results"][("scalar_rescale_inf_0p1", "no_cond_cap")]
        for case in case_records
    ]
    scalar_no_heur = [
        case["pair_results"][("scalar_rescale_inf_0p1", "no_heuristic_cap")]
        for case in case_records
    ]
    raw_full = [
        case["pair_results"][("raw_no_clip", "current_full_cap")]
        for case in case_records
    ]

    scalar_beats_clip = _fraction([
        sf["best_fresh_residual"] is not None
        and pr["best_fresh_residual"] is not None
        and sf["best_fresh_residual"] < pr["best_fresh_residual"]
        for sf, pr in zip(scalar_full, prod)
    ]) or 0.0
    no_cond_beats_scalar = _fraction([
        snc["best_fresh_residual"] is not None
        and sf["best_fresh_residual"] is not None
        and snc["best_fresh_residual"] < sf["best_fresh_residual"]
        for snc, sf in zip(scalar_no_cond, scalar_full)
    ]) or 0.0
    no_heur_beats_scalar = _fraction([
        snh["best_fresh_residual"] is not None
        and sf["best_fresh_residual"] is not None
        and snh["best_fresh_residual"] < sf["best_fresh_residual"]
        for snh, sf in zip(scalar_no_heur, scalar_full)
    ]) or 0.0
    raw_beats_prod = _fraction([
        rf["best_fresh_residual"] is not None
        and pr["best_fresh_residual"] is not None
        and rf["best_fresh_residual"] < pr["best_fresh_residual"]
        for rf, pr in zip(raw_full, prod)
    ]) or 0.0
    no_sk_only_infeasible = _fraction([
        snh["best_fresh_residual"] is not None
        and snh["sk_infeasible_trial_count"] > 0
        for snh in scalar_no_heur
    ]) or 0.0

    messages = []
    if scalar_beats_clip >= 0.6 and no_cond_beats_scalar < scalar_beats_clip - 0.2:
        messages.append("componentwise clipping is the main remaining RGIE loss")
    if no_cond_beats_scalar >= scalar_beats_clip + 0.2:
        messages.append("the condensate heuristic lambda cap is too restrictive")
    if no_sk_only_infeasible >= 0.5:
        messages.append("the sk guard is doing necessary safety work")
    if no_heur_beats_scalar >= 0.5 and scalar_beats_clip >= 0.5:
        messages.append("the main issue is clip + cap interaction, not either one alone")
    if raw_beats_prod < scalar_beats_clip and scalar_beats_clip >= 0.5:
        messages.append("direction-preserving magnitude control is the right next production candidate")
    if not messages:
        messages.append("globalization is no longer the main RGIE bottleneck; inspect another layer")

    next_move = (
        "promote a direction-preserving RGIE globalization policy"
        if any(
            msg in messages
            for msg in (
                "componentwise clipping is the main remaining RGIE loss",
                "the condensate heuristic lambda cap is too restrictive",
                "the main issue is clip + cap interaction, not either one alone",
                "direction-preserving magnitude control is the right next production candidate",
            )
        )
        else "keep current globalization and inspect another bottleneck"
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

                    transform_variants = {}
                    pair_results = {}
                    for transform_name in TRANSFORM_POLICIES:
                        transform_variant = build_rgie_condensate_direction_transform_variant(
                            step_metrics["raw_delta_ln_mk"],
                            transform_name,
                        )
                        if transform_name == "current_component_clip_0p1":
                            lam1_cond = step_metrics["lam1_cond"]
                            lam2_cond = step_metrics["lam2_cond"]
                        else:
                            from exogibbs.optimize.stepsize import stepsize_cond_heurstic, stepsize_sk

                            lam1_cond = stepsize_cond_heurstic(transform_variant["delta_ln_mk"])
                            lam2_cond = stepsize_sk(transform_variant["delta_ln_mk"], ln_mk, epsilon)
                        transform_variants[transform_name] = {
                            "variant": transform_variant,
                            "lam1_cond": lam1_cond,
                            "lam2_cond": lam2_cond,
                        }
                        for cap_name in CAP_POLICIES:
                            cap_policy = compute_rgie_lambda_cap_policy(
                                cap_name,
                                lam1_gas=step_metrics["lam1_gas"],
                                lam1_cond=lam1_cond,
                                lam2_cond=lam2_cond,
                            )
                            pair_results[(transform_name, cap_name)] = _evaluate_transform_cap_combo(
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
                                step_metrics=step_metrics,
                                transform_variant=transform_variant,
                                cap_policy=cap_policy,
                                lambda_trials=lambda_trials,
                                production_lam=float(step_metrics["lam"]),
                                current_residual=current_residual,
                            )

                    case_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "current_residual": current_residual,
                            "pair_results": pair_results,
                        }
                    )
                    trace_records.append(
                        {
                            "layer_index": layer_meta["layer_index"],
                            "epsilon": float(epsilon),
                            "startup_policy_name": startup["name"],
                            "pair_results": pair_results,
                        }
                    )

        focus_rows = []
        for startup in STARTUP_POLICIES:
            matching = [case for case in case_records if case["startup_policy_name"] == startup["name"]]
            focus_rows.append(
                {
                    "label": startup["name"],
                    "fraction_raw_no_clip_beats_prod": _fraction([
                        case["pair_results"][("raw_no_clip", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("current_component_clip_0p1", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("raw_no_clip", "current_full_cap")]["best_fresh_residual"]
                        < case["pair_results"][("current_component_clip_0p1", "current_full_cap")]["best_fresh_residual"]
                        for case in matching
                    ]),
                    "fraction_scalar_beats_prod": _fraction([
                        case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("current_component_clip_0p1", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"]
                        < case["pair_results"][("current_component_clip_0p1", "current_full_cap")]["best_fresh_residual"]
                        for case in matching
                    ]),
                    "fraction_no_cond_cap_beats_scalar_full": _fraction([
                        case["pair_results"][("scalar_rescale_inf_0p1", "no_cond_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("scalar_rescale_inf_0p1", "no_cond_cap")]["best_fresh_residual"]
                        < case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"]
                        for case in matching
                    ]),
                    "fraction_no_heuristic_cap_beats_scalar_full": _fraction([
                        case["pair_results"][("scalar_rescale_inf_0p1", "no_heuristic_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"] is not None
                        and case["pair_results"][("scalar_rescale_inf_0p1", "no_heuristic_cap")]["best_fresh_residual"]
                        < case["pair_results"][("scalar_rescale_inf_0p1", "current_full_cap")]["best_fresh_residual"]
                        for case in matching
                    ]),
                    "mean_best_production_residual": _mean([
                        case["pair_results"][("current_component_clip_0p1", "current_full_cap")]["best_fresh_residual"]
                        for case in matching
                    ]),
                }
            )

        trajectory_runs = []
        trajectory_subset = [case for case in case_records if case["layer_index"] == 45 and abs(case["epsilon"] + 5.0) < 1.0e-12]
        layer_meta = next(item for item in profile_states if item["layer_index"] == 45)
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
                epsilon=-5.0,
                support_indices=support_indices,
                startup_policy=startup["policy"],
                dtype=jnp.float64,
                **startup["kwargs"],
            )
            for transform_name, cap_name in TRAJECTORY_COMBOS:
                trajectory = _run_custom_trajectory(
                    state,
                    ln_nk_init=gas_init["ln_nk"],
                    ln_mk_init=ln_mk,
                    ln_ntot_init=gas_init["ln_ntot"],
                    formula_matrix=formula_matrix,
                    formula_matrix_cond=formula_matrix_cond,
                    b=b,
                    hvector=gas_hvector,
                    hvector_cond=hvector_cond,
                    epsilon=-5.0,
                    transform_policy_name=transform_name,
                    cap_policy_name=cap_name,
                    max_iter=args.trajectory_max_iter,
                )
                trajectory_runs.append(
                    {
                        "startup_policy_name": startup["name"],
                        **trajectory,
                    }
                )

        decision = _decision_summary(case_records, focus_rows)
        audit_payload = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": list(args.layers),
            "epsilons": list(args.epsilons),
            "lambda_trials": lambda_trials,
            "startup_policies": [dict(policy) for policy in STARTUP_POLICIES],
            "transform_policies": list(TRANSFORM_POLICIES),
            "cap_policies": list(CAP_POLICIES),
            "focus_combos": [list(combo) for combo in FOCUS_COMBOS],
            "cases": [
                {
                    **{k: v for k, v in case.items() if k != "pair_results"},
                    "pair_results": {
                        f"{transform}|{cap}": result
                        for (transform, cap), result in case["pair_results"].items()
                    },
                }
                for case in case_records
            ],
            "focus_rows": focus_rows,
            "trajectory_runs": trajectory_runs,
            "decision": decision,
        }
        traces_payload = {
            "timestamp_utc": audit_payload["timestamp_utc"],
            "backend": backend,
            "platform": device.platform,
            "traces": [
                {
                    **{k: v for k, v in trace.items() if k != "pair_results"},
                    "pair_results": {
                        f"{transform}|{cap}": result
                        for (transform, cap), result in trace["pair_results"].items()
                    },
                }
                for trace in trace_records
            ],
            "trajectory_runs": trajectory_runs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(audit_payload), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2))

    _print_terminal_table(focus_rows)
    print()
    for message in decision["messages"]:
        print(f"decision: {message}")
    print(f"next_move: {decision['next_move']}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
