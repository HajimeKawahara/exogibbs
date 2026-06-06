"""Audit patched condensate PIPM convergence for reduced RGIE vs full GIE.

This diagnostic runs the current patched RGIE and patched full GIE condensate
solvers on exactly the same initial states. It is sequential by design and does
not modify production defaults.

Example:

    PYTHONPATH=src python examples/comparisons/audit_condensate_pipm_line_search_convergence.py --platform cpu
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
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.pipm_gie_cond import trace_minimize_gibbs_cond_iterations as trace_gie_iterations
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_iterations as trace_rgie_iterations
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "condensate_pipm_line_search_convergence_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "condensate_pipm_line_search_convergence_traces.json"
DEFAULT_INITIAL_AMOUNT = 1.0e-30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        default=None,
        choices=("cpu", "gpu"),
        help="Optional JAX platform selection. CPU is recommended for this sequential audit.",
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
        default=200,
        help="Maximum condensate iterations per run.",
    )
    parser.add_argument(
        "--gas-max-iter",
        type=int,
        default=1000,
        help="Maximum gas-equilibrium iterations used to construct shared initial states.",
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


def _build_profile_states(element_vector: jax.Array) -> list[dict[str, Any]]:
    states = []
    for layer_index in DEFAULT_LAYER_INDICES:
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


def _extract_last_iteration(trace: dict[str, Any]) -> dict[str, Any]:
    if not trace["history"]:
        return {
            "lam_heuristic": None,
            "lam_selected": None,
            "n_backtracks": None,
            "line_search_accept_kind": None,
            "residual_before": None,
            "residual_after": None,
        }
    last = trace["history"][-1]
    return {
        "lam_heuristic": last["lam_heuristic"],
        "lam_selected": last["lam_selected"],
        "n_backtracks": last["n_backtracks"],
        "line_search_accept_kind": last["line_search_accept_kind"],
        "residual_before": last["residual_before"],
        "residual_after": last["residual_after"],
    }


def _build_run_record(
    *,
    solver_name: str,
    layer_meta: dict[str, Any],
    epsilon: float,
    init_state: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    last_iteration = _extract_last_iteration(trace)
    invalid_numbers_detected = bool(
        jnp.any(~jnp.isfinite(trace["ln_nk"]))
        | jnp.any(~jnp.isfinite(trace["ln_mk"]))
        | ~jnp.isfinite(jnp.asarray(trace["ln_ntot"]))
    )
    return {
        "solver": solver_name,
        "layer_index": layer_meta["layer_index"],
        "temperature_K": layer_meta["temperature_K"],
        "pressure_bar": layer_meta["pressure_bar"],
        "epsilon": float(epsilon),
        "residual_crit": float(math.exp(float(epsilon))),
        "converged": bool(trace["converged"]),
        "n_iter": int(trace["n_iter"]),
        "final_residual": float(trace["final_residual"]),
        "hit_max_iter": bool(trace["hit_max_iter"]),
        "invalid_numbers_detected": invalid_numbers_detected,
        "initial_state": {
            "gas_init_n_iter": init_state["gas_init_n_iter"],
            "gas_init_final_residual": init_state["gas_init_final_residual"],
            "ln_nk": to_python(jax.device_get(init_state["ln_nk"])),
            "ln_mk": to_python(jax.device_get(init_state["ln_mk"])),
            "ln_ntot": to_python(jax.device_get(init_state["ln_ntot"])),
        },
        "support_names": list(init_state["support_names"]),
        **last_iteration,
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


def _summarize_solver(records: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    n_iters = [record["n_iter"] for record in records]
    final_residuals = [record["final_residual"] for record in records]
    all_history = [step for trace in traces for step in trace["history"]]
    lam_reduced = [step["lam_selected"] < step["lam_heuristic"] - 1.0e-15 for step in all_history]
    accept_monotone = [step["line_search_accept_kind"] == "monotone" for step in all_history]
    accept_best = [step["line_search_accept_kind"] == "best_finite_fallback" for step in all_history]
    accept_zero = [step["line_search_accept_kind"] == "zero_step" for step in all_history]
    monotone_residual = [
        math.isfinite(step["residual_before"]) and step["residual_after"] <= step["residual_before"] + 1.0e-15
        for step in all_history
    ]
    clipping_active = [
        step.get("max_abs_raw_delta_ln_mk", 0.0) > step.get("max_abs_clipped_delta_ln_mk", 0.0) + 1.0e-12
        for step in all_history
    ]
    return {
        "run_count": len(records),
        "iteration_count": len(all_history),
        "convergence_rate": _fraction([record["converged"] for record in records]),
        "max_iter_hit_rate": _fraction([record["hit_max_iter"] for record in records]),
        "mean_n_iter": _mean([float(value) for value in n_iters]),
        "median_n_iter": _median([float(value) for value in n_iters]),
        "mean_final_residual": _mean(final_residuals),
        "median_final_residual": _median(final_residuals),
        "fraction_lam_selected_lt_lam_heuristic": _fraction(lam_reduced),
        "fraction_accept_monotone": _fraction(accept_monotone),
        "fraction_accept_best_finite_fallback": _fraction(accept_best),
        "fraction_accept_zero_step": _fraction(accept_zero),
        "fraction_monotone_residual_decrease": _fraction(monotone_residual),
        "fraction_condensate_clipping_active": _fraction(clipping_active),
    }


def _decision_text(summary_by_solver: dict[str, dict[str, Any]]) -> str:
    rgie = summary_by_solver["rgie"]
    gie = summary_by_solver["gie"]
    gie_better_hits = (gie["max_iter_hit_rate"] or 0.0) + 1.0e-12 < (rgie["max_iter_hit_rate"] or 0.0)
    gie_better_residual = (
        gie["mean_final_residual"] is not None
        and rgie["mean_final_residual"] is not None
        and gie["mean_final_residual"] <= 0.8 * rgie["mean_final_residual"]
    )
    if gie_better_hits or gie_better_residual:
        return "full-vs-reduced remains worth exploring"
    fallback_heavy = (
        ((rgie["fraction_accept_best_finite_fallback"] or 0.0) + (rgie["fraction_accept_zero_step"] or 0.0) >= 0.25)
        and ((gie["fraction_accept_best_finite_fallback"] or 0.0) + (gie["fraction_accept_zero_step"] or 0.0) >= 0.25)
    )
    return (
        "line search helped globalization, but the main bottleneck is elsewhere"
        if fallback_heavy
        else "line search helped globalization, but the main bottleneck is elsewhere"
    )


def _bottleneck_text(summary_by_solver: dict[str, dict[str, Any]]) -> str:
    lam_reduction = max(
        summary_by_solver["rgie"]["fraction_lam_selected_lt_lam_heuristic"] or 0.0,
        summary_by_solver["gie"]["fraction_lam_selected_lt_lam_heuristic"] or 0.0,
    )
    fallback = max(
        (summary_by_solver["rgie"]["fraction_accept_best_finite_fallback"] or 0.0)
        + (summary_by_solver["rgie"]["fraction_accept_zero_step"] or 0.0),
        (summary_by_solver["gie"]["fraction_accept_best_finite_fallback"] or 0.0)
        + (summary_by_solver["gie"]["fraction_accept_zero_step"] or 0.0),
    )
    clipping = max(
        summary_by_solver["rgie"]["fraction_condensate_clipping_active"] or 0.0,
        summary_by_solver["gie"]["fraction_condensate_clipping_active"] or 0.0,
    )
    if clipping >= 0.5 and fallback < 0.25:
        return "clipping"
    if lam_reduction >= 0.5 and fallback >= 0.25:
        return "direction quality"
    if lam_reduction >= 0.5:
        return "lambda selection"
    return "direction quality"


def _print_table(records: list[dict[str, Any]]) -> None:
    header = (
        f"{'solver':<6} {'layer':>5} {'eps':>7} {'conv':>5} {'iter':>5} "
        f"{'final_res':>12} {'maxit':>5} {'inv':>4} {'lam_h':>9} "
        f"{'lam_s':>9} {'bt':>3} {'accept':>22}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        lam_h = "-" if record["lam_heuristic"] is None else f"{record['lam_heuristic']:.3e}"
        lam_s = "-" if record["lam_selected"] is None else f"{record['lam_selected']:.3e}"
        bt = "-" if record["n_backtracks"] is None else str(record["n_backtracks"])
        accept = "-" if record["line_search_accept_kind"] is None else record["line_search_accept_kind"]
        print(
            f"{record['solver']:<6} {record['layer_index']:>5d} {record['epsilon']:>7.1f} "
            f"{str(record['converged']):>5} {record['n_iter']:>5d} {record['final_residual']:>12.3e} "
            f"{str(record['hit_max_iter']):>5} {str(record['invalid_numbers_detected']):>4} "
            f"{lam_h:>9} {lam_s:>9} {bt:>3} {accept:>22}"
        )


def main() -> None:
    args = build_parser().parse_args()
    backend, device = device_for_platform(args.platform)
    with jax.default_device(device):
        gas_setup = gas_chemsetup(silent=True)
        cond_setup = cond_chemsetup(silent=True)
        element_vector = jnp.asarray(gas_setup.element_vector_reference, dtype=jnp.float64)
        layer_states = _build_profile_states(element_vector)

        records: list[dict[str, Any]] = []
        traces_payload: dict[str, Any] = {
            "timestamp_utc": current_timestamp_utc(),
            "backend": backend,
            "platform": device.platform,
            "layers": [],
        }
        solver_to_traces: dict[str, list[dict[str, Any]]] = {"rgie": [], "gie": []}

        for layer_meta in layer_states:
            state = layer_meta["state"]
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            shared_init = _build_shared_condensate_init(
                state,
                gas_setup,
                cond_setup,
                layer_meta["support_names"],
                gas_hvector,
                cond_hvector_full,
                gas_max_iter=args.gas_max_iter,
            )
            layer_trace_entry = {
                "layer_index": layer_meta["layer_index"],
                "temperature_K": layer_meta["temperature_K"],
                "pressure_bar": layer_meta["pressure_bar"],
                "support_size": layer_meta["support_size"],
                "support_names": list(layer_meta["support_names"]),
                "support_source_json": layer_meta["support_source_json"],
                "initial_state": {
                    "gas_init_n_iter": shared_init["gas_init_n_iter"],
                    "gas_init_final_residual": shared_init["gas_init_final_residual"],
                    "ln_nk": to_python(jax.device_get(shared_init["ln_nk"])),
                    "ln_mk": to_python(jax.device_get(shared_init["ln_mk"])),
                    "ln_ntot": to_python(jax.device_get(shared_init["ln_ntot"])),
                },
                "runs": [],
            }
            for epsilon in DEFAULT_EPSILONS:
                residual_crit = float(math.exp(float(epsilon)))
                rgie_trace = trace_rgie_iterations(
                    state,
                    ln_nk_init=jnp.array(shared_init["ln_nk"], dtype=jnp.float64),
                    ln_mk_init=jnp.array(shared_init["ln_mk"], dtype=jnp.float64),
                    ln_ntot_init=jnp.asarray(shared_init["ln_ntot"], dtype=jnp.float64),
                    formula_matrix=gas_setup.formula_matrix,
                    formula_matrix_cond=shared_init["formula_matrix_cond"],
                    hvector_func=shared_init["gas_hvector_func"],
                    hvector_cond_func=shared_init["hvector_cond_func"],
                    epsilon=float(epsilon),
                    residual_crit=residual_crit,
                    max_iter=args.max_iter,
                )
                gie_trace = trace_gie_iterations(
                    state,
                    ln_nk_init=jnp.array(shared_init["ln_nk"], dtype=jnp.float64),
                    ln_mk_init=jnp.array(shared_init["ln_mk"], dtype=jnp.float64),
                    ln_ntot_init=jnp.asarray(shared_init["ln_ntot"], dtype=jnp.float64),
                    formula_matrix=gas_setup.formula_matrix,
                    formula_matrix_cond=shared_init["formula_matrix_cond"],
                    hvector_func=shared_init["gas_hvector_func"],
                    hvector_cond_func=shared_init["hvector_cond_func"],
                    epsilon=float(epsilon),
                    residual_crit=residual_crit,
                    max_iter=args.max_iter,
                )

                rgie_record = _build_run_record(
                    solver_name="rgie",
                    layer_meta=layer_meta,
                    epsilon=epsilon,
                    init_state=shared_init,
                    trace=rgie_trace,
                )
                gie_record = _build_run_record(
                    solver_name="gie",
                    layer_meta=layer_meta,
                    epsilon=epsilon,
                    init_state=shared_init,
                    trace=gie_trace,
                )
                records.extend([rgie_record, gie_record])
                solver_to_traces["rgie"].append(rgie_trace)
                solver_to_traces["gie"].append(gie_trace)
                layer_trace_entry["runs"].append(
                    {
                        "epsilon": float(epsilon),
                        "rgie": to_python(rgie_trace),
                        "gie": to_python(gie_trace),
                    }
                )
            traces_payload["layers"].append(layer_trace_entry)

    summary_by_solver = {
        "rgie": _summarize_solver(
            [record for record in records if record["solver"] == "rgie"],
            solver_to_traces["rgie"],
        ),
        "gie": _summarize_solver(
            [record for record in records if record["solver"] == "gie"],
            solver_to_traces["gie"],
        ),
    }
    decision = _decision_text(summary_by_solver)
    monotone_assessment = {
        solver: summary["fraction_monotone_residual_decrease"] for solver, summary in summary_by_solver.items()
    }
    report = {
        "timestamp_utc": current_timestamp_utc(),
        "platform": args.platform,
        "layer_indices": list(DEFAULT_LAYER_INDICES),
        "epsilons": list(DEFAULT_EPSILONS),
        "max_iter": args.max_iter,
        "gas_max_iter": args.gas_max_iter,
        "runs": records,
        "summary_by_solver": summary_by_solver,
        "decision": decision,
        "monotone_residual_decrease_fraction_by_solver": monotone_assessment,
        "likely_bottleneck": _bottleneck_text(summary_by_solver),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(report), indent=2) + "\n", encoding="utf-8")
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2) + "\n", encoding="utf-8")

    _print_table(records)
    print()
    print("Summary:")
    for solver_name, summary in summary_by_solver.items():
        print(
            f"  {solver_name}: conv_rate={summary['convergence_rate']:.3f} "
            f"mean_iter={summary['mean_n_iter']:.1f} median_iter={summary['median_n_iter']:.1f} "
            f"mean_final_res={summary['mean_final_residual']:.3e} median_final_res={summary['median_final_residual']:.3e} "
            f"lam_reduced={summary['fraction_lam_selected_lt_lam_heuristic']:.3f} "
            f"monotone={summary['fraction_accept_monotone']:.3f} "
            f"best_finite={summary['fraction_accept_best_finite_fallback']:.3f} "
            f"zero_step={summary['fraction_accept_zero_step']:.3f}"
        )
    print(f"Decision: {decision}")
    print(f"Likely bottleneck: {report['likely_bottleneck']}")
    print(f"Summary JSON: {args.output}")
    print(f"Trace JSON: {args.traces_output}")


if __name__ == "__main__":
    main()
