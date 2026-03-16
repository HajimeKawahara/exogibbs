"""Run the Stage 1/2 single-layer ExoGibbs benchmark.

Execute from the repository root, for example:

    PYTHONPATH=src python -m benchmarks.run_single_layer
    PYTHONPATH=src python -m benchmarks.run_single_layer --output results/benchmarks/single_layer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from benchmarks.cases import FASTCHEM_EXTENDED_SINGLE_ANCHOR
from benchmarks.cases import get_single_layer_case
from benchmarks.common import device_for_platform
from benchmarks.common import extract_diag_value
from benchmarks.common import has_nan_tree
from benchmarks.common import current_timestamp_utc
from benchmarks.common import load_normalized_element_abundances
from benchmarks.common import to_python
from benchmarks.models import BenchmarkResult
from benchmarks.models import ExecutionConfig
from benchmarks.timing import block_tree
from benchmarks.timing import time_first_and_repeated_calls
from exogibbs.api.equilibrium import EquilibriumOptions
from exogibbs.api.equilibrium import equilibrium
from exogibbs.presets.fastchem import chemsetup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        default=FASTCHEM_EXTENDED_SINGLE_ANCHOR.case_id,
        help="Single-layer benchmark case identifier.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=("cpu", "gpu"),
        help="Optional JAX platform selection.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup calls after the first measured call.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of measured repeated warm calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/single_layer.json"),
        help="Path for the JSON result file.",
    )
    return parser


def _build_result(
    case_id: str,
    platform: Optional[str],
    warmup: int,
    repeat: int,
    output_path: Path,
) -> BenchmarkResult:
    case = get_single_layer_case(case_id)
    backend, device = device_for_platform(platform)

    opts = EquilibriumOptions(
        epsilon_crit=float(case.solver_options["epsilon_crit"]),
        max_iter=int(case.solver_options["max_iter"]),
    )

    with jax.default_device(device):
        setup = chemsetup(**case.setup_kwargs)
        b = load_normalized_element_abundances(
            case.setup_metadata["element_file"],
            tuple(setup.elements),
        ).astype(setup.formula_matrix.dtype)
        T = jnp.asarray(case.axes["temperature_K"])
        P = jnp.asarray(case.axes["pressure_bar"])

        solve_fn = jax.jit(
            lambda T_in, P_in, b_in: equilibrium(
                setup,
                T_in,
                P_in,
                b_in,
                Pref=case.pref_bar,
                options=opts,
                return_diagnostics=False,
            )
        )

        timing = time_first_and_repeated_calls(
            solve_fn,
            T,
            P,
            b,
            warmup_count=warmup,
            repeat_count=repeat,
        )

        diagnostics = None
        diagnostic_result = None
        try:
            diag_fn = jax.jit(
                lambda T_in, P_in, b_in: equilibrium(
                    setup,
                    T_in,
                    P_in,
                    b_in,
                    Pref=case.pref_bar,
                    options=opts,
                    return_diagnostics=True,
                )
            )
            diagnostic_result, diagnostics = diag_fn(T, P, b)
            block_tree((diagnostic_result, diagnostics))
        except Exception:
            diagnostics = None
            diagnostic_result = None

        result_to_check = diagnostic_result if diagnostic_result is not None else timing.last_result
        has_nan = has_nan_tree(result_to_check) or has_nan_tree(diagnostics)

    diag_converged = extract_diag_value(diagnostics, "converged")
    diag_final_residual = extract_diag_value(diagnostics, "final_residual")
    diag_iteration_count = extract_diag_value(diagnostics, "n_iter")
    diag_hit_max_iter = extract_diag_value(diagnostics, "hit_max_iter")
    diag_max_iter = extract_diag_value(diagnostics, "max_iter")
    diag_epsilon_crit = extract_diag_value(diagnostics, "epsilon_crit")

    if has_nan:
        status = "fail_runtime"
    elif diag_converged is False or diag_hit_max_iter is True:
        status = "fail_runtime"
    elif diag_converged is True:
        status = "pass"
    else:
        status = "error"

    execution_config = ExecutionConfig(
        backend=backend,
        platform=device.platform,
        dtype=str(b.dtype),
        warmup_count=warmup,
        repeat_count=repeat,
    )
    metrics = {
        "converged": diag_converged,
        "final_residual": diag_final_residual,
        "iteration_count": diag_iteration_count,
        "hit_max_iter": diag_hit_max_iter,
        "max_iter": diag_max_iter,
        "epsilon_crit": diag_epsilon_crit,
        "has_nan": has_nan,
        "first_call_wall_s": timing.first_call_wall_s,
        "first_call_s": timing.first_call_wall_s,
        "warm_call_wall_s": timing.warm_call_wall_s,
        "warm_call_times_s": timing.warm_call_wall_s,
        "warm_call_mean_s": timing.warm_call_mean_s,
        "warm_call_median_s": timing.warm_call_median_s,
        "warm_call_p95_s": timing.warm_call_p95_s,
        "warm_call_min_s": timing.warm_call_min_s,
        "result_ntot": to_python(jax.device_get(timing.last_result.ntot)),
        "output_path": str(output_path),
    }

    setup_metadata = dict(case.setup_metadata)
    setup_metadata["n_elements"] = int(setup.formula_matrix.shape[0])
    setup_metadata["n_species"] = int(setup.formula_matrix.shape[1])
    setup_metadata["diagnostics_collection"] = "separate_jitted_call"

    return BenchmarkResult(
        benchmark_version="0.1",
        case_id=case.case_id,
        category=case.category,
        timestamp_utc=current_timestamp_utc(),
        setup_metadata=setup_metadata,
        axes=dict(case.axes),
        solver_options=dict(case.solver_options),
        execution_config=execution_config,
        metrics=metrics,
        status=status,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    result = _build_result(
        case_id=args.case_id,
        platform=args.platform,
        warmup=args.warmup,
        repeat=args.repeat,
        output_path=args.output,
    )

    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(payload), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
