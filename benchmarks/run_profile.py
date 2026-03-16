"""Run the minimal atmospheric-profile ExoGibbs benchmark.

Execute from the repository root, for example:

    PYTHONPATH=src python -m benchmarks.run_profile --method vmap_cold
    PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_top
    PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_bottom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Optional

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from benchmarks.cases import FASTCHEM_EXTENDED_PROFILE_ANCHOR
from benchmarks.cases import get_profile_case
from benchmarks.common import device_for_platform
from benchmarks.common import extract_diag_value
from benchmarks.common import has_nan_tree
from benchmarks.common import load_normalized_element_abundances
from benchmarks.common import to_python
from benchmarks.models import BenchmarkResult
from benchmarks.models import ExecutionConfig
from benchmarks.timing import block_tree
from benchmarks.timing import time_first_and_repeated_calls
from exogibbs.api.equilibrium import EquilibriumOptions
from exogibbs.api.equilibrium import equilibrium_profile
from exogibbs.presets.fastchem import chemsetup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        default=FASTCHEM_EXTENDED_PROFILE_ANCHOR.case_id,
        help="Profile benchmark case identifier.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom"),
        help="Profile execution method to benchmark.",
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
        default=Path("results/benchmarks/profile.json"),
        help="Path for the JSON result file.",
    )
    return parser


def _build_profile_inputs(case_axes: dict[str, Any], dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    layer_count = int(case_axes["layer_count"])
    log_pressure = jnp.linspace(
        jnp.log10(jnp.asarray(case_axes["pressure_min_bar"], dtype=dtype)),
        jnp.log10(jnp.asarray(case_axes["pressure_max_bar"], dtype=dtype)),
        layer_count,
    )
    pressure = jnp.power(10.0, log_pressure)
    temperature = jnp.linspace(
        jnp.asarray(case_axes["temperature_min_K"], dtype=dtype),
        jnp.asarray(case_axes["temperature_max_K"], dtype=dtype),
        layer_count,
    )
    return temperature, pressure


def _diag_array(diagnostics: Optional[dict[str, Any]], key: str) -> Any:
    value = extract_diag_value(diagnostics, key)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _uniform_scalar_or_value(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    if not value:
        return value
    first = value[0]
    if all(item == first for item in value):
        return first
    return value


def _mean_or_none(values: Any) -> Optional[float]:
    if values is None:
        return None
    return float(sum(values) / len(values))


def _max_or_none(values: Any) -> Optional[float]:
    if values is None:
        return None
    return max(values)


def _sum_true(values: Any) -> Optional[int]:
    if values is None:
        return None
    return int(sum(bool(value) for value in values))


def _converged_fraction(values: Any) -> Optional[float]:
    if values is None:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def _build_result(
    case_id: str,
    method: str,
    platform: Optional[str],
    warmup: int,
    repeat: int,
    output_path: Path,
) -> BenchmarkResult:
    case = get_profile_case(case_id)
    backend, device = device_for_platform(platform)

    with jax.default_device(device):
        setup = chemsetup(**case.setup_kwargs)
        dtype = setup.formula_matrix.dtype
        b = load_normalized_element_abundances(
            case.setup_metadata["element_file"],
            tuple(setup.elements),
        ).astype(dtype)
        temperature, pressure = _build_profile_inputs(case.axes, dtype)

        opts = EquilibriumOptions(
            epsilon_crit=float(case.solver_options["epsilon_crit"]),
            max_iter=int(case.solver_options["max_iter"]),
            method=method,
        )

        solve_fn = jax.jit(
            lambda T_in, P_in, b_in: equilibrium_profile(
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
            temperature,
            pressure,
            b,
            warmup_count=warmup,
            repeat_count=repeat,
        )

        diagnostics = None
        diagnostic_result = None
        try:
            diag_fn = jax.jit(
                lambda T_in, P_in, b_in: equilibrium_profile(
                    setup,
                    T_in,
                    P_in,
                    b_in,
                    Pref=case.pref_bar,
                    options=opts,
                    return_diagnostics=True,
                )
            )
            diagnostic_result, diagnostics = diag_fn(temperature, pressure, b)
            block_tree((diagnostic_result, diagnostics))
        except Exception:
            diagnostics = None
            diagnostic_result = None

        result_to_check = diagnostic_result if diagnostic_result is not None else timing.last_result
        has_nan = has_nan_tree(result_to_check) or has_nan_tree(diagnostics)

    residual_per_layer = _diag_array(diagnostics, "final_residual")
    iteration_count_per_layer = _diag_array(diagnostics, "n_iter")
    hit_max_iter_per_layer = _diag_array(diagnostics, "hit_max_iter")
    converged_per_layer = _diag_array(diagnostics, "converged")
    epsilon_crit = _uniform_scalar_or_value(extract_diag_value(diagnostics, "epsilon_crit"))
    max_iter = _uniform_scalar_or_value(extract_diag_value(diagnostics, "max_iter"))

    overall_converged = None
    if converged_per_layer is not None:
        overall_converged = all(bool(value) for value in converged_per_layer)

    severe_convergence_failure = False
    if hit_max_iter_per_layer is not None and any(bool(value) for value in hit_max_iter_per_layer):
        severe_convergence_failure = True
    if converged_per_layer is not None and not all(bool(value) for value in converged_per_layer):
        severe_convergence_failure = True

    if has_nan:
        status = "fail_runtime"
    elif severe_convergence_failure:
        status = "fail_runtime"
    elif overall_converged is True:
        status = "pass"
    else:
        status = "error"

    execution_config = ExecutionConfig(
        backend=backend,
        platform=device.platform,
        dtype=str(dtype),
        warmup_count=warmup,
        repeat_count=repeat,
    )
    setup_metadata = dict(case.setup_metadata)
    setup_metadata["n_elements"] = int(setup.formula_matrix.shape[0])
    setup_metadata["n_species"] = int(setup.formula_matrix.shape[1])
    setup_metadata["diagnostics_collection"] = "separate_jitted_call"

    axes = dict(case.axes)
    axes["method"] = method

    metrics = {
        "converged": overall_converged,
        "layer_converged_fraction": _converged_fraction(converged_per_layer),
        "layer_max_residual": _max_or_none(residual_per_layer),
        "layer_mean_residual": _mean_or_none(residual_per_layer),
        "layer_max_iteration_count": _max_or_none(iteration_count_per_layer),
        "layer_mean_iteration_count": _mean_or_none(iteration_count_per_layer),
        "layer_max_iter_hit_count": _sum_true(hit_max_iter_per_layer),
        "has_nan": has_nan,
        "epsilon_crit": epsilon_crit,
        "max_iter": max_iter,
        "first_call_s": timing.first_call_wall_s,
        "warm_call_times_s": timing.warm_call_wall_s,
        "warm_call_mean_s": timing.warm_call_mean_s,
        "warm_call_median_s": timing.warm_call_median_s,
        "warm_call_p95_s": timing.warm_call_p95_s,
        "warm_call_min_s": timing.warm_call_min_s,
        "residual_per_layer": residual_per_layer,
        "iteration_count_per_layer": iteration_count_per_layer,
        "hit_max_iter_per_layer": hit_max_iter_per_layer,
        "converged_per_layer": converged_per_layer,
        "result_ntot_per_layer": to_python(jax.device_get(timing.last_result.ntot)),
        "output_path": str(output_path),
    }

    return BenchmarkResult(
        case_id=case.case_id,
        category=case.category,
        setup_metadata=setup_metadata,
        axes=axes,
        solver_options={
            **dict(case.solver_options),
            "method": method,
        },
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
        method=args.method,
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
