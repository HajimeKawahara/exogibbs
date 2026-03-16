"""Run the Stage 1/2 single-layer ExoGibbs benchmark.

Execute from the repository root, for example:

    PYTHONPATH=src python -m benchmarks.run_single_layer
    PYTHONPATH=src python -m benchmarks.run_single_layer --output results/benchmarks/single_layer.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from benchmarks.cases import FASTCHEM_EXTENDED_SINGLE_ANCHOR
from benchmarks.cases import get_single_layer_case
from benchmarks.models import BenchmarkResult
from benchmarks.models import ExecutionConfig
from benchmarks.timing import block_tree
from benchmarks.timing import time_first_and_repeated_calls
from exogibbs.api.equilibrium import EquilibriumOptions
from exogibbs.api.equilibrium import equilibrium
from exogibbs.io.load_data import get_data_filepath
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


def _device_for_platform(platform: Optional[str]) -> tuple[str, jax.Device]:
    if platform is None:
        return jax.default_backend(), jax.devices()[0]
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX device found for platform={platform!r}.")
    return platform, devices[0]


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return _to_python(converted)
    if isinstance(value, (bool, int, float, str)):
        return value
    return value


def _has_nan_tree(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            if bool(jnp.any(jnp.isnan(leaf)).item()):
                return True
    return False


def _extract_diag_value(diagnostics: Optional[Mapping[str, Any]], key: str) -> Any:
    if diagnostics is None:
        return None
    if key not in diagnostics:
        return None
    return _to_python(jax.device_get(diagnostics[key]))


def _load_normalized_element_abundances(element_file: str, element_order: tuple[str, ...]) -> jax.Array:
    """Convert FastChem-style A(X)=log10(n_X/n_H)+12 abundances to normalized linear abundances."""
    path = get_data_filepath(element_file)
    log_abundances = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            log_abundances[parts[0]] = float(parts[1])

    linear_abundances = []
    for element in element_order:
        if element == "e-":
            linear_abundances.append(0.0)
            continue
        if element not in log_abundances:
            linear_abundances.append(1.0e-14)
            continue
        linear_abundances.append(math.pow(10.0, log_abundances[element] - 12.0))

    abundances = jnp.asarray(linear_abundances)
    total = jnp.sum(abundances)
    return abundances / jnp.where(total > 0.0, total, 1.0)


def _build_result(
    case_id: str,
    platform: Optional[str],
    warmup: int,
    repeat: int,
    output_path: Path,
) -> BenchmarkResult:
    case = get_single_layer_case(case_id)
    backend, device = _device_for_platform(platform)

    opts = EquilibriumOptions(
        epsilon_crit=float(case.solver_options["epsilon_crit"]),
        max_iter=int(case.solver_options["max_iter"]),
    )

    with jax.default_device(device):
        setup = chemsetup(**case.setup_kwargs)
        b = _load_normalized_element_abundances(
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
        has_nan = _has_nan_tree(result_to_check) or _has_nan_tree(diagnostics)

    diag_converged = _extract_diag_value(diagnostics, "converged")
    diag_final_residual = _extract_diag_value(diagnostics, "final_residual")
    diag_iteration_count = _extract_diag_value(diagnostics, "n_iter")
    diag_hit_max_iter = _extract_diag_value(diagnostics, "hit_max_iter")
    diag_max_iter = _extract_diag_value(diagnostics, "max_iter")
    diag_epsilon_crit = _extract_diag_value(diagnostics, "epsilon_crit")

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
        "warm_call_wall_s": timing.warm_call_wall_s,
        "warm_call_mean_s": timing.warm_call_mean_s,
        "warm_call_median_s": timing.warm_call_median_s,
        "warm_call_p95_s": timing.warm_call_p95_s,
        "warm_call_min_s": timing.warm_call_min_s,
        "result_ntot": _to_python(jax.device_get(timing.last_result.ntot)),
        "output_path": str(output_path),
    }

    setup_metadata = dict(case.setup_metadata)
    setup_metadata["n_elements"] = int(setup.formula_matrix.shape[0])
    setup_metadata["n_species"] = int(setup.formula_matrix.shape[1])

    return BenchmarkResult(
        case_id=case.case_id,
        category=case.category,
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
    args.output.write_text(json.dumps(_to_python(payload), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
