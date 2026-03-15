"""Benchmark one representative ExoGibbs equilibrium solve.

This script is intentionally small and solver-agnostic. It benchmarks the
high-level single-state API and separates:

* JIT compilation time
* First execution time
* Steady-state execution time

It follows JAX timing practice:

* wrap the solve in ``jax.jit``
* explicitly compile once
* use ``block_until_ready`` on outputs
* time repeated executions after warmup

Example
-------
CPU:
    python examples/benchmark/single_solve_benchmark.py --platform cpu

GPU:
    python examples/benchmark/single_solve_benchmark.py --platform gpu
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp

from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.presets.fastchem import chemsetup
from jax import config

config.update("jax_enable_x64", True)


def _block_tree(tree: Any) -> Any:
    """Synchronize all array leaves before reading wall-clock time."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _get_device(platform: str) -> jax.Device:
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX device found for platform={platform!r}.")
    return devices[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default=None,
        help="Execution platform. Defaults to JAX default backend.",
    )
    parser.add_argument("--temperature", type=float, default=3000.0, help="Temperature in K.")
    parser.add_argument("--pressure", type=float, default=1.0e-8, help="Pressure in bar.")
    parser.add_argument(
        "--pref",
        type=float,
        default=1.0,
        help="Reference pressure in bar.",
    )
    parser.add_argument(
        "--epsilon-crit",
        type=float,
        default=1.0e-10,
        help="Solver convergence tolerance.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Solver iteration cap.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed post-first-run warmup executions before steady-state timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timed steady-state executions.",
    )
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help="Also run the diagnostic path once outside the timed section.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.platform is None:
        backend = jax.default_backend()
        device = jax.devices()[0]
    else:
        device = _get_device(args.platform)
        backend = device.platform

    with jax.default_device(device):
        #setup = chemsetup()
        setup = chemsetup(path="fastchem/logK/logK_extended.dat", species_defalt_elements=False, element_file="fastchem/element_abundances/asplund_2020_extended.dat")
        from exojax.utils.zsol import nsol
        import jax.numpy as jnp

        solar_abundance = nsol()
        na_value = 1.e-14 # abundance for elements solar abundance is unavailable
        nsol_vector = []
        for el in setup.elements[:-1]:
            try:
                nsol_vector.append(solar_abundance[el])
            except:
                nsol_vector.append(na_value)
                print("no info on " ,el, "solar abundance. set",na_value)
        nsol_vector = jnp.array([nsol_vector])  # no solar abundance for e-
        element_vector = jnp.append(nsol_vector, 0.0)

        options = EquilibriumOptions(
            epsilon_crit=args.epsilon_crit,
            max_iter=args.max_iter,
        )
        T = jnp.asarray(args.temperature)
        P = jnp.asarray(args.pressure)
        b = jnp.asarray(element_vector)

    def solve_once(T_in: jax.Array, P_in: jax.Array, b_in: jax.Array):
        return equilibrium(
            setup,
            T_in,
            P_in,
            b_in,
            Pref=args.pref,
            options=options,
            return_diagnostics=False,
        )

    jitted_solve = jax.jit(solve_once)

    compile_t0 = time.perf_counter()
    compiled_solve = jitted_solve.lower(T, P, b).compile()
    compile_s = time.perf_counter() - compile_t0

    first_t0 = time.perf_counter()
    first_result = compiled_solve(T, P, b)
    _block_tree(first_result)
    first_exec_s = time.perf_counter() - first_t0

    for _ in range(args.warmup):
        _block_tree(compiled_solve(T, P, b))

    steady_times_s = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        result = compiled_solve(T, P, b)
        _block_tree(result)
        steady_times_s.append(time.perf_counter() - t0)

    last_result = result

    print("ExoGibbs single-solve benchmark")
    print(f"backend: {backend}")
    print(f"device: {device}")
    print(f"temperature_K: {args.temperature}")
    print(f"pressure_bar: {args.pressure}")
    print(f"n_elements: {int(setup.formula_matrix.shape[0])}")
    print(f"n_species: {int(setup.formula_matrix.shape[1])}")
    print(f"epsilon_crit: {options.epsilon_crit}")
    print(f"max_iter: {options.max_iter}")
    print(f"warmup_runs: {args.warmup}")
    print(f"timed_repeats: {args.repeat}")
    print(f"compile_time_s: {compile_s:.6f}")
    print(f"first_execution_time_s: {first_exec_s:.6f}")
    print(f"steady_mean_ms: {1.0e3 * statistics.mean(steady_times_s):.6f}")
    print(f"steady_median_ms: {1.0e3 * statistics.median(steady_times_s):.6f}")
    print(f"steady_min_ms: {1.0e3 * min(steady_times_s):.6f}")
    print(f"steady_max_ms: {1.0e3 * max(steady_times_s):.6f}")
    if len(steady_times_s) > 1:
        print(f"steady_std_ms: {1.0e3 * statistics.stdev(steady_times_s):.6f}")
    print(f"steady_solves_per_s: {1.0 / statistics.mean(steady_times_s):.3f}")
    print(f"result_ntot: {float(jax.device_get(last_result.ntot)):.16e}")

    if args.print_diagnostics:
        diag_fn = jax.jit(
            lambda T_in, P_in, b_in: equilibrium(
                setup,
                T_in,
                P_in,
                b_in,
                Pref=args.pref,
                options=options,
                return_diagnostics=True,
            )
        )
        diag_result, diagnostics = diag_fn(T, P, b)
        _block_tree((diag_result, diagnostics))
        print(f"diagnostic_n_iter: {int(jax.device_get(diagnostics['n_iter']))}")
        print(
            "diagnostic_final_residual: "
            f"{float(jax.device_get(diagnostics['final_residual'])):.16e}"
        )
        print(f"diagnostic_converged: {bool(jax.device_get(diagnostics['converged']))}")
        print(
            f"diagnostic_hit_max_iter: {bool(jax.device_get(diagnostics['hit_max_iter']))}"
        )


if __name__ == "__main__":
    main()
