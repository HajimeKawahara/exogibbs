"""Profile ExoGibbs Newton iteration cost for one representative solve.

This script measures two things:

1. Total steady-state solve time for the public single-state API.
2. Approximate per-iteration cost breakdown by running the same Newton loop
   under Python control and timing jitted sub-steps with synchronization.

The production solver is unchanged. The iteration breakdown is a profiling
path, so its part-level timings are approximate, but it is usually the most
practical way to understand cost structure inside a fully-jitted `while_loop`.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp

from exogibbs.api.chemistry import ThermoState
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.optimize.minimize import profile_minimize_gibbs_iterations
from exogibbs.presets.fastchem import chemsetup

from jax import config

config.update("jax_enable_x64", True)


def _block_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _default_init(b_vec: jax.Array, n_species: int) -> tuple[jax.Array, jax.Array]:
    ln_nk0 = jnp.zeros((n_species,), dtype=b_vec.dtype)
    ln_ntot0 = jnp.log(jnp.asarray(float(n_species), dtype=ln_nk0.dtype))
    return ln_nk0, ln_ntot0


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
        help="Execution platform. Defaults to the JAX default backend.",
    )
    parser.add_argument("--temperature", type=float, default=3000.0, help="Temperature in K.")
    parser.add_argument("--pressure", type=float, default=1.0e-8, help="Pressure in bar.")
    parser.add_argument("--pref", type=float, default=1.0, help="Reference pressure in bar.")
    parser.add_argument("--epsilon-crit", type=float, default=1.0e-10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--solve-repeat",
        type=int,
        default=10,
        help="Timed steady-state repetitions for total solve time.",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=3,
        help="Number of full profiling runs for the Python-controlled iteration loop.",
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
        ln_nk0, ln_ntot0 = _default_init(b, int(setup.formula_matrix.shape[1]))
        state = ThermoState(
            temperature=T,
            ln_normalized_pressure=jnp.log(P / args.pref),
            element_vector=b,
        )

    solve_fn = jax.jit(
        lambda T_in, P_in, b_in: equilibrium(
            setup,
            T_in,
            P_in,
            b_in,
            Pref=args.pref,
            options=options,
            return_diagnostics=False,
        )
    )
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

    compiled_solve = solve_fn.lower(T, P, b).compile()
    compiled_diag = diag_fn.lower(T, P, b).compile()

    result = compiled_solve(T, P, b)
    _block_tree(result)
    _, diagnostics = compiled_diag(T, P, b)
    _block_tree(diagnostics)

    solve_times_s = []
    for _ in range(args.solve_repeat):
        t0 = time.perf_counter()
        timed_result = compiled_solve(T, P, b)
        _block_tree(timed_result)
        solve_times_s.append(time.perf_counter() - t0)

    profiled_runs = []
    for _ in range(args.profile_repeat):
        profiled = profile_minimize_gibbs_iterations(
            state,
            ln_nk0,
            ln_ntot0,
            setup.formula_matrix,
            setup.hvector_func,
            epsilon_crit=options.epsilon_crit,
            max_iter=options.max_iter,
        )
        profiled_runs.append(profiled)

    n_iter_diag = int(jax.device_get(diagnostics["n_iter"]))
    final_residual = float(jax.device_get(diagnostics["final_residual"]))
    converged = bool(jax.device_get(diagnostics["converged"]))
    hit_max_iter = bool(jax.device_get(diagnostics["hit_max_iter"]))

    profiled_n_iter = [run["n_iter"] for run in profiled_runs]
    avg_profiled_total_s = statistics.mean(run["total_profiled_s"] for run in profiled_runs)
    avg_profiled_iter_s = statistics.mean(run["average_iteration_s"] for run in profiled_runs)
    profiled_ln_n = jax.device_get(profiled_runs[0]["ln_nk"])
    solve_ln_n = jax.device_get(result.ln_n)
    final_ln_n_max_abs_diff = float(jnp.max(jnp.abs(profiled_ln_n - solve_ln_n)))

    part_names = [
        "prepare_system",
        "linear_solve",
        "finish_solve",
        "step_update_damping",
        "residual_evaluation",
        "convergence_check",
    ]
    avg_parts_s = {
        name: statistics.mean(run["average_breakdown_s"][name] for run in profiled_runs)
        for name in part_names
    }
    major_parts_s = {
        "jacobian_construction": avg_parts_s["prepare_system"],
        "linear_solve": avg_parts_s["linear_solve"] + avg_parts_s["finish_solve"],
        "step_update_damping": avg_parts_s["step_update_damping"],
        "residual_evaluation": avg_parts_s["residual_evaluation"],
        "convergence_check": avg_parts_s["convergence_check"],
    }

    total_solve_s = statistics.mean(solve_times_s)
    avg_solve_iter_s = total_solve_s / n_iter_diag if n_iter_diag else 0.0

    print("ExoGibbs Newton iteration profile")
    print(f"backend: {backend}")
    print(f"device: {device}")
    print(f"temperature_K: {args.temperature}")
    print(f"pressure_bar: {args.pressure}")
    print("b_assembly_mode: einsum")
    print("linear_solve_mode: stacked_rhs")
    print("residual_eval_mode: reused_terms")
    print(f"n_elements: {int(setup.formula_matrix.shape[0])}")
    print(f"n_species: {int(setup.formula_matrix.shape[1])}")
    print(f"epsilon_crit: {options.epsilon_crit}")
    print(f"max_iter: {options.max_iter}")
    print(f"solve_repeat: {args.solve_repeat}")
    print(f"profile_repeat: {args.profile_repeat}")
    print(f"solve_total_mean_ms: {1.0e3 * total_solve_s:.6f}")
    print(f"solve_total_median_ms: {1.0e3 * statistics.median(solve_times_s):.6f}")
    print(f"newton_iterations: {n_iter_diag}")
    print(f"solve_avg_time_per_iteration_ms: {1.0e3 * avg_solve_iter_s:.6f}")
    print(f"diagnostic_final_residual: {final_residual:.16e}")
    print(f"diagnostic_converged: {converged}")
    print(f"diagnostic_hit_max_iter: {hit_max_iter}")
    print(f"profiled_loop_total_mean_ms: {1.0e3 * avg_profiled_total_s:.6f}")
    print(f"profiled_avg_iteration_ms: {1.0e3 * avg_profiled_iter_s:.6f}")
    print(
        f"profiled_vs_solve_iteration_ratio: "
        f"{(avg_profiled_iter_s / avg_solve_iter_s) if avg_solve_iter_s else 0.0:.6f}"
    )
    print(
        "profiled_iteration_count_match: "
        f"{all(count == n_iter_diag for count in profiled_n_iter)}"
    )
    print(f"profiled_final_ln_n_max_abs_diff: {final_ln_n_max_abs_diff:.16e}")

    for name, value_s in major_parts_s.items():
        part_ms = 1.0e3 * value_s
        frac = value_s / avg_profiled_iter_s if avg_profiled_iter_s else 0.0
        print(f"{name}_avg_ms: {part_ms:.6f}")
        print(f"{name}_fraction_of_profiled_iteration: {frac:.6f}")


if __name__ == "__main__":
    main()
