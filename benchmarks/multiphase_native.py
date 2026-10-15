"""Measure native phase selection and implicit derivatives via public APIs.

Run with JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu. This is a two-phase numerical
benchmark, not adoption of a planetary M2 material recipe.
"""

import json
import platform
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

import exoeos
import exogibbs
from exoeos import magma
from exogibbs.api import multiphase
from exogibbs.interop.exoeos import make_magma_liquid_phase, make_magma_pure_phase


def main():
    elements = ("Mg", "Si", "O")
    liquid = make_magma_liquid_phase(elements=elements, components=magma.COMPONENTS)
    solid = make_magma_pure_phase(elements=elements, component="Mg2SiO4",
                                 molar_gibbs_j=magma.forsterite_gibbs, name="forsterite")
    problem = multiphase.Problem(elements, (liquid, solid))
    support = multiphase.Support((0, 1, 2))

    solve = jax.jit(lambda t, p, b, n: multiphase.solve_fixed_support(
        problem, temperature_k=t, pressure_pa=p, element_amounts_mol=b,
        init=n, support=support,
    ))
    inputs = (jnp.array(2000.), jnp.array(1e5), jnp.array([1.5, 1., 3.5]),
              jnp.array([.25, .45, .3]))
    started = time.perf_counter()
    result = solve(*inputs)
    jax.block_until_ready(result)
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(20):
        result = solve(*inputs)
        jax.block_until_ready(result)
    runtime_seconds = (time.perf_counter() - started) / 20
    assert result.converged and result.minimum_valid and result.support_valid
    selected_states = []
    for temperature in (2000., 2200.):
        selected = multiphase.solve(problem, temperature_k=temperature, pressure_pa=1e5,
                                     element_amounts_mol=inputs[2], init=inputs[3])
        assert selected.converged and selected.local_kkt_valid and selected.minimum_valid
        selected_states.append({
            "temperature_k": temperature,
            "amounts_mol": np.asarray(selected.amounts_mol).tolist(),
            "support": list(map(int, selected.support.components)),
            "stability_status": selected.stability_status,
            "pure_insertion_lower": float(selected.insertion_lower[1]),
            "liquid_global_bound_available": bool(np.isfinite(selected.insertion_lower[0])),
        })

    def amounts(theta):
        t, p, silica, forsterite = theta
        initial = jnp.array([silica, .6*forsterite, .4*forsterite])
        return solve(t, p, jnp.asarray(problem.atom_matrix) @ initial, initial).amounts_mol

    theta = jnp.array([2000., 1e5, .25, .75])
    jacobian = jax.jit(jax.jacfwd(amounts))(theta)
    finite_errors = []
    for factor in (1., .5):
        finite = []
        for i, step in enumerate(factor*np.array([.1, 1000., .001, .001])):
            delta = jnp.zeros(4).at[i].set(step)
            finite.append((amounts(theta+delta)-amounts(theta-delta))/(2*step))
        finite_errors.append(float(np.max(np.abs(np.asarray(finite).T-jacobian))))
    direction, cotangent = jnp.array([2., 200., .02, -.01]), jnp.array([.1, -.2, .3])
    forward = jax.jvp(amounts, (theta,), (direction,))[1]
    reverse = jax.vjp(amounts, theta)[1](cotangent)[0]
    expected_budget_jacobian = np.column_stack((np.zeros((3, 2)), problem.atom_matrix[:, :2]))
    print(json.dumps({
        "model_id": "exoeos-native-binary-liquid-forsterite",
        "scope": "R3/R4 small native reference; no M2 material adoption or planetary derivative",
        "python": platform.python_version(), "jax": jax.__version__,
        "jax_x64": jax.config.jax_enable_x64,
        "devices": [str(device) for device in jax.devices()],
        "exogibbs_import": exogibbs.__file__, "exoeos_import": exoeos.__file__,
        "exogibbs_version": exogibbs.__version__, "exoeos_version": exoeos.__version__,
        "elements": elements, "components": [p.components for p in problem.phases],
        "atom_matrix": problem.atom_matrix.tolist(),
        "temperature_k": 2000., "pressure_pa": 1e5,
        "element_amounts_mol": [1.5, 1., 3.5],
        "init_mol": [.25, .45, .3], "amounts_mol": np.asarray(result.amounts_mol).tolist(),
        "gibbs_j": float(result.gibbs_j), "iterations": int(result.iterations),
        "conservation_error": float(result.conservation_error),
        "stationarity_error": float(result.stationarity_error),
        "minimum_curvature": float(result.minimum_curvature),
        "stability_status": result.stability_status,
        "phase_selection": selected_states,
        "implicit_derivative": {
            "valid": bool(result.derivative_valid),
            "input_order": ["T_K", "P_Pa", "total_SiO2_mol", "total_Mg2SiO4_mol"],
            "amount_jacobian": np.asarray(jacobian).tolist(),
            "refined_fd_max_absolute_errors": finite_errors,
            "jvp_vjp_transpose_error": float(abs(forward @ cotangent - direction @ reverse)),
            "conservation_jacobian_error": float(np.max(np.abs(
                problem.atom_matrix @ jacobian - expected_budget_jacobian))),
        },
        "first_call_seconds": first_seconds, "warm_call_mean_seconds": runtime_seconds,
        "warm_repetitions": 20,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2))


if __name__ == "__main__":
    main()
