"""Measure the small native R2 primal using installed public provider APIs.

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
    print(json.dumps({
        "model_id": "exoeos-native-binary-liquid-forsterite",
        "scope": "R2 fixed support; no phase selection or M2 material adoption",
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
        "first_call_seconds": first_seconds, "warm_call_mean_seconds": runtime_seconds,
        "warm_repetitions": 20,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2))


if __name__ == "__main__":
    main()
