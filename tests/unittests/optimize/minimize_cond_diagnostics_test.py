import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics


def test_minimize_gibbs_cond_with_diagnostics_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    ln_nk, ln_mk, ln_ntot, diagnostics = minimize_gibbs_cond_with_diagnostics(
        state,
        ln_nk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_mk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot_init=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
    )

    assert ln_nk.shape == (1,)
    assert ln_mk.shape == (1,)
    assert ln_ntot.shape == ()

    expected_fields = {
        "n_iter",
        "converged",
        "hit_max_iter",
        "final_residual",
        "residual_crit",
        "max_iter",
        "epsilon",
        "final_step_size",
        "invalid_numbers_detected",
        "debug_nan",
    }
    assert expected_fields.issubset(diagnostics.keys())

    assert diagnostics["n_iter"].shape == ()
    assert diagnostics["final_residual"].shape == ()
    assert diagnostics["residual_crit"].shape == ()
    assert diagnostics["epsilon"].shape == ()
    assert diagnostics["final_step_size"].shape == ()

    assert int(diagnostics["n_iter"]) == 0
    assert int(diagnostics["max_iter"]) == 0
    assert not bool(diagnostics["converged"])
    assert bool(diagnostics["hit_max_iter"])
    assert not bool(diagnostics["invalid_numbers_detected"])
    assert not bool(diagnostics["debug_nan"])
    assert not (bool(diagnostics["converged"]) and bool(diagnostics["hit_max_iter"]))
