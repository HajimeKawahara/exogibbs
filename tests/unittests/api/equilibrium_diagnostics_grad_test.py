"""First-order composition derivatives with diagnostics from the same solve."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.equilibrium import (
    EquilibriumOptions,
    equilibrium,
    equilibrium_profile,
)
from exogibbs.equilibrium.gas.kernel import solver


def _setup():
    return ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
        hvector_func=lambda t: jnp.array([0.0, -6000.0 / t, 0.0]),
        elements=("H", "He"),
        species=("H", "H2", "He"),
        element_vector_reference=jnp.array([1.0, 0.08]),
    )


def test_diagnostics_preserve_jvp_vjp_and_independent_finite_differences():
    setup = _setup()
    opts = EquilibriumOptions(epsilon_crit=1e-12, max_iter=200, method="vmap_cold")

    def evaluate(parameters):
        temperature, log_pressure, log_helium = parameters
        return equilibrium_profile(
            setup,
            jnp.array([temperature, temperature + 300.0]),
            jnp.exp(log_pressure) * jnp.array([0.1, 1.0]),
            jnp.array([1.0, jnp.exp(log_helium)]),
            options=opts,
            return_diagnostics=True,
        )

    run = jax.jit(evaluate)
    point = jnp.array([2000.0, 0.0, jnp.log(0.08)])
    direction = jnp.array([300.0, 0.2, 0.3])
    with jax.checking_leaks():
        result, diagnostics = run(point)
        shifted, shifted_diagnostics = run(point + 0.01 * direction)
    assert np.all(diagnostics["converged"])
    assert np.all(shifted_diagnostics["converged"])
    assert not np.allclose(result.x, shifted.x)

    def objective(p):
        return jnp.sum(run(p)[0].x * jnp.array([0.3, 1.0, 2.0]))

    _, tangent = jax.jvp(objective, (point,), (direction,))
    gradient = jax.jit(jax.grad(objective))(point)
    assert tangent == pytest.approx(jnp.vdot(gradient, direction), rel=1e-10)
    for step in (1e-3, 3e-4):
        plus, plus_diagnostics = run(point + step * direction)
        minus, minus_diagnostics = run(point - step * direction)
        assert np.all(plus_diagnostics["converged"])
        assert np.all(minus_diagnostics["converged"])
        finite_difference = jnp.sum((plus.x - minus.x) * jnp.array([0.3, 1.0, 2.0])) / (
            2 * step
        )
        assert tangent == pytest.approx(finite_difference, rel=1e-6, abs=1e-10)

    diagnostic_derivative = jax.grad(lambda p: jnp.sum(run(p)[1]["final_residual"]))(
        point
    )
    np.testing.assert_array_equal(diagnostic_derivative, 0.0)


def test_diagnostics_report_failure_and_use_one_primal_solve(monkeypatch):
    setup = _setup()
    original = solver.minimize_gibbs_core
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "minimize_gibbs_core", counted)
    options = EquilibriumOptions(max_iter=1)
    result, diagnostics = equilibrium(
        setup,
        2000.0,
        0.1,
        setup.element_vector_reference,
        options=options,
        return_diagnostics=True,
    )
    assert len(calls) == 1
    assert not diagnostics["converged"]
    assert diagnostics["hit_max_iter"]
    assert diagnostics["n_iter"] == 1
    assert np.all(np.isfinite(result.n))

    calls.clear()
    jax.jvp(
        lambda t: equilibrium(
            setup,
            t,
            0.1,
            setup.element_vector_reference,
            return_diagnostics=True,
        )[0].x,
        (2000.0,),
        (1.0,),
    )
    assert len(calls) == 1
