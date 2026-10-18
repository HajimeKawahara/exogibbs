"""Weak ionization must satisfy relative charge closure at default tolerance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.gas import solve
from exogibbs.thermo.models import ChemicalSetup


@pytest.mark.parametrize("kp", [1e-16, 1e-30, 1e-40])
def test_trace_ionization_matches_analytic_solution(kp):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        # H -> H+ + e-, P = 1 bar, total number of nuclei = 1.
        matrix = jnp.array([[1., 1., 0.], [0., -1., 1.]])
        setup = ChemicalSetup(matrix, lambda t: jnp.array([0., -np.log(kp), 0.]),
                              metadata={"relative_charge_convergence": True})
        result, diagnostics = solve(setup, T=2000., P=1., b=jnp.array([1., 0.]),
                                    return_diagnostics=True)
        expected = np.sqrt(kp / (1 + kp))
        assert bool(diagnostics["converged"])
        np.testing.assert_allclose(result.n[1:], expected, rtol=1e-9)
        relative_charge = abs(float(result.n[1] - result.n[2])) / float(result.n[1] + result.n[2])
        assert relative_charge <= 1e-10
        # The optional numerical mode must retain the implicit JVP/VJP.
        def log_electron(pressure):
            return solve(setup, T=2000., P=pressure, b=jnp.array([1., 0.])).ln_n[2]

        expected_slope = -.5 / (1. + kp)
        assert float(jax.jit(jax.grad(log_electron))(1.)) == pytest.approx(expected_slope, rel=1e-9)
        _, tangent = jax.jvp(log_electron, (1.,), (1.,))
        assert float(tangent) == pytest.approx(expected_slope, rel=1e-9)
    finally:
        jax.config.update("jax_enable_x64", previous)
