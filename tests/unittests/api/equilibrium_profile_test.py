import importlib
import jax.numpy as jnp

import exogibbs.api.equilibrium as eqmod
from exogibbs.api.equilibrium import EquilibriumOptions


class FakeSetup:
    """Minimal stand-in for ChemicalSetup for vectorized API testing."""

    def __init__(self, A):
        self.formula_matrix = A

    def hvector_func(self, T):
        K = self.formula_matrix.shape[1]
        return jnp.zeros((K,))


def test_equilibrium_profile_shapes_and_values(monkeypatch):
    """Simple happy-path test for equilibrium_profile with a stubbed minimizer.

    Ensures batched shapes are correct and values match the stub behavior
    (ln_n = 0 => n = 1, x uniform, ntot = K) across all layers.
    """
    E, K, N = 2, 3, 4
    A = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.float64)
    setup = FakeSetup(A)

    def stub_minimize_gibbs(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        # Basic sanity checks on inputs
        assert A_in.shape == A.shape
        assert ln_nk0.shape == (K,)
        assert ln_ntot0.shape == ()
        return jnp.zeros((K,), dtype=jnp.result_type(ln_nk0, A_in.dtype))

    monkeypatch.setattr(
        "exogibbs.api.equilibrium.minimize_gibbs", stub_minimize_gibbs, raising=True
    )

    # Profile inputs (N layers)
    T = jnp.linspace(1000.0, 2000.0, N)
    P = jnp.linspace(0.1, 1.0, N)
    b = jnp.array([1.0, 2.0], dtype=jnp.float64)

    out = eqmod.equilibrium_profile(
        setup, T, P, b, options=EquilibriumOptions(epsilon_crit=1e-11, max_iter=50)
    )

    # Batched shapes
    assert out.ln_n.shape == (N, K)
    assert out.n.shape == (N, K)
    assert out.x.shape == (N, K)
    assert out.ntot.shape == (N,)

    # Stub behavior reflected across layers
    assert jnp.allclose(out.ln_n, 0.0)
    assert jnp.allclose(out.n, 1.0)
    assert jnp.allclose(out.x, jnp.ones((N, K)) / K)
    assert jnp.allclose(out.ntot, K)
    assert jnp.allclose(jnp.sum(out.x, axis=1), 1.0)

