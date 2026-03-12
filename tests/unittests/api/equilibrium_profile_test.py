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


def test_equilibrium_profile_return_diagnostics(monkeypatch):
    E, K, N = 2, 3, 4
    A = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.float64)
    setup = FakeSetup(A)

    def stub_minimize_gibbs_with_diagnostics(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        ln_n = jnp.zeros((K,), dtype=jnp.result_type(ln_nk0, A_in.dtype))
        diagnostics = {
            "n_iter": jnp.asarray(3, dtype=jnp.int32),
            "converged": jnp.asarray(True),
            "hit_max_iter": jnp.asarray(False),
            "final_residual": jnp.asarray(1e-12, dtype=jnp.float64),
            "epsilon_crit": jnp.asarray(kwargs["epsilon_crit"], dtype=jnp.float64),
            "max_iter": jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
        }
        return ln_n, diagnostics

    monkeypatch.setattr(
        "exogibbs.api.equilibrium.minimize_gibbs_with_diagnostics",
        stub_minimize_gibbs_with_diagnostics,
        raising=True,
    )

    T = jnp.linspace(1000.0, 2000.0, N)
    P = jnp.linspace(0.1, 1.0, N)
    b = jnp.array([1.0, 2.0], dtype=jnp.float64)
    out, diag = eqmod.equilibrium_profile(
        setup,
        T,
        P,
        b,
        options=EquilibriumOptions(epsilon_crit=1e-11, max_iter=50),
        return_diagnostics=True,
    )

    assert out.ln_n.shape == (N, K)
    assert diag["n_iter"].shape == (N,)
    assert diag["converged"].shape == (N,)
    assert diag["hit_max_iter"].shape == (N,)
    assert diag["final_residual"].shape == (N,)
    assert jnp.all(diag["n_iter"] == 3)
    assert jnp.all(diag["converged"])
    assert not jnp.any(diag["hit_max_iter"])


def test_equilibrium_profile_scan_hot_from_top_uses_previous_layer_init(monkeypatch):
    E, K, N = 2, 2, 4
    A = jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)
    setup = FakeSetup(A)

    def stub_minimize_gibbs_with_diagnostics(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        # Return ln_n shifted from initial guess so carry behavior is observable.
        ln_n = ln_nk0 + 1.0
        diagnostics = {
            "n_iter": jnp.asarray(2, dtype=jnp.int32),
            "converged": jnp.asarray(True),
            "hit_max_iter": jnp.asarray(False),
            "final_residual": jnp.asarray(1e-12, dtype=jnp.float32),
            "epsilon_crit": jnp.asarray(kwargs["epsilon_crit"], dtype=jnp.float32),
            "max_iter": jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
        }
        return ln_n, diagnostics

    monkeypatch.setattr(
        "exogibbs.api.equilibrium.minimize_gibbs_with_diagnostics",
        stub_minimize_gibbs_with_diagnostics,
        raising=True,
    )

    T = jnp.linspace(1000.0, 1300.0, N)
    P = jnp.linspace(0.1, 1.0, N)
    b = jnp.array([1.0, 1.0], dtype=jnp.float32)
    out, diag = eqmod.equilibrium_profile(
        setup,
        T,
        P,
        b,
        options=EquilibriumOptions(method="scan_hot_from_top", epsilon_crit=1e-11, max_iter=50),
        return_diagnostics=True,
    )

    # default init ln_nk = 0; each layer adds +1 from previous layer init
    expected = jnp.arange(1, N + 1, dtype=jnp.float32)[:, None] * jnp.ones((N, K), dtype=jnp.float32)
    assert jnp.allclose(out.ln_n, expected)
    assert jnp.all(diag["converged"])


def test_equilibrium_profile_scan_hot_from_bottom_uses_previous_layer_init(monkeypatch):
    E, K, N = 2, 2, 4
    A = jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)
    setup = FakeSetup(A)

    def stub_minimize_gibbs_with_diagnostics(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        ln_n = ln_nk0 + 1.0
        diagnostics = {
            "n_iter": jnp.asarray(2, dtype=jnp.int32),
            "converged": jnp.asarray(True),
            "hit_max_iter": jnp.asarray(False),
            "final_residual": jnp.asarray(1e-12, dtype=jnp.float32),
            "epsilon_crit": jnp.asarray(kwargs["epsilon_crit"], dtype=jnp.float32),
            "max_iter": jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
        }
        return ln_n, diagnostics

    monkeypatch.setattr(
        "exogibbs.api.equilibrium.minimize_gibbs_with_diagnostics",
        stub_minimize_gibbs_with_diagnostics,
        raising=True,
    )

    T = jnp.linspace(1000.0, 1300.0, N)
    P = jnp.linspace(0.1, 1.0, N)
    b = jnp.array([1.0, 1.0], dtype=jnp.float32)
    out, diag = eqmod.equilibrium_profile(
        setup,
        T,
        P,
        b,
        options=EquilibriumOptions(method="scan_hot_from_bottom", epsilon_crit=1e-11, max_iter=50),
        return_diagnostics=True,
    )

    # In original index order, the deepest layer gets first solve in bottom-up scan.
    expected = jnp.array([4, 3, 2, 1], dtype=jnp.float32)[:, None] * jnp.ones((N, K), dtype=jnp.float32)
    assert jnp.allclose(out.ln_n, expected)
    assert jnp.all(diag["converged"])
