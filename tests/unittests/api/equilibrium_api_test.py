import pytest
import jax
import jax.numpy as jnp

from exogibbs.presets.ykb4 import chemsetup
from exogibbs.api.equilibrium import (
    equilibrium,
)
# tests/unittests/api/test_equilibrium_interface.py
import pytest
import jax.numpy as jnp

import exogibbs.api.equilibrium as eqmod
from exogibbs.api.equilibrium import EquilibriumInit, EquilibriumOptions

from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.smoke
def test_equilibrium_grad_wrt_temperature():
    setup = setup()
    b_vec = setup.element_vector_reference

    def f(T):
        return jnp.sum(equilibrium(setup, T, 1.0, b_vec).ln_n)

    g = jax.grad(f)(300.0)
    assert jnp.isfinite(g)

# tests/unittests/api/test_equilibrium_interface.py
import importlib
import pytest
import jax.numpy as jnp

# Load the module explicitly to avoid aliasing a function as a module by mistake
eqmod = importlib.import_module("exogibbs.api.equilibrium")
from exogibbs.api.equilibrium import EquilibriumInit, EquilibriumOptions


class FakeSetup:
    """Minimal stand-in for ChemicalSetup for interface testing."""
    def __init__(self, A, elements=None):
        self.formula_matrix = A
        self.elements = elements

    def hvector_func(self, T):
        # Shape must be (K,), but values are irrelevant for these tests
        K = self.formula_matrix.shape[1]
        return jnp.zeros((K,))


def test_equilibrium_happy_path(monkeypatch):
    """
    Happy path: patch minimize_gibbs to always return ln_n = 0.
    Expectation:
      - n = 1 for all species
      - ntot = K
      - x is uniform and sums to 1
      - Shapes and dtypes are consistent
    """
    E, K = 3, 5
    A = jnp.array([[1, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 1]], dtype=jnp.float64)
    setup = FakeSetup(A)

    def stub_minimize_gibbs(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        # Sanity check on inputs
        assert A_in.shape == A.shape
        assert ln_nk0.shape == (K,)
        assert ln_ntot0.shape == ()
        return jnp.zeros((K,), dtype=jnp.result_type(ln_nk0, A_in.dtype))

    # Patch by fully-qualified name so it works regardless of how the module was imported elsewhere
    monkeypatch.setattr("exogibbs.api.equilibrium.minimize_gibbs", stub_minimize_gibbs, raising=True)

    b = jnp.array([2.0, 1.0, 3.0], dtype=jnp.float64)
    out = eqmod.equilibrium(setup, T=1000.0, P=1.0, b=b,
                            options=EquilibriumOptions(epsilon_crit=1e-11, max_iter=50))

    assert out.ln_n.shape == (K,)
    assert out.n.shape == (K,)
    assert out.x.shape == (K,)
    assert out.ntot.shape == ()

    assert jnp.allclose(out.ln_n, 0.0)
    assert jnp.allclose(out.n, 1.0)
    assert jnp.isclose(out.ntot, K)
    assert jnp.allclose(out.x, jnp.ones(K) / K)
    assert jnp.isclose(out.x.sum(), 1.0)


def test_equilibrium_b_shape_validation(monkeypatch):
    """
    If b length does not match the number of elements E, raise ValueError.
    """
    E, K = 2, 3
    A = jnp.ones((E, K))
    setup = FakeSetup(A)

    # Not strictly needed for this test (should fail before calling the minimizer),
    # but patch anyway to keep isolation consistent.
    monkeypatch.setattr(
        "exogibbs.api.equilibrium.minimize_gibbs",
        lambda *args, **kw: jnp.zeros((K,)),
        raising=True,
    )

    b_bad = jnp.array([1.0, 2.0, 3.0])  # length mismatch (3 vs E=2)
    with pytest.raises(ValueError):
        eqmod.equilibrium(setup, T=500.0, P=1.0, b=b_bad)


def test_equilibrium_respects_init(monkeypatch):
    """
    If EquilibriumInit is provided, its values must be passed to the minimizer.
    Stub returns ln_n = ln_nk0 + c, so the output reflects the given init.
    """
    E, K = 2, 4
    A = jnp.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]], dtype=jnp.float32)
    setup = FakeSetup(A)

    c = 1.2345
    captured = {}

    def stub_minimize_gibbs(state, ln_nk0, ln_ntot0, A_in, hfunc, **kwargs):
        captured["ln_nk0"] = ln_nk0
        captured["ln_ntot0"] = ln_ntot0
        return ln_nk0 + c

    monkeypatch.setattr("exogibbs.api.equilibrium.minimize_gibbs", stub_minimize_gibbs, raising=True)

    ln_nk_init = jnp.full((K,), 0.3, dtype=jnp.float32)
    ln_ntot_init = jnp.asarray(0.7, dtype=jnp.float32)

    b = jnp.array([1.0, 1.0], dtype=jnp.float32)
    out = eqmod.equilibrium(
        setup, T=800.0, P=0.1, b=b,
        init=EquilibriumInit(ln_nk=ln_nk_init, ln_ntot=ln_ntot_init),
        options=EquilibriumOptions()
    )

    # Ensure minimizer saw the init values
    assert jnp.allclose(captured["ln_nk0"], ln_nk_init)
    assert jnp.allclose(captured["ln_ntot0"], ln_ntot_init)

    # Stub spec: ln_n = ln_nk0 + c
    assert jnp.allclose(out.ln_n, ln_nk_init + c)

    # Output consistency
    assert out.ln_n.shape == (K,)
    assert out.n.shape == (K,)
    assert out.x.shape == (K,)
    assert out.ntot.shape == ()
    assert jnp.isclose(out.x.sum(), 1.0)

if __name__ == "__main__":
    chemsetup = chemsetup()
    b_vec = chemsetup.element_vector_reference

    def f(T):
        return jnp.sum(equilibrium(chemsetup, T, 1.0, b_vec).ln_n)

    g = jax.grad(f)(300.0)
    print(g)