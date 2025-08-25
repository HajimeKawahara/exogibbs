import pytest
import jax
import jax.numpy as jnp

from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import (
    equilibrium,
    equilibrium_diagnostics,
    EquilibriumOptions,
)

from jax import config

config.update("jax_enable_x64", True)

@pytest.mark.smoke
def test_equilibrium_basic_shapes_and_sum():
    setup = prepare_ykb4_setup()

    # Use the packaged reference b vector if available for dimension safety
    assert setup.b_element_vector_reference is not None
    b_vec = setup.b_element_vector_reference

    T = 1200.0
    P = 1.0
    res = equilibrium(setup, T, P, b_vec)

    K = setup.formula_matrix.shape[1]
    assert isinstance(res.ln_n, jnp.ndarray)
    assert res.ln_n.shape == (K,)
    assert res.n.shape == (K,)
    assert res.x.shape == (K,)

    # Mole fractions sum to ~1
    s = jnp.sum(res.x)
    assert jnp.isfinite(s)
    assert jnp.abs(s - 1.0) < 1e-8


@pytest.mark.smoke
def test_equilibrium_with_dict_b_when_elems_available():
    setup = prepare_ykb4_setup()
    if setup.elems is None:
        pytest.skip("setup.elems not available; skipping dict-b test")

    # Build a dict from the reference vector for a small subset
    b_vec = setup.b_element_vector_reference
    elems = setup.elems
    # Map first few elements only; missing ones default to zero
    subset = {e: float(b_vec[i]) for i, e in enumerate(elems[: min(3, len(elems))])}

    T = 1000.0
    P = 0.5
    res = equilibrium(setup, T, P, subset)
    assert jnp.all(res.n >= 0.0)


@pytest.mark.smoke
def test_equilibrium_diagnostics_iterations():
    setup = prepare_ykb4_setup()
    b_vec = setup.b_element_vector_reference
    opts = EquilibriumOptions(epsilon_crit=1e-10, max_iter=200)

    res = equilibrium_diagnostics(setup, 900.0, 1.0, b_vec, options=opts)
    assert res.iterations is not None
    assert isinstance(res.iterations, int)
    assert res.iterations >= 0


@pytest.mark.smoke
def test_equilibrium_grad_wrt_temperature():
    setup = prepare_ykb4_setup()
    b_vec = setup.b_element_vector_reference

    def f(T):
        return jnp.sum(equilibrium(setup, T, 1.0, b_vec).ln_n)

    g = jax.grad(f)(1100.0)
    assert jnp.isfinite(g)

if __name__ == "__main__":
    setup = prepare_ykb4_setup()
    b_vec = setup.b_element_vector_reference

    def f(T):
        ln_n = equilibrium(setup, T, 1.0, b_vec).ln_n
        return ln_n
    
    Tin = 900.0
    from jax import jacrev
    g = jacrev(f)(Tin)
    print("df/dT(Tin)=",g)

    