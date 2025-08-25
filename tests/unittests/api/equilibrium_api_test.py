import pytest
import jax
import jax.numpy as jnp

from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import (
    equilibrium,
)

from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.smoke
def test_equilibrium_grad_wrt_temperature():
    setup = prepare_ykb4_setup()
    b_vec = setup.b_element_vector_reference

    def f(T):
        return jnp.sum(equilibrium(setup, T, 1.0, b_vec).ln_n)

    g = jax.grad(f)(300.0)
    assert jnp.isfinite(g)
