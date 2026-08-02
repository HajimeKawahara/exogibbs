import jax
import jax.numpy as jnp
import pytest

from jax import config

from exogibbs.presets.ykb4 import chemsetup
from exogibbs.api.equilibrium import (
    EquilibriumOptions,
    equilibrium,
    equilibrium_profile,
)


config.update("jax_enable_x64", True)


def test_equilibrium_profile_jit_under_grad():
    """
    Guard against regressions where a traced K-length init vector is
    captured as a JIT constant, causing InvalidInputException during
    NumPyro-like tracing (grad/JVP over a jitted call).

    This test differentiates through a jitted equilibrium_profile call.
    It should compile and return a finite gradient.
    """
    setup = chemsetup()
    b = setup.element_vector_reference

    # Small profile to keep runtime minimal
    N = 5
    P = jnp.linspace(0.1, 1.0, N)

    opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=200)

    # Jitted function as in the user workflow (captures only static objects)
    get_res = jax.jit(lambda T, Q: equilibrium_profile(setup, T, Q, b, options=opts))

    def f(alpha):
        T = jnp.clip(P ** alpha * 1000.0, 200.0, 1500.0)
        res = get_res(T, P)
        # Any smooth scalar; sum of ln n is simple and exercise AD path
        return jnp.sum(res.ln_n)

    g = jax.grad(f)(0.1)
    assert jnp.isfinite(g)


def test_equilibrium_forward_mode_jvp_is_explicitly_unsupported():
    """The gas solver's custom VJP intentionally supports reverse mode only."""

    setup = chemsetup()
    b = setup.element_vector_reference

    def f(temperature):
        return jnp.sum(equilibrium(setup, temperature, 1.0, b).ln_n)

    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(f, (300.0,), (1.0,))
