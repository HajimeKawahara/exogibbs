import jax
import jax.numpy as jnp

from jax import config

from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions


config.update("jax_enable_x64", True)


def test_equilibrium_profile_jit_under_grad():
    """
    Guard against regressions where a traced K-length init vector is
    captured as a JIT constant, causing InvalidInputException during
    NumPyro-like tracing (grad/JVP over a jitted call).

    This test differentiates through a jitted equilibrium_profile call.
    It should compile and return a finite gradient.
    """
    setup = prepare_ykb4_setup()
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

