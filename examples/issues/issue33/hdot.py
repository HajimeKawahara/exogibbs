from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium
import jax.numpy as jnp
from jax import jacrev
from jax import config

config.update("jax_enable_x64", True)


setup = prepare_ykb4_setup()
b_vec = setup.b_element_vector_reference

def f(T):
    ln_n = equilibrium(setup, T, 1.0, b_vec).ln_n
    return ln_n

#this works
Tin = 300.0
g = jacrev(f)(Tin)
assert jnp.isfinite(jnp.sum(g))


#this does not works
Tin = 900.0
g = jacrev(f)(Tin)
assert jnp.isfinite(jnp.sum(g))

