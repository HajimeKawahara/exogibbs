"""
Validation of Gibbs Minimization for Layered Systems 
===================================================================================

This example demonstrates and validates the ExoGibbs thermochemical equilibrium for the layered system

Updated to use the high-level API: exogibbs.api.equilibrium.equilibrium.
"""

from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium, EquilibriumOptions
from jax import vmap
import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


# Thermodynamic conditions
Pref = 1.0  # bar, reference pressure
Parr = jnp.logspace(-8, 2, 10)/Pref  # bar
Tarr = 1500*Parr**0.1


#chemical setup
chem = prepare_ykb4_setup()

##############################################################################
# Solve equilibrium via high-level API
# ------------------------------------
opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=1000)

# Vectorize over T and P via a wrapper to avoid mapping kwargs
equilibrium_layer = vmap(
    lambda T, P: equilibrium(
        chem,
        T,
        P,
        chem.b_element_vector_reference,
        Pref=Pref,
        options=opts,
    ),
    in_axes=(0, 0),
)
res = equilibrium_layer(Tarr, Parr)

##############################################################################
nk_result = res.n
print(nk_result)