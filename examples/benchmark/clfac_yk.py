"""
Validation of Gibbs Minimization Against ykawashima's B4 code
============================================================

This example demonstrates and validates the ExoGibbs thermochemical equilibrium
solver against the code by ykawashima when she was at B4.

Updated to use the high-level API: exogibbs.api.equilibrium.equilibrium.
"""
from exogibbs.presets.ykb4 import prepare_ykb4_setup
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.minimize import minimize_gibbs_core

from exogibbs.optimize.core import compute_ln_normalized_pressure

from jax import config

config.update("jax_enable_x64", True)

#chemical setup
chem = prepare_ykb4_setup()

# Thermodynamic conditions
temperature = 500.0  # K
P = 10.0  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.zeros(chem.formula_matrix.shape[1])  # log(n_species)
ln_ntot = 0.0  # log(total number density)

# old reference elemental abundance b from yk's sample number densities
b_old_ref = jnp.array(
    [
        4.8774824e-04,
        1.6749767e00,
        1.6143440e-01,
        2.5438149e-07,
        1.3435642e-04,
        3.9624806e-06,
        9.7356725e-04,
        5.7690579e-07,
        3.0653933e-05,
        1.6687756e-07,
        1.9870969e-08,
        0.0000000e00,
    ]
)
# ThermoState instance
thermo_state = ThermoState(
    temperature=temperature,
    ln_normalized_pressure=ln_normalized_pressure,
    element_vector=b_old_ref,
)

# Convergence criteria
epsilon_crit = 1e-11
max_iter = 1000

ln_nk_result, _, icount = minimize_gibbs_core(
    thermo_state,
    ln_nk,
    ln_ntot,
    chem.formula_matrix,
    chem.hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
)
nk_result = jnp.exp(ln_nk_result)
print("icount", icount) 
#702 for clfac = 0.1, 347 for _cea_lambda