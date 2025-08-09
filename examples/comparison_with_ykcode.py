"""
Validation of Gibbs Minimization Against ykawashima's B4 code
============================================================

This example demonstrates and validates the ExoGibbs thermochemical equilibrium
solver against the code by ykawashima when she was at B4.

"""

from exogibbs.optimize.minimize import minimize_gibbs
from exogibbs.test.analytic_hcosystem import HCOSystem
from exogibbs.optimize.core import compute_ln_normalized_pressure
import numpy as np
from jax import jacrev
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

##############################################################################
# Setup Test System and Parameters
# ---------------------------------
# We initialize the analytical H system and define the thermochemical
# equilibrium problem parameters.

# Initialize the analytic HCO system
hcosystem = HCOSystem()

# Define stoichiometric constraint matrix: [H atoms per species]
# Species order: [H₂, CO, CH₄, H₂O]
# Elements order: [H, C, O]
formula_matrix = jnp.array(
    [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [4.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
).T

# check if the formula matrix is full raw rank
rank = np.linalg.matrix_rank(formula_matrix)
print("formula matrix is row-full rank",rank == formula_matrix.shape[0])

# Thermodynamic conditions
temperature = 1500.0  # K
P = 1.5  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.array([0.0, 0.0, 0.0, 0.0])  # log(n_H₂), log(n_CO), log(n_CH₄), log(n_H₂O)
ln_ntot = 0.0  # log(total number density)


def hvector_func(temperature):
    """Chemical potential function h(T) = μ°(T)/RT for [H₂, CO, CH₄, H₂O]"""
    return hcosystem.hv_hco(temperature)


# Element abundance constraint: total H nuclei = 1.0
bH = 0.5
bC = 0.2
bO = 0.3
b_element_vector = jnp.array([bH, bC, bO])  # H, C, O

# Convergence criteria
epsilon_crit = 1e-11
max_iter = 1000

##############################################################################
# Single-Point Equilibrium Validation
# ------------------------------------
# First, we solve for equilibrium at a single temperature and pressure point
# using both the core and main minimize_gibbs functions.

# Run Gibbs minimization using core function (returns iteration count)

ln_nk_result = minimize_gibbs(
    temperature,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk,
    ln_ntot,
    formula_matrix,
    hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
)


print(
    f"Log number densities: ln(n_H2)={ln_nk_result[0]:.6f}, ln(n_CO)={ln_nk_result[1]:.6f}, ln(n_CH4)={ln_nk_result[2]:.6f}, ln(n_H2O)={ln_nk_result[3]:.6f}"
)
from exogibbs.test.analytic_hcosystem import function_equilibrium

hco_system = HCOSystem()
k = hco_system.equilibrium_constant(temperature, P / Pref)
n_CO = jnp.exp(ln_nk_result[1])
res = function_equilibrium(n_CO, k, bC, bH, bO)
assert jnp.abs(res) < epsilon_crit * 10.0

# mainly from here

# element derivatives
from exogibbs.test.analytic_hcosystem import derivative_dlnnCO_db

dlnn_db = jacrev(
    lambda b_element_vector_in: minimize_gibbs(
        temperature,
        ln_normalized_pressure,
        b_element_vector_in,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
)(b_element_vector) # (n_species, n_elements)

# analytical derivatives
gradf = derivative_dlnnCO_db(ln_nk_result[1], bC, bH, bO, k)

diff = jnp.abs(dlnn_db[1,:] / gradf - 1.0)

assert jnp.all(
    diff < 1.0e-5
), f"Derivative mismatch: {diff}"  # 2.32238010e-06 4.02220479e-11 1.80632038e-06 2025/8/7

# to here
