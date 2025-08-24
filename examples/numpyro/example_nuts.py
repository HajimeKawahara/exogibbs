##############################################################################
# Example: NUTS Sampling with Numpyro for Thermochemical Equilibrium
# -------------------------------------------------------------
# This example demonstrates how to use Numpyro's NUTS sampler to perform
# Bayesian inference on a thermochemical equilibrium problem, specifically
# for the HCO system. It includes Gibbs minimization and analytical validation.
#
# We infer the temperature and element abundances (bH, bC, bO) from noisy
# log number densities of species in the HCO system.
#

# Generates Ground-Truth
from exogibbs.api.thermochem import ThermoState
from exogibbs.optimize.minimize import minimize_gibbs
from exogibbs.test.analytic_hcosystem import HCOSystem
from exogibbs.optimize.core import compute_ln_normalized_pressure
import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

##############################################################################
# Setup Test System and Parameters
# ---------------------------------
# We initialize the analytical HCO system and define the thermochemical
# equilibrium problem parameters.

# Initialize the analytic HCO system
hcosystem = HCOSystem()

# Define stoichiometric constraint matrix:
# Species order: [H₂, CO, CH₄, H₂O]
# Elements order: [H, C, O]
formula_matrix = jnp.array(
    [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [4.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
).T

# check if the formula matrix is full row rank
rank = np.linalg.matrix_rank(formula_matrix)
print("formula matrix is row-full rank",rank == formula_matrix.shape[0])

# Thermodynamic conditions
temperature_gt = 1500.0  # K
P = 1.5  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.array([0.0, 0.0, 0.0, 0.0])  # log(n_H₂), log(n_CO), log(n_CH₄), log(n_H₂O)
ln_ntot = 0.0  # log(total number density)

def hvector_func(temperature):
    """Chemical potential function h(T) = μ°(T)/RT for [H₂, CO, CH₄, H₂O]"""
    return hcosystem.hv_hco(temperature)

# Element abundance constraint:
bH_gt = 0.5
bC_gt = 0.2
bO_gt = 0.3
b_element_vector_gt = jnp.array([bH_gt, bC_gt, bO_gt])  # H, C, O

# ThermoState instance
thermo_state_gt = ThermoState(temperature_gt, ln_normalized_pressure, b_element_vector_gt)

# Convergence criteria
epsilon_crit = 1e-11
max_iter = 1000
ln_nk_ground_truth = minimize_gibbs(
    thermo_state_gt,
    ln_nk,
    ln_ntot,
    formula_matrix,
    hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
)


print(
    f"Log number densities (GT): ln(n_H2)={ln_nk_ground_truth[0]:.6f}, ln(n_CO)={ln_nk_ground_truth[1]:.6f}, ln(n_CH4)={ln_nk_ground_truth[2]:.6f}, ln(n_H2O)={ln_nk_ground_truth[3]:.6f}"
)

from numpy.random import normal
sigma = 0.1
ln_nk_obs = jnp.array(ln_nk_ground_truth + normal(0, sigma, ln_nk_ground_truth.shape))
print(ln_nk_obs)

# Numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from jax import random

def model_prob(ln_nk_obs):

    #atmospheric/spectral model parameters priors
    bH = numpyro.sample('bH', dist.Uniform(0.1, 1.0))
    bC = numpyro.sample('bC', dist.Uniform(0.1, 0.3))
    bO = numpyro.sample('bO', dist.Uniform(0.1, 0.5))
    b_element_vector = jnp.array([bH, bC, bO]) 
    logT = numpyro.sample('logT', dist.Uniform(3.0, 3.5)) 
    temperature = 10**(logT)
    numpyro.deterministic('temperature', temperature)
    
    # ThermoState instance
    thermo_state = ThermoState(temperature, ln_normalized_pressure, b_element_vector)

    mu_ln_nk = minimize_gibbs(
    thermo_state,
    ln_nk,
    ln_ntot,
    formula_matrix,
    hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
    )

    numpyro.sample('ln_nk', dist.Normal(jnp.array(mu_ln_nk), sigma), obs=ln_nk_obs)

# Set up MCMC sampling
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1500, 1000
#kernel = NUTS(model_prob, forward_mode_differentiation=True)
kernel = NUTS(model_prob, forward_mode_differentiation=False)

# Run MCMC
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, ln_nk_obs=jnp.array(ln_nk_obs))
mcmc.print_summary()

import arviz
import matplotlib.pyplot as plt
pararr = ['temperature', 'bH', 'bC', 'bO']
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True,
                var_names=pararr)
plt.show()