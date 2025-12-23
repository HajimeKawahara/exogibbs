#!/usr/bin/env python
# coding: utf-8

# # All Fastchem Gas and Cond for atmospheric profile
#
# Hajime Kawahara 2025/11/27
#
#

# In[1]:


from jax import config

config.update("jax_enable_x64", True)

# load reference
import numpy as np
import matplotlib.pyplot as plt

data = np.load("vmr_fastchem_prof.npz")
vmr_ref = data["vmr_fastchem"]
temperatures = np.atleast_1d(data["temperature"])
pressures = np.atleast_1d(data["pressure"])

# In[2]:


from exogibbs.presets.fastchem_cond import chemsetup as condsetup

cond = condsetup()
from exogibbs.presets.fastchem import chemsetup as gassetup

gas = gassetup()


# In[3]:


from exojax.utils.zsol import nsol
import jax.numpy as jnp

solar_abundance = nsol()
nsol_vector = jnp.array(
    [solar_abundance[el] for el in gas.elements[:-1]]
)  # no solar abundance for e-
element_vector = jnp.append(nsol_vector, 0.0)

formula_matrix_gas = gas.formula_matrix

print("Formula matrix (gas):")
print(formula_matrix_gas)

formula_matrix_cond = cond.formula_matrix

print("Formula matrix (cond):")
print(formula_matrix_cond)

b_ref = gas.element_vector_reference


# This setting yields rank(Ac, Ag) < |b_element| because formula_matrix_gas[:,0] = formula_matrix_cond[:,0]. We need to redefine the formulation using the matrix contraction.

# In[4]:


from exogibbs.thermo.stoichiometry import contract_formula_matrix

formula_matrix_gas_eff, formula_matrix_cond_eff, indep_element_mask = (
    contract_formula_matrix(formula_matrix_gas, formula_matrix_cond)
)
# elements_eff =elements[indep_element_mask]

print("Formula matrix (gas):")
print(formula_matrix_gas_eff)
print("Formula matrix (cond):")
print(formula_matrix_cond_eff)
# print("independent elements:")
# print(elements_eff)


# Output the reference-state value of ( $h = \mu / (RT)$ ) at temperature ( T ).
#

# ## minimization using minimize_gibbs_cond_core

#

# In[5]:


from exogibbs.optimize.pipm_cond import minimize_gibbs_cond_core
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState

from exogibbs.optimize.core import compute_ln_normalized_pressure


# In[7]:


# Thermodynamic conditions
Pref = 1.0  # bar, reference pressure
ln_normalized_pressures = compute_ln_normalized_pressure(pressures, Pref)
ln_normalized_pressures = jnp.atleast_1d(ln_normalized_pressures)

plot_species = gas.species[29:]
N = len(plot_species)
if N != len(vmr_ref):
    raise ValueError("Length mismatch between ln_nk[29:] and vmr_ref")
# for i in range(0, N):
#    idx_exogibbs = gas.species.index(plot_species[i])
#    print(idx_exogibbs)


import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import logsumexp

def minimize_gibbs_cond(temperature, ln_normalized_pressure):
    thermo_state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=ln_normalized_pressure,
        element_vector=b_ref,
    )

    # initial values
    ln_nk = jnp.zeros(formula_matrix_gas_eff.shape[1])
    ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])
    ln_ntot = logsumexp(ln_nk)


    epsilon_start = 0.0
    epsilon_crit = -30.0
    n_step = 300

    # epsilon schedule (static, safe)
    epsilons = jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:]

    def body_fn(i, state):
        ln_nk, ln_mk, ln_ntot = state

        epsilon = epsilons[i]
        rcrit = jnp.exp(epsilon)

        ln_nk, ln_mk, ln_ntot, _ = minimize_gibbs_cond_core(
            thermo_state,
            ln_nk_init=ln_nk,
            ln_mk_init=ln_mk,
            ln_ntot_init=ln_ntot,
            formula_matrix=formula_matrix_gas_eff,
            formula_matrix_cond=formula_matrix_cond_eff,
            hvector_func=gas.hvector_func,
            hvector_cond_func=cond.hvector_func,
            epsilon=epsilon,
            residual_crit=rcrit,
            max_iter=100,
        )

        return (ln_nk, ln_mk, ln_ntot)

    ln_nk, ln_mk, ln_ntot = lax.fori_loop(
        0,
        n_step,
        body_fn,
        (ln_nk, ln_mk, ln_ntot),
    )

    return ln_nk, ln_mk, ln_ntot

from jax import vmap
from jax import jit


vmap_minimize_gibbs_cond = vmap(minimize_gibbs_cond, in_axes=(0, 0))
jit_vmap_minimize_gibbs_cond = jit(vmap_minimize_gibbs_cond)

import time

start = time.time()
ln_nk, ln_mk, ln_ntot = jit_vmap_minimize_gibbs_cond(
    jnp.array(temperatures),
    jnp.array(ln_normalized_pressures),
)
end = time.time()
print("Computation time (s):", end - start)


#vmr_exogibbs = np.exp(ln_nk[:, 29:]) / np.sum(np.exp(ln_nk), axis=1)[:, None]
vmr_exogibbs = np.exp(ln_nk[:, 29:] - logsumexp(ln_nk, axis=1)[:, None])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(0, N):
    plt.plot(vmr_exogibbs[:, i], pressures, ".", alpha=0.3)

plt.xlim(1.0e-300, 1.0)
plt.xscale("log")
plt.yscale("log")
ax.invert_yaxis()
plt.legend()
plt.savefig("prof.png")  # want to make "output/vmr_comparison0001.png"
plt.show()
plt.close()

# In[ ]:


# In[ ]:
