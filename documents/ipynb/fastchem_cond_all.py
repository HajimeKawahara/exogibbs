#!/usr/bin/env python
# coding: utf-8

# # All Fastchem Gas and Cond
# 
# Hajime Kawahara 2025/11/27
# 
# 

# In[1]:


from jax import config
config.update("jax_enable_x64", True)


# We assume N2+H2O (gas, water, ice) system using fastchem/fastchem_cond presets. 

# In[2]:


from exogibbs.presets.fastchem_cond import chemsetup as condsetup
cond = condsetup()
from exogibbs.presets.fastchem import chemsetup as gassetup
gas = gassetup()


# In[3]:


import numpy as np
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
formula_matrix_gas_eff, formula_matrix_cond_eff, indep_element_mask = contract_formula_matrix(formula_matrix_gas, formula_matrix_cond)
#elements_eff =elements[indep_element_mask]

print("Formula matrix (gas):")
print(formula_matrix_gas_eff)
print("Formula matrix (cond):")
print(formula_matrix_cond_eff)
#print("independent elements:")
#print(elements_eff)


# Output the reference-state value of ( $h = \mu / (RT)$ ) at temperature ( T ).
# 

# ## minimization using minimize_gibbs_cond_core

# 

# In[5]:


from exogibbs.optimize.minimize_cond import minimize_gibbs_cond_core
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState

from exogibbs.optimize.core import compute_ln_normalized_pressure


# In[7]:



# Thermodynamic conditions
P = 1.0  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.zeros(formula_matrix_gas_eff.shape[1])  # log(n_species)
ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])   # log(n_condensates)
ln_ntot = jnp.log(jnp.sum(jnp.exp(ln_nk)))  # log(total number density)

epsilon = 0.0
epsilon_crit = -20.0

for i, temperature in enumerate([200.0]):

    #PD-IPM
    nkpath=[]
    mkpath=[]
    eppath=[]
    
    thermo_state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=ln_normalized_pressure,
        element_vector=b_ref,
    )



    while epsilon > epsilon_crit:
        epsilon = epsilon - 0.1
        rcrit = jnp.exp(epsilon)
        ln_nk, ln_mk, ln_ntot, counter = minimize_gibbs_cond_core(
            thermo_state,
            ln_nk_init=ln_nk,
            ln_mk_init=ln_mk,
            ln_ntot_init=ln_ntot,
            formula_matrix=formula_matrix_gas_eff,
            formula_matrix_cond=formula_matrix_cond_eff,
            hvector_func=gas.hvector_func,
            hvector_cond_func=cond.hvector_func,
            epsilon=epsilon,  ### new argument
            residual_crit=rcrit,
            max_iter=1000,
        )

        nkpath.append(jnp.exp(ln_nk)[0])
        mkpath.append(jnp.exp(ln_mk)[0])
        eppath.append(epsilon)
        print("Optimization:", ln_nk, "counter=", counter, "epsilon=", epsilon, "rcrit=", rcrit)
    

    


# In[ ]:





# In[ ]:




