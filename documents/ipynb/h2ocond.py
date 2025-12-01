#!/usr/bin/env python
# coding: utf-8

# # N2 + H2O (gas, condensate)
# 
# Hajime Kawahara 2025/11/16
# 
# In this notebook, we adopt N2 as the background atmosphere and water as a species that can exist either in the gas phase or as a condensate. We then examine how the phase of water is determined through Gibbs free‐energy minimization.

# In[1]:


from math import e
from jax import config
config.update("jax_enable_x64", True)


# We assume N2+H2O (gas, water, ice) system using fastchem/fastchem_cond presets. 

# In[2]:


from exogibbs.presets.fastchem_cond import chemsetup as condsetup
cond = condsetup()
from exogibbs.presets.fastchem import chemsetup as gassetup
gas = gassetup()


# In[3]:


gas_species = list(gas.species)
gas_system = ['H2O1', 'N2']
index_h2o_gas = gas_species.index('H2O1')  
index_n2_gas = gas_species.index('N2')

cond_species = list(cond.species)
cond_system = ['H2O(s,l)']
index_h2o_cond = cond_species.index('H2O(s,l)')  


# In[4]:


import numpy as np
from exogibbs.thermo.stoichiometry import build_formula_matrix
from exogibbs.utils.nameparser import set_elements_from_components
from exogibbs.utils.nameparser import generate_components_from_formula_list

components_g = generate_components_from_formula_list(gas_system)
elements = np.array(["H", "N", "O"])  # fixed ordering for this notebook
formula_matrix_gas = build_formula_matrix(components_g, elements)

print("Formula matrix (gas):")
print(formula_matrix_gas)

components_c = generate_components_from_formula_list(cond_system)
formula_matrix_cond = build_formula_matrix(components_c, elements)

print("Formula matrix (cond):")
print(formula_matrix_cond)



# This setting yields rank(Ac, Ag) < |b_element| because formula_matrix_gas[:,0] = formula_matrix_cond[:,0]. We need to redefine the formulation using the matrix contraction. 

# In[5]:


from exogibbs.thermo.stoichiometry import contract_formula_matrix
formula_matrix_gas_eff, formula_matrix_cond_eff, indep_element_mask = contract_formula_matrix(formula_matrix_gas, formula_matrix_cond)
elements_eff =elements[indep_element_mask]

print("Formula matrix (gas):")
print(formula_matrix_gas_eff)
print("Formula matrix (cond):")
print(formula_matrix_cond_eff)
print("independent elements:")
print(elements_eff)


# Output the reference-state value of ( $h = \mu / (RT)$ ) at temperature ( T ).
# 

# In[6]:


def h2o_cond_h_values(T): 
    return  cond.hvector_func(T)[index_h2o_cond]

def h2o_gas_h_values(T, p, nH2O, nN2):
    ntot = nH2O + nN2
    return  gas.hvector_func(T)[index_h2o_gas] + np.log(p*nH2O/ntot)

def n2_gas_h_values(T, p, nH2, nN2):
    ntot = nH2 + nN2
    return  gas.hvector_func(T)[index_n2_gas] + np.log(p*nN2/ntot)


# We have the analytic solution this system, in fact...

# In[7]:


import jax.numpy as jnp
def nh2o_analytic(p, delta_hvector, n_gas_max, n_bkgd_max):
    k = jnp.exp(- delta_hvector)
    nast = n_bkgd_max * k / (p - k)
    return jnp.min(jnp.array([n_gas_max, nast]))
        


# 
# If there exists an nH2O for which the chemical potential of the gas equals that of the condensate, the gas and condensate coexist in a gas–condensate equilibrium. If no such value exists, the chemical potential of the gas is always lower and only the gas is present.

# In[8]:


import matplotlib.pyplot as plt
bN = 0.99
bH = 0.01
p = 1.0 # bar
        


# The amount of nH2O can also be verified directly by minimizing the total Gibbs energy.

# In[9]:

# Derive the minimum using a grid search.

# In[12]:




from exogibbs.optimize.minimize_cond import minimize_gibbs_cond_core
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState

from exogibbs.optimize.core import compute_ln_normalized_pressure


# In[22]:


# Thermodynamic conditions


P = 1.0  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
epsilon = -1.0
ln_nk = jnp.zeros(formula_matrix_gas_eff.shape[1])  # log(n_species)
ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])   # log(n_condensates)
ln_ntot = jnp.log(jnp.sum(jnp.exp(ln_nk)))  # log(total number density)

b_ref = jnp.array([0.01,0.99]) #bH,bN


def hvector_cond_func(T): 
    return  jnp.array([cond.hvector_func(T)[index_h2o_cond]])

def hvector_func(T):
    return  jnp.array([gas.hvector_func(T)[index_h2o_gas],gas.hvector_func(T)[index_n2_gas]])


for i, temperature in enumerate([250.0]):

    #analytic solution
    dh =  gas.hvector_func(temperature)[index_h2o_gas] - cond.hvector_func(temperature)[index_h2o_cond]
    nh2_ana = nh2o_analytic(P, dh, bH/2, bN/2)

    #PD-IPM
    nkpath=[]
    mkpath=[]
    
    thermo_state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=ln_normalized_pressure,
        element_vector=b_ref,
    )


    epsilon_crit = -20.0
    while epsilon > epsilon_crit:
        epsilon = epsilon - 0.5
    
        ln_nk, ln_mk, ln_ntot, counter = minimize_gibbs_cond_core(
            thermo_state,
            ln_nk_init=ln_nk,
            ln_mk_init=ln_mk,
            ln_ntot_init=ln_ntot,
            formula_matrix=formula_matrix_gas_eff,
            formula_matrix_cond=formula_matrix_cond_eff,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=epsilon,  ### new argument
            residual_crit=1.0e-10,
            max_iter=100,
        )

        nkpath.append(jnp.exp(ln_nk)[0])
        mkpath.append(jnp.exp(ln_mk)[0])
        print("Optimization:", jnp.exp(ln_nk)[0], "Analytic:",nh2_ana, "counter=", counter)
        #print(jnp.exp(ln_etak), jnp.exp(ln_nk), jnp.exp(ln_mk), jnp.exp(ln_ntot), jnp.exp(epsilon))
    

import matplotlib.pyplot as plt
nkpath = jnp.array(nkpath)
mkpath = jnp.array(mkpath)
plt.plot(nkpath, mkpath, ".")
plt.plot(nkpath, mkpath, alpha=0.3)
plt.plot(nh2_ana, bH/2 - nh2_ana, "rx", markersize=12)
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("n (gas)")
plt.ylabel("m (condensate)")
plt.show()

# In[ ]:




