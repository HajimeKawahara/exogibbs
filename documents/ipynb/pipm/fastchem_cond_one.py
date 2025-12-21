#!/usr/bin/env python
# coding: utf-8

# # All Fastchem Gas and Cond
# 
# Hajime Kawahara 2025/11/27
# 
# 

# In[1]:


from turtle import st
from blosc2 import unpack
from jax import config
config.update("jax_enable_x64", True)

# load reference
import numpy as np
import matplotlib.pyplot as plt

data = np.load("vmr_fastchem.npz")
vmr_ref = data["vmr_fastchem"]
tin = data["temperature"][0]
pin = data["pressure"][0]

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


from exogibbs.optimize.pipm_cond import minimize_gibbs_cond_core
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState

from exogibbs.optimize.core import compute_ln_normalized_pressure


# In[7]:



# Thermodynamic conditions
P = pin  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.zeros(formula_matrix_gas_eff.shape[1])  # log(n_species)


plot_species = gas.species[29:]
N = len(plot_species)
if N != len(vmr_ref):
    raise ValueError("Length mismatch between ln_nk[29:] and vmr_ref")
#for i in range(0, N):
#    idx_exogibbs = gas.species.index(plot_species[i])
#    print(idx_exogibbs)

ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])   # log(n_condensates)
ln_ntot = jnp.log(jnp.sum(jnp.exp(ln_nk)))  # log(total number density)

epsilon = 0.0
epsilon_crit = -40.0

for i, temperature in enumerate([tin]):

    #PD-IPM
    nkpath=[]
    mkpath=[]
    eppath=[]
    
    thermo_state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=ln_normalized_pressure,
        element_vector=b_ref,
    )


    iter=0
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
    


        vmr_exogibbs = np.exp(ln_nk[29:])/np.sum(np.exp(ln_nk))

        fig = plt.figure()
        plt.plot(vmr_exogibbs, ".", label="ExoGibbs", alpha=0.3)
        plt.plot(vmr_ref, "o", label="FastChem", alpha=0.3)
        plt.ylim(1.e-300,1.0)
        plt.yscale("log")
        plt.legend()
        plt.savefig("output/vmr_comparison"+str(iter).zfill(4)+".png") # want to make "output/vmr_comparison0001.png"
        plt.close()
        iter = iter+1

# In[ ]:





# In[ ]:




