#!/usr/bin/env python
# coding: utf-8

# # All Fastchem Gas and Cond
# 
# Hajime Kawahara 2025/11/27
# 
# 

# In[1]:

import jax.numpy as jnp
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

# load reference
import numpy as np
import matplotlib.pyplot as plt

data = np.load("vmr_fastchem.npz")
vmr_ref = data["vmr_fastchem"]
tin = data["temperature"][0]
pin = data["pressure"][0]


# we solved nan issue for this set:
#tin = 1.102800e+03
#pin = 3.679186e+00	

#still have nan issue for this set:
#tin = 6.055400e+02
#pin = 3.583005e-01	
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


from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics
import jax.numpy as jnp
from exogibbs.api.chemistry import ThermoState

from exogibbs.optimize.core import compute_ln_normalized_pressure


# In[7]:



# Thermodynamic conditions
P = pin  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

#init_setup = "gas_only"  # "zeros" or "gas_only"
init_setup = "gas_only"
# (0 setup) Initial guess for log number densities
if init_setup == "zeros":
    ln_nk = jnp.zeros(formula_matrix_gas_eff.shape[1])  # log(n_species)
    plot_species = gas.species[29:]
    N = len(plot_species)
    if N != len(vmr_ref):
        raise ValueError("Length mismatch between ln_nk[29:] and vmr_ref")
    ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])   # log(n_condensates)
    ln_ntot = jnp.log(jnp.sum(jnp.exp(ln_nk)))  # log(total number density)

elif init_setup == "gas_only":
    from exogibbs.api.equilibrium import equilibrium
    b = gas.element_vector_reference  # or your own jnp.array([...])
    result = equilibrium(gas, T=tin, P=pin, b=b)
    ln_nk = result.ln_n
    ln_ntot = jnp.log(jnp.sum(jnp.exp(ln_nk)))  # log(total number density)
    ln_mk = jnp.zeros(formula_matrix_cond_eff.shape[1])   # log(n_condensates)
else:
    raise ValueError("Invalid init_setup option")


epsilon_start = 0.0
epsilon_crit = -30.0
n_step = 300

# epsilon schedule (static, safe)
epsilons = jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:]

#n_step = 400
#epsilons = jnp.linspace(-20.0, epsilon_crit, n_step + 1)[1:]


temperature = tin

#P-IPM

thermo_state = ThermoState(
    temperature=temperature,
    ln_normalized_pressure=ln_normalized_pressure,
    element_vector=b_ref,
)


def run_condensate_step(ln_nk, ln_mk, ln_ntot, epsilon, residual_crit):
    return minimize_gibbs_cond_with_diagnostics(
        thermo_state,
        ln_nk_init=ln_nk,
        ln_mk_init=ln_mk,
        ln_ntot_init=ln_ntot,
        formula_matrix=formula_matrix_gas_eff,
        formula_matrix_cond=formula_matrix_cond_eff,
        hvector_func=gas.hvector_func,
        hvector_cond_func=cond.hvector_func,
        epsilon=epsilon,
        residual_crit=residual_crit,
        max_iter=100,
        debug_nan=False,
    )


def pipm_fori_body(i, state):
    ln_nk, ln_mk, ln_ntot = state
    epsilon = epsilons[i]
    rcrit = jnp.exp(epsilon)

    ln_nk, ln_mk, ln_ntot, diagnostics = run_condensate_step(
        ln_nk, ln_mk, ln_ntot, epsilon, rcrit
    )

    _ = diagnostics
    return (ln_nk, ln_mk, ln_ntot)


ln_nk, ln_mk, ln_ntot = lax.fori_loop(
    0,
    n_step,
    pipm_fori_body,
    (ln_nk, ln_mk, ln_ntot),
)

final_epsilon = epsilons[-1]
final_rcrit = jnp.exp(final_epsilon)
_, _, _, diagnostics = run_condensate_step(
    ln_nk, ln_mk, ln_ntot, final_epsilon, final_rcrit
)

vmr_exogibbs = np.exp(ln_nk[29:])/np.sum(np.exp(ln_nk))
print(ln_nk)
print("Final condensate diagnostics:", diagnostics)
#print(vmr_exogibbs)



fig = plt.figure()
plt.plot(vmr_exogibbs, ".", label="ExoGibbs", alpha=0.3)
plt.plot(vmr_ref, "o", label="FastChem", alpha=0.3)
plt.ylim(1.e-300,1.0)
plt.yscale("log")
plt.legend()
plt.savefig("output/vmr_comparison_final.png") 
plt.close()
    
exit()
# %%
# if you need to see the iteration process, use the following loop instead of lax.while_loop (but slow)
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
        max_iter=100,
        debug_nan=True,
    )

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
