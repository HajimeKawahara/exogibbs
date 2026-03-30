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


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(temperatures,pressures)
ax1.invert_yaxis()
plt.yscale("log")
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [bar]")
ax2 = fig.add_subplot(1, 2, 2)
hmatrix = cond.hvector_func(temperatures)
for i in range(0, hmatrix.shape[1]):
    if np.any(hmatrix[:, i] > 0.0):
        plt.plot(hmatrix[:, i],pressures, label=cond.species[i])
plt.yscale("log")
ax2.invert_yaxis()
plt.xlabel("Condensate h (mu/RT)")
plt.legend(loc="best")
plt.savefig("cond_hvector.png")
plt.show()
plt.close()
#exit()


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


from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import minimize_gibbs_cond as structured_minimize_gibbs_cond
from exogibbs.optimize.minimize_cond import minimize_gibbs_cond_profile
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
if N != vmr_ref.shape[1]:
    raise ValueError("Length mismatch between ln_nk[29:] and vmr_ref")
# for i in range(0, N):
#    idx_exogibbs = gas.species.index(plot_species[i])
#    print(idx_exogibbs)
print("Set up complete.")

import jax.numpy as jnp
from jax.scipy.special import logsumexp

init_setup = "gas_only"  # "zeros" or "gas_only"
profile_method = "scan_hot_from_bottom"  # or "vmap_cold" or "scan_hot_from_top"


def minimize_gibbs_cond_diagnostics(
    temperature,
    ln_normalized_pressure,
    ln_nk_init,
    ln_mk_init,
    ln_ntot_init,
):
    thermo_state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=ln_normalized_pressure,
        element_vector=b_ref,
    )
    epsilon = jnp.asarray(-40.0)
    residual_crit = jnp.exp(epsilon)
    return structured_minimize_gibbs_cond(
        thermo_state,
        init=CondensateEquilibriumInit(
            ln_nk=ln_nk_init,
            ln_mk=ln_mk_init,
            ln_ntot=ln_ntot_init,
        ),
        formula_matrix=formula_matrix_gas_eff,
        formula_matrix_cond=formula_matrix_cond_eff,
        hvector_func=gas.hvector_func,
        hvector_cond_func=cond.hvector_func,
        epsilon=epsilon,
        residual_crit=residual_crit,
        max_iter=100,
    )

if init_setup == "gas_only":
    from exogibbs.api.equilibrium import equilibrium

    ln_nk_init_list = []
    ln_ntot_init_list = []
    for temp, pres in zip(temperatures, pressures):
        result = equilibrium(gas, T=temp, P=pres, b=b_ref)
        ln_nk_init_list.append(result.ln_n)
        ln_ntot_init_list.append(logsumexp(result.ln_n))
    ln_nk_init = jnp.stack(ln_nk_init_list)
    ln_ntot_init = jnp.stack(ln_ntot_init_list)
    ln_mk_init = jnp.zeros(
        (ln_nk_init.shape[0], formula_matrix_cond_eff.shape[1])
    )
elif init_setup == "zeros":
    ln_nk_init = jnp.zeros(
        (len(temperatures), formula_matrix_gas_eff.shape[1])
    )
    ln_mk_init = jnp.zeros(
        (len(temperatures), formula_matrix_cond_eff.shape[1])
    )
    ln_ntot_init = logsumexp(ln_nk_init, axis=1)
else:
    raise ValueError("Invalid init_setup option")

import time

start = time.time()
profile_result = minimize_gibbs_cond_profile(
    jnp.array(temperatures),
    jnp.array(ln_normalized_pressures),
    b_ref,
    init=CondensateEquilibriumInit(
        ln_nk=ln_nk_init,
        ln_mk=ln_mk_init,
        ln_ntot=ln_ntot_init,
    ),
    formula_matrix=formula_matrix_gas_eff,
    formula_matrix_cond=formula_matrix_cond_eff,
    hvector_func=gas.hvector_func,
    hvector_cond_func=cond.hvector_func,
    epsilon_start=0.0,
    epsilon_crit=-40.0,
    n_step=100,
    max_iter=100,
    method=profile_method,
)
end = time.time()
print("Computation time (s):", end - start)
print("Profile solve method:", profile_method)

ln_nk = profile_result.ln_nk
ln_mk = profile_result.ln_mk
ln_ntot = profile_result.ln_ntot[:, None]
profile_diagnostics = profile_result.diagnostics.asdict()
print("Layer-0 condensate diagnostics:", {k: v[0] for k, v in profile_diagnostics.items()})

# Gibbs energy
from exogibbs.api.potential import gibbs_energies

ge = gibbs_energies(
    temperatures,
    pressures,
    gas,
    ln_nk,
    cond,
    ln_mk,
    nomalize=True,
)

print(ge)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(-ge, pressures)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("negative Normalized Gibbs energy - G/RT")
plt.ylabel("Pressure [bar]")
ax.invert_yaxis()
plt.legend()
plt.savefig("gibbs_energy.png")  # want to make "output/vmr_comparison0001.png"
plt.show()
plt.close()



# plotting
Nelespec = 29 
vmr_exogibbs = np.exp(ln_nk[:, Nelespec:] - ln_ntot)
vmr_elespec_exogibbs = np.exp(ln_nk[:, :Nelespec] - ln_ntot)
cond_exogibbs = np.exp(ln_mk[:, :] - ln_ntot)


fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
for i in range(0, N):
    color = "C"+str(i)
    plt.plot(vmr_ref[:, i], pressures, ".", alpha=0.3, color=color)
    plt.plot(vmr_exogibbs[:, i], pressures, alpha=0.3, color=color)

plt.xlim(1.0e-300, 1.0)
plt.xscale("log")
plt.yscale("log")
ax.invert_yaxis()
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
for i in range(0, Nelespec):
    color = "C"+str(i)
    plt.plot(vmr_elespec_exogibbs[:, i], pressures, alpha=0.3, color=color)
plt.xscale("log")
plt.yscale("log")
ax2.invert_yaxis()
plt.legend()

ax3 = fig.add_subplot(1, 3, 3)
for i in range(0, cond_exogibbs.shape[1]):
    color = "C"+str(i)
    plt.plot(cond_exogibbs[:, i], pressures, alpha=0.3, color=color)
plt.xscale("log")
plt.yscale("log")
ax3.invert_yaxis()
plt.legend()

plt.savefig("prof.png")  
plt.show()
plt.close()
