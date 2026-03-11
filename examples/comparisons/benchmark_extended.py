# comparison_with_fastchem.py
# This script compares the chemical equilibrium calculations of FastChem and ExoGibbs (fastchem preset).
# It requires the FastChem Python bindings to be installed
# also ExoJAX is required to set solar abundances (you can cahnge if you want)
from more_itertools import strip

import pyfastchem
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions


from jax import config

config.update("jax_enable_x64", True)

# some input values for temperature (in K) and pressure (in bar)
T = 3000 #K
Nlayer = 100
temperature = np.full(Nlayer, T)
pressure = np.logspace(-8, 2, num=Nlayer)


# define the directory for the output
# here, we currently use the standard one from FastChem
output_dir = "../output"


# First, we have to create a FastChem object
fastchem = pyfastchem.FastChem(
    "../input/element_abundances/asplund_2020_extended.dat", "../input/logK/logK_extended.dat", 1
)

# create the input and output structures for FastChem
input_data = pyfastchem.FastChemInput()
output_data = pyfastchem.FastChemOutput()

input_data.temperature = temperature
input_data.pressure = pressure


# run FastChem on the entire p-T structure
fastchem_flag = fastchem.calcDensities(input_data, output_data)

print("FastChem reports:")
print("  -", pyfastchem.FASTCHEM_MSG[fastchem_flag])

# ExoGibbs comparison###############################################################
# Thermodynamic conditions
# from exogibbs.presets.ykb4 import chemsetup
from exogibbs.presets.fastchem import chemsetup

from exojax.utils.zsol import nsol
import jax.numpy as jnp

chem = chemsetup(path="fastchem/logK/logK_extended.dat", species_defalt_elements=False, element_file="fastchem/element_abundances/asplund_2020_extended.dat")
solar_abundance = nsol()
na_value = 1.e-14 # abundance for elements solar abundance is unavailable
nsol_vector = []
for el in chem.elements[:-1]:
    try:
        nsol_vector.append(solar_abundance[el])
    except:
        nsol_vector.append(na_value)
        print("no info on " ,el, "solar abundance. set",na_value)
nsol_vector = jnp.array([nsol_vector])  # no solar abundance for e-
element_vector = jnp.append(nsol_vector, 0.0)

import time
ts = time.time()
#opts = EquilibriumOptions(method="scan_hot_from_top", epsilon_crit=1e-10, max_iter=1000) "1.07sec/run"
opts = EquilibriumOptions(method="scan_hot_from_bottom", epsilon_crit=1e-10, max_iter=1000) #1.05sec/run
#opts = EquilibriumOptions(method="vmap_cold", epsilon_crit=1e-10, max_iter=1000) #2.11sec/run
niter = 10
temperature = temperature - niter
for j in range(0, niter):
    temperature = temperature + 1.0
    res, diag = equilibrium_profile(
        chem,
        temperature,
        pressure,
        element_vector,
        Pref=1.0,
        options=opts,
        return_diagnostics=True
    )
    nk_result = res.x
    #print(diag)
te = time.time() - ts
print("ExoGibbs calculation time:", te, "seconds")
##################################################################################
    

plot_species = ["H2O1", "C1O2", "C1O1", "C1H4", "H3N1", "Fe1", "H1", "e1-"]
plot_species_labels = ["H2O", "CO2", "CO", "CH4", "NH3", "Fe", "H", "e-"]

# when you want to plot all species, use the following lines instead of the above two lines
#plot_species = chem.species
#plot_species_labels = plot_species

# check the species we want to plot and get their indices from FastChem
plot_species_indices = []
plot_species_symbols = []
from exogibbs.utils.nameparser import strip_trailing_one
for i, species in enumerate(plot_species):
    index = fastchem.getGasSpeciesIndex(species)
    
    # try Fe instead of Fe1, etc. if the species is not found in FastChem
    if index == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
        index = fastchem.getGasSpeciesIndex(strip_trailing_one(species))        
            
    if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
        plot_species_indices.append(index)
        plot_species_symbols.append(plot_species_labels[i])
    else:
        print("Species", species, "to plot not found in FastChem")


# convert the output into a numpy array
number_densities = np.array(output_data.number_densities)


# total gas particle number density from the ideal gas law
# used later to convert the number densities to mixing ratios
gas_number_density = pressure * 1e6 / (const.k_B.cgs * temperature)

# and plot...
N = len(plot_species_symbols)
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in np.linspace(0, 1, N)]


fig, (ax1, ax2) = plt.subplots(
    1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(8, 4)
)
crit = 1.0e-10
label_points = []
for i in range(0, N):
    vmr_fastchem = number_densities[:, plot_species_indices[i]] / gas_number_density
    if np.max(np.array(vmr_fastchem)) > crit:
        lab = plot_species_symbols[i]
        ax1.plot(vmr_fastchem, pressure, alpha=0.6, color=colors[i])

        idx_exogibbs = chem.species.index(plot_species[i])
        ax1.plot(nk_result[:, idx_exogibbs], pressure, "--", label=lab, color=colors[i])
        vmr_fastchem_top = float(np.asarray(vmr_fastchem[0]))
        nk_result_top = float(np.asarray(nk_result[0, idx_exogibbs]))
        label_points.append(
            (
                lab,
                nk_result_top,
                float(pressure[0] * 1.05),
                colors[i],
            )
        )
        label_points.append(
            (
                lab,
                vmr_fastchem_top,
                float(pressure[0] * 1.0),
                colors[i],
            )
        )
if False:
    for i in range(0, len(element_vector)):
        ax1.plot(nk_result[:, i], pressure, ".", label=lab, lw=2, color="black")
        label_points.append(
            (
                chem.elements[i],
                float(nk_result[0, i]),
                float(pressure[0] * 1.05),
                "black",
            )
        )

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlim(left=max(ax1.get_xlim()[0], 1.0e-30))

xmin, xmax = ax1.get_xlim()
for lab, xval, yval, color in label_points:
    xtext = min(max(xval, xmin * 1.05), xmax / 1.05)
    ax1.text(
        xtext,
        yval,
        lab,
        color=color,
        fontsize=8,
        ha="center",
        va="bottom",
        clip_on=True,
    )

ax1.set_xlabel("Mixing ratios")
ax1.set_ylabel("Pressure (bar)")
if N < 10:
    ax1.legend()
ax1.set_title("FastChem (solid) and ExoGibbs (dashed): T = " + str(T) + " K")
for i in range(0, N):
    vmr_fastchem = number_densities[:, plot_species_indices[i]] / gas_number_density
    if np.max(np.array(vmr_fastchem)) > crit:
        lab = plot_species_symbols[i]
        vmr_fastchem = number_densities[:, plot_species_indices[i]] / gas_number_density
        idx_exogibbs = chem.species.index(plot_species[i])
        deviation = 100 * (
            np.array(vmr_fastchem / nk_result[:, idx_exogibbs]) - 1.0
        )  # %
        if np.max(np.abs(deviation)) > 0.01:
            ax2.plot(deviation, pressure, color=colors[i], label=lab)

        else:
            ax2.plot(deviation, pressure, color=colors[i])
ax2.legend()
ax2.set_yscale("log")
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.set_xlabel("deviation (%)")


plt.savefig("comparison_fastchem_exogibbs_" + str(T) + ".png", dpi=300)
plt.show()
