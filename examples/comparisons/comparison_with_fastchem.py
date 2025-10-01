# comparison_with_fastchem.py
# This script compares the chemical equilibrium calculations of FastChem and ExoGibbs (fastchem preset).
# It requires the FastChem Python bindings to be installed
# also ExoJAX is required to set solar abundances (you can cahnge if you want)
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
    "../input/element_abundances/asplund_2020.dat", "../input/logK/logK.dat", 1
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

chem = chemsetup()
solar_abundance = nsol()
nsol_vector = jnp.array(
    [solar_abundance[el] for el in chem.elements[:-1]]
)  # no solar abundance for e-
element_vector = jnp.append(nsol_vector, 0.0)
opts = EquilibriumOptions(epsilon_crit=1e-15, max_iter=1000)
res = equilibrium_profile(
    chem,
    temperature,
    pressure,
    element_vector,
    Pref=1.0,
    options=opts,
)
nk_result = res.x
##################################################################################

# plot_species = ["H2O1", "C1O2", "C1O1", "C1H4", "H3N1"]
# plot_species_labels = ["H2O", "CO2", "CO", "CH4", "NH3"]

plot_species = chem.species[29:]
plot_species_labels = plot_species
print(plot_species)

# check the species we want to plot and get their indices from FastChem
plot_species_indices = []
plot_species_symbols = []

for i, species in enumerate(plot_species):
    index = fastchem.getGasSpeciesIndex(species)

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
for i in range(0, N):
    vmr_fastchem = number_densities[:, plot_species_indices[i]] / gas_number_density
    if np.max(np.array(vmr_fastchem)) > crit:
        lab = plot_species_symbols[i]

        ax1.plot(vmr_fastchem, pressure, alpha=0.3, color=colors[i])

        idx_exogibbs = chem.species.index(plot_species[i])
        ax1.plot(nk_result[:, idx_exogibbs], pressure, "--", label=lab, color=colors[i])

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim(ax1.get_ylim()[::-1])

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
