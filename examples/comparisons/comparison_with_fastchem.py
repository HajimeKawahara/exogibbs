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
Nlayer = 100
temperature = np.full(Nlayer, 1500)
pressure = np.logspace(-6, 1, num=Nlayer)


# define the directory for the output
# here, we currently use the standard one from FastChem
output_dir = "../output"

# the chemical species we want to plot later
# note that the standard FastChem input files use the Hill notation
plot_species = ["H2O1", "C1O2", "C1O1", "C1H4", "H3N1"]
# for the plot labels, we therefore use separate strings in the usual notation
plot_species_labels = ["H2O", "CO2", "CO", "CH4", "NH3"]

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
from exogibbs.presets.fastchem import chemsetup

# from exogibbs.presets.ykb4 import chemsetup

from exojax.utils.zsol import nsol
import jax.numpy as jnp

chem = chemsetup()
solar_abundance = nsol()
nsol_vector = jnp.array(
    [solar_abundance[el] for el in chem.elements[:-1]]
)  # no solar abundance for e-
element_vector = jnp.append(nsol_vector, 0.0)

idx_h2o = chem.species.index("H2O1")  # H2O
idx_co2 = chem.species.index("C1O2")  # CO2
idx_co = chem.species.index("C1O1")  # CO
idx_ch4 = chem.species.index("C1H4")  # CH4
idx_nh3 = chem.species.index("H3N1")  # NH3
idx_h2 = chem.species.index("H2")

Pref = 1.0  # bar, reference pressure
opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=1000)

res = equilibrium_profile(
    chem,
    temperature,
    pressure,
    element_vector,
    Pref=Pref,
    options=opts,
)
nk_result = res.x
print(nk_result)
vmr_h2o = nk_result[:, idx_h2o]
vmr_co2 = nk_result[:, idx_co2]
vmr_co = nk_result[:, idx_co]
vmr_ch4 = nk_result[:, idx_ch4]
vmr_nh3 = nk_result[:, idx_nh3]
vmr_h2 = nk_result[:, idx_h2]
##################################################################################

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
for i in range(0, len(plot_species_symbols)):
    fig = plt.plot(
        number_densities[:, plot_species_indices[i]] / gas_number_density,
        pressure,
        alpha=0.3,
    )

plt.plot(vmr_h2o, pressure, ls="dashed", color="C0")
plt.plot(vmr_co2, pressure, ls="dashed", color="C1")
plt.plot(vmr_co, pressure, ls="dashed", color="C2")
plt.plot(vmr_ch4, pressure, ls="dashed", color="C3")
plt.plot(vmr_nh3, pressure, ls="dashed", color="C4")

plt.xscale("log")
plt.yscale("log")
plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

plt.xlabel("Mixing ratios")
plt.ylabel("Pressure (bar)")
plt.legend(plot_species_symbols)
plt.title("Comparison of FastChem (solid) and ExoGibbs setting (dashed)")
plt.savefig("comparison_fastchem_exogibbs.png", dpi=300)
plt.show()
