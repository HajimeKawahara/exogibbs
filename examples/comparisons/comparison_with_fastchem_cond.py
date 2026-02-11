# comparison_with_fastchem_cond.py
# This script compares the chemical equilibrium calculations of FastChem and ExoGibbs (fastchem_cond preset).
# It requires the FastChem Python bindings to be installed
# also ExoJAX is required to set solar abundances (you can cahnge if you want)
#
# fastchem original file: python/fastchem_cond.py
#
# soft link this file into the fastchem/input directory and run it there:
#
# cd fastchem/input
# python comparison_with_fastchem_cond.py
#
import pyfastchem
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions


from jax import config

config.update("jax_enable_x64", True)


# we read in a p-T structure for a brown dwarf
prof = True
if prof:
    data = np.loadtxt("../input/example_p_t_structures/Brown_dwarf_Sonora.dat")
    tag = "_prof"
else:
    #data = np.array([[1.0, 700.0]]) # 1 bar, 700 K
    data = np.array([[1.0, 200.0]]) # 1 bar, 700 K
    tag = ""
# and extract temperature and pressure values
temperature = data[:, 1]
pressure = data[:, 0]
print("T=",temperature, "P=",pressure)

# define the directory for the output
# here, we currently use the standard one from FastChem
output_dir = "../output"


# the chemical species we want to plot later
# note that the standard FastChem input files use the Hill notation
plot_species = ["H2O1", "C1O2", "C1O1", "C1H4", "H3N1", "Fe1S1", "H2S1"]
# for the plot labels, we therefore use separate strings in the usual notation
plot_species_labels = ["H2O", "CO2", "CO", "CH4", "NH3", "FeS", "H2S"]

# the default condensate data doesn't use the Hill notation
plot_species_cond = ["Fe(s,l)", "FeS(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)"]


# create a FastChem object
fastchem = pyfastchem.FastChem(
    "../input/element_abundances/asplund_2009.dat",
    "../input/logK/logK.dat",
    "../input/logK/logK_condensates.dat",
    1,
)


# create the input and output structures for FastChem
input_data = pyfastchem.FastChemInput()
output_data = pyfastchem.FastChemOutput()

# use equilibrium condensation
input_data.equilibrium_condensation = True


input_data.temperature = temperature
input_data.pressure = pressure

# run FastChem on the entire p-T structure
fastchem_flag = fastchem.calcDensities(input_data, output_data)

print("FastChem reports:")
print("  -", pyfastchem.FASTCHEM_MSG[fastchem_flag])

if fastchem_flag != pyfastchem.FASTCHEM_SUCCESS:
    raise RuntimeError("FastChem calculation did not complete successfully. maybe try in fastchem/python/ directory?")

if np.amin(output_data.element_conserved[:]) == 1:
    print("  - element conservation: ok")
else:
    print("  - element conservation: fail")

# exogibbs
from exogibbs.presets.fastchem_cond import chemsetup as condsetup
cond = condsetup()
from exogibbs.presets.fastchem import chemsetup as gassetup
gas = gassetup()


plot_species = gas.species[29:]
plot_species_labels = plot_species

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

N = len(plot_species_symbols)
print(N)
if prof:
    vmr_fastchem = np.array(number_densities[:, plot_species_indices] / gas_number_density[:, np.newaxis])
else:
    vmr_fastchem = np.array(number_densities[:, plot_species_indices] / gas_number_density)[0]      

print(np.sum(number_densities, axis=1)/gas_number_density)

np.savez("vmr_fastchem"+tag+".npz", vmr_fastchem=vmr_fastchem, temperature=temperature, pressure=pressure)
