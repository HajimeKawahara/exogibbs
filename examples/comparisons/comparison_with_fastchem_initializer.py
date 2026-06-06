"""Compare FastChem and ExoGibbs using a saved equilibrium-grid initializer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyfastchem
from astropy import constants as const
from exojax.utils.zsol import nsol
from jax import config
import jax.numpy as jnp

from exogibbs.api import get_default_equilibrium_grid_path
from exogibbs.api import load_equilibrium_grid_netcdf
from exogibbs.api.equilibrium import (
    EquilibriumOptions,
    GridEquilibriumInitializer,
    equilibrium_profile,
)
from exogibbs.api.equilibrium_grid import compute_physical_log10_z_over_z_sun
from exogibbs.presets.fastchem import chemsetup

config.update("jax_enable_x64", True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
FASTCHEM_INPUT_DIR = REPO_DIR / "ref" / "FastChem" / "input"
FASTCHEM_ELEMENT_FILE = FASTCHEM_INPUT_DIR / "element_abundances" / "asplund_2020.dat"
FASTCHEM_LOGK_FILE = FASTCHEM_INPUT_DIR / "logK" / "logK.dat"


def main() -> None:
    grid_path = get_default_equilibrium_grid_path("fastchem")
    T = 2870.0
    nlayer = 100
    temperature = np.full(nlayer, T)
    pressure = np.logspace(-8, 2, num=nlayer)

    print("== FastChem / ExoGibbs Grid-Initializer Comparison ==")
    print("Loaded grid file:", grid_path)
    print("Using initializer: GridEquilibriumInitializer")
    print("FastChem element file:", FASTCHEM_ELEMENT_FILE)
    print("FastChem logK file:", FASTCHEM_LOGK_FILE)

    if not FASTCHEM_ELEMENT_FILE.exists():
        raise FileNotFoundError(f"FastChem element abundance file not found at {FASTCHEM_ELEMENT_FILE}.")
    if not FASTCHEM_LOGK_FILE.exists():
        raise FileNotFoundError(f"FastChem logK file not found at {FASTCHEM_LOGK_FILE}.")

    fastchem = pyfastchem.FastChem(
        str(FASTCHEM_ELEMENT_FILE),
        str(FASTCHEM_LOGK_FILE),
        1,
    )
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()
    input_data.temperature = temperature
    input_data.pressure = pressure
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    print("FastChem reports:")
    print("  -", pyfastchem.FASTCHEM_MSG[fastchem_flag])

    chem = chemsetup(silent=True)
    solar_abundance = nsol()
    nsol_vector = jnp.array([solar_abundance[el] for el in chem.elements[:-1]])
    element_vector = jnp.append(nsol_vector, 0.0)
    log10_z_over_z_sun = float(
        compute_physical_log10_z_over_z_sun(chem, np.asarray(element_vector))
    )
    #method = "scan_hot_from_top"
    #method = "scan_hot_from_bottom"
    method = "vmap_cold"

    opts = EquilibriumOptions(epsilon_crit=1e-10, max_iter=1000, method=method)

    grid = load_equilibrium_grid_netcdf(str(grid_path))
    initializer = GridEquilibriumInitializer(grid=grid, preset_name="fastchem")

    print("Abundance setup: solar abundances from exojax.utils.zsol.nsol()")
    print("Test temperature [K]:", T)
    print(
        "Test pressure range [bar]:",
        f"{pressure[0]:.3e} -> {pressure[-1]:.3e}",
    )
    print("Inferred physical log10(Z/Zsun):", f"{log10_z_over_z_sun:.6f}")

    res = equilibrium_profile(
        chem,
        temperature,
        pressure,
        element_vector,
        Pref=1.0,
        initializer=initializer,
        options=opts,
    )
    nk_result = np.asarray(res.x)

    print("ExoGibbs solve completed with grid-backed initializer.")
    print("Result shape:", nk_result.shape)

    n_elem = len(chem.elements)
    if len(element_vector) != n_elem:
        raise AssertionError("comparison_with_fastchem_initializer: len(element_vector) must equal len(chem.elements)")
    if list(chem.elements)[-1] != "e-":
        raise AssertionError("comparison_with_fastchem_initializer: ExoGibbs chem.elements must keep e- last")
    if list(chem.species[:n_elem]) != list(chem.elements):
        raise AssertionError(
            "comparison_with_fastchem_initializer: molecule species slice must start at len(chem.elements)"
        )
    plot_species = chem.species[n_elem:]
    plot_species_labels = plot_species

    plot_species_indices = []
    plot_species_symbols = []
    for i, species in enumerate(plot_species):
        index = fastchem.getGasSpeciesIndex(species)
        if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            plot_species_indices.append(index)
            plot_species_symbols.append(plot_species_labels[i])
        else:
            print("Species", species, "to plot not found in FastChem")

    number_densities = np.array(output_data.number_densities)
    gas_number_density = pressure * 1e6 / (const.k_B.cgs * temperature)

    n_plot = len(plot_species_symbols)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in np.linspace(0, 1, n_plot)]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [4, 1]},
        figsize=(8, 4),
    )
    crit = 1.0e-10
    max_abs_deviation = 0.0
    for i in range(n_plot):
        vmr_fastchem = number_densities[:, plot_species_indices[i]] / gas_number_density
        if np.max(np.array(vmr_fastchem)) > crit:
            label = plot_species_symbols[i]
            ax1.plot(vmr_fastchem, pressure, alpha=0.3, color=colors[i])

            idx_exogibbs = chem.species.index(plot_species[i])
            ax1.plot(nk_result[:, idx_exogibbs], pressure, "--", label=label, color=colors[i])

            deviation = 100.0 * (np.array(vmr_fastchem / nk_result[:, idx_exogibbs]) - 1.0)
            max_abs_deviation = max(max_abs_deviation, float(np.nanmax(np.abs(deviation))))
            if np.max(np.abs(deviation)) > 0.01:
                ax2.plot(deviation, pressure, color=colors[i], label=label)
            else:
                ax2.plot(deviation, pressure, color=colors[i])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xlabel("Mixing ratios")
    ax1.set_ylabel("Pressure (bar)")
    if n_plot < 10:
        ax1.legend()
    ax1.set_title(f"FastChem (solid) and ExoGibbs (dashed): T = {T:g} K")

    ax2.legend()
    ax2.set_yscale("log")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xlabel("deviation (%)")

    output_path = SCRIPT_DIR / f"comparison_fastchem_exogibbs_initializer_{int(T)}.png"
    plt.savefig(output_path, dpi=300)

    print("Comparison outputs:")
    print("  - Saved figure:", output_path)
    print("  - Maximum absolute deviation [%]:", f"{max_abs_deviation:.6f}")

    plt.show()


if __name__ == "__main__":
    main()
