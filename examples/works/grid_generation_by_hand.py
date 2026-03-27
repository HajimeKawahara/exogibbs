from pathlib import Path
import numpy as np
from jax import config

# Match the manual FastChem comparison workflow more closely.
# This must be set before importing JAX-based ExoGibbs modules.
config.update("jax_enable_x64", True)

from exogibbs.api import (
    build_equilibrium_grid,
    load_equilibrium_grid_netcdf,
    save_equilibrium_grid_netcdf,
    validate_equilibrium_grid_compatibility,
)
from exogibbs.api.equilibrium import (
    EquilibriumInitRequest,
    EquilibriumOptions,
    GridEquilibriumInitializer,
)
from exogibbs.api.equilibrium_grid import compute_physical_metal_mass_fraction
from exogibbs.presets.fastchem import chemsetup as fastchem_chemsetup

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "tmp_grid_check"


def main() -> None:
    # Small manual smoke-test axes.
    temperature_axis = np.array([1000.0, 2000.0, 3000.0], dtype=float)
    pressure_axis = np.array([1.0e-3, 1.0e-2, 1.0e-1], dtype=float)
    log10_z_over_z_sun_axis = np.array([-0.1, 0.1, 0.2], dtype=float)

    # Change these by hand when you want tighter or looser ExoGibbs convergence.
    equilibrium_options = EquilibriumOptions(
        # epsilon_crit=1.0e-10,
        # max_iter=1000,
    )

    OUTDIR.mkdir(exist_ok=True)
    path_exo = OUTDIR / "grid_exogibbs.nc"
    print("== Grid Generation Smoke Check ==")
    print("Output directory:", OUTDIR)
    print("Temperature axis [K]:", temperature_axis)
    print("Pressure axis [bar]:", pressure_axis)
    print("Composition axis log10(Z/Zsun):", log10_z_over_z_sun_axis)
    print(
        "Using ExoGibbs options:",
        f"epsilon_crit={equilibrium_options.epsilon_crit},",
        f"max_iter={equilibrium_options.max_iter}",
    )

    print("\n[1/5] Build ExoGibbs-backed grid with FastChem verification")
    grid_exo = build_equilibrium_grid(
        preset_name="fastchem",
        temperature_axis=temperature_axis,
        pressure_axis=pressure_axis,
        log10_z_over_z_sun_axis=log10_z_over_z_sun_axis,
        source="exogibbs",
        options=equilibrium_options,
        verify_exogibbs_against_fastchem=True,
    )
    print("ln_n shape:", grid_exo.outputs.ln_n.shape)
    print("ntot shape:", grid_exo.outputs.ntot.shape)
    print("verification_passed:", grid_exo.metadata.verification_passed)
    print(
        "verification_max_abs_percent_deviation:",
        grid_exo.metadata.verification_max_abs_percent_deviation,
    )

    print("\n[2/5] Build FastChem-backed comparison grid")
    grid_fc = build_equilibrium_grid(
        preset_name="fastchem",
        temperature_axis=temperature_axis,
        pressure_axis=pressure_axis,
        log10_z_over_z_sun_axis=log10_z_over_z_sun_axis,
        source="fastchem",
    )
    print("FastChem grid ln_n shape:", grid_fc.outputs.ln_n.shape)
    print("FastChem grid metadata source:", grid_fc.metadata.source)

    print(f"\n[3/5] Save and load ExoGibbs grid: {path_exo}")
    save_equilibrium_grid_netcdf(grid_exo, path_exo)
    grid_exo_loaded = load_equilibrium_grid_netcdf(path_exo)
    print("Loaded ln_n shape:", grid_exo_loaded.outputs.ln_n.shape)
    print("Loaded metadata source:", grid_exo_loaded.metadata.source)
    print("Loaded composition axis name:", grid_exo_loaded.metadata.composition_axis_name)

    print("\n[4/5] Validate loaded-grid compatibility")
    setup = fastchem_chemsetup(silent=True)
    b_ref = np.asarray(setup.element_vector_reference)
    Z_sun = float(compute_physical_metal_mass_fraction(setup, b_ref))
    print(Z_sun)
    validate_equilibrium_grid_compatibility(
        grid_exo_loaded,
        setup=setup,
        preset_name="fastchem",
    )
    print("Compatibility check passed.")
    print("\n[5/5] Check GridEquilibriumInitializer on a matching request")
    initializer = GridEquilibriumInitializer(
        grid=grid_exo_loaded,
        preset_name="fastchem",
    )
    request = EquilibriumInitRequest(
        setup=setup,
        T=float(temperature_axis[0]),
        P=float(pressure_axis[0]),
        b=np.asarray(setup.element_vector_reference),
        K=len(setup.species),
    )
    init = initializer(request)
    print("Initializer returned ln_nk shape:", np.asarray(init.ln_nk).shape)
    print("Initializer returned ln_ntot:", float(init.ln_ntot))

    assert grid_exo.outputs.ln_n.shape == (3, 3, 3, len(grid_exo.metadata.preset_species))
    assert grid_exo.outputs.ntot.shape == (3, 3, 3)
    assert np.all(np.isfinite(grid_exo.outputs.ln_n))
    assert np.all(np.isfinite(grid_exo.outputs.ntot))
    assert np.all(grid_exo.outputs.ntot > 0.0)

    xsum = np.sum(grid_exo.outputs.x, axis=-1)
    print("Species-fraction sum min/max:", np.min(xsum), np.max(xsum))
    print("\nGrid generation smoke check completed successfully.")


if __name__ == "__main__":
    main()
