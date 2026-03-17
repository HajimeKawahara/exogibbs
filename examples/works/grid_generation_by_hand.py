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
from exogibbs.presets.fastchem import chemsetup as fastchem_chemsetup

# ---- small test axes ----
temperature_axis = np.array([1000.0, 3000.0], dtype=float)
pressure_axis = np.array([1.0e-3, 1.0e-1], dtype=float)
log10_z_over_z_sun_axis = np.array([1.0, 1.0], dtype=float)

# ---- ExoGibbs solver options ----
# Change these by hand when you want tighter or looser ExoGibbs convergence.
equilibrium_options = EquilibriumOptions(
#epsilon_crit=1.0e-10, # HERE
#max_iter=1000, # HERE
)

outdir = Path("tmp_grid_check")
outdir.mkdir(exist_ok=True)

# ---- 1) ExoGibbs-backed grid with verification ----
print("Building ExoGibbs-backed grid...")
print(
    "Using ExoGibbs options:",
    f"epsilon_crit={equilibrium_options.epsilon_crit},",
    f"max_iter={equilibrium_options.max_iter}",
)
grid_exo = build_equilibrium_grid(
    preset_name="fastchem",
    temperature_axis=temperature_axis,
    pressure_axis=pressure_axis,
    log10_z_over_z_sun_axis=log10_z_over_z_sun_axis,
    source="exogibbs",
    options=equilibrium_options,
    verify_exogibbs_against_fastchem=True,
)

print("ExoGibbs grid built.")
print("ln_n shape:", grid_exo.outputs.ln_n.shape)
print("n shape:", grid_exo.outputs.n.shape)
print("x shape:", grid_exo.outputs.x.shape)
print("ntot shape:", grid_exo.outputs.ntot.shape)
print("metadata verification_passed:", grid_exo.metadata.verification_passed)
print("metadata verification_max_abs_percent_deviation:",
      grid_exo.metadata.verification_max_abs_percent_deviation)

# ---- 2) FastChem-backed grid ----
print("\nBuilding FastChem-backed grid...")
grid_fc = build_equilibrium_grid(
    preset_name="fastchem",
    temperature_axis=temperature_axis,
    pressure_axis=pressure_axis,
    log10_z_over_z_sun_axis=log10_z_over_z_sun_axis,
    source="fastchem",
)

print("FastChem grid built.")
print("ln_n shape:", grid_fc.outputs.ln_n.shape)
print("metadata source:", grid_fc.metadata.source)

# ---- 3) save/load roundtrip ----
path_exo = outdir / "grid_exogibbs.nc"
print(f"\nSaving ExoGibbs grid to {path_exo} ...")
save_equilibrium_grid_netcdf(grid_exo, path_exo)

print("Loading back...")
grid_exo_loaded = load_equilibrium_grid_netcdf(path_exo)

print("Loaded.")
print("loaded ln_n shape:", grid_exo_loaded.outputs.ln_n.shape)
print("loaded metadata source:", grid_exo_loaded.metadata.source)
print("loaded composition axis name:", grid_exo_loaded.metadata.composition_axis_name)

# ---- 4) compatibility check ----
print("\nChecking compatibility...")
setup = fastchem_chemsetup(silent=True)
validate_equilibrium_grid_compatibility(
    grid_exo_loaded,
    setup=setup,
    preset_name="fastchem",
)
print("Compatibility check passed.")

# ---- 5) grid initializer shell ----
print("\nChecking GridEquilibriumInitializer shell...")
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
try:
    initializer(request)
except NotImplementedError as exc:
    print("Initializer validation passed, lookup is still unimplemented:")
    print(" ", exc)

# ---- 6) simple numerical sanity checks ----
assert grid_exo.outputs.ln_n.shape == (2, 2, 2, len(grid_exo.metadata.preset_species))
assert grid_exo.outputs.ntot.shape == (2, 2, 2)
assert np.all(np.isfinite(grid_exo.outputs.ln_n))
assert np.all(np.isfinite(grid_exo.outputs.ntot))
assert np.all(grid_exo.outputs.ntot > 0.0)

# x should sum to ~1 along species axis
xsum = np.sum(grid_exo.outputs.x, axis=-1)
print("x sum min/max:", np.min(xsum), np.max(xsum))

print("\nSmoke test completed successfully.")
