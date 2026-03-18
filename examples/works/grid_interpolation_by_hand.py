from pathlib import Path
import numpy as np
from jax import config

# Keep the example aligned with the manual grid-generation workflow.
config.update("jax_enable_x64", True)

from exogibbs.api import load_equilibrium_grid_netcdf


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    grid_path = SCRIPT_DIR / "tmp_grid_check" / "grid_exogibbs.nc"

    temperature = 2500.0
    pressure = 3.0 * 10.0 ** (-2.5)
    log10_z_over_z_sun = 0.4

    print("== Grid Interpolation Smoke Check ==")
    print("Loading grid from:", grid_path)
    print("Query temperature [K]:", temperature)
    print("Query pressure [bar]:", pressure)
    print("Query log10(Z/Zsun):", log10_z_over_z_sun)

    grid = load_equilibrium_grid_netcdf(grid_path)
    result = grid.interpolate(
        temperature=temperature,
        pressure=pressure,
        log10_z_over_z_sun=log10_z_over_z_sun,
    )

    species = grid.metadata.preset_species
    if species is None:
        raise ValueError("The loaded grid does not store species metadata.")

    print("\nInterpolated state summary")
    print("Stored species count:", len(species))
    print("ntot:", float(result.ntot))
    print("ln_ntot:", float(result.ln_ntot))
    print("sum(x):", float(np.sum(np.asarray(result.x))))

    print("\nFirst 10 species entries in stored grid order")
    for name, ln_n_i, x_i in zip(species[:10], np.asarray(result.ln_n)[:10], np.asarray(result.x)[:10]):
        print(f"{name:>12s}  ln_n={ln_n_i: .6e}  x={x_i: .6e}")

    print("\nGrid interpolation smoke check completed successfully.")


if __name__ == "__main__":
    main()
