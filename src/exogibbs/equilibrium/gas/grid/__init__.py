"""Gas-equilibrium grid models, interpolation, storage, and construction."""

from exogibbs.equilibrium.gas.grid.service import (
    EquilibriumGrid,
    EquilibriumGridInterpolationOptions,
    EquilibriumGridInterpolationResult,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
    build_equilibrium_grid,
    build_h_he_element_vector_from_log10_z_over_z_sun,
    compute_physical_log10_z_over_z_sun,
    equilibrium_grid_from_dataset,
    equilibrium_grid_to_dataset,
    interpolate_equilibrium_grid,
    load_equilibrium_grid_netcdf,
    save_equilibrium_grid_netcdf,
    validate_equilibrium_grid_compatibility,
)


__all__ = (
    "EquilibriumGrid",
    "EquilibriumGridInterpolationOptions",
    "EquilibriumGridInterpolationResult",
    "EquilibriumGridMetadata",
    "EquilibriumGridOutputs",
    "build_equilibrium_grid",
    "build_h_he_element_vector_from_log10_z_over_z_sun",
    "compute_physical_log10_z_over_z_sun",
    "equilibrium_grid_from_dataset",
    "equilibrium_grid_to_dataset",
    "interpolate_equilibrium_grid",
    "load_equilibrium_grid_netcdf",
    "save_equilibrium_grid_netcdf",
    "validate_equilibrium_grid_compatibility",
)
