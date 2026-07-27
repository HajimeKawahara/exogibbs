"""Compatibility aggregation surface for gas-equilibrium grids."""

from exogibbs.equilibrium.gas.grid.build import (
    _create_fastchem_point_solver,
    _verify_exogibbs_grid_against_fastchem,
    build_equilibrium_grid,
    build_h_he_element_vector_from_log10_z_over_z_sun,
)
from exogibbs.equilibrium.gas.grid.interpolation import (
    Interpolator3D,
    compute_physical_log10_z_over_z_sun,
    compute_physical_metal_mass_fraction,
    compute_reference_physical_metal_mass_fraction,
    interpolate_equilibrium_grid,
    validate_equilibrium_grid_compatibility,
)
from exogibbs.equilibrium.gas.grid.storage import (
    equilibrium_grid_from_dataset,
    equilibrium_grid_to_dataset,
    load_equilibrium_grid_netcdf,
    save_equilibrium_grid_netcdf,
)
from exogibbs.equilibrium.gas.grid.types import (
    EquilibriumGrid,
    EquilibriumGridInterpolationOptions,
    EquilibriumGridInterpolationResult,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
    EquilibriumGridSource,
)

# These private names remain module attributes for historical diagnostic
# monkeypatches; they are intentionally excluded from ``__all__``.
_COMPATIBILITY_PRIVATE_EXPORTS = (
    _create_fastchem_point_solver,
    _verify_exogibbs_grid_against_fastchem,
    Interpolator3D,
)


__all__ = (
    "EquilibriumGrid",
    "EquilibriumGridInterpolationOptions",
    "EquilibriumGridInterpolationResult",
    "EquilibriumGridMetadata",
    "EquilibriumGridOutputs",
    "EquilibriumGridSource",
    "build_equilibrium_grid",
    "build_h_he_element_vector_from_log10_z_over_z_sun",
    "compute_physical_log10_z_over_z_sun",
    "compute_physical_metal_mass_fraction",
    "compute_reference_physical_metal_mass_fraction",
    "equilibrium_grid_from_dataset",
    "equilibrium_grid_to_dataset",
    "interpolate_equilibrium_grid",
    "load_equilibrium_grid_netcdf",
    "save_equilibrium_grid_netcdf",
    "validate_equilibrium_grid_compatibility",
)
