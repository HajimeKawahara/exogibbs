from .chemistry import ChemicalSetup, ThermoState

__all__ = [
    "ChemicalSetup",
    "ThermoState",
    "get_default_equilibrium_grid_path",
    "build_equilibrium_grid",
    "build_h_he_element_vector_from_log10_z_over_z_sun",
    "compute_physical_log10_z_over_z_sun",
    "equilibrium_grid_from_dataset",
    "equilibrium_grid_to_dataset",
    "validate_equilibrium_grid_compatibility",
    "EquilibriumGrid",
    "EquilibriumGridMetadata",
    "EquilibriumGridOutputs",
    "EquilibriumGridInterpolationOptions",
    "EquilibriumGridInterpolationResult",
    "interpolate_equilibrium_grid",
    "load_equilibrium_grid_netcdf",
    "save_equilibrium_grid_netcdf",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumResult",
]


def __getattr__(name):
    if name == "get_default_equilibrium_grid_path":
        from exogibbs.io.load_data import get_default_equilibrium_grid_path

        return get_default_equilibrium_grid_path
    # Delay importing the equilibrium module until one of its public symbols is
    # requested. This avoids import cycles for modules that only need
    # `exogibbs.api.chemistry`.
    if name in {
        "build_equilibrium_grid",
        "build_h_he_element_vector_from_log10_z_over_z_sun",
        "compute_physical_log10_z_over_z_sun",
        "equilibrium_grid_from_dataset",
        "equilibrium_grid_to_dataset",
        "validate_equilibrium_grid_compatibility",
        "EquilibriumGrid",
        "EquilibriumGridMetadata",
        "EquilibriumGridOutputs",
        "EquilibriumGridInterpolationOptions",
        "EquilibriumGridInterpolationResult",
        "interpolate_equilibrium_grid",
        "load_equilibrium_grid_netcdf",
        "save_equilibrium_grid_netcdf",
    }:
        from .equilibrium_grid import (
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

        return {
            "build_equilibrium_grid": build_equilibrium_grid,
            "build_h_he_element_vector_from_log10_z_over_z_sun": build_h_he_element_vector_from_log10_z_over_z_sun,
            "compute_physical_log10_z_over_z_sun": compute_physical_log10_z_over_z_sun,
            "equilibrium_grid_from_dataset": equilibrium_grid_from_dataset,
            "equilibrium_grid_to_dataset": equilibrium_grid_to_dataset,
            "validate_equilibrium_grid_compatibility": validate_equilibrium_grid_compatibility,
            "EquilibriumGrid": EquilibriumGrid,
            "EquilibriumGridMetadata": EquilibriumGridMetadata,
            "EquilibriumGridOutputs": EquilibriumGridOutputs,
            "EquilibriumGridInterpolationOptions": EquilibriumGridInterpolationOptions,
            "EquilibriumGridInterpolationResult": EquilibriumGridInterpolationResult,
            "interpolate_equilibrium_grid": interpolate_equilibrium_grid,
            "load_equilibrium_grid_netcdf": load_equilibrium_grid_netcdf,
            "save_equilibrium_grid_netcdf": save_equilibrium_grid_netcdf,
        }[name]
    if name in {"EquilibriumOptions", "EquilibriumInit", "EquilibriumResult"}:
        from .equilibrium import EquilibriumInit, EquilibriumOptions, EquilibriumResult

        return {
            "EquilibriumOptions": EquilibriumOptions,
            "EquilibriumInit": EquilibriumInit,
            "EquilibriumResult": EquilibriumResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
