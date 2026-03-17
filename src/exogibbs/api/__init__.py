from .chemistry import ChemicalSetup, ThermoState

__all__ = [
    "ChemicalSetup",
    "ThermoState",
    "build_equilibrium_grid",
    "build_h_he_element_vector_from_log10_z_over_z_sun",
    "equilibrium_grid_from_dataset",
    "equilibrium_grid_to_dataset",
    "validate_equilibrium_grid_compatibility",
    "EquilibriumGrid",
    "EquilibriumGridMetadata",
    "EquilibriumGridOutputs",
    "load_equilibrium_grid_netcdf",
    "save_equilibrium_grid_netcdf",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumResult",
]


def __getattr__(name):
    # Delay importing the equilibrium module until one of its public symbols is
    # requested. This avoids import cycles for modules that only need
    # `exogibbs.api.chemistry`.
    if name in {
        "build_equilibrium_grid",
        "build_h_he_element_vector_from_log10_z_over_z_sun",
        "equilibrium_grid_from_dataset",
        "equilibrium_grid_to_dataset",
        "validate_equilibrium_grid_compatibility",
        "EquilibriumGrid",
        "EquilibriumGridMetadata",
        "EquilibriumGridOutputs",
        "load_equilibrium_grid_netcdf",
        "save_equilibrium_grid_netcdf",
    }:
        from .equilibrium_grid import (
            EquilibriumGrid,
            EquilibriumGridMetadata,
            EquilibriumGridOutputs,
            build_equilibrium_grid,
            build_h_he_element_vector_from_log10_z_over_z_sun,
            equilibrium_grid_from_dataset,
            equilibrium_grid_to_dataset,
            load_equilibrium_grid_netcdf,
            save_equilibrium_grid_netcdf,
            validate_equilibrium_grid_compatibility,
        )

        return {
            "build_equilibrium_grid": build_equilibrium_grid,
            "build_h_he_element_vector_from_log10_z_over_z_sun": build_h_he_element_vector_from_log10_z_over_z_sun,
            "equilibrium_grid_from_dataset": equilibrium_grid_from_dataset,
            "equilibrium_grid_to_dataset": equilibrium_grid_to_dataset,
            "validate_equilibrium_grid_compatibility": validate_equilibrium_grid_compatibility,
            "EquilibriumGrid": EquilibriumGrid,
            "EquilibriumGridMetadata": EquilibriumGridMetadata,
            "EquilibriumGridOutputs": EquilibriumGridOutputs,
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
