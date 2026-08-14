"""Public ExoGibbs API namespaces and compatibility exports."""

from importlib import import_module
from typing import Final

from .chemistry import (
    ChemicalSetup,
    LogFugacityCoefficientFunction,
    ThermoState,
)


_MODULE_EXPORTS: Final = {
    "condensate": ".condensate",
    "condensate_equilibrium": ".condensate_equilibrium",
    "equilibrium": ".equilibrium",
    "equilibrium_grid": ".equilibrium_grid",
    "gas": ".gas",
}

_ATTRIBUTE_EXPORTS: Final = {
    "get_default_equilibrium_grid_path": (
        "exogibbs.io.load_data",
        "get_default_equilibrium_grid_path",
    ),
    "build_equilibrium_grid": (".equilibrium_grid", "build_equilibrium_grid"),
    "build_h_he_element_vector_from_log10_z_over_z_sun": (
        ".equilibrium_grid",
        "build_h_he_element_vector_from_log10_z_over_z_sun",
    ),
    "compute_physical_log10_z_over_z_sun": (
        ".equilibrium_grid",
        "compute_physical_log10_z_over_z_sun",
    ),
    "equilibrium_grid_from_dataset": (
        ".equilibrium_grid",
        "equilibrium_grid_from_dataset",
    ),
    "equilibrium_grid_to_dataset": (
        ".equilibrium_grid",
        "equilibrium_grid_to_dataset",
    ),
    "validate_equilibrium_grid_compatibility": (
        ".equilibrium_grid",
        "validate_equilibrium_grid_compatibility",
    ),
    "EquilibriumGrid": (".equilibrium_grid", "EquilibriumGrid"),
    "EquilibriumGridMetadata": (
        ".equilibrium_grid",
        "EquilibriumGridMetadata",
    ),
    "EquilibriumGridOutputs": (".equilibrium_grid", "EquilibriumGridOutputs"),
    "EquilibriumGridInterpolationOptions": (
        ".equilibrium_grid",
        "EquilibriumGridInterpolationOptions",
    ),
    "EquilibriumGridInterpolationResult": (
        ".equilibrium_grid",
        "EquilibriumGridInterpolationResult",
    ),
    "interpolate_equilibrium_grid": (
        ".equilibrium_grid",
        "interpolate_equilibrium_grid",
    ),
    "load_equilibrium_grid_netcdf": (
        ".equilibrium_grid",
        "load_equilibrium_grid_netcdf",
    ),
    "save_equilibrium_grid_netcdf": (
        ".equilibrium_grid",
        "save_equilibrium_grid_netcdf",
    ),
    "EquilibriumOptions": (".equilibrium", "EquilibriumOptions"),
    "EquilibriumInit": (".equilibrium", "EquilibriumInit"),
    "EquilibriumResult": (".equilibrium", "EquilibriumResult"),
    "CondensateChemicalSetup": (
        ".condensate_equilibrium",
        "CondensateChemicalSetup",
    ),
    "CondensateEquilibriumOptions": (
        ".condensate_equilibrium",
        "CondensateEquilibriumOptions",
    ),
    "CondensateEquilibriumResult": (
        ".condensate_equilibrium",
        "CondensateEquilibriumResult",
    ),
    "CondensateFixedSupportV2Preset": (
        ".condensate_equilibrium",
        "CondensateFixedSupportV2Preset",
    ),
    "FIXED_SUPPORT_V2_VALIDATED_PRESET": (
        ".condensate_equilibrium",
        "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    ),
    "HEAD_ROUTE_V2": (".condensate_equilibrium", "HEAD_ROUTE_V2"),
    "build_condensate_chemical_setup": (
        ".condensate_equilibrium",
        "build_condensate_chemical_setup",
    ),
    "condensate_equilibrium_profile": (
        ".condensate_equilibrium",
        "condensate_equilibrium_profile",
    ),
    "validate_condensate_chemical_setup": (
        ".condensate_equilibrium",
        "validate_condensate_chemical_setup",
    ),
}


__all__ = [
    "ChemicalSetup",
    "LogFugacityCoefficientFunction",
    "ThermoState",
    "condensate",
    "condensate_equilibrium",
    "equilibrium",
    "equilibrium_grid",
    "gas",
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
    "CondensateChemicalSetup",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumResult",
    "CondensateFixedSupportV2Preset",
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "HEAD_ROUTE_V2",
    "build_condensate_chemical_setup",
    "condensate_equilibrium_profile",
    "validate_condensate_chemical_setup",
]


def __getattr__(name):
    """Resolve public modules and compatibility symbols lazily."""

    module_name = _MODULE_EXPORTS.get(name)
    if module_name is not None:
        return import_module(module_name, __name__)

    export = _ATTRIBUTE_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    if module_name.startswith("."):
        module = import_module(module_name, __name__)
    else:
        module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    """Return deterministic names without importing lazy implementations."""

    return sorted(set(globals()) | set(__all__))
