"""Canonical public API for gas-plus-condensate equilibrium."""

from exogibbs.equilibrium.condensate.initialization import (
    DefaultCondensateEquilibriumInitializer,
    FixedSupportCondensateEquilibriumGrid,
    GridCondensateEquilibriumInitializer,
)
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
    validate_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.solve import (
    condensate_equilibrium,
    condensate_equilibrium_profile,
)
from exogibbs.equilibrium.condensate.types import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
    HEAD_ROUTE_V2,
    CondensateEquilibriumInit,
    CondensateEquilibriumInitRequest,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumOptions,
    CondensateEquilibriumPoint,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
    CondensateFixedSupportV2Preset,
    CondensateProfileMethod,
)
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction


solve = condensate_equilibrium
solve_profile = condensate_equilibrium_profile


__all__ = (
    "CONDENSATE_HEAD_V2_ROUTE_NAME",
    "CONDENSATE_HEAD_V2_ROUTE_VERSION",
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "HEAD_ROUTE_V2",
    "CondensateChemicalSetup",
    "CondensateEquilibriumInit",
    "CondensateEquilibriumInitRequest",
    "CondensateEquilibriumInitializer",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumPoint",
    "CondensateEquilibriumProfileResult",
    "CondensateEquilibriumResult",
    "CondensateFixedSupportV2Preset",
    "CondensateProfileMethod",
    "DefaultCondensateEquilibriumInitializer",
    "FixedSupportCondensateEquilibriumGrid",
    "GridCondensateEquilibriumInitializer",
    "LogFugacityCoefficientFunction",
    "build_condensate_chemical_setup",
    "solve",
    "solve_profile",
    "validate_condensate_chemical_setup",
)
