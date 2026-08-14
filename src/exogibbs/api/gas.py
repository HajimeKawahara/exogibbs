"""Canonical public API for gas-only equilibrium."""

from exogibbs.equilibrium.gas.initialization import (
    DefaultEquilibriumInitializer,
    LearnedEquilibriumInitializer,
)
from exogibbs.equilibrium.gas.grid.initialization import (
    GridEquilibriumInitializer,
)
from exogibbs.equilibrium.gas.profile import equilibrium_profile
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import (
    EquilibriumInit,
    EquilibriumInitRequest,
    EquilibriumInitializer,
    EquilibriumOptions,
    EquilibriumResult,
)
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction


solve = equilibrium
solve_profile = equilibrium_profile


__all__ = (
    "DefaultEquilibriumInitializer",
    "EquilibriumInit",
    "EquilibriumInitRequest",
    "EquilibriumInitializer",
    "EquilibriumOptions",
    "EquilibriumResult",
    "GridEquilibriumInitializer",
    "LearnedEquilibriumInitializer",
    "LogFugacityCoefficientFunction",
    "solve",
    "solve_profile",
)
