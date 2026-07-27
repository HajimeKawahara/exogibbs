"""Compatibility facade for the gas-only equilibrium API."""

from exogibbs.equilibrium.gas.initialization import (
    DefaultEquilibriumInitializer,
    LearnedEquilibriumInitializer,
)
from exogibbs.equilibrium.gas.grid.initialization import (
    GridEquilibriumInitializer,
)
from exogibbs.equilibrium.gas.kernel.solver import (
    minimize_gibbs,
    minimize_gibbs_with_diagnostics,
)
from exogibbs.equilibrium.gas.profile import equilibrium_profile
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import (
    EquilibriumInit,
    EquilibriumInitializer,
    EquilibriumInitRequest,
    EquilibriumOptions,
    EquilibriumResult,
)

__all__ = (
    "DefaultEquilibriumInitializer",
    "EquilibriumInit",
    "EquilibriumInitRequest",
    "EquilibriumInitializer",
    "EquilibriumOptions",
    "EquilibriumResult",
    "GridEquilibriumInitializer",
    "LearnedEquilibriumInitializer",
    "equilibrium",
    "equilibrium_profile",
)
