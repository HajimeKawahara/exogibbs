"""Gas-only equilibrium implementation.

The ``solve`` and ``profile`` attributes intentionally refer to their child
modules.  User-facing function aliases live in :mod:`exogibbs.api.gas`.
"""

from exogibbs.equilibrium.gas import profile, solve
from exogibbs.equilibrium.gas.profile import equilibrium_profile, solve_profile
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import (
    EquilibriumInit,
    EquilibriumInitializer,
    EquilibriumInitRequest,
    EquilibriumOptions,
    EquilibriumResult,
    ThermoState,
)


__all__ = (
    "EquilibriumInit",
    "EquilibriumInitRequest",
    "EquilibriumInitializer",
    "EquilibriumOptions",
    "EquilibriumResult",
    "ThermoState",
    "equilibrium",
    "equilibrium_profile",
    "profile",
    "solve",
    "solve_profile",
)
