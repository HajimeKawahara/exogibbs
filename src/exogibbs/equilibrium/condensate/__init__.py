"""Gas-plus-condensate equilibrium implementation.

The ``solve`` attribute intentionally refers to its child module.  User-facing
function aliases live in :mod:`exogibbs.api.condensate`.
"""

from exogibbs.equilibrium.condensate import solve
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
    condensate_temperature_validity_upper,
    validate_condensate_chemical_setup,
)


__all__ = (
    "CondensateChemicalSetup",
    "build_condensate_chemical_setup",
    "condensate_temperature_validity_upper",
    "solve",
    "validate_condensate_chemical_setup",
)
