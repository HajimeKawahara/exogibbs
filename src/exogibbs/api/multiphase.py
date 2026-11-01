"""Local multiphase equilibrium from full extensive Gibbs callbacks (SI units)."""

from exogibbs.equilibrium.multiphase.solve import solve, solve_fixed_support
from exogibbs.equilibrium.multiphase.linearize import (
    EquilibriumTangent, FixedSupportLinearization, linearize_fixed_support,
)
from exogibbs.equilibrium.multiphase.selection import (
    PurePhaseUpdateResult, audit_stability, solve_with_pure_phase_updates,
)
from exogibbs.equilibrium.multiphase.types import (
    GibbsFunction, Options, Phase, Problem, Result, Support,
)

__all__ = (
    "GibbsFunction", "Options", "Phase", "Problem", "Result", "Support",
    "solve", "solve_fixed_support", "audit_stability",
    "EquilibriumTangent", "FixedSupportLinearization", "linearize_fixed_support",
    "PurePhaseUpdateResult", "solve_with_pure_phase_updates",
)
