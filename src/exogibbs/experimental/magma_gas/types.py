"""State and option types for the experimental magma--gas interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax

from exogibbs.equilibrium.gas.types import EquilibriumOptions


Array = jax.Array

MELTYQ_MELT_QUANTITIES = ("H2", "H2O", "CO", "CO2", "CH4", "N")
MELTYQ_ROOT_RESIDUALS = ("O2", "CO_melt", "N_melt", "H2_He")


@dataclass(frozen=True)
class MagmaAtmosphereInterfaceOptions:
    """Numerical options for the four-variable interface root."""

    root_tolerance: float = 1.0e-8
    max_iter: int = 30
    line_search_steps: int = 12
    backtracking_factor: float = 0.5
    max_log_step: float = 5.0
    h2_fraction_in_h_he: float = 0.84
    equilibrium_options: EquilibriumOptions = field(
        default_factory=lambda: EquilibriumOptions(
            epsilon_crit=1.0e-11,
            max_iter=1000,
        )
    )


@dataclass(frozen=True)
class MagmaAtmosphereInterfaceInit:
    """Optional initial logarithmic element ratios.

    ``log_element_ratios`` follows ``(C/H, O/H, N/H, He/H)`` order and
    contains natural logarithms.
    """

    log_element_ratios: Optional[Array] = None


class MagmaGasRootDiagnostics(NamedTuple):
    """JAX-compatible convergence information for both nested solves.

    ``converged`` requires both ``outer_converged`` and ``inner_converged``.
    Diagnostic fields are not part of the implicit differentiation contract.
    """

    converged: Array
    outer_converged: Array
    inner_converged: Array
    iterations: Array
    inner_iterations: Array
    residual: Array
    residual_norm: Array
    root_tolerance: Array
    inner_residual_norm: Array
    inner_tolerance: Array
    step_accepted: Array


class MagmaAtmosphereInterfaceState(NamedTuple):
    """Coupled gas composition and dilute melt volatile abundances.

    Element abundances follow ``(H, C, O, N, He)`` order.  Gas arrays follow
    the canonical species order of the prepared chemistry.
    ``gas_ln_n`` and ``gas_ntot`` use the interface's ``b_H = 1`` amount gauge;
    they are numerical equilibrium amounts, not a physical number density.
    ``melt_volatile_mole_ratios`` follows ``MELTYQ_MELT_QUANTITIES``.  These
    values mix native mole-fraction outputs for H2 and CH4 with MELTYQ's
    dilute 60 g/mol matrix conversion for H2O, CO, CO2, and N.  The vector is
    not renormalized.
    """

    element_abundances: Array
    gas_ln_n: Array
    gas_ntot: Array
    gas_log_mole_fractions: Array
    gas_mole_fractions: Array
    partial_pressures_bar: Array
    fugacities_bar: Array
    melt_volatile_mole_ratios: Array
    delta_iw: Array
    root_variables: Array
    diagnostics: MagmaGasRootDiagnostics


__all__ = (
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MagmaAtmosphereInterfaceInit",
    "MagmaAtmosphereInterfaceOptions",
    "MagmaAtmosphereInterfaceState",
    "MagmaGasRootDiagnostics",
)
