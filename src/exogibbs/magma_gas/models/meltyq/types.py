"""MELTYQ model inputs, outputs, and compatibility types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax

from exogibbs.equilibrium.gas.types import EquilibriumOptions
from exogibbs.magma_gas.types import MagmaGasDiagnostics


Array = jax.Array

MELTYQ_MELT_QUANTITIES = ("H2", "H2O", "CO", "CO2", "CH4", "N")
MELTYQ_ROOT_RESIDUALS = ("O2", "CO_melt", "N_melt", "H2_He")


class MeltyqMagmaGasInputs(NamedTuple):
    """Dynamic MELTYQ boundary constraints."""

    oxygen_fugacity_bar: Array
    co_melt_mole_ratio: Array
    n_melt_mole_ratio: Array
    h2_fraction_in_h_he: Array = 0.84


class MeltyqMagmaGasState(NamedTuple):
    """MELTYQ-specific melt quantities evaluated at the coupled root."""

    log_melt_volatile_mole_ratios: Array
    melt_volatile_mole_ratios: Array
    delta_iw: Array


@dataclass(frozen=True)
class MagmaAtmosphereInterfaceOptions:
    """Legacy numerical and MELTYQ settings for compatibility."""

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
    """Legacy initial logarithmic ``(C/H, O/H, N/H, He/H)`` ratios."""

    log_element_ratios: Optional[Array] = None


MagmaGasRootDiagnostics = MagmaGasDiagnostics


class MagmaAtmosphereInterfaceState(NamedTuple):
    """Legacy flat MELTYQ magma--gas state."""

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
    "MeltyqMagmaGasInputs",
    "MeltyqMagmaGasState",
)
