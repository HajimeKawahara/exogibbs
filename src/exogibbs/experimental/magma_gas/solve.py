"""Compatibility solver for the former experimental MELTYQ interface."""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from exogibbs.magma_gas.models.meltyq.model import MeltyqMagmaGasModel
from exogibbs.magma_gas.models.meltyq.setup import PreparedMagmaGasChemistry
from exogibbs.magma_gas.models.meltyq.types import (
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    MagmaAtmosphereInterfaceState,
    MeltyqMagmaGasInputs,
)
from exogibbs.magma_gas.solve import solve
from exogibbs.magma_gas.types import (
    MagmaGasInit,
    MagmaGasOptions,
    MagmaGasProblem,
)


def solve_magma_atmosphere_interface(
    chemistry: PreparedMagmaGasChemistry,
    temperature_melt_k: ArrayLike,
    pressure_melt_bar: ArrayLike,
    oxygen_fugacity_bar: ArrayLike,
    co_melt_mole_ratio: ArrayLike,
    n_melt_mole_ratio: ArrayLike,
    *,
    init: Optional[MagmaAtmosphereInterfaceInit] = None,
    options: Optional[MagmaAtmosphereInterfaceOptions] = None,
) -> MagmaAtmosphereInterfaceState:
    """Solve through the stable engine while preserving the legacy result."""

    active_options = options or MagmaAtmosphereInterfaceOptions()
    if (
        not math.isfinite(active_options.h2_fraction_in_h_he)
        or not 0.0 < active_options.h2_fraction_in_h_he < 1.0
    ):
        raise ValueError("h2_fraction_in_h_he must be between zero and one.")

    generic_init = None
    if init is not None and init.log_element_ratios is not None:
        root_variables = jnp.asarray(init.log_element_ratios)
        if root_variables.shape != (4,):
            raise ValueError("init.log_element_ratios must have shape (4,).")
        generic_init = MagmaGasInit(root_variables=root_variables)

    result = solve(
        MagmaGasProblem(
            setup=chemistry.setup,
            model=MeltyqMagmaGasModel(),
            lnphi_func=chemistry.lnphi_func,
        ),
        temperature_melt_k,
        pressure_melt_bar,
        MeltyqMagmaGasInputs(
            oxygen_fugacity_bar=oxygen_fugacity_bar,
            co_melt_mole_ratio=co_melt_mole_ratio,
            n_melt_mole_ratio=n_melt_mole_ratio,
            h2_fraction_in_h_he=active_options.h2_fraction_in_h_he,
        ),
        init=generic_init,
        options=MagmaGasOptions(
            root_tolerance=active_options.root_tolerance,
            max_iter=active_options.max_iter,
            line_search_steps=active_options.line_search_steps,
            backtracking_factor=active_options.backtracking_factor,
            max_step=active_options.max_log_step,
            equilibrium_options=active_options.equilibrium_options,
        ),
    )
    gas = result.gas
    model_state = result.model_state
    return MagmaAtmosphereInterfaceState(
        element_abundances=result.element_abundances,
        gas_ln_n=gas.equilibrium.ln_n,
        gas_ntot=gas.equilibrium.ntot,
        gas_log_mole_fractions=gas.log_mole_fractions,
        gas_mole_fractions=jnp.exp(gas.log_mole_fractions),
        partial_pressures_bar=gas.partial_pressures_bar,
        fugacities_bar=gas.fugacities_bar,
        melt_volatile_mole_ratios=model_state.melt_volatile_mole_ratios,
        delta_iw=model_state.delta_iw,
        root_variables=result.root_variables,
        diagnostics=result.diagnostics,
    )


__all__ = ("solve_magma_atmosphere_interface",)
