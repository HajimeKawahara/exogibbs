"""Thermpodynamic potential functions for ExoGibbs API."""

import jax.numpy as jnp
from typing import Optional
from exogibbs.utils.constants import R_gas_constant_si


def gibbs_energy(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    chem_gas: ChemicalSetup,
    ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ncond: Optional[jnp.ndarray] = None,
):
    """Calculate Gibbs energy from chemical setup, temperature, and pressure.

    Args:
        temperature: jnp.ndarray
            Temperature(s) at which to evaluate Gibbs energy.
        pressure: jnp.ndarray
            Pressure(s) at which to evaluate Gibbs energy.
        chem: ChemicalSetup
            The chemical setup containing formula matrix and enthalpy function.

    """
    RT = R_gas_constant_si * temperature
    ntot = jnp.sum(ngas)
    hvector_gas = chem_gas.hvector_func(temperature) + jnp.log(pressure * ngas / ntot)
    g_gas = jnp.dot(ngas, hvector_gas) * RT

    if chem_cond is not None and ncond is not None:
        hvector_cond = chem_cond.hvector_func(temperature)
        g_cond = jnp.dot(ncond, hvector_cond) * RT
        return g_gas + g_cond

    return g_gas
