"""Thermpodynamic potential functions for ExoGibbs API."""

import jax.numpy as jnp
from jax import vmap
from typing import Optional
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.utils.constants import R_gas_constant_si


def gibbs_energy(
    temperature: float,
    pressure: float,
    chem_gas: ChemicalSetup,
    ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ncond: Optional[jnp.ndarray] = None,
):
    """Calculate Gibbs energy from chemical setup, temperature, and pressure.

    Args:
        temperature: float
            Temperature at which to evaluate Gibbs energy.
        pressure: float
            Pressure at which to evaluate Gibbs energy.
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


def gibbs_energies(
    temperatures: jnp.ndarray,
    pressures: jnp.ndarray,
    chem_gas: ChemicalSetup,
    ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ncond: Optional[jnp.ndarray] = None,
    ):
    """Vectorized Gibbs energy calculation over temperature and pressure arrays."""
    gibbs_energy_vmapped = vmap(
        gibbs_energy,
        in_axes=(0, 0, None, None, None, None),
    )   
    return gibbs_energy_vmapped(
        temperatures,
        pressures,
        chem_gas,
        ngas,
        chem_cond,
        ncond,
    )


if __name__ == "__main__":

    from exogibbs.presets.fastchem_cond import chemsetup as condsetup
    from exogibbs.presets.fastchem import chemsetup as gassetup

    from jax import config

    config.update("jax_enable_x64", True)

    gas = gassetup()
    cond = condsetup()
    temperature = 1000.0 
    pressure = 1.0 
    ngas = jnp.ones(len(gas.species))
    ncond = jnp.ones(len(cond.species))

    g = gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        chem_gas=gas,
        ngas=ngas,
        chem_cond=cond,
        ncond=ncond,
    )
    print("Gibbs energy:", g)

    n=100
    temperatures = jnp.linspace(500.0, 3000.0, n)
    pressures = jnp.linspace(0.1, 10.0, n)
    gs = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=gas,
        ngas=ngas,
        chem_cond=cond,
        ncond=ncond,
    )
    print("Gibbs energies:", gs)