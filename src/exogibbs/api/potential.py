"""Thermpodynamic potential functions for ExoGibbs API."""

import jax.numpy as jnp
from jax import vmap
from typing import Optional

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.utils.constants import R_gas_constant_si
from jax.scipy.special import logsumexp

def gibbs_energy(
    temperature: float,
    pressure: float,
    chem_gas: ChemicalSetup,
    ln_ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ln_ncond: Optional[jnp.ndarray] = None,
    nomalize: bool = False,
):
    """Calculate Gibbs energy from chemical setup, temperature, and pressure.

    Args:
        temperature: float
            Temperature at which to evaluate Gibbs energy.
        pressure: float
            Pressure at which to evaluate Gibbs energy.
        chem_gas: ChemicalSetup
            The chemical setup for gas phase.
        ln_ngas: jnp.ndarray
            Logarithm of amounts of gas species (K_gas,).
        chem_cond: Optional[ChemicalSetup]
            The chemical setup for condensed phase.
        ln_ncond: Optional[jnp.ndarray]
            Logarithm of amounts of condensed species (K_cond,).
        nomalize: bool
            If True, return normalized Gibbs energy (G/RT).

    Returns:
        float
            Gibbs energy at given temperature and pressure.

    """
    if nomalize:
        RT = 1.0
    else:
        RT = R_gas_constant_si * temperature
    
    ln_ntot = logsumexp(ln_ngas)
    #hvector_gas = chem_gas.hvector_func(temperature) + jnp.log(pressure * ngas / ntot)
    hvector_gas = chem_gas.hvector_func(temperature) + jnp.log(pressure) + ln_ngas - ln_ntot
    g_gas = jnp.dot(jnp.exp(ln_ngas), hvector_gas) * RT

    if chem_cond is not None and ln_ncond is not None:
        hvector_cond = chem_cond.hvector_func(temperature)
        g_cond = jnp.dot(jnp.exp(ln_ncond), hvector_cond) * RT
        return g_gas + g_cond

    return g_gas


def gibbs_energies(
    temperatures: jnp.ndarray,
    pressures: jnp.ndarray,
    chem_gas: ChemicalSetup,
    ln_ngas: jnp.ndarray,
    chem_cond: Optional[ChemicalSetup] = None,
    ln_ncond: Optional[jnp.ndarray] = None,
    nomalize: bool = False,
    ):
    """Vectorized Gibbs energy calculation over temperature and pressure arrays.
    
    Args:
        temperatures: jnp.ndarray
            Array of temperatures at which to evaluate Gibbs energy.
        pressures: jnp.ndarray
            Array of pressures at which to evaluate Gibbs energy.
        chem_gas: ChemicalSetup
            The chemical setup for gas phase.
        ln_ngas: jnp.ndarray
            Logarithm of amounts of gas species (K_gas,).
        chem_cond: Optional[ChemicalSetup]
            The chemical setup for condensed phase.
        ln_ncond: Optional[jnp.ndarray]
            Logarithm of amounts of condensed species (K_cond,).
        nomalize: bool
            If True, return normalized Gibbs energy (G/RT).

    Returns:
        jnp.ndarray
            Array of Gibbs energies corresponding to input temperatures and pressures.
    
    """
    gibbs_energy_vmapped = vmap(
        gibbs_energy,
        in_axes=(0, 0, None, 0, None, 0, None),
    )   
    return gibbs_energy_vmapped(
        temperatures,
        pressures,
        chem_gas,
        ln_ngas,
        chem_cond,
        ln_ncond,
        nomalize,
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
    ln_ngas = jnp.log(jnp.ones(len(gas.species)))
    ln_ncond = jnp.log(jnp.ones(len(cond.species)))
    g = gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        chem_gas=gas,
        ln_ngas=ln_ngas,
        chem_cond=cond,
        ln_ncond=ln_ncond,
        nomalize=True,
    )
    print("Gibbs energy:", g)

    n=100
    
    
    temperatures = jnp.linspace(500.0, 3000.0, n)
    pressures = jnp.linspace(0.1, 10.0, n)
    
    ln_ngas = jnp.log(jnp.ones((n, len(gas.species))))
    ln_ncond = jnp.log(jnp.ones((n, len(cond.species))))
    
    gs = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=gas,
        ln_ngas=ln_ngas,
        chem_cond=cond,
        ln_ncond=ln_ncond,
        nomalize=True,
    )
    print("Gibbs energies:", gs)