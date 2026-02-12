"""Thermodynamic potential functions for ExoGibbs API."""

import jax.numpy as jnp
from typing import Optional

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.utils.constants import R_gas_constant_si
from jax.scipy.special import logsumexp


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
    temperatures = jnp.asarray(temperatures)
    pressures = jnp.asarray(pressures)
    ln_ngas = jnp.asarray(ln_ngas)

    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("temperatures and pressures must be 1D arrays.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("temperatures and pressures must have the same length.")
    if ln_ngas.ndim != 2 or ln_ngas.shape[0] != temperatures.shape[0]:
        raise ValueError("ln_ngas must have shape (N, K_gas).")

    if nomalize:
        RT = jnp.ones_like(temperatures)
    else:
        RT = R_gas_constant_si * temperatures

    hvector_gases = compute_hvector_gases_at_tp(temperatures, pressures, chem_gas, ln_ngas)
    g_gas = jnp.sum(jnp.exp(ln_ngas) * hvector_gases, axis=1) * RT

    if chem_cond is None or ln_ncond is None:
        return g_gas

    ln_ncond = jnp.asarray(ln_ncond)
    if ln_ncond.ndim != 2 or ln_ncond.shape[0] != temperatures.shape[0]:
        raise ValueError("ln_ncond must have shape (N, K_cond).")

    hvector_cond = chem_cond.hvector_func(temperatures)
    g_cond = jnp.sum(jnp.exp(ln_ncond) * hvector_cond, axis=1) * RT
    return g_gas + g_cond

def compute_hvector_gases_at_tp(temperatures, pressures, chem_gas, ln_ngas):
    ln_ntot = logsumexp(ln_ngas, axis=1)
    hvector_gases = (
        chem_gas.hvector_func(temperatures)
        + jnp.log(pressures)[:, None]
        + ln_ngas
        - ln_ntot[:, None]
    )
    
    return hvector_gases


if __name__ == "__main__":

    from exogibbs.presets.fastchem_cond import chemsetup as condsetup
    from exogibbs.presets.fastchem import chemsetup as gassetup

    from jax import config

    config.update("jax_enable_x64", True)

    gas = gassetup()
    cond = condsetup()
    temperature = 1000.0
    pressure = 1.0
    ln_ngas = jnp.log(jnp.ones((1, len(gas.species))))
    ln_ncond = jnp.log(jnp.ones((1, len(cond.species))))
    g = gibbs_energies(
        temperatures=jnp.array([temperature]),
        pressures=jnp.array([pressure]),
        chem_gas=gas,
        ln_ngas=ln_ngas,
        chem_cond=cond,
        ln_ncond=ln_ncond,
        nomalize=True,
    )
    print("Gibbs energy:", g)

    n = 100

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
