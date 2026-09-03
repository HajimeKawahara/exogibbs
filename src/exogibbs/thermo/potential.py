"""Phase-aware thermodynamic potential functions."""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.constants import R_gas_constant_si


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
    gas_species_count = int(chem_gas.formula_matrix.shape[1])
    expected_gas_shape = (temperatures.shape[0], gas_species_count)
    if ln_ngas.shape != expected_gas_shape:
        raise ValueError(
            f"ln_ngas must have shape {expected_gas_shape}, got {ln_ngas.shape}."
        )

    if (chem_cond is None) != (ln_ncond is None):
        raise ValueError("chem_cond and ln_ncond must be provided together.")

    if nomalize:
        RT = jnp.ones_like(temperatures)
    else:
        RT = R_gas_constant_si * temperatures

    ln_ntot = logsumexp(ln_ngas, axis=1)
    ln_mole_fractions = ln_ngas - ln_ntot[:, None]
    ln_mole_fractions = jnp.where(
        jnp.isneginf(ln_ngas),
        0.0,
        ln_mole_fractions,
    )
    hvector_gas = (
        chem_gas.hvector_func(temperatures)
        + jnp.log(pressures)[:, None]
        + ln_mole_fractions
    )
    g_gas = jnp.sum(jnp.exp(ln_ngas) * hvector_gas, axis=1) * RT
    g_gas = jnp.where(jnp.isneginf(ln_ntot), jnp.nan, g_gas)

    if chem_cond is None or ln_ncond is None:
        return g_gas

    ln_ncond = jnp.asarray(ln_ncond)
    condensate_species_count = int(chem_cond.formula_matrix.shape[1])
    expected_condensate_shape = (
        temperatures.shape[0],
        condensate_species_count,
    )
    if ln_ncond.shape != expected_condensate_shape:
        raise ValueError(
            "ln_ncond must have shape "
            f"{expected_condensate_shape}, got {ln_ncond.shape}."
        )

    hvector_cond = chem_cond.hvector_func(temperatures)
    g_cond = jnp.sum(jnp.exp(ln_ncond) * hvector_cond, axis=1) * RT
    return g_gas + g_cond


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
