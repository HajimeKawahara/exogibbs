import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.potential import gibbs_energy, gibbs_energies
from exogibbs.utils.constants import R_gas_constant_si


def test_gibbs_energy_with_condensed_phase_normalized():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda T: jnp.array([1.0, 2.0]),
    )
    chem_cond = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 1)),
        hvector_func=lambda T: jnp.array([4.0]),
    )

    temperature = 1000.0
    pressure = 2.0
    ngas = jnp.array([2.0, 3.0])
    ncond = jnp.array([5.0])

    ntot = jnp.sum(ngas)
    expected_gas = jnp.dot(ngas, jnp.array([1.0, 2.0]) + jnp.log(pressure * ngas / ntot))
    expected_cond = jnp.dot(ncond, jnp.array([4.0]))
    expected = expected_gas + expected_cond

    out = gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        chem_gas=chem_gas,
        ngas=ngas,
        chem_cond=chem_cond,
        ncond=ncond,
        nomalize=True,
    )

    assert jnp.allclose(out, expected)


def test_gibbs_energies_vectorized_matches_scalar_non_normalized():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda T: jnp.array([1.0, 2.0]),
    )

    temperatures = jnp.array([1000.0, 1500.0])
    pressures = jnp.array([2.0, 2.0])
    ngas = jnp.array([2.0, 3.0])

    ntot = jnp.sum(ngas)
    base = jnp.dot(ngas, jnp.array([1.0, 2.0]) + jnp.log(pressures[0] * ngas / ntot))
    expected = jnp.array(
        [
            base * R_gas_constant_si * temperatures[0],
            base * R_gas_constant_si * temperatures[1],
        ]
    )

    out = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=chem_gas,
        ngas=ngas,
        nomalize=False,
    )

    assert out.shape == (2,)
    assert jnp.allclose(out, expected)
