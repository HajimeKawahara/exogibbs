import jax.numpy as jnp
from jax.scipy.special import logsumexp

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.potential import gibbs_energy, gibbs_energies


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
    ln_ngas = jnp.log(jnp.array([2.0, 3.0]))
    ln_ncond = jnp.log(jnp.array([5.0]))

    ln_ntot = logsumexp(ln_ngas)
    expected_gas = jnp.dot(jnp.exp(ln_ngas), jnp.array([1.0, 2.0]) + jnp.log(pressure) + ln_ngas - ln_ntot)
    expected_cond = jnp.dot(jnp.exp(ln_ncond), jnp.array([4.0]))
    expected = expected_gas + expected_cond

    out = gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        chem_gas=chem_gas,
        ln_ngas=ln_ngas,
        chem_cond=chem_cond,
        ln_ncond=ln_ncond,
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
    ln_ngas = jnp.log(jnp.array([[2.0, 3.0], [2.0, 3.0]]))

    expected = jnp.array(
        [
            gibbs_energy(
                temperature=temperatures[i],
                pressure=pressures[i],
                chem_gas=chem_gas,
                ln_ngas=ln_ngas[i],
                nomalize=False,
            )
            for i in range(2)
        ]
    )

    out = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=chem_gas,
        ln_ngas=ln_ngas,
        nomalize=False,
    )

    assert out.shape == (2,)
    assert jnp.allclose(out, expected)
