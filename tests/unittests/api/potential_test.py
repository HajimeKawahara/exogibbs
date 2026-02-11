import jax.numpy as jnp
from jax.scipy.special import logsumexp

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.potential import gibbs_energies
from exogibbs.utils.constants import R_gas_constant_si


def test_gibbs_energies_with_condensed_phase_normalized():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda T: jnp.array([1.0, 2.0]),
    )
    chem_cond = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 1)),
        hvector_func=lambda T: jnp.array([4.0]),
    )

    temperatures = jnp.array([1000.0])
    pressures = jnp.array([2.0])
    ln_ngas = jnp.log(jnp.array([[2.0, 3.0]]))
    ln_ncond = jnp.log(jnp.array([[5.0]]))

    ln_ntot = logsumexp(ln_ngas[0])
    expected_gas = jnp.dot(
        jnp.exp(ln_ngas[0]),
        jnp.array([1.0, 2.0]) + jnp.log(pressures[0]) + ln_ngas[0] - ln_ntot,
    )
    expected_cond = jnp.dot(jnp.exp(ln_ncond[0]), jnp.array([4.0]))
    expected = jnp.array([expected_gas + expected_cond])

    out = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=chem_gas,
        ln_ngas=ln_ngas,
        chem_cond=chem_cond,
        ln_ncond=ln_ncond,
        nomalize=True,
    )

    assert jnp.allclose(out, expected)


def test_gibbs_energies_vectorized_non_normalized():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda T: jnp.array([1.0, 2.0]),
    )

    temperatures = jnp.array([1000.0, 1500.0])
    pressures = jnp.array([2.0, 2.0])
    ln_ngas = jnp.log(jnp.array([[2.0, 3.0], [2.0, 3.0]]))

    rt = R_gas_constant_si * temperatures
    expected = []
    for i in range(2):
        ln_ntot = logsumexp(ln_ngas[i])
        hvector_gas = jnp.array([1.0, 2.0]) + jnp.log(pressures[i]) + ln_ngas[i] - ln_ntot
        expected.append(jnp.dot(jnp.exp(ln_ngas[i]), hvector_gas) * rt[i])
    expected = jnp.array(expected)

    out = gibbs_energies(
        temperatures=temperatures,
        pressures=pressures,
        chem_gas=chem_gas,
        ln_ngas=ln_ngas,
        nomalize=False,
    )

    assert out.shape == (2,)
    assert jnp.allclose(out, expected)
