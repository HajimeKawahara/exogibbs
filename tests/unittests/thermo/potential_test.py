import jax.numpy as jnp
import pytest
from jax.scipy.special import logsumexp

from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.potential import gibbs_energies
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


def test_gibbs_energies_handles_zero_species_amount():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda temperature: jnp.asarray([1.0, 2.0]),
    )

    result = gibbs_energies(
        temperatures=jnp.asarray([1000.0]),
        pressures=jnp.asarray([2.0]),
        chem_gas=chem_gas,
        ln_ngas=jnp.asarray([[0.0, -jnp.inf]]),
        nomalize=True,
    )

    assert jnp.isfinite(result[0])
    assert result[0] == pytest.approx(1.0 + jnp.log(2.0), abs=1.0e-12)


def test_gibbs_energies_remains_finite_for_large_log_amounts():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda temperature: jnp.zeros(2),
    )
    ln_ngas = jnp.asarray([[84.0, 84.0]])

    result = gibbs_energies(
        temperatures=jnp.asarray([1000.0]),
        pressures=jnp.asarray([1.0]),
        chem_gas=chem_gas,
        ln_ngas=ln_ngas,
        nomalize=True,
    )
    expected = jnp.sum(
        jnp.exp(ln_ngas) * (ln_ngas - logsumexp(ln_ngas, axis=1)[:, None]),
        axis=1,
    )

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_gibbs_energies_rejects_an_all_zero_gas_state_with_nan():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda temperature: jnp.asarray([1.0, 2.0]),
    )

    result = gibbs_energies(
        temperatures=jnp.asarray([1000.0]),
        pressures=jnp.asarray([1.0]),
        chem_gas=chem_gas,
        ln_ngas=jnp.asarray([[-jnp.inf, -jnp.inf]]),
        nomalize=True,
    )

    assert jnp.isnan(result[0])


def test_gibbs_energies_rejects_wrong_species_width():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 2)),
        hvector_func=lambda temperature: jnp.asarray([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="ln_ngas must have shape"):
        gibbs_energies(
            temperatures=jnp.asarray([1000.0]),
            pressures=jnp.asarray([2.0]),
            chem_gas=chem_gas,
            ln_ngas=jnp.zeros((1, 1)),
        )


def test_gibbs_energies_requires_both_condensate_arguments():
    chem_gas = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 1)),
        hvector_func=lambda temperature: jnp.asarray([1.0]),
    )
    chem_cond = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 1)),
        hvector_func=lambda temperature: jnp.asarray([2.0]),
    )
    common = {
        "temperatures": jnp.asarray([1000.0]),
        "pressures": jnp.asarray([2.0]),
        "chem_gas": chem_gas,
        "ln_ngas": jnp.zeros((1, 1)),
    }

    with pytest.raises(ValueError, match="must be provided together"):
        gibbs_energies(**common, chem_cond=chem_cond)
    with pytest.raises(ValueError, match="must be provided together"):
        gibbs_energies(**common, ln_ncond=jnp.zeros((1, 1)))
