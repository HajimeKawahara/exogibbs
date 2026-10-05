"""Caloric identities and full-network coverage for the opt-in gas model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.thermo.standard import prepare_fastchem_thermodynamics, _minimum_cp


@pytest.fixture(scope="module")
def thermo():
    return prepare_fastchem_thermodynamics()


@pytest.mark.parametrize("temperature", [1500.0, 2200.0, 3000.0, 4500.0])
def test_all_species_thermodynamic_identities(thermo, temperature):
    g = thermo.standard_gibbs_rt(temperature)
    entropy = thermo.standard_entropy_r(temperature)
    cp = thermo.standard_cp_r(temperature)
    dg = jax.jacfwd(thermo.standard_gibbs_rt)(temperature)
    ds = jax.jacfwd(thermo.standard_entropy_r)(temperature)
    assert g.shape == (538,)
    np.testing.assert_allclose(entropy, -g - temperature * dg, rtol=3e-14, atol=1e-12)
    np.testing.assert_allclose(cp, temperature * ds, rtol=3e-14, atol=1e-12)
    assert np.all(np.isfinite(g)) and np.all(cp >= 2.49)
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.sum(thermo.standard_entropy_r(t)))(temperature),
        jnp.sum(cp) / temperature,
        rtol=3e-14,
    )


def test_helium_and_electron_statistical_reference(thermo):
    # Independent Sackur--Tetrode standard entropy, including electron spin.
    k_b, planck, atomic_mass, pressure_pa = 1.380649e-23, 6.62607015e-34, 1.66053906660e-27, 1e5
    temperature = 2400.0
    for element, species, degeneracy in [("He", "He1", 1), ("e-", "e1-", 2)]:
        mass = thermo.element_masses_u[element] * atomic_mass
        expected = 2.5 + np.log(degeneracy * (2 * np.pi * mass * k_b * temperature / planck**2)**1.5 * k_b * temperature / pressure_pa)
        index = thermo.species.index(species)
        # NASA's older constants differ slightly from current SI constants.
        np.testing.assert_allclose(thermo.standard_entropy_r(temperature)[index], expected, rtol=2e-6)
        np.testing.assert_allclose(thermo.standard_cp_r(temperature)[index], 2.5, atol=1e-13)


def test_hydrogen_absolute_reference_tables(thermo):
    # Independent NIST-JANAF values at 1 bar, J/(mol K):
    # https://janaf.nist.gov/tables/H-001.html (atomic H)
    # https://janaf.nist.gov/tables/H-050.html (molecular H2)
    temperatures = np.array([1500.0, 3000.0, 4500.0])
    indices = [thermo.species.index(name) for name in ["H1", "H2"]]
    entropy = np.array([[148.299, 178.846], [162.706, 202.891], [171.135, 218.508]])
    cp = np.array([[20.786, 32.298], [20.786, 37.087], [20.786, 40.017]])
    gas_constant = 8.31446261815324
    # Reaction-fit derivatives are approximate source data: the H2 cp
    # discrepancy reaches 0.85%, despite exact model thermodynamic identities.
    np.testing.assert_allclose(thermo.standard_entropy_r(temperatures)[:, indices] * gas_constant, entropy, rtol=5e-4)
    np.testing.assert_allclose(thermo.standard_cp_r(temperatures)[:, indices] * gas_constant, cp, rtol=0.01)


def test_gauge_preserves_unmodified_reaction_thermochemistry(thermo):
    baseline = chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    temperature = jnp.array([1500.0, 2750.0, 4500.0])
    untouched = [name for name in thermo.species if name not in thermo.metadata["nasa_replacements"]]
    old_indices = [baseline.species.index(name) for name in untouched]
    new_indices = [thermo.species.index(name) for name in untouched]
    np.testing.assert_allclose(
        thermo.hvector_func(temperature)[:, new_indices],
        baseline.hvector_func(temperature)[:, old_indices], rtol=2e-14, atol=2e-13,
    )
    assert thermo.hvector_func is thermo.chemical_setup.hvector_func
    assert len(thermo.metadata["nasa_replacements"]) == 34
    assert set(thermo.metadata["excluded_species"]) == {"C4H6O4"}
    assert "C4H6O4" in baseline.species and "C4H6O4" not in thermo.species
    # Every restored gauge contribution is the same conserved elemental sum.
    atom_indices = [thermo.species.index("e1-" if e == "e-" else e + "1") for e in thermo.elements]
    g = thermo.standard_gibbs_rt(temperature)
    restored = g[:, atom_indices] @ thermo.chemical_setup.formula_matrix
    np.testing.assert_allclose(g - thermo.hvector_func(temperature), restored, rtol=1e-13, atol=1e-13)


def test_domain_and_preparation_fail_closed(thermo):
    for temperature in [1499.0, 4501.0, np.nan]:
        assert np.all(np.isnan(thermo.standard_entropy_r(temperature)))
    for bounds in [(300, 4500), (1500, 6000), (4500, 1500), (2000, 2000), (np.nan, 4000)]:
        with pytest.raises(ValueError, match="temperature_range"):
            prepare_fastchem_thermodynamics(temperature_range=bounds)


def test_preparation_snapshots_temperature_domain():
    bounds = np.array([1500.0, 4500.0])
    thermo = prepare_fastchem_thermodynamics(temperature_range=bounds)
    bounds[0] = 3500.0
    assert thermo.temperature_range == (1500.0, 4500.0)
    assert np.all(np.isfinite(thermo.standard_entropy_r(2000.0)))


def test_cp_minimum_finds_interior_failure():
    # cp = (T / 3000 - 1)**2 - 1/4: positive at both endpoints.
    coefficients = np.array([[0, 0, 0.75, -2 / 3000, 1 / 3000**2, 0, 0]])
    np.testing.assert_allclose(_minimum_cp(coefficients, (1200, 4800)), [-0.25], atol=1e-14)


def test_full_network_jit_vmap_and_finite_difference(thermo):
    temperatures = jnp.array([1735.0, 2517.0, 3973.0])
    batched = jax.jit(jax.vmap(thermo.standard_entropy_r))(temperatures)
    np.testing.assert_allclose(batched, thermo.standard_entropy_r(temperatures), rtol=1e-14)
    step = 0.02
    finite_difference = (thermo.standard_entropy_r(temperatures + step) - thermo.standard_entropy_r(temperatures - step)) / (2 * step)
    np.testing.assert_allclose(finite_difference, thermo.standard_cp_r(temperatures) / temperatures[:, None], rtol=2e-9, atol=2e-12)
