"""Regression tests for the selective NASA Ni/Co ionization preset."""

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
import numpy as np
import pytest

from exogibbs.api.gas import solve
from exogibbs.presets import fastchem4, fastchem4_nasa9_ions


@pytest.fixture(scope="module")
def setup():
    return fastchem4_nasa9_ions.chemsetup(silent=True)


@pytest.mark.parametrize(
    "element, expected_log10_kp",
    [
        ("Ni", [-37.590976039921394, -10.81500378586127, -3.5665645066351765]),
        ("Co", [-38.369429678481914, -10.755751699191713, -3.354624542771229]),
    ],
)
def test_ionization_matches_independent_nasa_reference(setup, element, expected_log10_kp):
    h = setup.hvector_func(jnp.array([1000.0, 3000.0, 6000.0]))
    log_kp = (h[:, setup.species.index(element + "1")]
              - h[:, setup.species.index(element + "1+")]
              - h[:, setup.species.index("e1-")])
    np.testing.assert_allclose(log_kp / np.log(10.0), expected_log10_kp, atol=1e-12, rtol=0)


def test_only_two_chemical_potentials_change(setup):
    baseline = fastchem4.chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    assert setup.species == baseline.species
    assert setup.elements == baseline.elements
    assert len(setup.species) == 539
    assert "C4H6O4" in setup.species
    np.testing.assert_array_equal(setup.formula_matrix, baseline.formula_matrix)
    np.testing.assert_array_equal(setup.element_vector_reference, baseline.element_vector_reference)
    temperatures = jnp.array([1000.0, 3000.0, 6000.0])
    changed = np.any(np.asarray(setup.hvector_func(temperatures))
                     != np.asarray(baseline.hvector_func(temperatures)), axis=0)
    assert {name for name, different in zip(setup.species, changed) if different} == {"Ni1+", "Co1+"}
    assert setup.metadata["nasa_replacements"] == {"Ni1+": "Ni+", "Co1+": "Co+"}
    assert not {"Ni1+", "Co1+"} & set(setup.metadata["fastchem_logk_source_records"])
    assert set(setup.metadata["replaced_fastchem_logk_source_records"]) == {"Ni1+", "Co1+"}


def test_jit_batch_and_temperature_derivative(setup):
    temperatures = jnp.array([1735.0, 2517.0, 3973.0])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(setup.hvector_func))(temperatures),
        setup.hvector_func(temperatures), rtol=1e-14, atol=1e-13,
    )
    indices = jnp.array([setup.species.index(name) for name in ("Ni1+", "Co1+")])
    derivative = jax.jacfwd(setup.hvector_func)(3000.0)
    step = 0.05
    finite_difference = (setup.hvector_func(3000.0 + step)
                         - setup.hvector_func(3000.0 - step)) / (2 * step)
    np.testing.assert_allclose(derivative[indices], finite_difference[indices], rtol=1e-8)


@pytest.mark.parametrize("enabled", [False, True])
def test_integer_temperature_and_dtype(enabled):
    with enable_x64(enabled):
        setup = fastchem4_nasa9_ions.chemsetup(silent=True)
        dtype = np.float64 if enabled else np.float32
        temperatures = np.array([[1000, 3000], [4500, 6000]], dtype=np.int32)
        actual = setup.hvector_func(temperatures)
        assert actual.shape == (2, 2, 539)
        assert actual.dtype == dtype
        assert np.all(np.isfinite(actual))
        np.testing.assert_array_equal(actual, setup.hvector_func(temperatures.astype(dtype)))


def test_temperature_bounds_do_not_fall_back_to_fastchem(setup):
    h = np.asarray(setup.hvector_func(jnp.array([999.0, 1000.0, 6000.0, 6001.0, np.nan])))
    assert np.all(np.isnan(h[[0, 3, 4]]))
    assert np.all(np.isfinite(h[[1, 2]]))
    assert setup.metadata["temperature_range_K"] == (1000.0, 6000.0)


def test_missing_ions_are_rejected():
    with pytest.raises(ValueError, match="missing species: Ni1\\+, Co1\\+"):
        fastchem4_nasa9_ions.chemsetup(path="FastChem4/logK/logK_wo_ions.dat", silent=True)


@pytest.mark.parametrize("element, log10_kp", [("Ni", -10.81500378586127), ("Co", -10.755751699191713)])
def test_pure_metal_ionization_matches_analytic_charge_neutral_equilibrium(setup, element, log10_kp):
    # M + M+ + e- with one conserved metal atom and zero net charge.
    species = (element + "1", element + "1+", "e1-")
    indices = jnp.array([setup.species.index(name) for name in species])
    element_indices = jnp.array([setup.elements.index(element), setup.elements.index("e-")])
    reduced = replace(
        setup,
        species=species,
        elements=(element, "e-"),
        formula_matrix=setup.formula_matrix[element_indices[:, None], indices],
        hvector_func=lambda t: setup.hvector_func(t)[indices],
        element_vector_reference=jnp.array([1.0, 0.0]),
    )
    pressure_bar = 1e-9
    result = solve(reduced, T=3000.0, P=pressure_bar, b=reduced.element_vector_reference)
    # Kp = alpha^2 * (P / 1 bar) / (1 - alpha^2), including electron pressure.
    expected_alpha = np.sqrt(10**log10_kp / (pressure_bar + 10**log10_kp))
    actual_alpha = result.x[1] / (result.x[0] + result.x[1])
    assert float(actual_alpha) == pytest.approx(expected_alpha, rel=1e-9)
    np.testing.assert_allclose(result.x[1], result.x[2], rtol=1e-10)
