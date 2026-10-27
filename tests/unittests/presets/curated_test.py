"""Offline source, temperature, network and equilibrium checks for curated gas."""

from collections import Counter
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.gas import EquilibriumOptions, solve, solve_profile
from exogibbs.presets import curated, fastchem4
from exogibbs.thermo.standard import _nasa9


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def setup():
    return curated.chemsetup()


@pytest.fixture(scope="module")
def neutral():
    return curated.chemsetup(ions=False)


@pytest.fixture(scope="module")
def data():
    return json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())


def test_catalog_and_solar_budget_match_approved_study(setup, neutral):
    with np.load(ROOT / "tests/data/validation/gas_thermochemistry_impact/equilibria.npz") as archive:
        assert setup.species == tuple(archive["species"])
        assert setup.elements == tuple(archive["elements"])
        np.testing.assert_array_equal(setup.formula_matrix, archive["formula_matrix"])
        np.testing.assert_allclose(setup.element_vector_reference, archive["budget"], rtol=1e-14)
    assert setup.formula_matrix.shape == (28, 537)
    assert neutral.formula_matrix.shape == (27, 421)
    assert "e-" not in neutral.elements
    assert "e1-" not in neutral.species
    assert set(setup.metadata["excluded_species"]) == {"N1-", "Zn1-"}
    assert not {"N1-", "Zn1-"} & set(setup.species)
    keep = np.asarray(setup.formula_matrix)[-1] == 0
    assert neutral.species == tuple(s for s, take in zip(setup.species, keep) if take)
    np.testing.assert_array_equal(neutral.formula_matrix, np.asarray(setup.formula_matrix)[:-1, keep])
    np.testing.assert_array_equal(neutral.element_vector_reference, setup.element_vector_reference[:-1])
    t = jnp.array([1500., 1735., 2517., 3973., 4000.])
    np.testing.assert_allclose(neutral.hvector_func(t), setup.hvector_func(t)[:, keep], atol=1e-12)


def test_source_choices_and_unreviewed_species_are_explicit(setup, data):
    assert setup.metadata["dataset"] == "curated-gas-v1"
    assert set(setup.metadata["species_provenance"]) == set(setup.species)
    counts = Counter(r["status"] for r in setup.metadata["species_provenance"].values())
    assert counts["inherited_unreviewed"] == 427
    assert counts["reference_gauge"] == 28
    assert len(data["decisions"]) == 82
    base = fastchem4.chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    changed = (set(data["nasa9_reactions"]) | set(data["janaf_reactions"])
               | set(data["partition_reactions"]) | set(data["logk_energy_shifts_K"]))
    assert len(changed) == 70
    retained = [name for name in setup.species if name not in changed]
    t = jnp.array([1601., 2527., 3997.])
    # Separate JIT graphs can round the same inherited constants differently.
    np.testing.assert_allclose(
        setup.hvector_func(t)[:, [setup.species.index(s) for s in retained]],
        base.hvector_func(t)[:, [base.species.index(s) for s in retained]],
        atol=1e-12, rtol=0,
    )


def test_continuous_constants_reproduce_approved_nodes(setup):
    with np.load(ROOT / "tests/data/validation/gas_thermochemistry_impact/equilibria.npz") as archive:
        actual = np.asarray(setup.hvector_func(archive["temperature_K"]))
        expected = archive["h"][3]
    # Rounded Ca partition values require a small positive thermal contribution.
    partitions = [setup.species.index(s) for s in ("Co1+", "C1-", "Ca1-", "Ge1-")]
    np.testing.assert_allclose(actual[:, partitions], expected[:, partitions], atol=5e-6, rtol=0)
    others = [i for i in range(537) if i not in partitions]
    np.testing.assert_allclose(actual[:, others], expected[:, others], atol=2e-12, rtol=0)


@pytest.mark.parametrize("species, name", [("C1H4", "CH4"), ("C1O1S1", "COS"),
                                           ("O1Ti1", "TiO"), ("O1V1", "VO")])
def test_janaf_free_energy_and_derivatives_at_withheld_nodes(setup, data, species, name):
    reference = json.loads((ROOT / "examples/comparisons/data/curated_continuous_reference.json").read_text())["janaf"][name]
    record = data["janaf_reactions"][species]
    index = setup.species.index(species)
    coefficients = jnp.array([record["background_nasa9"]])

    def thermal_free_energy(t):
        return t * (setup.hvector_func(t)[index] - _nasa9(t, coefficients)[0][0])

    temperatures = jnp.array([float(t) for t in reference["rows"] if float(t) not in record["knots_K"]])
    rows = np.array([reference["rows"][str(float(t))] for t in temperatures])
    g = np.asarray(jax.vmap(thermal_free_energy)(temperatures))
    dg = np.asarray(jax.vmap(jax.grad(thermal_free_energy))(temperatures))
    d2g = np.asarray(jax.vmap(jax.grad(jax.grad(thermal_free_energy)))(temperatures))
    r = 8.31446261815324
    t = np.asarray(temperatures)
    enthalpy = 1000 * (rows[:, 2] - reference["h0_minus_h298_kJ_mol"])
    np.testing.assert_allclose(g/t, (enthalpy/t - rows[:, 1])/r, atol=6e-5*np.log(10), rtol=0)
    np.testing.assert_allclose(r*(g - t*dg), enthalpy, atol=9., rtol=0)
    np.testing.assert_allclose(-r*dg, rows[:, 1], atol=.004, rtol=0)
    np.testing.assert_allclose(-r*t*d2g, rows[:, 0], atol=.04, rtol=0)


def test_jit_batch_bounds_and_continuous_temperature_derivatives(setup):
    temperatures = jnp.array([[1500, 2000], [3000, 4000]])
    assert setup.hvector_func(temperatures).shape == (2, 2, 537)
    np.testing.assert_allclose(jax.jit(jax.vmap(setup.hvector_func))(temperatures),
                               setup.hvector_func(temperatures), atol=1e-12)
    invalid = jnp.array([0., 1499., 4001., np.nan, np.inf])
    assert np.all(np.isnan(setup.hvector_func(invalid)))
    for t in (1735., 2000., 2517., 3000., 3973.):
        first = jax.jacfwd(setup.hvector_func)(t)
        second = jax.jacfwd(jax.jacfwd(setup.hvector_func))(t)
        delta = .05
        finite = (setup.hvector_func(t+delta) - setup.hvector_func(t-delta))/(2*delta)
        np.testing.assert_allclose(first, finite, atol=2e-10, rtol=2e-8)
        np.testing.assert_allclose(second, (jax.jacfwd(setup.hvector_func)(t+delta)
                                           - jax.jacfwd(setup.hvector_func)(t-delta))/(2*delta),
                                   atol=2e-10, rtol=1e-5)
        assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    for edge, direction in ((1500., 1.), (4000., -1.)):
        step = direction * .01
        inward_derivative = (-3*setup.hvector_func(edge) + 4*setup.hvector_func(edge+step)
                             - setup.hvector_func(edge+2*step))/(2*step)
        np.testing.assert_allclose(jax.jacfwd(setup.hvector_func)(edge), inward_derivative,
                                   atol=2e-9, rtol=2e-8)


def test_partitions_fit_source_precision_with_positive_weights(data):
    reference = json.loads((ROOT / "examples/comparisons/data/curated_continuous_reference.json").read_text())
    t = np.array(reference["partition_temperature_K"])
    for name, record in reference["partitions"].items():
        fit = data["partition_functions"][name]
        weights, energy = np.array(fit["weights"]), np.array(fit["excitation_K"])
        assert np.all(weights > 0) and np.all(energy >= 0)
        actual = np.exp(-energy[None, :]/t[:, None]) @ weights
        np.testing.assert_allclose(actual, record["values"], rtol=4e-6)


def test_default_solver_resolves_trace_charge(setup):
    result, diagnostics = solve(setup, T=1500., P=100., b=setup.element_vector_reference,
                                return_diagnostics=True)
    charge = np.asarray(setup.formula_matrix)[-1]
    n = np.asarray(result.n)
    assert bool(diagnostics["converged"])
    assert abs(charge @ n) / (np.abs(charge) @ n) < 1e-10


@pytest.mark.parametrize("ions", [True, False])
def test_equilibrium_closure_and_archived_fastchem_comparison(setup, neutral, ions):
    model = setup if ions else neutral
    temperatures = jnp.repeat(jnp.array([1500., 1735., 2000., 2517., 3000., 3973., 4000.]), 5)
    pressures = jnp.tile(jnp.array([1e-6, 1e-4, .01, 1., 100.]), 7)
    result, diagnostics = solve_profile(
        model, T=temperatures, P=pressures, b=model.element_vector_reference,
        options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"), return_diagnostics=True)
    assert np.all(diagnostics["converged"])
    n = np.asarray(result.n)
    a, b = np.asarray(model.formula_matrix), np.asarray(model.element_vector_reference)
    assert np.max(np.abs(n@a.T - b)[:, b>0]/b[b>0]) < 1e-5
    references = [model.species.index("e1-" if el == "e-" else el + "1") for el in model.elements]
    logp = np.asarray(result.ln_n) - np.log(n.sum(axis=-1))[:, None] + np.log(pressures)[:, None]
    residual = logp - logp[:, references] @ a + model.hvector_func(temperatures)
    assert np.max(np.abs(residual)) < 1e-8
    if ions:
        charge = a[-1]
        assert np.max(np.abs(n@charge)/(n@np.abs(charge))) < 1e-12
        with np.load(ROOT / "tests/data/validation/gas_thermochemistry_impact/equilibria.npz") as archive:
            reference = archive["fastchem_ln_n"][3]
        selected = np.isin(np.asarray(temperatures), [1500., 2000., 3000., 4000.])
        logx = np.asarray(result.ln_n)[selected] - np.log(n.sum(axis=-1))[selected, None]
        reference -= np.log(np.exp(reference).sum(axis=-1))[:, None]
        assert np.max(np.abs(logx - reference))/np.log(10) < 1e-3
        electron = model.species.index("e1-")
        assert np.max(np.abs(logx[:, electron] - reference[:, electron])) < 1e-5


def test_invalid_options_and_changed_source_are_rejected(monkeypatch):
    with pytest.raises(TypeError, match="ions must be a bool"):
        curated.chemsetup(ions="false")
    monkeypatch.setattr(curated, "_DATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned version"):
        curated.chemsetup()


@pytest.mark.parametrize("enabled", [False, True])
def test_integer_temperature_preserves_setup_dtype(enabled):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", enabled)
    try:
        model = curated.chemsetup()
        actual = model.hvector_func(jnp.array([1500, 2517, 4000]))
        assert actual.dtype == (jnp.float64 if enabled else jnp.float32)
        assert actual.shape == (3, 537)
        assert np.all(np.isfinite(actual))
    finally:
        jax.config.update("jax_enable_x64", previous)
