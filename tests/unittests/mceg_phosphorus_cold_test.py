"""Check thermodynamic identities and reference conventions of the cold study."""

from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.thermo.models import ChemicalSetup


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_20261030.py"))


@pytest.fixture
def data():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield STUDY["load_data"]()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_source_identities_and_formation_enthalpy_anchors(data):
    names = ["HPO2", "H3PO4", "PH3", "P4O6"]
    assert data["identity_notes"]["HPO2"] == "HOPO"
    assert data["identity_notes"]["H3PO4"] == "O=P(OH)3"
    isomers = [data["records"][s] for s in ("H3PO3_phosphorous", "H3PO3_phosphonic")]
    assert isomers[0]["composition"] == isomers[1]["composition"]
    assert isomers[0]["raw_record_sha256"] != isomers[1]["raw_record_sha256"]
    g, s, _ = STUDY["thermodynamics"](data, names, 298.15)
    h = np.asarray((g + s) * 298.15 * data["source_gas_constant_J_mol_K"])
    expected = [data["records"][name]["hf298_J_mol"] for name in names]
    np.testing.assert_allclose(h, expected, rtol=0, atol=2.)
    assert all(data["records"][name]["phase"] == "gas" for name in names)


def test_burcat_entropy_probe_distinguishes_bar_from_atmosphere(data):
    probe = data["standard_state_evidence"]["burcat_pressure"]["probe_record_outside_network"]
    isolated = {**data, "records": {"O_1D": probe}}
    _, entropy, _ = STUDY["thermodynamics"](isolated, ["O_1D"], 298.15)
    source_entropy = float(entropy[0]) * data["source_gas_constant_J_mol_K"]
    k, planck, avogadro = 1.380649e-23, 6.62607015e-34, 6.02214076e23
    mass, temperature = 15.9994 * 1.66053906660e-27, 298.15
    translation = (2*np.pi*mass*k*temperature)**1.5 * k*temperature / planck**3
    ideal_entropy = lambda pressure: k*avogadro*(np.log(translation/pressure) + 2.5 + np.log(5.))
    assert abs(source_entropy - ideal_entropy(1e5)) < .001
    assert abs(source_entropy - ideal_entropy(101325.)) > .1


@pytest.mark.parametrize("temperature", [350., 700., 1250.])
def test_gibbs_and_enthalpy_derivatives_match_entropy_and_heat_capacity(data, temperature):
    names = list(data["records"])
    step = .01
    low = STUDY["thermodynamics"](data, names, temperature - step)
    high = STUDY["thermodynamics"](data, names, temperature + step)
    _, entropy, heat_capacity = STUDY["thermodynamics"](data, names, temperature)
    dg_dt = ((temperature + step)*high[0] - (temperature - step)*low[0]) / (2*step)
    dh_dt = ((temperature + step)*(high[0]+high[1]) - (temperature - step)*(low[0]+low[1])) / (2*step)
    np.testing.assert_allclose(dg_dt, -entropy, rtol=2e-7, atol=2e-5)
    np.testing.assert_allclose(dh_dt, heat_capacity, rtol=2e-7, atol=2e-5)


def test_additions_preserve_background_and_reject_extrapolation(data):
    models = STUDY["networks"](data)
    base = STUDY["make_setup"](data, *models["base"])
    both = STUDY["make_setup"](data, *models["add_both"])
    t = jnp.asarray([200., 700., 1000., 1500.])
    shared = [both.species.index(name) for name in base.species]
    np.testing.assert_allclose(np.asarray(both.hvector_func(t))[:, shared], base.hvector_func(t), rtol=0, atol=2e-12)
    np.testing.assert_array_equal(base.element_vector_reference, both.element_vector_reference)
    atoms = [both.species.index(e) for e in both.elements]
    np.testing.assert_allclose(np.asarray(both.hvector_func(t))[:, atoms], 0., rtol=0, atol=1e-12)
    invalid = jnp.asarray([199., 1501., np.nan, np.inf, -1.])
    assert np.isnan(np.asarray(both.hvector_func(invalid))).all()
    for values in STUDY["thermodynamics"](data, list(both.species), invalid):
        assert np.isnan(np.asarray(values)).all()


def test_hydrogen_mass_action_and_elemental_gauge_invariance(data):
    setup = STUDY["make_setup"](data, ["H", "H2"])
    temperature, pressure = 1500., .001
    options = EquilibriumOptions(epsilon_crit=1e-13, max_iter=1000)
    baseline, info = solve(setup, temperature, pressure, setup.element_vector_reference,
                           options=options, return_diagnostics=True)
    assert bool(info["converged"])
    kp = np.exp(-float(setup.hvector_func(temperature)[1]))
    expected_h = 2. / (1. + np.sqrt(1. + 4.*kp*pressure))
    np.testing.assert_allclose(baseline.x, [expected_h, 1.-expected_h], rtol=2e-7, atol=1e-12)
    shift = setup.formula_matrix.T @ jnp.asarray([7.25])
    shifted = ChemicalSetup(setup.formula_matrix, lambda t: setup.hvector_func(t) + shift,
                            setup.elements, setup.species, setup.element_vector_reference)
    changed, info = solve(shifted, temperature, pressure, shifted.element_vector_reference,
                          options=options, return_diagnostics=True)
    assert bool(info["converged"])
    np.testing.assert_allclose(changed.ln_n, baseline.ln_n, rtol=0, atol=1e-9)
