"""Finite background-element gas catalogs, reference invariance and contact."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("m1_chemistry", "m2_common_gas", "m2_omitted_gas", "m2_janaf", "m2_finite_gas",
         "local", "reference", "source", "hydrogen", "full_potential", "common_gibbs",
         "melts_coupled", "run_melts_reference", "run_bse_common_gibbs", "phase_selection",
         "m2_atmosphere", "m2_expanded_source", "m2_standards_audit", "run_m2_contact")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    GAS = importlib.import_module("m2_finite_gas")
    ATM = importlib.import_module("m2_atmosphere")
    SOURCE = importlib.import_module("m2_expanded_source")
    CONTACT = importlib.import_module("run_m2_contact")
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture(scope="module")
def setup():
    return GAS.build_atmosphere_setup("janaf")


def test_catalog_contains_all_41_omitted_neutral_gases_and_original_clouds(setup):
    assert len(setup.elements) == 13
    assert len(setup.gas_species) == 76
    assert len(setup.condensate_species) == 26
    assert tuple(setup.gas_species[:35]) == GAS.EXPANDED_GAS_SPECIES
    assert tuple(setup.condensate_species) == GAS.CONDENSATE_SPECIES
    assert len(set(setup.gas_species)) == 76
    assert np.linalg.matrix_rank(setup.gas_setup.formula_matrix) == 13
    assert np.all(np.asarray(setup.condensate_setup.formula_matrix)[7:] == 0)
    assert len(GAS.catalog_sha256(setup)) == 64
    assert GAS.catalog_sha256(setup) != GAS.catalog_sha256(GAS.build_atmosphere_setup())
    with pytest.raises(ValueError, match="gas_model"):
        GAS.build_atmosphere_setup("unknown")


def test_janaf_gauge_keeps_every_existing_gas_and_cloud_standard(setup):
    t = 2173.15
    reference, _ = SOURCE.source_standards_rt(t)
    old = GAS.build_atmosphere_setup()
    original = GAS.atmosphere_gauge_rt(old, t, reference)
    new = GAS.atmosphere_gauge_rt(setup, t, reference)
    np.testing.assert_array_equal(new[:7], original)
    for old_phase, new_phase in ((old.gas_setup, setup.gas_setup),
                                 (old.condensate_setup, setup.condensate_setup)):
        old_h = np.asarray(old_phase.hvector_func(t)) + np.asarray(old_phase.formula_matrix).T @ original
        new_h = np.asarray(new_phase.hvector_func(t)) + np.asarray(new_phase.formula_matrix).T @ new
        np.testing.assert_array_equal(new_h[:len(old_h)], old_h)
    with pytest.raises(ValueError, match="no extrapolation"):
        GAS.atmosphere_gauge_rt(setup, 1000., reference)


@pytest.mark.parametrize("temperature", [1000., 2173.15])
def test_finite_phosphorus_and_upper_composition_are_gauge_invariant(setup, temperature):
    # A finite atmospheric allocation with all thirteen elements; the column
    # re-evaluates reactions at its own T and uses no JANAF extrapolation.
    b = np.array([1., .1, .03, .001, .001, .001, .001,
                  1e-7, 1e-7, 1e-7, 1e-8, 1e-7, 2e-4])
    phase = ATM.make_atmosphere_phase(setup, np.zeros(13))
    report = phase.parcel(temperature, 267., b)
    shifted = ATM.make_atmosphere_phase(setup, np.arange(13) * .37).parcel(temperature, 267., b)
    assert report["accepted"] and shifted["accepted"]
    np.testing.assert_allclose(report["gas_amounts_mol"], shifted["gas_amounts_mol"], rtol=1e-12)
    np.testing.assert_allclose(report["condensate_amounts_mol"], shifted["condensate_amounts_mol"], rtol=1e-12)
    assert shifted["gibbs_rt"] - report["gibbs_rt"] == pytest.approx(np.arange(13) @ b * .37, abs=1e-10)
    phosphorus = setup.elements.index("P")
    demand = np.asarray(setup.gas_setup.formula_matrix)[phosphorus] @ report["gas_amounts_mol"]
    assert demand == pytest.approx(b[phosphorus], rel=1e-9)
    assert sum(report["gas_mole_fractions"]) == pytest.approx(1.)


def test_thirteen_carrier_unpack_and_contact_include_background_species(setup):
    b = np.array([1., .1, .03, .001, .001, .001, .001,
                  1e-7, 1e-7, 1e-7, 1e-8, 1e-7, 2e-4])
    reference, _ = SOURCE.source_standards_rt(2173.15)
    q = GAS.atmosphere_gauge_rt(setup, 2173.15, reference)
    phase = ATM.make_atmosphere_phase(setup, q)
    names = [element + "_atmosphere_atom" for element in setup.elements]
    record = {"elements": list(setup.elements), "phases": {"atmosphere": names},
              "component_formulas": {name: {element: 1.} for name, element in zip(names, setup.elements)},
              "atmosphere_element_order": list(setup.elements), "atmosphere_gas_model": "janaf", "reactions": []}
    state = phase(2173.15, 267., b)
    result = {"accepted": True, "component_amounts_mol": b, "gibbs_rt": state.gibbs_rt}
    public, values, callbacks, parcel = SOURCE.unpack_expanded_source(record, result, {"atmosphere": phase}, 2173.15, 267.)
    assert len(values["component_amounts_mol"]) == 102
    np.testing.assert_allclose(np.sum(values["phase_element_amounts_mol"], axis=0), b, rtol=1e-9)
    audit = CONTACT.audit_expanded_contact(public, values, callbacks, 2173.15, 267., parcel)
    assert audit["accepted"]
    assert len(audit["gas_contact"]["species"]) == 76
    assert CONTACT.diagnose_contact(public, b, values, callbacks, 2173.15, 267.)["matched_contact_accepted"]


@pytest.mark.parametrize("mode", ["janaf", "janaf_condensed"])
def test_bse_builder_preserves_all_atoms_and_pins_identical_model_files(tmp_path, mode):
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    path = checkout / "examples/m2_material/bse_inventory.json"
    if not path.exists():
        pytest.skip("Requires the explicitly selected ExoEOS BSE checkout.")
    outputs = [SOURCE.build_expanded_bse_problem(path, checkout, tmp_path, sys.executable,
                                                gas_model=selected) for selected in ("m1", mode)]
    record, budget, callbacks, initial, metadata = outputs[1]
    names = [name for group in record["phases"].values() for name in group]
    formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                        for element in record["elements"]])
    np.testing.assert_allclose(formula @ initial, budget, rtol=1e-12)
    assert len(record["phases"]["atmosphere"]) == 13
    assert callbacks["atmosphere"].setup.gas_species == GAS.build_atmosphere_setup("janaf").gas_species
    assert metadata["standards"]["gas_model"] == mode + "_retained"
    assert len(metadata["standards"]["common_gas"]["element_gauge_rt"]) == 13
    assert metadata["provenance"]["file_sha256"] == outputs[0][-1]["provenance"]["file_sha256"]
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    assert len(metadata["atmosphere"]["condensate_species"]) == (67 if mode == "janaf_condensed" else 26)


def test_background_condensates_keep_the_gas_catalog_and_pure_phase_validity(setup):
    expanded = GAS.build_atmosphere_setup("janaf_condensed")
    assert expanded.gas_species == setup.gas_species
    assert len(expanded.condensate_species) == 67
    assert expanded.condensate_species[:26] == setup.condensate_species
    assert len(set(expanded.condensate_species)) == 67
    matrix = np.asarray(expanded.condensate_setup.formula_matrix)
    assert np.all(np.any(matrix[7:, 26:] > 0, axis=0))
    candidates = expanded.condensate_species
    upper = expanded.condensate_setup.temperature_validity_upper
    assert upper[candidates.index("Mg3P2O8(s,l)")] == 4500.
    assert upper[candidates.index("PH3(s,l)")] == 185.56
    assert GAS.catalog_sha256(expanded) != GAS.catalog_sha256(setup)


@pytest.mark.parametrize("temperature", [1000., 2173.15])
def test_67_condensates_preserve_finite_atoms_and_cannot_raise_minimum(setup, temperature):
    b = np.array([1., .1, .03, .001, .001, .001, .001,
                  1e-7, 1e-7, 1e-7, 1e-8, 1e-7, 2e-4])
    expanded = GAS.build_atmosphere_setup("janaf_condensed")
    original = ATM.make_atmosphere_phase(setup, np.zeros(13)).parcel(temperature, 267., b)
    report = ATM.make_atmosphere_phase(expanded, np.zeros(13)).parcel(temperature, 267., b)
    assert report["accepted"]
    assert report["gibbs_rt"] <= original["gibbs_rt"] + 1e-10
    np.testing.assert_allclose(np.asarray(report["gas_element_amounts_mol"])
                               + report["cloud_element_amounts_mol"], b, rtol=1e-9)
    assert report["condensate_amounts_mol"][expanded.condensate_species.index("PH3(s,l)")] == 0.
    assert max(abs(x) for x in report["relative_element_residual"]) < 1e-9


@pytest.mark.parametrize("mode", ["m1", "janaf", "janaf_condensed"])
def test_canonical_interior_initialization_changes_only_a_conserved_start(tmp_path, mode):
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    path = checkout / "examples/m2_material/bse_inventory.json"
    if not path.exists():
        pytest.skip("Requires the explicitly selected ExoEOS BSE checkout.")
    default = SOURCE.build_expanded_bse_problem(path, checkout, tmp_path, sys.executable, gas_model=mode)
    record, b, callbacks, canonical, metadata = SOURCE.build_expanded_bse_problem(
        path, checkout, tmp_path, sys.executable, gas_model=mode, initialization="canonical")
    seed = np.array(metadata["numerical_initialization"]["initial_component_amounts_mol"])
    names = [name for phase in record["phases"].values() for name in phase]
    formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                        for element in record["elements"]])
    np.testing.assert_allclose(formula @ seed, b, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(canonical, default[3])
    assert default[-1]["numerical_initialization"]["initial_component_amounts_mol"] is None
    assert metadata["numerical_initialization"]["canonical_interior_fraction"] == 1e-4
    assert metadata["standards"] == default[-1]["standards"]
    assert metadata["atmosphere"] == default[-1]["atmosphere"]
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    metal = [names.index(name) for name in record["phases"]["metal"]]
    np.testing.assert_array_equal(seed[metal], 0.)
    for name in record["phases"]["atmosphere"]:
        assert seed[names.index(name)] > 0
    if mode != "m1":
        p = names.index("P_atmosphere_atom")
        assert canonical[p] == 0. and 0 < seed[p] < 1e-4 * b[record["elements"].index("P")]
    zero_he, no_he = b.copy(), canonical.copy()
    zero_he[record["elements"].index("He")] = 0.
    no_he[names.index("He_atmosphere_atom")] = 0.
    restricted = SOURCE.canonical_interior_seed(record, zero_he, no_he)
    assert restricted[names.index("He_atmosphere_atom")] == 0.
    np.testing.assert_allclose(formula @ restricted, zero_he, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="ledger"):
        SOURCE.canonical_interior_seed(record, b, canonical * 1.001)
    with pytest.raises(ValueError, match="initialization"):
        SOURCE.build_expanded_bse_problem(path, checkout, tmp_path, sys.executable, initialization="bad")


def test_warm_start_execution_is_outside_the_physical_contract(tmp_path):
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    path = checkout / "examples/m2_material/bse_inventory.json"
    if not path.exists():
        pytest.skip("Requires the explicitly selected ExoEOS BSE checkout.")
    cold = SOURCE.build_expanded_bse_problem(path, checkout, tmp_path, sys.executable, gas_model="janaf")
    warm = SOURCE.build_expanded_bse_problem(path, checkout, tmp_path, sys.executable, gas_model="janaf",
                                           atmosphere_warm_start=True)
    assert cold[-1]["standards"] == warm[-1]["standards"]
    assert cold[-1]["atmosphere"] == warm[-1]["atmosphere"]
    assert cold[-1]["provenance"] == warm[-1]["provenance"]
    assert warm[-1]["numerical_execution"]["atmosphere"] is warm[2]["atmosphere"].numerical_execution
    assert not cold[-1]["numerical_execution"]["atmosphere"]["warm_start_enabled"]
    assert warm[-1]["numerical_execution"]["atmosphere"]["warm_start_enabled"]
