"""Primitive retained-atmosphere source transfer and independent contact checks."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "reference", "source", "hydrogen", "full_potential", "common_gibbs",
         "melts_coupled", "run_melts_reference", "run_bse_common_gibbs", "m1_chemistry",
         "m2_common_gas", "m2_standards_audit", "run_m2_contact", "m2_atmosphere", "m2_expanded_source")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    ATM = importlib.import_module("m2_atmosphere")
    EXPANDED = importlib.import_module("m2_expanded_source")
    CONTACT = importlib.import_module("run_m2_contact")
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("budget", [np.array([1., .1, .3, .02, .01, .01, .01]),
                                   np.array([1., .1, 0., 0., 0., 0., 0.])])
def test_primitive_unpack_retains_full_catalog_and_audits_independent_contact(budget):
    _, setup = EXPANDED.build_setups()
    temperature, pressure = 2173.15, 1.
    reference, _ = EXPANDED.source_standards_rt(temperature, pressure)
    _, gauge = EXPANDED.anchored_standards_rt(setup.gas_setup, temperature, reference)
    atmosphere = ATM.make_atmosphere_phase(setup, gauge)
    names = [element + "_atmosphere_atom" for element in setup.elements]
    internal = {"elements": list(setup.elements), "phases": {"atmosphere": names},
                "component_formulas": {name: {element: 1.} for name, element in zip(names, setup.elements)},
                "atmosphere_element_order": list(setup.elements), "reactions": []}
    state = atmosphere(temperature, pressure, budget)
    result = {"accepted": True, "component_amounts_mol": budget, "gibbs_rt": state.gibbs_rt}
    record, public, callbacks, parcel = EXPANDED.unpack_expanded_source(
        internal, result, {"atmosphere": atmosphere}, temperature, pressure)
    assert list(record["phases"]) == ["gas", "cloud"]
    assert len(public["component_amounts_mol"]) == 61
    assert "reaction_residual" not in public
    assert set(record["retained_condensate_components"].values()) == set(setup.condensate_species)
    np.testing.assert_allclose(np.sum(public["phase_element_amounts_mol"], axis=0), budget, rtol=1e-12)
    assert public["gibbs_rt"] == pytest.approx(state.gibbs_rt, rel=1e-12)
    audit = CONTACT.audit_expanded_contact(record, public, callbacks, temperature, pressure, parcel)
    assert audit["accepted"]
    assert len(audit["gas_contact"]["species"]) == 35
    assert len(audit["common_standards"]["species"]) >= 35
    if np.any(budget == 0):
        assert CONTACT.diagnose_contact(record, budget, public, callbacks, temperature, pressure)["matched_contact_accepted"]
    with pytest.raises(ValueError, match="catalog"):
        CONTACT.audit_expanded_contact(record, public, callbacks, temperature, pressure,
                                       {**parcel, "elements": list(reversed(parcel["elements"]))})
    rejected = EXPANDED.unpack_expanded_source(internal, {**result, "accepted": "false"},
        {"atmosphere": atmosphere}, temperature, pressure)[1]
    assert rejected["accepted"] is False
    invalid = {**internal, "atmosphere_element_order": list(reversed(setup.elements))}
    with pytest.raises(ValueError, match="carriers"):
        EXPANDED.unpack_expanded_source(invalid, result, {"atmosphere": atmosphere}, temperature, pressure)
    # A provider acceptance flag cannot hide changed primitive atom amounts.
    corrupted = dict(parcel, gas_amounts_mol=(1.001 * np.asarray(parcel["gas_amounts_mol"])).tolist())
    assert not CONTACT.audit_expanded_contact(record, public, callbacks, temperature, pressure, corrupted)["accepted"]
    # A pure condensate's energy must use the same elemental gauge as the gas.
    original = callbacks["cloud"]
    def wrong_cloud(t, p, n):
        state = original(t, p, n)
        return EXPANDED.PhaseState(state.mu_rt + .01, state.gibbs_rt + .01 * np.sum(n))
    assert not CONTACT.audit_expanded_contact(record, public, {**callbacks, "cloud": wrong_cloud},
                                               temperature, pressure, parcel)["accepted"]


def test_expanded_bse_internal_basis_preserves_canonical_thirteen_element_ledger(tmp_path):
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    path = checkout / "examples/m2_material/bse_inventory.json"
    if not path.exists():
        pytest.skip("Requires the explicitly selected ExoEOS BSE checkout.")
    record, budget, callbacks, initial, metadata = EXPANDED.build_expanded_bse_problem(
        path, checkout, tmp_path, sys.executable)
    names = [name for group in record["phases"].values() for name in group]
    formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                        for element in record["elements"]])
    np.testing.assert_allclose(formula @ initial, budget, rtol=1e-12)
    assert list(record["phases"]) == ["silicate", "metal", "atmosphere"]
    assert len(record["phases"]["atmosphere"]) == 7
    assert metadata["standards"]["gas_model"] == "m1_retained"
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    assert len(metadata["atmosphere"]["condensate_species"]) == 26
