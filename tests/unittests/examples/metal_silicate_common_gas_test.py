"""Independent reaction coefficients, gauge invariance, and contact controls."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "reference", "source", "hydrogen", "full_potential", "common_gibbs",
         "melts_coupled", "run_melts_reference", "run_bse_common_gibbs", "m1_chemistry",
         "m2_common_gas", "m2_standards_audit", "run_m2_contact")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    COMMON = importlib.import_module("m2_common_gas")
    BSE = importlib.import_module("run_bse_common_gibbs")
    M1 = importlib.import_module("m1_chemistry")
    CONTACT = importlib.import_module("run_m2_contact")
    FULL = importlib.import_module("full_potential")
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("temperature", [1800., 2173.15, 2350., 2600.])
def test_water_reaction_matches_independent_janaf_coefficients_and_preserves_anchors(temperature):
    # Pinned FastChem4 JANAF fits, species H2, O2, H2O; this evaluates the
    # polynomial directly, independently of the preset's parser/evaluator.
    coefficients = np.array([
        [5.1909637142380554e4, -1.8011701211306956, .087224583233705744, 2.5613890164973008e-4, -5.354025536740606e-9],
        [5.9533604009055794e4, -1.6151664799086234, -3.6097651261292185, 4.0627380887831022e-4, -1.8694620741402745e-8],
        [1.103364538879382e5, -4.1783597409582285, 3.1744691010633233, 9.4064684023068e-4, -4.048246148286689e-8],
    ])
    independent = -coefficients @ np.array([1 / temperature, np.log(temperature), 1., temperature, temperature**2])
    setup = COMMON.build_common_gas_setup()
    reference, _ = BSE.source_standards_rt(temperature)
    values, gauge = COMMON.anchored_standards_rt(setup, temperature, reference)
    reaction = np.array([-1., -.5, 1.])
    assert reaction @ values[:3] == pytest.approx(reaction @ independent, abs=3e-14)
    for name in COMMON.REFERENCE_ANCHORS:
        assert values[COMMON.SHARED_SPECIES.index(name)] == pytest.approx(reference[name + "_gas"], abs=2e-14)
    assert np.max(np.abs(gauge)) > 1.
    assert np.max(np.abs(values - setup.hvector_func(temperature))) > 1.


def test_common_callback_varies_continuously_in_temperature_and_keeps_pressure_once(tmp_path):
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    path = checkout / "examples/m2_material/bse_inventory.json"
    if not path.exists():
        pytest.skip("Requires the explicitly selected ExoEOS BSE checkout.")
    record, _, callbacks, _, metadata = BSE.build_bse_problem(
        path, checkout, tmp_path, sys.executable, gas_model="m1_shared",
    )
    n = np.linspace(.1, 1., len(record["phases"]["gas"]))
    center = callbacks["gas"](2173.15, 1., n)
    low = callbacks["gas"](2173.149, 1., n)
    high = callbacks["gas"](2173.151, 1., n)
    assert 1e-8 < np.max(np.abs(high.mu_rt - low.mu_rt)) < 1e-3
    np.testing.assert_allclose((low.mu_rt + high.mu_rt) / 2, center.mu_rt, atol=1e-10, rtol=0)
    pressured = callbacks["gas"](2173.15, 13., n)
    np.testing.assert_allclose(pressured.mu_rt - center.mu_rt, np.log(13.), atol=1e-14)
    assert pressured.gibbs_rt - center.gibbs_rt == pytest.approx(n.sum() * np.log(13.), abs=1e-12)
    assert metadata["standards"]["common_gas"]["pressure_standard_bar"] == 1.
    assert metadata["scientific_acceptance"] == {"M2_A": "pending", "M2_B": "pending"}


def test_common_element_reference_does_not_change_conserved_parcel_equilibrium():
    setup = COMMON.build_common_gas_setup()
    b = np.array([1., .1, .3, .02, .01, .01, .01])
    reference, _ = BSE.source_standards_rt(2173.15)
    values, gauge = COMMON.anchored_standards_rt(setup, 2173.15, reference)
    original = M1.solve_parcel(setup, 2173.15, 1., b)
    from exogibbs.thermo.models import ChemicalSetup

    shifted_setup = ChemicalSetup(formula_matrix=setup.formula_matrix, elements=setup.elements,
                                 species=setup.species, hvector_func=lambda t: values)
    shifted = M1.solve_parcel(shifted_setup, 2173.15, 1., b)
    assert original["accepted"] and shifted["accepted"]
    np.testing.assert_allclose(original["gas_amounts_mol"], shifted["gas_amounts_mol"], rtol=1e-11)
    amount = np.asarray(original["gas_amounts_mol"])
    assert amount @ (values - setup.hvector_func(2173.15)) == pytest.approx(b @ gauge, abs=1e-13)


def test_contact_recounts_source_and_distinguishes_catalog_change():
    setup = COMMON.build_common_gas_setup()
    b = np.array([1., .1, .3, .02, .01, .01, .01])
    native = M1.solve_parcel(setup, 2173.15, 1., b)
    names = [name + "_gas" for name in COMMON.SHARED_SPECIES]
    record = {"elements": list(setup.elements), "phases": {"gas": names},
              "component_formulas": {name: {element: float(setup.formula_matrix[row, column])
                  for row, element in enumerate(setup.elements)} for column, name in enumerate(names)}}
    callbacks = {"gas": FULL.ideal_phase(lambda t, p: setup.hvector_func(t), gas=True)}
    source = {"accepted": True, "component_amounts_mol": native["gas_amounts_mol"]}
    report = CONTACT.diagnose_contact(record, b, source, callbacks, 2173.15, 1.)
    assert report["shared_contact_accepted"] and report["numerical_diagnostics_completed"]
    assert [entry["catalog"] for entry in report["catalogs"]] == ["shared_gas", "expanded_gas", "expanded_condensed"]
    assert report["catalogs"][1]["contact"]["maximum_residual"] > 1e-4
    source["component_amounts_mol"] = np.asarray(source["component_amounts_mol"]) * 1.01
    with pytest.raises(ValueError, match="ledger"):
        CONTACT.diagnose_contact(record, b, source, callbacks, 2173.15, 1.)
