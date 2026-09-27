"""Conserved He extension and independent whole-scalar derivatives."""

import importlib
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    SOURCE = importlib.import_module("m2_expanded_source")
    HE = importlib.import_module("m2_helium")
    FULL = importlib.import_module("full_potential")
    STABILITY = importlib.import_module("m2_host_stability")
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def checkout():
    eos = pytest.importorskip("exoeos")
    path = Path(eos.__file__).resolve().parents[2]
    if not (path / "examples/m2_material/helium_dissolution.py").exists():
        pytest.skip("Requires the finite-He EOS provider")
    return path


def build(checkout, tmp_path, **options):
    return SOURCE.build_expanded_bse_problem(
        checkout / "examples/m2_material/bse_inventory.json", checkout, tmp_path,
        sys.executable, pressure_bar=269.6545152505794,
        gas_model="janaf_condensed", initialization="canonical", **options)


def test_finite_atomic_helium_keeps_budgets_and_records_actual_retained_gas_anchor(checkout, tmp_path):
    record, budget, callbacks, initial, metadata = build(
        checkout, tmp_path, helium_solubility_model="guillot2012_olivine")
    names = [n for phase in record["phases"].values() for n in phase]
    formula = np.array([[record["component_formulas"][n].get(e, 0.) for n in names]
                        for e in record["elements"]])
    assert record["phases"]["silicate"][-1] == "He_dissolved"
    assert record["component_formulas"]["He_dissolved"] == {"He": 1.}
    assert initial[names.index("He_dissolved")] == 0.
    np.testing.assert_allclose(formula @ initial, budget, rtol=1e-12)
    np.testing.assert_allclose(formula @ metadata["numerical_initialization"]["initial_component_amounts_mol"],
                               budget, rtol=1e-12)
    receipt = metadata["helium_dissolution"]
    anchor = receipt["gas_anchor"]
    assert anchor["common_standard_rt"] == pytest.approx(
        anchor["retained_raw_standard_rt"] + np.dot(anchor["formula"], anchor["element_gauge_rt"]))
    assert receipt["gas_standard_rt"] == anchor["common_standard_rt"]
    assert receipt["pressure_standard_bar"] == 1.
    assert receipt["pressure_bar"] == 269.6545152505794
    assert receipt["dry_host_molar_masses_kg"][-2:] == [0., 0.]
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    assert "He silicate dissolution" not in metadata["atmosphere"]["missing_paths"]
    assert {"Na alloy component", "He alloy dissolution"} <= set(metadata["atmosphere"]["missing_paths"])
    assert metadata["helium_solubility_model"] == "guillot2012_olivine"
    changed = copy.deepcopy(receipt)
    changed["gas_anchor"]["formula"] = [0.] * len(anchor["elements"])
    # Even preserving the numeric gauge sum cannot turn another atom into He.
    changed["gas_anchor"]["retained_raw_standard_rt"] = receipt["gas_standard_rt"]
    with pytest.raises(ValueError, match="formula"):
        HE.reconstruct_helium_model(checkout, changed)


def test_wrapped_host_potentials_and_saved_recipe_agree_with_closed_henry_solution(checkout):
    provider = HE._provider(checkout)
    added = provider.make_helium_dissolution("guillot2012_morb", 2173.15, [.04, 0.], -10.)
    receipt = {**added.receipt, "temperature_K": 2173.15, "pressure_bar": 270.}
    host = FULL.ideal_phase(lambda t, p: np.array([-5., -3.]))
    phase = HE.wrap_helium_phase(host, checkout, receipt)
    gas = FULL.ideal_phase(lambda t, p: np.array([-10., 0.]), gas=True)
    dry_amount, wet_amount, gas_background, total_he = 25., .01, 2., .003
    capacity = receipt["capacity"]["He_mol_per_kg_dry_host_per_bar_fugacity"]
    mass = dry_amount * .04
    # Conserve total He at fixed P with a finite ideal-gas denominator.
    gas_he = brentq(lambda g: g + mass*capacity*270.*g/(gas_background+g)-total_he,
                   0., total_he, xtol=1e-16)
    dissolved = total_he-gas_he
    n = np.array([dry_amount, wet_amount, dissolved])
    state = phase(2173.15, 270., n)
    gas_state = gas(2173.15, 270., [gas_he, gas_background])
    assert state.mu_rt[-1] == pytest.approx(gas_state.mu_rt[0], abs=1e-11)
    energy, gradient = phase.energy_value_and_grad_rt(2173.15, 270., n)
    np.testing.assert_allclose(gradient, state.mu_rt, atol=1e-12)
    assert energy == pytest.approx(state.gibbs_rt, abs=1e-12)
    assert n @ gradient == pytest.approx(energy, abs=1e-12)
    # He affects the dry-host potential, not the added water's zero mass weight.
    base = host(2173.15, 270., n[:-1])
    assert state.mu_rt[0]-base.mu_rt[0] == pytest.approx(-dissolved/dry_amount)
    assert state.mu_rt[1] == base.mu_rt[1]
    with pytest.raises(ValueError, match="recipe"):
        HE.wrap_helium_phase(host, checkout, {**receipt, "gas_standard_rt": -9.,
                                            "scalar": "inconsistent saved law"})
    with pytest.raises(ValueError, match="T/P"):
        phase(2200., 270., n)
    wet_only = phase(2173.15, 270., [0., 1., 0.])
    bare_wet_only = host(2173.15, 270., [0., 1.])
    assert wet_only.gibbs_rt == bare_wet_only.gibbs_rt
    np.testing.assert_array_equal(wet_only.mu_rt[:-1], bare_wet_only.mu_rt)
    assert wet_only.mu_rt[-1] == np.inf
    with pytest.raises(ValueError, match="Positive He"):
        phase(2173.15, 270., [0., 1., .001])


def test_default_preserves_gas_only_species_and_marks_each_omission(checkout, tmp_path):
    record, _, _, _, metadata = build(checkout, tmp_path)
    assert "He_dissolved" not in record["phases"]["silicate"]
    assert metadata["helium_solubility_model"] == "gas_only"
    assert "He silicate dissolution" in metadata["atmosphere"]["missing_paths"]
    with pytest.raises(ValueError, match="helium_solubility_model"):
        build(checkout, tmp_path, helium_solubility_model="BSE")


def test_fresh_candidate_audit_preserves_bare_provider_and_reconstructs_he(checkout):
    names = ["A_melts", "B_melts", "H2_dissolved", "He_dissolved"]
    record = {"phases": {"silicate": names}, "component_formulas": {
        "A_melts": {"A": 1.}, "B_melts": {"B": 1.},
        "H2_dissolved": {"H": 2.}, "He_dissolved": {"He": 1.}}}
    bare = {"model_id": STABILITY.PROVIDER_MODEL_ID,
        "status": "ok_supplied_liquid_properties", "T_K": 2173.15, "P_Pa": 270.e5,
        "component_order": ["A", "B"], "component_moles": [2., 3.], "mu_RT": [-2., -3.],
        "oxide_molar_masses_g_mol": [1., 1.], "gibbs_J": -13.*STABILITY.COMMON_R*2173.15,
        "basis": {"component_oxide_matrix": [[1., 0.], [0., 1.]],
                  "component_element_matrix": [[1., 0.], [0., 1.]],
                  "element_order": ["A", "B"], "common_R_J_mol_K": STABILITY.COMMON_R},
        "phase_policy": {"oxygen_buffer": "None", "equilibrated": False},
        "saturation": {"equilibrated": False, "candidate_order": ["AB"], "candidates": [{
            "phase": "AB", "status": "ok_candidate_properties", "reason": None,
            "native_affinity_J": 0., "oxide_mass_g": [1., 1.],
            "gibbs_J": -5.*STABILITY.COMMON_R*2173.15}]}}
    evaluator = SimpleNamespace(COMPONENTS=["A", "B"], ELEMENTS=["A", "B"],
        FORMULA_MATRIX=np.eye(2), evaluate_liquid=lambda *a, **k: copy.deepcopy(bare))
    model = HE._provider(checkout).make_helium_dissolution("guillot2012_morb", 2173.15,
                                                         [.001, .001, 0.], -10.)
    receipt = {**model.receipt, "temperature_K": 2173.15, "pressure_bar": 270.,
               "host_component_order": names[:-1], "component_order": names}
    amounts = [2., 3., 5., .01]
    options = dict(evaluator=evaluator, runtime=None, python_executable=sys.executable,
                   amount_scale=1., exoeos_checkout=checkout)
    result = STABILITY.evaluate_host_stability(record, amounts, 2173.15, 270.,
                                               helium_dissolution=receipt, **options)
    assert result["provider_properties"] == bare
    assert result["native_candidate_properties"] == bare
    assert result["log_native_host_fraction"] == pytest.approx(-np.log(2.))
    he = result["helium_dissolution"]
    np.testing.assert_allclose(he["native_host_mu_correction_rt"], [-.002, -.002])
    assert he["native_additional_gibbs_rt"] == pytest.approx(model.state(amounts)["gibbs_rt"])
    assert result["candidates"][0]["helium_dissolution_correction_gibbs_rt"] == pytest.approx(.004)
    with pytest.raises(ValueError, match="saved He"):
        STABILITY.evaluate_host_stability(record, amounts, 2173.15, 270., **options)
