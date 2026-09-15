"""Thermochemical anchors, finite closure, and common boundary regressions."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import build_condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup

PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "m1_thermochemical_control.py"
SPEC = importlib.util.spec_from_file_location("m1_thermochemical_control", PATH)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


@pytest.fixture(scope="module")
def upper():
    return CONTROL.CHEMISTRY.build_setups()[1]


def test_declared_gas_data_change_propagates_to_dissolution(upper):
    network, case, _ = CONTROL.CHEMISTRY.source_inputs()
    original = deepcopy(case)
    control = CONTROL.build_control(network, case, upper)
    assert case == original
    shift = np.asarray(control["case"]["source_delta_g_over_rt"]) - case["source_delta_g_over_rt"]
    expected = np.zeros_like(shift)
    expected[[9, 15, 18]] = [0.001332583393, -0.001332583393, 0.026788593378]
    np.testing.assert_allclose(shift, expected, rtol=0, atol=1e-11)
    assert control["standard_reconstruction_max_abs_rt"] < 1e-12
    assert len(control["contact_reaction_indices"]) == 14


def test_common_element_gauge_leaves_control_reaction_data_unchanged(upper):
    network, case, _ = CONTROL.CHEMISTRY.source_inputs()
    gauge = jnp.asarray([0.7, -0.9, 0.5, 1.4, -2.1, 0.8, 0.3])

    def shifted(setup):
        return ChemicalSetup(
            formula_matrix=setup.formula_matrix, elements=setup.elements,
            species=setup.species, temperature_validity_upper=setup.temperature_validity_upper,
            hvector_func=lambda temperature: setup.hvector_func(temperature) + setup.formula_matrix.T @ gauge,
        )

    changed = build_condensate_chemical_setup(gas_setup=shifted(upper.gas_setup),
                                             condensate_setup=shifted(upper.condensate_setup))
    first = CONTROL.build_control(network, case, upper)
    second = CONTROL.build_control(network, case, changed)
    np.testing.assert_allclose(first["case"]["source_delta_g_over_rt"],
                               second["case"]["source_delta_g_over_rt"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("temperature", [1000., 3000.])
def test_control_refuses_unavailable_source_temperature(upper, temperature):
    network, case, _ = CONTROL.CHEMISTRY.source_inputs()
    case["T_K"] = temperature
    with pytest.raises(ValueError, match="2350 K"):
        CONTROL.build_control(network, case, upper)


@pytest.fixture(scope="module", params=[0.9, 1.0, 1.1], ids=["low_oxygen", "central", "high_oxygen"])
def local_control(request, upper):
    network, case, budget = CONTROL.CHEMISTRY.source_inputs(request.param)
    pressure = {0.9: 20.43, 1.0: 16.224, 1.1: 12.9}[request.param]
    report = CONTROL.solve_control(network, case, upper, element_amounts_mol=budget, pressure_bar=pressure)
    return network, case, budget, report


def test_shared_partial_pressures_close_all_source_reactions(local_control, upper):
    network, case, budget, report = local_control
    audit = CONTROL.audit_control(network, case, upper, report)
    assert report["accepted"] and audit["accepted"]
    assert report["common_gas_contact_equilibrium"]
    assert not report["global_phase_stability_certified"]
    assert len(audit["reaction_residual"]) == 16
    assert np.max(np.abs(audit["reaction_residual"])) < 1e-8
    assert np.max(np.abs(audit["relative_element_residual"])) < 1e-9
    np.testing.assert_allclose(np.sum(list(audit["phase_element_amounts_mol"].values()), axis=0),
                               budget, rtol=1e-9, atol=0)
    parcel = report["bottom_parcel"]
    np.testing.assert_allclose(np.asarray(parcel["gas_element_amounts_mol"]) + parcel["cloud_element_amounts_mol"],
                               report["atmosphere_element_amounts_mol"], rtol=1e-9, atol=0)


def test_audit_recounts_primitive_arrays_instead_of_saved_acceptance(local_control, upper):
    network, case, _, report = local_control
    bad = deepcopy(report)
    column = bad["deep_source_species"].index("MgO_silicate")
    bad["deep_component_amounts_mol"][column] *= 1.01
    assert not CONTROL.audit_control(network, case, upper, bad)["accepted"]


def test_control_preserves_absolute_amount_scale(upper):
    network, case, budget = CONTROL.CHEMISTRY.source_inputs()
    report = CONTROL.solve_control(network, case, upper, element_amounts_mol=budget, pressure_bar=16.224)
    scaled = CONTROL.solve_control(network, case, upper, element_amounts_mol=budget * 1e20,
                                   pressure_bar=16.224, initial_control=report)
    for key in ("deep_component_amounts_mol", "atmosphere_element_amounts_mol"):
        np.testing.assert_allclose(np.asarray(scaled[key]) / 1e20, report[key], rtol=1e-8, atol=0)


def test_large_pressure_steps_then_neighboring_and_fresh_roots(upper):
    saved = json.loads(Path(__file__).with_name("data").joinpath("m1_contact_pressure_seed.json").read_text())
    initial = saved["seed"]
    network, case, _ = CONTROL.CHEMISTRY.source_inputs()
    budget = np.asarray(initial["element_amounts_mol"])
    results = []
    for pressure in saved["pressure_probes_bar"]:
        result = CONTROL.solve_control(network, case, upper, element_amounts_mol=budget,
                                        pressure_bar=pressure, initial_control=initial)
        assert result["accepted"]
        assert max(np.abs(result["reaction_residual"])) < 1e-8
        assert max(np.abs(result["relative_element_residual"])) < 1e-9
        results.append(result)
        initial = result
    # Repeating a local pressure solve must reproduce all physical amounts.
    for key in ("deep_component_amounts_mol", "atmosphere_element_amounts_mol"):
        np.testing.assert_allclose(results[-1][key], results[-2][key], rtol=1e-8, atol=0)


@pytest.fixture
def low_oxygen_pressure_seed():
    path = Path(__file__).with_name("data") / "m1_contact_inventory_seed.json"
    return json.loads(path.read_text())["seed"]


@pytest.mark.parametrize("pressure,warm", [(80., True), (80., False), (60., True)])
def test_pressure_steps_keep_nested_parcels_within_finite_supply(
    upper, low_oxygen_pressure_seed, monkeypatch, pressure, warm,
):
    seed = low_oxygen_pressure_seed
    network, case, _ = CONTROL.CHEMISTRY.source_inputs(0.9)
    budget = np.asarray(seed["element_amounts_mol"])
    rows = [network["elements"].index(e) for e in CONTROL.CHEMISTRY.ELEMENTS]
    original_parcel = CONTROL.CHEMISTRY.solve_parcel

    def checked_parcel(setup, temperature, pressure_bar, atoms):
        assert np.max(np.asarray(atoms) / budget[rows]) <= 1. + 64. * np.finfo(float).eps
        return original_parcel(setup, temperature, pressure_bar, atoms)

    monkeypatch.setattr(CONTROL.CHEMISTRY, "solve_parcel", checked_parcel)
    report = CONTROL.solve_control(
        network, case, upper, element_amounts_mol=budget, pressure_bar=pressure,
        initial_control=seed if warm else None,
    )
    assert report["accepted"] and CONTROL.audit_control(network, case, upper, report)["accepted"]
    helium = CONTROL.CHEMISTRY.ELEMENTS.index("He")
    assert report["atmosphere_element_amounts_mol"][helium] / budget[rows[helium]] == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("phase", ["deep", "atmosphere"])
def test_control_rejects_seed_above_finite_capacity(upper, low_oxygen_pressure_seed, phase):
    seed = low_oxygen_pressure_seed
    network, case, _ = CONTROL.CHEMISTRY.source_inputs(0.9)
    budget = np.asarray(seed["element_amounts_mol"])
    if phase == "deep":
        index = seed["deep_source_species"].index("Fe_metal")
        seed["deep_component_amounts_mol"][index] = 2. * budget[network["elements"].index("Fe")]
    else:
        index = CONTROL.CHEMISTRY.ELEMENTS.index("He")
        seed["atmosphere_element_amounts_mol"][index] = 2. * budget[network["elements"].index("He")]
    with pytest.raises(ValueError, match="finite elemental capacities"):
        CONTROL.solve_control(network, case, upper, element_amounts_mol=budget,
                              pressure_bar=seed["P_bar"], initial_control=seed)


def test_control_allows_roundoff_at_helium_capacity(upper, low_oxygen_pressure_seed):
    seed = low_oxygen_pressure_seed
    network, case, _ = CONTROL.CHEMISTRY.source_inputs(0.9)
    budget = np.asarray(seed["element_amounts_mol"])
    helium = CONTROL.CHEMISTRY.ELEMENTS.index("He")
    capacity = budget[network["elements"].index("He")]
    seed["atmosphere_element_amounts_mol"][helium] = np.nextafter(capacity, np.inf)
    report = CONTROL.solve_control(network, case, upper, element_amounts_mol=budget,
                                   pressure_bar=seed["P_bar"], initial_control=seed)
    assert CONTROL.audit_control(network, case, upper, report)["accepted"]
