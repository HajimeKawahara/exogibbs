"""Approximate source boundary audits and saved low-oxygen regressions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "m1_chemistry.py"
SPEC = importlib.util.spec_from_file_location("m1_boundary_chemistry", PATH)
M1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1)


def test_shared_gas_local_acceptance_does_not_certify_common_equilibrium():
    network, case, budget = M1.source_inputs()
    source = M1.SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget,
                                           pressure_bar=100.)
    shared, _ = M1.build_setups()
    report = M1.solve_parcel(shared, case["T_K"], source["P_bar"], M1.source_gas_inventory(network, source))
    boundary = M1.audit_boundary(network, case, source, shared, report)
    assert boundary["contract_met"] and boundary["local_accepted"]
    assert not boundary["common_equilibrium_certified"]
    assert not boundary["m1_a_accepted"]
    assert {9, 18} <= set(boundary["unmet_source_reaction_indices"])
    comparison = M1.shared_reaction_comparison(network, case, report)
    positions = [boundary["source_reaction_indices"].index(i) for i in comparison["source_reaction_indices"]]
    np.testing.assert_allclose(
        np.asarray(boundary["fixed_deep_source_reaction_residual_rt"])[positions],
        comparison["source_reaction_residual"], atol=1e-12,
    )


BOUNDARY_CASES = json.loads(Path(__file__).with_name("data").joinpath("m1_boundary_cases.json").read_text())["cases"]


@pytest.fixture(scope="module", params=BOUNDARY_CASES, ids=lambda saved: saved["id"])
def low_oxygen_boundary(request):
    saved = request.param
    network, case, _ = M1.source_inputs(saved["oxygen_factor"])
    assert network["elements"] == saved["source_elements"]
    source = M1.SOURCE.solve_reduced_source(
        network, case, element_amounts_mol=saved["source_element_amounts_mol"],
        pressure_bar=saved["P_bar"],
    )
    budget = M1.source_gas_inventory(network, source)
    np.testing.assert_allclose(budget, saved["source_gas_element_amounts_mol"], rtol=1e-10, atol=0.)
    _, upper = M1.build_setups()
    report = M1.solve_parcel(upper, saved["T_K"], saved["P_bar"], budget)
    return saved, network, case, source, upper, report


def test_low_oxygen_contract_quantifies_condensation_and_unmet_deep_reactions(low_oxygen_boundary):
    saved, network, case, source, upper, report = low_oxygen_boundary
    boundary = M1.audit_boundary(network, case, source, upper, report)
    assert boundary["local_accepted"] and boundary["contract_met"]
    assert not boundary["common_equilibrium_certified"]
    assert not boundary["m1_a_accepted"]
    np.testing.assert_allclose(boundary["cloud_element_fraction"], saved["saved_cloud_element_fraction"], atol=1e-8)
    assert boundary["max_abs_delta_source_reaction_residual_rt"] == pytest.approx(
        saved["saved_max_abs_delta_source_reaction_residual_rt"], abs=1e-8,
    )
    np.testing.assert_allclose(boundary["source_reaction_residual_rt"], source["reaction_residual"], atol=1e-12)
    positions = [boundary["source_reaction_indices"].index(i) for i in (9, 18)]
    np.testing.assert_allclose(np.asarray(boundary["fixed_deep_source_reaction_residual_rt"])[positions],
                               [-0.0013325834, -0.0267885934], atol=1e-8)
    # Independently replace the source's gas log activities. Expanded upper
    # gases dilute these activities; the shared subset must not be renormalized.
    species, _, _ = M1.SOURCE.component_matrices(network)
    fractions = np.asarray(source["component_amounts_mol"]).copy()
    for phase in network["phases"].values():
        columns = [species.index(name) for name in phase]
        fractions[columns] /= fractions[columns].sum()
    for source_name, upper_name in zip(M1.SHARED_SOURCE_SPECIES, M1.SHARED_GAS_SPECIES):
        fractions[species.index(source_name)] = report["gas_mole_fractions"][report["gas_species"].index(upper_name)]
    supported = fractions > 0
    callback = M1.SOURCE._make_reaction_residual(network, case, np.flatnonzero(supported),
                                               np.asarray(source["active_reaction_indices"]))
    np.testing.assert_allclose(callback(np.log(fractions[supported]), source["P_bar"]),
                               boundary["fixed_deep_source_reaction_residual_rt"], atol=1e-12)


@pytest.mark.parametrize("field", ["T_K", "P_bar"])
def test_boundary_requires_common_temperature_and_pressure(low_oxygen_boundary, field):
    _, network, case, source, upper, report = low_oxygen_boundary
    changed = {**report, field: report[field] * 1.01}
    with pytest.raises(ValueError, match="same source and upper temperature and pressure"):
        M1.audit_boundary(network, case, source, upper, changed)


@pytest.mark.parametrize("phase", ["gas", "condensate"])
def test_boundary_recounts_actual_phase_amounts_despite_accepted_report(low_oxygen_boundary, phase):
    _, network, case, source, upper, report = low_oxygen_boundary
    changed = {**report, f"{phase}_amounts_mol": (1.01 * np.asarray(report[f"{phase}_amounts_mol"])).tolist()}
    boundary = M1.audit_boundary(network, case, source, upper, changed)
    assert boundary["local_accepted"]
    assert not boundary["contract_met"]
    assert np.max(np.abs(boundary["relative_element_residual"])) > M1.ELEMENT_TOLERANCE


def test_boundary_does_not_use_stale_element_or_partial_pressure_summaries(low_oxygen_boundary):
    _, network, case, source, upper, report = low_oxygen_boundary
    changed = {**report, **{key: [0.] * len(report[key]) for key in (
        "element_amounts_mol", "gas_element_amounts_mol", "cloud_element_amounts_mol",
        "gas_mole_fractions", "partial_pressures_bar",
    )}}
    assert M1.audit_boundary(network, case, source, upper, changed) == M1.audit_boundary(network, case, source, upper, report)


def test_boundary_rechecks_source_atom_closure_despite_accepted_flag(low_oxygen_boundary):
    _, network, case, source, upper, report = low_oxygen_boundary
    amounts = np.asarray(source["component_amounts_mol"]).copy()
    phase = network["phases"]["silicate"]
    amounts[[source["species"].index(name) for name in phase]] *= 1.01
    changed = {**source, "component_amounts_mol": amounts.tolist()}
    boundary = M1.audit_boundary(network, case, changed, upper, report)
    assert boundary["local_accepted"]
    assert not boundary["contract_met"]
    assert np.max(np.abs(boundary["source_relative_element_residual"])) > M1.ELEMENT_TOLERANCE
