"""Local CNS scans conserve host atoms and report molecular gas mass."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
sys.path.insert(0, str(EXAMPLES))
try:
    SPEC = importlib.util.spec_from_file_location("cns_inventory", EXAMPLES / "cns_inventory.py")
    SCAN = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SCAN)
finally:
    sys.path.pop(0)
REFERENCE = SCAN.load_reference()


@pytest.mark.parametrize("element,absent", [("C", ("S", "N")), ("S", ("C", "N")), ("N", ())])
def test_scan_crosses_exact_zero_without_changing_background_inventory(element, absent) -> None:
    network = REFERENCE["networks"]["sulfur_nitrogen"]
    budget = np.asarray(network["element_amounts_mol"]).copy()
    for zero in absent:
        budget[network["elements"].index(zero)] = 0.0
    column = network["elements"].index(element)
    values = [0.0, budget[column], 0.0]
    original = budget.copy()
    result = SCAN.scan_source(network, network["cases"][0], element, values,
                              element_amounts_mol=budget)
    np.testing.assert_array_equal(budget, original)
    assert "no atmospheric column closure" in result["pressure_policy"]
    for value, point in zip(values, result["states"]):
        local = point["local_state"]
        summary = point["summary"]
        assert local["accepted"]
        assert local["P_bar"] == network["cases"][0]["P_bar"]
        expected = original.copy()
        expected[column] = value
        np.testing.assert_array_equal(local["element_amounts_mol"], expected)
        phase_totals = summary["phase_element_amounts_mol"]
        actual = [sum(phase[e] for phase in phase_totals.values()) for e in network["elements"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
        fractions = summary["gas_mole_fractions"]
        assert sum(fractions.values()) == pytest.approx(1.0)
        assert summary["gas_mass_g"] > 0
        assert summary["gas_mean_molar_mass_g_mol"] > 0
        assert summary["gas_h2o_h2_ratio"] == pytest.approx(fractions["H2O_gas"] / fractions["H2_gas"])
        if value == 0:
            assert all(phase[element] == 0 for phase in summary["phase_cns_amounts_mol"].values())
    np.testing.assert_allclose(result["states"][0]["local_state"]["component_amounts_mol"],
                               result["states"][-1]["local_state"]["component_amounts_mol"],
                               rtol=1e-7, atol=0)


def test_carbon_scan_scales_absolute_reservoirs_but_not_composition() -> None:
    network = REFERENCE["networks"]["carbon"]
    case = network["cases"][0]
    budget = np.asarray(network["element_amounts_mol"])
    values = [0.0, budget[-1]]
    first = SCAN.scan_source(network, case, "C", values, pressure_bar=1000.0)
    factor = 7.5
    scaled = SCAN.scan_source(network, case, "C", factor * np.asarray(values),
                              element_amounts_mol=factor * budget, pressure_bar=1000.0)
    for point, larger in zip(first["states"], scaled["states"]):
        np.testing.assert_allclose(larger["local_state"]["component_amounts_mol"],
                                   factor * np.asarray(point["local_state"]["component_amounts_mol"]),
                                   rtol=1e-7, atol=0)
        assert larger["summary"]["gas_mass_g"] == pytest.approx(factor * point["summary"]["gas_mass_g"])
        assert larger["summary"]["gas_mean_molar_mass_g_mol"] == pytest.approx(
            point["summary"]["gas_mean_molar_mass_g_mol"])
        for phase in point["summary"]["phase_cns_amounts_mol"].values():
            assert phase["N"] == phase["S"] == 0.0


def test_gas_mass_counts_molecules_instead_of_atom_moles() -> None:
    network = REFERENCE["networks"]["sulfur_nitrogen"]
    species, _, _ = SCAN.component_matrices(network)
    amounts = np.zeros(len(species))
    for name, value in {"H2_gas": 2.0, "H2O_gas": 1.0, "CO_gas": 3.0,
                        "CO2_gas": 4.0, "CH4_gas": 5.0, "N2_gas": 6.0,
                        "NH3_gas": 7.0, "HCN_gas": 8.0, "H2S_gas": 9.0,
                        "SO2_gas": 10.0, "C_metal": 1000.0}.items():
        amounts[species.index(name)] = value
    summary = SCAN.summarize_state(network, {"component_amounts_mol": amounts})
    expected = (2 * 2.016 + 18.015 + 3 * 28.010 + 4 * 44.009 + 5 * 16.043
                + 6 * 28.014 + 7 * 17.031 + 8 * 27.026 + 9 * 34.076 + 10 * 64.058)
    assert summary["gas_mass_g"] == pytest.approx(expected)
    assert summary["gas_mean_molar_mass_g_mol"] == pytest.approx(expected / 55)
    assert summary["phase_cns_amounts_mol"]["metal"]["C"] == 1000.0
    assert summary["phase_masses_g"]["metal"] == pytest.approx(12011.0)


@pytest.mark.parametrize("element,values", [("O", [1.0]), ("N", [1.0]), ("C", []),
                                            ("C", [-1.0]), ("C", [np.nan]), ("C", [[1.0]])])
def test_scan_rejects_invalid_sequence(element, values) -> None:
    network = REFERENCE["networks"]["carbon"]
    with pytest.raises(ValueError):
        SCAN.scan_source(network, network["cases"][0], element, values)


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_scan_validates_baseline_before_overwriting_scanned_budget(invalid) -> None:
    network = REFERENCE["networks"]["carbon"]
    budget = np.asarray(network["element_amounts_mol"]).copy()
    budget[-1] = invalid
    with pytest.raises(ValueError, match="Budgets must be finite"):
        SCAN.scan_source(network, network["cases"][0], "C", [0.0], element_amounts_mol=budget)
