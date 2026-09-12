"""Exact-zero inventory controls on the separately pinned GCE source hosts."""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "sulfur_source.py"
SPEC = importlib.util.spec_from_file_location("cns_source", PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
RECORD = SOURCE.load_reference()
ZERO_CASES = [("carbon", ("C",))] + [
    ("sulfur_nitrogen", absent)
    for count in (1, 2, 3) for absent in combinations(("C", "N", "S"), count)
]


def reduced_inputs(name, absent, case_index=0):
    network = RECORD["networks"][name]
    budget = np.asarray(network["element_amounts_mol"]).copy()
    budget[[network["elements"].index(element) for element in absent]] = 0.0
    return network, network["cases"][case_index], budget


@pytest.mark.parametrize("name,absent", ZERO_CASES)
@pytest.mark.parametrize("case_index", [0, 1])
def test_zero_cns_support_closes_finite_source_equations(name, absent, case_index):
    network, case, budget = reduced_inputs(name, absent, case_index)
    result = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget)
    species, formula, reactions = SOURCE.component_matrices(network)
    amounts = np.asarray(result["component_amounts_mol"])
    excluded = np.any(formula[budget == 0] != 0, axis=0)
    active_rows = np.flatnonzero(np.all(reactions[:, excluded] == 0, axis=1))

    assert result["accepted"] and result["support_policy"] == "exact_zero_cns"
    assert set(result["zero_budget_elements"]) == set(absent)
    assert result["active_species"] == [name for name, removed in zip(species, excluded) if not removed]
    assert result["active_elements"] == [name for name, total in zip(network["elements"], budget) if total > 0]
    np.testing.assert_array_equal(result["active_reaction_indices"], active_rows)
    np.testing.assert_array_equal(amounts[excluded], 0.0)
    assert np.all(amounts[~excluded] > 0)
    assert np.all(np.asarray(result["phase_amounts_mol"]) > 0)
    np.testing.assert_allclose(formula @ amounts, budget, rtol=1e-9, atol=0)
    np.testing.assert_allclose(np.sum(result["phase_element_amounts_mol"], axis=0), budget,
                               rtol=1e-9, atol=0)
    for names, contribution in zip(network["phases"].values(), result["phase_element_amounts_mol"]):
        columns = [species.index(name) for name in names]
        np.testing.assert_allclose(formula[:, columns] @ amounts[columns], contribution,
                                   rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.asarray(result["relative_element_residual"])[budget == 0], 0.0)

    # Absent columns have zero coefficients in every retained equation.
    # Neutral logarithms let the independently checked full callback audit
    # those equations without claiming a positive amount of absent material.
    ln_x = np.zeros(len(species))
    for names in network["phases"].values():
        indices = [species.index(name) for name in names if name in result["active_species"]]
        ln_x[indices] = np.log(amounts[indices] / amounts[indices].sum())
    checked = np.asarray(SOURCE.make_reaction_residual(network, case)(ln_x, result["P_bar"]))[active_rows]
    np.testing.assert_allclose(checked, 0.0, atol=1e-8)
    np.testing.assert_allclose(result["reaction_residual"], checked, atol=1e-12)
    assert len(active_rows) + np.count_nonzero(budget) == np.count_nonzero(~excluded)


@pytest.mark.parametrize("name", ["carbon", "sulfur_nitrogen"])
def test_positive_reduced_entry_point_preserves_source_fixture(name):
    network = RECORD["networks"][name]
    case = network["cases"][1]
    full = SOURCE.solve_source(network, case)
    reduced = SOURCE.solve_reduced_source(network, case)
    assert reduced["zero_budget_elements"] == []
    for key, expected in full.items():
        assert reduced[key] == expected


@pytest.mark.parametrize("name,absent", [("carbon", ("C",)), ("sulfur_nitrogen", ("C", "N"))])
@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_reduced_chemistry_is_invariant_to_amount_scale(name, absent, scale):
    network, case, budget = reduced_inputs(name, absent)
    reference = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget)
    expected = np.asarray(reference["component_amounts_mol"])
    initial = expected * scale * np.exp(np.linspace(-0.2, 0.2, expected.size))
    result = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget * scale,
                                         initial_component_amounts_mol=initial)
    np.testing.assert_allclose(np.asarray(result["component_amounts_mol"]) / scale,
                               expected, rtol=2e-8, atol=0)


@pytest.mark.parametrize("name,absent", [
    ("carbon", ()), ("carbon", ("C",)),
    ("sulfur_nitrogen", ()), ("sulfur_nitrogen", ("C", "N")),
])
@pytest.mark.parametrize("scale", [1e-6, 1e20])
def test_default_reduced_seed_follows_absolute_inventory_scale(name, absent, scale):
    network, case, budget = reduced_inputs(name, absent)
    reference = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget)
    scaled = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget * scale)

    assert scaled["accepted"]
    amounts = np.asarray(scaled["component_amounts_mol"])
    np.testing.assert_allclose(amounts / scale, reference["component_amounts_mol"],
                               rtol=2e-8, atol=0)
    _, formula, _ = SOURCE.component_matrices(network)
    np.testing.assert_allclose(formula @ amounts, budget * scale, rtol=1e-9, atol=0)
    np.testing.assert_allclose(scaled["reaction_residual"], 0.0, atol=1e-8)


@pytest.mark.parametrize("element,value", [("C", -1), ("N", np.nan), ("S", np.inf), ("O", 0), ("Fe", 0)])
def test_reduced_source_rejects_invalid_or_absent_background_budget(element, value):
    network, case, budget = reduced_inputs("sulfur_nitrogen", ("C",))
    budget[network["elements"].index(element)] = value
    with pytest.raises(ValueError, match="nonnegative C/N/S and positive background"):
        SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget)


@pytest.mark.parametrize("seed_error", ["positive_absent", "zero_active", "nonfinite", "shape"])
def test_explicit_reduced_initial_amounts_must_match_support(seed_error):
    network, case, budget = reduced_inputs("carbon", ("C",))
    species, formula, _ = SOURCE.component_matrices(network)
    active = np.all(formula[budget == 0] == 0, axis=0)
    initial = np.where(active, case["initial_component_amounts_mol"], 0.0)
    if seed_error == "positive_absent":
        initial[species.index("C_metal")] = 1.0
    elif seed_error == "zero_active":
        initial[species.index("Fe_metal")] = 0.0
    elif seed_error == "nonfinite":
        initial[0] = np.nan
    else:
        initial = initial[:-1]
    with pytest.raises(ValueError, match="positive on active components and exactly zero elsewhere"):
        SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget,
                                    initial_component_amounts_mol=initial)


@pytest.mark.parametrize("pressure", [0.0, -1.0, np.nan, np.inf])
def test_reduced_source_requires_finite_positive_local_pressure(pressure):
    network, case, budget = reduced_inputs("carbon", ("C",))
    with pytest.raises(ValueError, match="pressure_bar must be finite and positive"):
        SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget, pressure_bar=pressure)
