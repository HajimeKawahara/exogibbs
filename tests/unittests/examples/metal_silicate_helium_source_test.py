"""Finite inert He, dilution, and exact-zero controls on the source S/N host."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "sulfur_source.py"
SPEC = importlib.util.spec_from_file_location("helium_source", PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
REFERENCE = SOURCE.load_reference()
BASE = REFERENCE["networks"]["sulfur_nitrogen"]
HELIUM = SOURCE.build_helium_network(BASE)


def inputs(case_index=0, *, helium=0.002, absent=()):
    budget = np.asarray(HELIUM["element_amounts_mol"]).copy()
    budget[-1] = helium
    for element in absent:
        budget[HELIUM["elements"].index(element)] = 0.0
    return HELIUM["cases"][case_index], budget


def audit(result, case, budget):
    """Count atoms independently and re-evaluate retained reaction equations."""
    species, formula, reactions = SOURCE.component_matrices(HELIUM)
    amounts = np.asarray(result["component_amounts_mol"])
    excluded = np.any(formula[budget == 0] != 0, axis=0)
    active_rows = np.flatnonzero(np.all(reactions[:, excluded] == 0, axis=1))
    np.testing.assert_array_equal(amounts[excluded], 0.0)
    assert np.all(amounts[~excluded] > 0)
    np.testing.assert_allclose(formula @ amounts, budget, rtol=1e-9, atol=0)
    ln_x = np.zeros(len(species))
    for phase_index, names in enumerate(HELIUM["phases"].values()):
        indices = [species.index(name) for name in names]
        np.testing.assert_allclose(
            formula[:, indices] @ amounts[indices],
            result["phase_element_amounts_mol"][phase_index], rtol=1e-12, atol=0,
        )
        active = [index for index in indices if not excluded[index]]
        ln_x[active] = np.log(amounts[active] / amounts[active].sum())
    # Neutral logs for excluded species do not contribute to retained rows.
    residual = np.asarray(SOURCE.make_reaction_residual(HELIUM, case)(ln_x, result["P_bar"]))[active_rows]
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(result["reaction_residual"], residual, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(result["phase_element_amounts_mol"])[:2, -1], 0.0)
    assert amounts[-1] == pytest.approx(budget[-1], rel=1e-9, abs=0)
    assert result["accepted"]
    assert result["source_model_id"] == BASE["model_id"]
    assert result["source_case_id"] == case["source_case_id"]
    assert result["evidence_level"] == "ideal-He extension of source equations"
    if "active_reaction_indices" in result:
        np.testing.assert_array_equal(result["active_reaction_indices"], active_rows)


def test_builder_preserves_source_and_records_only_derived_inputs():
    original = deepcopy(BASE)
    derived = SOURCE.build_helium_network(original)
    assert original == BASE
    assert derived["model_id"] != BASE["model_id"]
    assert derived["source_model_id"] == BASE["model_id"]
    assert derived["elements"] == BASE["elements"] + ["He"]
    assert derived["element_amounts_mol"] == BASE["element_amounts_mol"] + [0.0]
    species, formula, reactions = SOURCE.component_matrices(derived)
    _, base_formula, base_reactions = SOURCE.component_matrices(BASE)
    assert formula.shape == (10, 38) and reactions.shape == (28, 38)
    assert species[-1] == "He_gas"
    np.testing.assert_array_equal(formula[:-1, :-1], base_formula)
    np.testing.assert_array_equal(formula[-1], [0] * 37 + [1])
    np.testing.assert_array_equal(formula[:-1, -1], 0)
    np.testing.assert_array_equal(reactions[:, :-1], base_reactions)
    np.testing.assert_array_equal(reactions[:, -1], 0)
    np.testing.assert_array_equal(reactions @ formula.T, 0)
    for case, base in zip(derived["cases"], BASE["cases"]):
        assert set(case) == {"id", "source_case_id", "T_K", "P_bar",
                             "source_delta_g_over_rt", "initial_component_amounts_mol"}
        assert case["source_case_id"] == base["id"] and case["id"] != base["id"]
        assert case["initial_component_amounts_mol"] == base["component_amounts_mol"] + [0.0]
        assert case["source_delta_g_over_rt"] == base["source_delta_g_over_rt"]
    derived["reactions"][0]["stoichiometry"].clear()
    derived["cases"][0]["source_delta_g_over_rt"][0] = 999.0
    assert original == BASE


@pytest.mark.parametrize("case_index,absent", [(0, ()), (1, ()), (0, ("C", "N"))])
def test_zero_helium_recovers_original_source_root(case_index, absent):
    case, budget = inputs(case_index, helium=0.0, absent=absent)
    if absent:
        base = SOURCE.solve_reduced_source(
            BASE, BASE["cases"][case_index], element_amounts_mol=budget[:-1],
        )["component_amounts_mol"]
    else:
        base = BASE["cases"][case_index]["component_amounts_mol"]
    result = SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget)
    np.testing.assert_allclose(result["component_amounts_mol"][:-1], base, rtol=2e-8, atol=0)
    assert result["component_amounts_mol"][-1] == 0.0
    assert set(result["zero_budget_elements"]) == {*absent, "He"}
    assert result["support_policy"] == "exact_zero_cns_he"
    audit(result, case, budget)


@pytest.mark.parametrize("case_index,entry,default_seed", [
    (0, "solve_reduced_source", False), (1, "solve_source", False),
    (0, "solve_reduced_source", True), (1, "solve_source", True),
])
def test_helium_pressure_compensation_preserves_reactive_partial_pressures(case_index, entry, default_seed):
    base_case = BASE["cases"][case_index]
    base_species, _, _ = SOURCE.component_matrices(BASE)
    gas = [base_species.index(name) for name in BASE["phases"]["gas"]]
    base_amounts = np.asarray(base_case["component_amounts_mol"])
    base_gas_moles = base_amounts[gas].sum()
    helium_moles = 0.5 * base_gas_moles
    case, budget = inputs(case_index, helium=helium_moles)
    expected = np.append(base_amounts, helium_moles)
    pressure = base_case["P_bar"] * (1 + helium_moles / base_gas_moles)
    kwargs = {}
    if not default_seed:
        seed = expected * np.exp(np.linspace(-0.2, 0.2, expected.size))
        before = seed.copy()
        kwargs["initial_component_amounts_mol"] = seed
    result = getattr(SOURCE, entry)(HELIUM, case, element_amounts_mol=budget,
                                   pressure_bar=pressure, **kwargs)
    if not default_seed:
        np.testing.assert_array_equal(seed, before)
    amounts = np.asarray(result["component_amounts_mol"])
    np.testing.assert_allclose(amounts, expected, rtol=2e-8, atol=0)
    np.testing.assert_allclose(
        amounts[gas] / (amounts[gas].sum() + amounts[-1]) * pressure,
        base_amounts[gas] / base_gas_moles * base_case["P_bar"], rtol=2e-8, atol=0,
    )
    audit(result, case, budget)

    # Adding He at unchanged total pressure must not leave this composition at equilibrium.
    species, _, _ = SOURCE.component_matrices(HELIUM)
    ln_x = np.empty(len(species))
    for names in HELIUM["phases"].values():
        indices = [species.index(name) for name in names]
        ln_x[indices] = np.log(expected[indices] / expected[indices].sum())
    uncompensated = SOURCE.make_reaction_residual(HELIUM, case)(ln_x, base_case["P_bar"])
    assert np.max(np.abs(uncompensated)) > 0.1


@pytest.fixture(scope="module")
def reduced_helium_root():
    case, budget = inputs(absent=("C", "N"))
    result = SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget)
    audit(result, case, budget)
    return case, budget, result


@pytest.mark.parametrize("scale", [1e-6, 1e20])
def test_default_helium_seed_respects_reduced_inventory_scale(reduced_helium_root, scale):
    case, budget, reference = reduced_helium_root
    result = SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget * scale)
    np.testing.assert_allclose(np.asarray(result["component_amounts_mol"]) / scale,
                               reference["component_amounts_mol"], rtol=2e-8, atol=0)
    audit(result, case, budget * scale)


def test_positive_helium_with_all_cns_absent_retains_exact_support():
    case, budget = inputs(absent=("C", "N", "S"))
    result = SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget)
    assert set(result["zero_budget_elements"]) == {"C", "N", "S"}
    assert "He_gas" in result["active_species"]
    audit(result, case, budget)


@pytest.mark.parametrize("element,value", [("He", -1), ("He", np.nan), ("He", np.inf), ("O", 0)])
def test_helium_branch_rejects_invalid_budgets(element, value):
    case, budget = inputs()
    budget[HELIUM["elements"].index(element)] = value
    with pytest.raises(ValueError, match="finite nonnegative C/N/S/He and positive background"):
        SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget)


@pytest.mark.parametrize("seed_error", ["positive_absent_he", "zero_present_he", "shape", "nonfinite"])
def test_explicit_helium_seed_must_match_support(seed_error):
    case, budget = inputs(helium=0.0 if seed_error == "positive_absent_he" else 0.002)
    seed = np.asarray(case["initial_component_amounts_mol"]).copy()
    seed[-1] = budget[-1]
    if seed_error == "positive_absent_he":
        seed[-1] = 0.002
    elif seed_error == "zero_present_he":
        seed[-1] = 0.0
    elif seed_error == "shape":
        seed = seed[:-1]
    else:
        seed[-1] = np.nan
    with pytest.raises(ValueError, match="positive on active components and exactly zero elsewhere"):
        SOURCE.solve_reduced_source(HELIUM, case, element_amounts_mol=budget,
                                    initial_component_amounts_mol=seed)


@pytest.mark.parametrize("network", [HELIUM, REFERENCE["networks"]["carbon"]])
def test_builder_rejects_repeated_extension_and_separate_carbon_model(network):
    with pytest.raises(ValueError, match="unextended sulfur/nitrogen"):
        SOURCE.build_helium_network(network)


def test_full_source_entry_still_rejects_exact_zero_helium():
    case, budget = inputs(helium=0.0)
    with pytest.raises(ValueError, match="strictly positive element budgets"):
        SOURCE.solve_source(HELIUM, case, element_amounts_mol=budget)
