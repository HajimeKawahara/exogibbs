"""Full-network checks against unmodified, pinned GCE equations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
SPEC = importlib.util.spec_from_file_location("sulfur_source", EXAMPLES / "sulfur_source.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
RECORD = SOURCE.load_reference()
CASES = [(network, case) for network in RECORD["networks"].values() for case in network["cases"]]


@pytest.mark.parametrize("name,components,elements,reactions", [
    ("sulfur_nitrogen", 37, 9, 28), ("carbon", 26, 7, 19),
])
def test_source_network_retains_all_components_and_balanced_reactions(
    name, components, elements, reactions,
) -> None:
    network = RECORD["networks"][name]
    species, formula, nu = SOURCE.component_matrices(network)
    assert len(species) == components
    assert formula.shape == (elements, components)
    assert nu.shape == (reactions, components)
    np.testing.assert_array_equal(formula @ nu.T, 0.0)
    assert np.linalg.matrix_rank(formula) == elements
    assert np.linalg.matrix_rank(nu) == reactions
    if name == "sulfur_nitrogen":
        assert network["component_formulas"]["FeO15_silicate"]["O"] == 1.5
        assert {"N2_gas", "NH3_gas", "HCN_gas", "N2_silicate", "S_metal",
                "FeS_silicate", "FeSO4_silicate", "SiH4_gas"} <= set(species)
    else:
        assert "S" not in network["elements"]
        assert "C_metal" in species


@pytest.mark.parametrize("network,case", CASES, ids=[case["id"] for _, case in CASES])
def test_jitted_full_residual_reproduces_independent_author_equations(network, case) -> None:
    residual = jax.jit(SOURCE.make_reaction_residual(network, case))
    for probe in case["probes"]:
        actual = residual(np.log(probe["mole_fractions"]), probe["P_bar"])
        np.testing.assert_allclose(actual, probe["source_reaction_residual"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("network,case", CASES, ids=[case["id"] for _, case in CASES])
def test_finite_source_roots_match_independent_reference(network, case) -> None:
    result = SOURCE.solve_source(network, case)
    assert result["accepted"]
    np.testing.assert_allclose(result["component_amounts_mol"], case["component_amounts_mol"],
                               rtol=2e-8, atol=1e-25)
    np.testing.assert_allclose(result["reaction_residual"], 0.0, atol=1e-8)
    np.testing.assert_allclose(result["relative_element_residual"], 0.0, atol=1e-9)
    np.testing.assert_allclose(np.sum(result["phase_element_amounts_mol"], axis=0),
                               network["element_amounts_mol"], rtol=1e-9)


def test_sulfur_root_rescales_with_finite_inventory_and_changed_initial_composition() -> None:
    network = RECORD["networks"]["sulfur_nitrogen"]
    case = network["cases"][1]
    factor = 7.5
    initial = factor * np.asarray(case["component_amounts_mol"])
    initial *= np.exp(np.linspace(-0.25, 0.25, initial.size))
    result = SOURCE.solve_source(network, case,
                                 element_amounts_mol=factor * np.asarray(network["element_amounts_mol"]),
                                 initial_component_amounts_mol=initial)
    np.testing.assert_allclose(result["component_amounts_mol"],
                               factor * np.asarray(case["component_amounts_mol"]),
                               rtol=2e-8, atol=1e-25)


def test_source_pressure_derivative_has_no_young_r14_offset() -> None:
    network = RECORD["networks"]["sulfur_nitrogen"]
    case = network["cases"][0]
    residual = SOURCE.make_reaction_residual(network, case)
    ln_x = np.log(case["probes"][0]["mole_fractions"])
    difference = np.asarray(residual(ln_x, 10.0) - residual(ln_x, 1.0))
    species, _, nu = SOURCE.component_matrices(network)
    gas = [species.index(name) for name in network["phases"]["gas"]]
    np.testing.assert_allclose(difference, nu[:, gas].sum(axis=1) * np.log(10.0), atol=1e-13)
    assert difference[14] == pytest.approx(-np.log(10.0))


def test_sulfur_correction_uses_full_host_fraction_and_preserves_source_log_bases() -> None:
    network = RECORD["networks"]["sulfur_nitrogen"]
    case = network["cases"][0]
    species, _, nu = SOURCE.component_matrices(network)
    probe = np.asarray(case["probes"][0]["mole_fractions"])
    ln_x = np.log(probe)
    x_feo = probe[species.index("FeO_silicate")]
    log_c_s = (-5.704 + 3.15 * x_feo + 0.12 * probe[species.index("MgO_silicate")]
               + 0.75 * probe[species.index("Na2O_silicate")])
    source_correction = -2.302585093 * (-log_c_s + np.log(x_feo))
    conventional_log10_correction = -2.302585093 * (-log_c_s + np.log10(x_feo))
    result = SOURCE.make_reaction_residual(network, case)(ln_x, 1.0)
    ordinary = case["source_delta_g_over_rt"][21] + nu[21] @ ln_x
    assert float(result[21] - ordinary) == pytest.approx(source_correction)
    assert abs(source_correction - conventional_log10_correction) > 1.0


def test_full_source_rejects_zero_budgets_instead_of_inserting_trace_material() -> None:
    network = RECORD["networks"]["sulfur_nitrogen"]
    budget = np.asarray(network["element_amounts_mol"])
    budget[-2] = 0.0
    with pytest.raises(ValueError, match="strictly positive element budgets"):
        SOURCE.solve_source(network, network["cases"][0], element_amounts_mol=budget)


def test_source_provenance_pins_original_files_and_does_not_claim_calibration() -> None:
    assert RECORD["evidence_level"] == "source reproduction"
    assert RECORD["source"]["commit"] == "31558873d8da460c3cd11986b574cb347621b43d"
    assert len(RECORD["source"]["files"]["Gibbs.py"]) == 64
    assert "No joint calibration" in RECORD["domain"]["calibration"]
    assert any("mixed log-base" in note for note in RECORD["audit"])
