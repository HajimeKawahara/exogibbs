"""Independent contracts for the metal--silicate reference thermochemistry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
SPEC = importlib.util.spec_from_file_location("metal_silicate_reference", EXAMPLE_DIRECTORY / "reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
RECORD = REFERENCE.load_reference()


def test_full_young_network_is_balanced_and_independent() -> None:
    species, formula, reactions = REFERENCE.component_matrices(RECORD)
    assert tuple(RECORD["elements"]) == ("Si", "Mg", "O", "Fe", "H", "Na", "C")
    assert len(species) == len(set(species)) == 25
    assert [len(names) for names in RECORD["phases"].values()] == [11, 4, 10]
    assert formula.shape == (7, 25)
    assert reactions.shape == (18, 25)
    assert np.linalg.matrix_rank(formula) == 7
    assert np.linalg.matrix_rank(reactions) == 25 - 7
    np.testing.assert_array_equal(formula @ reactions.T, np.zeros((7, 18)))
    assert RECORD["phases"]["silicate"][:3] == [
        "MgO_silicate", "SiO2_silicate", "MgSiO3_silicate",
    ]
    assert "SiH4_gas" not in species  # A different GCE version adds this component.


@pytest.mark.parametrize("case", RECORD["cases"], ids=lambda case: case["id"])
def test_independent_standards_recover_all_source_reaction_constants(case) -> None:
    species, _, reactions = REFERENCE.component_matrices(RECORD)
    potentials = REFERENCE.source_standard_potentials(RECORD, case)
    rt = RECORD["source"]["gas_constant_J_mol_K"] * case["T_K"]
    np.testing.assert_allclose(
        reactions @ potentials / rt,
        case["source_delta_g_over_rt"], rtol=5e-12, atol=5e-12,
    )
    # Separately captured GCE species values check the Shomate kJ/J and S units.
    for name, expected in case["source_mu0_J_mol"].items():
        np.testing.assert_allclose(potentials[species.index(name)], expected, rtol=5e-13)


@pytest.mark.parametrize("case", RECORD["cases"], ids=lambda case: case["id"])
def test_empirical_log10_constants_and_reaction_direction(case) -> None:
    t = case["T_K"]
    ln10 = np.log(10.0)
    # Use ln(10), independently of the source's rounded conversion constant.
    expected_log_k = {
        0: ln10 * (1.33 - 13870.0 / t),
        1: 0.5 * ln10 * (2.97 - 21800.0 / t),
        3: -ln10 * ((2.736 - 11439.0 / t) + 0.5 * (2.97 - 21800.0 / t)),
        5: ln10 * (0.63 - 3103.0 / t),
        14: -12.500076,
        15: 2565.0 / t - 14.21,
    }
    for index, ln_k in expected_log_k.items():
        np.testing.assert_allclose(-case["source_delta_g_over_rt"][index], ln_k, atol=1e-9, rtol=0)
    assert RECORD["reactions"][3]["stoichiometry"] == {
        "Si_metal": -0.5, "O_metal": -1.0, "SiO2_silicate": 0.5,
    }


def test_common_elemental_reference_shift_preserves_all_reactions() -> None:
    _, formula, reactions = REFERENCE.component_matrices(RECORD)
    for case in RECORD["cases"]:
        potentials = REFERENCE.source_standard_potentials(RECORD, case)
        elemental_offsets = 1e4 * np.asarray([2.5, -4.0, 7.0, -3.0, 1.5, 9.0, -8.0])
        shifted = potentials + formula.T @ elemental_offsets
        np.testing.assert_allclose(reactions @ shifted, reactions @ potentials, atol=2e-9, rtol=1e-13)


@pytest.mark.parametrize("state", RECORD["exoeos_standard_reference"]["states"], ids=lambda state: state["id"])
def test_completed_metal_standard_conversion_preserves_chemical_potentials(state) -> None:
    case = next(case for case in RECORD["cases"] if case["T_K"] == state["T_K"])
    species, _, reactions = REFERENCE.component_matrices(RECORD)
    rt = RECORD["source"]["gas_constant_J_mol_K"] * state["T_K"]
    mu0 = REFERENCE.source_standard_potentials(RECORD, case) / rt
    indices = [species.index(name + "_metal") for name in ("Fe", "Si", "O")]
    shift = REFERENCE.metal_standard_shift(state["T_K"])
    np.testing.assert_allclose(shift, state["mu0_formal_minus_source_over_RT"], atol=5e-12)
    source_mu = mu0[indices] + np.log(state["x"]) + state["ln_gamma_source"]
    formal_mu = mu0[indices] + shift + np.log(state["x"]) + state["ln_gamma_formal"]
    np.testing.assert_allclose(formal_mu, source_mu, atol=5e-12, rtol=5e-12)
    # A standard-only change is consequential for interphase reactions.
    assert np.max(np.abs(reactions[:, indices] @ shift)) > 1.0
    np.testing.assert_allclose(reactions[:, indices] @ (formal_mu - source_mu), 0.0, atol=5e-12)


def test_absolute_component_amounts_count_each_phase_and_trace_element() -> None:
    species, _, _ = REFERENCE.component_matrices(RECORD)
    amounts = np.zeros(len(species))
    for name, value in {
        "MgSiO3_silicate": 2, "FeO_silicate": 3,
        "Fe_metal": 4, "Si_metal": 0.5, "O_metal": 0.25, "H_metal": 0.125,
        "H2_gas": 7, "CO2_gas": 0.25, "Na_gas": 1e-12,
    }.items():
        amounts[species.index(name)] = value
    expected = {
        "silicate": [2, 2, 9, 3, 0, 0, 0],
        "metal": [0.5, 0, 0.25, 4, 0.125, 0, 0],
        "gas": [0, 0, 0.5, 0, 14, 1e-12, 0.25],
    }
    for scale in (1.0, 1e-6, 1e6):
        contributions = REFERENCE.phase_element_amounts(RECORD, scale * amounts)
        for phase, element_amounts in expected.items():
            np.testing.assert_allclose(contributions[phase], scale * np.asarray(element_amounts), atol=0, rtol=1e-14)
    for value in REFERENCE.phase_element_amounts(RECORD, np.zeros(25)).values():
        np.testing.assert_array_equal(value, np.zeros(7))


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_amounts_are_rejected(value) -> None:
    amounts = np.ones(25)
    amounts[0] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        REFERENCE.phase_element_amounts(RECORD, amounts)


def test_invalid_amount_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        REFERENCE.phase_element_amounts(RECORD, np.ones(24))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_standard_conversion_temperature_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        REFERENCE.metal_standard_shift(value)


def test_audit_reports_thermochemistry_without_claiming_equilibrium() -> None:
    report = REFERENCE.audit_reference(RECORD)
    assert report["component_count"] == 25
    assert len(report["cases"]) == 2
    assert "no equilibrium solve" in report["status"]
