"""Finite sulfur closure and empirical phase-branch regression controls."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "sulfide.py"
SPEC = importlib.util.spec_from_file_location("metal_silicate_sulfide", PATH)
SULFIDE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SULFIDE
SPEC.loader.exec_module(SULFIDE)


def test_scss_converts_speciation_and_declared_mass_denominators():
    _, amounts, _, saturation = SULFIDE.analytic_control()
    scss = saturation(1873, 1, amounts)
    dry_sulfide = SULFIDE.melt_sulfur_ppm(amounts, sulfur_state="sulfide", mass_basis="dry")
    dry_total = SULFIDE.melt_sulfur_ppm(amounts, sulfur_state="total", mass_basis="dry")
    hydrous_total = SULFIDE.melt_sulfur_ppm(amounts, sulfur_state="total", mass_basis="hydrous")
    sulfate_fraction = amounts[9] / (amounts[8] + amounts[9])
    assert SULFIDE.scss_total_ppm(scss, sulfate_fraction) == pytest.approx(dry_total)
    assert dry_total == pytest.approx(dry_sulfide * 1.25)
    assert hydrous_total < dry_total
    dry_mass = amounts[[3, 4, 5, 8, 9]] @ SULFIDE.MOLAR_MASS_G[[3, 4, 5, 8, 9]]
    water_mass = amounts[[6, 7]] @ SULFIDE.MOLAR_MASS_G[[6, 7]]
    assert hydrous_total == pytest.approx(dry_total * dry_mass / (dry_mass + water_mass))
    with pytest.raises(ValueError, match="nonzero sulfide fraction"):
        SULFIDE.scss_total_ppm(scss, 1.0)


def test_known_saturated_state_multistart_and_independent_acceptance():
    budgets, expected, partition, saturation = SULFIDE.analytic_control()
    np.testing.assert_array_equal(SULFIDE.FORMULA @ SULFIDE.REACTIONS.T, 0)
    assert np.linalg.matrix_rank(SULFIDE.REACTIONS) == 7
    states = SULFIDE.solve_branches(1873, 1, budgets, partition, saturation, initial_amounts=(expected * 0.4, expected * 1.7))
    assert len(states) == 4
    for state in states:
        amounts = state.component_amounts_mol
        np.testing.assert_allclose(SULFIDE.FORMULA @ amounts, budgets, rtol=1e-9, atol=0)
        np.testing.assert_allclose(partition(1873, 1, amounts), 0, atol=1e-8)
        if state.branch == "present":
            assert state.accepted
            np.testing.assert_allclose(amounts, expected, rtol=1e-9, atol=1e-12)
            assert abs(state.saturation_log_ratio) < 1e-8
            # Each mole of separate FeS removes precisely one mole Fe and S.
            reduced = SULFIDE.FORMULA[:, :-1] @ amounts[:-1]
            np.testing.assert_allclose(budgets - reduced, [0, 0, 0.15, 0, 0, 0.15], atol=1e-12)
        else:
            assert not state.accepted
            assert amounts[-1] == 0
            assert state.saturation_log_ratio > 0
    corrupted = expected.copy()
    corrupted[-1] += 0.01
    checked = SULFIDE.evaluate(corrupted, budgets, 1873, 1, partition, saturation, branch="present")
    assert not checked.accepted


def test_zero_sulfur_removes_all_s_components_and_s_reactions_exactly():
    budgets, expected, partition, saturation = SULFIDE.analytic_control()
    budgets[-1] = 0
    states = SULFIDE.solve_branches(1873, 1, budgets, partition, saturation, initial_amounts=(expected,))
    assert len(states) == 1
    state = states[0]
    assert state.accepted
    assert state.branch == "absent"
    assert state.reaction_residual.shape == (4,)
    assert state.saturation_log_ratio == -np.inf
    np.testing.assert_array_equal(state.component_amounts_mol[SULFIDE.FORMULA[-1] > 0], 0)
    np.testing.assert_allclose(SULFIDE.FORMULA @ state.component_amounts_mol, budgets, atol=1e-10)


def test_tiny_positive_sulfur_budget_remains_finite_without_a_trace_floor():
    budgets, _, partition, saturation = SULFIDE.analytic_control()
    budgets[-1] = 1e-50
    states = SULFIDE.solve_branches(1873, 1, budgets, partition, saturation)
    accepted = [state for state in states if state.accepted]
    assert len(accepted) == 1
    state = accepted[0]
    assert state.branch == "absent"
    assert state.component_amounts_mol[-1] == 0
    assert np.all(state.component_amounts_mol[[2, 8, 9, 12]] > 0)
    np.testing.assert_allclose(SULFIDE.FORMULA @ state.component_amounts_mol,
                               budgets, rtol=1e-9, atol=0)
    np.testing.assert_allclose(state.reaction_residual, 0, atol=1e-8)


@pytest.mark.parametrize("temperature,pressure", [
    (-1.0, 1.0), (0.0, 1.0), (np.nan, 1.0), (np.inf, 1.0),
    (1873.0, -1.0), (1873.0, 0.0), (1873.0, np.nan), (1873.0, np.inf),
])
def test_independent_acceptance_rejects_invalid_temperature_or_pressure(temperature, pressure):
    budgets, known, partition, saturation = SULFIDE.analytic_control()
    with pytest.raises(ValueError, match="Temperature in K and pressure in bar"):
        SULFIDE.evaluate(known, budgets, temperature, pressure, partition, saturation, branch="present")


def test_forward_backward_continuation_crosses_exact_phase_disappearance():
    budgets, known, partition, saturation = SULFIDE.analytic_control()
    sulfur = [0.0, 0.03, 0.08, 0.14, budgets[-1]]
    forward = SULFIDE.continue_sulfur(sulfur, budgets, 1873, 1, partition, saturation, initial_amounts=(known * 0.7,))
    backward = SULFIDE.continue_sulfur(sulfur[::-1], budgets, 1873, 1, partition, saturation, initial_amounts=(known * 1.3,))[::-1]
    phases = []
    for left, right in zip(forward, backward):
        accepted_left = [state for state in left if state.accepted]
        accepted_right = [state for state in right if state.accepted]
        assert accepted_left and accepted_right
        phases.append(accepted_left[0].branch)
        for state in accepted_left + accepted_right:
            np.testing.assert_allclose(state.component_amounts_mol, accepted_left[0].component_amounts_mol, rtol=1e-7, atol=1e-11)
            if state.branch == "absent":
                assert state.component_amounts_mol[-1] == 0
    assert phases[0] == "absent"
    assert phases[-1] == "present"
    assert any(state.component_amounts_mol[-1] < 0 and not state.accepted for state in forward[1] if state.branch == "present")


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_amount_scaling_preserves_partition_and_boundary(scale):
    budgets, expected, partition, saturation = SULFIDE.analytic_control()
    states = SULFIDE.solve_branches(1873, 1, budgets * scale, partition, saturation, initial_amounts=(expected * scale * 1.2,))
    accepted = [state for state in states if state.accepted]
    assert len(accepted) == 1
    np.testing.assert_allclose(accepted[0].component_amounts_mol / scale, expected, rtol=1e-8, atol=1e-11)


def test_scss_is_evaluated_at_final_composition_on_its_own_basis():
    budgets, expected, partition, saturation = SULFIDE.analytic_control()
    base = saturation(1873, 1, expected)
    calls = []

    def equivalent_total_hydrous(temperature, pressure, amounts):
        calls.append(amounts.copy())
        dry_mass = SULFIDE.MOLAR_MASS_G[[3, 4, 5, 8, 9]] @ amounts[[3, 4, 5, 8, 9]]
        hydrous_mass = SULFIDE.MOLAR_MASS_G[3:10] @ amounts[3:10]
        sulfate = amounts[9] / (amounts[8] + amounts[9])
        threshold = SULFIDE.scss_total_ppm(base, sulfate) * dry_mass / hydrous_mass
        return SULFIDE.SCSS(threshold, "total", "hydrous", "equivalent_analytic_control", "conditional")

    states = SULFIDE.solve_branches(1873, 1, budgets, partition, equivalent_total_hydrous, initial_amounts=(expected * 1.2,))
    state = next(state for state in states if state.accepted)
    np.testing.assert_allclose(state.component_amounts_mol, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(calls[-1], state.component_amounts_mol)


@pytest.mark.parametrize("field,value", [("sulfur_state", "sulfate"), ("mass_basis", "mole_fraction"), ("ppm_s", 0), ("phase_state", "unknown")])
def test_invalid_scss_metadata_is_rejected(field, value):
    kwargs = dict(ppm_s=1000, sulfur_state="sulfide", mass_basis="dry", calibration_id="test", phase_state="conditional")
    kwargs[field] = value
    with pytest.raises(ValueError):
        SULFIDE.SCSS(**kwargs)
