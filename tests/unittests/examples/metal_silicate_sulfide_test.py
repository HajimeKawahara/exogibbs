"""Finite sulfur closure and empirical phase-branch regression controls."""

import importlib.util
import json
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


def test_final_absent_domain_failure_preserves_valid_present_root_and_continuation():
    budgets, known, partition, saturation = SULFIDE.analytic_control()
    known_feo = known[5] / known[3:10].sum()
    absent_calls = []

    def bounded_saturation(temperature, pressure, amounts):
        if amounts[-1] == 0:
            absent_calls.append(amounts.copy())
            # SCSS must not reject an intermediate absent iteration.
            np.testing.assert_allclose(SULFIDE.FORMULA @ amounts, budgets, rtol=1e-9)
        if abs(amounts[5] / amounts[3:10].sum() - known_feo) > 1e-4:
            raise SULFIDE.SulfideDomainError("Outside explicit FeO composition domain")
        return saturation(temperature, pressure, amounts)

    assert SULFIDE.evaluate(known, budgets, 1873, 1, partition, bounded_saturation, branch="present").accepted
    sequence = SULFIDE.continue_sulfur(
        [budgets[-1], budgets[-1]], budgets, 1873, 1, partition,
        bounded_saturation, initial_amounts=(known,),
    )
    for states in sequence:
        for state in states:
            if state.branch == "absent":
                assert not state.accepted
                assert state.solver_success  # Convergence does not imply admissibility.
                assert state.failure_reason == "SulfideDomainError: Outside explicit FeO composition domain"
                assert state.scss is state.reaction_residual is state.saturation_log_ratio is None
                json.dumps({"failure_reason": state.failure_reason, "saturation_log_ratio": state.saturation_log_ratio}, allow_nan=False)
            else:
                assert state.accepted
                assert state.failure_reason is None
                np.testing.assert_allclose(state.component_amounts_mol, known, rtol=1e-9, atol=1e-12)
    assert len(absent_calls) == sum(state.branch == "absent" for states in sequence for state in states)


@pytest.mark.parametrize("failure", [
    SULFIDE.SulfideDomainError, FloatingPointError, OverflowError, np.linalg.LinAlgError,
])
def test_declared_domain_or_numerical_failure_preserves_later_starts(failure):
    budgets, known, partition, saturation = SULFIDE.analytic_control()

    def bounded_partition(temperature, pressure, amounts):
        if amounts[5] < 0.3:
            raise failure("Invalid FeO trial")
        return partition(temperature, pressure, amounts)

    states = SULFIDE.solve_branches(
        1873, 1, budgets, bounded_partition, saturation,
        initial_amounts=(known * 0.4, known),
    )
    assert [(state.start_index, state.branch) for state in states] == [
        (0, "absent"), (0, "present"), (1, "absent"), (1, "present"),
    ]
    for state in states[:2]:
        assert not state.accepted and not state.solver_success
        assert state.failure_reason == f"{failure.__name__}: Invalid FeO trial"
    assert states[-1].accepted
    np.testing.assert_allclose(states[-1].component_amounts_mol, known, rtol=1e-9, atol=1e-12)


def test_continuation_advances_after_every_attempt_fails():
    budgets, known, partition, saturation = SULFIDE.analytic_control()

    def sulfur_partition(temperature, pressure, amounts):
        if np.all(amounts[SULFIDE.FORMULA[-1] > 0] == 0):
            raise SULFIDE.SulfideDomainError("This calibration requires sulfur")
        return partition(temperature, pressure, amounts)

    failed, recovered = SULFIDE.continue_sulfur(
        [0.0, budgets[-1]], budgets, 1873, 1, sulfur_partition,
        saturation, initial_amounts=(known,),
    )
    assert len(failed) == 1 and not failed[0].accepted
    assert failed[0].failure_reason == "SulfideDomainError: This calibration requires sulfur"
    assert any(state.accepted and state.branch == "present" for state in recovered)


@pytest.mark.parametrize("failure", [ValueError, TypeError, RuntimeError])
def test_undeclared_callback_errors_propagate(failure):
    budgets, known, partition, _ = SULFIDE.analytic_control()

    def broken_saturation(temperature, pressure, amounts):
        raise failure("Malformed calibration")

    with pytest.raises(failure, match="Malformed calibration"):
        SULFIDE.solve_branches(1873, 1, budgets, partition, broken_saturation, initial_amounts=(known,))


def test_malformed_partition_residual_shape_propagates():
    budgets, known, _, saturation = SULFIDE.analytic_control()
    with pytest.raises(ValueError, match="seven residuals"):
        SULFIDE.solve_branches(1873, 1, budgets, lambda t, p, n: np.zeros(6), saturation, initial_amounts=(known,))


def test_nonfinite_partition_attempt_is_retained():
    budgets, known, partition, saturation = SULFIDE.analytic_control()

    def numerical_partition(temperature, pressure, amounts):
        return np.full(7, np.nan) if amounts[5] < 0.3 else partition(temperature, pressure, amounts)

    states = SULFIDE.solve_branches(
        1873, 1, budgets, numerical_partition, saturation,
        initial_amounts=(known * 0.4, known),
    )
    assert states[0].failure_reason == "FloatingPointError: Partition callback returned nonfinite active residuals."
    assert states[-1].accepted
