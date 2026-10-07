"""Analytic CPU descent tests preserve inventory, support and scalar ownership."""

import builtins
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    names = ("local", "full_potential", "common_gibbs", "feasible_initial_descent")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded = {name: importlib.import_module(name) for name in names}
    yield SimpleNamespace(local=loaded["local"], full=loaded["full_potential"],
                          scalar=loaded["common_gibbs"], helper=loaded["feasible_initial_descent"])
    for name in names:
        sys.modules.pop(name, None)


def toy_case(modules, scale=1., gauge=None):
    phases = {"silicate": ["Fe_l", "P_l"], "metal": ["Fe_m", "P_m", "Ca_m"], "excluded": ["C_g"]}
    record = {"elements": ["Fe", "P", "Ca", "C"], "phases": phases,
              "component_formulas": {name: {name.split("_")[0]: 1.} for phase in phases.values() for name in phase},
              "reactions": []}
    initial = np.array([1., .3, .98 - 1e-12, .02, 1e-12, 0.]) * scale
    target = np.array([1.49, .31, .49 - 1e-12, .01, 1e-12, 0.]) * scale
    budget = np.array([1.98 - 1e-12, .32, 1e-12, 0.]) * scale
    problem = modules.local.build_problem(record, budget, lambda t, p: np.zeros(6), phases=("silicate", "metal"))
    bounds = {"metal": (np.array([.75, 0., 0.]), np.array([1., .02, 1.]))}
    desired_mu = np.array([0., 0., .04, -1.96, .04])
    if gauge is not None:
        desired_mu += problem.formula_matrix.T @ np.asarray(gauge)
    callbacks, seen = {}, []
    for phase, section in zip(problem.phases, problem.phase_slices):
        standard = desired_mu[section] - np.log(target[section] / target[section].sum())
        def callback(temperature, pressure, amounts, *, standard=standard, phase=phase):
            seen.append((phase, temperature, pressure, amounts.copy()))
            mu = standard + np.log(amounts / amounts.sum())
            return SimpleNamespace(mu_rt=mu, gibbs_rt=float(amounts @ mu))
        callback.energy_value_and_grad_rt = modules.full.ideal_phase(
            lambda t, p, standard=standard: standard).energy_value_and_grad_rt
        callbacks[phase] = callback
    return SimpleNamespace(problem=problem, budget=budget, initial=initial, target=target,
                           bounds=bounds, callbacks=callbacks, seen=seen)


def run(modules, case, **kwargs):
    return modules.helper.descend_initial_amounts(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=case.initial, phase_composition_bounds=case.bounds, **kwargs)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e26])
def test_all_evaluated_ledgers_conserve_atoms_active_face_and_positive_support(modules, scale):
    case = toy_case(modules, scale)
    initial = case.initial.copy()
    events = []
    result = run(modules, case, progress=events.append)
    assert result.adopted, result.diagnostics
    report = result.diagnostics
    assert report["final_gibbs_rt"] < report["initial_gibbs_rt"]
    np.testing.assert_allclose(result.component_amounts_mol / scale, case.target / scale, rtol=1e-4, atol=1e-14)
    np.testing.assert_array_equal(case.initial, initial)
    assert result.component_amounts_mol[4] > 0 and result.component_amounts_mol[5] == 0
    assert report["callback_evaluations"] == len(case.seen) == 2 * report["evaluations"]
    for left, right in zip(case.seen[::2], case.seen[1::2]):
        ledger = np.r_[left[3], right[3]]
        np.testing.assert_allclose(case.problem.formula_matrix @ ledger, case.budget[:3], rtol=2e-14, atol=0)
        assert np.all(ledger > 0)
        assert abs(ledger[3] / ledger[2:].sum() - .02) < 1e-14
        assert ledger[2] / ledger[2:].sum() >= .75 - 1e-14
        assert left[1:3] == right[1:3] == (3000., 60.)
    assert events[-1]["event"] == "complete"
    assert len([e for e in events if e["event"] == "iteration"]) == report["accepted_steps"]
    energies = [h["gibbs_rt_per_inventory_atom"] for h in report["history"]]
    assert all(right < left for left, right in zip(energies, energies[1:]))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("allow_interior", [False, True])
def test_original_scalar_can_reject_an_adopted_partial_descent(modules, allow_interior):
    case = toy_case(modules)
    if allow_interior:
        case.bounds["metal"][1][1] = .021
    result = run(modules, case, max_iterations=1, allow_interior=allow_interior)
    assert result.adopted and result.diagnostics["final_stationarity_max_abs"] > 1e-8
    assert "accepted" not in result._fields
    scalar = modules.scalar.minimize_gibbs(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=result.component_amounts_mol, phase_composition_bounds=case.bounds, maxiter=1)
    assert not scalar.accepted
    assert "scalar minimizer did not converge" in scalar.audit_reasons


def test_element_potential_gauge_does_not_change_descent(modules):
    ordinary = run(modules, toy_case(modules), max_iterations=5)
    shifted = run(modules, toy_case(modules, gauge=[17., -31., 23.]), max_iterations=5)
    assert ordinary.adopted and shifted.adopted
    np.testing.assert_allclose(ordinary.component_amounts_mol, shifted.component_amounts_mol, rtol=2e-12, atol=1e-15)
    assert ordinary.diagnostics["accepted_steps"] == shifted.diagnostics["accepted_steps"] == 5


@pytest.mark.parametrize("kind", ["atoms", "domain", "zero_trace", "excluded", "complex", "nan"])
@pytest.mark.parametrize("allow_interior", [False, True])
def test_invalid_initial_input_has_no_callback_side_effect(modules, kind, allow_interior):
    case = toy_case(modules)
    if kind == "atoms":
        case.initial[0] *= 1.01
    elif kind == "domain":
        case.initial[3] += .001
        case.initial[1] -= .001
    elif kind == "zero_trace":
        case.initial[4] = 0.
    elif kind == "excluded":
        case.initial[5] = 1e-20
    elif kind == "complex":
        case.initial = case.initial.astype(complex)
    else:
        case.initial[0] = np.nan
    with pytest.raises(ValueError, match="ledger"):
        run(modules, case, allow_interior=allow_interior)
    assert not case.seen


def test_no_active_metal_face_skips_without_callbacks(modules):
    case = toy_case(modules)
    case.bounds["metal"][1][1] = .03
    result = run(modules, case)
    assert not result.adopted and result.diagnostics["stage"] == "skipped"
    assert result.diagnostics["evaluations"] == 0 and not case.seen


@pytest.mark.parametrize("scale", [1e-6, 1., 1e26])
def test_opt_in_interior_uses_current_bound_and_preserves_every_evaluated_ledger(modules, scale):
    case = toy_case(modules, scale)
    # The donor's old .02 face is interior to the supplied .021 domain.
    case.bounds["metal"][1][1] = .021
    result = run(modules, case, allow_interior=True)
    report = result.diagnostics
    assert result.adopted and report["allow_interior"]
    assert report["active_constraints"] == []
    assert report["tangent_rank"] == 3
    assert report["final_gibbs_rt"] < report["initial_gibbs_rt"]
    assert result.component_amounts_mol[5] == 0
    for left, right in zip(case.seen[::2], case.seen[1::2]):
        ledger = np.r_[left[3], right[3]]
        np.testing.assert_allclose(case.problem.formula_matrix @ ledger,
                                   case.budget[:3], rtol=2e-14, atol=0)
        assert np.all(ledger > 0)
        assert ledger[3] / ledger[2:].sum() <= .021
        assert ledger[2] / ledger[2:].sum() >= .75
    final_metal = result.component_amounts_mol[2:5]
    assert .02099 < final_metal[1] / final_metal.sum() <= .021
    energies = [report["initial_gibbs_rt"] / case.budget.sum()] + [
        row["gibbs_rt_per_inventory_atom"] for row in report["history"]]
    assert all(right < left for left, right in zip(energies, energies[1:]))
    assert all(row["max_relative_active_face_drift"] == 0 for row in report["history"])
    json.dumps(report, allow_nan=False)


def test_opt_in_leaves_existing_active_face_descent_identical(modules):
    ordinary = run(modules, toy_case(modules))
    enabled = run(modules, toy_case(modules), allow_interior=True)
    assert ordinary.adopted and enabled.adopted
    np.testing.assert_array_equal(ordinary.component_amounts_mol, enabled.component_amounts_mol)
    for key in ("history", "active_constraints", "evaluations", "final_gibbs_rt"):
        assert ordinary.diagnostics[key] == enabled.diagnostics[key]


@pytest.mark.parametrize("scale", [1e-6, 1., 1e26])
def test_reached_face_opt_in_continues_descent_inside_current_box(modules, scale):
    case = toy_case(modules, scale)
    case.bounds["metal"][1][1] = .021
    fixed = run(modules, case, allow_interior=True)
    case.seen.clear()
    events = []
    updated = run(modules, case, allow_interior=True, update_active_faces=True, progress=events.append)
    report = updated.diagnostics
    assert updated.adopted and report["update_active_faces"]
    assert report["final_gibbs_rt"] < fixed.diagnostics["final_gibbs_rt"]
    assert report["final_stationarity_max_abs"] < fixed.diagnostics["final_stationarity_max_abs"]
    assert len(report["activated_constraints"]) == 1
    added = report["activated_constraints"][0]
    assert 0 <= added["relative_slack"] < 1e-8
    assert added["constraint"] in report["active_constraints"]
    assert report["tangent_rank"] == 4
    assert report["accepted_steps"] > fixed.diagnostics["accepted_steps"]
    assert any(e["event"] == "active_faces_updated" for e in events)
    for left, right in zip(case.seen[::2], case.seen[1::2]):
        ledger = np.r_[left[3], right[3]]
        np.testing.assert_allclose(case.problem.formula_matrix @ ledger,
                                   case.budget[:3], rtol=2e-14, atol=0)
        assert np.all(ledger > 0)
        assert ledger[3] / ledger[2:].sum() <= .021
        assert ledger[2] / ledger[2:].sum() >= .75
    assert updated.component_amounts_mol[5] == 0
    assert all(h["max_relative_active_face_drift"] < 1e-10 for h in report["history"])
    energies = [report["initial_gibbs_rt"] / case.budget.sum()] + [
        h["gibbs_rt_per_inventory_atom"] for h in report["history"]]
    assert all(b < a for a, b in zip(energies, energies[1:]))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("allow_interior", [False, True])
def test_reached_face_default_keeps_original_trajectory(modules, allow_interior):
    case = toy_case(modules)
    if allow_interior:
        case.bounds["metal"][1][1] = .021
    ordinary = run(modules, case, allow_interior=allow_interior)
    explicit = run(modules, case, allow_interior=allow_interior, update_active_faces=False)
    np.testing.assert_array_equal(ordinary.component_amounts_mol, explicit.component_amounts_mol)
    for key in ("history", "active_constraints", "evaluations", "final_gibbs_rt"):
        assert ordinary.diagnostics[key] == explicit.diagnostics[key]
    assert explicit.diagnostics["activated_constraints"] == []


def test_reached_face_rows_are_not_passed_to_original_scalar(modules):
    case = toy_case(modules)
    case.bounds["metal"][1][1] = .021
    bounds = {phase: tuple(v.copy() for v in values) for phase, values in case.bounds.items()}
    result = run(modules, case, allow_interior=True, update_active_faces=True, max_iterations=8)
    assert result.adopted and result.diagnostics["activated_constraints"]
    for phase in bounds:
        for original, final in zip(bounds[phase], case.bounds[phase]):
            np.testing.assert_array_equal(original, final)
    scalar = modules.scalar.minimize_gibbs(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=result.component_amounts_mol,
        phase_composition_bounds=case.bounds, maxiter=1)
    assert not scalar.accepted
    assert "scalar minimizer did not converge" in scalar.audit_reasons


@pytest.mark.parametrize("flag", [None, 0, 1, "yes", np.bool_(True)])
def test_reached_face_updates_require_explicit_boolean(modules, flag):
    case = toy_case(modules)
    with pytest.raises(ValueError, match="update_active_faces"):
        run(modules, case, update_active_faces=flag)
    assert not case.seen


def test_interior_opt_in_does_not_admit_donor_outside_tighter_current_bound(modules):
    case = toy_case(modules)
    case.bounds["metal"][1][1] = .01
    with pytest.raises(ValueError, match="composition bound"):
        run(modules, case, allow_interior=True)
    assert not case.seen


def test_interior_opt_in_does_not_evaluate_metal_free_problem(modules):
    case = toy_case(modules)
    case.problem = replace(case.problem, phases=("silicate", "atmosphere"))
    case.callbacks["atmosphere"] = case.callbacks.pop("metal")
    case.bounds["atmosphere"] = case.bounds.pop("metal")
    result = run(modules, case, allow_interior=True)
    assert not result.adopted and result.diagnostics["stage"] == "skipped"
    assert result.diagnostics["evaluations"] == 0 and not case.seen


@pytest.mark.parametrize("flag", [None, 0, 1, "yes", np.bool_(True)])
def test_interior_permission_requires_explicit_boolean(modules, flag):
    case = toy_case(modules)
    with pytest.raises(ValueError, match="boolean allow_interior"):
        run(modules, case, allow_interior=flag)
    assert not case.seen


def test_invalid_initial_callback_fails_without_adopting(modules):
    case = toy_case(modules)
    case.callbacks["metal"] = lambda *args: None
    result = run(modules, case)
    assert not result.adopted
    assert result.diagnostics["evaluations"] == 1
    assert result.diagnostics["callback_evaluations"] == 2
    assert result.diagnostics["invalid_callbacks"][0]["stage"] == "initial_energy"
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)


def test_invalid_trial_is_rejected_then_smaller_trial_can_descend(modules):
    case = toy_case(modules)
    original = case.callbacks["metal"]
    calls = 0
    def fail_once(t, p, n):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic nested solve failure")
        return original(t, p, n)
    case.callbacks["metal"] = fail_once
    result = run(modules, case, max_iterations=1)
    assert result.adopted
    assert result.diagnostics["rejected_trials"] == 1
    assert result.diagnostics["evaluations"] == 3
    assert result.diagnostics["callback_evaluations"] == 6
    assert result.diagnostics["history"][0]["step"] == .5


def test_budget_exhaustion_retains_best_valid_candidate(modules):
    case = toy_case(modules)
    unchanged = run(modules, case, max_evaluations=1)
    assert not unchanged.adopted and unchanged.diagnostics["evaluations"] == 1
    result = run(modules, toy_case(modules), max_evaluations=2)
    assert result.adopted and result.diagnostics["evaluations"] == 2
    assert result.diagnostics["accepted_steps"] == 1
    assert "budget" in result.diagnostics["reason"]


def test_newly_approached_inactive_bound_limits_trials(modules):
    case = toy_case(modules)
    case.bounds["silicate"] = (np.zeros(2), np.array([.78, 1.]))
    result = run(modules, case)
    assert result.adopted
    silicate = [n for phase, _, _, n in case.seen if phase == "silicate"]
    assert all(n[0] / n.sum() <= .78 for n in silicate)
    assert silicate[-1][0] / silicate[-1].sum() > .779


def test_elapsed_time_budget_is_checked_between_provider_calls(modules, monkeypatch):
    case = toy_case(modules)
    clock = [0.]
    monkeypatch.setattr(modules.helper, "perf_counter", lambda: clock[0])
    original = case.callbacks["silicate"]
    def slow(t, p, n):
        value = original(t, p, n)
        clock[0] += 2.
        return value
    case.callbacks["silicate"] = slow
    result = run(modules, case, max_seconds=1.)
    assert not result.adopted
    assert result.diagnostics["evaluations"] == result.diagnostics["callback_evaluations"] == 1
    assert "between callbacks" in result.diagnostics["reason"]


@pytest.mark.parametrize("limits", [{"max_iterations": 257}, {"max_evaluations": 1025}, {"max_seconds": 1801.}])
def test_hard_caps_are_not_silently_expanded(modules, limits):
    case = toy_case(modules)
    with pytest.raises(ValueError):
        run(modules, case, **limits)
    assert not case.seen


def test_import_does_not_load_physical_checkout(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ("common_gibbs", "full_potential", "local"):
            pytest.fail("The caller must initialize its physical checkout first")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("standalone_feasible_descent", DIRECTORY / "feasible_initial_descent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.descend_initial_amounts)
