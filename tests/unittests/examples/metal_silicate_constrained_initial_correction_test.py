"""CPU-only analytic active-face guesses preserve atoms and original scalar ownership."""

import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    names = ("local", "full_potential", "common_gibbs", "constrained_initial_correction")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded = {name: importlib.import_module(name) for name in names}
    yield SimpleNamespace(local=loaded["local"], full=loaded["full_potential"], scalar=loaded["common_gibbs"],
                          helper=loaded["constrained_initial_correction"])
    for name in names:
        sys.modules.pop(name, None)


def toy_case(modules, scale=1.):
    phases = {"silicate": ["Fe_l", "P_l"], "metal": ["Fe_m", "P_m", "Ca_m"], "excluded": ["C_g"]}
    record = {"elements": ["Fe", "P", "Ca", "C"], "phases": phases,
              "component_formulas": {name: {name.split("_")[0]: 1.} for names in phases.values() for name in names},
              "reactions": []}
    initial = np.array([1., .3, .98 - 1e-12, .02, 1e-12, 0.]) * scale
    target = np.array([1.49, .31, .49 - 1e-12, .01, 1e-12, 0.]) * scale
    budget = np.array([1.98 - 1e-12, .32, 1e-12, 0.]) * scale
    problem = modules.local.build_problem(record, budget, lambda t, p: np.zeros(6), phases=("silicate", "metal"))
    bounds = {"metal": (np.array([.75, 0., 0.]), np.array([1., .02, 1.]))}
    # An ideal constrained minimum at P_m/metal_total=.02 has a nonzero
    # inequality multiplier. Ordinary unconstrained mu equations are wrong.
    desired_mu = np.array([0., 0., .04, -1.96, .04])
    callbacks, seen = {}, []
    for name, section in zip(problem.phases, problem.phase_slices):
        standard = desired_mu[section] - np.log(target[section] / target[section].sum())
        def callback(temperature, pressure, amounts, *, standard=standard, name=name):
            seen.append((name, temperature, pressure, amounts.copy()))
            mu = standard + np.log(amounts / amounts.sum())
            return SimpleNamespace(mu_rt=mu, gibbs_rt=float(amounts @ mu))
        callback.energy_value_and_grad_rt = modules.full.ideal_phase(
            lambda t, p, standard=standard: standard).energy_value_and_grad_rt
        callbacks[name] = callback
    return SimpleNamespace(problem=problem, budget=budget, initial=initial, target=target,
                           bounds=bounds, callbacks=callbacks, seen=seen)


def correct(modules, case, **kwargs):
    return modules.helper.correct_initial_amounts(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=case.initial, phase_composition_bounds=case.bounds, **kwargs)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e26])
def test_active_face_correction_closes_scaled_atoms_without_clipping_trace_or_simplex(modules, scale):
    case = toy_case(modules, scale)
    original = case.initial.copy()
    result = correct(modules, case)
    assert result.adopted, result.diagnostics
    np.testing.assert_allclose(result.component_amounts_mol / scale, case.target / scale, rtol=2e-8, atol=1e-14)
    np.testing.assert_allclose(np.asarray(case.problem.full_formula_matrix) @ result.component_amounts_mol,
                               case.budget, rtol=1e-9, atol=0)
    assert result.component_amounts_mol[4] > 0 and result.component_amounts_mol[5] == 0
    assert result.diagnostics["active_constraints"] == [{"phase": "metal", "component": "P_m", "bound": "upper", "value": .02}]
    assert result.diagnostics["energy_nonincreasing"] and result.diagnostics["equations_closed"]
    assert result.diagnostics["nfev"] <= 12 and result.diagnostics["residual_evaluations"] >= result.diagnostics["nfev"]
    assert all(temperature == 3000. and pressure == 60. for _, temperature, pressure, _ in case.seen)
    np.testing.assert_array_equal(case.initial, original)


def test_correction_is_only_a_guess_and_the_original_scalar_still_decides(modules):
    case = toy_case(modules)
    result = correct(modules, case)
    assert result.adopted
    scalar = modules.scalar.minimize_gibbs(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=result.component_amounts_mol, phase_composition_bounds=case.bounds, maxiter=100)
    assert scalar.accepted, scalar.audit_reasons
    np.testing.assert_allclose(scalar.component_amounts_mol, case.target, rtol=3e-8, atol=1e-14)
    assert "accepted" not in result._fields


def test_no_explicit_active_face_skips_without_evaluating_callbacks(modules):
    case = toy_case(modules)
    case.bounds["metal"][1][1] = .03
    result = correct(modules, case)
    assert not result.adopted and result.diagnostics["stage"] == "skipped"
    assert result.diagnostics["callback_evaluations"] == 0 and case.seen == []
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)


@pytest.mark.parametrize("kind", ["atoms", "domain", "zero_trace", "excluded"])
def test_invalid_input_donor_is_rejected_before_any_callbacks(modules, kind):
    case = toy_case(modules)
    if kind == "atoms":
        case.initial[0] *= 1. + 2.48e-6
    elif kind == "domain":
        case.initial[3] += .001
        case.initial[1] -= .001
    elif kind == "zero_trace":
        case.initial[4] = 0.
    else:
        case.initial[5] = 1e-20
    with pytest.raises(ValueError, match="ledger"):
        correct(modules, case)
    assert case.seen == []


def test_callback_failure_is_recorded_and_original_ledger_returned(modules):
    case = toy_case(modules)
    def invalid(*args):
        raise ValueError("native activity unavailable")
    case.callbacks["metal"] = invalid
    result = correct(modules, case)
    assert not result.adopted
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)
    assert result.diagnostics["invalid_callbacks"] == [{"stage": "initial_energy", "phase": "metal",
        "residual_evaluation": 0, "error": "ValueError: native activity unavailable"}]


def test_bounded_nonconvergence_keeps_the_valid_original_ledger(modules):
    case = toy_case(modules)
    result = correct(modules, case, max_nfev=1)
    assert not result.adopted
    assert result.diagnostics["nfev"] == 1
    assert not result.diagnostics["equations_closed"]
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)


def test_even_tiny_energy_increase_cannot_be_adopted(modules, monkeypatch):
    case = toy_case(modules)
    case.initial = case.target.copy()
    counts = {name: 0 for name in case.callbacks}
    for name, original in list(case.callbacks.items()):
        def callback(t, p, n, *, name=name, original=original):
            state = original(t, p, n)
            counts[name] += 1
            if counts[name] > 1:
                # A common elemental gauge leaves tangent stationarity intact,
                # but fresh final energy must still not increase at all.
                shift = 1e-13
                return SimpleNamespace(mu_rt=state.mu_rt + shift,
                                       gibbs_rt=state.gibbs_rt + shift * n.sum())
            return state
        case.callbacks[name] = callback
    monkeypatch.setattr(modules.helper, "least_squares", lambda *args, **kwargs:
        SimpleNamespace(x=np.log(case.initial[:5] / case.budget.sum()), nfev=1,
                        success=True, message="synthetic current-root candidate"))
    result = correct(modules, case)
    assert result.diagnostics["equations_closed"] and result.diagnostics["inequalities_passed"]
    assert not result.adopted and result.diagnostics["energy_nonincreasing"] is False
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)


def test_helper_load_does_not_import_a_physical_example_checkout(monkeypatch):
    import scipy.linalg
    import scipy.optimize
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ("common_gibbs", "full_potential", "local"):
            pytest.fail("The separately loaded helper imported provider examples before source initialization")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("standalone_constrained_correction", DIRECTORY / "constrained_initial_correction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.correct_initial_amounts)


def test_all_inequalities_are_checked_even_when_only_one_face_was_active(modules, monkeypatch):
    case = toy_case(modules)
    candidate = np.array([case.budget[0] - .47e-12, case.budget[1] - .03e-12,
                          .47e-12, .03e-12, 1e-12])
    monkeypatch.setattr(modules.helper, "least_squares", lambda *args, **kwargs:
        SimpleNamespace(x=np.log(candidate / case.budget.sum()), nfev=1,
                        success=True, message="synthetic out-of-domain candidate"))
    result = correct(modules, case)
    assert result.diagnostics["final_max_relative_atom_residual"] < 1e-9
    assert result.diagnostics["inequalities_passed"] is False
    assert not result.adopted
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)


def test_invalid_callback_during_correction_keeps_stage_and_actual_counts(modules):
    case = toy_case(modules)
    original = case.callbacks["metal"]
    calls = 0
    def callback(t, p, n):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("invalid trial composition")
        return original(t, p, n)
    case.callbacks["metal"] = callback
    result = correct(modules, case)
    assert not result.adopted
    assert result.diagnostics["solver_result_returned"] is False
    assert result.diagnostics["residual_evaluations"] == 1
    assert result.diagnostics["callback_evaluations"] == 4
    assert result.diagnostics["invalid_callbacks"][0]["stage"] == "correction"
    np.testing.assert_array_equal(result.component_amounts_mol, case.initial)
