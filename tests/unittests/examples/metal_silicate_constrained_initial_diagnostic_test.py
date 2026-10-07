"""Analytic CPU diagnostics never call a minimizer or accept a new ledger."""

import builtins
from functools import lru_cache
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
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    names = ("local", "full_potential", "common_gibbs", "constrained_initial_diagnostic")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    modules = {name: importlib.import_module(name) for name in names}
    def forbidden(*args, **kwargs):
        pytest.fail("A Jacobian diagnostic must not run an equilibrium minimizer")
    monkeypatch.setattr(modules["common_gibbs"], "minimize_gibbs", forbidden)
    monkeypatch.setattr(modules["common_gibbs"], "least_squares", forbidden)
    monkeypatch.setattr(modules["common_gibbs"], "minimize", forbidden)
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "least_squares", forbidden)
    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    yield modules["constrained_initial_diagnostic"]
    for name in names:
        sys.modules.pop(name, None)


def toy_case(scale=1.):
    names = ["Fe_l", "P_l", "Fe_m", "P_m"]
    formula = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    initial = np.array([1., .3, .98, .02]) * scale
    problem = SimpleNamespace(
        reaction_offset=None, elements=("Fe", "P"), element_indices=np.arange(2),
        species_indices=np.arange(4), full_species=names, species=names,
        formula_matrix=formula, phases=("silicate", "metal"), phase_slices=(slice(0, 2), slice(2, 4)))
    calls = []
    def ideal(t, p, n):
        calls.append((t, p, n.copy()))
        mu = np.log(n / n.sum())
        return SimpleNamespace(mu_rt=mu, gibbs_rt=float(n @ mu))
    bounds = {"metal": (np.array([.75, 0.]), np.array([1., .02]))}
    return SimpleNamespace(problem=problem, budget=formula @ initial, initial=initial,
                           callbacks={"silicate": ideal, "metal": ideal}, bounds=bounds, calls=calls)


def run(diagnostic, case, **kwargs):
    return diagnostic.diagnose_initialization(case.problem, 3000., 60., case.budget, case.callbacks,
        initial_component_amounts_mol=case.initial, phase_composition_bounds=case.bounds, **kwargs)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e26])
def test_noise_free_jacobians_match_independent_analytic_derivatives(diagnostic, scale):
    case = toy_case(scale)
    original = case.initial.copy()
    events = []
    report = run(diagnostic, case, progress=events.append)
    assert report["completed"], report
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(case.initial, original)
    assert report["residual_evaluations"] == 16  # 1 + 3 * (4 + 1)
    assert report["callback_evaluations"] == 32 == len(case.calls)
    assert all(value["calls"] == 16 for value in report["phase_evaluations"].values())
    assert len([event for event in events if event["event"] == "column"]) == 12
    assert len([event for event in events if event["event"] == "base"]) == 4
    assert events[-1]["event"] == "complete"
    assert all(t == 3000. and p == 60. for t, p, n in case.calls)
    assert all(np.all(n > 0) for t, p, n in case.calls)
    for repeat in report["base_repeats"][1:]:
        assert repeat["mu_drift_max_abs"] == repeat["residual_drift_max_abs"] == 0.
    z = case.initial / case.budget.sum()
    atom_jac = case.problem.formula_matrix * case.initial / case.budget[:, None]
    face_jac = np.array([0., 0., .0196, -.0196])
    derivative = np.zeros((4, 4))
    for section in case.problem.phase_slices:
        fraction = z[section] / z[section].sum()
        derivative[section, section] = np.eye(2) - fraction[None, :]
    reaction = np.asarray(report["reaction_basis"])
    exact = np.vstack([atom_jac, face_jac, reaction @ derivative])
    np.testing.assert_allclose(report["analytic_atom_face_jacobian"], exact[:3], rtol=1e-14, atol=1e-16)
    for grid, tolerance in zip(report["grids"], [2e-7, 6e-5, 6e-4]):
        np.testing.assert_allclose(grid["jacobian"], exact, rtol=0, atol=tolerance)
        assert grid["rank"] == 4
        assert grid["analytic_atom_face_error_max_abs"] < tolerance
        assert grid["common_scale_derivative_error_max_abs"] < 2 * tolerance
        assert grid["linear_newton_probe"]["predicted_residual_max_abs"] < 1e-12
    assert "accepted" not in report and "component_amounts_mol" not in report


def test_cached_nested_noise_is_observed_after_composition_cache_eviction(diagnostic):
    case = toy_case()
    unique_evaluations = []
    @lru_cache(maxsize=1)
    def nested(amount_tuple):
        n = np.asarray(amount_tuple)
        unique_evaluations.append(n.copy())
        mu = np.log(n / n.sum())
        # Synthetic nested convergence error, with an exact one-entry cache.
        mu[0] += 2e-9 * len(unique_evaluations)
        return SimpleNamespace(mu_rt=mu, gibbs_rt=float(n @ mu))
    def callback(t, p, n):
        return nested(tuple(n))
    case.callbacks["metal"] = callback
    report = run(diagnostic, case)
    assert report["completed"], report
    assert len(unique_evaluations) == 10  # initial plus two displaced columns/base per grid
    assert all(base["mu_drift_max_abs"] > 0 for base in report["base_repeats"][1:])
    default_difference, large_difference = report["comparisons"][:2]
    assert default_difference["difference_frobenius_norm"] > 20 * large_difference["difference_frobenius_norm"]
    assert not np.array_equal(unique_evaluations[-2], unique_evaluations[-1])
    np.testing.assert_allclose(unique_evaluations[-1], case.initial[2:], rtol=1e-15)


def test_singular_spectrum_identifies_the_weak_component(diagnostic):
    report = diagnostic._matrix_report(np.diag([1., 1e-7, 1e-14]), ["bulk", "minor", "trace"])
    assert report["condition_number"] == pytest.approx(1e14)
    assert report["weak_directions"][0]["largest_components"][0]["component"] == "trace"
    deficient = diagnostic._matrix_report(np.diag([1., 0.]), ["bulk", "free"])
    assert deficient["rank"] == 1 and deficient["condition_number"] is None
    json.dumps(deficient, allow_nan=False)


@pytest.mark.parametrize("kind", ["atoms", "domain", "zero", "all_zero", "nan", "shape"])
def test_invalid_initial_input_fails_before_callbacks(diagnostic, kind):
    case = toy_case()
    if kind == "atoms":
        case.initial[0] *= 1.01
    elif kind == "domain":
        case.initial[3] += .001
        case.initial[1] -= .001
    elif kind == "zero":
        case.initial[3] = 0.
    elif kind == "all_zero":
        case.initial[:] = 0.
        case.budget[:] = 0.
    elif kind == "nan":
        case.initial[0] = np.nan
    else:
        case.initial = case.initial[:-1]
    with pytest.raises(ValueError):
        run(diagnostic, case)
    assert not case.calls


def test_provider_failure_preserves_actual_probe_amounts_and_partial_progress(diagnostic):
    case = toy_case()
    original = case.callbacks["metal"]
    observed = []
    def failing(t, p, n):
        observed.append(n.copy())
        if len(observed) == 4:
            raise ValueError("synthetic provider failure")
        return original(t, p, n)
    case.callbacks["metal"] = failing
    events = []
    report = run(diagnostic, case, progress=events.append)
    assert not report["completed"]
    assert report["residual_evaluations"] == 4 and report["callback_evaluations"] == 8
    assert report["grids"][0]["completed_columns"] == 2
    failure = report["invalid_callbacks"][0]
    np.testing.assert_array_equal(failure["phase_component_amounts_mol"], observed[-1])
    assert not np.array_equal(observed[-1], observed[0])
    assert failure["phase"] == "metal" and failure["residual_evaluation"] == 4
    assert events[-1]["event"] == "failure"
    json.dumps(report, allow_nan=False)


def test_probe_outside_declared_cap_is_identified_as_derivative_only(diagnostic):
    case = toy_case()
    report = run(diagnostic, case)
    assert report["completed"]
    assert any(n[1] / n.sum() > .02 + 1e-8 for _, _, n in case.calls if n[0] < 1.)
    assert "Derivative probes only" in report["scope"]


def test_malformed_provider_result_is_an_explicit_failure(diagnostic):
    case = toy_case()
    case.callbacks["metal"] = lambda *args: None
    report = run(diagnostic, case)
    assert not report["completed"]
    assert report["invalid_callbacks"][0]["error"].startswith("AttributeError:")
    assert report["phase_evaluations"]["metal"]["calls"] == 1
    json.dumps(report, allow_nan=False)


def test_no_active_face_skips_without_provider_calls(diagnostic):
    case = toy_case()
    case.bounds["metal"][1][1] = .03
    report = run(diagnostic, case)
    assert report["stage"] == "skipped" and not report["completed"]
    assert report["residual_evaluations"] == 0 and case.calls == []


def test_module_import_remains_independent_of_physical_checkout(monkeypatch):
    import scipy.linalg
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ("common_gibbs", "full_potential", "local"):
            pytest.fail("Physical modules must be loaded by the caller before runtime imports")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("standalone_diagnostic", DIRECTORY / "constrained_initial_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.diagnose_initialization)
