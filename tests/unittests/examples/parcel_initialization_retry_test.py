"""A bounded initialization retry cannot replace chemistry acceptance gates."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exogibbs.api.condensate import CondensateEquilibriumOptions, CondensateEquilibriumResult


PATH = Path(__file__).resolve().parents[3] / "examples/metal_silicate/parcel_initialization_retry.py"
SPEC = importlib.util.spec_from_file_location("parcel_retry_test_helper", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


@pytest.fixture
def setup():
    return SimpleNamespace(
        elements=("H", "He"), gas_species=("H", "He"), condensate_species=("H(s)", "He(s)"),
        gas_setup=SimpleNamespace(formula_matrix=np.eye(2)),
        condensate_setup=SimpleNamespace(formula_matrix=np.eye(2), temperature_validity_upper=(1200., 1200.)),
    )


OPTIONS = CondensateEquilibriumOptions(return_diagnostics=True, rainout=False,
                                      full_condensate_budget_relative_tolerance=1e-9)
BUDGET = np.array([1., 2.])


def result(converged, *, scale=1.):
    gas, cloud = scale * np.array([.8, 1.5]), scale * np.array([.2, .5])
    return CondensateEquilibriumResult(
        gas_ln_n=np.log(gas), gas_n=gas, gas_x=gas/gas.sum(), gas_ntot=gas.sum(),
        condensate_amounts=cloud, condensate_support_indices=np.array([0, 1]),
        condensate_support_names=("H(s)", "He(s)"), acceptance_tier="accepted" if converged else "failed",
        selected_route="head_v2_fixed_support_lifecycle", status="converged" if converged else "not_converged",
        converged=converged, diagnostics={"physical_audit": {"accepted": converged}, "reason": "test evidence"},
    )


def sequence(values):
    calls = []
    values = iter(values)

    def solver(*args, **kwargs):
        calls.append((args, kwargs))
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value
    return solver, calls


def test_default_returns_original_result_and_forwards_arguments_unchanged(setup):
    failed = result(False)
    solver, calls = sequence([failed])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, record=receipts.append)
    assert wrapped(setup, 1000., 1., BUDGET, options=OPTIONS) is failed
    assert len(calls) == 1 and calls[0][0][3] is BUDGET
    assert calls[0][1] == {"options": OPTIONS} and not receipts


@pytest.mark.parametrize("enabled", [1, "yes", None])
def test_opt_in_is_strict_bool(enabled):
    with pytest.raises(TypeError, match="bool"):
        HELPER.make_previous_parcel_retry(lambda: None, enabled=enabled)


def test_successful_cold_call_is_returned_unchanged_without_retry(setup):
    accepted = result(True)
    solver, calls = sequence([accepted])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    assert wrapped(setup, 1000., 1., BUDGET, options=OPTIONS) is accepted
    assert len(calls) == 1 and not receipts


@pytest.mark.parametrize("scale", [1., 1e24, 1e-24])
def test_one_retry_uses_amounts_in_caller_gauge_and_preserves_failed_evidence(setup, scale):
    prior, cold, warm = result(True, scale=scale), result(False, scale=scale), result(True, scale=scale)
    solver, calls = sequence([prior, cold, warm])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    b = scale * BUDGET
    wrapped(setup, 1000., 1., b, options=OPTIONS)
    assert wrapped(setup, 1010., 1.1, b, options=OPTIONS) is warm
    assert len(calls) == 3
    assert calls[-1][0][3] is b and calls[-1][1]["options"] is OPTIONS
    initial = calls[-1][1]["init"]
    np.testing.assert_array_equal(initial.gas_ln_n, np.log(prior.gas_n))
    assert float(initial.gas_ntot) == prior.gas_ntot
    np.testing.assert_array_equal(initial.condensate_amounts, prior.condensate_amounts)
    for name in ("support_indices", "support_amounts", "element_potential", "rho", "barrier_epsilon", "inventory_bridge_origin"):
        assert getattr(initial, name) is None
    receipt = receipts[0]
    assert receipt["cold_result"]["converged"] is False and receipt["warm_result"]["converged"] is True
    assert receipt["donor"]["point"]["temperature_K"] == 1000.
    assert receipt["point"]["caller_element_amounts"] == b.tolist()
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("change", ["budget", "setup", "Pref", "invalid_temperature", "explicit_init", "lnphi"])
def test_incompatible_or_out_of_scope_seed_never_retries(setup, change):
    cold = result(False)
    solver, calls = sequence([result(True), cold])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    wrapped(setup, 1000., 1., BUDGET, options=OPTIONS)
    target, b, temperature, kwargs = setup, BUDGET.copy(), 1010., {"options": OPTIONS}
    if change == "budget":
        b[0] = np.nextafter(b[0], np.inf)
    elif change == "setup":
        target = SimpleNamespace(**vars(setup))
    elif change == "Pref":
        kwargs["Pref"] = 2.
    elif change == "invalid_temperature":
        temperature = 1300.
    elif change == "explicit_init":
        kwargs["init"] = object()
    else:
        kwargs["lnphi_func"] = lambda *args: np.zeros(2)
    assert wrapped(target, temperature, 1., b, **kwargs) is cold
    assert len(calls) == 2 and receipts[0]["retry_attempted"] is False
    assert receipts[0]["skip_reason"]


@pytest.mark.parametrize("bad", ["nonconserved", "zero_gas", "negative_cloud"])
def test_invalid_converged_amounts_are_not_reused_as_donors(setup, bad):
    donor = result(True)
    if bad == "nonconserved":
        donor.gas_n[0] *= 2.
    elif bad == "zero_gas":
        donor.gas_n[0] = 0.
    else:
        donor.condensate_amounts[0] = -.1
    cold = result(False)
    solver, calls = sequence([donor, cold])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    wrapped(setup, 1000., 1., BUDGET, options=OPTIONS)
    assert wrapped(setup, 1010., 1., BUDGET, options=OPTIONS) is cold
    assert len(calls) == 2 and not receipts[0]["retry_attempted"]


def test_failed_warm_result_stays_failed_and_cannot_retry_recursively(setup):
    warm = result(False)
    solver, calls = sequence([result(True), result(False), warm])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    wrapped(setup, 1000., 1., BUDGET, options=OPTIONS)
    assert wrapped(setup, 1010., 1., BUDGET, options=OPTIONS) is warm
    assert len(calls) == 3 and receipts[0]["warm_result"]["converged"] is False


@pytest.mark.parametrize("where", ["cold", "warm"])
def test_solver_exception_is_recorded_and_propagated(setup, where):
    error = ValueError("solver unavailable")
    values = [error] if where == "cold" else [result(True), result(False), error]
    solver, calls = sequence(values)
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    if where == "warm":
        wrapped(setup, 1000., 1., BUDGET, options=OPTIONS)
    with pytest.raises(ValueError, match="unavailable"):
        wrapped(setup, 1010., 1., BUDGET, options=OPTIONS)
    assert receipts[0][where + "_exception"] == "ValueError: solver unavailable"


def test_nonfinite_failed_diagnostics_are_explicit_json_evidence(setup):
    failed = result(False)
    failed.diagnostics["failed_residual"] = np.nan
    solver, _ = sequence([failed])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(solver, enabled=True, record=receipts.append)
    assert wrapped(setup, 1000., 1., BUDGET, options=OPTIONS) is failed
    assert receipts[0]["cold_result"]["diagnostics"]["failed_residual"] == {"nonfinite_float": "nan"}
    json.dumps(receipts[0], allow_nan=False)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_nonideal_opt_in_is_strict_bool(value):
    with pytest.raises(TypeError, match="allow_nonideal must be a bool"):
        HELPER.make_previous_parcel_retry(lambda: None, allow_nonideal=value)


def test_nonideal_opt_in_requires_explicit_fixed_context():
    with pytest.raises(ValueError, match="fixed EOS/column identity"):
        HELPER.make_previous_parcel_retry(lambda: None, enabled=True, allow_nonideal=True)


def test_nonideal_retry_preserves_current_fugacity_and_options(setup):
    prior, cold, warm = result(True), result(False), result(True)
    solver, calls = sequence([prior, cold, warm])
    context = object()
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(
        solver, enabled=True, record=receipts.append,
        allow_nonideal=True, nonideal_identity=context,
    )
    previous_phi = lambda *args: np.array([.1, .2])
    current_phi = lambda *args: np.array([.3, .4])
    wrapped(setup, 1000., 1., BUDGET, options=OPTIONS, lnphi_func=previous_phi)
    assert wrapped(setup, 1010., 1.1, BUDGET, options=OPTIONS, lnphi_func=current_phi) is warm
    assert len(calls) == 3
    assert calls[-2][1]["lnphi_func"] is current_phi
    assert calls[-1][1]["lnphi_func"] is current_phi
    assert calls[-1][1]["options"] is OPTIONS
    assert calls[-1][0] == calls[-2][0]
    initial = calls[-1][1]["init"]
    np.testing.assert_array_equal(initial.gas_ln_n, np.log(prior.gas_n))
    np.testing.assert_array_equal(initial.condensate_amounts, prior.condensate_amounts)
    assert initial.element_potential is None and initial.support_indices is None
    assert receipts[0]["nonideal_initialization"]["enabled"] is True
    assert receipts[0]["retry_attempted"] is True
    json.dumps(receipts[0], allow_nan=False)


@pytest.mark.parametrize("change", ["options", "nonideal_mode", "new_context"])
def test_nonideal_donor_rejected_when_its_options_or_context_change(setup, change):
    cold = result(False)
    solver, calls = sequence([result(True), cold])
    receipts = []
    wrapped = HELPER.make_previous_parcel_retry(
        solver, enabled=True, record=receipts.append,
        allow_nonideal=True, nonideal_identity=object(),
    )
    phi = lambda *args: np.array([.1, .2])
    wrapped(setup, 1000., 1., BUDGET, options=OPTIONS, lnphi_func=phi)
    options = OPTIONS
    if change == "options":
        options = replace(OPTIONS, full_condensate_budget_relative_tolerance=1e-8)
    elif change == "nonideal_mode":
        phi = None
    else:
        wrapped = HELPER.make_previous_parcel_retry(
            solver, enabled=True, record=receipts.append,
            allow_nonideal=True, nonideal_identity=object(),
        )
    assert wrapped(setup, 1010., 1.1, BUDGET, options=options, lnphi_func=phi) is cold
    assert len(calls) == 2 and receipts[0]["retry_attempted"] is False


def test_nonideal_options_snapshot_rejects_in_place_tolerance_change(setup):
    cold = result(False)
    solver, calls = sequence([result(True), cold])
    receipts = []
    options = replace(OPTIONS)
    wrapped = HELPER.make_previous_parcel_retry(
        solver, enabled=True, record=receipts.append,
        allow_nonideal=True, nonideal_identity=object(),
    )
    phi = lambda *args: np.array([.1, .2])
    wrapped(setup, 1000., 1., BUDGET, options=options, lnphi_func=phi)
    object.__setattr__(options, "full_condensate_budget_relative_tolerance", 1e-8)
    assert wrapped(setup, 1010., 1.1, BUDGET, options=options, lnphi_func=phi) is cold
    assert len(calls) == 2 and receipts[0]["skip_reason"] == "the donor options or fixed nonideal context differs"
