"""Finite-barrier decisions and the contract for exact refinement."""

from dataclasses import replace

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support.types import (
    KKTComponentNorms,
    TerminalStatus,
)
from exogibbs.equilibrium.condensate.lifecycle import (
    _assess_finite_barrier_layer,
)
from exogibbs.equilibrium.condensate.policy import (
    fixed_support_v2_production_policy,
)


def _batch_result(*, gas_stationarity=0.0, condensate_stationarity=0.0):
    return {
        "terminal_status": np.asarray([TerminalStatus.CONVERGED]),
        "fixed_support_converged": np.asarray([True]),
        "support_closed": np.asarray([True]),
        "final_state_values_finite": np.asarray([True]),
        "final_kkt_norms": KKTComponentNorms(
            gas_stationarity=np.asarray([gas_stationarity]),
            condensate_stationarity=np.asarray([condensate_stationarity]),
            budget_scaled=np.zeros(1),
            complementarity=np.zeros(1),
            total_density_scaled=np.zeros(1),
        ),
    }


def _assess_result(raw):
    return _assess_finite_barrier_layer(
        raw,
        0,
        terminal_status=int(raw["terminal_status"][0]),
        fixed_support_converged=bool(raw["fixed_support_converged"][0]),
        support_closed=bool(raw["support_closed"][0]),
        policy=fixed_support_v2_production_policy(),
    )


def test_phase_barrier_bias_only_authorizes_exact_initialization() -> None:
    raw = _batch_result(condensate_stationarity=1.0)
    raw["fixed_support_converged"][0] = False
    raw["terminal_status"][0] = TerminalStatus.NORMAL_LINE_SEARCH_FAILED
    assessment = _assess_result(raw)

    assert assessment.terminal_initializer_eligible
    assert not assessment.fixed_support_accepted
    assert not assessment.early_initializer_eligible
    assert assessment.terminal_outcome == "fixed_support_failed"
    report = assessment.diagnostics()
    assert report["zero_barrier_initializer_kkt_passed"]
    assert not report["independent_kkt_passed"]

    # Presentation may be augmented without granting numerical acceptance.
    report["fixed_support_converged"] = True
    report["independent_kkt_passed"] = True
    assert not assessment.fixed_support_accepted
    assert assessment.terminal_outcome == "fixed_support_failed"


@pytest.mark.parametrize("gas_stationarity", [1.1e-5, np.inf, np.nan])
def test_exact_handoff_rejects_bad_noncondensate_residuals(
    gas_stationarity,
) -> None:
    assessment = _assess_result(
        _batch_result(gas_stationarity=gas_stationarity)
    )
    assert not assessment.terminal_initializer_eligible
    assert not assessment.fixed_support_accepted


def test_open_converged_state_only_authorizes_early_exact_closure() -> None:
    raw = _batch_result()
    raw["support_closed"][0] = False
    assessment = _assess_result(raw)

    assert assessment.early_initializer_eligible
    assert not assessment.terminal_initializer_eligible
    assert not assessment.fixed_support_accepted
    assert assessment.terminal_outcome is None


@pytest.mark.parametrize(
    "converged,kkt_passed,finite,closed,outcome",
    [
        (False, False, False, True, "fixed_support_failed"),
        (True, False, False, True, "independent_kkt_failed"),
        (True, True, False, True, "nonfinite_final_state"),
        (True, True, True, True, "closed"),
        (True, True, True, False, None),
    ],
)
def test_finite_barrier_stop_precedence_is_independent_of_diagnostics(
    converged, kkt_passed, finite, closed, outcome
) -> None:
    assessment = _assess_result(_batch_result())
    assessment = replace(
        assessment,
        fixed_support_converged=converged,
        independent_kkt_passed=kkt_passed,
        final_state_values_finite=finite,
        support_closed=closed,
    )
    assert assessment.terminal_outcome == outcome
