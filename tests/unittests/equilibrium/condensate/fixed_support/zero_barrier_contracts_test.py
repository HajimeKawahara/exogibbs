"""Contracts distinguishing terminal roots from support transitions."""

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    _audit_zero_barrier_state,
    _physical_zero_barrier_audit,
)


def _audit_arguments(*, amount=1.0, inactive_driving=1.0, status=1):
    return dict(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 2)),
        target_inventory=np.asarray([1.0 + amount]),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.asarray([0.0, inactive_driving]),
        gas_log_amounts=np.zeros(1),
        condensate_amounts=np.asarray([amount, 0.0]),
        total_gas_log_amount=0.0,
        element_potential=np.zeros(1),
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.ones(1),
        optimizer_success=status > 0,
        optimizer_status=status,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )


@pytest.mark.parametrize(
    "amount,inactive_driving,status,accepted,local_passed,drop_allowed",
    [
        (1.0, 1.0, 1, True, True, True),
        (1.0, -1.0, 1, False, True, True),
        (-0.5, 1.0, 1, False, False, True),
        (1.0, 1.0, 0, True, True, False),
        (1.0, -1.0, 0, False, False, False),
        (-0.5, 1.0, 0, False, False, False),
        (1.0, 1.0, -1, False, False, False),
    ],
)
def test_terminal_acceptance_and_support_permissions_are_distinct(
    amount, inactive_driving, status, accepted, local_passed, drop_allowed,
):
    arguments = _audit_arguments(
        amount=amount, inactive_driving=inactive_driving, status=status,
    )
    audit = _audit_zero_barrier_state(**arguments)
    assert audit.accepted is accepted
    assert audit.local_kkt_passed is local_passed
    assert audit.root_blocks_passed is drop_allowed
    report = audit.to_report()
    legacy = _physical_zero_barrier_audit(**arguments)
    assert report.keys() == legacy.keys()
    for key in report:
        np.testing.assert_equal(report[key], legacy[key])


def test_exported_diagnostics_cannot_change_audit_control_state():
    audit = _audit_zero_barrier_state(**_audit_arguments(inactive_driving=-1.0))
    report = audit.to_report()
    report["accepted"] = True
    report["full_driving"][:] = 100.0
    assert not audit.accepted
    assert audit.local_kkt_passed
    np.testing.assert_array_equal(audit.full_driving, [0.0, -1.0])
    np.testing.assert_array_equal(audit.to_report()["full_driving"], [0.0, -1.0])


def test_failed_equality_blocks_cannot_authorize_a_support_transition():
    arguments = _audit_arguments(amount=-0.5)
    arguments["target_inventory"] = np.asarray([2.0])
    audit = _audit_zero_barrier_state(**arguments)
    assert not audit.accepted
    assert not audit.root_blocks_passed
    assert "element_budget" in audit.local_kkt_failure_reasons
