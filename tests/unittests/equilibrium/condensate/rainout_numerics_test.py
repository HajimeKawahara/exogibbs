"""Range and failure-report contracts for rainout post-solve operations."""

from types import SimpleNamespace

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.profile import (
    _certify_rainout_candidate,
    _conservation_rainout_inventory,
    _floorless_budget_certification,
)


def test_rainout_certification_and_propagation_share_depleted_species_inventory():
    setup = SimpleNamespace(
        elements=("A", "B", "e-"),
        formula_matrix=np.asarray([[1., 0., 0.], [1., 1., 0.], [0., 1., -1.]]),
        formula_matrix_cond=np.asarray([[1., 0.], [1., 1.], [0., 0.]]),
    )
    result = SimpleNamespace(
        gas_n=np.asarray([0.01, 0.75, 0.75]),
        condensate_amounts=np.asarray([0.1, 0.25]),
    )
    target = np.asarray([0.0, 1.0, 0.0])
    conserved = np.asarray([True, True, False])

    certificate = _floorless_budget_certification(
        setup=setup, result=result, conserved_mask=conserved,
        inventory_target=target, relative_tolerance=1.0e-3,
    )
    propagation = _conservation_rainout_inventory(
        setup=setup, result=result, conserved_mask=conserved,
        normalization_mask=target > 0.0, inventory_target=target,
        inventory_sum=1.0, roundoff_multiplier=64.0,
    )

    assert certificate["accepted"]
    for certified_name, propagated_name, expected in (
        ("raw_solver_gas_element_inventory", "gas_inventory", [0.01, 0.76, 0.]),
        ("gas_element_inventory", "propagation_gas_inventory", [0., 0.75, 0.]),
        ("raw_solver_condensate_element_inventory", "raw_condensate_inventory",
         [0.1, 0.35, 0.]),
        ("condensate_element_inventory", "condensate_inventory", [0., 0.25, 0.]),
    ):
        np.testing.assert_array_equal(
            certificate[certified_name], propagation[propagated_name]
        )
        np.testing.assert_allclose(propagation[propagated_name], expected)
    assert propagation["ignored_gas_species_indices"] == (0,)
    assert propagation["ignored_condensate_species_indices"] == (0,)
    np.testing.assert_array_equal(propagation["next_inventory"], target)


@pytest.mark.parametrize(
    ("target", "gas", "condensate", "expected"),
    (
        ([1.0e300, 1.0e-20, 0.0], [0.0, 1.0e-20, 0.0], 1.0e300,
         [0.0, 1.0e300, 0.0]),
        ([1.0e300, 1.0e300, np.nextafter(0.0, 1.0)],
         [0.0, 1.0e300, np.nextafter(0.0, 1.0)], 1.0e300,
         [0.0, 2.0e300, 2.0 * np.nextafter(0.0, 1.0)]),
        ([0.5, 0.5, 0.0], [0.25, 0.5, 0.0], 0.25,
         [1.0 / 3.0, 2.0 / 3.0, 0.0]),
    ),
)
def test_rainout_normalization_preserves_representable_survivors(
    target, gas, condensate, expected
):
    setup = SimpleNamespace(
        elements=("A", "B", "C"),
        formula_matrix=np.eye(3),
        formula_matrix_cond=np.array([[1.0], [0.0], [0.0]]),
    )
    target = np.asarray(target)
    result = SimpleNamespace(
        gas_n=np.asarray(gas), condensate_amounts=np.asarray([condensate])
    )
    conserved = np.ones(3, dtype=bool)
    assert _floorless_budget_certification(
        setup=setup, result=result, conserved_mask=conserved,
        inventory_target=target, relative_tolerance=1.0e-3,
    )["accepted"]

    propagation = _conservation_rainout_inventory(
        setup=setup, result=result, conserved_mask=conserved,
        normalization_mask=target > 0.0, inventory_target=target,
        inventory_sum=float(np.sum(target)), roundoff_multiplier=64.0,
    )

    np.testing.assert_allclose(
        propagation["next_inventory"], expected, rtol=1.0e-14, atol=0.0,
    )
    assert np.all(np.isfinite(propagation["next_inventory"]))
    assert np.isfinite(propagation["log_normalization"])
    assert propagation["normalization"] is None or np.isfinite(
        propagation["normalization"]
    )


def test_rejected_rainout_attempt_retains_compact_physical_failure_reason():
    kkt = {"gas_stationarity": 2.0e-4, "budget_scaled": 1.0e-15}
    caller_kkt = {"accepted": False, "gas_stationarity_max_abs": 2.0e-4}
    polish = {
        "accepted": False,
        "gas_stationarity_max_abs": 2.0e-4,
        "stationarity_tolerance": 1.0e-8,
        "selected_numerical_formulation": "reduced_log_domain_support_search",
        "support_release_portfolio": {"large_internal_trace": [1, 2, 3]},
    }
    candidate = SimpleNamespace(
        converged=False, status="not_converged", acceptance_tier="solver_failed",
        condensate_support_indices=np.asarray([], dtype=int),
        condensate_support_names=(),
        diagnostics={"fixed_support_v2": {
            "outcome": "zero_barrier_active_support_polish_failed",
            "independent_kkt": kkt,
            "caller_gauge_zero_barrier_kkt": caller_kkt,
            "zero_barrier_active_support_polish": polish,
        }},
    )
    assessment = _certify_rainout_candidate(
        setup=None, candidate=candidate, initialization="resolved",
        abundance_scale=1.0, conserved_mask=np.asarray([True]),
        inventory_target=np.asarray([1.0]), relative_tolerance=1.0e-3,
    )
    assert assessment.accepted_result is None
    assert assessment.attempt["independent_kkt"] == kkt
    assert assessment.attempt["caller_gauge_zero_barrier_kkt"] == caller_kkt
    reported = assessment.attempt["zero_barrier_active_support_polish"]
    assert reported["gas_stationarity_max_abs"] > reported["stationarity_tolerance"]
    assert reported["selected_numerical_formulation"] == polish[
        "selected_numerical_formulation"
    ]
    assert "support_release_portfolio" not in reported
