"""Boundary geometry and independent zero-barrier certification contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_basic_portfolio_preserves_an_occupied_proper_face(amount_scale: float) -> None:
    formula = np.asarray([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    amounts = amount_scale * np.asarray([0.0, 0.0, 1.0])
    candidates, report = zero_barrier._build_alternative_basic_support_candidates(
        condensate_formula_matrix_full=formula,
        target_inventory=amount_scale * np.asarray([2.0, 1.0]),
        condensate_amounts=amounts,
        support_indices=(2, 1, 0),
        budget_scale=np.full(2, 1.0 / amount_scale),
        budget_tolerance=1.0e-8,
    )
    assert tuple(candidate["support_indices"] for candidate in candidates) == (
        (2,),
    )
    np.testing.assert_allclose(
        formula @ candidates[0]["condensate_amounts"], formula @ amounts,
        rtol=1.0e-12, atol=0.0,
    )
    assert report["positive_input_face"]["eligible"]
    assert report["initial_support_rank"] == 2


@pytest.mark.parametrize(
    "amount,support,valid,accepted",
    (
        (0.0, (), True, True),
        (0.5, (0,), True, True),
        (-0.5, (), True, False),
        (-0.5, (0,), True, False),
        (0.5, (), True, False),
        (0.5, (0,), False, False),
        (0.5, (0, 0), True, False),
        (0.0, (1,), True, False),
    ),
)
def test_certificate_checks_full_amount_vector_and_support(
    amount: float, support: tuple[int, ...], valid: bool, accepted: bool,
) -> None:
    audit = zero_barrier._physical_zero_barrier_audit(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 1)),
        target_inventory=np.asarray([1.0 + amount]),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(1),
        gas_log_amounts=np.zeros(1),
        condensate_amounts=np.asarray([amount]),
        total_gas_log_amount=0.0,
        element_potential=np.zeros(1),
        support_indices=support,
        condensate_valid_mask=np.asarray([valid]),
        budget_scale=np.ones(1),
        optimizer_success=True,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )
    assert audit["accepted"] is accepted
    assert audit["physical_root_certified"] is accepted
    assert audit["budget_scaled_max_abs"] == 0.0
    tolerances = {
        "stationarity_tolerance": 1.0e-8,
        "budget_tolerance": 1.0e-8,
        "total_density_tolerance": 1.0e-8,
    }
    assert zero_barrier._physical_audit_local_kkt_passed(
        audit, optimizer_success=True, **tolerances
    ) is accepted
    reasons = zero_barrier._local_zero_barrier_kkt_failure_reasons(
        audit | {"optimizer_success": True}, **tolerances
    )
    assert bool(reasons) is not accepted
    # A negative active amount may authorize release of that phase, but an
    # off-support amount or invalid support must never authorize a pivot.
    assert zero_barrier._physical_audit_root_blocks_passed(
        audit, optimizer_success=True, **tolerances
    ) is audit["support_consistent"]


@pytest.mark.parametrize("charge", (0.0, -0.1))
def test_structural_zero_keeps_signed_conservation_rows(charge: float) -> None:
    gas = np.asarray([0.6 - charge, 0.2 + charge, 0.2, 1.0e-12])
    formula = np.asarray([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, -2.0],
    ])
    potential = np.asarray([0.3, 0.0, 0.4])
    result = zero_barrier.polish_zero_barrier_active_support(
        gas_formula_matrix=formula,
        condensate_formula_matrix_full=np.asarray([[1.0], [0.0], [0.0]]),
        target_inventory=np.asarray([0.8, 0.0, charge]),
        gas_standard_source=formula.T @ potential - np.log(gas),
        condensate_standard_source_full=np.ones(1),
        gas_log_amounts_init=np.log(gas),
        condensate_amounts_init=np.zeros(1),
        total_gas_log_amount_init=0.0,
        element_potential_init=potential,
        support_indices=(),
    )
    report = result.report["structural_zero_reduced_log_rescue"]
    assert result.accepted
    assert report["structural_zero_target_rows"] == (1,)
    assert report["retained_budget_rows"] == (0, 2)
    assert report["retained_zero_target_rows"] == (
        (2,) if charge == 0.0 else ()
    )
    assert report["suppressed_gas_indices"] == (3,)
    np.testing.assert_allclose(
        np.exp(result.gas_log_amounts)[:3], gas[:3], rtol=1e-8
    )
    assert result.element_potential[2] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "controlled_boundary", (False, True), ids=("numerical", "controlled_boundary")
)
def test_full_rank_boundary_can_release_a_phase(
    monkeypatch: pytest.MonkeyPatch, controlled_boundary: bool,
) -> None:
    boundary_calls = []
    if controlled_boundary:
        real_solve = zero_barrier._solve_normalized_gas_reduced_linear_support

        def stop_at_full_rank_boundary(**kwargs):
            if tuple(kwargs["support_indices"]) != (0, 1):
                return real_solve(**kwargs)
            # Fix only the optimizer outcome at this seam. The real release
            # search, face solves, and physical certificate remain in use.
            boundary_calls.append(tuple(kwargs["support_indices"]))
            kwargs["function_evaluation_budget"].consume(1)
            state = {
                "gas_log_amounts": np.log(np.asarray([0.5, 1.0])),
                "condensate_amounts": np.asarray([0.5, 0.0]),
                "total_gas_log_amount": float(np.log(1.5)),
                "element_potential": np.zeros(2),
                "support_indices": (0, 1),
                "optimizer_success": True,
                "optimizer_status": 1,
            }
            audit = zero_barrier._physical_zero_barrier_audit(
                **state,
                **{
                    key: kwargs[key]
                    for key in (
                        "gas_formula_matrix", "condensate_formula_matrix_full",
                        "target_inventory", "gas_standard_source",
                        "condensate_standard_source_full", "condensate_valid_mask",
                        "budget_scale", "stationarity_tolerance",
                        "budget_tolerance", "total_density_tolerance",
                        "support_closure_tolerance",
                    )
                },
            )
            assert audit["finite"]
            assert not audit["positive_active_amounts"]
            assert not audit["accepted"]
            return {
                "accepted": False,
                "candidate": state | {
                    "accepted": False,
                    "optimizer_message": "Controlled full-rank boundary",
                    "function_evaluations": 1,
                    "active_phase_at_lower_bound": True,
                    "audit": audit,
                },
                "report": {
                    "attempted": True,
                    "accepted": False,
                    "attempts": ({"function_evaluations": 1},),
                },
            }

        monkeypatch.setattr(
            zero_barrier, "_solve_normalized_gas_reduced_linear_support",
            stop_at_full_rank_boundary,
        )

    budget = zero_barrier._FunctionEvaluationBudget(800)
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.eye(2),
        condensate_formula_matrix_full=np.eye(2),
        target_inventory=np.ones(2),
        gas_standard_source=np.log(np.asarray([3.0, 1.5])),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2),
        support_indices=(0, 1),
        max_function_evaluations=100,
        function_evaluation_budget=budget,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )
    assert result.accepted
    assert result.support_indices == (0,)
    np.testing.assert_allclose(
        np.exp(result.gas_log_amounts), [0.5, 1.0], rtol=1.0e-8, atol=0.0
    )
    np.testing.assert_allclose(
        result.condensate_amounts, [0.5, 0.0], rtol=1.0e-8, atol=0.0
    )
    assert result.report["support_consistent"]
    assert result.report["nonnegative_condensate_amounts"]
    assert result.report["selected_physical_audit"]["support_consistent"]
    assert result.report["selected_physical_audit"]["nonnegative_condensate_amounts"]
    assert result.report["basic_support_reduction"]["initial_support_nullity"] == 0
    if controlled_boundary:
        assert boundary_calls
        release = result.report["support_release_portfolio"]
        assert release["attempted"]
        assert release["trigger"] == "full_rank_support_boundary_reached"
    assert budget.used <= budget.limit


def test_dual_coordinate_scaling_keeps_representable_trace_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected = []

    def inspect_optimizer(objective, initial, *, jac, constraints, **kwargs):
        del kwargs
        with np.errstate(over="raise", invalid="raise"):
            objective(initial)
            inspected.append(jac(initial))
            for constraint in constraints:
                assert np.all(np.isfinite(constraint["fun"](initial)))
                assert np.all(np.isfinite(constraint["jac"](initial)))
        return SimpleNamespace(
            x=initial, success=False, status=9, message="test", nit=1
        )

    monkeypatch.setattr(zero_barrier, "minimize", inspect_optimizer)
    with np.errstate(over="raise", invalid="raise"):
        result = zero_barrier._select_support_with_zero_barrier_dual(
            gas_formula_matrix=np.eye(2),
            condensate_formula_matrix_full=np.eye(2),
            target_inventory=np.asarray([1.0, 1.0e-320]),
            gas_standard_source=np.zeros(2),
            condensate_standard_source_full=np.ones(2),
            gas_log_amounts_init=np.zeros(2),
            condensate_amounts_init=np.zeros(2),
            total_gas_log_amount_init=0.0,
            element_potential_init=np.zeros(2),
            condensate_valid_mask=np.ones(2, dtype=bool),
            stationarity_tolerance=1.0e-8,
            support_closure_tolerance=1.0e-8,
            max_function_evaluations=10,
            enabled=True,
        )
    assert len(inspected) == 1
    assert np.all(np.isfinite(inspected[0]))
    assert inspected[0][1] != 0.0
    assert not result["applied"]
