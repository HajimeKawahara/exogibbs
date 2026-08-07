"""Tests for the physical zero-barrier active-support refinement."""

import warnings

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    _least_squares_with_scipy_overflow_guard,
    _solve_reduced_log_domain_active_support,
    _solve_reduced_log_domain_support_branches,
    polish_zero_barrier_active_support,
)


def test_least_squares_guard_is_local_to_scipy_scalar_divide_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def warning_solver(*args, **kwargs):
        del args, kwargs
        warnings.warn_explicit(
            "overflow encountered in scalar divide",
            RuntimeWarning,
            filename="scipy/optimize/_lsq/common.py",
            lineno=1,
            module="scipy.optimize._lsq.common",
        )
        return sentinel

    monkeypatch.setattr(
        "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
        "least_squares",
        warning_solver,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _least_squares_with_scipy_overflow_guard(
            lambda values: values,
            np.ones(1, dtype=np.float64),
        )
        assert result is sentinel
        with pytest.raises(RuntimeWarning, match="scalar divide"):
            warnings.warn_explicit(
                "overflow encountered in scalar divide",
                RuntimeWarning,
                filename="other_solver.py",
                lineno=1,
                module="other_solver",
            )


def _one_active_phase_problem():
    gas_formula = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    condensate_formula = np.eye(2, dtype=np.float64)
    expected_gas = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    expected_condensates = np.asarray([0.7, 0.0], dtype=np.float64)
    expected_potential = np.asarray([0.4, -0.2], dtype=np.float64)
    gamma = (
        gas_formula.T @ expected_potential - np.log(expected_gas)
    )
    hcond = np.asarray([0.4, -0.1], dtype=np.float64)
    target = (
        gas_formula @ expected_gas
        + condensate_formula @ expected_condensates
    )
    return (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    )


def test_zero_barrier_polish_restores_all_physical_kkt_blocks() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas) + 2.0e-2,
        condensate_amounts_init=np.asarray([0.65, 0.0]),
        total_gas_log_amount_init=1.0e-2,
        element_potential_init=expected_potential + 1.0e-2,
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    assert result.accepted
    assert result.support_indices == (0,)
    assert np.exp(result.gas_log_amounts) == pytest.approx(expected_gas)
    assert result.condensate_amounts == pytest.approx(expected_condensates)
    assert result.element_potential == pytest.approx(expected_potential)
    assert result.report["active_condensate_driving_max_abs"] < 1.0e-10
    assert result.report["inactive_condensate_violation_max_abs"] == 0.0
    assert result.report["budget_scaled_max_abs"] < 1.0e-10
    assert result.report["total_density_scaled_abs"] < 1.0e-10


def test_zero_barrier_polish_rejects_negative_inactive_driving() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=expected_condensates,
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    assert not result.accepted
    assert result.report["active_condensate_driving_max_abs"] < 1.0e-10
    assert result.report["inactive_condensate_violation_max_abs"] == (
        pytest.approx(0.1)
    )


def test_zero_barrier_polish_drops_a_negative_phase_and_updates_support() -> None:
    gas_formula = np.eye(2, dtype=np.float64)
    condensate_formula = np.asarray([[0.0], [1.0]], dtype=np.float64)
    target = np.asarray([1.0, 0.01], dtype=np.float64)
    active_gas = np.asarray([1.0, 0.25], dtype=np.float64)

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=np.zeros(2),
        condensate_standard_source_full=np.asarray([np.log(0.2)]),
        gas_log_amounts_init=np.log(active_gas),
        condensate_amounts_init=np.asarray([1.0e-3]),
        total_gas_log_amount_init=np.log(np.sum(active_gas)),
        element_potential_init=np.log(np.asarray([0.8, 0.2])),
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    assert result.accepted
    assert result.support_indices == ()
    assert result.report["dropped_support_indices"] == (0,)
    assert result.condensate_amounts.tolist() == [0.0]
    assert np.exp(result.gas_log_amounts) == pytest.approx(target)
    assert result.report["full_condensate_driving"][0] > 0.0


def test_zero_barrier_polish_rejects_incompatible_active_stationarity() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=np.asarray([0.65, 0.03]),
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0, 1),
        budget_relative_floor=1.0e-12,
    )

    assert not result.accepted
    assert (
        result.report["active_condensate_driving_max_abs"] > 1.0e-8
        or result.report["total_density_scaled_abs"] > 1.0e-8
    )


def test_zero_barrier_polish_uses_relative_scaling_for_positive_trace() -> None:
    gas_formula = np.eye(2, dtype=np.float64)
    condensate_formula = np.asarray([[1.0], [0.0]], dtype=np.float64)
    expected_gas = np.asarray([0.5, 1.0e-20], dtype=np.float64)
    expected_condensate = np.asarray([0.5], dtype=np.float64)
    target = np.asarray([1.0, 1.0e-20], dtype=np.float64)
    qtot = float(np.log(np.sum(expected_gas)))
    potential = np.asarray([0.2, -0.4], dtype=np.float64)
    gamma = potential - np.log(expected_gas) + qtot

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=np.asarray([potential[0]]),
        gas_log_amounts_init=np.log(np.asarray([0.5, 1.0e-18])),
        condensate_amounts_init=expected_condensate,
        total_gas_log_amount_init=qtot,
        element_potential_init=potential,
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
    )

    reconstructed = (
        gas_formula @ np.exp(result.gas_log_amounts)
        + condensate_formula @ result.condensate_amounts
    )
    assert result.accepted
    assert reconstructed[1] == pytest.approx(target[1], rel=1.0e-8)
    assert result.report["budget_scaled_max_abs"] < 1.0e-8
    assert result.report["budget_scaling"].startswith("relative_for_nonzero")


def test_zero_barrier_polish_regularizes_trace_barrier_initializer() -> None:
    gas_formula = np.asarray(
        [
            [2.0, 2.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    condensate_formula = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    target = np.asarray(
        [
            2.8959369615671957e8,
            7.7510262401527523e-22,
            1.0406303843280405e7,
        ],
        dtype=np.float64,
    )
    gamma = np.asarray(
        [
            -78.68871595,
            -171.79173500,
            -89.90713332,
            -154.77418944,
            -222.42561704,
        ],
        dtype=np.float64,
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=np.asarray(
            [-227.97601192, -346.40090319],
            dtype=np.float64,
        ),
        gas_log_amounts_init=np.asarray(
            [
                18.63566502187771,
                -2.15476346e16,
                -4.30952692e16,
                -2.15476346e16,
                16.157922319260653,
            ],
            dtype=np.float64,
        ),
        condensate_amounts_init=np.asarray(
            [target[1], 0.0],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=18.71626062818451,
        element_potential_init=np.asarray(
            [-39.3846558, -2.15476346e16, -67.4453322],
            dtype=np.float64,
        ),
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
    )

    regularization = result.report["initializer_regularization"]
    reconstructed = (
        gas_formula @ np.exp(result.gas_log_amounts)
        + condensate_formula @ result.condensate_amounts
    )
    assert result.accepted
    assert result.support_indices == (0,)
    assert regularization["applied"]
    assert regularization["capacity_fraction"] == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps)
    )
    assert regularization["regularized_gas_count"] == 3
    assert regularization["regularized_gas_mask"] == (
        False,
        True,
        True,
        True,
        False,
    )
    assert regularization["element_potential_recomputed"]
    assert regularization["element_potential_fit_rank"] == 3
    assert reconstructed == pytest.approx(target, rel=1.0e-12)
    assert result.condensate_amounts[0] == pytest.approx(
        1.27414995e-22,
        rel=1.0e-6,
    )
    assert result.report["active_condensate_driving_max_abs"] < 1.0e-8


def test_reduced_log_fallback_branches_layer_839_trace_support() -> None:
    gas_formula = np.asarray(
        [
            [2.0, 2.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    condensate_formula = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    target = (
        np.asarray(
            [0.9653123147, 5.43352573e-38, 0.0346876853],
            dtype=np.float64,
        )
        * 3.0e8
    )
    gamma = np.asarray(
        [
            -104.36002887563116,
            -225.45008100585233,
            -119.1085856180165,
            -201.02350479285428,
            -296.7652055580194,
        ],
        dtype=np.float64,
    )
    hcond = np.asarray(
        [-296.684067078042, -451.80912549650316],
        dtype=np.float64,
    )
    gas_init = np.asarray(
        [
            (target[0] - 4.0 * target[2]) / 2.0,
            target[1] * 1.0e-3,
            target[1] * 1.0e-3,
            target[1] * 1.0e-3,
            target[2],
        ],
        dtype=np.float64,
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(gas_init),
        condensate_amounts_init=np.asarray(
            [0.5 * target[1], 0.25 * target[1]],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=float(np.log(np.sum(gas_init))),
        element_potential_init=np.zeros(3, dtype=np.float64),
        support_indices=(0, 1),
        budget_relative_floor=1.0e-6,
    )

    gas_inventory = gas_formula @ np.exp(result.gas_log_amounts)
    reconstructed = (
        gas_inventory + condensate_formula @ result.condensate_amounts
    )
    linear_supports = tuple(
        attempt["support_indices"] for attempt in result.report["attempts"]
    )
    fallback = result.report["reduced_log_domain_fallback"]

    assert result.accepted
    assert result.support_indices == (0,)
    selected = result.report["selected_numerical_formulation"]
    # BLAS-level differences can reverse the order of two negative amounts
    # near 1e-27.  Either route is valid if its own audit and closure pass.
    assert selected in {
        "capacity_scaled_linear_amounts",
        "reduced_log_domain_support_search",
    }
    assert linear_supports[0] == (0, 1)
    if selected == "reduced_log_domain_support_search":
        assert linear_supports == ((0, 1), (1,), ())
        assert all(
            amount < 0.0
            for amount in result.report["attempts"][0][
                "active_condensate_amounts"
            ]
        )
        assert fallback["accepted"]
        assert fallback["visited_supports"] == ((0, 1), (1,), (0,))
        assert tuple(node["accepted"] for node in fallback["nodes"]) == (
            False,
            False,
            True,
        )
        assert all(
            not node["solve"]["greedy_drop_enabled"]
            for node in fallback["nodes"]
        )
    else:
        assert linear_supports[-1] == (0,)
        assert result.report["linear_amount_physical_audit"]["accepted"]
        assert not fallback["attempted"]
        assert not fallback["accepted"]
        assert fallback["skip_reason"] == (
            "linear_amount_physical_audit_accepted"
        )
    assert reconstructed == pytest.approx(target, rel=1.0e-12)
    assert gas_inventory[1] / target[1] == pytest.approx(
        0.7947931234846488,
        rel=1.0e-10,
    )
    assert result.condensate_amounts[0] > 0.0
    assert result.condensate_amounts[1] == 0.0
    assert result.report["full_condensate_driving"][1] > 0.0

    # Exercise the breadth-first fallback independently of the linear route.
    branched = _solve_reduced_log_domain_support_branches(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(gas_init),
        condensate_amounts_init=np.asarray(
            [0.5 * target[1], 0.25 * target[1]],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=float(np.log(np.sum(gas_init))),
        element_potential_init=np.zeros(3, dtype=np.float64),
        support_indices=(0, 1),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
    )
    branch_report = branched["report"]

    assert branched["accepted"]
    assert branched["candidate"]["support_indices"] == (0,)
    assert branch_report["visited_supports"] == ((0, 1), (1,), (0,))
    assert tuple(
        node["accepted"] for node in branch_report["nodes"]
    ) == (False, False, True)
    assert all(
        not node["solve"]["greedy_drop_enabled"]
        for node in branch_report["nodes"]
    )


def test_reduced_log_support_rejects_active_phase_at_amount_floor() -> None:
    true_relative_amount = 1.0e-12
    target = np.ones(2, dtype=np.float64)
    gas_fractions = np.asarray(
        [
            (1.0 - true_relative_amount) / (2.0 - true_relative_amount),
            1.0 / (2.0 - true_relative_amount),
        ],
        dtype=np.float64,
    )
    common_arguments = {
        "gas_formula_matrix": np.eye(2, dtype=np.float64),
        "condensate_formula_matrix_full": np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        "target_inventory": target,
        "gas_standard_source": np.zeros(2, dtype=np.float64),
        "condensate_standard_source_full": np.asarray(
            [np.log(gas_fractions[0])], dtype=np.float64
        ),
        "gas_log_amounts_init": np.log(target),
        "condensate_amounts_init": np.asarray(
            [true_relative_amount], dtype=np.float64
        ),
        "total_gas_log_amount_init": float(np.log(np.sum(target))),
        "element_potential_init": np.log(
            np.asarray([0.5, 0.5], dtype=np.float64)
        ),
        "support_indices": (0,),
        "condensate_valid_mask": np.asarray([True]),
        "budget_scale": np.reciprocal(target),
        "stationarity_tolerance": 1.0e-8,
        "budget_tolerance": 1.0e-8,
        "total_density_tolerance": 1.0e-8,
        "support_closure_tolerance": 1.0e-8,
        "max_function_evaluations": 400,
    }

    branched = _solve_reduced_log_domain_support_branches(
        **common_arguments
    )
    first_node = branched["report"]["nodes"][0]
    first_attempt = first_node["solve"]["attempts"][0]

    assert first_attempt["physical_audit_accepted"]
    assert first_attempt["active_phase_at_lower_bound"]
    assert first_attempt["lower_bound_support_indices"] == (0,)
    assert not first_node["accepted"]
    assert branched["accepted"]
    assert branched["candidate"]["support_indices"] == ()
    assert branched["report"]["visited_supports"] == ((0,), ())

    greedy = _solve_reduced_log_domain_active_support(
        **common_arguments,
        allow_greedy_drop=True,
    )

    assert greedy["accepted"]
    assert greedy["candidate"]["support_indices"] == ()
    assert greedy["report"]["dropped_support_indices"] == (0,)


@pytest.mark.parametrize(
    ("gas_formula", "target", "skip_reason"),
    (
        (
            np.eye(2, dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            "nonpositive_target_row",
        ),
        (
            np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
            "signed_stoichiometry_row",
        ),
    ),
)
def test_reduced_log_fallback_skips_nonlogarithmic_budget_rows(
    gas_formula,
    target,
    skip_reason,
) -> None:
    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        target_inventory=target,
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.asarray([0.5, 1.0e-300], dtype=np.float64)
        ),
        condensate_amounts_init=np.asarray([0.5], dtype=np.float64),
        total_gas_log_amount_init=float(np.log(0.5 + 1.0e-300)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
        max_function_evaluations=10,
    )

    fallback = result.report["reduced_log_domain_fallback"]
    assert not fallback["eligible"]
    assert not fallback["attempted"]
    assert not fallback["accepted"]
    assert fallback["skip_reason"] == skip_reason
