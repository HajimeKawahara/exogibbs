"""Tests for algorithm-v1.1 condensate direction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from exogibbs.optimize.condensate_algorithm_v11_direction import (
    build_active_condensate_budget_correction_direction,
    build_linear_budget_total_density_amount_gas_direction,
)


def test_active_condensate_budget_correction_reduces_linearized_budget() -> None:
    direction = build_active_condensate_budget_correction_direction(
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        external_condensate_budget=[0.2],
        q=[np.log(0.7)],
        r=[np.log(0.5)],
        lambda_size=1,
        rho_size=1,
        max_abs_delta_r=10.0,
        damping=0.0,
    )

    assert direction.direction_kind == "active_condensate_budget_correction_direction"
    assert direction.delta_q == pytest.approx([0.0])
    assert direction.delta_lambda == pytest.approx([0.0])
    assert direction.delta_rho == pytest.approx([0.0])
    assert direction.delta_qtot == pytest.approx(0.0)
    assert direction.delta_r == pytest.approx([-0.8])


def test_active_condensate_budget_correction_clips_large_log_amount_step() -> None:
    direction = build_active_condensate_budget_correction_direction(
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        q=[np.log(0.7)],
        r=[np.log(1.0e-6)],
        lambda_size=1,
        rho_size=1,
        max_abs_delta_r=2.0,
        damping=0.0,
    )

    assert np.max(np.abs(direction.delta_r)) == pytest.approx(2.0)


def test_relative_active_condensate_budget_correction_weights_trace_element() -> None:
    absolute_direction = build_active_condensate_budget_correction_direction(
        formula_matrix=[[0.0], [0.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[1.0, 1.0e-8],
        q=[np.log(1.0e-30)],
        r=[np.log(1.0)],
        lambda_size=2,
        rho_size=1,
        max_abs_delta_r=10.0,
        damping=0.0,
    )
    relative_direction = build_active_condensate_budget_correction_direction(
        formula_matrix=[[0.0], [0.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[1.0, 1.0e-8],
        q=[np.log(1.0e-30)],
        r=[np.log(1.0)],
        lambda_size=2,
        rho_size=1,
        max_abs_delta_r=10.0,
        damping=0.0,
        relative_budget_weighting=True,
    )

    assert absolute_direction.delta_r[0] == pytest.approx(-0.5)
    assert relative_direction.delta_r[0] == pytest.approx(-1.0)


def test_active_condensate_budget_correction_enforces_elemental_capacity() -> None:
    direction = build_active_condensate_budget_correction_direction(
        formula_matrix=[[0.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0e-8],
        q=[np.log(1.0e-30)],
        r=[np.log(1.0e-6)],
        lambda_size=1,
        rho_size=1,
        max_abs_delta_r=2.0,
        damping=0.0,
        relative_budget_weighting=True,
        enforce_condensate_capacity=True,
    )

    assert direction.delta_r[0] == pytest.approx(np.log(1.0e-8) - np.log(1.0e-6))


def test_joint_budget_direction_can_weight_trace_element_relatively() -> None:
    absolute_direction = build_linear_budget_total_density_amount_gas_direction(
        formula_matrix=[[1.0], [1.0]],
        formula_matrix_cond_active=[[0.0], [0.0]],
        element_inventory_target=[1.0, 1.0e-8],
        gas_stationarity_source=[0.0],
        q=[np.log(1.0)],
        r=[np.log(1.0e-300)],
        lam=[0.0, 0.0],
        rho=[0.0],
        qtot=np.log(1.0),
        budget_weight=1.0,
        total_density_weight=0.0,
        amount_gas_weight=0.0,
        target_direction_weight=0.0,
        max_abs_delta_q=10.0,
    )
    relative_direction = build_linear_budget_total_density_amount_gas_direction(
        formula_matrix=[[1.0], [1.0]],
        formula_matrix_cond_active=[[0.0], [0.0]],
        element_inventory_target=[1.0, 1.0e-8],
        gas_stationarity_source=[0.0],
        q=[np.log(1.0)],
        r=[np.log(1.0e-300)],
        lam=[0.0, 0.0],
        rho=[0.0],
        qtot=np.log(1.0),
        budget_weight=1.0,
        total_density_weight=0.0,
        amount_gas_weight=0.0,
        target_direction_weight=0.0,
        budget_row_scaling_policy="relative_target",
        max_abs_delta_q=10.0,
    )

    assert abs(relative_direction.delta_q[0]) > abs(absolute_direction.delta_q[0])
    assert relative_direction.delta_q[0] == pytest.approx(-1.0)
