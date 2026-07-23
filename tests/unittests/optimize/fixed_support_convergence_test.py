import pytest

from exogibbs.optimize.fixed_support_convergence import (
    fixed_support_batch_converged,
    fixed_support_budget_relative_max,
)


def test_fixed_support_budget_relative_max_uses_trace_floor():
    value = fixed_support_budget_relative_max(
        budget_residual=[2.0e-13, 2.0e-9],
        target=[1.0e-9, 1.0e-3],
        relative_floor=1.0e-8,
    )

    assert float(value) == pytest.approx(2.0e-5)


def test_fixed_support_batch_converged_uses_componentwise_tolerances():
    assert bool(
        fixed_support_batch_converged(
            gas_norm=1.0e-6,
            condensate_stationarity_norm=2.0e-6,
            complementarity_norm=9.0e-6,
            total_density_norm=1.0e-6,
            budget_relative_max=9.0e-5,
            log_tolerance=1.0e-5,
            budget_relative_tolerance=1.0e-4,
            total_density_tolerance=1.0e-5,
        )
    )
    assert not bool(
        fixed_support_batch_converged(
            gas_norm=1.0e-6,
            condensate_stationarity_norm=2.0e-6,
            complementarity_norm=1.1e-5,
            total_density_norm=1.0e-6,
            budget_relative_max=9.0e-5,
            log_tolerance=1.0e-5,
            budget_relative_tolerance=1.0e-4,
            total_density_tolerance=1.0e-5,
        )
    )
    assert not bool(
        fixed_support_batch_converged(
            gas_norm=1.0e-6,
            condensate_stationarity_norm=2.0e-6,
            complementarity_norm=9.0e-6,
            total_density_norm=1.0e-6,
            budget_relative_max=1.1e-4,
            log_tolerance=1.0e-5,
            budget_relative_tolerance=1.0e-4,
            total_density_tolerance=1.0e-5,
        )
    )
