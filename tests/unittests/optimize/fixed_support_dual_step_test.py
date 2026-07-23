import jax.numpy as jnp

from exogibbs.optimize.fixed_support_dual_step import (
    fixed_support_min_dual_infeasibility_step,
    fixed_support_min_equality_dual_infeasibility_step,
)


def test_min_dual_infeasibility_step_selects_quadratic_minimizer():
    alpha = fixed_support_min_dual_infeasibility_step(
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
        q_trial=jnp.asarray([2.0], dtype=jnp.float64),
        r_trial=jnp.asarray([-1.0], dtype=jnp.float64),
        lambda_current=jnp.asarray([0.0], dtype=jnp.float64),
        rho_current=jnp.asarray([0.0], dtype=jnp.float64),
        delta_lambda=jnp.asarray([2.0], dtype=jnp.float64),
        delta_rho=jnp.asarray([0.0], dtype=jnp.float64),
        gas_stationarity_source=jnp.asarray([0.0], dtype=jnp.float64),
        condensate_standard_source=jnp.asarray([1.0], dtype=jnp.float64),
        epsilon=jnp.asarray([-1.0], dtype=jnp.float64),
    )

    assert jnp.isclose(alpha, 0.5)


def test_min_dual_infeasibility_step_clips_to_unit_interval():
    alpha = fixed_support_min_dual_infeasibility_step(
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
        q_trial=jnp.asarray([-2.0], dtype=jnp.float64),
        r_trial=jnp.asarray([-1.0], dtype=jnp.float64),
        lambda_current=jnp.asarray([0.0], dtype=jnp.float64),
        rho_current=jnp.asarray([0.0], dtype=jnp.float64),
        delta_lambda=jnp.asarray([1.0], dtype=jnp.float64),
        delta_rho=jnp.asarray([0.0], dtype=jnp.float64),
        gas_stationarity_source=jnp.asarray([0.0], dtype=jnp.float64),
        condensate_standard_source=jnp.asarray([1.0], dtype=jnp.float64),
        epsilon=jnp.asarray([-1.0], dtype=jnp.float64),
    )

    assert alpha == 0.0


def test_min_equality_dual_step_uses_fixed_bound_dual_trial():
    alpha = fixed_support_min_equality_dual_infeasibility_step(
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
        q_trial=jnp.asarray([2.0], dtype=jnp.float64),
        rho_trial=jnp.asarray([0.0], dtype=jnp.float64),
        lambda_current=jnp.asarray([0.0], dtype=jnp.float64),
        delta_lambda=jnp.asarray([2.0], dtype=jnp.float64),
        gas_stationarity_source=jnp.asarray([0.0], dtype=jnp.float64),
        condensate_standard_source=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert jnp.isclose(alpha, 0.5)
