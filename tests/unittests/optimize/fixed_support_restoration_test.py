import jax
import jax.numpy as jnp

from exogibbs.optimize.fixed_support_kkt import fixed_support_filter_theta
from exogibbs.optimize.fixed_support_restoration import (
    fixed_support_amount_space_restoration,
    fixed_support_full_restoration,
    fixed_support_ipopt_bound_multiplier_update,
    fixed_support_ipopt_restoration_dual_return,
    fixed_support_restoration_phase_exit,
    fixed_support_restoration_phase_transition,
)

jax.config.update("jax_enable_x64", True)


def test_full_restoration_reduces_primal_infeasibility_with_positive_slacks():
    kwargs = {
        "formula_matrix": jnp.asarray([[1.0]]),
        "formula_matrix_cond_active": jnp.asarray([[1.0]]),
        "element_inventory_target": jnp.asarray([1.0]),
        "q_reference": jnp.log(jnp.asarray([0.4])),
        "r_reference": jnp.log(jnp.asarray([0.1])),
        "qtot_reference": jnp.log(jnp.asarray(0.8)),
        "relative_floor": 1.0e-8,
    }
    initial_theta = fixed_support_filter_theta(
        **{key: kwargs[key] for key in (
            "formula_matrix",
            "formula_matrix_cond_active",
            "element_inventory_target",
        )},
        q=kwargs["q_reference"],
        r=kwargs["r_reference"],
        qtot=kwargs["qtot_reference"],
        relative_floor=kwargs["relative_floor"],
    )
    q, r, qtot, positive, negative = fixed_support_full_restoration(
        **kwargs,
        proximity_weight=1.0e-3,
        max_abs_primal_step=2.0,
        passes=4,
    )
    final_theta = fixed_support_filter_theta(
        formula_matrix=kwargs["formula_matrix"],
        formula_matrix_cond_active=kwargs["formula_matrix_cond_active"],
        element_inventory_target=kwargs["element_inventory_target"],
        q=q,
        r=r,
        qtot=qtot,
        relative_floor=kwargs["relative_floor"],
    )

    assert bool(jnp.all(positive > 0.0))
    assert bool(jnp.all(negative > 0.0))
    assert float(final_theta) < float(initial_theta)


def test_amount_space_restoration_reduces_primal_infeasibility():
    kwargs = {
        "formula_matrix": jnp.asarray([[1.0, 1.0]]),
        "formula_matrix_cond_active": jnp.asarray([[1.0]]),
        "element_inventory_target": jnp.asarray([1.0]),
        "q_reference": jnp.log(jnp.asarray([0.05, 0.05])),
        "r_reference": jnp.log(jnp.asarray([0.05])),
        "qtot_reference": jnp.log(jnp.asarray(0.8)),
        "relative_floor": 1.0e-8,
    }
    initial_theta = fixed_support_filter_theta(
        formula_matrix=kwargs["formula_matrix"],
        formula_matrix_cond_active=kwargs["formula_matrix_cond_active"],
        element_inventory_target=kwargs["element_inventory_target"],
        q=kwargs["q_reference"],
        r=kwargs["r_reference"],
        qtot=kwargs["qtot_reference"],
        relative_floor=kwargs["relative_floor"],
    )
    q, r, qtot, positive, negative = fixed_support_amount_space_restoration(
        **kwargs,
        proximity_weight=1.0e-3,
        max_abs_primal_step=10.0,
        passes=8,
    )
    final_theta = fixed_support_filter_theta(
        formula_matrix=kwargs["formula_matrix"],
        formula_matrix_cond_active=kwargs["formula_matrix_cond_active"],
        element_inventory_target=kwargs["element_inventory_target"],
        q=q,
        r=r,
        qtot=qtot,
        relative_floor=kwargs["relative_floor"],
    )

    assert jnp.all(jnp.isfinite(q))
    assert jnp.all(jnp.isfinite(r))
    assert jnp.isfinite(qtot)
    assert jnp.all(positive > 0.0)
    assert jnp.all(negative > 0.0)
    assert float(final_theta) < 0.1 * float(initial_theta)


def test_amount_space_restoration_keeps_phase_entry_proximity_reference():
    ag = jnp.asarray([[1.0, 1.0]])
    ac = jnp.asarray([[1.0]])
    target = jnp.asarray([1.0])
    q0 = jnp.log(jnp.asarray([0.05, 0.05]))
    r0 = jnp.log(jnp.asarray([0.05]))
    qtot0 = jnp.log(jnp.asarray(0.8))
    common = {
        "formula_matrix": ag,
        "formula_matrix_cond_active": ac,
        "element_inventory_target": target,
        "relative_floor": 1.0e-8,
        "proximity_weight": 1.0e-3,
        "max_abs_primal_step": 10.0,
        "passes": 1,
    }
    q1, r1, qtot1, _positive, _negative = fixed_support_amount_space_restoration(
        **common,
        q_reference=q0,
        r_reference=r0,
        qtot_reference=qtot0,
    )
    q2, r2, qtot2, _positive, _negative = fixed_support_amount_space_restoration(
        **common,
        q_reference=q1,
        r_reference=r1,
        qtot_reference=qtot1,
        q_proximity_reference=q0,
        r_proximity_reference=r0,
        qtot_proximity_reference=qtot0,
    )

    theta1 = fixed_support_filter_theta(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        q=q1,
        r=r1,
        qtot=qtot1,
        relative_floor=1.0e-8,
    )
    theta2 = fixed_support_filter_theta(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        q=q2,
        r=r2,
        qtot=qtot2,
        relative_floor=1.0e-8,
    )

    assert float(theta2) < float(theta1)


def test_ipopt_bound_multiplier_return_uses_linearized_complementarity_step():
    updated, alpha, reset = fixed_support_ipopt_bound_multiplier_update(
        current_amount=jnp.asarray([0.2]),
        restored_amount=jnp.asarray([0.1]),
        current_multiplier=jnp.asarray([0.5]),
        barrier=jnp.asarray(0.1),
    )

    assert float(alpha) == 1.0
    assert not bool(reset)
    assert float(updated[0]) == 0.75


def test_ipopt_bound_multiplier_return_resets_excessive_multipliers():
    updated, _alpha, reset = fixed_support_ipopt_bound_multiplier_update(
        current_amount=jnp.asarray([1.0e-12]),
        restored_amount=jnp.asarray([1.0e-15]),
        current_multiplier=jnp.asarray([1.0]),
        barrier=jnp.asarray(1.0e-5),
    )

    assert bool(reset)
    assert jnp.array_equal(updated, jnp.ones_like(updated))


def test_ipopt_bound_multiplier_return_uses_requested_fraction_to_boundary():
    updated, alpha, reset = fixed_support_ipopt_bound_multiplier_update(
        current_amount=jnp.asarray([1.0]),
        restored_amount=jnp.asarray([2.0]),
        current_multiplier=jnp.asarray([1.0]),
        barrier=jnp.asarray(0.0),
        fraction_to_boundary=0.9,
    )

    assert float(alpha) == 0.45
    assert not bool(reset)
    assert jnp.allclose(updated, jnp.asarray([0.1]))


def test_ipopt_restoration_dual_return_satisfies_compatible_stationarity():
    returned_lambda, returned_rho, alpha, bound_reset, equality_reset = (
        fixed_support_ipopt_restoration_dual_return(
            formula_matrix=jnp.asarray([[1.0]]),
            formula_matrix_cond_active=jnp.asarray([[1.0]]),
            restored_q=jnp.asarray([1.5]),
            restored_r=jnp.log(jnp.asarray([0.2])),
            restored_qtot=jnp.asarray(0.0),
            qtot_reference=jnp.asarray(0.0),
            gas_stationarity_source=jnp.asarray([0.5]),
            condensate_standard_source=jnp.asarray([2.5]),
            current_r=jnp.log(jnp.asarray([0.2])),
            current_rho=jnp.log(jnp.asarray([0.5])),
            barrier=jnp.asarray(0.1),
        )
    )

    assert jnp.allclose(returned_lambda, jnp.asarray([2.0]))
    assert jnp.allclose(jnp.exp(returned_rho), jnp.asarray([0.5]))
    assert float(alpha) == 1.0
    assert not bool(bound_reset)
    assert not bool(equality_reset)


def test_ipopt_restoration_dual_return_resets_large_equality_multiplier():
    returned_lambda, _rho, _alpha, _bound_reset, equality_reset = (
        fixed_support_ipopt_restoration_dual_return(
            formula_matrix=jnp.asarray([[1.0]]),
            formula_matrix_cond_active=jnp.asarray([[1.0]]),
            restored_q=jnp.asarray([1.5]),
            restored_r=jnp.log(jnp.asarray([0.2])),
            restored_qtot=jnp.asarray(0.0),
            qtot_reference=jnp.asarray(0.0),
            gas_stationarity_source=jnp.asarray([0.5]),
            condensate_standard_source=jnp.asarray([2.5]),
            current_r=jnp.log(jnp.asarray([0.2])),
            current_rho=jnp.log(jnp.asarray([0.5])),
            barrier=jnp.asarray(0.1),
            equality_multiplier_reset_threshold=1.0,
        )
    )

    assert bool(equality_reset)
    assert jnp.array_equal(returned_lambda, jnp.zeros_like(returned_lambda))


def test_ipopt_restoration_dual_return_is_jittable_and_minimum_norm():
    def dual_return(restored_q):
        return fixed_support_ipopt_restoration_dual_return(
            formula_matrix=jnp.asarray([[1.0], [1.0]]),
            formula_matrix_cond_active=jnp.empty((2, 0), dtype=jnp.float64),
            restored_q=restored_q,
            restored_r=jnp.empty((0,), dtype=jnp.float64),
            restored_qtot=jnp.asarray(0.0),
            qtot_reference=jnp.asarray(0.0),
            gas_stationarity_source=jnp.asarray([0.0]),
            condensate_standard_source=jnp.empty((0,), dtype=jnp.float64),
            current_r=jnp.empty((0,), dtype=jnp.float64),
            current_rho=jnp.empty((0,), dtype=jnp.float64),
            barrier=jnp.asarray(0.1),
        )

    returned_lambda, _rho, _alpha, bound_reset, equality_reset = jax.jit(
        dual_return
    )(jnp.asarray([2.0]))

    assert jnp.allclose(returned_lambda, jnp.asarray([1.0, 1.0]))
    assert not bool(bound_reset)
    assert not bool(equality_reset)


def test_restoration_phase_exit_requires_all_return_conditions():
    kwargs = {
        "selected_amount_restoration": True,
        "trial_theta": 0.08,
        "entry_theta": 1.0,
        "theta_reduction": 0.9,
        "budget_relative_residual_max": 1.0e-8,
        "budget_relative_tolerance": 1.0e-6,
        "total_density_residual": 1.0e-8,
        "total_density_tolerance": 1.0e-6,
        "original_filter_accepted": True,
    }

    assert bool(fixed_support_restoration_phase_exit(**kwargs))
    assert not bool(
        fixed_support_restoration_phase_exit(
            **{**kwargs, "original_filter_accepted": False}
        )
    )
    assert not bool(
        fixed_support_restoration_phase_exit(
            **{**kwargs, "budget_relative_residual_max": 1.0e-3}
        )
    )


def test_restoration_phase_transition_enters_exits_and_counts_down():
    entered, active, cooldown = fixed_support_restoration_phase_transition(
        phase_active=False,
        cooldown=0,
        normal_iteration_attempted=False,
        selected_amount_restoration=True,
        phase_exit=False,
        cooldown_iterations=2,
    )
    assert bool(entered)
    assert bool(active)
    assert int(cooldown) == 0

    entered, active, cooldown = fixed_support_restoration_phase_transition(
        phase_active=active,
        cooldown=cooldown,
        normal_iteration_attempted=False,
        selected_amount_restoration=True,
        phase_exit=True,
        cooldown_iterations=2,
    )
    assert not bool(entered)
    assert not bool(active)
    assert int(cooldown) == 2

    _entered, _active, cooldown = fixed_support_restoration_phase_transition(
        phase_active=active,
        cooldown=cooldown,
        normal_iteration_attempted=True,
        selected_amount_restoration=False,
        phase_exit=False,
        cooldown_iterations=2,
    )
    assert int(cooldown) == 1
