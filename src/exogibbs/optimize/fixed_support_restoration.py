"""Slack-based feasibility restoration for fixed-support PD-IPM."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp


def fixed_support_ipopt_bound_multiplier_update(
    *,
    current_amount: Any,
    restored_amount: Any,
    current_multiplier: Any,
    barrier: Any,
    fraction_to_boundary: float = 0.995,
    reset_threshold: float = 1.0e3,
) -> tuple[Any, Any, Any]:
    """Map Ipopt's restoration bound-multiplier return step to ``m, eta``."""

    if not 0.0 < fraction_to_boundary <= 1.0:
        raise ValueError("fraction_to_boundary must be in the interval (0, 1].")

    amount = jnp.asarray(current_amount)
    dtype = amount.dtype
    restored = jnp.asarray(restored_amount, dtype=dtype)
    multiplier = jnp.asarray(current_multiplier, dtype=dtype)
    mu = jnp.asarray(barrier, dtype=dtype)
    delta = (mu - restored * multiplier) / jnp.maximum(
        amount,
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    alpha_bound = jnp.min(
        jnp.where(delta < 0.0, -multiplier / delta, jnp.inf),
        initial=jnp.asarray(jnp.inf, dtype=dtype),
    )
    alpha = jnp.minimum(
        1.0,
        jnp.asarray(fraction_to_boundary, dtype=dtype) * alpha_bound,
    )
    updated = multiplier + alpha * delta
    reset = jnp.max(updated, initial=0.0) > jnp.asarray(
        reset_threshold, dtype=dtype
    )
    updated = jnp.where(reset, jnp.ones_like(updated), updated)
    return updated, alpha, reset


def fixed_support_ipopt_restoration_dual_return(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    restored_q: Any,
    restored_r: Any,
    restored_qtot: Any,
    qtot_reference: Any,
    gas_stationarity_source: Any,
    condensate_standard_source: Any,
    current_r: Any,
    current_rho: Any,
    barrier: Any,
    fraction_to_boundary: float = 0.995,
    bound_multiplier_reset_threshold: float = 1.0e3,
    equality_multiplier_reset_threshold: float = float("inf"),
) -> tuple[Any, Any, Any, Any, Any]:
    """Return original R-GIE duals after an amount restoration phase.

    The condensate bound multipliers use Ipopt's linearized complementarity
    return.  Equality multipliers are the column-scaled minimum-norm solution
    of the stacked gas/condensate stationarity least-squares problem.  A finite
    equality threshold resets the entire equality multiplier vector to zero.
    """

    q = jnp.asarray(restored_q)
    dtype = q.dtype
    r = jnp.asarray(restored_r, dtype=dtype)
    qtot = jnp.asarray(restored_qtot, dtype=dtype)
    qtot_ref = jnp.asarray(qtot_reference, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    gas_source = jnp.asarray(gas_stationarity_source, dtype=dtype)
    cond_source = jnp.asarray(condensate_standard_source, dtype=dtype)
    old_r = jnp.asarray(current_r, dtype=dtype)
    old_rho = jnp.asarray(current_rho, dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    eta, alpha_dual, bound_reset = fixed_support_ipopt_bound_multiplier_update(
        current_amount=jnp.exp(old_r),
        restored_amount=jnp.exp(r),
        current_multiplier=jnp.exp(old_rho),
        barrier=jnp.asarray(barrier, dtype=dtype),
        fraction_to_boundary=fraction_to_boundary,
        reset_threshold=bound_multiplier_reset_threshold,
    )
    eta = jnp.maximum(eta, tiny)
    rho = jnp.log(eta)

    stationarity_matrix = jnp.concatenate([ag.T, ac.T], axis=0)
    stationarity_rhs = jnp.concatenate(
        [
            q + gas_source + qtot_ref - qtot,
            cond_source - eta,
        ]
    )
    column_scale = jnp.maximum(
        jnp.linalg.norm(stationarity_matrix, axis=0),
        tiny,
    )
    scaled_matrix = stationarity_matrix / column_scale[None, :]
    scaled_lambda = jnp.linalg.lstsq(
        scaled_matrix,
        stationarity_rhs,
        rcond=None,
    )[0]
    candidate_lambda = scaled_lambda / column_scale
    lambda_finite = jnp.all(jnp.isfinite(candidate_lambda))
    lambda_max = jnp.max(
        jnp.abs(candidate_lambda),
        initial=jnp.asarray(0.0, dtype=dtype),
    )
    equality_reset = (~lambda_finite) | (
        lambda_max
        > jnp.asarray(equality_multiplier_reset_threshold, dtype=dtype)
    )
    returned_lambda = jnp.where(
        equality_reset,
        jnp.zeros_like(candidate_lambda),
        candidate_lambda,
    )
    return returned_lambda, rho, alpha_dual, bound_reset, equality_reset


def fixed_support_restoration_phase_exit(
    *,
    selected_amount_restoration: Any,
    trial_theta: Any,
    entry_theta: Any,
    theta_reduction: Any,
    budget_relative_residual_max: Any,
    budget_relative_tolerance: Any,
    total_density_residual: Any,
    total_density_tolerance: Any,
    original_filter_accepted: Any,
) -> Any:
    """Return whether an accepted amount step may leave restoration mode."""

    theta = jnp.asarray(trial_theta)
    dtype = theta.dtype
    return (
        jnp.asarray(selected_amount_restoration, dtype=bool)
        & jnp.isfinite(theta)
        & (theta <= jnp.asarray(theta_reduction, dtype=dtype) * entry_theta)
        & (
            jnp.asarray(budget_relative_residual_max, dtype=dtype)
            <= jnp.asarray(budget_relative_tolerance, dtype=dtype)
        )
        & (
            jnp.asarray(total_density_residual, dtype=dtype)
            <= jnp.asarray(total_density_tolerance, dtype=dtype)
        )
        & jnp.asarray(original_filter_accepted, dtype=bool)
    )


def fixed_support_restoration_phase_transition(
    *,
    phase_active: Any,
    cooldown: Any,
    normal_iteration_attempted: Any,
    selected_amount_restoration: Any,
    phase_exit: Any,
    cooldown_iterations: int,
) -> tuple[Any, Any, Any]:
    """Advance the fixed-shape normal/restoration/cooldown lifecycle."""

    active = jnp.asarray(phase_active, dtype=bool)
    cooldown_value = jnp.asarray(cooldown, dtype=jnp.int32)
    selected_amount = jnp.asarray(selected_amount_restoration, dtype=bool)
    exited = jnp.asarray(phase_exit, dtype=bool)
    entered = selected_amount & (~active)
    next_active = (active | entered) & (~exited)
    next_cooldown = jnp.where(
        exited,
        jnp.asarray(cooldown_iterations, dtype=jnp.int32),
        jnp.where(
            jnp.asarray(normal_iteration_attempted, dtype=bool)
            & (~selected_amount)
            & (cooldown_value > 0),
            cooldown_value - jnp.asarray(1, dtype=jnp.int32),
            cooldown_value,
        ),
    )
    return entered, next_active, next_cooldown


def fixed_support_full_restoration(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    element_inventory_target: Any,
    q_reference: Any,
    r_reference: Any,
    qtot_reference: Any,
    relative_floor: Any,
    proximity_weight: Any,
    max_abs_primal_step: Any,
    passes: int,
    slack_penalty: float = 1.0,
    slack_barrier: float = 1.0e-3,
) -> tuple[Any, Any, Any, Any, Any]:
    """Approximately solve the positive/negative-slack restoration NLP."""

    q0 = jnp.asarray(q_reference)
    dtype = q0.dtype
    r0 = jnp.asarray(r_reference, dtype=dtype)
    qtot0 = jnp.asarray(qtot_reference, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    target = jnp.asarray(element_inventory_target, dtype=dtype)
    floor = jnp.maximum(
        jnp.asarray(relative_floor, dtype=dtype),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    budget_scale = jnp.where(
        target > 0.0,
        jnp.maximum(jnp.abs(target), floor),
        jnp.asarray(1.0, dtype=dtype),
    )
    mu = jnp.asarray(slack_barrier, dtype=dtype)
    penalty = jnp.asarray(slack_penalty, dtype=dtype)
    proximity = jnp.maximum(jnp.asarray(proximity_weight, dtype=dtype), 1.0e-12)
    limit = jnp.maximum(jnp.asarray(max_abs_primal_step, dtype=dtype), 0.0)

    def constraints_and_jacobian(q, r, qtot):
        n = jnp.exp(q)
        m = jnp.exp(r)
        ntot = jnp.exp(qtot)
        budget = (ag @ n + ac @ m - target) / budget_scale
        total_scale = jnp.maximum(jnp.abs(ntot), floor)
        total = (jnp.sum(n) - ntot) / total_scale
        jac_budget = jnp.concatenate(
            [
                ag * n[None, :] / budget_scale[:, None],
                ac * m[None, :] / budget_scale[:, None],
                jnp.zeros((target.shape[0], 1), dtype=dtype),
            ],
            axis=1,
        )
        jac_total = jnp.concatenate(
            [n / total_scale, jnp.zeros_like(m), jnp.asarray([-ntot / total_scale])]
        )[None, :]
        return jnp.concatenate([budget, jnp.reshape(total, (1,))]), jnp.concatenate(
            [jac_budget, jac_total], axis=0
        )

    c0, _ = constraints_and_jacobian(q0, r0, qtot0)
    slack_center = jnp.sqrt(mu)
    positive0 = jnp.maximum(c0, 0.0) + slack_center
    negative0 = jnp.maximum(-c0, 0.0) + slack_center
    dual0 = jnp.zeros_like(c0)

    def body(_index, state):
        q, r, qtot, positive, negative, dual = state
        constraints, jacobian = constraints_and_jacobian(q, r, qtot)
        displacement = jnp.concatenate([q - q0, r - r0, jnp.reshape(qtot - qtot0, (1,))])
        residual_primal = constraints - positive + negative
        residual_d = proximity * displacement + jacobian.T @ dual
        residual_positive = penalty - mu / positive - dual
        residual_negative = penalty - mu / negative + dual
        positive_scale = positive * positive / mu
        negative_scale = negative * negative / mu
        hessian = proximity * jnp.eye(displacement.shape[0], dtype=dtype)
        kkt = jnp.block(
            [
                [hessian, jacobian.T],
                [
                    jacobian,
                    -jnp.diag(positive_scale + negative_scale),
                ],
            ]
        )
        rhs_constraint = (
            -residual_primal
            - positive_scale * residual_positive
            + negative_scale * residual_negative
        )
        rhs = jnp.concatenate([-residual_d, rhs_constraint])
        solution = jnp.linalg.lstsq(kkt, rhs, rcond=None)[0]
        solution = jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
        delta_d = solution[: displacement.shape[0]]
        delta_dual = solution[displacement.shape[0] :]
        delta_positive = positive_scale * (-residual_positive + delta_dual)
        delta_negative = negative_scale * (-residual_negative - delta_dual)
        alpha_positive = jnp.min(
            jnp.where(delta_positive < 0.0, -positive / delta_positive, jnp.inf),
            initial=jnp.asarray(jnp.inf, dtype=dtype),
        )
        alpha_negative = jnp.min(
            jnp.where(delta_negative < 0.0, -negative / delta_negative, jnp.inf),
            initial=jnp.asarray(jnp.inf, dtype=dtype),
        )
        primal_max = jnp.max(
            jnp.abs(delta_d), initial=jnp.asarray(0.0, dtype=dtype)
        )
        alpha = jnp.minimum(
            1.0,
            jnp.minimum(
                0.995 * jnp.minimum(alpha_positive, alpha_negative),
                limit / jnp.maximum(primal_max, jnp.asarray(1.0e-300, dtype=dtype)),
            ),
        )
        delta_q = delta_d[: q.shape[0]]
        delta_r = delta_d[q.shape[0] : q.shape[0] + r.shape[0]]
        delta_qtot = delta_d[-1]
        return (
            q + alpha * delta_q,
            r + alpha * delta_r,
            qtot + alpha * delta_qtot,
            positive + alpha * delta_positive,
            negative + alpha * delta_negative,
            dual + alpha * delta_dual,
        )

    qf, rf, qtotf, positivef, negativef, _dualf = jax.lax.fori_loop(
        0,
        int(passes),
        body,
        (q0, r0, qtot0, positive0, negative0, dual0),
    )
    return qf, rf, qtotf, positivef, negativef


def fixed_support_amount_space_restoration(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    element_inventory_target: Any,
    q_reference: Any,
    r_reference: Any,
    qtot_reference: Any,
    relative_floor: Any,
    proximity_weight: Any,
    max_abs_primal_step: Any,
    passes: int,
    slack_penalty: float = 1.0e3,
    slack_barrier: float = 1.0e-3,
    scale_floor_fraction: float = 1.0e-12,
    q_proximity_reference: Optional[Any] = None,
    r_proximity_reference: Optional[Any] = None,
    qtot_proximity_reference: Optional[Any] = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Solve the elastic restoration problem in physical amount coordinates."""

    q0 = jnp.asarray(q_reference)
    dtype = q0.dtype
    r0 = jnp.asarray(r_reference, dtype=dtype)
    qtot0 = jnp.asarray(qtot_reference, dtype=dtype)
    qref = q0 if q_proximity_reference is None else jnp.asarray(
        q_proximity_reference, dtype=dtype
    )
    rref = r0 if r_proximity_reference is None else jnp.asarray(
        r_proximity_reference, dtype=dtype
    )
    qtotref = qtot0 if qtot_proximity_reference is None else jnp.asarray(
        qtot_proximity_reference, dtype=dtype
    )
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    target = jnp.asarray(element_inventory_target, dtype=dtype)
    n0 = jnp.exp(q0)
    m0 = jnp.exp(r0)
    ntot0 = jnp.exp(qtot0)
    x0 = jnp.concatenate([n0, m0, jnp.reshape(ntot0, (1,))])
    nref = jnp.exp(qref)
    mref = jnp.exp(rref)
    ntotref = jnp.exp(qtotref)
    xref = jnp.concatenate([nref, mref, jnp.reshape(ntotref, (1,))])
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    floor = jnp.maximum(jnp.asarray(relative_floor, dtype=dtype), tiny)
    budget_scale = jnp.where(target > 0.0, jnp.maximum(target, floor), 1.0)
    total_scale = jnp.maximum(ntotref, floor)
    row_scale = jnp.concatenate(
        [1.0 / budget_scale, jnp.reshape(1.0 / total_scale, (1,))]
    )
    jacobian_unscaled = jnp.block(
        [
            [ag, ac, jnp.zeros((target.shape[0], 1), dtype=dtype)],
            [
                jnp.ones((1, n0.shape[0]), dtype=dtype),
                jnp.zeros((1, m0.shape[0]), dtype=dtype),
                -jnp.ones((1, 1), dtype=dtype),
            ],
        ]
    )
    jacobian = row_scale[:, None] * jacobian_unscaled
    constraint_offset = jnp.concatenate([-target, jnp.zeros((1,), dtype=dtype)])

    capacity = jnp.min(
        jnp.where(ac > 0.0, target[:, None] / ac, jnp.inf),
        axis=0,
    )
    amount_floor = jnp.asarray(scale_floor_fraction, dtype=dtype)
    gas_scale = jnp.maximum(nref, amount_floor * ntotref)
    cond_scale = jnp.maximum(
        mref,
        amount_floor * jnp.where(jnp.isfinite(capacity), capacity, ntotref),
    )
    x_scale = jnp.maximum(
        jnp.concatenate([gas_scale, cond_scale, jnp.reshape(ntotref, (1,))]),
        tiny,
    )
    d2 = 1.0 / (x_scale * x_scale)
    mu = jnp.asarray(slack_barrier, dtype=dtype)
    penalty = jnp.asarray(slack_penalty, dtype=dtype)
    proximity = jnp.maximum(
        jnp.asarray(proximity_weight, dtype=dtype) * jnp.sqrt(mu), tiny
    )
    step_limit = jnp.maximum(jnp.asarray(max_abs_primal_step, dtype=dtype), 0.0)

    def constraints(x):
        return row_scale * (jacobian_unscaled @ x + constraint_offset)

    c0 = constraints(x0)
    slack_center = jnp.sqrt(mu)
    positive0 = jnp.maximum(c0, 0.0) + slack_center
    negative0 = jnp.maximum(-c0, 0.0) + slack_center
    dual0 = jnp.zeros_like(c0)

    def body(_index, state):
        x, positive, negative, dual = state
        c = constraints(x)
        residual_primal = c - positive + negative
        residual_x = proximity * d2 * (x - xref) - mu / x + jacobian.T @ dual
        residual_positive = penalty - mu / positive - dual
        residual_negative = penalty - mu / negative + dual
        positive_scale = positive * positive / mu
        negative_scale = negative * negative / mu
        slack_scale = positive_scale + negative_scale
        hessian_diagonal = proximity * d2 + mu / (x * x)
        hessian_inverse = 1.0 / jnp.maximum(hessian_diagonal, tiny)
        rhs_constraint = (
            -residual_primal
            - positive_scale * residual_positive
            + negative_scale * residual_negative
        )
        weighted_jacobian = jacobian * hessian_inverse[None, :]
        schur = weighted_jacobian @ jacobian.T + jnp.diag(slack_scale)
        schur_rhs = -rhs_constraint - weighted_jacobian @ residual_x
        delta_dual = jnp.linalg.lstsq(schur, schur_rhs, rcond=None)[0]
        delta_x = hessian_inverse * (-residual_x - jacobian.T @ delta_dual)
        delta_positive = positive_scale * (-residual_positive + delta_dual)
        delta_negative = negative_scale * (-residual_negative - delta_dual)

        def fraction_to_boundary(value, direction):
            return jnp.min(
                jnp.where(direction < 0.0, -value / direction, jnp.inf),
                initial=jnp.asarray(jnp.inf, dtype=dtype),
            )

        alpha_bound = 0.995 * jnp.minimum(
            fraction_to_boundary(x, delta_x),
            jnp.minimum(
                fraction_to_boundary(positive, delta_positive),
                fraction_to_boundary(negative, delta_negative),
            ),
        )
        relative_step = jnp.max(jnp.abs(delta_x) / x_scale, initial=0.0)
        alpha_trust = step_limit / jnp.maximum(relative_step, tiny)
        alpha = jnp.minimum(1.0, jnp.minimum(alpha_bound, alpha_trust))
        return (
            x + alpha * delta_x,
            positive + alpha * delta_positive,
            negative + alpha * delta_negative,
            dual + alpha * delta_dual,
        )

    xf, positivef, negativef, _dualf = jax.lax.fori_loop(
        0,
        int(passes),
        body,
        (x0, positive0, negative0, dual0),
    )
    n_end = n0.shape[0]
    m_end = n_end + m0.shape[0]
    qf = jnp.log(jnp.maximum(xf[:n_end], tiny))
    rf = jnp.log(jnp.maximum(xf[n_end:m_end], tiny))
    qtotf = jnp.log(jnp.maximum(xf[-1], tiny))
    return qf, rf, qtotf, positivef, negativef


__all__ = [
    "fixed_support_amount_space_restoration",
    "fixed_support_full_restoration",
    "fixed_support_ipopt_bound_multiplier_update",
    "fixed_support_ipopt_restoration_dual_return",
    "fixed_support_restoration_phase_exit",
    "fixed_support_restoration_phase_transition",
]
