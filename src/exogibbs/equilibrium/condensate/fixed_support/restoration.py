"""Standalone persistent elastic restoration in physical amount coordinates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.filter import accept_to_history
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    amount_space_equality_jacobian,
    barrier_objective,
    filter_violation,
    physical_amounts,
    residual_components,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FilterState,
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    RestorationDirection,
    RestorationDirectionDiagnostics,
    RestorationDirectionResult,
    RestorationIterationResult,
    RestorationResiduals,
    RestorationSolveResult,
    RestorationState,
    RestorationTrialBatch,
    TerminalStatus,
)


def _validate_config(config: FixedSupportV2Config) -> None:
    restoration = config.restoration
    limits = config.limits
    if restoration.elastic_penalty <= 0.0:
        raise ValueError("elastic_penalty must be positive.")
    if restoration.proximity_weight < 0.0:
        raise ValueError("proximity_weight must be non-negative.")
    if restoration.amount_scale_floor_fraction <= 0.0:
        raise ValueError("amount_scale_floor_fraction must be positive.")
    if restoration.interior_push_fraction < 0.0:
        raise ValueError("interior_push_fraction must be non-negative.")
    if not 0.0 < restoration.fraction_to_boundary <= 1.0:
        raise ValueError("fraction_to_boundary must be in (0, 1].")
    if not 0.0 < restoration.backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must be in (0, 1).")
    if not 0.0 < restoration.armijo_fraction <= 1.0:
        raise ValueError("armijo_fraction must be in (0, 1].")
    if not 0.0 < restoration.return_dual_fraction_to_boundary <= 1.0:
        raise ValueError("return_dual_fraction_to_boundary must be in (0, 1].")
    if restoration.bound_multiplier_reset_threshold < 0.0:
        raise ValueError("bound_multiplier_reset_threshold must be non-negative.")
    if restoration.representation_floor <= 0.0:
        raise ValueError("representation_floor must be positive.")
    if restoration.representation_floor_injection_tolerance < 0.0:
        raise ValueError(
            "representation_floor_injection_tolerance must be non-negative."
        )
    if not 0.0 <= restoration.required_reduction < 1.0:
        raise ValueError("required_reduction must be in [0, 1).")
    if limits.max_restoration_line_search_trials < 1:
        raise ValueError("max_restoration_line_search_trials must be positive.")
    if limits.max_restoration_iterations < 0:
        raise ValueError("max_restoration_iterations must be non-negative.")


def restoration_constraint_jacobian(
    problem: FixedSupportProblem,
    row_scales,
):
    """Return the constant scaled Jacobian with respect to ``(n,m,ntot)``."""

    jacobian = amount_space_equality_jacobian(problem)
    scales = jnp.asarray(row_scales, dtype=jacobian.dtype)
    return scales[:, None] * jacobian


def restoration_constraint_offset(
    problem: FixedSupportProblem,
    row_scales,
):
    """Return the offset such that ``c(x) = J x + offset``."""

    target = jnp.asarray(problem.target_inventory)
    dtype = target.dtype
    unscaled = jnp.concatenate([-target, jnp.zeros((1,), dtype=dtype)])
    return jnp.asarray(row_scales, dtype=dtype) * unscaled


def restoration_equalities(
    problem: FixedSupportProblem,
    state: RestorationState,
):
    """Return ``c(x) - p + v`` for the elastic restoration NLP."""

    jacobian = restoration_constraint_jacobian(problem, state.row_scales)
    offset = restoration_constraint_offset(problem, state.row_scales)
    return (
        jacobian @ state.x
        + offset
        - state.positive_slack
        + state.negative_slack
    )


def _proximity_coefficient(state: RestorationState, config) -> jax.Array:
    return jnp.asarray(config.proximity_weight, dtype=state.x.dtype) * jnp.sqrt(
        state.restoration_mu
    )


def restoration_elastic_objective(
    state: RestorationState,
    config,
):
    """Return the elastic L1 plus scaled proximity objective."""

    displacement = (state.x - state.entry_x) / state.variable_scales
    proximity = 0.5 * _proximity_coefficient(state, config) * jnp.dot(
        displacement, displacement
    )
    elastic = jnp.asarray(config.elastic_penalty, dtype=state.x.dtype) * jnp.sum(
        state.positive_slack + state.negative_slack
    )
    return elastic + proximity


def restoration_barrier_objective(
    state: RestorationState,
    config,
):
    """Return the bound-barrier objective used for internal globalization."""

    barrier = -state.restoration_mu * (
        jnp.sum(jnp.log(state.x))
        + jnp.sum(jnp.log(state.positive_slack))
        + jnp.sum(jnp.log(state.negative_slack))
    )
    return restoration_elastic_objective(state, config) + barrier


def restoration_barrier_directional_derivative(
    state: RestorationState,
    direction: RestorationDirection,
    config,
):
    """Return the internal barrier-objective directional derivative."""

    zeta = _proximity_coefficient(state, config)
    gradient_x = (
        zeta * (state.x - state.entry_x) / (state.variable_scales**2)
        - state.restoration_mu / state.x
    )
    penalty = jnp.asarray(config.elastic_penalty, dtype=state.x.dtype)
    gradient_positive = penalty - state.restoration_mu / state.positive_slack
    gradient_negative = penalty - state.restoration_mu / state.negative_slack
    return (
        jnp.dot(gradient_x, direction.x)
        + jnp.dot(gradient_positive, direction.positive_slack)
        + jnp.dot(gradient_negative, direction.negative_slack)
    )


def restoration_residuals(
    problem: FixedSupportProblem,
    state: RestorationState,
    config,
) -> RestorationResiduals:
    """Evaluate all seven primal-dual restoration KKT blocks."""

    jacobian = restoration_constraint_jacobian(problem, state.row_scales)
    zeta = _proximity_coefficient(state, config)
    penalty = jnp.asarray(config.elastic_penalty, dtype=state.x.dtype)
    return RestorationResiduals(
        dual_x=(
            zeta * (state.x - state.entry_x) / (state.variable_scales**2)
            + jacobian.T @ state.equality_dual
            - state.lower_bound_dual_x
        ),
        dual_positive=(
            penalty - state.equality_dual - state.lower_bound_dual_positive
        ),
        dual_negative=(
            penalty + state.equality_dual - state.lower_bound_dual_negative
        ),
        equality=restoration_equalities(problem, state),
        complementarity_x=(
            state.x * state.lower_bound_dual_x - state.restoration_mu
        ),
        complementarity_positive=(
            state.positive_slack * state.lower_bound_dual_positive
            - state.restoration_mu
        ),
        complementarity_negative=(
            state.negative_slack * state.lower_bound_dual_negative
            - state.restoration_mu
        ),
    )


def restoration_residual_vector(residual: RestorationResiduals):
    """Concatenate KKT blocks in the restoration dense-audit order."""

    return jnp.concatenate([jnp.ravel(block) for block in residual])


def _scaled_euclidean_norm(value):
    """Evaluate a 2-norm without squaring very large finite entries."""

    flat = jnp.ravel(jnp.asarray(value))
    scale = jnp.max(jnp.abs(flat), initial=0.0)
    return jnp.where(
        scale == 0.0,
        jnp.asarray(0.0, dtype=flat.dtype),
        scale * jnp.linalg.norm(flat / scale),
    )


def _relative_residual(residual, matrix, direction, rhs):
    """Return ``||r|| / (||A|| ||d|| + ||b||)`` without overflow."""

    residual_norm = _scaled_euclidean_norm(residual)
    matrix_norm = _scaled_euclidean_norm(matrix)
    direction_norm = _scaled_euclidean_norm(direction)
    rhs_norm = _scaled_euclidean_norm(rhs)
    negative_infinity = jnp.asarray(-jnp.inf, dtype=residual_norm.dtype)

    def logarithm(value):
        return jnp.where(value > 0.0, jnp.log(value), negative_infinity)

    log_denominator = jnp.logaddexp(
        logarithm(matrix_norm) + logarithm(direction_norm),
        logarithm(rhs_norm),
    )
    relative = jnp.exp(logarithm(residual_norm) - log_denominator)
    return residual_norm, jnp.where(residual_norm == 0.0, 0.0, relative)


def restoration_kkt_jacobian(
    problem: FixedSupportProblem,
    state: RestorationState,
    config,
):
    """Return the dense restoration KKT Jacobian for CPU contract audits."""

    x = state.x
    p = state.positive_slack
    v = state.negative_slack
    zx = state.lower_bound_dual_x
    zp = state.lower_bound_dual_positive
    zv = state.lower_bound_dual_negative
    jacobian = restoration_constraint_jacobian(problem, state.row_scales)
    nx, nc = x.shape[0], p.shape[0]
    zeros = lambda rows, columns: jnp.zeros((rows, columns), dtype=x.dtype)
    hessian = jnp.diag(
        _proximity_coefficient(state, config) / (state.variable_scales**2)
    )
    return jnp.block(
        [
            [hessian, zeros(nx, nc), zeros(nx, nc), jacobian.T, -jnp.eye(nx), zeros(nx, nc), zeros(nx, nc)],
            [zeros(nc, nx), zeros(nc, nc), zeros(nc, nc), -jnp.eye(nc), zeros(nc, nx), -jnp.eye(nc), zeros(nc, nc)],
            [zeros(nc, nx), zeros(nc, nc), zeros(nc, nc), jnp.eye(nc), zeros(nc, nx), zeros(nc, nc), -jnp.eye(nc)],
            [jacobian, -jnp.eye(nc), jnp.eye(nc), zeros(nc, nc), zeros(nc, nx), zeros(nc, nc), zeros(nc, nc)],
            [jnp.diag(zx), zeros(nx, nc), zeros(nx, nc), zeros(nx, nc), jnp.diag(x), zeros(nx, nc), zeros(nx, nc)],
            [zeros(nc, nx), jnp.diag(zp), zeros(nc, nc), zeros(nc, nc), zeros(nc, nx), jnp.diag(p), zeros(nc, nc)],
            [zeros(nc, nx), zeros(nc, nc), jnp.diag(zv), zeros(nc, nc), zeros(nc, nx), zeros(nc, nc), jnp.diag(v)],
        ]
    )


def _zero_direction(state: RestorationState) -> RestorationDirection:
    return RestorationDirection(
        x=jnp.zeros_like(state.x),
        positive_slack=jnp.zeros_like(state.positive_slack),
        negative_slack=jnp.zeros_like(state.negative_slack),
        equality_dual=jnp.zeros_like(state.equality_dual),
        lower_bound_dual_x=jnp.zeros_like(state.lower_bound_dual_x),
        lower_bound_dual_positive=jnp.zeros_like(
            state.lower_bound_dual_positive
        ),
        lower_bound_dual_negative=jnp.zeros_like(
            state.lower_bound_dual_negative
        ),
    )


def restoration_direction(
    problem: FixedSupportProblem,
    state: RestorationState,
    config,
) -> RestorationDirectionResult:
    """Solve the constant-Jacobian constraint-space Schur system."""

    residual = restoration_residuals(problem, state, config)
    jacobian = restoration_constraint_jacobian(problem, state.row_scales)
    x, p, v = state.x, state.positive_slack, state.negative_slack
    zx = state.lower_bound_dual_x
    zp = state.lower_bound_dual_positive
    zv = state.lower_bound_dual_negative
    zeta_diagonal = _proximity_coefficient(state, config) / (
        state.variable_scales**2
    )
    # These are algebraically ``1 / (H + z/x)``, ``1 / (zp/p)`` and
    # ``1 / (zv/v)``.  Forming ``z/x`` overflows for perfectly valid trace
    # amounts (for example x=1e-300), so keep the same Schur equation in a
    # quotient order that remains representable.
    inverse_wx = x / (zeta_diagonal * x + zx)
    inverse_wp = p / zp
    inverse_wv = v / zv
    proximity_gradient = zeta_diagonal * (x - state.entry_x)
    jacobian_dual = jacobian.T @ state.equality_dual
    scaled_ax = (
        state.restoration_mu - x * (proximity_gradient + jacobian_dual)
    ) / (zeta_diagonal * x + zx)
    scaled_ap = (
        state.restoration_mu
        + p * (state.equality_dual - jnp.asarray(config.elastic_penalty, dtype=x.dtype))
    ) / zp
    scaled_av = (
        state.restoration_mu
        - v * (jnp.asarray(config.elastic_penalty, dtype=x.dtype) + state.equality_dual)
    ) / zv
    weighted_jacobian = jacobian * inverse_wx[None, :]
    schur = weighted_jacobian @ jacobian.T + jnp.diag(
        inverse_wp + inverse_wv
    )
    schur_rhs = (
        residual.equality
        + jacobian @ scaled_ax
        - scaled_ap
        + scaled_av
    )
    delta_y = jnp.linalg.solve(schur, schur_rhs)
    delta_x = scaled_ax - inverse_wx * (jacobian.T @ delta_y)
    delta_p = scaled_ap + inverse_wp * delta_y
    delta_v = scaled_av - inverse_wv * delta_y
    # Recover bound-dual directions from the stationarity rows.  This is the
    # same Newton system as complementarity back-substitution, but avoids
    # dividing by trace amounts a second time.
    delta_zx = residual.dual_x + zeta_diagonal * delta_x + jacobian.T @ delta_y
    delta_zp = residual.dual_positive - delta_y
    delta_zv = residual.dual_negative + delta_y
    raw_direction = RestorationDirection(
        x=delta_x,
        positive_slack=delta_p,
        negative_slack=delta_v,
        equality_dual=delta_y,
        lower_bound_dual_x=delta_zx,
        lower_bound_dual_positive=delta_zp,
        lower_bound_dual_negative=delta_zv,
    )
    flat_direction = jnp.concatenate([jnp.ravel(block) for block in raw_direction])
    raw_finite = jnp.all(jnp.isfinite(flat_direction))
    schur_residual = schur @ delta_y - schur_rhs
    schur_residual_norm, relative_schur_residual = _relative_residual(
        schur_residual, schur, delta_y, schur_rhs
    )
    singular_values = jnp.linalg.svd(schur, compute_uv=False)
    smallest = jnp.min(singular_values)
    largest = jnp.max(singular_values)
    condition = largest / jnp.maximum(
        smallest, jnp.asarray(jnp.finfo(x.dtype).tiny, dtype=x.dtype)
    )
    full_residual = jnp.concatenate(
        [
            residual.dual_x
            + zeta_diagonal * delta_x
            + jacobian.T @ delta_y
            - delta_zx,
            residual.dual_positive - delta_y - delta_zp,
            residual.dual_negative + delta_y - delta_zv,
            residual.equality + jacobian @ delta_x - delta_p + delta_v,
            residual.complementarity_x + zx * delta_x + x * delta_zx,
            residual.complementarity_positive + zp * delta_p + p * delta_zp,
            residual.complementarity_negative + zv * delta_v + v * delta_zv,
        ]
    )
    full_matrix = restoration_kkt_jacobian(problem, state, config)
    full_rhs = restoration_residual_vector(residual)
    full_residual_norm, relative_full_residual = _relative_residual(
        full_residual, full_matrix, flat_direction, full_rhs
    )
    solve_ok = (
        raw_finite
        & jnp.isfinite(relative_schur_residual)
        & jnp.isfinite(relative_full_residual)
        & (relative_schur_residual <= config.relative_linear_solve_tolerance)
        & (relative_full_residual <= config.relative_linear_solve_tolerance)
    )
    direction = jax.tree_util.tree_map(
        lambda raw, zero: jnp.where(solve_ok, raw, zero),
        raw_direction,
        _zero_direction(state),
    )
    return RestorationDirectionResult(
        direction=direction,
        diagnostics=RestorationDirectionDiagnostics(
            raw_direction_finite=raw_finite,
            schur_residual_norm=schur_residual_norm,
            relative_schur_residual=relative_schur_residual,
            full_kkt_residual_norm=full_residual_norm,
            relative_full_kkt_residual=relative_full_residual,
            smallest_singular_value=smallest,
            largest_singular_value=largest,
            condition_estimate=condition,
        ),
        status=jnp.where(
            solve_ok,
            jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
            jnp.asarray(
                TerminalStatus.RESTORATION_LINEAR_SOLVE_FAILED,
                dtype=jnp.int32,
            ),
        ),
    )


def _fraction_to_boundary(value, direction, fraction):
    limit = jnp.min(
        jnp.where(direction < 0.0, -value / direction, jnp.inf),
        initial=jnp.asarray(jnp.inf, dtype=value.dtype),
    )
    return jnp.minimum(1.0, jnp.asarray(fraction, dtype=value.dtype) * limit)


def restoration_step_limits(
    state: RestorationState,
    direction: RestorationDirection,
    fraction: float,
):
    """Return separate primal and dual fraction-to-boundary limits."""

    primal = jnp.minimum(
        _fraction_to_boundary(state.x, direction.x, fraction),
        jnp.minimum(
            _fraction_to_boundary(
                state.positive_slack, direction.positive_slack, fraction
            ),
            _fraction_to_boundary(
                state.negative_slack, direction.negative_slack, fraction
            ),
        ),
    )
    dual = jnp.minimum(
        _fraction_to_boundary(
            state.lower_bound_dual_x, direction.lower_bound_dual_x, fraction
        ),
        jnp.minimum(
            _fraction_to_boundary(
                state.lower_bound_dual_positive,
                direction.lower_bound_dual_positive,
                fraction,
            ),
            _fraction_to_boundary(
                state.lower_bound_dual_negative,
                direction.lower_bound_dual_negative,
                fraction,
            ),
        ),
    )
    return primal, dual


def _broadcast(value, count):
    value = jnp.asarray(value)
    return jnp.broadcast_to(value, (count,) + value.shape)


def _trial_states(state, direction, alphas):
    count = alphas.shape[0]

    def advance(value, delta):
        value = jnp.asarray(value)
        return value[None, ...] + alphas.reshape(
            (count,) + (1,) * value.ndim
        ) * jnp.asarray(delta)[None, ...]

    entry_original = jax.tree_util.tree_map(
        lambda value: _broadcast(value, count), state.entry_original_state
    )
    return RestorationState(
        x=advance(state.x, direction.x),
        positive_slack=advance(state.positive_slack, direction.positive_slack),
        negative_slack=advance(state.negative_slack, direction.negative_slack),
        equality_dual=advance(state.equality_dual, direction.equality_dual),
        lower_bound_dual_x=advance(
            state.lower_bound_dual_x, direction.lower_bound_dual_x
        ),
        lower_bound_dual_positive=advance(
            state.lower_bound_dual_positive,
            direction.lower_bound_dual_positive,
        ),
        lower_bound_dual_negative=advance(
            state.lower_bound_dual_negative,
            direction.lower_bound_dual_negative,
        ),
        restoration_mu=_broadcast(state.restoration_mu, count),
        entry_x=_broadcast(state.entry_x, count),
        entry_original_state=entry_original,
        entry_phi=_broadcast(state.entry_phi, count),
        entry_theta=_broadcast(state.entry_theta, count),
        variable_scales=_broadcast(state.variable_scales, count),
        row_scales=_broadcast(state.row_scales, count),
        iteration=_broadcast(state.iteration + 1, count),
        accepted_iteration_count=_broadcast(
            state.accepted_iteration_count + 1, count
        ),
    )


def restoration_iteration(
    problem: FixedSupportProblem,
    state: RestorationState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> RestorationIterationResult:
    """Take at most one accepted internal step without reinitializing state."""

    _validate_config(config)
    direction_result = restoration_direction(
        problem, state, config.restoration
    )
    primal_limit, dual_limit = restoration_step_limits(
        state,
        direction_result.direction,
        config.restoration.fraction_to_boundary,
    )
    alpha_max = jnp.minimum(primal_limit, dual_limit)
    indices = jnp.arange(
        config.limits.max_restoration_line_search_trials, dtype=state.x.dtype
    )
    alphas = alpha_max * jnp.asarray(
        config.restoration.backtracking_factor, dtype=state.x.dtype
    ) ** indices
    trial_states = _trial_states(state, direction_result.direction, alphas)
    objectives = jax.vmap(
        lambda trial: restoration_barrier_objective(
            trial, config.restoration
        )
    )(trial_states)
    elastic_objectives = jax.vmap(
        lambda trial: restoration_elastic_objective(
            trial, config.restoration
        )
    )(trial_states)
    violations = jax.vmap(
        lambda trial: jnp.linalg.norm(
            restoration_equalities(problem, trial), ord=1
        )
    )(trial_states)
    current_objective = restoration_barrier_objective(
        state, config.restoration
    )
    current_violation = jnp.linalg.norm(
        restoration_equalities(problem, state), ord=1
    )
    derivative = restoration_barrier_directional_derivative(
        state, direction_result.direction, config.restoration
    )
    objective_acceptable = objectives <= (
        current_objective
        + config.restoration.armijo_fraction * alphas * derivative
    )
    constraint_acceptable = violations <= (
        current_violation + config.restoration.constraint_nonincrease_tolerance
    )
    finite_positive = jax.vmap(
        lambda trial: (
            jnp.isfinite(restoration_barrier_objective(trial, config.restoration))
            & jnp.all(jnp.isfinite(trial.x))
            & jnp.all(jnp.isfinite(trial.positive_slack))
            & jnp.all(jnp.isfinite(trial.negative_slack))
            & jnp.all(jnp.isfinite(trial.equality_dual))
            & jnp.all(jnp.isfinite(trial.lower_bound_dual_x))
            & jnp.all(jnp.isfinite(trial.lower_bound_dual_positive))
            & jnp.all(jnp.isfinite(trial.lower_bound_dual_negative))
            & jnp.all(trial.x > 0.0)
            & jnp.all(trial.positive_slack > 0.0)
            & jnp.all(trial.negative_slack > 0.0)
            & jnp.all(trial.lower_bound_dual_x > 0.0)
            & jnp.all(trial.lower_bound_dual_positive > 0.0)
            & jnp.all(trial.lower_bound_dual_negative > 0.0)
        )
    )(trial_states)
    direction_ok = direction_result.status == int(TerminalStatus.NOT_TERMINATED)
    accepted_mask = (
        finite_positive
        & objective_acceptable
        & constraint_acceptable
        & direction_ok
    )
    accepted = jnp.any(accepted_mask)
    selected_index = jnp.argmax(accepted_mask).astype(jnp.int32)
    selected_state = jax.tree_util.tree_map(
        lambda values: values[selected_index], trial_states
    )
    next_state = jax.tree_util.tree_map(
        lambda selected, current: jnp.where(accepted, selected, current),
        selected_state,
        state,
    )
    status = jnp.where(
        direction_ok,
        jnp.where(
            accepted,
            jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
            jnp.asarray(
                TerminalStatus.RESTORATION_LINE_SEARCH_FAILED,
                dtype=jnp.int32,
            ),
        ),
        direction_result.status,
    )
    return RestorationIterationResult(
        state=next_state,
        direction_result=direction_result,
        trials=RestorationTrialBatch(
            states=trial_states,
            alphas=alphas,
            elastic_objective=elastic_objectives,
            barrier_objective=objectives,
            equality_violation=violations,
            finite_and_positive=finite_positive,
            objective_acceptable=objective_acceptable,
            constraint_acceptable=constraint_acceptable,
            accepted=accepted_mask,
        ),
        accepted=accepted,
        selected_index=jnp.where(accepted, selected_index, -1),
        selected_alpha=jnp.where(
            accepted, alphas[selected_index], jnp.asarray(jnp.nan, dtype=alphas.dtype)
        ),
        status=status,
    )


def initialize_restoration(
    problem: FixedSupportProblem,
    original_state: OriginalState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> RestorationState:
    """Initialize persistent state exactly once on restoration entry."""

    _validate_config(config)
    amounts = physical_amounts(original_state)
    raw_x = jnp.concatenate(
        [amounts.gas, amounts.condensate, amounts.total_gas.reshape((1,))]
    )
    dtype = raw_x.dtype
    row_scales = jnp.concatenate(
        [
            jnp.asarray(problem.budget_row_scale, dtype=dtype),
            jnp.asarray(problem.total_density_row_scale, dtype=dtype).reshape((1,)),
        ]
    )
    ng = amounts.gas.shape[0]
    nc = amounts.condensate.shape[0]
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    target = jnp.asarray(problem.target_inventory, dtype=dtype)
    valid_capacity = (ac > 0.0) & (target[:, None] > 0.0)
    capacity = jnp.min(
        jnp.where(valid_capacity, target[:, None] / ac, jnp.inf), axis=0
    )
    scale_floor_fraction = jnp.asarray(
        config.restoration.amount_scale_floor_fraction, dtype=dtype
    )
    gas_scales = jnp.maximum(
        raw_x[:ng], scale_floor_fraction * raw_x[-1]
    )
    condensate_reference = jnp.where(
        jnp.isfinite(capacity), capacity, raw_x[-1]
    )
    condensate_scales = jnp.maximum(
        raw_x[ng : ng + nc],
        scale_floor_fraction * condensate_reference,
    )
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    variable_scales = jnp.maximum(
        jnp.concatenate([gas_scales, condensate_scales, raw_x[-1:]]),
        tiny,
    )
    # Restoration is an interior-point NLP.  Merely replacing an underflowed
    # amount by 1e-300 gives z=mu/x near 1e295 and destroys useful scaling.
    # Push every bound variable to a fixed fraction of its entry-derived scale,
    # then accept that representation only after the same inventory audit used
    # for the return map.
    representation_floor = jnp.asarray(
        config.restoration.representation_floor, dtype=dtype
    )
    interior_floor = jnp.maximum(
        representation_floor,
        jnp.asarray(config.restoration.interior_push_fraction, dtype=dtype)
        * variable_scales,
    )
    floored_x = jnp.maximum(raw_x, interior_floor)
    injection = floored_x - raw_x
    scaled_budget_injection = row_scales[:-1] * (
        ag @ injection[:ng] + ac @ injection[ng : ng + nc]
    )
    scaled_total_injection = row_scales[-1] * (
        jnp.sum(injection[:ng]) - injection[-1]
    )
    injection_tolerance = jnp.asarray(
        config.restoration.representation_floor_injection_tolerance,
        dtype=dtype,
    )
    floor_audit_ok = (
        jnp.all(jnp.isfinite(floored_x))
        & (
            jnp.max(jnp.abs(scaled_budget_injection), initial=0.0)
            <= injection_tolerance
        )
        & (jnp.abs(scaled_total_injection) <= injection_tolerance)
    )
    x = jnp.where(floor_audit_ok, floored_x, raw_x)
    jacobian = restoration_constraint_jacobian(problem, row_scales)
    offset = restoration_constraint_offset(problem, row_scales)
    c = jacobian @ x + offset
    mu = jnp.exp(jnp.asarray(original_state.epsilon, dtype=dtype))
    slack_center = jnp.sqrt(mu)
    positive = jnp.maximum(c, 0.0) + slack_center
    negative = jnp.maximum(-c, 0.0) + slack_center
    equality_dual = jnp.zeros_like(c)
    return RestorationState(
        x=x,
        positive_slack=positive,
        negative_slack=negative,
        equality_dual=equality_dual,
        lower_bound_dual_x=mu / x,
        lower_bound_dual_positive=mu / positive,
        lower_bound_dual_negative=mu / negative,
        restoration_mu=mu,
        entry_x=x,
        entry_original_state=original_state,
        entry_phi=barrier_objective(problem, original_state),
        entry_theta=filter_violation(problem, original_state),
        variable_scales=variable_scales,
        row_scales=row_scales,
        iteration=jnp.asarray(0, dtype=jnp.int32),
        accepted_iteration_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _original_metrics(problem, state, filter_state, config):
    ng = state.entry_original_state.q.shape[0]
    nc = state.entry_original_state.r.shape[0]
    primal = OriginalState(
        q=jnp.log(state.x[:ng]),
        r=jnp.log(state.x[ng : ng + nc]),
        lambda_=state.entry_original_state.lambda_,
        rho=state.entry_original_state.rho,
        qtot=jnp.log(state.x[-1]),
        epsilon=state.entry_original_state.epsilon,
        iteration=state.entry_original_state.iteration,
    )
    phi = barrier_objective(problem, primal)
    theta = filter_violation(problem, primal)
    history = accept_to_history(
        phi.reshape((1,)), theta.reshape((1,)), filter_state
    )[0]
    current = (
        theta <= (1.0 - config.filter.gamma_theta) * state.entry_theta
    ) | (
        phi - state.entry_phi
        <= -config.filter.gamma_phi * state.entry_theta
    )
    original_residual = residual_components(problem, primal)
    scaled_budget = (
        jnp.asarray(problem.budget_row_scale, dtype=state.x.dtype)
        * original_residual.budget
    )
    scaled_total = (
        jnp.asarray(problem.total_density_row_scale, dtype=state.x.dtype)
        * original_residual.total_density[0]
    )
    finite = (
        jnp.all(jnp.isfinite(state.x))
        & jnp.all(state.x > 0.0)
        & jnp.isfinite(phi)
        & jnp.isfinite(theta)
    )
    primal_feasible = (
        finite
        & (theta <= config.restoration.required_reduction * state.entry_theta)
        & (jnp.max(jnp.abs(scaled_budget), initial=0.0) <= config.restoration.budget_tolerance)
        & (jnp.abs(scaled_total) <= config.restoration.total_density_tolerance)
    )
    return_ready = (
        (state.accepted_iteration_count > 0)
        & primal_feasible
        & history
        & current
    )
    return phi, theta, return_ready, finite, primal_feasible


def _restoration_kkt_norm(problem, state, config):
    residual = restoration_residual_vector(
        restoration_residuals(problem, state, config.restoration)
    )
    return jnp.max(jnp.abs(residual), initial=0.0)


def restoration_advance(
    problem: FixedSupportProblem,
    state: RestorationState,
    filter_state: FilterState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> RestorationSolveResult:
    """Advance one restoration super-iteration to a typed state."""

    _validate_config(config)
    phi, theta, return_ready, finite, primal_feasible = _original_metrics(
        problem, state, filter_state, config
    )
    kkt_converged = (
        _restoration_kkt_norm(problem, state, config)
        <= config.restoration.kkt_tolerance
    )
    converged_failure = jnp.where(
        primal_feasible,
        jnp.asarray(
            TerminalStatus.RESTORATION_FEASIBLE_BUT_UNACCEPTABLE,
            dtype=jnp.int32,
        ),
        jnp.asarray(
            TerminalStatus.RESTORATION_LOCALLY_INFEASIBLE,
            dtype=jnp.int32,
        ),
    )
    current_status = jnp.where(
        ~finite,
        jnp.asarray(TerminalStatus.RESTORATION_NONFINITE, dtype=jnp.int32),
        jnp.where(
            return_ready,
            jnp.asarray(
                TerminalStatus.RESTORATION_RETURN_ACCEPTED,
                dtype=jnp.int32,
            ),
            jnp.where(
                kkt_converged,
                converged_failure,
                jnp.where(
                    state.iteration >= config.limits.max_restoration_iterations,
                    jnp.asarray(
                        TerminalStatus.RESTORATION_MAX_ITER, dtype=jnp.int32
                    ),
                    jnp.asarray(
                        TerminalStatus.NOT_TERMINATED, dtype=jnp.int32
                    ),
                ),
            ),
        ),
    )
    current_result = RestorationSolveResult(
        state=state,
        status=current_status,
        return_accepted=return_ready,
        original_phi=phi,
        original_theta=theta,
    )

    def take_iteration(_operand):
        iteration = restoration_iteration(problem, state, config)
        next_phi, next_theta, next_return, next_finite, next_primal = (
            _original_metrics(problem, iteration.state, filter_state, config)
        )
        next_kkt_converged = (
            _restoration_kkt_norm(problem, iteration.state, config)
            <= config.restoration.kkt_tolerance
        )
        next_converged_failure = jnp.where(
            next_primal,
            jnp.asarray(
                TerminalStatus.RESTORATION_FEASIBLE_BUT_UNACCEPTABLE,
                dtype=jnp.int32,
            ),
            jnp.asarray(
                TerminalStatus.RESTORATION_LOCALLY_INFEASIBLE,
                dtype=jnp.int32,
            ),
        )
        next_status = jnp.where(
            ~next_finite,
            jnp.asarray(TerminalStatus.RESTORATION_NONFINITE, dtype=jnp.int32),
            jnp.where(
                next_return,
                jnp.asarray(
                    TerminalStatus.RESTORATION_RETURN_ACCEPTED,
                    dtype=jnp.int32,
                ),
                jnp.where(
                    next_kkt_converged,
                    next_converged_failure,
                    jnp.where(
                        iteration.status
                        != int(TerminalStatus.NOT_TERMINATED),
                        iteration.status,
                        jnp.where(
                            iteration.state.iteration
                            >= config.limits.max_restoration_iterations,
                            jnp.asarray(
                                TerminalStatus.RESTORATION_MAX_ITER,
                                dtype=jnp.int32,
                            ),
                            iteration.status,
                        ),
                    ),
                ),
            ),
        )
        return RestorationSolveResult(
            state=iteration.state,
            status=next_status,
            return_accepted=next_return,
            original_phi=next_phi,
            original_theta=next_theta,
        )

    return jax.lax.cond(
        current_status == int(TerminalStatus.NOT_TERMINATED),
        take_iteration,
        lambda _operand: current_result,
        operand=None,
    )


def solve_restoration(
    problem: FixedSupportProblem,
    initial_state: RestorationState,
    filter_state: FilterState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> RestorationSolveResult:
    """Run one standalone persistent restoration call to a typed outcome."""

    _validate_config(config)
    (
        initial_phi,
        initial_theta,
        initial_return,
        initial_finite,
        _initial_primal_feasible,
    ) = _original_metrics(problem, initial_state, filter_state, config)
    initial_status = jnp.where(
        ~initial_finite,
        jnp.asarray(TerminalStatus.RESTORATION_NONFINITE, dtype=jnp.int32),
        jnp.where(
            initial_return,
            jnp.asarray(
                TerminalStatus.RESTORATION_RETURN_ACCEPTED,
                dtype=jnp.int32,
            ),
            jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
        ),
    )
    carry = RestorationSolveResult(
        state=initial_state,
        status=initial_status,
        return_accepted=initial_return,
        original_phi=initial_phi,
        original_theta=initial_theta,
    )

    def condition(result):
        return (
            (result.status == int(TerminalStatus.NOT_TERMINATED))
            & (~result.return_accepted)
            & (result.state.iteration < config.limits.max_restoration_iterations)
        )

    def body(result):
        return restoration_advance(problem, result.state, filter_state, config)

    result = jax.lax.while_loop(condition, body, carry)
    maxed = (
        (result.status == int(TerminalStatus.NOT_TERMINATED))
        & (~result.return_accepted)
        & (result.state.iteration >= config.limits.max_restoration_iterations)
    )
    return result._replace(
        status=jnp.where(
            maxed,
            jnp.asarray(TerminalStatus.RESTORATION_MAX_ITER, dtype=jnp.int32),
            result.status,
        )
    )


__all__ = [
    "initialize_restoration",
    "restoration_barrier_directional_derivative",
    "restoration_advance",
    "restoration_barrier_objective",
    "restoration_constraint_jacobian",
    "restoration_constraint_offset",
    "restoration_direction",
    "restoration_elastic_objective",
    "restoration_equalities",
    "restoration_iteration",
    "restoration_kkt_jacobian",
    "restoration_residual_vector",
    "restoration_residuals",
    "restoration_step_limits",
    "solve_restoration",
]
