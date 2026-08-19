"""Private damped root solver for magma--gas coupling models."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import stop_gradient

from exogibbs.magma_gas.types import MagmaGasOptions


class _RootCarry(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    progressing: jax.Array


class _LineSearchCarry(NamedTuple):
    trial: jax.Array
    alpha: jax.Array
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    accepted: jax.Array


class _OuterRootSolution(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    tolerance: jax.Array
    step_accepted: jax.Array


class InnerRootDiagnostics(NamedTuple):
    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    tolerance: jax.Array


class RootSolution(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    outer_converged: jax.Array
    inner_converged: jax.Array
    converged: jax.Array
    root_tolerance: jax.Array
    inner_iterations: jax.Array
    inner_residual_norm: jax.Array
    inner_tolerance: jax.Array
    step_accepted: jax.Array


ResidualFunction = Callable[[Any, jax.Array], jax.Array]
InnerDiagnosticsFunction = Callable[[Any, jax.Array], InnerRootDiagnostics]


def _residual_norm(residual: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(residual))


def _effective_root_tolerance(
    options: MagmaGasOptions,
    dtype: jnp.dtype,
) -> jax.Array:
    requested = jnp.asarray(options.root_tolerance, dtype=dtype)
    roundoff_floor = jnp.asarray(64.0 * jnp.finfo(dtype).eps, dtype=dtype)
    return jnp.maximum(requested, roundoff_floor)


def _solve_root_core(
    residual_func: ResidualFunction,
    parameters: Any,
    root_variables_init: jax.Array,
    options: MagmaGasOptions,
) -> _OuterRootSolution:
    root0 = jnp.asarray(root_variables_init)
    residual0 = residual_func(parameters, root0)
    norm0 = _residual_norm(residual0)
    initial = _RootCarry(
        root_variables=root0,
        residual=residual0,
        residual_norm=norm0,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        progressing=jnp.asarray(True),
    )
    tolerance = _effective_root_tolerance(options, root0.dtype)

    def root_cond(carry: _RootCarry) -> jax.Array:
        return (
            (carry.residual_norm > tolerance)
            & (carry.iterations < options.max_iter)
            & carry.progressing
            & jnp.isfinite(carry.residual_norm)
        )

    def root_body(carry: _RootCarry) -> _RootCarry:
        jacobian = jax.jacrev(lambda root: residual_func(parameters, root))(
            carry.root_variables
        )
        direction = jnp.linalg.solve(jacobian, -carry.residual)
        direction_norm = jnp.max(jnp.abs(direction))
        direction_scale = jnp.minimum(
            jnp.asarray(1.0, dtype=root0.dtype),
            jnp.asarray(options.max_step, dtype=root0.dtype)
            / jnp.maximum(
                direction_norm,
                jnp.asarray(jnp.finfo(root0.dtype).tiny, dtype=root0.dtype),
            ),
        )
        direction = direction_scale * direction
        line_initial = _LineSearchCarry(
            trial=jnp.asarray(0, dtype=jnp.int32),
            alpha=jnp.asarray(1.0, dtype=root0.dtype),
            root_variables=carry.root_variables,
            residual=carry.residual,
            residual_norm=carry.residual_norm,
            accepted=jnp.asarray(False),
        )

        def line_cond(line: _LineSearchCarry) -> jax.Array:
            return (
                (line.trial < options.line_search_steps) & (~line.accepted)
            )

        def line_body(line: _LineSearchCarry) -> _LineSearchCarry:
            candidate_root = carry.root_variables + line.alpha * direction
            candidate_residual = residual_func(parameters, candidate_root)
            candidate_norm = _residual_norm(candidate_residual)
            acceptable = (
                jnp.all(jnp.isfinite(candidate_root))
                & jnp.all(jnp.isfinite(candidate_residual))
                & jnp.isfinite(candidate_norm)
                & (candidate_norm < carry.residual_norm)
            )
            return _LineSearchCarry(
                trial=line.trial + 1,
                alpha=line.alpha
                * jnp.asarray(options.backtracking_factor, dtype=root0.dtype),
                root_variables=jnp.where(
                    acceptable,
                    candidate_root,
                    line.root_variables,
                ),
                residual=jnp.where(
                    acceptable,
                    candidate_residual,
                    line.residual,
                ),
                residual_norm=jnp.where(
                    acceptable,
                    candidate_norm,
                    line.residual_norm,
                ),
                accepted=line.accepted | acceptable,
            )

        line = jax.lax.while_loop(line_cond, line_body, line_initial)
        return _RootCarry(
            root_variables=line.root_variables,
            residual=line.residual,
            residual_norm=line.residual_norm,
            iterations=carry.iterations + 1,
            progressing=line.accepted,
        )

    solved = jax.lax.while_loop(root_cond, root_body, initial)
    converged = (
        jnp.isfinite(solved.residual_norm)
        & (solved.residual_norm <= tolerance)
    )
    return _OuterRootSolution(
        root_variables=solved.root_variables,
        residual=solved.residual,
        residual_norm=solved.residual_norm,
        iterations=solved.iterations,
        converged=converged,
        tolerance=tolerance,
        step_accepted=(solved.iterations > 0) & solved.progressing,
    )


def _complete_root_solution(
    outer: _OuterRootSolution,
    inner: InnerRootDiagnostics,
) -> RootSolution:
    return RootSolution(
        root_variables=outer.root_variables,
        residual=outer.residual,
        residual_norm=outer.residual_norm,
        iterations=outer.iterations,
        outer_converged=outer.converged,
        inner_converged=inner.converged,
        converged=outer.converged & inner.converged,
        root_tolerance=outer.tolerance,
        inner_iterations=inner.iterations,
        inner_residual_norm=inner.residual_norm,
        inner_tolerance=inner.tolerance,
        step_accepted=outer.step_accepted,
    )


def make_implicit_root_solver(
    residual_func: ResidualFunction,
    inner_diagnostics_func: InnerDiagnosticsFunction,
    options: MagmaGasOptions,
):
    """Return a custom-VJP root whose model and policy remain static."""

    def solve_with_diagnostics(parameters, root_variables_init):
        outer = _solve_root_core(
            residual_func,
            parameters,
            root_variables_init,
            options,
        )
        inner = inner_diagnostics_func(parameters, outer.root_variables)
        return _complete_root_solution(outer, inner)

    @custom_vjp
    def implicit_root(parameters, root_variables_init):
        return solve_with_diagnostics(parameters, root_variables_init)

    def implicit_root_fwd(parameters, root_variables_init):
        solution = solve_with_diagnostics(parameters, root_variables_init)
        residuals = (
            solution.root_variables,
            parameters,
            solution.converged,
        )
        return solution, residuals

    def implicit_root_bwd(residuals, cotangent):
        root_variables, parameters, converged = residuals
        jacobian = jax.jacrev(
            lambda root: residual_func(parameters, root)
        )(root_variables)
        adjoint = jnp.linalg.solve(
            jacobian.T,
            jnp.asarray(cotangent.root_variables),
        )
        _, parameter_pullback = jax.vjp(
            lambda dynamic_parameters: residual_func(
                dynamic_parameters,
                stop_gradient(root_variables),
            ),
            parameters,
        )
        parameter_cotangent = parameter_pullback(-adjoint)[0]

        def require_converged(value):
            value = jnp.asarray(value)
            return jnp.where(
                converged,
                value,
                jnp.full_like(value, jnp.nan),
            )

        parameter_cotangent = jax.tree_util.tree_map(
            require_converged,
            parameter_cotangent,
        )
        return parameter_cotangent, None

    implicit_root.defvjp(implicit_root_fwd, implicit_root_bwd)
    return implicit_root


__all__ = (
    "InnerRootDiagnostics",
    "RootSolution",
    "make_implicit_root_solver",
)
