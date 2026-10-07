"""Reduced R-GIE linear solve for fixed-support v2 normal steps."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.problem import physical_amounts
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FixedSupportProblem,
    LinearSolveDiagnostics,
    LinearSolverConfig,
    NormalDirectionResult,
    OriginalDirection,
    OriginalState,
    ResidualComponents,
    TerminalStatus,
)


def _symmetric_ruiz_equilibration(
    matrix,
    rhs,
    iterations: int,
):
    scaled_matrix = jnp.asarray(matrix)
    dtype = scaled_matrix.dtype
    scaled_rhs = jnp.asarray(rhs, dtype=dtype)
    total_scale = jnp.ones((scaled_matrix.shape[0],), dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    for _ in range(iterations):
        row_norm = jnp.max(jnp.abs(scaled_matrix), axis=1)
        step_scale = jnp.where(row_norm > tiny, 1.0 / jnp.sqrt(row_norm), 1.0)
        scaled_matrix = step_scale[:, None] * scaled_matrix * step_scale[None, :]
        scaled_rhs = step_scale * scaled_rhs
        total_scale = total_scale * step_scale
    return scaled_matrix, scaled_rhs, total_scale


def _scaled_solve(scaled_matrix, scaled_rhs, total_scale):
    scaled_solution = jnp.linalg.solve(scaled_matrix, scaled_rhs)
    return total_scale * scaled_solution


def solve_symmetric_reduced_system(
    matrix: Any,
    rhs: Any,
    config: LinearSolverConfig = LinearSolverConfig(),
) -> Any:
    """Solve one symmetric reduced system with the canonical scaling policy.

    Both the finite-barrier R-GIE direction and the zero-barrier implicit
    derivative use symmetric, generally indefinite, reduced systems.  Keeping
    their Ruiz equilibration and iterative-refinement policy in one helper
    prevents the forward and adjoint paths from silently acquiring different
    numerics.
    """

    scaled_matrix, scaled_rhs, total_scale = _symmetric_ruiz_equilibration(
        matrix, rhs, config.ruiz_iterations
    )
    solution = _scaled_solve(scaled_matrix, scaled_rhs, total_scale)
    for _ in range(config.iterative_refinement_steps):
        unscaled_residual = rhs - matrix @ solution
        correction = _scaled_solve(
            scaled_matrix,
            total_scale * unscaled_residual,
            total_scale,
        )
        solution = solution + correction
    return solution


def reduced_direction_from_rhs(
    problem: FixedSupportProblem,
    state: OriginalState,
    *,
    gas_rhs,
    condensate_rhs,
    budget_rhs,
    complementarity_rhs,
    total_density_rhs,
    config: LinearSolverConfig = LinearSolverConfig(),
    failure_status: TerminalStatus = TerminalStatus.NORMAL_LINEAR_SOLVE_FAILED,
) -> NormalDirectionResult:
    """Solve the exact reduced log-KKT equation for arbitrary RHS blocks."""

    amounts = physical_amounts(state)
    n = jnp.asarray(amounts.gas)
    dtype = n.dtype
    m = jnp.asarray(amounts.condensate, dtype=dtype)
    ntot = jnp.asarray(amounts.total_gas, dtype=dtype)
    eta = jnp.exp(jnp.asarray(state.rho, dtype=dtype))
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    rg = jnp.asarray(gas_rhs, dtype=dtype)
    rc = jnp.asarray(condensate_rhs, dtype=dtype)
    rb = jnp.asarray(budget_rhs, dtype=dtype)
    rt_comp = jnp.asarray(complementarity_rhs, dtype=dtype)
    rt_total = jnp.asarray(total_density_rhs, dtype=dtype).reshape(())
    j_vec = m / eta
    gas_inventory = ag @ n
    qhat = ag @ (n[:, None] * ag.T) + ac @ (j_vec[:, None] * ac.T)
    matrix = jnp.block(
        [
            [qhat, gas_inventory[:, None]],
            [
                gas_inventory[None, :],
                (jnp.sum(n) - ntot).reshape((1, 1)),
            ],
        ]
    )
    reduced_budget_rhs = (
        -rb
        + ag @ (n * rg)
        + ac @ (m * rt_comp)
        + ac @ (j_vec * rc)
    )
    reduced_total_rhs = -rt_total + jnp.dot(n, rg)
    rhs = jnp.concatenate(
        [reduced_budget_rhs, reduced_total_rhs.reshape((1,))]
    )
    solution = solve_symmetric_reduced_system(matrix, rhs, config)

    delta_lambda = solution[:-1]
    delta_qtot = solution[-1]
    delta_q = ag.T @ delta_lambda + delta_qtot - rg
    delta_rho = (rc - ac.T @ delta_lambda) / eta
    delta_r = -rt_comp - delta_rho
    direction = OriginalDirection(
        q=delta_q,
        r=delta_r,
        lambda_=delta_lambda,
        rho=delta_rho,
        qtot=delta_qtot,
    )
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    raw_solution_finite = jnp.all(
        jnp.isfinite(
            jnp.concatenate(
                [
                    delta_q,
                    delta_r,
                    delta_lambda,
                    delta_rho,
                    delta_qtot.reshape((1,)),
                ]
            )
        )
    )
    residual_norm = jnp.linalg.norm(matrix @ solution - rhs)
    solution_norm = jnp.linalg.norm(solution)
    denominator = jnp.maximum(
        jnp.linalg.norm(matrix) * solution_norm + jnp.linalg.norm(rhs),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    relative_residual = residual_norm / denominator
    smallest = jnp.min(singular_values)
    largest = jnp.max(singular_values)
    condition = largest / jnp.maximum(
        smallest, jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    )
    solve_ok = (
        raw_solution_finite
        & jnp.isfinite(relative_residual)
        & (relative_residual <= config.relative_residual_tolerance)
    )
    status = jnp.where(
        solve_ok,
        jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
        jnp.asarray(failure_status, dtype=jnp.int32),
    )
    return NormalDirectionResult(
        direction=direction,
        diagnostics=LinearSolveDiagnostics(
            raw_solution_finite=raw_solution_finite,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            solution_norm=solution_norm,
            smallest_singular_value=smallest,
            largest_singular_value=largest,
            condition_estimate=condition,
        ),
        status=status,
    )


def normal_reduced_direction(
    problem: FixedSupportProblem,
    state: OriginalState,
    residual: ResidualComponents,
    config: LinearSolverConfig = LinearSolverConfig(),
) -> NormalDirectionResult:
    """Solve the ordinary Newton equation by exact R-GIE elimination."""

    return reduced_direction_from_rhs(
        problem,
        state,
        gas_rhs=residual.gas_stationarity,
        condensate_rhs=residual.condensate_stationarity,
        budget_rhs=residual.budget,
        complementarity_rhs=residual.complementarity,
        total_density_rhs=residual.total_density,
        config=config,
    )


__all__ = [
    "normal_reduced_direction",
    "reduced_direction_from_rhs",
    "solve_symmetric_reduced_system",
]
