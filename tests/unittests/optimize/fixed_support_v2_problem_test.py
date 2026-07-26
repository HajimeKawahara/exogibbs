import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_v2.problem import (
    amount_space_equality_jacobian,
    barrier_objective,
    barrier_objective_directional_derivative,
    canonical_gas_source,
    filter_violation,
    linearized_residual_components,
    log_primal_coordinates,
    physical_amounts,
    residual_components,
    residual_jacobian,
    residual_vector,
)
from exogibbs.optimize.fixed_support_v2.types import (
    FixedSupportProblem,
    OriginalDirection,
    OriginalState,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def deterministic_fixture():
    ag = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    ac = jnp.asarray([[1.0, 2.0], [1.0, 0.0]])
    target = jnp.asarray([0.42, 0.55])
    hgas = jnp.asarray([0.12, -0.07, 0.21])
    pressure = jnp.asarray(0.3)
    q = jnp.log(jnp.asarray([0.21, 0.18, 0.04]))
    r = jnp.log(jnp.asarray([0.025, 0.015]))
    qtot = jnp.log(0.45)
    problem = FixedSupportProblem(
        gas_formula_matrix=ag,
        condensate_formula_matrix=ac,
        target_inventory=target,
        gamma=canonical_gas_source(hgas, pressure),
        condensate_standard_source=jnp.asarray([0.31, -0.22]),
        support_indices=jnp.asarray([2, 5], dtype=jnp.int32),
        budget_row_scale=1.0 / target,
        total_density_row_scale=1.0 / jnp.exp(qtot),
    )
    state = OriginalState(
        q=q,
        r=r,
        lambda_=jnp.asarray([0.08, -0.11]),
        rho=jnp.log(jnp.asarray([0.004, 0.006])),
        qtot=qtot,
        epsilon=jnp.log(1.0e-7),
        iteration=jnp.asarray(3, dtype=jnp.int32),
    )
    return problem, state, hgas, pressure


def test_canonical_residual_matches_direct_formulas_and_qtot_convention(
    deterministic_fixture,
):
    problem, state, hgas, pressure = deterministic_fixture
    residual = residual_components(problem, state)
    amounts = physical_amounts(state)
    eta = jnp.exp(state.rho)
    assert residual.gas_stationarity == pytest.approx(
        state.q
        + problem.gamma
        - state.qtot
        - problem.gas_formula_matrix.T @ state.lambda_
    )
    assert residual.condensate_stationarity == pytest.approx(
        problem.condensate_standard_source
        - problem.condensate_formula_matrix.T @ state.lambda_
        - eta
    )
    assert residual.budget == pytest.approx(
        problem.gas_formula_matrix @ amounts.gas
        + problem.condensate_formula_matrix @ amounts.condensate
        - problem.target_inventory
    )
    assert residual.complementarity == pytest.approx(
        state.r + state.rho - state.epsilon
    )
    assert residual.total_density == pytest.approx(
        [jnp.sum(amounts.gas) - amounts.total_gas]
    )

    trial = state._replace(qtot=state.qtot + 0.17)
    gas_change = (
        residual_components(problem, trial).gas_stationarity
        - residual.gas_stationarity
    )
    assert gas_change == pytest.approx(jnp.full_like(state.q, -0.17))


def test_objective_and_theta_match_their_canonical_formulas(
    deterministic_fixture,
):
    problem, state, _hgas, _pressure = deterministic_fixture
    amounts = physical_amounts(state)
    residual = residual_components(problem, state)
    expected_phi = (
        jnp.dot(
            amounts.gas,
            problem.gamma + state.q - state.qtot,
        )
        + jnp.dot(
            amounts.condensate,
            problem.condensate_standard_source,
        )
        - jnp.exp(state.epsilon) * jnp.sum(state.r)
    )
    expected_theta = jnp.linalg.norm(
        jnp.concatenate(
            [
                problem.budget_row_scale * residual.budget,
                jnp.ravel(
                    problem.total_density_row_scale
                    * residual.total_density
                ),
            ]
        ),
        ord=1,
    )

    assert barrier_objective(problem, state) == pytest.approx(expected_phi)
    assert filter_violation(problem, state) == pytest.approx(expected_theta)


def test_residual_linearization_and_objective_derivative_match_jax(
    deterministic_fixture,
):
    problem, state, _hgas, _pressure = deterministic_fixture
    direction = OriginalDirection(
        q=jnp.asarray([0.4, -0.3, 0.1]),
        r=jnp.asarray([0.2, -0.13]),
        lambda_=jnp.asarray([-0.1, 0.15]),
        rho=jnp.asarray([0.05, -0.08]),
        qtot=jnp.asarray(-0.15),
    )

    def vector_for_args(q, r, lambda_, rho, qtot):
        return residual_vector(
            residual_components(
                problem,
                state._replace(q=q, r=r, lambda_=lambda_, rho=rho, qtot=qtot),
            )
        )

    args = (state.q, state.r, state.lambda_, state.rho, state.qtot)
    tangent = tuple(direction)
    current, jacobian_direction = jax.jvp(vector_for_args, args, tangent)
    actual = residual_vector(linearized_residual_components(problem, state, direction))
    objective_jvp = jax.jvp(
        lambda q, r, lambda_, rho, qtot: barrier_objective(
            problem,
            state._replace(q=q, r=r, lambda_=lambda_, rho=rho, qtot=qtot),
        ),
        args,
        tangent,
    )[1]

    assert actual == pytest.approx(current + jacobian_direction, abs=1.0e-14)
    dense_direction = jnp.concatenate(
        [
            direction.q,
            direction.r,
            direction.lambda_,
            direction.rho,
            jnp.asarray([direction.qtot]),
        ]
    )
    assert residual_jacobian(problem, state) @ dense_direction == pytest.approx(
        jacobian_direction, abs=1.0e-14
    )
    assert barrier_objective_directional_derivative(
        problem, state, direction
    ) == pytest.approx(objective_jvp, abs=1.0e-14)


def test_coordinate_maps_and_amount_equality_jacobian(deterministic_fixture):
    problem, state, _hgas, _pressure = deterministic_fixture
    amounts = physical_amounts(state)

    q, r, qtot = log_primal_coordinates(amounts)
    assert q == pytest.approx(state.q)
    assert r == pytest.approx(state.r)
    assert qtot == pytest.approx(state.qtot)

    def equalities(x):
        ng = state.q.size
        nc = state.r.size
        n, m, ntot = x[:ng], x[ng : ng + nc], x[-1]
        return jnp.concatenate(
            [
                problem.gas_formula_matrix @ n
                + problem.condensate_formula_matrix @ m
                - problem.target_inventory,
                jnp.asarray([jnp.sum(n) - ntot]),
            ]
        )

    x = jnp.concatenate([amounts.gas, amounts.condensate, amounts.total_gas[None]])
    assert amount_space_equality_jacobian(problem) == pytest.approx(
        jax.jacfwd(equalities)(x)
    )
