"""Tests for the model-neutral magma--gas application service."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.magma_gas import (
    MagmaGasModel,
    MagmaGasModelEvaluation,
    MagmaGasOptions,
    MagmaGasProblem,
    solve,
)


_TEMPERATURE_K = 1000.0
_PRESSURE_BAR = 10.0
_FORMULA_MATRIX = np.asarray(
    [
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ]
)
_KNOWN_GAS_MOLE_FRACTIONS = np.asarray([0.4, 0.3, 0.3])
_KNOWN_ELEMENT_ABUNDANCES = _FORMULA_MATRIX @ _KNOWN_GAS_MOLE_FRACTIONS
_KNOWN_ELEMENT_ABUNDANCES /= _KNOWN_ELEMENT_ABUNDANCES[0]
_KNOWN_ROOT = np.log(_KNOWN_ELEMENT_ABUNDANCES[1])
_TARGET_LOG_MOLE_FRACTION = np.log(_KNOWN_GAS_MOLE_FRACTIONS[0])


class _SyntheticInputs(NamedTuple):
    target_log_mole_fraction: jax.Array


class _SyntheticState(NamedTuple):
    constrained_log_mole_fraction: jax.Array


class _SyntheticModel:
    """One-root boundary model for a two-element, three-species gas."""

    def initial_root(self, conditions):
        return jnp.zeros((1,), dtype=conditions.temperature_k.dtype)

    def element_abundances(self, conditions, root_variables):
        del conditions
        return jnp.stack(
            (
                jnp.asarray(1.0, dtype=root_variables.dtype),
                jnp.exp(root_variables[0]),
            )
        )

    def evaluate(self, conditions, root_variables, gas):
        del root_variables
        constrained = gas.log_mole_fractions[0]
        return MagmaGasModelEvaluation(
            residual=jnp.asarray(
                [constrained - conditions.model_inputs.target_log_mole_fraction]
            ),
            state=_SyntheticState(
                constrained_log_mole_fraction=constrained,
            ),
        )


class _WrongResidualShapeModel(_SyntheticModel):
    def evaluate(self, conditions, root_variables, gas):
        evaluation = super().evaluate(conditions, root_variables, gas)
        return MagmaGasModelEvaluation(
            residual=jnp.repeat(evaluation.residual, 2),
            state=evaluation.state,
        )


def _problem(model=None) -> MagmaGasProblem:
    hvector = jnp.asarray(
        -np.log(_PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS)
    )
    setup = ChemicalSetup(
        formula_matrix=jnp.asarray(_FORMULA_MATRIX),
        hvector_func=lambda temperature: hvector,
        elements=("A", "B"),
        species=("A2", "B2", "AB"),
    )
    return MagmaGasProblem(
        setup=setup,
        model=model or _SyntheticModel(),
    )


def _inputs(target=_TARGET_LOG_MOLE_FRACTION) -> _SyntheticInputs:
    return _SyntheticInputs(jnp.asarray(target))


def _options() -> MagmaGasOptions:
    return MagmaGasOptions(root_tolerance=1.0e-10, max_iter=20)


def test_solve_supports_model_neutral_one_root_problem() -> None:
    problem = _problem()

    result = solve(
        problem,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        _inputs(),
        options=_options(),
    )

    assert isinstance(problem.model, MagmaGasModel)
    assert bool(result.diagnostics.converged)
    assert result.element_abundances.shape == (2,)
    assert result.gas.equilibrium.x.shape == (3,)
    np.testing.assert_allclose(
        result.root_variables,
        np.asarray([_KNOWN_ROOT]),
        rtol=0.0,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        result.element_abundances,
        _KNOWN_ELEMENT_ABUNDANCES,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        result.gas.equilibrium.x,
        _KNOWN_GAS_MOLE_FRACTIONS,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        result.model_state.constrained_log_mole_fraction,
        _TARGET_LOG_MOLE_FRACTION,
        rtol=0.0,
        atol=2.0e-5,
    )


def test_solve_supports_jit_and_reverse_mode_gradient() -> None:
    problem = _problem()
    options = _options()

    def solve_target(target):
        return solve(
            problem,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            _inputs(target),
            options=options,
        )

    compiled_solve = jax.jit(solve_target)
    target = jnp.asarray(_TARGET_LOG_MOLE_FRACTION)
    result = compiled_solve(target)
    calculated = jax.jit(
        jax.grad(lambda value: solve_target(value).root_variables[0])
    )(target)
    epsilon = jnp.asarray(1.0e-3, dtype=target.dtype)
    finite_difference = (
        compiled_solve(target + epsilon).root_variables[0]
        - compiled_solve(target - epsilon).root_variables[0]
    ) / (2.0 * epsilon)

    assert bool(result.diagnostics.converged)
    assert jnp.isfinite(calculated)
    assert jnp.abs(calculated) > 0.1
    np.testing.assert_allclose(
        calculated,
        finite_difference,
        rtol=2.0e-3,
        atol=2.0e-4,
    )


def test_solve_rejects_model_residual_with_wrong_root_shape() -> None:
    with pytest.raises(
        ValueError,
        match="model residual must match the root shape",
    ):
        solve(
            _problem(_WrongResidualShapeModel()),
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            _inputs(),
            options=_options(),
        )
