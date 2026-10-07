"""Model-neutral coupled magma--gas application service."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import EquilibriumResult
from exogibbs.applications.magma_gas._root import (
    InnerRootDiagnostics,
    make_implicit_root_solver,
)
from exogibbs.applications.magma_gas.types import (
    MagmaGasConditions,
    MagmaGasDiagnostics,
    MagmaGasEquilibriumState,
    MagmaGasInit,
    MagmaGasModelEvaluation,
    MagmaGasOptions,
    MagmaGasProblem,
    MagmaGasResult,
)


def _validate_options(options: MagmaGasOptions) -> None:
    if (
        not math.isfinite(options.root_tolerance)
        or options.root_tolerance <= 0.0
    ):
        raise ValueError("root_tolerance must be finite and positive.")
    if (
        isinstance(options.max_iter, bool)
        or not isinstance(options.max_iter, Integral)
        or options.max_iter < 0
    ):
        raise ValueError("max_iter must be a nonnegative integer.")
    if (
        isinstance(options.line_search_steps, bool)
        or not isinstance(options.line_search_steps, Integral)
        or options.line_search_steps <= 0
    ):
        raise ValueError("line_search_steps must be a positive integer.")
    if (
        not math.isfinite(options.backtracking_factor)
        or not 0.0 < options.backtracking_factor < 1.0
    ):
        raise ValueError("backtracking_factor must be between zero and one.")
    if not math.isfinite(options.max_step) or options.max_step <= 0.0:
        raise ValueError("max_step must be finite and positive.")

    inner = options.equilibrium_options
    if not math.isfinite(inner.epsilon_crit) or inner.epsilon_crit <= 0.0:
        raise ValueError(
            "equilibrium_options.epsilon_crit must be finite and positive."
        )
    if (
        isinstance(inner.max_iter, bool)
        or not isinstance(inner.max_iter, Integral)
        or inner.max_iter < 0
    ):
        raise ValueError(
            "equilibrium_options.max_iter must be a nonnegative integer."
        )


def _as_scalar(value: ArrayLike, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return array


def _conditions(
    problem: MagmaGasProblem,
    temperature_k: ArrayLike,
    pressure_bar: ArrayLike,
    model_inputs,
) -> MagmaGasConditions:
    temperature = _as_scalar(temperature_k, "temperature_k")
    pressure = _as_scalar(pressure_bar, "pressure_bar")
    input_leaves = [
        jnp.asarray(value)
        for value in jax.tree_util.tree_leaves(model_inputs)
    ]
    dtype = jnp.result_type(
        problem.setup.formula_matrix,
        temperature,
        pressure,
        *input_leaves,
        jnp.float32,
    )
    normalized_inputs = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=dtype),
        model_inputs,
    )
    return MagmaGasConditions(
        temperature_k=jnp.asarray(temperature, dtype=dtype),
        pressure_bar=jnp.asarray(pressure, dtype=dtype),
        model_inputs=normalized_inputs,
    )


def _evaluate_lnphi(
    problem: MagmaGasProblem,
    conditions: MagmaGasConditions,
    dtype: jnp.dtype,
) -> jax.Array:
    species_count = int(problem.setup.formula_matrix.shape[1])
    if problem.lnphi_func is None:
        return jnp.zeros((species_count,), dtype=dtype)

    lnphi = jnp.asarray(
        problem.lnphi_func(
            conditions.temperature_k,
            conditions.pressure_bar,
            None,
        ),
        dtype=dtype,
    )
    expected_shape = (species_count,)
    if lnphi.shape != expected_shape:
        raise ValueError(
            "lnphi_func must return one value per gas species: "
            f"expected {expected_shape}, got {lnphi.shape}."
        )
    return lnphi


def _solve_gas(
    problem: MagmaGasProblem,
    options: MagmaGasOptions,
    conditions: MagmaGasConditions,
    element_abundances: jax.Array,
    *,
    return_diagnostics: bool = False,
) -> Tuple[EquilibriumResult, jax.Array, Optional[Mapping[str, jax.Array]]]:
    dtype = jnp.result_type(
        element_abundances,
        conditions.temperature_k,
        conditions.pressure_bar,
        problem.setup.formula_matrix,
        jnp.float32,
    )
    lnphi = _evaluate_lnphi(problem, conditions, dtype)

    def evaluated_lnphi_func(temperature, pressure_bar, mole_fractions):
        del temperature, pressure_bar
        if mole_fractions is not None:
            raise ValueError("magma-gas equilibrium requires pure lnphi values.")
        return lnphi

    solved = equilibrium(
        problem.setup,
        conditions.temperature_k,
        conditions.pressure_bar,
        element_abundances,
        options=options.equilibrium_options,
        return_diagnostics=return_diagnostics,
        lnphi_func=evaluated_lnphi_func,
    )
    if return_diagnostics:
        gas_result, diagnostics = solved
        return gas_result, lnphi, diagnostics
    return solved, lnphi, None


def _gas_state(
    gas_result: EquilibriumResult,
    lnphi: jax.Array,
    pressure_bar: jax.Array,
) -> MagmaGasEquilibriumState:
    log_mole_fractions = gas_result.ln_n - logsumexp(gas_result.ln_n)
    log_partial_pressures_bar = log_mole_fractions + jnp.log(pressure_bar)
    log_fugacities_bar = log_partial_pressures_bar + lnphi
    return MagmaGasEquilibriumState(
        equilibrium=gas_result,
        ln_fugacity_coefficients=lnphi,
        log_mole_fractions=log_mole_fractions,
        log_partial_pressures_bar=log_partial_pressures_bar,
        log_fugacities_bar=log_fugacities_bar,
        partial_pressures_bar=jnp.exp(log_partial_pressures_bar),
        fugacities_bar=jnp.exp(log_fugacities_bar),
    )


def _evaluate_problem(
    problem: MagmaGasProblem,
    options: MagmaGasOptions,
    conditions: MagmaGasConditions,
    root_variables: jax.Array,
):
    element_abundances = jnp.asarray(
        problem.model.element_abundances(conditions, root_variables)
    )
    gas_result, lnphi, _ = _solve_gas(
        problem,
        options,
        conditions,
        element_abundances,
    )
    gas = _gas_state(gas_result, lnphi, conditions.pressure_bar)
    evaluation = problem.model.evaluate(
        conditions,
        root_variables,
        gas,
    )
    if not isinstance(evaluation, MagmaGasModelEvaluation):
        raise TypeError(
            "model.evaluate must return MagmaGasModelEvaluation."
        )
    residual = jnp.asarray(evaluation.residual, dtype=root_variables.dtype)
    if residual.shape != root_variables.shape:
        raise ValueError(
            "model residual must match the root shape: "
            f"expected {root_variables.shape}, got {residual.shape}."
        )
    return element_abundances, gas, MagmaGasModelEvaluation(
        residual=residual,
        state=evaluation.state,
    )


def _inner_root_diagnostics(
    problem: MagmaGasProblem,
    options: MagmaGasOptions,
    conditions: MagmaGasConditions,
    root_variables: jax.Array,
) -> InnerRootDiagnostics:
    audit_conditions = jax.tree_util.tree_map(stop_gradient, conditions)
    audit_root = stop_gradient(root_variables)
    element_abundances = jnp.asarray(
        problem.model.element_abundances(audit_conditions, audit_root)
    )
    _, _, diagnostics = _solve_gas(
        problem,
        options,
        audit_conditions,
        element_abundances,
        return_diagnostics=True,
    )
    if diagnostics is None:
        raise RuntimeError("gas diagnostics were not returned.")
    return InnerRootDiagnostics(
        converged=diagnostics["converged"],
        iterations=diagnostics["n_iter"],
        residual_norm=diagnostics["final_residual"],
        tolerance=diagnostics["epsilon_crit"],
    )


def solve(
    problem: MagmaGasProblem,
    temperature_k: ArrayLike,
    pressure_bar: ArrayLike,
    model_inputs,
    *,
    init: Optional[MagmaGasInit] = None,
    options: Optional[MagmaGasOptions] = None,
) -> MagmaGasResult:
    """Solve a model-defined magma boundary coupled to gas equilibrium.

    The model supplies a square, dimensionless residual in its own root
    coordinates. Pressure inputs and gas-state pressure outputs use bar.
    Dynamic model inputs must be JAX PyTrees of differentiable array values.
    """

    active_options = options or MagmaGasOptions()
    _validate_options(active_options)
    active_conditions = _conditions(
        problem,
        temperature_k,
        pressure_bar,
        model_inputs,
    )

    if init is None or init.root_variables is None:
        root_variables_init = jnp.asarray(
            problem.model.initial_root(active_conditions)
        )
    else:
        root_variables_init = jnp.asarray(init.root_variables)
    root_dtype = jnp.result_type(
        root_variables_init,
        active_conditions.temperature_k,
        active_conditions.pressure_bar,
        problem.setup.formula_matrix,
        jnp.float32,
    )
    root_variables_init = jnp.asarray(root_variables_init, dtype=root_dtype)
    if root_variables_init.ndim != 1 or root_variables_init.shape[0] == 0:
        raise ValueError("root_variables must be a non-empty 1D array.")
    root_variables_init = stop_gradient(root_variables_init)

    def residual_func(dynamic_conditions, root_variables):
        _, _, evaluation = _evaluate_problem(
            problem,
            active_options,
            dynamic_conditions,
            root_variables,
        )
        return evaluation.residual

    def inner_diagnostics_func(dynamic_conditions, root_variables):
        return _inner_root_diagnostics(
            problem,
            active_options,
            dynamic_conditions,
            root_variables,
        )

    implicit_root = make_implicit_root_solver(
        residual_func,
        inner_diagnostics_func,
        active_options,
    )
    root_solution = implicit_root(active_conditions, root_variables_init)
    element_abundances, gas, evaluation = _evaluate_problem(
        problem,
        active_options,
        active_conditions,
        root_solution.root_variables,
    )
    diagnostics = MagmaGasDiagnostics(
        converged=root_solution.converged,
        outer_converged=root_solution.outer_converged,
        inner_converged=root_solution.inner_converged,
        iterations=root_solution.iterations,
        inner_iterations=root_solution.inner_iterations,
        residual=root_solution.residual,
        residual_norm=root_solution.residual_norm,
        root_tolerance=root_solution.root_tolerance,
        inner_residual_norm=root_solution.inner_residual_norm,
        inner_tolerance=root_solution.inner_tolerance,
        step_accepted=root_solution.step_accepted,
    )
    return MagmaGasResult(
        element_abundances=element_abundances,
        root_variables=root_solution.root_variables,
        gas=gas,
        model_state=evaluation.state,
        diagnostics=diagnostics,
    )


__all__ = ("solve",)
