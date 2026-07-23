"""Ipopt-style SOC contracts mapped to the fixed-support log-KKT system."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def fixed_support_symmetric_ruiz_equilibration(
    matrix: Any,
    rhs: Any,
    *,
    iterations: int = 4,
) -> tuple[Any, Any, Any]:
    """Equilibrate ``M x=b`` as ``(D M D) z=D b``, with ``x=D z``."""

    scaled_matrix = jnp.asarray(matrix)
    dtype = scaled_matrix.dtype
    scaled_rhs = jnp.asarray(rhs, dtype=dtype)
    total_scale = jnp.ones((scaled_matrix.shape[0],), dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    for _ in range(iterations):
        row_norm = jnp.max(jnp.abs(scaled_matrix), axis=1)
        step_scale = jnp.where(
            row_norm > tiny,
            1.0 / jnp.sqrt(row_norm),
            jnp.asarray(1.0, dtype=dtype),
        )
        scaled_matrix = (
            step_scale[:, None] * scaled_matrix * step_scale[None, :]
        )
        scaled_rhs = step_scale * scaled_rhs
        total_scale = total_scale * step_scale
    return scaled_matrix, scaled_rhs, total_scale


def _solve_equilibrated_system(
    scaled_matrix: Any,
    scaled_rhs: Any,
    total_scale: Any,
) -> tuple[Any, Any]:
    scaled_solution, _residuals, _rank, singular_values = jnp.linalg.lstsq(
        jnp.asarray(scaled_matrix),
        jnp.asarray(scaled_rhs),
        rcond=None,
    )
    return jnp.asarray(total_scale) * scaled_solution, singular_values


def _solve_charge_schur_system(
    matrix: Any,
    rhs: Any,
    *,
    charge_row_index: int,
) -> Any:
    """Eliminate one charge-potential coordinate before the dense solve."""

    matrix_array = jnp.asarray(matrix)
    rhs_array = jnp.asarray(rhs, dtype=matrix_array.dtype)
    size = matrix_array.shape[0]
    keep = jnp.asarray(
        [index for index in range(size) if index != charge_row_index]
    )
    coupling = matrix_array[keep, charge_row_index]
    charge_diagonal = matrix_array[charge_row_index, charge_row_index]
    charge_rhs = rhs_array[charge_row_index]
    schur_matrix = matrix_array[jnp.ix_(keep, keep)] - (
        coupling[:, None] * coupling[None, :] / charge_diagonal
    )
    schur_rhs = rhs_array[keep] - coupling * charge_rhs / charge_diagonal
    scaled_matrix, scaled_rhs, total_scale = fixed_support_symmetric_ruiz_equilibration(
        schur_matrix,
        schur_rhs,
    )
    kept_solution, _ = _solve_equilibrated_system(
        scaled_matrix,
        scaled_rhs,
        total_scale,
    )
    charge_solution = (
        charge_rhs - jnp.dot(coupling, kept_solution)
    ) / charge_diagonal
    return jnp.zeros_like(rhs_array).at[keep].set(kept_solution).at[
        charge_row_index
    ].set(charge_solution)


def fixed_support_reduced_direction_from_rhs_with_diagnostics(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    gas_amounts: Any,
    condensate_amounts: Any,
    condensate_duals: Any,
    total_gas_amount: Any,
    gas_rhs: Any,
    condensate_rhs: Any,
    budget_rhs: Any,
    complementarity_rhs: Any,
    total_density_rhs: Any,
    charge_solve_policy: str = "coupled",
    charge_row_index: int = -1,
    reduced_mode_policy: str = "full",
    diagnostic_mode_vector_policy: str = "smallest_right_singular",
) -> tuple[Any, ...]:
    """Solve the reduced log-KKT system and expose raw solve quality."""

    n = jnp.asarray(gas_amounts)
    dtype = n.dtype
    m = jnp.asarray(condensate_amounts, dtype=dtype)
    eta = jnp.asarray(condensate_duals, dtype=dtype)
    ntot = jnp.asarray(total_gas_amount, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    rg = jnp.asarray(gas_rhs, dtype=dtype)
    rc = jnp.asarray(condensate_rhs, dtype=dtype)
    rb = jnp.asarray(budget_rhs, dtype=dtype)
    rt_comp = jnp.asarray(complementarity_rhs, dtype=dtype)
    rt_total = jnp.asarray(total_density_rhs, dtype=dtype).reshape(())
    eta_safe = jnp.maximum(eta, jnp.asarray(1.0e-300, dtype=dtype))
    j_vec = m / eta_safe
    gas_inventory = ag @ n
    qhat = ag @ (n[:, None] * ag.T) + ac @ (j_vec[:, None] * ac.T)
    reduced_matrix = jnp.block(
        [
            [qhat, gas_inventory[:, None]],
            [
                gas_inventory[None, :],
                jnp.asarray([[jnp.sum(n) - ntot]], dtype=dtype),
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
    reduced_rhs = jnp.concatenate(
        [reduced_budget_rhs, jnp.asarray([reduced_total_rhs], dtype=dtype)]
    )
    _unscaled_left_vectors, unscaled_singular_values, unscaled_vh = (
        jnp.linalg.svd(reduced_matrix, full_matrices=False)
    )
    smallest_right_singular_vector = unscaled_vh[-1]
    scaled_matrix, scaled_rhs, total_scale = (
        fixed_support_symmetric_ruiz_equilibration(
            reduced_matrix,
            reduced_rhs,
        )
    )
    scaled_singular_values = jnp.linalg.svd(
        scaled_matrix,
        compute_uv=False,
    )
    if charge_solve_policy == "coupled":
        raw_solution, _ = _solve_equilibrated_system(
            scaled_matrix,
            scaled_rhs,
            total_scale,
        )
    elif charge_solve_policy == "charge_schur":
        if not 0 <= charge_row_index < ag.shape[0]:
            raise ValueError("charge_row_index must identify an element row.")
        raw_solution = _solve_charge_schur_system(
            reduced_matrix,
            reduced_rhs,
            charge_row_index=charge_row_index,
        )
    else:
        raise ValueError(
            "charge_solve_policy must be 'coupled' or 'charge_schur'."
        )
    for _ in range(2):
        unscaled_residual = reduced_rhs - reduced_matrix @ raw_solution
        if charge_solve_policy == "coupled":
            correction, _ = _solve_equilibrated_system(
                scaled_matrix,
                total_scale * unscaled_residual,
                total_scale,
            )
        else:
            correction = _solve_charge_schur_system(
                reduced_matrix,
                unscaled_residual,
                charge_row_index=charge_row_index,
            )
        raw_solution = raw_solution + correction
    if reduced_mode_policy == "remove_smallest_mode":
        raw_solution = raw_solution - jnp.dot(
            raw_solution, smallest_right_singular_vector
        ) * smallest_right_singular_vector
    elif reduced_mode_policy != "full":
        raise ValueError(
            "reduced_mode_policy must be 'full' or 'remove_smallest_mode'."
        )
    raw_solution_finite = jnp.all(jnp.isfinite(raw_solution))
    linear_system_residual_norm = jnp.linalg.norm(
        reduced_matrix @ raw_solution - reduced_rhs
    )
    raw_solution_norm = jnp.linalg.norm(raw_solution)
    smallest_singular_value = jnp.min(unscaled_singular_values)
    largest_singular_value = jnp.max(unscaled_singular_values)
    scaled_smallest_singular_value = jnp.min(scaled_singular_values)
    scaled_largest_singular_value = jnp.max(scaled_singular_values)
    relative_linear_system_residual = linear_system_residual_norm / jnp.maximum(
        jnp.linalg.norm(reduced_matrix) * raw_solution_norm
        + jnp.linalg.norm(reduced_rhs),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    solution = jnp.nan_to_num(raw_solution, nan=0.0, posinf=0.0, neginf=0.0)
    if diagnostic_mode_vector_policy == "smallest_right_singular":
        diagnostic_mode_vector = smallest_right_singular_vector
        diagnostic_mode_metrics = jnp.zeros((23,), dtype=dtype)
    elif diagnostic_mode_vector_policy == "dominant_solution_component":
        mode_coefficients = unscaled_vh @ solution
        dominant_mode_index = jnp.argmax(jnp.abs(mode_coefficients))
        dominant_right_vector = unscaled_vh[dominant_mode_index]
        diagnostic_mode_vector = (
            mode_coefficients[dominant_mode_index]
            * dominant_right_vector
        )
        mode_lambda = dominant_right_vector[:-1]
        mode_qtot = dominant_right_vector[-1]
        gas_mode_log = ag.T @ mode_lambda + mode_qtot
        gas_curvature = jnp.sum(n * gas_mode_log * gas_mode_log) - (
            ntot * mode_qtot * mode_qtot
        )
        cond_stationarity_mode = ac.T @ mode_lambda
        cond_curvature_terms = j_vec * cond_stationarity_mode**2
        cond_rhs_terms = cond_stationarity_mode * (
            m * rt_comp + j_vec * rc
        )
        budget_rhs_terms = -mode_lambda * rb
        total_density_rhs_projection = -mode_qtot * rt_total
        gas_rhs_terms = n * rg * gas_mode_log
        cond_complementarity_rhs_terms = (
            cond_stationarity_mode * m * rt_comp
        )
        cond_stationarity_rhs_terms = cond_stationarity_mode * j_vec * rc
        top_cond_curvature_index = jnp.argmax(jnp.abs(cond_curvature_terms))
        top_cond_rhs_index = jnp.argmax(jnp.abs(cond_rhs_terms))
        top_budget_rhs_index = jnp.argmax(jnp.abs(budget_rhs_terms))
        top_gas_rhs_index = jnp.argmax(jnp.abs(gas_rhs_terms))
        top_cond_complementarity_rhs_index = jnp.argmax(
            jnp.abs(cond_complementarity_rhs_terms)
        )
        top_cond_stationarity_rhs_index = jnp.argmax(
            jnp.abs(cond_stationarity_rhs_terms)
        )
        diagnostic_mode_metrics = jnp.asarray(
            [
                dominant_mode_index,
                unscaled_singular_values[dominant_mode_index],
                mode_coefficients[dominant_mode_index],
                gas_curvature,
                jnp.sum(cond_curvature_terms),
                top_cond_curvature_index,
                cond_curvature_terms[top_cond_curvature_index],
                top_cond_rhs_index,
                cond_rhs_terms[top_cond_rhs_index],
                jnp.sum(cond_rhs_terms),
                jnp.sum(budget_rhs_terms),
                total_density_rhs_projection,
                jnp.sum(gas_rhs_terms),
                top_gas_rhs_index,
                gas_rhs_terms[top_gas_rhs_index],
                jnp.sum(cond_complementarity_rhs_terms),
                jnp.sum(cond_stationarity_rhs_terms),
                top_budget_rhs_index,
                budget_rhs_terms[top_budget_rhs_index],
                top_cond_complementarity_rhs_index,
                cond_complementarity_rhs_terms[
                    top_cond_complementarity_rhs_index
                ],
                top_cond_stationarity_rhs_index,
                cond_stationarity_rhs_terms[top_cond_stationarity_rhs_index],
            ],
            dtype=dtype,
        )
    else:
        raise ValueError(
            "diagnostic_mode_vector_policy must be "
            "'smallest_right_singular' or 'dominant_solution_component'."
        )
    delta_lambda = solution[:-1]
    delta_qtot = solution[-1]
    delta_q = ag.T @ delta_lambda + delta_qtot - rg
    delta_rho = (rc - ac.T @ delta_lambda) / eta_safe
    delta_r = -rt_comp - delta_rho
    return (
        delta_q,
        delta_r,
        delta_lambda,
        delta_rho,
        delta_qtot,
        raw_solution_finite,
        linear_system_residual_norm,
        raw_solution_norm,
        smallest_singular_value,
        largest_singular_value,
        scaled_smallest_singular_value,
        scaled_largest_singular_value,
        relative_linear_system_residual,
        jnp.concatenate([diagnostic_mode_vector, diagnostic_mode_metrics]),
    )


def fixed_support_reduced_direction_from_rhs(
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Solve the exact reduced log-KKT system for generic residual blocks."""

    return fixed_support_reduced_direction_from_rhs_with_diagnostics(**kwargs)[:5]


def fixed_support_soc_constraint_rhs(
    *,
    trial_budget_residual: Any,
    trial_total_density_residual: Any,
    previous_soc_budget_rhs: Any,
    previous_soc_total_density_rhs: Any,
    alpha_soc: Any,
) -> tuple[Any, Any]:
    """Apply Ipopt's repeated ``c_soc <- c_trial + alpha_soc*c_soc`` rule."""

    trial_budget = jnp.asarray(trial_budget_residual)
    dtype = trial_budget.dtype
    alpha = jnp.asarray(alpha_soc, dtype=dtype)
    return (
        trial_budget
        + alpha * jnp.asarray(previous_soc_budget_rhs, dtype=dtype),
        jnp.asarray(trial_total_density_residual, dtype=dtype)
        + alpha * jnp.asarray(previous_soc_total_density_rhs, dtype=dtype),
    )


def fixed_support_soc_trial_from_current(
    *,
    q: Any,
    r: Any,
    element_potential: Any,
    rho: Any,
    qtot: Any,
    delta_q: Any,
    delta_r: Any,
    delta_element_potential: Any,
    delta_rho: Any,
    delta_qtot: Any,
    alpha_test: Any,
    alpha_soc: Any,
    alpha_y: Any,
    alpha_dual: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build an SOC state from the current iterate while retaining alpha_test."""

    q_array = jnp.asarray(q)
    dtype = q_array.dtype
    alpha_soc_array = jnp.asarray(alpha_soc, dtype=dtype)
    alpha_y_array = jnp.asarray(alpha_y, dtype=dtype)
    alpha_dual_array = jnp.asarray(alpha_dual, dtype=dtype)
    return (
        q_array + alpha_soc_array * jnp.asarray(delta_q, dtype=dtype),
        jnp.asarray(r, dtype=dtype)
        + alpha_soc_array * jnp.asarray(delta_r, dtype=dtype),
        jnp.asarray(element_potential, dtype=dtype)
        + alpha_y_array * jnp.asarray(delta_element_potential, dtype=dtype),
        jnp.asarray(rho, dtype=dtype)
        + alpha_dual_array * jnp.asarray(delta_rho, dtype=dtype),
        jnp.asarray(qtot, dtype=dtype)
        + alpha_soc_array * jnp.asarray(delta_qtot, dtype=dtype),
        jnp.asarray(alpha_test, dtype=dtype),
    )


def fixed_support_linearized_rhs_residual_blocks(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    gas_amounts: Any,
    condensate_amounts: Any,
    condensate_duals: Any,
    total_gas_amount: Any,
    gas_rhs: Any,
    condensate_rhs: Any,
    budget_rhs: Any,
    complementarity_rhs: Any,
    total_density_rhs: Any,
    delta_q: Any,
    delta_r: Any,
    delta_element_potential: Any,
    delta_rho: Any,
    delta_qtot: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return the five block residuals ``R + J(current) delta``."""

    n = jnp.asarray(gas_amounts)
    dtype = n.dtype
    m = jnp.asarray(condensate_amounts, dtype=dtype)
    eta = jnp.asarray(condensate_duals, dtype=dtype)
    ntot = jnp.asarray(total_gas_amount, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    dq = jnp.asarray(delta_q, dtype=dtype)
    dr = jnp.asarray(delta_r, dtype=dtype)
    dlam = jnp.asarray(delta_element_potential, dtype=dtype)
    drho = jnp.asarray(delta_rho, dtype=dtype)
    dqtot = jnp.asarray(delta_qtot, dtype=dtype)
    return (
        jnp.asarray(gas_rhs, dtype=dtype) + dq - ag.T @ dlam - dqtot,
        jnp.asarray(condensate_rhs, dtype=dtype) - ac.T @ dlam - eta * drho,
        jnp.asarray(budget_rhs, dtype=dtype)
        + ag @ (n * dq)
        + ac @ (m * dr),
        jnp.asarray(complementarity_rhs, dtype=dtype) + dr + drho,
        jnp.asarray(total_density_rhs, dtype=dtype)
        + jnp.dot(n, dq)
        - ntot * dqtot,
    )


__all__ = [
    "fixed_support_linearized_rhs_residual_blocks",
    "fixed_support_reduced_direction_from_rhs",
    "fixed_support_reduced_direction_from_rhs_with_diagnostics",
    "fixed_support_symmetric_ruiz_equilibration",
    "fixed_support_soc_constraint_rhs",
    "fixed_support_soc_trial_from_current",
]
