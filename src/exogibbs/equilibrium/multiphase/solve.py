"""Host preparation and a JAX primal on a declared component/constraint face."""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import minimize
from .types import Options, Problem, R_SCALE, Result, Support


def _geometry(problem, support):
    """Factor only static matrices; preserve redundant elemental equations."""
    atoms, domain = problem.atom_matrix, problem.domain_matrix
    indices = np.asarray(support.components, dtype=int)
    active = np.asarray(support.active_constraints, dtype=int)
    if np.any(indices >= atoms.shape[1]) or np.any(active >= len(domain)):
        raise ValueError("Support index is outside the problem.")
    matrix = np.concatenate((atoms[:, indices], domain[active][:, indices]))
    row_scale = np.linalg.norm(matrix, axis=1)
    row_scale = np.where(row_scale > 0, row_scale, 1.)
    scaled = matrix / row_scale[:, None]
    u, singular, vt = np.linalg.svd(scaled, full_matrices=True)
    tolerance = max(scaled.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != np.linalg.matrix_rank(atoms[:, indices]) + len(active):
        raise ValueError("Active domain rows must be independent of element constraints.")
    inverse = (vt[:rank].T / singular[:rank]) @ u[:, :rank].T
    return indices, active, row_scale, scaled, inverse, vt[rank:].T, rank


def solve_fixed_support(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, support: Support, init: Any,
    parameters: Any = (), options: Options = Options(),
) -> Result:
    """Minimize full G on a fixed support, with a JIT-compatible JAX primal.

    Capture problem, support and options as static closure values under jit.
    T, P, b, init and parameters are dynamic. ``init`` is a full mol vector;
    its projection onto the equalities must lie strictly inside this face.
    Excluded components are never evaluated as separate phases and remain zero.
    No phase selection or global certificate is performed. Implicit JVP/VJP
    is a subsequent contract; differentiation raises explicitly in this version.
    Invalid numeric inputs return invalid diagnostics, not a physical solution.
    """
    indices, active, row_scale, matrix, inverse, tangent, rank = _geometry(
        problem, support,
    )
    temperature = jnp.asarray(temperature_k, dtype=float)
    pressure = jnp.asarray(pressure_pa, dtype=float)
    budget = jnp.asarray(element_amounts_mol, dtype=float)
    initial = jnp.asarray(init, dtype=float)
    atoms = jnp.asarray(problem.atom_matrix)
    domain = jnp.asarray(problem.domain_matrix)
    count = atoms.shape[1]
    if temperature.shape or pressure.shape:
        raise ValueError("temperature_k and pressure_pa must be scalars.")
    if budget.shape != (len(problem.elements),) or initial.shape != (count,):
        raise ValueError("Budget and init shapes must match elements and components.")
    total = jnp.sum(budget)
    target = jnp.concatenate((budget / total, jnp.zeros(len(active)))) / row_scale
    q = initial[indices] / total
    q = q + jnp.asarray(inverse) @ (target - jnp.asarray(matrix) @ q)
    sections = problem.component_slices
    present = tuple(any(section.start <= i < section.stop for i in indices)
                    for section in sections)

    def energy(q, t, p, params):
        # Homogeneous callbacks can be evaluated on q mol to avoid overflow at
        # planetary scales. Restore B exactly once in the public extensive G.
        full = jnp.zeros(count, dtype=q.dtype).at[indices].set(q)
        value = jnp.zeros((), dtype=q.dtype)
        for phase, section, exists in zip(problem.phases, sections, present):
            if exists:
                value = value + phase.gibbs_j(t, p, full[section], params)
        if value.shape:
            raise ValueError("Every gibbs_j callback must return an extensive scalar.")
        return value / (R_SCALE * t)

    inactive = np.asarray([i for i in range(len(domain)) if i not in active], int)
    inequalities = np.concatenate((np.eye(len(indices)),
                                    problem.domain_matrix[inactive][:, indices]))
    # Domain rows belonging entirely to excluded phases impose 0 >= 0.
    inequalities = jnp.asarray(inequalities[np.any(inequalities != 0, axis=1)])
    valid_input = (jnp.isfinite(temperature) & (temperature > 0)
                   & jnp.isfinite(pressure) & (pressure > 0)
                   & jnp.all(jnp.isfinite(budget)) & jnp.all(budget >= 0)
                   & (total > 0) & jnp.all(jnp.isfinite(initial))
                   & jnp.all(initial >= 0) & jnp.all(inequalities @ q > 0))

    @jax.custom_jvp
    def primal(q, t, p, params):
        return minimize(lambda x: energy(x, t, p, params), q,
                        jnp.asarray(tangent), inequalities, valid_input, options)

    @primal.defjvp
    def no_implicit_derivative(primals, tangents):
        raise NotImplementedError(
            "Multiphase implicit derivatives are not implemented (R4). "
            "Thermodynamic callbacks remain differentiable."
        )

    q, iterations, line_search_failed = primal(q, temperature, pressure, parameters)
    value, gradient = jax.value_and_grad(energy)(q, temperature, pressure, parameters)
    hessian = jax.hessian(energy)(q, temperature, pressure, parameters)
    dual = (jnp.asarray(inverse).T @ gradient) / row_scale
    raw_matrix = jnp.asarray(matrix) * jnp.asarray(row_scale)[:, None]
    stationarity = jnp.max(jnp.abs(gradient - raw_matrix.T @ dual))
    curvature = (jnp.linalg.eigvalsh(jnp.asarray(tangent).T @ hessian
                                    @ jnp.asarray(tangent))[0]
                 if tangent.shape[1] else jnp.asarray(jnp.inf))
    amounts = jnp.zeros(count).at[indices].set(total * q)
    phase_amounts = jnp.stack([jnp.sum(amounts[s]) for s in sections])
    phase_elements = jnp.stack([atoms[:, s] @ amounts[s] for s in sections])
    residual = jnp.sum(phase_elements, axis=0) - budget
    conservation = jnp.max(jnp.abs(residual) / jnp.where(budget > 0, budget, total))
    zero_conserved = jnp.all(jnp.where(budget == 0, residual == 0, True))
    full_q = jnp.zeros(count).at[indices].set(q)
    domain_scales = []
    for phase, section in zip(problem.phases, sections):
        for row in phase.domain_matrix:
            domain_scales.append(jnp.sum(full_q[section]) * np.linalg.norm(row))
    domain_scales = jnp.asarray(domain_scales)
    slack = (domain @ full_q) / jnp.where(domain_scales > 0, domain_scales, 1.)
    multipliers = jnp.zeros(len(domain)).at[active].set(dual[len(budget):])
    complementarity = jnp.max(jnp.abs(multipliers * (domain @ full_q)), initial=0.)
    kkt = (valid_input & jnp.isfinite(value) & (conservation <= options.conservation_tolerance)
           & zero_conserved & (stationarity <= options.kkt_tolerance)
           & jnp.all(q > 0) & jnp.all(slack >= -options.kkt_tolerance)
           & jnp.all(multipliers >= -options.kkt_tolerance)
           & (complementarity <= options.kkt_tolerance)
           & jnp.all(jnp.abs(slack[active]) <= options.kkt_tolerance))
    relevant_inactive = np.asarray([i for i in inactive
                                   if np.any(np.asarray(problem.domain_matrix)[i, indices])], int)
    support_valid = kkt & jnp.all(slack[relevant_inactive] > options.support_tolerance)
    minimum_valid = kkt & (curvature > options.curvature_tolerance)
    converged = (valid_input & ~line_search_failed & jnp.isfinite(value)
                 & (stationarity <= options.stationarity_tolerance)
                 & (curvature >= -options.curvature_tolerance))
    rt = R_SCALE * temperature
    element_dual = jnp.where(jnp.any(atoms[:, indices] != 0, axis=1),
                             rt * dual[:len(budget)], jnp.nan)
    return Result(
        amounts, phase_amounts, phase_elements, value * rt * total,
        jnp.full(count, jnp.nan).at[indices].set(rt * gradient), element_dual,
        rt * multipliers, residual, conservation, stationarity, slack,
        complementarity, curvature, iterations, converged, kkt, minimum_valid,
        support_valid, jnp.asarray(False), jnp.asarray(rank == len(matrix)),
        line_search_failed, jnp.full(len(sections), jnp.nan),
        jnp.full(len(sections), jnp.nan),
        tuple(jnp.full(len(p.components), jnp.nan) for p in problem.phases), support,
    )


def solve(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, init: Any, parameters: Any = (),
    support: Optional[Support] = None, options: Options = Options(),
) -> Result:
    """Select component/phase supports and audit their insertion stability.

    This host entry is not JIT/vmap/AD compatible. An explicit Support keeps
    the fixed-face behavior; omit it to search supports and active domains.
    Use solve_fixed_support with the returned support for JAX transforms.
    """
    budget = np.asarray(element_amounts_mol, dtype=float)
    if (budget.shape != (len(problem.elements),) or not np.all(np.isfinite(budget))
            or np.any(budget < 0) or budget.sum() <= 0):
        raise ValueError("element_amounts_mol must be finite, nonnegative and nonzero.")
    reachable = ~np.any(problem.atom_matrix[budget == 0] > 0, axis=0)
    if support is None:
        from .selection import select

        return select(problem, temperature_k=temperature_k, pressure_pa=pressure_pa,
                      element_amounts_mol=budget, parameters=parameters, init=init,
                      reachable=reachable, options=options)
    elif any(i >= len(reachable) or not reachable[i] for i in support.components):
        raise ValueError("Support contains components forbidden by a zero element budget.")
    return solve_fixed_support(
        problem, temperature_k=temperature_k, pressure_pa=pressure_pa,
        element_amounts_mol=budget, parameters=parameters, init=init,
        support=support, options=options,
    )
