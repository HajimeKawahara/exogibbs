"""Host preparation and a JAX primal on a declared component/constraint face."""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import minimize, refine_stationarity, scaled_basis
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


def _energy(problem, indices):
    count = problem.atom_matrix.shape[1]
    present = tuple(any(section.start <= i < section.stop for i in indices)
                    for section in problem.component_slices)

    def energy(q, t, p, params):
        # Evaluate homogeneous G on q mol; restore the physical scale once.
        full = jnp.zeros(count, dtype=q.dtype).at[indices].set(q)
        value = jnp.zeros((), dtype=q.dtype)
        for phase, section, exists in zip(problem.phases, problem.component_slices, present):
            if exists:
                value = value + phase.gibbs_j(t, p, full[section], params)
        if value.shape:
            raise ValueError("Every gibbs_j callback must return an extensive scalar.")
        return value / (R_SCALE * t)

    return energy


def _physical_state(problem, support, energy, q, temperature, pressure, budget, parameters):
    """The differentiable physical outputs, separate from numerical audits."""
    indices, active, row_scale, _, inverse, _, _ = _geometry(problem, support)
    atoms = jnp.asarray(problem.atom_matrix)
    sections = problem.component_slices
    total, rt = jnp.sum(budget), R_SCALE * temperature
    value, gradient = jax.value_and_grad(energy)(q, temperature, pressure, parameters)
    dual = (jnp.asarray(inverse).T @ gradient) / row_scale
    amounts = jnp.zeros(atoms.shape[1]).at[indices].set(total * q)
    phase_amounts = jnp.stack([jnp.sum(amounts[s]) for s in sections])
    phase_elements = jnp.stack([atoms[:, s] @ amounts[s] for s in sections])
    element_dual = jnp.where(jnp.any(atoms[:, indices] != 0, axis=1),
                             rt * dual[:len(budget)], jnp.nan)
    multipliers = jnp.zeros(len(problem.domain_matrix)).at[active].set(dual[len(budget):])
    return (amounts, phase_amounts, phase_elements, value * rt * total,
            jnp.full(atoms.shape[1], jnp.nan).at[indices].set(rt * gradient), element_dual,
            rt * multipliers, jnp.sum(phase_elements, axis=0) - budget)


def _phase_derivative_domain(problem, support, temperature, pressure, parameters, physical, options):
    """Check material branches and strict margins against excluded phases."""
    amounts, _, _, _, _, element_dual, _, _ = physical
    represented = np.any(problem.atom_matrix[:, support.components] > 0, axis=1)
    potentials = jnp.where(jnp.isfinite(element_dual), element_dual, 0.)
    valid = jnp.asarray(True)
    for phase, section in zip(problem.phases, problem.component_slices):
        atoms = np.asarray(phase.atom_matrix)
        allowed = tuple(np.flatnonzero(~np.any(atoms[~represented] > 0, axis=0)))
        included = tuple(i-section.start for i in support.components if section.start <= i < section.stop)
        if included:
            # Missing reachable components require a one-sided provider
            # derivative contract, outside the current smooth-face API.
            valid &= set(allowed) == set(included)
            if phase.derivative_domain is not None:
                interior = jnp.asarray(phase.derivative_domain(
                    temperature, pressure, amounts[section], parameters,
                ), dtype=bool)
                if interior.shape:
                    raise ValueError("derivative_domain must return a scalar predicate.")
                valid &= interior
        elif allowed:
            if len(phase.components) == 1:
                margin = (phase.gibbs_j(temperature, pressure, jnp.ones(1), parameters)
                          - potentials @ jnp.asarray(atoms[:, 0]))
            elif phase.insertion_lower_bound_j_mol is not None:
                # This optional bound must be JAX-compatible for transformed
                # fixed-support solves. Host-only proofs remain usable in R3.
                margin = phase.insertion_lower_bound_j_mol(
                    temperature, pressure, potentials, parameters, allowed,
                )
            else:
                margin = jnp.nan
            if jnp.shape(margin):
                raise ValueError("insertion_lower_bound_j_mol must return a scalar.")
            valid &= jnp.isfinite(margin) & (margin / (R_SCALE * temperature) > options.kkt_tolerance)
    return valid


def _primal(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, support: Support, init: Any,
    parameters: Any = (), options: Options = Options(), check_derivatives=False,
) -> Result:
    """Minimize full G on a fixed support, with a JIT-compatible JAX primal.

    Capture problem, support and options as static closure values under jit.
    T, P, b, init and parameters are dynamic. ``init`` is a full mol vector;
    its projection onto the equalities must lie strictly inside this face.
    Excluded components are never evaluated as separate phases and remain zero.
    No phase selection or global certificate is performed.
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
    row_basis = np.linalg.svd(matrix, full_matrices=False)[2][:rank]
    row = jnp.asarray(row_basis)
    # A relative correction preserves positive trace species when a feasible
    # warm start differs from the ledger only by floating-point summation.
    weights = jnp.where(jnp.all(q > 0), q, jnp.ones_like(q))
    weighted = row * weights
    q = q + weighted.T @ jnp.linalg.solve(
        weighted @ row.T, row @ jnp.asarray(inverse) @ (target-jnp.asarray(matrix) @ q))
    sections = problem.component_slices
    energy = _energy(problem, indices)

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

    q, iterations, line_search_failed = minimize(
        lambda x: energy(x, temperature, pressure, parameters), q,
        jnp.asarray(tangent), row, inequalities, valid_input, options,
    )
    if options.max_refinement_iterations and not len(active):
        q, refinement_iterations, refined = refine_stationarity(
            lambda x: energy(x, temperature, pressure, parameters), q,
            row, row @ jnp.asarray(inverse) @ target, inequalities, valid_input, options,
        )
        iterations += refinement_iterations
        line_search_failed &= ~refined
    value, gradient = jax.value_and_grad(energy)(q, temperature, pressure, parameters)
    hessian = jax.hessian(energy)(q, temperature, pressure, parameters)
    dual = (jnp.asarray(inverse).T @ gradient) / row_scale
    raw_matrix = jnp.asarray(matrix) * jnp.asarray(row_scale)[:, None]
    stationarity = jnp.max(jnp.abs(gradient - raw_matrix.T @ dual))
    scale, basis = scaled_basis(q, row)
    curvature = (jnp.linalg.eigvalsh(basis.T @ (scale[:, None] * hessian * scale[None, :]) @ basis)[0]
                 if tangent.shape[1] else jnp.asarray(jnp.inf))
    physical = _physical_state(problem, support, energy, q, temperature, pressure, budget, parameters)
    residual = physical[-1]
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
    derivative_valid = jnp.asarray(False)
    if check_derivatives:
        singular = np.linalg.svd(matrix, compute_uv=False)
        regular = jnp.asarray(singular[rank-1] > options.derivative_rcond * singular[0])
        if tangent.shape[1]:
            eigenvalues = jnp.linalg.eigvalsh(jnp.asarray(tangent).T @ hessian @ jnp.asarray(tangent))
            regular &= eigenvalues[0] > options.derivative_rcond * eigenvalues[-1]
        derivative_valid = (converged & minimum_valid & support_valid & regular
                            & jnp.all(q > options.support_tolerance)
                            & jnp.all(dual[len(budget):] > options.kkt_tolerance)
                            & _phase_derivative_domain(problem, support, temperature, pressure,
                                                       parameters, physical, options))
    return Result(
        *physical, conservation, stationarity, slack,
        complementarity, curvature, iterations, converged, kkt, minimum_valid,
        support_valid, derivative_valid, jnp.asarray(rank == len(matrix)),
        line_search_failed, jnp.full(len(sections), jnp.nan),
        jnp.full(len(sections), jnp.nan),
        tuple(jnp.full(len(p.components), jnp.nan) for p in problem.phases), support,
    )


def solve_fixed_support(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, support: Support, init: Any,
    parameters: Any = (), options: Options = Options(),
) -> Result:
    """Solve a fixed face with local implicit JVP/VJP for T, P, b and parameters.

    Capture problem/support/options as static values under JIT. Initialization
    is nondifferentiable. Check derivative_valid before using derivatives;
    invalid roots, switches or ill-conditioned systems produce NaN tangents
    and cotangents. Diagnostics themselves are nondifferentiable.
    """
    from .implicit import solve_differentiable

    return solve_differentiable(problem, temperature_k=temperature_k, pressure_pa=pressure_pa,
                               element_amounts_mol=element_amounts_mol, support=support, init=init,
                               parameters=parameters, options=options)


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
