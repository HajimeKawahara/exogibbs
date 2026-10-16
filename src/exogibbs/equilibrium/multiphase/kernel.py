"""Feasible reduced Newton steps with full-G curvature and Armijo descent."""

import jax
import jax.numpy as jnp


def scaled_basis(q, row_basis):
    """Congruence-scaled equality tangent, retaining arbitrarily dilute species."""
    scale = jnp.sqrt(jnp.where(q > 0, q, 1.))
    orthogonal, _ = jnp.linalg.qr((row_basis * scale).T, mode="complete")
    return scale, orthogonal[:, row_basis.shape[0]:]


def minimize(energy, initial, tangent, row_basis, inequalities, valid_input, options):
    """JAX-only primal; static null-space coordinates enforce the equalities."""
    value_grad = jax.value_and_grad(energy)
    hessian = jax.hessian(energy)
    if tangent.shape[1] == 0:
        return initial, jnp.asarray(0), jnp.asarray(False)

    def diagnostics(q):
        value, gradient = value_grad(q)
        scale, basis = scaled_basis(q, row_basis)
        directions = scale[:, None] * basis
        # Remove the conserved linear gauge before scaling. Otherwise its
        # roundoff can dominate a trace species' physically tiny direction.
        projected_gradient = gradient-row_basis.T @ (row_basis @ gradient)
        reduced_gradient = directions.T @ projected_gradient
        scaled_hessian = scale[:, None] * hessian(q) * scale[None, :]
        reduced_hessian = basis.T @ scaled_hessian @ basis
        eigenvalues, eigenvectors = jnp.linalg.eigh(reduced_hessian)
        return value, reduced_gradient, eigenvalues, eigenvectors, directions, tangent.T @ gradient

    def condition(state):
        q, iteration, failed = state
        value, gradient, eigenvalues, _, _, residual = diagnostics(q)
        unfinished = ((jnp.linalg.norm(residual) > options.stationarity_tolerance)
                      | (eigenvalues[0] < -options.curvature_tolerance))
        return (valid_input & ~failed & (iteration < options.max_iterations)
                & jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))
                & jnp.all(jnp.isfinite(eigenvalues)) & unfinished)

    def step(state):
        q, iteration, _ = state
        value, gradient, eigenvalues, eigenvectors, directions, residual = diagnostics(q)
        # Regularize the full Hessian only for the direction, never the energy
        # or final curvature audit. This ensures descent in nonconvex regions.
        shift = jnp.maximum(0., 1e-8 - eigenvalues[0])
        direction = -eigenvectors @ ((eigenvectors.T @ gradient) / (eigenvalues + shift))
        saddle = ((jnp.linalg.norm(residual) <= options.stationarity_tolerance)
                  & (eigenvalues[0] < -options.curvature_tolerance))
        negative_direction = eigenvectors[:, 0]
        negative_direction *= jnp.where(gradient @ negative_direction > 0, -1., 1.)
        direction = jnp.where(saddle, negative_direction, direction)
        full_direction = directions @ direction
        slack = inequalities @ q
        change = inequalities @ full_direction
        ratios = jnp.where(change < 0, -slack / jnp.where(change < 0, change, -1.), jnp.inf)
        alpha = jnp.minimum(1., .99 * jnp.min(ratios))
        slope = gradient @ direction

        def accepted(alpha):
            trial = q + alpha * full_direction
            trial_energy = energy(trial)
            roundoff = 8 * jnp.finfo(q.dtype).eps * jnp.maximum(1., jnp.abs(value))
            return (jnp.all(inequalities @ trial > 0) & jnp.isfinite(trial_energy)
                    & (trial_energy <= value + 1e-4 * alpha * slope + roundoff))

        def backtrack(state):
            alpha, count = state
            return (~accepted(alpha)) & (count < options.max_backtracks)

        alpha, _ = jax.lax.while_loop(
            backtrack, lambda state: (state[0] * .5, state[1] + 1), (alpha, 0),
        )
        success = accepted(alpha) & (alpha > 0)
        return jnp.where(success, q + alpha * full_direction, q), iteration + 1, ~success

    return jax.lax.while_loop(condition, step, (initial, 0, jnp.asarray(False)))
