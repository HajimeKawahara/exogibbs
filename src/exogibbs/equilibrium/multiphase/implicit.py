"""Local KKT sensitivities, independent of iterations and support search."""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
from .solve import _energy, _geometry, _physical_state, _primal
from .types import Result


def solve_differentiable(problem, *, temperature_k, pressure_pa, element_amounts_mol,
                         support, init, parameters, options):
    indices, active, row_scale, _, inverse, tangent, _ = _geometry(problem, support)
    energy = _energy(problem, indices)
    flags = ("converged", "local_kkt_valid", "minimum_valid", "support_valid",
             "derivative_valid", "dual_unique", "line_search_failed")

    @jax.custom_jvp
    def equilibrium(t, p, b, initial, params):
        result = _primal(problem, temperature_k=t, pressure_pa=p,
                         element_amounts_mol=b, support=support, init=initial,
                         parameters=params, options=options, check_derivatives=True)
        # JAX 0.4.30 requires matching primal/tangent dtypes in custom_jvp.
        # Restore integer/bool diagnostics outside the custom derivative.
        return jax.tree_util.tree_map(lambda value: value.astype(float), result)

    @equilibrium.defjvp
    def equilibrium_jvp(primals, tangents):
        t, p, b, initial, params = primals
        dt, dp, db, _, dparams = tangents
        result = equilibrium(*primals)
        q = result.amounts_mol[indices] / jnp.sum(b)
        projection, basis = jnp.asarray(inverse), jnp.asarray(tangent)

        def target(b):
            return jnp.concatenate((b / jnp.sum(b), jnp.zeros(len(active)))) / row_scale

        # This particular solution includes d(sum(b)); the reduced equation
        # enforces differentiated stationarity and hence the full active KKT.
        particular = projection @ jax.jvp(target, (b,), (db,))[1]
        hessian = jax.hessian(energy)(q, t, p, params)
        partial_gradient = jax.jvp(lambda t, p, a: jax.grad(energy)(q, t, p, a),
                                   (t, p, params), (dt, dp, dparams))[1]
        dq = particular
        if tangent.shape[1]:
            reduced = basis.T @ hessian @ basis
            rhs = basis.T @ (hessian @ particular + partial_gradient)
            dq = particular - basis @ jnp.linalg.solve(reduced, rhs)

        def physical(q, t, p, b, params):
            return _physical_state(problem, support, energy, q, t, p, b, params)

        _, physical_tangent = jax.jvp(physical, (q, t, p, b, params), (dq, dt, dp, db, dparams))
        # A NaN *coefficient* preserves linearity in tangents, so transposed
        # reverse mode also fails closed. A constant NaN branch would not.
        coefficient = jnp.where(result.derivative_valid > 0, 1., jnp.nan)
        # Keep even identically zero physical Jacobian columns invalid at a
        # rejected root, including coefficients used only by an absent phase.
        signal = sum(jnp.sum(value) for value in jax.tree_util.tree_leaves((dt, dp, db, dparams))
                     if getattr(value, "dtype", None) != jax.dtypes.float0)
        invalid_signal = jnp.where(result.derivative_valid > 0, 0., jnp.nan) * signal
        diagnostics = jax.tree_util.tree_map(jnp.zeros_like, result)
        derivative = replace(diagnostics, **{
            field.name: coefficient * value + invalid_signal
            for field, value in zip(fields(Result)[:8], physical_tangent)
        })
        return result, derivative

    result = equilibrium(jnp.asarray(temperature_k, dtype=float), jnp.asarray(pressure_pa, dtype=float),
                         jnp.asarray(element_amounts_mol, dtype=float), jnp.asarray(init, dtype=float), parameters)
    return replace(result, iterations=result.iterations.astype(int),
                   **{name: getattr(result, name).astype(bool) for name in flags})
