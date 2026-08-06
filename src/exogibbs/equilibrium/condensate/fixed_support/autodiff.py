"""Implicit reverse-mode sensitivities on a fixed condensate support.

The production condensate lifecycle discovers support on the host and then
polishes accepted states at zero barrier.  Support discovery is discrete, so
this module deliberately owns only the smooth local problem obtained after a
strictly positive support has been fixed.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import stop_gradient

from exogibbs.equilibrium.condensate.fixed_support.linear_solver import (
    solve_symmetric_reduced_system,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    DifferentiableFixedSupportResult,
    FixedSupportSolveDiagnostics,
    FixedSupportSourceCotangents,
    LinearSolverConfig,
)
from exogibbs.equilibrium.gas.types import ThermoState


_FRACTION_TO_BOUNDARY = 0.995
_LINE_SEARCH_TRIALS = 20
_BACKTRACKING_FACTOR = 0.5


class _ZeroBarrierState(NamedTuple):
    gas_log_amounts: Any
    condensate_amounts: Any
    element_potential: Any
    total_gas_log_amount: Any


class _ZeroBarrierResiduals(NamedTuple):
    gas_stationarity: Any
    condensate_stationarity: Any
    budget: Any
    total_density: Any


class _NewtonCarry(NamedTuple):
    state: _ZeroBarrierState
    residual_norm: Any
    iteration: Any
    progressing: Any


def _effective_residual_tolerance(residual_crit, dtype):
    requested = jnp.asarray(residual_crit, dtype=dtype)
    roundoff_floor = jnp.asarray(10.0 * jnp.finfo(dtype).eps, dtype=dtype)
    return jnp.maximum(requested, roundoff_floor)


def zero_barrier_residuals(
    target_inventory: Any,
    gas_source: Any,
    condensate_standard_source: Any,
    state: _ZeroBarrierState,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
) -> _ZeroBarrierResiduals:
    """Evaluate the physical fixed-support KKT residual at zero barrier."""

    q = jnp.asarray(state.gas_log_amounts)
    dtype = q.dtype
    m = jnp.asarray(state.condensate_amounts, dtype=dtype)
    element_potential = jnp.asarray(state.element_potential, dtype=dtype)
    qtot = jnp.asarray(state.total_gas_log_amount, dtype=dtype)
    target = jnp.asarray(target_inventory, dtype=dtype)
    gamma = jnp.asarray(gas_source, dtype=dtype)
    hcond = jnp.asarray(condensate_standard_source, dtype=dtype)
    ag = jnp.asarray(gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(condensate_formula_matrix, dtype=dtype)
    gas_amounts = jnp.exp(q)
    total_gas = jnp.exp(qtot)

    return _ZeroBarrierResiduals(
        gas_stationarity=(
            q + gamma - qtot - ag.T @ element_potential
        ),
        condensate_stationarity=hcond - ac.T @ element_potential,
        budget=ag @ gas_amounts + ac @ m - target,
        total_density=jnp.sum(gas_amounts) - total_gas,
    )


def zero_barrier_residual_vector(
    target_inventory: Any,
    gas_source: Any,
    condensate_standard_source: Any,
    gas_log_amounts: Any,
    condensate_amounts: Any,
    element_potential: Any,
    total_gas_log_amount: Any,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
) -> Any:
    """Return the zero-barrier residual in canonical block order."""

    residual = zero_barrier_residuals(
        target_inventory,
        gas_source,
        condensate_standard_source,
        _ZeroBarrierState(
            gas_log_amounts=gas_log_amounts,
            condensate_amounts=condensate_amounts,
            element_potential=element_potential,
            total_gas_log_amount=total_gas_log_amount,
        ),
        gas_formula_matrix,
        condensate_formula_matrix,
    )
    return jnp.concatenate(
        [
            jnp.ravel(residual.gas_stationarity),
            jnp.ravel(residual.condensate_stationarity),
            jnp.ravel(residual.budget),
            jnp.ravel(residual.total_density),
        ]
    )


def _scaled_residual_norm(residual, target_inventory, total_gas_log_amount):
    dtype = jnp.asarray(residual.gas_stationarity).dtype
    target = jnp.asarray(target_inventory, dtype=dtype)
    target_magnitude = jnp.abs(target)
    budget_scale = 1.0 / jnp.where(
        target_magnitude > 0.0,
        target_magnitude,
        jnp.ones_like(target_magnitude),
    )
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    total_scale = 1.0 / jnp.maximum(
        jnp.exp(jnp.asarray(total_gas_log_amount, dtype=dtype)), tiny
    )

    def max_abs(value):
        return jnp.max(jnp.abs(jnp.asarray(value)), initial=0.0)

    return jnp.max(
        jnp.asarray(
            [
                max_abs(residual.gas_stationarity),
                max_abs(residual.condensate_stationarity),
                max_abs(budget_scale * residual.budget),
                jnp.abs(total_scale * residual.total_density),
            ],
            dtype=dtype,
        )
    )


def _zero_barrier_reduced_matrix(
    gas_log_amounts,
    total_gas_log_amount,
    gas_formula_matrix,
    condensate_formula_matrix,
):
    q = jnp.asarray(gas_log_amounts)
    dtype = q.dtype
    qtot = jnp.asarray(total_gas_log_amount, dtype=dtype)
    ag = jnp.asarray(gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(condensate_formula_matrix, dtype=dtype)
    gas_amounts = jnp.exp(q)
    total_gas = jnp.exp(qtot)
    gas_inventory = ag @ gas_amounts
    qgas = ag @ (gas_amounts[:, None] * ag.T)
    condensate_count = ac.shape[1]
    return jnp.block(
        [
            [qgas, ac, gas_inventory[:, None]],
            [
                ac.T,
                jnp.zeros(
                    (condensate_count, condensate_count), dtype=dtype
                ),
                jnp.zeros((condensate_count, 1), dtype=dtype),
            ],
            [
                gas_inventory[None, :],
                jnp.zeros((1, condensate_count), dtype=dtype),
                (jnp.sum(gas_amounts) - total_gas).reshape((1, 1)),
            ],
        ]
    )


def _zero_barrier_newton_direction(
    target_inventory,
    gas_source,
    condensate_standard_source,
    state,
    gas_formula_matrix,
    condensate_formula_matrix,
):
    residual = zero_barrier_residuals(
        target_inventory,
        gas_source,
        condensate_standard_source,
        state,
        gas_formula_matrix,
        condensate_formula_matrix,
    )
    q = jnp.asarray(state.gas_log_amounts)
    dtype = q.dtype
    ag = jnp.asarray(gas_formula_matrix, dtype=dtype)
    gas_amounts = jnp.exp(q)
    matrix = _zero_barrier_reduced_matrix(
        q,
        state.total_gas_log_amount,
        ag,
        condensate_formula_matrix,
    )
    rhs = jnp.concatenate(
        [
            -residual.budget
            + ag @ (gas_amounts * residual.gas_stationarity),
            residual.condensate_stationarity,
            jnp.asarray(
                [
                    -residual.total_density
                    + jnp.dot(gas_amounts, residual.gas_stationarity)
                ],
                dtype=dtype,
            ),
        ]
    )
    solution = solve_symmetric_reduced_system(
        matrix, rhs, LinearSolverConfig()
    )
    element_count = ag.shape[0]
    condensate_count = jnp.asarray(
        condensate_formula_matrix
    ).shape[1]
    delta_element_potential = solution[:element_count]
    delta_condensate = solution[
        element_count : element_count + condensate_count
    ]
    delta_qtot = solution[-1]
    delta_q = (
        ag.T @ delta_element_potential
        + delta_qtot
        - residual.gas_stationarity
    )
    return _ZeroBarrierState(
        gas_log_amounts=delta_q,
        condensate_amounts=delta_condensate,
        element_potential=delta_element_potential,
        total_gas_log_amount=delta_qtot,
    )


def _zero_barrier_solve_core(
    target_inventory,
    gas_source,
    condensate_standard_source,
    gas_log_amounts_init,
    condensate_amounts_init,
    element_potential_init,
    total_gas_log_amount_init,
    gas_formula_matrix,
    condensate_formula_matrix,
    residual_crit,
    max_iter,
):
    q0 = jnp.asarray(gas_log_amounts_init)
    dtype = q0.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    initial_state = _ZeroBarrierState(
        gas_log_amounts=q0,
        condensate_amounts=jnp.maximum(
            jnp.asarray(condensate_amounts_init, dtype=dtype), tiny
        ),
        element_potential=jnp.asarray(element_potential_init, dtype=dtype),
        total_gas_log_amount=jnp.asarray(
            total_gas_log_amount_init, dtype=dtype
        ),
    )
    initial_residual = zero_barrier_residuals(
        target_inventory,
        gas_source,
        condensate_standard_source,
        initial_state,
        gas_formula_matrix,
        condensate_formula_matrix,
    )
    initial_norm = _scaled_residual_norm(
        initial_residual,
        target_inventory,
        initial_state.total_gas_log_amount,
    )
    initial = _NewtonCarry(
        state=initial_state,
        residual_norm=initial_norm,
        iteration=jnp.asarray(0, dtype=jnp.int32),
        progressing=jnp.asarray(True),
    )
    tolerance = _effective_residual_tolerance(residual_crit, dtype)

    def cond_fun(carry):
        return (
            (carry.residual_norm > tolerance)
            & (carry.iteration < max_iter)
            & carry.progressing
            & jnp.isfinite(carry.residual_norm)
        )

    def body_fun(carry):
        direction = _zero_barrier_newton_direction(
            target_inventory,
            gas_source,
            condensate_standard_source,
            carry.state,
            gas_formula_matrix,
            condensate_formula_matrix,
        )
        m = carry.state.condensate_amounts
        dm = direction.condensate_amounts
        boundary_ratios = jnp.where(dm < 0.0, -m / dm, jnp.inf)
        alpha_max = jnp.minimum(
            jnp.asarray(1.0, dtype=dtype),
            _FRACTION_TO_BOUNDARY * jnp.min(boundary_ratios),
        )
        alphas = alpha_max * _BACKTRACKING_FACTOR ** jnp.arange(
            _LINE_SEARCH_TRIALS, dtype=dtype
        )

        def evaluate(alpha):
            candidate = jax.tree_util.tree_map(
                lambda current, delta: current + alpha * delta,
                carry.state,
                direction,
            )
            residual = zero_barrier_residuals(
                target_inventory,
                gas_source,
                condensate_standard_source,
                candidate,
                gas_formula_matrix,
                condensate_formula_matrix,
            )
            norm = _scaled_residual_norm(
                residual,
                target_inventory,
                candidate.total_gas_log_amount,
            )
            values = jnp.concatenate(
                [
                    candidate.gas_log_amounts,
                    candidate.condensate_amounts,
                    candidate.element_potential,
                    candidate.total_gas_log_amount.reshape((1,)),
                ]
            )
            finite_and_positive = (
                jnp.all(jnp.isfinite(values))
                & jnp.isfinite(norm)
                & jnp.all(candidate.condensate_amounts > 0.0)
            )
            return candidate, norm, finite_and_positive

        candidates, norms, finite_and_positive = jax.vmap(evaluate)(alphas)
        acceptable = finite_and_positive & (norms < carry.residual_norm)
        accepted = jnp.any(acceptable)
        selected_index = jnp.argmax(acceptable.astype(jnp.int32))
        selected_state = jax.tree_util.tree_map(
            lambda values, fallback: jnp.where(
                accepted, values[selected_index], fallback
            ),
            candidates,
            carry.state,
        )
        selected_norm = jnp.where(
            accepted, norms[selected_index], carry.residual_norm
        )
        return _NewtonCarry(
            state=selected_state,
            residual_norm=selected_norm,
            iteration=carry.iteration + 1,
            progressing=accepted,
        )

    return jax.lax.while_loop(cond_fun, body_fun, initial)


def fixed_support_source_vjp(
    gas_cotangent: Any,
    condensate_cotangent: Any,
    gas_log_amounts: Any,
    condensate_amounts: Any,
    total_gas_log_amount: Any,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
) -> FixedSupportSourceCotangents:
    """Return source cotangents from the zero-barrier reduced adjoint.

    The caller supplies cotangents for ``(log(n_gas), m_active)``.  The
    returned fields correspond to ``(b, gamma, h_cond_active)``.  Formula
    matrices and support indices are treated as fixed mathematical structure.
    The supplied state must be a converged physical root with strictly
    positive active amounts and a nonsingular reduced KKT matrix.
    """

    q = jnp.asarray(gas_log_amounts)
    dtype = q.dtype
    m = jnp.asarray(condensate_amounts, dtype=dtype)
    gq = jnp.asarray(gas_cotangent, dtype=dtype)
    gm = jnp.asarray(condensate_cotangent, dtype=dtype)
    ag = jnp.asarray(gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(condensate_formula_matrix, dtype=dtype)
    if gq.shape != q.shape:
        raise ValueError("gas_cotangent must match gas_log_amounts.")
    if gm.shape != m.shape or m.shape != (ac.shape[1],):
        raise ValueError(
            "condensate cotangent, amounts, and formula columns must match."
        )
    matrix = _zero_barrier_reduced_matrix(
        q, total_gas_log_amount, ag, ac
    )
    rhs = jnp.concatenate([ag @ gq, gm, jnp.sum(gq).reshape((1,))])
    solution = solve_symmetric_reduced_system(
        matrix, rhs, LinearSolverConfig()
    )
    element_count = ag.shape[0]
    condensate_count = ac.shape[1]
    alpha = solution[:element_count]
    chi = solution[element_count : element_count + condensate_count]
    tau = solution[-1]
    gas_amounts = jnp.exp(q)
    gas_source_cotangent = (
        gas_amounts * (ag.T @ alpha + tau) - gq
    )
    return FixedSupportSourceCotangents(
        target_inventory=alpha,
        gas_source=gas_source_cotangent,
        condensate_standard_source=chi,
    )


def _solve_zero_barrier_sources_impl(
    target_inventory,
    gas_source,
    condensate_standard_source,
    gas_log_amounts_init,
    condensate_amounts_init,
    element_potential_init,
    total_gas_log_amount_init,
    gas_formula_matrix,
    condensate_formula_matrix,
    residual_crit,
    max_iter,
):
    solved = _zero_barrier_solve_core(
        target_inventory,
        gas_source,
        condensate_standard_source,
        gas_log_amounts_init,
        condensate_amounts_init,
        element_potential_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        residual_crit,
        max_iter,
    )
    final = solved.state
    return DifferentiableFixedSupportResult(
        gas_log_amounts=final.gas_log_amounts,
        condensate_amounts=final.condensate_amounts,
    )


@custom_vjp
def _solve_zero_barrier_sources(
    target_inventory,
    gas_source,
    condensate_standard_source,
    gas_log_amounts_init,
    condensate_amounts_init,
    element_potential_init,
    total_gas_log_amount_init,
    gas_formula_matrix,
    condensate_formula_matrix,
    residual_crit,
    max_iter,
):
    return _solve_zero_barrier_sources_impl(
        target_inventory,
        gas_source,
        condensate_standard_source,
        gas_log_amounts_init,
        condensate_amounts_init,
        element_potential_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        residual_crit,
        max_iter,
    )


def _solve_zero_barrier_sources_fwd(
    target_inventory,
    gas_source,
    condensate_standard_source,
    gas_log_amounts_init,
    condensate_amounts_init,
    element_potential_init,
    total_gas_log_amount_init,
    gas_formula_matrix,
    condensate_formula_matrix,
    residual_crit,
    max_iter,
):
    solved = _zero_barrier_solve_core(
        target_inventory,
        gas_source,
        condensate_standard_source,
        gas_log_amounts_init,
        condensate_amounts_init,
        element_potential_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        residual_crit,
        max_iter,
    )
    final = solved.state
    result = DifferentiableFixedSupportResult(
        gas_log_amounts=final.gas_log_amounts,
        condensate_amounts=final.condensate_amounts,
    )
    tolerance = _effective_residual_tolerance(
        residual_crit, solved.residual_norm.dtype
    )
    residuals = (
        final.gas_log_amounts,
        final.condensate_amounts,
        final.total_gas_log_amount,
        jnp.isfinite(solved.residual_norm)
        & (solved.residual_norm <= tolerance),
        gas_formula_matrix,
        condensate_formula_matrix,
    )
    return result, residuals


def _solve_zero_barrier_sources_bwd(
    residuals,
    cotangent,
):
    q, m, qtot, converged, ag, ac = residuals
    source_cotangents = fixed_support_source_vjp(
        cotangent.gas_log_amounts,
        cotangent.condensate_amounts,
        q,
        m,
        qtot,
        ag,
        ac,
    )

    def require_converged(value):
        value = jnp.asarray(value)
        return jnp.where(converged, value, jnp.full_like(value, jnp.nan))

    # Initialization, formula matrices, and support are not differentiated.
    # A failed primal has no certified implicit root, so its source cotangents
    # fail closed instead of reporting derivatives of an arbitrary iterate.
    return (
        require_converged(source_cotangents.target_inventory),
        require_converged(source_cotangents.gas_source),
        require_converged(source_cotangents.condensate_standard_source),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_solve_zero_barrier_sources.defvjp(
    _solve_zero_barrier_sources_fwd,
    _solve_zero_barrier_sources_bwd,
)


def _prepare_fixed_support_inputs(
    state: ThermoState,
    gas_log_amounts_init,
    condensate_amounts_init,
    total_gas_log_amount_init,
    gas_formula_matrix,
    condensate_formula_matrix,
    gas_hvector_func: Callable[[Any], Any],
    condensate_hvector_func: Callable[[Any], Any],
    *,
    element_potential_init=None,
):
    """Validate and normalize one fixed-support zero-barrier solve."""

    q_input = jnp.asarray(gas_log_amounts_init)
    m_input = jnp.asarray(condensate_amounts_init)
    target_input = jnp.asarray(state.element_vector)
    gas_source_input = jnp.asarray(gas_hvector_func(state.temperature))
    condensate_source_input = jnp.asarray(
        condensate_hvector_func(state.temperature)
    )
    dtype = jnp.result_type(
        q_input,
        m_input,
        target_input,
        gas_source_input,
        condensate_source_input,
        state.temperature,
        state.ln_normalized_pressure,
        jnp.float32,
    )
    ag = jnp.asarray(gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(condensate_formula_matrix, dtype=dtype)
    q0 = jnp.asarray(q_input, dtype=dtype)
    m0 = jnp.asarray(m_input, dtype=dtype)
    qtot0 = jnp.asarray(total_gas_log_amount_init, dtype=dtype)
    target = jnp.asarray(target_input, dtype=dtype)
    if ag.ndim != 2 or ac.ndim != 2:
        raise ValueError("gas and condensate formula matrices must be 2D.")
    if ag.shape[0] != ac.shape[0]:
        raise ValueError("gas and condensate element dimensions must match.")
    if ac.shape[1] == 0:
        raise ValueError(
            "fixed-support differentiation requires a nonempty support."
        )
    if q0.shape != (ag.shape[1],):
        raise ValueError("gas_log_amounts_init must have one value per gas species.")
    if m0.shape != (ac.shape[1],):
        raise ValueError(
            "condensate_amounts_init must have one value per support species."
        )
    if qtot0.ndim != 0:
        raise ValueError("total_gas_log_amount_init must be scalar.")
    if target.shape != (ag.shape[0],):
        raise ValueError("state.element_vector must have one value per element.")

    gas_standard_source = jnp.asarray(
        gas_source_input, dtype=dtype
    )
    condensate_standard_source = jnp.asarray(
        condensate_source_input, dtype=dtype
    )
    if gas_standard_source.shape != (ag.shape[1],):
        raise ValueError("gas_hvector_func must return one value per gas species.")
    if condensate_standard_source.shape != (ac.shape[1],):
        raise ValueError(
            "condensate_hvector_func must return one value per support species."
        )
    gas_source = gas_standard_source + jnp.asarray(
        state.ln_normalized_pressure, dtype=dtype
    )
    if element_potential_init is None:
        initial_stationarity_rhs = q0 + gas_source - qtot0
        element_potential0 = jnp.linalg.lstsq(
            ag.T, initial_stationarity_rhs, rcond=None
        )[0]
    else:
        element_potential0 = jnp.asarray(
            element_potential_init, dtype=dtype
        )
        if element_potential0.shape != (ag.shape[0],):
            raise ValueError(
                "element_potential_init must have one value per element."
            )

    return (
        target,
        gas_source,
        condensate_standard_source,
        stop_gradient(q0),
        stop_gradient(m0),
        stop_gradient(element_potential0),
        stop_gradient(qtot0),
        ag,
        ac,
    )


def minimize_gibbs_fixed_support(
    state: ThermoState,
    gas_log_amounts_init: Any,
    condensate_amounts_init: Any,
    total_gas_log_amount_init: Any,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
    gas_hvector_func: Callable[[Any], Any],
    condensate_hvector_func: Callable[[Any], Any],
    *,
    element_potential_init: Any = None,
    residual_crit: float = 1.0e-10,
    max_iter: int = 100,
) -> DifferentiableFixedSupportResult:
    """Solve and differentiate a zero-barrier equilibrium on fixed support.

    ``condensate_formula_matrix`` must contain exactly the caller-selected
    positive support, and ``condensate_amounts_init`` uses the same column
    order.  ``condensate_hvector_func`` must likewise return only that support
    in the same order; a full-catalog setup function should be sliced by the
    caller.  The support, formula matrices, and all initialization values are
    non-differentiable.  Reverse-mode derivatives are provided with respect
    to temperature, normalized log pressure, and target elemental inventory
    via ``state``.  The host-side support lifecycle and rainout propagation
    are not part of this local differentiability contract.

    The convergence threshold is never set below ten machine epsilons for the
    numerical dtype.  Thus the default remains ``1e-10`` in float64 and is
    automatically relaxed to a representable tolerance in float32.  Use
    :func:`minimize_gibbs_fixed_support_with_diagnostics` to certify a primal
    solve before consuming it.
    """

    prepared = _prepare_fixed_support_inputs(
        state,
        gas_log_amounts_init,
        condensate_amounts_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        gas_hvector_func,
        condensate_hvector_func,
        element_potential_init=element_potential_init,
    )
    return _solve_zero_barrier_sources(
        *prepared,
        residual_crit,
        max_iter,
    )


def minimize_gibbs_fixed_support_with_diagnostics(
    state: ThermoState,
    gas_log_amounts_init: Any,
    condensate_amounts_init: Any,
    total_gas_log_amount_init: Any,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
    gas_hvector_func: Callable[[Any], Any],
    condensate_hvector_func: Callable[[Any], Any],
    *,
    element_potential_init: Any = None,
    residual_crit: float = 1.0e-10,
    max_iter: int = 100,
) -> tuple[DifferentiableFixedSupportResult, FixedSupportSolveDiagnostics]:
    """Run the zero-barrier kernel and return non-differentiable diagnostics.

    This audit entry point executes the same primal iteration but intentionally
    bypasses the custom VJP.  Use :func:`minimize_gibbs_fixed_support` for
    reverse-mode derivatives after this report certifies convergence.
    """

    prepared = _prepare_fixed_support_inputs(
        state,
        gas_log_amounts_init,
        condensate_amounts_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        gas_hvector_func,
        condensate_hvector_func,
        element_potential_init=element_potential_init,
    )
    solved = _zero_barrier_solve_core(
        *prepared,
        residual_crit,
        max_iter,
    )
    final = solved.state
    tolerance = _effective_residual_tolerance(
        residual_crit, solved.residual_norm.dtype
    )
    result = DifferentiableFixedSupportResult(
        gas_log_amounts=final.gas_log_amounts,
        condensate_amounts=final.condensate_amounts,
    )
    diagnostics = FixedSupportSolveDiagnostics(
        converged=jnp.isfinite(solved.residual_norm)
        & (solved.residual_norm <= tolerance),
        residual_norm=solved.residual_norm,
        iterations=solved.iteration,
    )
    return result, diagnostics


__all__ = [
    "fixed_support_source_vjp",
    "minimize_gibbs_fixed_support",
    "minimize_gibbs_fixed_support_with_diagnostics",
    "zero_barrier_residual_vector",
]
