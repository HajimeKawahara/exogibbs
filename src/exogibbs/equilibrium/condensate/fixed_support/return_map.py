"""Accepted restoration return to the original log-coordinate variables."""

from __future__ import annotations

import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.problem import (
    kkt_component_norms,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    RestorationReturnDiagnostics,
    RestorationReturnResult,
    RestorationState,
    TerminalStatus,
)


def _bound_multiplier_return(
    *,
    entry_amount,
    restored_amount,
    entry_multiplier,
    barrier,
    fraction_to_boundary,
    reset_threshold,
):
    """Apply the Ipopt linearized bound-multiplier restoration return."""

    dtype = entry_amount.dtype
    delta = (barrier - restored_amount * entry_multiplier) / entry_amount
    alpha_bound = jnp.min(
        jnp.where(delta < 0.0, -entry_multiplier / delta, jnp.inf),
        initial=jnp.asarray(jnp.inf, dtype=dtype),
    )
    alpha = jnp.minimum(
        1.0,
        jnp.asarray(fraction_to_boundary, dtype=dtype) * alpha_bound,
    )
    candidate = entry_multiplier + alpha * delta
    reset = (~jnp.all(jnp.isfinite(candidate))) | (
        jnp.max(candidate, initial=0.0)
        > jnp.asarray(reset_threshold, dtype=dtype)
    )
    returned = jnp.where(reset, jnp.ones_like(candidate), candidate)
    return returned, alpha, reset


def apply_restoration_return(
    problem: FixedSupportProblem,
    restoration_state: RestorationState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> RestorationReturnResult:
    """Map one accepted physical restoration point back to original variables."""

    if not 0.0 < config.restoration.return_dual_fraction_to_boundary <= 1.0:
        raise ValueError("return_dual_fraction_to_boundary must be in (0, 1].")
    if config.restoration.bound_multiplier_reset_threshold < 0.0:
        raise ValueError("bound_multiplier_reset_threshold must be non-negative.")
    if config.restoration.representation_floor <= 0.0:
        raise ValueError("representation_floor must be positive.")
    if config.restoration.representation_floor_injection_tolerance < 0.0:
        raise ValueError(
            "representation_floor_injection_tolerance must be non-negative."
        )
    entry = restoration_state.entry_original_state
    ng = entry.q.shape[0]
    nc = entry.r.shape[0]
    dtype = restoration_state.x.dtype
    floor = jnp.asarray(config.restoration.representation_floor, dtype=dtype)
    floored_x = jnp.maximum(restoration_state.x, floor)
    injection = floored_x - restoration_state.x
    gas_injection = injection[:ng]
    condensate_injection = injection[ng : ng + nc]
    total_injection = injection[-1]
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    scaled_budget_injection = (
        jnp.asarray(problem.budget_row_scale, dtype=dtype)
        * (ag @ gas_injection + ac @ condensate_injection)
    )
    scaled_budget_injection_max = jnp.max(
        jnp.abs(scaled_budget_injection), initial=0.0
    )
    scaled_total_injection = jnp.abs(
        jnp.asarray(problem.total_density_row_scale, dtype=dtype)
        * (jnp.sum(gas_injection) - total_injection)
    )
    floor_applied = jnp.any(injection > 0.0)
    floor_audit_ok = (
        jnp.all(jnp.isfinite(floored_x))
        & jnp.all(floored_x > 0.0)
        & (
            scaled_budget_injection_max
            <= config.restoration.representation_floor_injection_tolerance
        )
        & (
            scaled_total_injection
            <= config.restoration.representation_floor_injection_tolerance
        )
    )

    q = jnp.log(floored_x[:ng])
    r = jnp.log(floored_x[ng : ng + nc])
    qtot = jnp.log(floored_x[-1])
    entry_m = restoration_state.entry_x[ng : ng + nc]
    entry_eta = jnp.exp(entry.rho)
    restored_eta, alpha_dual, bound_reset = _bound_multiplier_return(
        entry_amount=entry_m,
        restored_amount=floored_x[ng : ng + nc],
        entry_multiplier=entry_eta,
        barrier=restoration_state.restoration_mu,
        fraction_to_boundary=(
            config.restoration.return_dual_fraction_to_boundary
        ),
        reset_threshold=config.restoration.bound_multiplier_reset_threshold,
    )
    rho = jnp.log(restored_eta)
    pre_return_state = OriginalState(
        q=q,
        r=r,
        lambda_=entry.lambda_,
        rho=entry.rho,
        qtot=qtot,
        epsilon=entry.epsilon,
        iteration=entry.iteration,
    )
    returned_state = OriginalState(
        q=q,
        r=r,
        lambda_=jnp.zeros_like(entry.lambda_),
        rho=rho,
        qtot=qtot,
        epsilon=entry.epsilon,
        iteration=entry.iteration,
    )
    diagnostics = RestorationReturnDiagnostics(
        alpha_dual=alpha_dual,
        bound_multiplier_reset=bound_reset,
        equality_multiplier_reset=jnp.asarray(True),
        representation_floor_applied=floor_applied,
        scaled_budget_injection_max=scaled_budget_injection_max,
        scaled_total_density_injection=scaled_total_injection,
        pre_return_norms=kkt_component_norms(problem, pre_return_state),
        post_return_norms=kkt_component_norms(problem, returned_state),
    )
    return RestorationReturnResult(
        original_state=returned_state,
        diagnostics=diagnostics,
        accepted=floor_audit_ok,
        status=jnp.where(
            floor_audit_ok,
            jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
            jnp.asarray(
                TerminalStatus.RETURN_REPRESENTATION_FLOOR_FAILED,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = ["apply_restoration_return"]
