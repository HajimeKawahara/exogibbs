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
    active_mask=None,
):
    """Apply the Ipopt linearized bound-multiplier restoration return."""

    dtype = entry_amount.dtype
    if active_mask is None:
        active_mask = jnp.ones_like(entry_multiplier, dtype=jnp.bool_)
    else:
        active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    delta = (barrier - restored_amount * entry_multiplier) / entry_amount
    alpha_bound = jnp.min(
        jnp.where(
            active_mask & (delta < 0.0),
            -entry_multiplier / delta,
            jnp.inf,
        ),
        initial=jnp.asarray(jnp.inf, dtype=dtype),
    )
    alpha = jnp.minimum(
        1.0,
        jnp.asarray(fraction_to_boundary, dtype=dtype) * alpha_bound,
    )
    candidate = entry_multiplier + alpha * delta
    reset = (~jnp.all(jnp.where(active_mask, jnp.isfinite(candidate), True))) | (
        jnp.max(jnp.where(active_mask, candidate, 0.0), initial=0.0)
        > jnp.asarray(reset_threshold, dtype=dtype)
    )
    returned = jnp.where(
        active_mask,
        jnp.where(reset, jnp.ones_like(candidate), candidate),
        jnp.ones_like(candidate),
    )
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
    slot_mask = (
        jnp.ones_like(entry.r, dtype=jnp.bool_)
        if problem.condensate_slot_mask is None
        else jnp.asarray(problem.condensate_slot_mask, dtype=jnp.bool_)
    )
    dummy_mask = ~slot_mask
    dummy_amount = jnp.exp(jnp.asarray(entry.epsilon, dtype=dtype))
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

    returned_x = floored_x.at[ng : ng + nc].set(
        jnp.where(slot_mask, floored_x[ng : ng + nc], dummy_amount)
    )
    q = jnp.log(returned_x[:ng])
    r = jnp.where(
        slot_mask,
        jnp.log(returned_x[ng : ng + nc]),
        jnp.asarray(entry.epsilon, dtype=dtype),
    )
    qtot = jnp.log(floored_x[-1])
    entry_m = jnp.where(
        slot_mask,
        restoration_state.entry_x[ng : ng + nc],
        dummy_amount,
    )
    entry_eta = jnp.where(slot_mask, jnp.exp(entry.rho), 1.0)
    restored_m = jnp.where(
        slot_mask,
        returned_x[ng : ng + nc],
        dummy_amount,
    )
    restored_eta, alpha_dual, bound_reset = _bound_multiplier_return(
        entry_amount=entry_m,
        restored_amount=restored_m,
        entry_multiplier=entry_eta,
        barrier=restoration_state.restoration_mu,
        fraction_to_boundary=(
            config.restoration.return_dual_fraction_to_boundary
        ),
        reset_threshold=config.restoration.bound_multiplier_reset_threshold,
        active_mask=slot_mask,
    )
    rho = jnp.where(slot_mask, jnp.log(restored_eta), 0.0)
    pre_return_state = OriginalState(
        q=q,
        r=r,
        lambda_=entry.lambda_,
        rho=jnp.where(slot_mask, entry.rho, 0.0),
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
    hcond = jnp.asarray(problem.condensate_standard_source, dtype=dtype)
    dummy_contract_ok = (
        jnp.all(jnp.where(dummy_mask[None, :], ac == 0.0, True))
        & jnp.all(jnp.where(dummy_mask, hcond == 1.0, True))
        & jnp.all(
            jnp.where(
                dummy_mask,
                jnp.asarray(restoration_state.restoration_mu, dtype=dtype)
                == dummy_amount,
                True,
            )
        )
    )
    dummy_stationarity = hcond - ac.T @ returned_state.lambda_ - jnp.exp(rho)
    dummy_complementarity = r + rho - jnp.asarray(entry.epsilon, dtype=dtype)
    dummy_anchor_ok = (
        jnp.all(jnp.isfinite(jnp.where(dummy_mask, dummy_stationarity, 0.0)))
        & jnp.all(jnp.isfinite(jnp.where(dummy_mask, dummy_complementarity, 0.0)))
        & jnp.all(jnp.where(dummy_mask, dummy_stationarity == 0.0, True))
        & jnp.all(jnp.where(dummy_mask, dummy_complementarity == 0.0, True))
    )
    return_contract_ok = dummy_contract_ok & dummy_anchor_ok
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
        accepted=floor_audit_ok & return_contract_ok,
        status=jnp.where(
            ~return_contract_ok,
            jnp.asarray(
                TerminalStatus.INTERNAL_CONTRACT_ERROR,
                dtype=jnp.int32,
            ),
            jnp.where(
                floor_audit_ok,
                jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
                jnp.asarray(
                    TerminalStatus.RETURN_REPRESENTATION_FLOOR_FAILED,
                    dtype=jnp.int32,
                ),
            ),
        ),
    )


__all__ = ["apply_restoration_return"]
