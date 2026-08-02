"""Active-condensate support seed and validation decisions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
from exogibbs.equilibrium.condensate.types import CondensateEquilibriumInit


def positive_support_amounts_for_warm_start(
    amounts: Sequence[float],
    *,
    min_seed_amount: float,
) -> tuple[float, ...]:
    """Replace invalid or nonpositive active amounts by the seed floor."""

    floor = float(min_seed_amount)
    return tuple(
        (
            float(value)
            if math.isfinite(float(value)) and float(value) > 0.0
            else floor
        )
        for value in amounts
    )


def support_payload_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    min_seed_amount: float,
) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    """Return finite support payload from an optional profile initializer."""

    if init is None:
        return None
    if init.support_indices is not None:
        support_indices = tuple(int(index) for index in init.support_indices)
        if init.support_amounts is not None:
            support_amounts = positive_support_amounts_for_warm_start(
                init.support_amounts,
                min_seed_amount=min_seed_amount,
            )
        elif init.condensate_amounts is not None:
            amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
            if (
                amounts.ndim != 1
                or amounts.shape[0] != len(setup.condensate_species)
            ):
                return None
            support_amounts = positive_support_amounts_for_warm_start(
                (amounts[index] for index in support_indices),
                min_seed_amount=min_seed_amount,
            )
        else:
            return None
    elif init.condensate_amounts is not None:
        amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
        if (
            amounts.ndim != 1
            or amounts.shape[0] != len(setup.condensate_species)
        ):
            return None
        active = np.flatnonzero(np.isfinite(amounts) & (amounts > 0.0))
        support_indices = tuple(int(index) for index in active.tolist())
        support_amounts = positive_support_amounts_for_warm_start(
            (amounts[index] for index in support_indices),
            min_seed_amount=min_seed_amount,
        )
    else:
        return None
    if len(support_indices) != len(support_amounts):
        return None
    if len(set(support_indices)) != len(support_indices):
        return None
    if any(
        index < 0 or index >= len(setup.condensate_species)
        for index in support_indices
    ):
        return None
    if not all(
        math.isfinite(value) and value > 0.0 for value in support_amounts
    ):
        return None
    return support_indices, support_amounts


def evaluate_profile_support_closure(
    fixed_support_result: Mapping[str, Any],
    *,
    formula_matrix: Any,
    formula_matrix_cond_full: Any,
    condensate_standard_source_full: Any,
    condensate_valid_mask: Any = None,
    budget_relative_floor: float = 1.0e-6,
    support_closure_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Attach full-budget and inactive-condensate closure reports."""

    if support_closure_tolerance < 0.0:
        raise ValueError("support_closure_tolerance must be non-negative.")
    result = dict(fixed_support_result)
    ag = jnp.asarray(formula_matrix)
    ac_full = jnp.asarray(formula_matrix_cond_full, dtype=ag.dtype)
    gas_log_amounts = jnp.asarray(result["gas_log_amounts"], dtype=ag.dtype)
    condensate_amounts = jnp.asarray(
        result["condensate_amounts"],
        dtype=ag.dtype,
    )
    element_potential = jnp.asarray(
        result["element_potential"],
        dtype=ag.dtype,
    )
    target_by_layer = jnp.asarray(
        result["element_inventory_target"],
        dtype=ag.dtype,
    )
    support_mask = jnp.asarray(result["support_mask"], dtype=bool)
    fixed_support_converged = jnp.asarray(
        result["fixed_support_converged"],
        dtype=bool,
    )
    layer_count, condensate_count = condensate_amounts.shape
    hcond_full = jnp.asarray(
        condensate_standard_source_full,
        dtype=ag.dtype,
    )
    if ac_full.shape != (ag.shape[0], condensate_count):
        raise ValueError(
            "formula_matrix_cond_full must have shape "
            f"({ag.shape[0]}, {condensate_count})."
        )
    if hcond_full.shape != (layer_count, condensate_count):
        raise ValueError(
            "condensate_standard_source_full must have shape "
            f"({layer_count}, {condensate_count})."
        )
    if condensate_valid_mask is None:
        valid_mask = jnp.ones((layer_count, condensate_count), dtype=bool)
    else:
        valid_mask = jnp.asarray(condensate_valid_mask, dtype=bool)
        if valid_mask.shape != (layer_count, condensate_count):
            raise ValueError(
                "condensate_valid_mask must have shape "
                f"({layer_count}, {condensate_count})."
            )

    inventory_residual = (
        jax.vmap(lambda gas: ag @ gas)(jnp.exp(gas_log_amounts))
        + jax.vmap(lambda cond: ac_full @ cond)(condensate_amounts)
        - target_by_layer
    )
    inventory_residual_scaled = inventory_residual / jnp.maximum(
        jnp.abs(target_by_layer),
        jnp.asarray(budget_relative_floor, dtype=ag.dtype),
    )
    inactive_driving = hcond_full - jax.vmap(
        lambda multiplier: ac_full.T @ multiplier
    )(element_potential)
    inactive_violation = jnp.where(
        support_mask | ~valid_mask,
        0.0,
        jnp.maximum(-inactive_driving, 0.0),
    )
    support_expansion_mask = (
        inactive_violation > support_closure_tolerance
    )
    support_closed = ~jnp.any(support_expansion_mask, axis=1)
    result.update(
        {
            "inventory_residual": inventory_residual,
            "inventory_residual_scaled": inventory_residual_scaled,
            "inactive_condensate_driving": inactive_driving,
            "condensate_valid_mask": valid_mask,
            "support_expansion_mask": support_expansion_mask,
            "support_closed": support_closed,
            "converged_with_support_closure": (
                fixed_support_converged & support_closed
            ),
        }
    )
    return result


__all__ = (
    "positive_support_amounts_for_warm_start",
    "evaluate_profile_support_closure",
    "support_payload_from_condensate_init",
)
