"""Host-side amount-gauge transformations for condensate states."""

from __future__ import annotations

from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate.types import (
    Array,
    CondensateEquilibriumInit,
)


def transform_linear_amount_gauge_on_host(
    values: Array,
    amount_scale: float,
    *,
    to_canonical: bool,
) -> Array:
    """Transform concrete linear amounts without flushing subnormal values.

    :meta private:
    """

    if not math.isfinite(amount_scale) or amount_scale <= 0.0:
        raise ValueError("amount_scale must be finite and positive.")
    host_values = np.asarray(jax.device_get(values), dtype=np.float64)
    with np.errstate(
        divide="ignore", over="ignore", under="ignore", invalid="ignore"
    ):
        # Divide directly: the reciprocal scale may overflow or underflow.
        transformed = (
            np.divide(host_values, amount_scale)
            if to_canonical
            else np.multiply(host_values, amount_scale)
        )
    return jnp.asarray(transformed, dtype=jnp.float64)


def transform_condensate_init_amount_gauge(
    init: CondensateEquilibriumInit | None,
    amount_scale: float,
    *,
    to_canonical: bool,
) -> CondensateEquilibriumInit | None:
    """Transform extensive initial fields and provenance in either direction.

    Log amounts and log barrier parameters shift together. Element potentials,
    rho, support indices, temperature, and pressure are intensive or structural
    fields and retain their values.

    :meta private:
    """

    if not math.isfinite(amount_scale) or amount_scale <= 0.0:
        raise ValueError("amount_scale must be finite and positive.")
    if init is None or amount_scale == 1.0:
        return init
    log_shift = math.log(amount_scale) * (-1.0 if to_canonical else 1.0)

    def linear(values: Array | None) -> Array | None:
        if values is None:
            return None
        return transform_linear_amount_gauge_on_host(
            values, amount_scale, to_canonical=to_canonical
        )

    def logarithmic(values: Array | None) -> Array | None:
        if values is None:
            return None
        return jnp.asarray(values, dtype=jnp.float64) + log_shift

    support_amounts = linear(init.support_amounts)
    origin = init.inventory_bridge_origin
    return replace(
        init,
        gas_ln_n=logarithmic(init.gas_ln_n),
        gas_ntot=linear(init.gas_ntot),
        condensate_amounts=linear(init.condensate_amounts),
        support_amounts=(
            None
            if support_amounts is None
            else tuple(np.asarray(support_amounts).tolist())
        ),
        barrier_epsilon=logarithmic(init.barrier_epsilon),
        inventory_bridge_origin=(
            None
            if origin is None
            else replace(origin, element_inventory=linear(origin.element_inventory))
        ),
    )
