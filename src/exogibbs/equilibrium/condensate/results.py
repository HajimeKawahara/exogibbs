"""Pure construction of condensate equilibrium result objects."""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    validate_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.types import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
    HEAD_ROUTE_V2,
    AcceptedCondensateState,
    Array,
    CondensateEquilibriumResult,
    CondensateRoute,
)


def _full_condensate_amounts_numpy(
    *,
    support_indices: Sequence[int],
    support_amounts: Array,
    condensate_count: int,
) -> np.ndarray:
    """Expand active-support amounts into a host array."""

    indices = np.asarray(
        tuple(int(index) for index in support_indices),
        dtype=np.int64,
    )
    amounts = np.asarray(support_amounts)
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError(
            "support_indices and support_amounts must have the same length."
        )
    if np.any(indices < 0) or np.any(indices >= condensate_count):
        raise ValueError(
            "support_indices contain an out-of-range condensate index."
        )
    full = np.zeros((condensate_count,), dtype=amounts.dtype)
    if indices.size:
        full[indices] = amounts
    return full


def full_condensate_amounts(
    *,
    support_indices: Sequence[int],
    support_amounts: Array,
    condensate_count: int,
) -> Array:
    """Expand active-support amounts into the full condensate catalog."""

    return jnp.asarray(
        _full_condensate_amounts_numpy(
            support_indices=support_indices,
            support_amounts=support_amounts,
            condensate_count=condensate_count,
        )
    )


def _merge_external_condensate_amounts_numpy(
    *,
    condensate_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None,
) -> np.ndarray:
    """Add externally budgeted condensates in host precision."""

    amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if external_condensate_amounts is None:
        return amounts
    external = np.asarray(external_condensate_amounts, dtype=np.float64)
    if external.ndim != 1 or external.shape[0] != amounts.shape[0]:
        raise ValueError(
            "external_condensate_amounts must match condensate_count."
        )
    return amounts + external


def merge_external_condensate_amounts(
    *,
    condensate_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None,
) -> Array:
    """Add externally budgeted condensates back to the public full vector."""

    return jnp.asarray(
        _merge_external_condensate_amounts_numpy(
            condensate_amounts=condensate_amounts,
            external_condensate_amounts=external_condensate_amounts,
        ),
        dtype=jnp.float64,
    )


def build_condensate_equilibrium_result(
    *,
    setup: CondensateChemicalSetup,
    accepted_state: AcceptedCondensateState,
    support_indices: Sequence[int],
    selected_route: str,
    route: CondensateRoute = HEAD_ROUTE_V2,
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION,
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME,
) -> CondensateEquilibriumResult:
    """Format an already accepted numerical state as a public result."""

    validate_condensate_chemical_setup(setup)
    if route != HEAD_ROUTE_V2:
        raise ValueError("Only the head_v2 result route is supported.")
    normalized_support_indices = tuple(
        int(index) for index in support_indices
    )
    if any(
        index < 0 or index >= len(setup.condensate_species)
        for index in normalized_support_indices
    ):
        raise ValueError(
            "support_indices contain an out-of-range condensate index."
        )
    support_index_array = jnp.asarray(
        normalized_support_indices,
        dtype=jnp.int32,
    )
    support_names = tuple(
        setup.condensate_species[int(index)]
        for index in support_index_array.tolist()
    )
    metadata: dict[str, Any] = dict(accepted_state.diagnostics)
    metadata.setdefault("route", route)
    metadata.setdefault("head_route_version", head_route_version)
    metadata.setdefault("head_route_name", head_route_name)
    metadata.setdefault("selected_route", selected_route)
    metadata["acceptance_tier"] = accepted_state.acceptance_tier
    metadata["warning_messages"] = accepted_state.warning_messages
    metadata.setdefault(
        "fastchem4_trace_public_runtime_constructor_inputs_used",
        False,
    )
    return CondensateEquilibriumResult(
        gas_ln_n=accepted_state.gas_ln_n,
        gas_n=accepted_state.gas_n,
        gas_x=accepted_state.gas_x,
        gas_ntot=accepted_state.gas_ntot,
        condensate_amounts=accepted_state.condensate_amounts,
        condensate_support_indices=support_index_array,
        condensate_support_names=support_names,
        acceptance_tier=accepted_state.acceptance_tier,
        selected_route=selected_route,
        status=accepted_state.status,
        converged=accepted_state.status == "converged",
        diagnostics=metadata,
        head_route_version=head_route_version,
        head_route_name=head_route_name,
    )


__all__ = (
    "build_condensate_equilibrium_result",
    "full_condensate_amounts",
    "merge_external_condensate_amounts",
)
