"""Initialization policies for gas equilibrium."""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp

from exogibbs.equilibrium.gas.types import (
    Array,
    EquilibriumInit,
    EquilibriumInitializer,
    EquilibriumInitRequest,
)


def default_init(b_vec: Array, species_count: int) -> Tuple[Array, Array]:
    """Return the existing uniform gas initialization."""

    dtype = jnp.result_type(b_vec.dtype, jnp.float32)
    ln_nk = jnp.zeros((species_count,), dtype=dtype)
    ln_ntot = jnp.log(jnp.asarray(species_count, dtype=dtype))
    return ln_nk, ln_ntot


def prepare_init(
    init: Optional[EquilibriumInit],
    b_vec: Array,
    species_count: int,
) -> Tuple[Array, Array]:
    """Resolve a complete numerical initial state."""

    if init is not None and init.ln_nk is not None and init.ln_ntot is not None:
        return jnp.asarray(init.ln_nk), jnp.asarray(init.ln_ntot)
    return default_init(b_vec, species_count)


@dataclass(frozen=True)
class DefaultEquilibriumInitializer:
    """Prefer explicit state, then previous state, then uniform state."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        if (
            request.user_init is not None
            and request.user_init.ln_nk is not None
            and request.user_init.ln_ntot is not None
        ):
            return EquilibriumInit(
                ln_nk=jnp.asarray(request.user_init.ln_nk),
                ln_ntot=jnp.asarray(request.user_init.ln_ntot),
            )
        if (
            request.previous_solution is not None
            and request.previous_solution.ln_nk is not None
            and request.previous_solution.ln_ntot is not None
        ):
            return EquilibriumInit(
                ln_nk=jnp.asarray(request.previous_solution.ln_nk),
                ln_ntot=jnp.asarray(request.previous_solution.ln_ntot),
            )
        ln_nk, ln_ntot = default_init(request.b, request.K)
        return EquilibriumInit(ln_nk=ln_nk, ln_ntot=ln_ntot)


@dataclass(frozen=True)
class LearnedEquilibriumInitializer:
    """Placeholder for a future learned one-layer initializer."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        raise NotImplementedError(
            "LearnedEquilibriumInitializer is not implemented yet."
        )


DEFAULT_INITIALIZER = DefaultEquilibriumInitializer()


def resolve_initial_guess(
    initializer: Optional[EquilibriumInitializer],
    request: EquilibriumInitRequest,
) -> EquilibriumInit:
    """Apply the caller initializer or the default initialization policy."""

    active_initializer = initializer or DEFAULT_INITIALIZER
    return active_initializer(request)
