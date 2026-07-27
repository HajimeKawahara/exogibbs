"""Initialization policies for condensate equilibrium."""

from dataclasses import dataclass
from typing import Optional

from exogibbs.equilibrium.condensate.types import (
    CondensateEquilibriumInit,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumInitRequest,
)


@dataclass(frozen=True)
class DefaultCondensateEquilibriumInitializer:
    """Use explicit per-layer init first, then the previous profile solution."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        if request.user_init is not None:
            return request.user_init
        if request.previous_solution is not None:
            return request.previous_solution
        return CondensateEquilibriumInit()


DEFAULT_CONDENSATE_INITIALIZER = DefaultCondensateEquilibriumInitializer()


def resolve_condensate_initial_guess(
    initializer: Optional[CondensateEquilibriumInitializer],
    request: CondensateEquilibriumInitRequest,
) -> CondensateEquilibriumInit:
    """Apply the caller initializer or the default condensate policy."""

    active_initializer = initializer or DEFAULT_CONDENSATE_INITIALIZER
    return active_initializer(request)


__all__ = (
    "DefaultCondensateEquilibriumInitializer",
    "resolve_condensate_initial_guess",
)
