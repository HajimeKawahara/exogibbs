"""Initialization policies for condensate equilibrium."""

from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp

from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
from exogibbs.equilibrium.condensate.types import (
    Array,
    CondensateEquilibriumInit,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumInitRequest,
)
from exogibbs.equilibrium.gas.grid.initialization import (
    _resolve_equilibrium_grid_initialization_composition,
)
from exogibbs.equilibrium.gas.grid.interpolation import (
    _interpolate_equilibrium_grid_field,
    _validate_equilibrium_grid_metadata_compatibility,
)
from exogibbs.equilibrium.gas.grid.types import (
    EquilibriumGrid,
    EquilibriumGridMetadata,
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


@dataclass(frozen=True)
class FixedSupportCondensateEquilibriumGrid:
    """Fixed-support condensate states on a gas-grid coordinate system.

    ``gas_grid`` stores the gas components of fixed-support condensate
    solutions. ``condensate_amounts`` stores linear amounts with shape
    ``(T, P, Z, support)``. The support is identical at every grid point;
    full-catalog phase grids are outside this container's contract.
    """

    gas_grid: EquilibriumGrid
    condensate_amounts: Array
    support_indices: tuple[int, ...]
    condensate_setup_metadata: EquilibriumGridMetadata


def _validate_fixed_support_condensate_grid(
    grid: FixedSupportCondensateEquilibriumGrid,
    setup: CondensateChemicalSetup,
    preset_name: str,
    expected_composition_axis_name: str,
) -> tuple[int, ...]:
    """Validate fixed support, field shape, and condensate setup metadata."""

    if not isinstance(grid.gas_grid, EquilibriumGrid):
        raise TypeError(
            "Fixed-support condensate grid gas_grid must be an "
            "EquilibriumGrid."
        )
    if not isinstance(
        grid.condensate_setup_metadata,
        EquilibriumGridMetadata,
    ):
        raise TypeError(
            "Fixed-support condensate grid condensate_setup_metadata must be "
            "an EquilibriumGridMetadata."
        )
    support_indices = tuple(int(index) for index in grid.support_indices)
    if not support_indices:
        raise ValueError(
            "Fixed-support condensate grid support must not be empty."
        )
    if len(set(support_indices)) != len(support_indices):
        raise ValueError(
            "Fixed-support condensate grid support indices must be unique."
        )
    metadata = grid.condensate_setup_metadata
    _validate_equilibrium_grid_metadata_compatibility(
        metadata,
        setup.condensate_setup,
        preset_name,
        expected_composition_axis_name=expected_composition_axis_name,
        grid_label="Fixed-support condensate grid",
    )
    stored_species = metadata.preset_species
    if stored_species is None:
        raise ValueError(
            "Fixed-support condensate grid metadata must store condensate "
            "species."
        )
    if tuple(setup.condensate_species) != tuple(stored_species):
        raise ValueError(
            "Fixed-support condensate grid condensate species mismatch: "
            "metadata does not match the runtime combined setup."
        )
    if any(
        index < 0 or index >= len(stored_species)
        for index in support_indices
    ):
        raise ValueError(
            "Fixed-support condensate grid contains an out-of-range support "
            "index."
        )
    amounts = jnp.asarray(grid.condensate_amounts)
    gas_grid = grid.gas_grid
    expected_shape = (
        gas_grid.temperature_axis.shape[0],
        gas_grid.pressure_axis.shape[0],
        gas_grid.log10_z_over_z_sun_axis.shape[0],
        len(support_indices),
    )
    if amounts.shape != expected_shape:
        raise ValueError(
            "Fixed-support condensate grid amounts shape mismatch: expected "
            f"{expected_shape}, got {amounts.shape}."
        )
    return support_indices


@dataclass(frozen=True)
class GridCondensateEquilibriumInitializer:
    """Initialize gas and condensate fields from a fixed-support grid.

    Interpolated support amounts are also scattered into the full runtime
    condensate catalog. Non-grid fields from explicit or previous state are
    retained.
    """

    grid: FixedSupportCondensateEquilibriumGrid
    preset_name: str
    expected_composition_axis_name: str = "log10(Z/Zsun)"

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        if not isinstance(
            self.grid,
            FixedSupportCondensateEquilibriumGrid,
        ):
            raise TypeError(
                "GridCondensateEquilibriumInitializer requires a "
                "FixedSupportCondensateEquilibriumGrid."
            )
        support_indices = _validate_fixed_support_condensate_grid(
            self.grid,
            request.setup,
            self.preset_name,
            self.expected_composition_axis_name,
        )
        gas_grid = self.grid.gas_grid
        composition = _resolve_equilibrium_grid_initialization_composition(
            gas_grid,
            setup=request.setup.gas_setup,
            preset_name=self.preset_name,
            element_vector=request.b,
            explicit_log10_z_over_z_sun=(
                request.explicit_log10_z_over_z_sun
            ),
            expected_composition_axis_name=self.expected_composition_axis_name,
            initializer_name=type(self).__name__,
        )
        from exogibbs.equilibrium.gas.grid.service import (
            interpolate_equilibrium_grid,
        )

        gas_state = interpolate_equilibrium_grid(
            gas_grid,
            temperature=request.T,
            pressure=request.P,
            log10_z_over_z_sun=composition,
        )
        support_amounts = _interpolate_equilibrium_grid_field(
            gas_grid,
            self.grid.condensate_amounts,
            request.T,
            request.P,
            composition,
        )
        condensate_amounts = jnp.zeros(
            (len(request.setup.condensate_species),),
            dtype=support_amounts.dtype,
        ).at[jnp.asarray(support_indices)].set(support_amounts)
        base = DefaultCondensateEquilibriumInitializer()(request)
        return replace(
            base,
            gas_ln_n=jnp.asarray(gas_state.ln_n),
            gas_ntot=jnp.asarray(gas_state.ntot),
            condensate_amounts=condensate_amounts,
            support_indices=support_indices,
            support_amounts=jnp.asarray(support_amounts),
        )


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
    "FixedSupportCondensateEquilibriumGrid",
    "GridCondensateEquilibriumInitializer",
    "resolve_condensate_initial_guess",
)
