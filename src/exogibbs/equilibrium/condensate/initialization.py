"""Initialization policies for condensate equilibrium."""

from dataclasses import dataclass, replace
import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

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


def regauge_gas_only_warm_start(
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    element_inventory: Array,
) -> CondensateEquilibriumInit:
    """Preserve finite gas log ratios in an element-inventory amount gauge.

    The uniform shift matches the summed positive non-electron target rows.
    Finite species compatible with the target inventory retain their log
    ratios; absent or incompatible species alone receive a finite numerical
    floor.
    This is a Python-level host utility and returns gas fields only.
    """

    gas_log_amounts = np.asarray(
        jax.device_get(gas_ln_n),
        dtype=np.float64,
    )
    expected_gas_shape = (len(setup.gas_species),)
    if gas_log_amounts.shape != expected_gas_shape:
        raise ValueError(
            "gas_ln_n must have one value per gas species: expected shape "
            f"{expected_gas_shape}, got {gas_log_amounts.shape}."
        )
    if np.any(np.isnan(gas_log_amounts)) or np.any(
        np.isposinf(gas_log_amounts)
    ):
        raise ValueError("gas_ln_n must not contain NaN or positive infinity.")

    inventory = np.asarray(
        jax.device_get(element_inventory),
        dtype=np.float64,
    )
    expected_inventory_shape = (len(setup.elements),)
    if inventory.shape != expected_inventory_shape:
        raise ValueError(
            "element_inventory must have one value per element: expected "
            f"shape {expected_inventory_shape}, got {inventory.shape}."
        )
    if not np.all(np.isfinite(inventory)) or np.any(inventory < 0.0):
        raise ValueError(
            "element_inventory must contain only finite non-negative values."
        )

    gas_formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    physical = np.asarray(
        tuple(
            str(element).strip().lower() not in {"e-", "electron"}
            for element in setup.elements
        ),
        dtype=bool,
    )
    active = physical & (inventory > 0.0)
    target_amount = float(np.sum(inventory[active]))
    if not math.isfinite(target_amount) or target_amount <= 0.0:
        raise ValueError(
            "element_inventory must contain a positive physical-element "
            "amount."
        )

    depleted = physical & (inventory == 0.0)
    incompatible = (
        np.any(gas_formula[depleted, :] != 0.0, axis=0)
        if np.any(depleted)
        else np.zeros(expected_gas_shape, dtype=bool)
    )
    usable = np.isfinite(gas_log_amounts) & ~incompatible
    if not np.any(usable):
        raise ValueError(
            "gas_ln_n has no finite species compatible with element_inventory."
        )

    active_atom_weights = np.sum(gas_formula[active, :], axis=0)
    representing = usable & (active_atom_weights > 0.0)
    if not np.any(representing):
        raise ValueError(
            "gas_ln_n cannot be regauged to element_inventory."
        )
    reference_log_amount = float(np.max(gas_log_amounts[representing]))
    relative_amounts = np.zeros(expected_gas_shape, dtype=np.float64)
    relative_amounts[representing] = np.exp(
        gas_log_amounts[representing] - reference_log_amount
    )
    represented_amount = float(
        np.sum((gas_formula @ relative_amounts)[active])
    )
    if not math.isfinite(represented_amount) or represented_amount <= 0.0:
        raise ValueError(
            "gas_ln_n cannot be regauged to element_inventory."
        )

    log_gauge_shift = (
        math.log(target_amount)
        - math.log(represented_amount)
        - reference_log_amount
    )
    regauged_log_amounts = gas_log_amounts + log_gauge_shift
    if not np.all(np.isfinite(regauged_log_amounts[usable])):
        raise ValueError("The gas warm-start gauge shift is not finite.")
    log_gas_total = float(
        np.logaddexp.reduce(regauged_log_amounts[usable])
    )
    if not math.isfinite(log_gas_total):
        raise ValueError("The regauged gas warm start has no finite amount.")

    relative_log_floor = log_gas_total + math.log(1.0e-300)
    represented_log_floor = float(
        np.min(regauged_log_amounts[usable])
        + math.log(np.finfo(regauged_log_amounts.dtype).eps)
    )
    log_floor = min(
        relative_log_floor,
        represented_log_floor,
    )
    if not math.isfinite(log_floor):
        raise ValueError("The gas warm-start numerical floor is not finite.")
    regauged_log_amounts[~usable] = log_floor
    final_log_gas_total = float(np.logaddexp.reduce(regauged_log_amounts))
    if (
        not math.isfinite(final_log_gas_total)
        or final_log_gas_total > math.log(np.finfo(np.float64).max)
        or final_log_gas_total
        < math.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
    ):
        raise ValueError("The regauged gas warm start has no finite amount.")
    gas_total = math.exp(final_log_gas_total)
    if not math.isfinite(gas_total) or gas_total <= 0.0:
        raise ValueError("The regauged gas warm start has no finite amount.")

    return CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray(regauged_log_amounts, dtype=jnp.float64),
        gas_ntot=jnp.asarray(gas_total, dtype=jnp.float64),
    )


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
    condensate catalog and rescaled to the caller's element-inventory gauge.
    Non-grid fields from explicit or previous state are retained.
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
        gas_ln_n = jnp.asarray(gas_state.ln_n)
        grid_inventory = (
            jnp.asarray(request.setup.formula_matrix) @ jnp.exp(gas_ln_n)
            + jnp.asarray(request.setup.formula_matrix_cond)
            @ condensate_amounts
        )
        physical_rows = jnp.asarray(
            [
                str(element) not in {"e-", "electron"}
                for element in request.setup.elements
            ],
            dtype=bool,
        )
        request_inventory = jnp.asarray(request.b)
        request_amount_scale = jnp.sum(
            jnp.where(
                physical_rows & (request_inventory > 0.0),
                request_inventory,
                jnp.zeros_like(request_inventory),
            )
        )
        grid_amount_scale = jnp.sum(
            jnp.where(
                physical_rows & (grid_inventory > 0.0),
                grid_inventory,
                jnp.zeros_like(grid_inventory),
            )
        )
        amount_ratio = request_amount_scale / grid_amount_scale
        gas_ln_n = gas_ln_n + jnp.log(amount_ratio)
        gas_ntot = jnp.asarray(gas_state.ntot) * amount_ratio
        condensate_amounts = condensate_amounts * amount_ratio
        support_amounts = jnp.asarray(support_amounts) * amount_ratio
        base = DefaultCondensateEquilibriumInitializer()(request)
        return replace(
            base,
            gas_ln_n=gas_ln_n,
            gas_ntot=gas_ntot,
            condensate_amounts=condensate_amounts,
            support_indices=support_indices,
            support_amounts=support_amounts,
            inventory_bridge_origin=None,
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
    "regauge_gas_only_warm_start",
    "resolve_condensate_initial_guess",
)
