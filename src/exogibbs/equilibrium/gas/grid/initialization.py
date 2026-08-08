"""Grid-backed initialization for gas equilibrium."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from exogibbs.equilibrium.gas.types import (
    EquilibriumInit,
    EquilibriumInitRequest,
)

if TYPE_CHECKING:
    from exogibbs.equilibrium.gas.grid.service import EquilibriumGrid
    from exogibbs.equilibrium.gas.grid.types import Array
    from exogibbs.thermo.models import ChemicalSetup


def _resolve_equilibrium_grid_initialization_composition(
    grid: "EquilibriumGrid",
    *,
    setup: "ChemicalSetup",
    preset_name: str,
    element_vector,
    explicit_log10_z_over_z_sun: Optional[float],
    expected_composition_axis_name: str,
    initializer_name: str,
) -> "Array":
    """Validate an initializer grid and resolve its composition query."""

    from exogibbs.equilibrium.gas.grid.service import (
        compute_physical_log10_z_over_z_sun,
        validate_equilibrium_grid_compatibility,
    )

    validate_equilibrium_grid_compatibility(
        grid,
        setup,
        preset_name,
        expected_composition_axis_name=expected_composition_axis_name,
    )
    if explicit_log10_z_over_z_sun is not None:
        log10_z_over_z_sun = explicit_log10_z_over_z_sun
    else:
        try:
            log10_z_over_z_sun = compute_physical_log10_z_over_z_sun(
                setup,
                element_vector,
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"{initializer_name} could not infer physical "
                f"log10(Z/Zsun) from request.b: {exc}"
            ) from exc
    return jnp.asarray(log10_z_over_z_sun)


@dataclass(frozen=True)
class GridEquilibriumInitializer:
    """Initialize a gas solve by interpolating a compatible grid."""

    grid: "EquilibriumGrid"
    preset_name: str
    expected_composition_axis_name: str = "log10(Z/Zsun)"

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        from exogibbs.equilibrium.gas.grid.service import (
            interpolate_equilibrium_grid,
        )

        composition = _resolve_equilibrium_grid_initialization_composition(
            self.grid,
            setup=request.setup,
            preset_name=self.preset_name,
            element_vector=request.b,
            explicit_log10_z_over_z_sun=(
                request.explicit_log10_z_over_z_sun
            ),
            expected_composition_axis_name=self.expected_composition_axis_name,
            initializer_name=type(self).__name__,
        )
        interpolated = interpolate_equilibrium_grid(
            self.grid,
            temperature=request.T,
            pressure=request.P,
            log10_z_over_z_sun=composition,
        )
        return interpolated.to_equilibrium_init()


__all__ = ("GridEquilibriumInitializer",)
