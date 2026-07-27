"""Grid-backed initialization for gas equilibrium."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from exogibbs.equilibrium.gas.types import (
    EquilibriumInit,
    EquilibriumInitRequest,
)

if TYPE_CHECKING:
    from exogibbs.equilibrium.gas.grid.service import EquilibriumGrid


@dataclass(frozen=True)
class GridEquilibriumInitializer:
    """Initialize a gas solve by interpolating a compatible grid."""

    grid: "EquilibriumGrid"
    preset_name: str
    expected_composition_axis_name: str = "log10(Z/Zsun)"

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        from exogibbs.equilibrium.gas.grid.service import (
            compute_physical_log10_z_over_z_sun,
            interpolate_equilibrium_grid,
            validate_equilibrium_grid_compatibility,
        )

        validate_equilibrium_grid_compatibility(
            self.grid,
            request.setup,
            self.preset_name,
            expected_composition_axis_name=self.expected_composition_axis_name,
        )
        if request.explicit_log10_z_over_z_sun is not None:
            log10_z_over_z_sun = request.explicit_log10_z_over_z_sun
        else:
            try:
                log10_z_over_z_sun = compute_physical_log10_z_over_z_sun(
                    request.setup,
                    request.b,
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    "GridEquilibriumInitializer could not infer physical "
                    f"log10(Z/Zsun) from request.b: {exc}"
                ) from exc
        interpolated = interpolate_equilibrium_grid(
            self.grid,
            temperature=request.T,
            pressure=request.P,
            log10_z_over_z_sun=log10_z_over_z_sun,
        )
        return interpolated.to_equilibrium_init()


__all__ = ("GridEquilibriumInitializer",)
