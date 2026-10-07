"""Pure policy helpers for bounded rainout inventory bridges."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from exogibbs.equilibrium.condensate.types import CondensateEquilibriumPoint


@dataclass(frozen=True)
class InventoryBridgeConfig:
    """Bound the numerical work used to find an interior inventory anchor."""

    anchor_fractions: tuple[float, ...] = (0.5,)
    max_lifecycle_solves: int = 2


def validate_inventory_bridge_config(config: InventoryBridgeConfig) -> None:
    """Validate a deterministic interior-anchor portfolio."""

    if not isinstance(config.max_lifecycle_solves, int):
        raise TypeError("max_lifecycle_solves must be an int.")
    if config.max_lifecycle_solves < 1:
        raise ValueError("max_lifecycle_solves must be positive.")
    if not config.anchor_fractions:
        raise ValueError("anchor_fractions must not be empty.")
    fractions = tuple(float(value) for value in config.anchor_fractions)
    if any(
        not math.isfinite(value) or value <= 0.0 or value >= 1.0
        for value in fractions
    ):
        raise ValueError(
            "anchor_fractions must contain only finite values between 0 and 1."
        )
    if len(set(fractions)) != len(fractions):
        raise ValueError("anchor_fractions must not contain duplicates.")


def validate_equilibrium_point(
    point: CondensateEquilibriumPoint,
    *,
    expected_inventory_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a validated float64 inventory from an equilibrium provenance."""

    temperature = float(point.temperature)
    pressure = float(point.pressure)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Bridge-origin temperature must be finite and positive.")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("Bridge-origin pressure must be finite and positive.")
    inventory = np.asarray(point.element_inventory, dtype=np.float64)
    if inventory.shape != expected_inventory_shape:
        raise ValueError(
            "Bridge-origin element inventory has the wrong shape: expected "
            f"{expected_inventory_shape}, got {inventory.shape}."
        )
    if not np.all(np.isfinite(inventory)):
        raise ValueError(
            "Bridge-origin element inventory must contain only finite values."
        )
    if np.any(inventory < 0.0):
        raise ValueError(
            "Bridge-origin element inventory must be non-negative."
        )
    return inventory.copy()


def interpolate_element_inventory(
    origin: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate positive rows logarithmically and zero endpoints linearly."""

    source = np.asarray(origin, dtype=np.float64)
    destination = np.asarray(target, dtype=np.float64)
    if source.shape != destination.shape:
        raise ValueError("origin and target inventories must have the same shape.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("origin and target inventories must be finite.")
    if np.any(source < 0.0) or np.any(destination < 0.0):
        raise ValueError("origin and target inventories must be non-negative.")
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("fraction must be finite and lie between 0 and 1.")
    if value == 0.0:
        return source.copy()
    if value == 1.0:
        return destination.copy()

    result = (1.0 - value) * source + value * destination
    logarithmic = (source > 0.0) & (destination > 0.0)
    result[logarithmic] = np.exp(
        (1.0 - value) * np.log(source[logarithmic])
        + value * np.log(destination[logarithmic])
    )
    return result


__all__ = (
    "InventoryBridgeConfig",
    "interpolate_element_inventory",
    "validate_equilibrium_point",
    "validate_inventory_bridge_config",
)
