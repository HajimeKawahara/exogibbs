"""Contracts for the bounded rainout inventory bridge."""

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.equilibrium.condensate.inventory_bridge import (
    InventoryBridgeConfig,
    interpolate_element_inventory,
    validate_equilibrium_point,
    validate_inventory_bridge_config,
)
from exogibbs.equilibrium.condensate.types import CondensateEquilibriumPoint


def test_inventory_bridge_uses_log_interpolation_for_positive_rows() -> None:
    origin = np.asarray([1.0, 1.0e-8, 0.0])
    target = np.asarray([4.0, 1.0e-12, 2.0])

    midpoint = interpolate_element_inventory(origin, target, 0.5)

    np.testing.assert_allclose(midpoint, [2.0, 1.0e-10, 1.0])
    np.testing.assert_array_equal(
        interpolate_element_inventory(origin, target, 0.0),
        origin,
    )
    np.testing.assert_array_equal(
        interpolate_element_inventory(origin, target, 1.0),
        target,
    )


@pytest.mark.parametrize(
    "origin,target,fraction,match",
    [
        ([1.0], [1.0, 2.0], 0.5, "same shape"),
        ([-1.0], [1.0], 0.5, "non-negative"),
        ([1.0], [np.inf], 0.5, "finite"),
        ([1.0], [1.0], -0.1, "between 0 and 1"),
    ],
)
def test_inventory_bridge_rejects_invalid_interpolation_inputs(
    origin,
    target,
    fraction,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        interpolate_element_inventory(origin, target, fraction)


def test_inventory_bridge_validates_origin_and_bounded_policy() -> None:
    point = CondensateEquilibriumPoint(
        temperature=300.0,
        pressure=10.0,
        element_inventory=jnp.asarray([0.7, 0.3, 0.0]),
    )

    inventory = validate_equilibrium_point(
        point,
        expected_inventory_shape=(3,),
    )
    validate_inventory_bridge_config(InventoryBridgeConfig())

    np.testing.assert_array_equal(inventory, [0.7, 0.3, 0.0])
    with pytest.raises(ValueError, match="temperature"):
        validate_equilibrium_point(
            CondensateEquilibriumPoint(0.0, 10.0, point.element_inventory),
            expected_inventory_shape=(3,),
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_inventory_bridge_config(
            InventoryBridgeConfig(anchor_fractions=(1.0,))
        )
    with pytest.raises(ValueError, match="positive"):
        validate_inventory_bridge_config(
            InventoryBridgeConfig(max_lifecycle_solves=0)
        )
