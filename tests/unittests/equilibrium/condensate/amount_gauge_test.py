"""Consistent amount gauges across rainout and canonical initialization."""

import math

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.lifecycle import (
    _normalize_condensate_init_amount_gauge,
)
from exogibbs.equilibrium.condensate.profile import _scale_initial_guess
from exogibbs.equilibrium.condensate.types import (
    CondensateEquilibriumInit,
    CondensateEquilibriumPoint,
)


@pytest.mark.parametrize("amount_scale", [0.5, 8.0, 1.0e-310])
def test_rainout_to_canonical_gauge_round_trip_includes_bridge_origin(
    amount_scale: float,
) -> None:
    initial = CondensateEquilibriumInit(
        gas_ln_n=np.log([1.0, 0.5]),
        gas_ntot=np.asarray(1.5),
        condensate_amounts=np.asarray([0.5, 0.0]),
        support_indices=(0,),
        support_amounts=(0.5,),
        element_potential=np.asarray([2.0, 3.0, 4.0]),
        rho=np.asarray([0.25]),
        barrier_epsilon=np.asarray(-11.0),
        inventory_bridge_origin=CondensateEquilibriumPoint(
            temperature=300.0,
            pressure=10.0,
            element_inventory=np.asarray([1.0, 0.5, 0.0]),
        ),
    )

    caller = _scale_initial_guess(initial, amount_scale)
    restored = _normalize_condensate_init_amount_gauge(caller, amount_scale)

    for field in (
        "gas_ln_n", "gas_ntot", "condensate_amounts", "support_amounts",
        "barrier_epsilon",
    ):
        np.testing.assert_allclose(
            getattr(restored, field), getattr(initial, field),
            rtol=1.0e-12, atol=1.0e-13,
        )
    np.testing.assert_allclose(
        caller.inventory_bridge_origin.element_inventory,
        initial.inventory_bridge_origin.element_inventory * amount_scale,
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        restored.inventory_bridge_origin.element_inventory,
        initial.inventory_bridge_origin.element_inventory,
        rtol=1.0e-12, atol=0.0,
    )
    assert restored.inventory_bridge_origin.temperature == 300.0
    assert restored.inventory_bridge_origin.pressure == 10.0
    assert restored.support_indices == initial.support_indices
    assert restored.element_potential is initial.element_potential
    assert restored.rho is initial.rho
    assert float(caller.barrier_epsilon) == pytest.approx(
        float(initial.barrier_epsilon) + math.log(amount_scale)
    )


@pytest.mark.parametrize("amount_scale", [0.0, -1.0, np.inf, np.nan])
def test_rainout_gauge_rejects_invalid_scales_even_without_amount_fields(
    amount_scale: float,
) -> None:
    with pytest.raises(ValueError, match="amount_scale must be finite and positive"):
        _scale_initial_guess(CondensateEquilibriumInit(), amount_scale)
