"""Tests for production fixed-support v2 support seeding."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from exogibbs.condensates import (
    seed_fixed_support_payload as exported_seed_fixed_support_payload,
)
from exogibbs.condensates.fixed_support_payload import (
    seed_fixed_support_payload,
)


def _fake_setup() -> SimpleNamespace:
    return SimpleNamespace(
        formula_matrix_cond=jnp.asarray(
            [[1.0, 1.0, 1.0]],
            dtype=jnp.float64,
        ),
        condensate_species=("first_s", "second_s", "third_s"),
    )


def test_seed_helper_is_exported_from_condensates_namespace() -> None:
    assert exported_seed_fixed_support_payload is seed_fixed_support_payload


def test_seed_fixed_support_payload_deduplicates_and_preserves_budget() -> None:
    support, amounts = seed_fixed_support_payload(
        setup=_fake_setup(),
        element_inventory_target=jnp.asarray([1.0], dtype=jnp.float64),
        support_indices=(2, 1, 2),
        seed_fraction=1.0e-3,
        max_seed_amount=1.0e-3,
    )

    assert support == (2, 1)
    assert len(amounts) == 2
    assert all(amount > 0.0 for amount in amounts)
    assert sum(amounts) <= 1.0e-3


def test_seed_fixed_support_payload_rejects_zero_capacity_support() -> None:
    setup = SimpleNamespace(
        formula_matrix_cond=jnp.asarray([[1.0], [1.0]], dtype=jnp.float64),
        condensate_species=("AB_s",),
    )

    with pytest.raises(ValueError, match="cannot receive a positive seed"):
        seed_fixed_support_payload(
            setup=setup,
            element_inventory_target=jnp.asarray([1.0, 0.0], dtype=jnp.float64),
            support_indices=(0,),
        )
