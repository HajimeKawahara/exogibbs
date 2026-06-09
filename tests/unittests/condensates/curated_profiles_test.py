"""Tests for native curated condensate profile definitions."""

from __future__ import annotations

import jax.numpy as jnp

from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    case_id_for_profile,
    element_budget_for_profile,
    fresh_profile_definition,
    support_payload_for_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


def test_fresh_curated_profiles_expose_the_demo_family_set() -> None:
    assert len(FRESH_CURATED_PROFILES) == 10
    assert fresh_profile_definition("solar_water_condensation").support_species == (
        "H2O(s,l)",
    )


def test_case_id_for_profile_matches_existing_demo_labels() -> None:
    definition = fresh_profile_definition("solar_water_condensation")

    assert case_id_for_profile(definition, 300.0, 0.1) == (
        "solar_water_condensation__T300_P0p1"
    )


def test_support_payload_for_profile_uses_native_budget_seed() -> None:
    setup = condensate_chemical_setup(silent=True)
    definition = fresh_profile_definition("solar_water_condensation")
    budget = element_budget_for_profile(setup, definition)

    support_indices, support_amounts = support_payload_for_profile(setup, definition, budget)

    assert tuple(setup.condensate_species[index] for index in support_indices) == (
        "H2O(s,l)",
    )
    assert len(support_amounts) == len(support_indices)
    assert all(amount > 0.0 for amount in support_amounts)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(support_amounts))))
