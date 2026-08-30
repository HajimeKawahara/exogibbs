"""Regression for the finite-barrier solar metal-sulfide initializer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.fixed_support_v2.curated_profiles import (
    element_budget_for_profile,
    fresh_profile_definition,
    support_payload_for_profile,
)
from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


FAMILY = "solar_metal_sulfide_or_Fe_Ni_S_region"
FAILING_LAYER_INDICES = (2, 7, 8)


@pytest.fixture(scope="module")
def solar_metal_sulfide_profile():
    setup = condensate_chemical_setup(silent=True)
    definition = fresh_profile_definition(FAMILY)
    budget = element_budget_for_profile(setup, definition)
    support, amounts = support_payload_for_profile(
        setup, definition, budget
    )
    temperatures = np.asarray(definition.temperatures, dtype=np.float64)
    pressures = np.asarray(definition.pressures, dtype=np.float64)
    profile = solve_condensate_profile(
        setup,
        T=temperatures,
        P=pressures,
        b=budget,
        support_indices=support,
        support_amounts_init=amounts,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )
    return setup, temperatures, pressures, profile


def test_solar_metal_sulfide_profile_accepts_all_nine_layers(
    solar_metal_sulfide_profile,
) -> None:
    _, _, _, profile = solar_metal_sulfide_profile

    assert len(profile.layers) == 9
    assert all(layer.converged for layer in profile.layers)
    assert all(layer.status == "converged" for layer in profile.layers)


def test_previously_failing_layers_accept_basic_finite_barrier_initializers(
    solar_metal_sulfide_profile,
) -> None:
    setup, temperatures, pressures, profile = solar_metal_sulfide_profile
    failing_temperatures = temperatures[list(FAILING_LAYER_INDICES)]
    failing_pressures = pressures[list(FAILING_LAYER_INDICES)]
    failing_layers = tuple(
        profile.layers[index] for index in FAILING_LAYER_INDICES
    )

    np.testing.assert_allclose(
        failing_temperatures, [787.5, 631.25, 600.0]
    )
    np.testing.assert_allclose(
        failing_pressures,
        [0.01, 3.162277660168379, 10.0],
    )
    assert all(layer.converged for layer in failing_layers)
    assert all(layer.status == "converged" for layer in failing_layers)

    expected_input_counts = (11, 12, 12)
    for layer, expected_input_count in zip(
        failing_layers, expected_input_counts
    ):
        lifecycle = layer.diagnostics["fixed_support_v2"]
        reduction = lifecycle[
            "finite_barrier_initial_support_reduction"
        ]
        support = tuple(lifecycle["initial_support_indices"])
        positive_matrix = np.asarray(setup.formula_matrix_cond)[:, support]

        assert reduction["role"] == "finite_barrier_pdipm_initializer"
        assert reduction["attempted"]
        assert reduction["applied"]
        assert reduction["fallback_reason"] is None
        assert reduction["input_support_count"] == expected_input_count
        assert reduction["input_support_rank"] == 8
        assert reduction["output_support_count"] == 8
        assert reduction["output_support_rank"] == 8
        assert len(reduction["output_dropped_support_indices"]) == (
            expected_input_count - 8
        )
        assert reduction[
            "output_scaled_inventory_residual_max_abs"
        ] <= reduction["scaled_inventory_residual_tolerance"]
        assert np.linalg.matrix_rank(positive_matrix) == len(support)


def test_finite_initializer_reduction_keeps_full_condensate_catalog(
    solar_metal_sulfide_profile,
) -> None:
    setup, _, _, profile = solar_metal_sulfide_profile
    catalog_count = len(setup.condensate_species)

    assert catalog_count > 12
    assert all(
        np.asarray(layer.condensate_amounts).shape == (catalog_count,)
        for layer in profile.layers
    )
