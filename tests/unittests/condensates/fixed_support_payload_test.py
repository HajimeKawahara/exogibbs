"""Tests for experimental fixed-support payload construction."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from exogibbs.condensates import (
    FixedSupportPayloadOptions as ExportedFixedSupportPayloadOptions,
)
from exogibbs.condensates.fixed_support_payload import (
    FixedSupportPayloadOptions,
    ObjectivePayloadMetric,
    build_dynamic_expansion_payload,
    seed_fixed_support_payload,
    select_objective_aware_payload,
)


def _fake_setup() -> SimpleNamespace:
    return SimpleNamespace(
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float64),
        condensate_species=("active_s", "candidate_s", "weak_s"),
        gas_setup=SimpleNamespace(
            hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        ),
        condensate_setup=SimpleNamespace(
            hvector_func=lambda temperature: jnp.asarray(
                [-3.0, -2.0, -1.0],
                dtype=jnp.float64,
            ),
            metadata={"temperature_validity_upper": (1000.0, 1000.0, 1000.0)},
        ),
    )


def test_fixed_support_payload_options_are_exported_from_condensates_namespace() -> None:
    assert ExportedFixedSupportPayloadOptions is FixedSupportPayloadOptions


def test_seed_fixed_support_payload_deduplicates_and_seeds_from_budget() -> None:
    setup = _fake_setup()

    support, amounts = seed_fixed_support_payload(
        setup=setup,
        element_inventory_target=jnp.asarray([1.0], dtype=jnp.float64),
        support_indices=(2, 1, 2),
        seed_fraction=1.0e-3,
        max_seed_amount=1.0e-3,
    )

    assert support == (2, 1)
    assert len(amounts) == 2
    assert all(amount > 0.0 for amount in amounts)


def test_dynamic_expansion_payload_uses_inactive_driving_rank() -> None:
    setup = _fake_setup()
    result = SimpleNamespace(
        layers=(
            SimpleNamespace(
                gas_n=jnp.asarray([1.0], dtype=jnp.float64),
                condensate_amounts=jnp.asarray([1.0e-20, 0.0, 0.0], dtype=jnp.float64),
            ),
        )
    )

    payload = build_dynamic_expansion_payload(
        setup=setup,
        temperatures=(500.0,),
        pressures=(1.0,),
        element_inventory_target=jnp.asarray([1.0], dtype=jnp.float64),
        result=result,
        current_support_indices=(0,),
        round_index=1,
        topk=1,
        options=FixedSupportPayloadOptions(
            dynamic_topk_grid=(1,),
            max_support_count=3,
            dynamic_active_floor=1.0e-30,
        ),
    )

    assert payload is not None
    assert payload.variant == "dynamic_activity_expansion_round1_top1_cap3"
    assert payload.support_indices == (0, 1)
    assert payload.payload_policy["dynamic_topk"] == 1


def test_objective_selection_prefers_knee_min_support_over_best_inactive() -> None:
    selection = select_objective_aware_payload(
        metrics=(
            ObjectivePayloadMetric(
                variant="curated_baseline",
                support_count=1,
                inactive=100.0,
                budget=1.0e-6,
                exogibbs_gibbs_mean=10.0,
                exogibbs_gibbs_max=12.0,
                all_converged=True,
            ),
            ObjectivePayloadMetric(
                variant="cap_payload",
                support_count=48,
                inactive=10.0,
                budget=1.0e-6,
                exogibbs_gibbs_mean=9.0,
                exogibbs_gibbs_max=11.0,
                all_converged=True,
            ),
            ObjectivePayloadMetric(
                variant="knee_payload",
                support_count=13,
                inactive=14.0,
                budget=1.0e-6,
                exogibbs_gibbs_mean=9.5,
                exogibbs_gibbs_max=11.5,
                all_converged=True,
            ),
            ObjectivePayloadMetric(
                variant="gibbs_worse_payload",
                support_count=5,
                inactive=5.0,
                budget=1.0e-6,
                exogibbs_gibbs_mean=10.1,
                exogibbs_gibbs_max=12.1,
                all_converged=True,
            ),
        ),
        options=FixedSupportPayloadOptions(max_support_count=48),
    )

    assert selection["selected_variant"] == "knee_payload"
    assert selection["inactive_first_selected"]["variant"] == "cap_payload"
    assert selection["knee_candidate_count"] == 2
    assert selection["top_rejected"][0]["variant"] == "gibbs_worse_payload"
