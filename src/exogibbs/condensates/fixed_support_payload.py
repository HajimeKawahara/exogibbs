"""Experimental fixed-support payload construction for condensate profiles.

The helpers in this module build ExoGibbs-native explicit support payloads.
They do not import FastChem4, call pyfastchem, or modify the PD-IPM solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.condensate_equilibrium import _least_squares_element_potential
from exogibbs.condensates.inactive_driving import evaluate_inactive_condensate_driving
from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)


AMOUNT_FLOOR = 1.0e-300


@dataclass(frozen=True)
class FixedSupportPayloadOptions:
    """Options for experimental profile fixed-support payload construction."""

    dynamic_topk_grid: tuple[int, ...] = (8, 12, 16)
    dynamic_rounds: int = 3
    max_support_count: int = 48
    dynamic_active_floor: float = 1.0e-30
    activity_threshold: float = 0.0
    seed_fraction: float = 1.0e-3
    max_seed_amount: float = 1.0e-3
    min_seed_amount: float = AMOUNT_FLOOR
    selection_inactive_knee_factor: float = 1.5
    accept_budget_max: float = 1.0e-4
    accept_inactive_ratio_max: float = 0.75
    accept_gibbs_mean_slack: float = 0.0
    accept_gibbs_max_slack: float = 0.0

    def __post_init__(self) -> None:
        if not self.dynamic_topk_grid:
            raise ValueError("dynamic_topk_grid must be non-empty.")
        if any(int(value) <= 0 for value in self.dynamic_topk_grid):
            raise ValueError("dynamic_topk_grid values must be positive.")
        if int(self.dynamic_rounds) < 0:
            raise ValueError("dynamic_rounds must be non-negative.")
        if int(self.max_support_count) <= 0:
            raise ValueError("max_support_count must be positive.")
        if float(self.dynamic_active_floor) <= 0.0:
            raise ValueError("dynamic_active_floor must be positive.")
        if float(self.seed_fraction) <= 0.0:
            raise ValueError("seed_fraction must be positive.")
        if float(self.max_seed_amount) <= 0.0:
            raise ValueError("max_seed_amount must be positive.")
        if float(self.min_seed_amount) <= 0.0:
            raise ValueError("min_seed_amount must be positive.")
        if float(self.selection_inactive_knee_factor) < 1.0:
            raise ValueError("selection_inactive_knee_factor must be at least one.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedSupportPayload:
    """Explicit fixed-support payload for a condensate profile solve."""

    variant: str
    support_indices: tuple[int, ...]
    support_amounts: tuple[float, ...]
    payload_policy: Mapping[str, Any]

    @property
    def support_count(self) -> int:
        return len(self.support_indices)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectivePayloadMetric:
    """Minimal metric record for objective-aware payload selection."""

    variant: str
    support_count: int
    inactive: float
    budget: float
    exogibbs_gibbs_mean: float | None
    exogibbs_gibbs_max: float | None
    all_converged: bool
    inactive_count: int | None = None
    recall_mean: float | None = None
    recall_min: float | None = None
    extra: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.extra:
            payload.update(dict(self.extra))
        return payload


def condensate_validity_upper(setup: Any) -> np.ndarray:
    """Return condensate temperature-validity upper bounds."""

    upper = setup.condensate_setup.metadata.get("temperature_validity_upper")
    if upper is None:
        return np.full((len(setup.condensate_species),), np.inf, dtype=np.float64)
    return np.asarray(upper, dtype=np.float64)


def condensate_capacity(
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    index: int,
) -> float:
    """Return the maximum amount allowed by positive elemental budgets."""

    ac = np.asarray(formula_matrix_cond, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    column = ac[:, int(index)]
    positive = column > 0.0
    if not np.any(positive):
        return float("inf")
    values = target[positive]
    if np.any(values <= 0.0):
        return 0.0
    return float(np.min(values / column[positive]))


def seed_fixed_support_payload(
    *,
    setup: Any,
    element_inventory_target: Sequence[float],
    support_indices: Sequence[int],
    seed_fraction: float = 1.0e-3,
    max_seed_amount: float = 1.0e-3,
    min_seed_amount: float = AMOUNT_FLOOR,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return deduplicated support indices and conservative seed amounts."""

    support = tuple(dict.fromkeys(int(index) for index in support_indices))
    if not support:
        return (), ()
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=setup.condensate_species,
        support_indices=support,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
        min_seed_amount=float(min_seed_amount),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_fixed_support_payload_budget",
        },
    )
    return support, tuple(float(value) for value in seed.recommended_amounts)


def _solution_inactive_candidates(
    *,
    setup: Any,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    element_inventory_target: Sequence[float],
    result: Any,
    active_floor: float,
    activity_threshold: float,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    ac = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    budget = np.asarray(jax.device_get(element_inventory_target), dtype=np.float64)
    validity_upper = condensate_validity_upper(setup)
    selected: dict[int, dict[str, Any]] = {}
    layer_summaries = []
    for layer_index, layer in enumerate(result.layers):
        temperature = float(temperatures[layer_index])
        pressure = float(pressures[layer_index])
        gas_n = np.asarray(jax.device_get(layer.gas_n), dtype=np.float64)
        cond_n = np.asarray(jax.device_get(layer.condensate_amounts), dtype=np.float64)
        gas_ln_n = jnp.log(jnp.asarray(np.maximum(gas_n, AMOUNT_FLOOR), dtype=jnp.float64))
        potential = _least_squares_element_potential(
            formula_matrix=setup.formula_matrix,
            gas_ln_n=gas_ln_n,
            gas_stationarity_source=setup.gas_setup.hvector_func(temperature)
            + math.log(pressure),
        )
        hcond = np.asarray(setup.condensate_setup.hvector_func(temperature), dtype=np.float64)
        driving = np.asarray(
            jax.device_get(ac.T @ np.asarray(potential) - hcond),
            dtype=np.float64,
        )
        temperature_valid = temperature <= validity_upper
        active = cond_n > float(active_floor)
        candidates = []
        for index, value in enumerate(driving):
            if bool(active[index]) or not bool(temperature_valid[index]):
                continue
            cap = condensate_capacity(ac, budget, index)
            if cap <= 0.0 or not np.isfinite(cap):
                continue
            if float(value) <= float(activity_threshold):
                continue
            row = {
                "index": int(index),
                "species": str(setup.condensate_species[index]),
                "driving": float(value),
                "capacity": float(cap),
                "amount": float(cond_n[index]),
            }
            candidates.append(row)
            old = selected.get(index)
            if old is None or float(row["driving"]) > float(old["driving"]):
                selected[index] = row
        candidates.sort(
            key=lambda row: (-float(row["driving"]), -float(row["capacity"]), int(row["index"]))
        )
        layer_summaries.append(
            {
                "layer_index": int(layer_index),
                "temperature": temperature,
                "pressure": pressure,
                "active_floor": float(active_floor),
                "max_positive_driving": float(candidates[0]["driving"]) if candidates else 0.0,
                "positive_count": len(candidates),
                "top_candidates": candidates[:20],
            }
        )
    ranked = tuple(
        sorted(
            selected.values(),
            key=lambda row: (-float(row["driving"]), -float(row["capacity"]), int(row["index"])),
        )
    )
    return {
        "active_floor": float(active_floor),
        "activity_threshold": float(activity_threshold),
        "layer_candidate_summary": layer_summaries,
        "candidate_count": len(ranked),
        "top_candidates": ranked[:50],
    }, ranked


def build_solution_inactive_expansion_payload(
    *,
    setup: Any,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    element_inventory_target: Sequence[float],
    result: Any,
    current_support_indices: Sequence[int],
    topk: int,
    variant: str,
    policy_name: str,
    options: FixedSupportPayloadOptions = FixedSupportPayloadOptions(),
    extra_policy: Mapping[str, Any] | None = None,
) -> FixedSupportPayload | None:
    """Expand a support payload from inactive driving in a solved profile."""

    report, ranked = _solution_inactive_candidates(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        element_inventory_target=element_inventory_target,
        result=result,
        active_floor=float(options.dynamic_active_floor),
        activity_threshold=float(options.activity_threshold),
    )
    additions = [int(row["index"]) for row in ranked[: int(topk)]]
    if not additions:
        return None
    support = list(dict.fromkeys([*current_support_indices, *additions]))[
        : int(options.max_support_count)
    ]
    if tuple(support) == tuple(int(index) for index in current_support_indices):
        return None
    support_indices, support_amounts = seed_fixed_support_payload(
        setup=setup,
        element_inventory_target=element_inventory_target,
        support_indices=support,
        seed_fraction=float(options.seed_fraction),
        max_seed_amount=float(options.max_seed_amount),
        min_seed_amount=float(options.min_seed_amount),
    )
    policy = {
        "policy": policy_name,
        "topk": int(topk),
        "max_support_count": int(options.max_support_count),
        "expansion_report": report,
        "fastchem4_constructor_inputs_used": False,
    }
    if extra_policy:
        policy.update(dict(extra_policy))
    return FixedSupportPayload(
        variant=variant,
        support_indices=support_indices,
        support_amounts=support_amounts,
        payload_policy=policy,
    )


def build_baseline_inactive_expansion_payloads(
    *,
    setup: Any,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    element_inventory_target: Sequence[float],
    baseline_result: Any,
    baseline_support_indices: Sequence[int],
    topk_values: Sequence[int],
    options: FixedSupportPayloadOptions = FixedSupportPayloadOptions(),
) -> tuple[FixedSupportPayload, ...]:
    """Build baseline-solution inactive-driving expansion payloads."""

    payloads = []
    for topk in topk_values:
        payload = build_solution_inactive_expansion_payload(
            setup=setup,
            temperatures=temperatures,
            pressures=pressures,
            element_inventory_target=element_inventory_target,
            result=baseline_result,
            current_support_indices=baseline_support_indices,
            topk=int(topk),
            variant=(
                f"baseline_inactive_expansion_top{int(topk)}"
                f"_cap{int(options.max_support_count)}"
            ),
            policy_name="baseline_solution_temperature_valid_inactive_driving_expansion",
            options=options,
        )
        if payload is not None:
            payloads.append(payload)
    return tuple(payloads)


def build_dynamic_expansion_payload(
    *,
    setup: Any,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    element_inventory_target: Sequence[float],
    result: Any,
    current_support_indices: Sequence[int],
    round_index: int,
    topk: int,
    options: FixedSupportPayloadOptions = FixedSupportPayloadOptions(),
) -> FixedSupportPayload | None:
    """Build one iterative inactive-driving dynamic expansion payload."""

    return build_solution_inactive_expansion_payload(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        element_inventory_target=element_inventory_target,
        result=result,
        current_support_indices=current_support_indices,
        topk=int(topk),
        variant=(
            f"dynamic_activity_expansion_round{int(round_index)}"
            f"_top{int(topk)}_cap{int(options.max_support_count)}"
        ),
        policy_name="iterative_solution_inactive_driving_support_expansion",
        options=options,
        extra_policy={
            "dynamic_round": int(round_index),
            "dynamic_topk": int(topk),
        },
    )


def inactive_driving_summary_for_state(
    *,
    setup: Any,
    temperature: float,
    pressure: float,
    gas_n: Sequence[float],
    condensate_amounts: Sequence[float],
    active_floor: float = 1.0e-30,
    activity_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate inactive driving for a solved state using native arrays."""

    gas_ln_n = jnp.log(jnp.asarray(np.maximum(gas_n, AMOUNT_FLOOR), dtype=jnp.float64))
    potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=gas_ln_n,
        gas_stationarity_source=setup.gas_setup.hvector_func(float(temperature))
        + math.log(float(pressure)),
    )
    return evaluate_inactive_condensate_driving(
        formula_matrix_cond=setup.formula_matrix_cond,
        condensate_species_order=setup.condensate_species,
        condensate_amounts=condensate_amounts,
        hvector_cond=setup.condensate_setup.hvector_func(float(temperature)),
        element_potential=potential,
        temperature=float(temperature),
        condensate_temperature_validity_upper=condensate_validity_upper(setup),
        active_floor=float(active_floor),
        activity_threshold=float(activity_threshold),
    ).as_dict()


def select_objective_aware_payload(
    *,
    metrics: Sequence[ObjectivePayloadMetric],
    baseline_variant: str = "curated_baseline",
    options: FixedSupportPayloadOptions = FixedSupportPayloadOptions(),
) -> dict[str, Any]:
    """Select a payload with convergence, budget, Gibbs, inactive, and knee gates."""

    metric_rows = [metric.as_dict() for metric in metrics]
    baseline = next((row for row in metric_rows if row["variant"] == baseline_variant), None)
    if baseline is None:
        return {
            "selection_schema": "exogibbs_objective_aware_payload_selection_v1",
            "selected_variant": None,
            "reason": "missing_curated_baseline",
        }
    candidates = []
    rejected = []
    for row in metric_rows:
        mean_delta = None
        max_delta = None
        if row["exogibbs_gibbs_mean"] is not None and baseline["exogibbs_gibbs_mean"] is not None:
            mean_delta = float(row["exogibbs_gibbs_mean"]) - float(baseline["exogibbs_gibbs_mean"])
        if row["exogibbs_gibbs_max"] is not None and baseline["exogibbs_gibbs_max"] is not None:
            max_delta = float(row["exogibbs_gibbs_max"]) - float(baseline["exogibbs_gibbs_max"])
        inactive_ratio = (
            0.0
            if float(baseline["inactive"]) <= 0.0
            else float(row["inactive"]) / float(baseline["inactive"])
        )
        annotated = {
            **row,
            "gibbs_mean_delta_vs_baseline": mean_delta,
            "gibbs_max_delta_vs_baseline": max_delta,
            "inactive_ratio_vs_baseline": inactive_ratio,
            "inactive_improvement_factor": (
                None
                if float(row["inactive"]) <= 0.0
                else float(baseline["inactive"]) / float(row["inactive"])
            ),
        }
        gates = {
            "converged": bool(row["all_converged"]),
            "budget": float(row["budget"]) <= float(options.accept_budget_max),
            "gibbs_mean": (
                mean_delta is not None
                and mean_delta <= float(options.accept_gibbs_mean_slack)
            ),
            "gibbs_max": (
                max_delta is not None
                and max_delta <= float(options.accept_gibbs_max_slack)
            ),
            "inactive_improved": inactive_ratio <= float(options.accept_inactive_ratio_max),
            "support_size": int(row["support_count"]) <= int(options.max_support_count),
        }
        annotated["objective_acceptance_gates"] = gates
        if all(gates.values()):
            candidates.append(annotated)
        else:
            rejected.append(
                {
                    **annotated,
                    "reason": ",".join(key for key, value in gates.items() if not value),
                }
            )
    if not candidates:
        return {
            "selection_schema": "exogibbs_objective_aware_payload_selection_v1",
            "selected_variant": baseline["variant"],
            "selected_support_count": baseline["support_count"],
            "selected_reason": "fallback_to_baseline_no_objective_accepted_candidate",
            "baseline": baseline,
            "candidate_count": 0,
            "rejected_count": len(rejected),
            "top_rejected": sorted(
                rejected,
                key=lambda row: (
                    row.get("inactive_ratio_vs_baseline", float("inf")),
                    row.get("support_count", 0),
                ),
            )[:10],
        }
    best_inactive = min(float(row["inactive"]) for row in candidates)
    knee_candidates = [
        row
        for row in candidates
        if float(row["inactive"])
        <= best_inactive * float(options.selection_inactive_knee_factor)
    ] or candidates
    selected = min(
        knee_candidates,
        key=lambda row: (
            row["support_count"],
            row["inactive_ratio_vs_baseline"],
            row["gibbs_mean_delta_vs_baseline"],
        ),
    )
    inactive_first_selected = min(
        candidates,
        key=lambda row: (
            row["inactive_ratio_vs_baseline"],
            row["support_count"],
            row["gibbs_mean_delta_vs_baseline"],
        ),
    )
    return {
        "selection_schema": "exogibbs_objective_aware_payload_selection_v1",
        "selected_variant": selected["variant"],
        "selected_support_count": selected["support_count"],
        "selected_reason": "objective_accepted_knee_min_support",
        "baseline": baseline,
        "selected": selected,
        "inactive_first_selected": inactive_first_selected,
        "best_inactive": best_inactive,
        "selection_inactive_knee_factor": float(options.selection_inactive_knee_factor),
        "knee_candidate_count": len(knee_candidates),
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "top_candidates": sorted(
            candidates,
            key=lambda row: (
                row["inactive_ratio_vs_baseline"],
                row["support_count"],
                row["gibbs_mean_delta_vs_baseline"],
            ),
        )[:10],
        "top_rejected": sorted(
            rejected,
            key=lambda row: (
                row.get("inactive_ratio_vs_baseline", float("inf")),
                row.get("support_count", 0),
            ),
        )[:10],
    }


__all__ = (
    "AMOUNT_FLOOR",
    "FixedSupportPayload",
    "FixedSupportPayloadOptions",
    "ObjectivePayloadMetric",
    "build_baseline_inactive_expansion_payloads",
    "build_dynamic_expansion_payload",
    "build_solution_inactive_expansion_payload",
    "condensate_capacity",
    "condensate_validity_upper",
    "inactive_driving_summary_for_state",
    "seed_fixed_support_payload",
    "select_objective_aware_payload",
)
