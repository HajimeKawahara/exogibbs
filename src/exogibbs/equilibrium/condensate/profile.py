"""Dependent Python-level profile schedulers for condensate equilibrium."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate.initialization import (
    resolve_condensate_initial_guess,
)
from exogibbs.equilibrium.condensate.policy import (
    FixedSupportV2ProductionPolicy,
    fixed_support_v2_production_policy,
)
from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
from exogibbs.equilibrium.condensate.types import (
    CONVERGED,
    Array,
    CondensateEquilibriumInit,
    CondensateEquilibriumInitRequest,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumOptions,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
)


def _validate_rainout_inventory(
    setup: CondensateChemicalSetup,
    b: Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    inventory = np.asarray(jax.device_get(b), dtype=np.float64)
    expected_shape = (len(setup.elements),)
    if inventory.shape != expected_shape:
        raise ValueError(
            "b must have one entry per element; expected shape "
            f"{expected_shape}, got {inventory.shape}."
        )
    if not np.all(np.isfinite(inventory)):
        raise ValueError("b must contain only finite values.")
    if np.any(inventory < 0.0):
        raise ValueError("b must contain only non-negative values.")
    conserved_mask = np.asarray(
        tuple(
            str(element).strip().lower() not in {"e-", "electron"}
            for element in setup.elements
        ),
        dtype=bool,
    )
    normalization_mask = conserved_mask & (inventory > 0.0)
    inventory_sum = float(np.sum(inventory[normalization_mask]))
    if not math.isfinite(inventory_sum) or inventory_sum <= 0.0:
        raise ValueError(
            "b must contain at least one positive non-electron element "
            "abundance."
        )
    return (
        inventory.copy(),
        conserved_mask,
        normalization_mask,
        inventory_sum,
    )


def _rainout_gauge_scales(
    inventory: np.ndarray,
    normalization_mask: np.ndarray,
    *,
    minimum_targets: Sequence[float],
    maximum_total: float,
    total_targets: Sequence[float],
) -> tuple[float, ...]:
    """Return bounded gauges with the highest feasible gauge first.

    Rainout normalizes the total element inventory after every layer.  Using
    the largest feasible total as the primary gauge therefore keeps the
    working scale constant along the profile.  The remaining values are
    strictly descending retry gauges; trace-element thresholds never select
    the primary gauge.
    """

    active = inventory[normalization_mask]
    positive = active[active > 0.0]
    minimum = float(np.min(positive))
    total = float(np.sum(active))
    if not math.isfinite(maximum_total) or maximum_total <= 0.0:
        raise ValueError("maximum_total must be finite and positive.")
    maximum_scale = maximum_total / total
    safe_maximum_scale = float(np.nextafter(maximum_scale, 0.0))
    baseline_scale = min(1.0, safe_maximum_scale)
    candidates = [safe_maximum_scale]
    for target in total_targets:
        candidate = float(target) / total
        if total <= maximum_total:
            candidate = max(1.0, candidate)
        candidate = min(candidate, safe_maximum_scale)
        candidates.append(candidate)
    for target in minimum_targets:
        candidate = float(target) / minimum
        if total <= maximum_total:
            candidate = max(1.0, candidate)
        candidates.append(min(candidate, safe_maximum_scale))
    candidates.append(baseline_scale)
    candidates.sort(reverse=True)
    unique: list[float] = []
    for candidate in candidates:
        if (
            not math.isfinite(candidate)
            or candidate <= 0.0
            or candidate * total > maximum_total
        ):
            continue
        if not any(
            math.isclose(candidate, prior, rel_tol=1.0e-12, abs_tol=0.0)
            for prior in unique
        ):
            unique.append(candidate)
    if not unique:
        fallback = maximum_total / (2.0 * total)
        if not math.isfinite(fallback) or fallback <= 0.0:
            raise ValueError("Unable to construct a bounded rainout gauge scale.")
        unique.append(fallback)
    return tuple(unique)


def _scale_initial_guess(
    initial_guess: CondensateEquilibriumInit,
    scale: float,
) -> CondensateEquilibriumInit:
    if scale == 1.0:
        return initial_guess
    log_scale = math.log(scale)
    return replace(
        initial_guess,
        gas_ln_n=(
            None
            if initial_guess.gas_ln_n is None
            else jnp.asarray(initial_guess.gas_ln_n, dtype=jnp.float64)
            + log_scale
        ),
        gas_ntot=(
            None
            if initial_guess.gas_ntot is None
            else jnp.asarray(initial_guess.gas_ntot, dtype=jnp.float64)
            * scale
        ),
        condensate_amounts=(
            None
            if initial_guess.condensate_amounts is None
            else jnp.asarray(
                initial_guess.condensate_amounts,
                dtype=jnp.float64,
            )
            * scale
        ),
        support_amounts=(
            None
            if initial_guess.support_amounts is None
            else tuple(float(value) * scale for value in initial_guess.support_amounts)
        ),
    )


def _initialization_attempts(
    initial_guess: CondensateEquilibriumInit,
) -> tuple[tuple[str, CondensateEquilibriumInit], ...]:
    values = (
        initial_guess.gas_ln_n,
        initial_guess.gas_ntot,
        initial_guess.condensate_amounts,
        initial_guess.support_indices,
        initial_guess.support_amounts,
        initial_guess.element_potential,
        initial_guess.rho,
        initial_guess.barrier_epsilon,
    )
    if not any(value is not None for value in values):
        return (("cold", initial_guess),)
    return (
        ("resolved", initial_guess),
        ("cold_fallback", CondensateEquilibriumInit()),
    )


def _rescale_layer_result(
    result: CondensateEquilibriumResult,
    scale: float,
) -> CondensateEquilibriumResult:
    if scale == 1.0:
        return result
    inverse_scale = 1.0 / scale
    return replace(
        result,
        gas_ln_n=jnp.asarray(result.gas_ln_n) - math.log(scale),
        gas_n=jnp.asarray(result.gas_n) * inverse_scale,
        gas_ntot=jnp.asarray(result.gas_ntot) * inverse_scale,
        condensate_amounts=(
            jnp.asarray(result.condensate_amounts) * inverse_scale
        ),
    )


def _with_rainout_layer_diagnostics(
    result: CondensateEquilibriumResult,
    *,
    layer_index: int,
    abundance_scale: float,
    preferred_abundance_scale: float,
    previous_abundance_scale: float | None,
    working_inventory_total: float,
    attempts: Sequence[Mapping[str, Any]],
    depleted_projection: Mapping[str, Any],
    floorless_budget: Mapping[str, Any],
) -> CondensateEquilibriumResult:
    diagnostics = dict(result.diagnostics or {})
    diagnostics["rainout"] = {
        "schema": "exogibbs_condensate_rainout_layer_v1",
        "layer_index": layer_index,
        "scan_direction": "bottom_to_top",
        "abundance_scale": abundance_scale,
        "preferred_abundance_scale": preferred_abundance_scale,
        "previous_abundance_scale": previous_abundance_scale,
        "accepted_to_preferred_scale_ratio": (
            abundance_scale / preferred_abundance_scale
        ),
        "accepted_to_previous_scale_ratio": (
            None
            if previous_abundance_scale is None
            else abundance_scale / previous_abundance_scale
        ),
        "working_inventory_total": working_inventory_total,
        "solver_diagnostics_gauge": (
            "caller_abundance_gauge_times_abundance_scale"
        ),
        "public_result_gauge": "caller_abundance_gauge",
        "depleted_element_projection": dict(depleted_projection),
        "floorless_budget_certification": dict(floorless_budget),
        "attempts": tuple(dict(attempt) for attempt in attempts),
    }
    return replace(result, diagnostics=diagnostics)


def _is_retryable_numerical_value_error(error: ValueError) -> bool:
    message = str(error).lower()
    return (
        "must contain only finite values" in message
        or "non-finite" in message
        or "nonfinite" in message
    )


def _trace_capacity_acceptance_report(
    *,
    setup: CondensateChemicalSetup,
    inventory: np.ndarray,
    inventory_sum: float,
    candidate: CondensateEquilibriumResult,
    abundance_scale: float,
    policy: FixedSupportV2ProductionPolicy,
) -> Mapping[str, Any] | None:
    # A NORMAL_MAX_ITER state does not establish the sign of the active
    # condensate driving force.  In particular, accepting such a state can
    # turn a numerical trace capacity into irreversible physical rainout.
    # Keep this legacy escape hatch explicit and disabled in production.
    if not policy.rainout_allow_trace_capacity_acceptance:
        return None
    diagnostics = candidate.diagnostics or {}
    lifecycle = diagnostics.get("fixed_support_v2", {})
    if not isinstance(lifecycle, Mapping):
        return None
    kkt = lifecycle.get("independent_kkt", {})
    if not isinstance(kkt, Mapping):
        return None
    budget_gate = diagnostics.get(
        "full_condensate_budget_residual_gate", {}
    )
    if (
        not isinstance(budget_gate, Mapping)
        or not bool(budget_gate.get("enabled", False))
        or not bool(budget_gate.get("accepted", False))
    ):
        return None
    if not math.isfinite(abundance_scale) or abundance_scale <= 0.0:
        return None
    budget_target = np.asarray(
        budget_gate.get("element_budget_target", ()), dtype=np.float64
    )
    budget_reconstructed = np.asarray(
        budget_gate.get("element_budget_reconstructed", ()),
        dtype=np.float64,
    )
    expected_budget_shape = (len(setup.elements),)
    if (
        budget_target.shape != expected_budget_shape
        or budget_reconstructed.shape != expected_budget_shape
        or not np.all(np.isfinite(budget_target))
        or not np.all(np.isfinite(budget_reconstructed))
        or np.any(budget_target < 0.0)
    ):
        return None
    budget_relative_tolerance = float(
        budget_gate.get("relative_tolerance", math.nan)
    )
    if (
        not math.isfinite(budget_relative_tolerance)
        or budget_relative_tolerance <= 0.0
    ):
        return None
    budget_gate_mask = np.asarray(
        tuple(
            str(element) not in {"e-", "electron"}
            for element in setup.elements
        ),
        dtype=bool,
    )
    positive_budget_mask = budget_gate_mask & (budget_target > 0.0)
    positive_relative_residual = np.zeros_like(budget_target)
    positive_relative_residual[positive_budget_mask] = np.abs(
        budget_reconstructed[positive_budget_mask]
        - budget_target[positive_budget_mask]
    ) / budget_target[positive_budget_mask]
    maximum_positive_relative_residual = float(
        np.max(positive_relative_residual, initial=0.0)
    )
    if maximum_positive_relative_residual > budget_relative_tolerance:
        return None
    positive_targets = budget_target[positive_budget_mask]
    budget_scale = (
        float(np.max(positive_targets)) if positive_targets.size else 1.0
    )
    zero_budget_absolute_tolerance = (
        float(np.finfo(np.float64).tiny) * max(1.0, budget_scale)
    )
    zero_budget_mask = budget_gate_mask & (budget_target == 0.0)
    maximum_zero_budget_absolute_reconstructed = float(
        np.max(
            np.abs(budget_reconstructed[zero_budget_mask]),
            initial=0.0,
        )
    )
    if (
        maximum_zero_budget_absolute_reconstructed
        > zero_budget_absolute_tolerance
    ):
        return None
    if (
        lifecycle.get("outcome") != "fixed_support_failed"
        or lifecycle.get("terminal_status_name") != "NORMAL_MAX_ITER"
        or not bool(lifecycle.get("support_closed", False))
        or not bool(lifecycle.get("final_state_values_finite", False))
    ):
        return None
    tolerances = policy.solver_config.normal
    required_components = (
        ("gas_stationarity", tolerances.stationarity_tolerance),
        ("budget_scaled", tolerances.budget_tolerance),
        ("complementarity", tolerances.complementarity_tolerance),
        ("total_density_scaled", tolerances.total_density_tolerance),
    )
    if any(
        not math.isfinite(float(kkt.get(name, math.inf)))
        or float(kkt.get(name, math.inf)) > float(tolerance)
        for name, tolerance in required_components
    ):
        return None
    condensate_stationarity = float(
        kkt.get("condensate_stationarity", math.inf)
    )
    if (
        not math.isfinite(condensate_stationarity)
        or condensate_stationarity
        > policy.rainout_trace_condensate_stationarity_tolerance
    ):
        return None
    support = tuple(
        int(index)
        for index in np.asarray(
            jax.device_get(candidate.condensate_support_indices),
            dtype=np.int64,
        ).tolist()
    )
    if not support or len(set(support)) != len(support):
        return None
    formula_cond = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    condensate_amounts = np.asarray(
        jax.device_get(candidate.condensate_amounts), dtype=np.float64
    )
    if (
        condensate_amounts.shape != (len(setup.condensate_species),)
        or not np.all(np.isfinite(condensate_amounts))
        or np.any(condensate_amounts < 0.0)
    ):
        return None
    capacities = []
    actual_amounts = []
    for index in support:
        if index < 0 or index >= len(setup.condensate_species):
            return None
        column = formula_cond[:, index]
        positive = column > 0.0
        if not np.any(positive):
            return None
        if np.any(inventory[positive] <= 0.0):
            capacity = 0.0
        else:
            capacity = float(np.min(inventory[positive] / column[positive]))
        if capacity <= 0.0:
            return None
        actual_amount = float(condensate_amounts[index]) / abundance_scale
        if actual_amount > capacity * (1.0 + 1.0e-12):
            return None
        capacities.append(capacity)
        actual_amounts.append(actual_amount)
    maximum_relative_capacity = max(capacities) / inventory_sum
    if (
        not math.isfinite(maximum_relative_capacity)
        or maximum_relative_capacity
        > policy.rainout_trace_capacity_relative_tolerance
    ):
        return None
    if not np.all(
        np.isfinite(
            np.asarray(jax.device_get(candidate.gas_ln_n), dtype=np.float64)
        )
    ):
        return None
    return {
        "schema": "exogibbs_rainout_trace_capacity_acceptance_v1",
        "accepted": True,
        "reason": (
            "All non-condensate-stationarity KKT components passed and the "
            "closed active condensate capacity was below the configured "
            "rainout trace threshold."
        ),
        "support_indices": support,
        "support_names": tuple(
            setup.condensate_species[index] for index in support
        ),
        "support_capacities": tuple(capacities),
        "support_actual_amounts": tuple(actual_amounts),
        "maximum_relative_capacity": maximum_relative_capacity,
        "capacity_relative_tolerance": (
            policy.rainout_trace_capacity_relative_tolerance
        ),
        "condensate_stationarity": condensate_stationarity,
        "condensate_stationarity_tolerance": (
            policy.rainout_trace_condensate_stationarity_tolerance
        ),
        "floorless_maximum_positive_relative_budget_residual": (
            maximum_positive_relative_residual
        ),
        "floorless_positive_relative_budget_tolerance": (
            budget_relative_tolerance
        ),
        "zero_budget_maximum_absolute_reconstructed": (
            maximum_zero_budget_absolute_reconstructed
        ),
        "zero_budget_absolute_tolerance": zero_budget_absolute_tolerance,
    }


def _accept_trace_capacity_candidate(
    candidate: CondensateEquilibriumResult,
    report: Mapping[str, Any],
) -> CondensateEquilibriumResult:
    diagnostics = dict(candidate.diagnostics or {})
    diagnostics.setdefault(
        "pre_rainout_trace_capacity_acceptance_status",
        candidate.status,
    )
    diagnostics.setdefault(
        "pre_rainout_trace_capacity_acceptance_tier",
        candidate.acceptance_tier,
    )
    diagnostics["rainout_trace_capacity_acceptance"] = dict(report)
    diagnostics["acceptance_tier"] = "rainout_trace_capacity_accepted"
    return replace(
        candidate,
        status=CONVERGED,
        converged=True,
        acceptance_tier="rainout_trace_capacity_accepted",
        diagnostics=diagnostics,
    )


def _remove_depleted_element_species(
    *,
    setup: CondensateChemicalSetup,
    result: CondensateEquilibriumResult,
    conserved_mask: np.ndarray,
    inventory_target: np.ndarray,
) -> tuple[CondensateEquilibriumResult, Mapping[str, Any]]:
    """Report species excluded from exact-zero rainout propagation.

    The full solver uses positive log amounts, so a species containing an
    exactly-zero budget row can remain at its absolute numerical floor.  The
    raw public solver state is retained for auditability, while these species
    are excluded from budget certification and propagation.  Warm starts are
    therefore finite and the projection cannot violate result invariants.
    """

    depleted_rows = conserved_mask & (inventory_target == 0.0)
    gas_formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    condensate_formula = np.asarray(
        setup.formula_matrix_cond, dtype=np.float64
    )
    if not np.any(depleted_rows):
        return result, {
            "applied_to_public_state": False,
            "applied_to_propagation": False,
            "public_state": "raw_full_network_solver_state",
            "depleted_element_indices": (),
            "depleted_element_names": (),
            "removed_gas_species": (),
            "removed_condensate_species": (),
        }

    incompatible_gas = np.any(gas_formula[depleted_rows, :] != 0.0, axis=0)
    incompatible_condensates = np.any(
        condensate_formula[depleted_rows, :] != 0.0, axis=0
    )
    gas_n = np.asarray(jax.device_get(result.gas_n), dtype=np.float64)
    condensate_amounts = np.asarray(
        jax.device_get(result.condensate_amounts), dtype=np.float64
    )
    removed_gas_amounts = gas_n[incompatible_gas].copy()
    removed_condensate_amounts = condensate_amounts[
        incompatible_condensates
    ].copy()
    depleted_indices = tuple(np.flatnonzero(depleted_rows).tolist())
    return result, {
        "applied_to_public_state": False,
        "applied_to_propagation": bool(
            np.any(incompatible_gas) or np.any(incompatible_condensates)
        ),
        "public_state": "raw_full_network_solver_state",
        "depleted_element_indices": depleted_indices,
        "depleted_element_names": tuple(
            setup.elements[index] for index in depleted_indices
        ),
        "removed_gas_species": tuple(
            setup.gas_species[index]
            for index in np.flatnonzero(incompatible_gas)
        ),
        "removed_gas_amounts": tuple(float(x) for x in removed_gas_amounts),
        "removed_condensate_species": tuple(
            setup.condensate_species[index]
            for index in np.flatnonzero(incompatible_condensates)
        ),
        "removed_condensate_amounts": tuple(
            float(x) for x in removed_condensate_amounts
        ),
    }


def _floorless_budget_certification(
    *,
    setup: CondensateChemicalSetup,
    result: CondensateEquilibriumResult,
    conserved_mask: np.ndarray,
    inventory_target: np.ndarray,
    relative_tolerance: float,
) -> Mapping[str, Any]:
    """Certify every propagated budget row without an absolute floor."""

    gas_inventory = np.asarray(setup.formula_matrix, dtype=np.float64) @ np.asarray(
        jax.device_get(result.gas_n), dtype=np.float64
    )
    gas_formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    condensate_formula = np.asarray(
        setup.formula_matrix_cond, dtype=np.float64
    )
    gas_amounts = np.asarray(jax.device_get(result.gas_n), dtype=np.float64)
    condensate_amounts = np.asarray(
        jax.device_get(result.condensate_amounts), dtype=np.float64
    )
    raw_condensate_inventory = condensate_formula @ condensate_amounts
    depleted_rows = conserved_mask & (inventory_target == 0.0)
    incompatible_gas = (
        np.any(gas_formula[depleted_rows, :] != 0.0, axis=0)
        if np.any(depleted_rows)
        else np.zeros(gas_formula.shape[1], dtype=bool)
    )
    incompatible_condensates = (
        np.any(condensate_formula[depleted_rows, :] != 0.0, axis=0)
        if np.any(depleted_rows)
        else np.zeros(condensate_formula.shape[1], dtype=bool)
    )
    certified_gas_amounts = gas_amounts.copy()
    certified_gas_amounts[incompatible_gas] = 0.0
    certified_condensate_amounts = condensate_amounts.copy()
    certified_condensate_amounts[incompatible_condensates] = 0.0
    certified_gas_inventory = gas_formula @ certified_gas_amounts
    condensate_inventory = (
        condensate_formula @ certified_condensate_amounts
    )
    reconstructed = certified_gas_inventory + condensate_inventory
    finite = bool(
        np.all(np.isfinite(gas_inventory))
        and np.all(np.isfinite(raw_condensate_inventory))
        and np.all(np.isfinite(certified_gas_inventory))
        and np.all(np.isfinite(condensate_inventory))
        and np.all(np.isfinite(reconstructed))
    )
    positive_mask = conserved_mask & (inventory_target > 0.0)
    zero_mask = conserved_mask & (inventory_target == 0.0)
    relative_residual = np.zeros_like(inventory_target)
    if finite:
        relative_residual[positive_mask] = np.abs(
            reconstructed[positive_mask] - inventory_target[positive_mask]
        ) / inventory_target[positive_mask]
    else:
        relative_residual[positive_mask] = np.inf
    maximum_positive_relative_residual = float(
        np.max(relative_residual[positive_mask], initial=0.0)
    )
    maximum_zero_absolute_reconstructed = float(
        np.max(np.abs(reconstructed[zero_mask]), initial=0.0)
        if finite
        else math.inf
    )
    nonnegative = bool(
        finite
        and np.all(certified_gas_inventory[conserved_mask] >= 0.0)
        and np.all(condensate_inventory[conserved_mask] >= 0.0)
    )
    accepted = bool(
        finite
        and nonnegative
        and maximum_positive_relative_residual <= relative_tolerance
        and maximum_zero_absolute_reconstructed == 0.0
    )
    return {
        "schema": "exogibbs_rainout_floorless_budget_v1",
        "accepted": accepted,
        "finite": finite,
        "nonnegative": nonnegative,
        "relative_tolerance": float(relative_tolerance),
        "maximum_positive_relative_residual": (
            maximum_positive_relative_residual
        ),
        "maximum_zero_absolute_reconstructed": (
            maximum_zero_absolute_reconstructed
        ),
        "zero_budget_tolerance": 0.0,
        "zero_budget_handling": (
            "reduced_propagation"
            if np.any(depleted_rows)
            else "strict_full_network"
        ),
        "element_budget_target": tuple(float(x) for x in inventory_target),
        "raw_solver_gas_element_inventory": tuple(
            float(x) for x in gas_inventory
        ),
        "gas_element_inventory": tuple(
            float(x) for x in certified_gas_inventory
        ),
        "raw_solver_zero_budget_maximum_absolute_reconstructed": float(
            np.max(
                np.abs(
                    (
                        gas_inventory + raw_condensate_inventory
                    )[zero_mask]
                ),
                initial=0.0,
            )
            if finite
            else math.inf
        ),
        "raw_solver_condensate_element_inventory": tuple(
            float(x) for x in raw_condensate_inventory
        ),
        "rainout_propagation_condensate_element_inventory": tuple(
            float(x) for x in condensate_inventory
        ),
        "condensate_element_inventory": tuple(
            float(x) for x in condensate_inventory
        ),
        "condensate_element_inventory_alias_target": (
            "rainout_propagation_condensate_element_inventory"
        ),
        "element_budget_reconstructed": tuple(float(x) for x in reconstructed),
        "element_budget_residual": tuple(
            float(x) for x in (reconstructed - inventory_target)
        ),
        "positive_element_indices": tuple(
            np.flatnonzero(positive_mask).tolist()
        ),
        "zero_element_indices": tuple(np.flatnonzero(zero_mask).tolist()),
    }


def _conservation_rainout_inventory(
    *,
    setup: CondensateChemicalSetup,
    result: CondensateEquilibriumResult,
    conserved_mask: np.ndarray,
    normalization_mask: np.ndarray,
    inventory_target: np.ndarray,
    inventory_sum: float,
    roundoff_multiplier: float,
) -> Mapping[str, Any]:
    """Subtract condensates from the input budget and normalize the remainder."""

    gas_formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    condensate_formula = np.asarray(
        setup.formula_matrix_cond, dtype=np.float64
    )
    gas_amounts = np.asarray(jax.device_get(result.gas_n), dtype=np.float64)
    condensate_amounts = np.asarray(
        jax.device_get(result.condensate_amounts), dtype=np.float64
    )
    gas_inventory = gas_formula @ gas_amounts
    raw_condensate_inventory = condensate_formula @ condensate_amounts
    depleted_rows = conserved_mask & (inventory_target == 0.0)
    incompatible_gas = (
        np.any(gas_formula[depleted_rows, :] != 0.0, axis=0)
        if np.any(depleted_rows)
        else np.zeros(gas_formula.shape[1], dtype=bool)
    )
    incompatible_condensates = (
        np.any(condensate_formula[depleted_rows, :] != 0.0, axis=0)
        if np.any(depleted_rows)
        else np.zeros(condensate_formula.shape[1], dtype=bool)
    )
    propagation_gas_amounts = gas_amounts.copy()
    propagation_gas_amounts[incompatible_gas] = 0.0
    propagation_condensate_amounts = condensate_amounts.copy()
    propagation_condensate_amounts[incompatible_condensates] = 0.0
    propagation_gas_inventory = gas_formula @ propagation_gas_amounts
    condensate_inventory = (
        condensate_formula @ propagation_condensate_amounts
    )
    if not np.all(np.isfinite(gas_inventory)):
        raise RuntimeError("Rainout gas element inventory is not finite.")
    if not np.all(np.isfinite(raw_condensate_inventory)):
        raise RuntimeError("Rainout condensate element inventory is not finite.")
    conservation_inventory = inventory_target - condensate_inventory
    reconstruction_error = (
        gas_inventory + raw_condensate_inventory - inventory_target
    )
    propagation_crosscheck_error = (
        propagation_gas_inventory
        + condensate_inventory
        - inventory_target
    )
    roundoff_bound = (
        float(roundoff_multiplier)
        * np.finfo(np.float64).eps
        * np.maximum(np.abs(inventory_target), np.abs(condensate_inventory))
    )
    depletion_error_bound = (
        np.abs(propagation_crosscheck_error) + roundoff_bound
    )
    positive_target = normalization_mask & (inventory_target > 0.0)
    snap_mask = (
        positive_target
        & (condensate_inventory > 0.0)
        & (conservation_inventory <= depletion_error_bound)
    )
    material_negative = (
        positive_target
        & (conservation_inventory < -depletion_error_bound)
    )
    if np.any(material_negative):
        names = tuple(
            setup.elements[index]
            for index in np.flatnonzero(material_negative)
        )
        raise RuntimeError(
            "Rainout condensate removal exceeds the available element "
            f"inventory for {names!r}."
        )
    conservation_inventory = conservation_inventory.copy()
    conservation_inventory[snap_mask] = 0.0
    conservation_inventory[~positive_target] = 0.0
    if np.any(conservation_inventory[positive_target] < 0.0):
        raise RuntimeError("Rainout conservation inventory is negative.")

    no_condensate_removal = not bool(
        np.any(condensate_inventory[conserved_mask] > 0.0)
    )
    if no_condensate_removal:
        # This exact copy is important: a gas-only layer cannot change an
        # elemental rainout inventory merely because its gas solve has a
        # finite residual.
        next_inventory = inventory_target.copy()
        conservation_inventory = inventory_target.copy()
        conservation_inventory[~normalization_mask] = 0.0
        conservation_sum = inventory_sum
        normalization = 1.0
    else:
        conservation_sum = float(
            np.sum(conservation_inventory[normalization_mask])
        )
        if not math.isfinite(conservation_sum) or conservation_sum <= 0.0:
            raise RuntimeError(
                "Rainout cannot normalize an empty gas element inventory."
            )
        normalization = inventory_sum / conservation_sum
        next_inventory = np.zeros_like(conservation_inventory)
        surviving = normalization_mask & (conservation_inventory > 0.0)
        next_inventory[surviving] = (
            conservation_inventory[surviving] * normalization
        )
    return {
        "gas_inventory": gas_inventory,
        "propagation_gas_amounts": propagation_gas_amounts,
        "propagation_gas_inventory": propagation_gas_inventory,
        "raw_condensate_inventory": raw_condensate_inventory,
        "condensate_inventory": condensate_inventory,
        "conservation_inventory": conservation_inventory,
        "next_inventory": next_inventory,
        "conservation_sum": float(conservation_sum),
        "normalization": float(normalization),
        "crosscheck_residual": reconstruction_error,
        "propagation_crosscheck_residual": propagation_crosscheck_error,
        "ignored_gas_species_indices": tuple(
            np.flatnonzero(incompatible_gas).tolist()
        ),
        "ignored_condensate_species_indices": tuple(
            np.flatnonzero(incompatible_condensates).tolist()
        ),
        "snap_mask": snap_mask,
        "snap_amount": np.where(snap_mask, np.maximum(
            inventory_target - condensate_inventory, 0.0
        ), 0.0),
        "snap_error_bound": depletion_error_bound,
        "no_condensate_removal": no_condensate_removal,
    }


def _gas_warm_start_for_next_layer(
    gas_amounts: np.ndarray,
    *,
    inventory_sum: float,
    conservation_inventory_sum: float,
) -> CondensateEquilibriumInit:
    """Build a finite warm start from the exact-zero-compatible gas state."""

    normalization = inventory_sum / conservation_inventory_sum
    gas_n = np.asarray(gas_amounts, dtype=np.float64)
    scaled_gas_n = gas_n * normalization
    warm_floor = 1.0e-300 * max(1.0, float(np.sum(scaled_gas_n)))
    warm_gas_n = np.maximum(scaled_gas_n, warm_floor)
    return CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray(warm_gas_n, dtype=jnp.float64)),
        gas_ntot=jnp.asarray(np.sum(warm_gas_n), dtype=jnp.float64),
    )


def run_rainout_profile(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: Optional[CondensateEquilibriumInitializer],
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    options: CondensateEquilibriumOptions,
    return_diagnostics: bool,
) -> CondensateEquilibriumProfileResult:
    """Run dependent equilibrium layers from the bottom of a profile.

    Input and output arrays retain the package-wide top-to-bottom ordering.
    The final input entry is the bottom boundary and is solved first.  Each
    next inventory is obtained by subtracting accepted condensates from the
    current element budget and then normalizing the conserved gas remainder.
    The independently reconstructed gas inventory is retained as an audit.
    """

    (
        initial_inventory,
        conserved_mask,
        normalization_mask,
        inventory_sum,
    ) = (
        _validate_rainout_inventory(setup, b)
    )
    policy = fixed_support_v2_production_policy(
        options.fixed_support_v2_preset
    )
    n_layers = int(temperatures.shape[0])
    processing_indices = tuple(range(n_layers - 1, -1, -1))
    layer_results: list[CondensateEquilibriumResult | None] = [None] * n_layers
    target_by_layer = np.zeros((n_layers, len(setup.elements)), dtype=np.float64)
    gas_by_layer = np.zeros_like(target_by_layer)
    propagation_gas_by_layer = np.zeros_like(target_by_layer)
    suppressed_reintroduction_by_layer = np.zeros_like(target_by_layer)
    condensate_by_layer = np.zeros_like(target_by_layer)
    raw_condensate_by_layer = np.zeros_like(target_by_layer)
    conservation_by_layer = np.zeros_like(target_by_layer)
    crosscheck_by_layer = np.zeros_like(target_by_layer)
    depletion_snap_by_layer = np.zeros_like(target_by_layer)
    depletion_bound_by_layer = np.zeros_like(target_by_layer)
    out_by_layer = np.zeros_like(target_by_layer)
    abundance_scale_by_layer = np.ones(n_layers, dtype=np.float64)
    working_total_by_layer = np.ones(n_layers, dtype=np.float64)
    layer_records: list[Mapping[str, Any] | None] = [None] * n_layers

    current_inventory = initial_inventory
    previous_solution: CondensateEquilibriumInit | None = None
    previous_abundance_scale: float | None = None
    for layer_index in processing_indices:
        target_by_layer[layer_index] = current_inventory
        initial_guess = resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=jnp.asarray(current_inventory, dtype=jnp.float64),
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=previous_solution,
            ),
        )
        scales = _rainout_gauge_scales(
            current_inventory,
            normalization_mask,
            minimum_targets=policy.rainout_gauge_minimum_targets,
            maximum_total=policy.rainout_gauge_maximum_total,
            total_targets=policy.rainout_gauge_total_targets,
        )
        attempts: list[Mapping[str, Any]] = []
        accepted_profile: CondensateEquilibriumProfileResult | None = None
        accepted_scale: float | None = None
        accepted_result: CondensateEquilibriumResult | None = None
        accepted_projection: Mapping[str, Any] | None = None
        accepted_floorless_budget: Mapping[str, Any] | None = None
        trace_candidates: list[
            tuple[
                float,
                CondensateEquilibriumProfileResult,
                float,
                Mapping[str, Any],
            ]
        ] = []
        trace_retry_scales = 0
        for abundance_scale in scales:
            for initialization, attempt_guess in _initialization_attempts(
                initial_guess
            ):
                try:
                    scaled_profile = _lifecycle._run_head_v2_profile(
                        setup=setup,
                        temperatures=(
                            temperatures[layer_index : layer_index + 1]
                        ),
                        pressures=pressures[layer_index : layer_index + 1],
                        b=jnp.asarray(
                            current_inventory * abundance_scale,
                            dtype=jnp.float64,
                        ),
                        Pref=Pref,
                        explicit_inits=(
                            _scale_initial_guess(
                                attempt_guess,
                                abundance_scale,
                            ),
                        ),
                        initializer=None,
                        support_indices=support_indices,
                        support_amounts_init=(
                            None
                            if support_amounts_init is None
                            else tuple(
                                float(value) * abundance_scale
                                for value in support_amounts_init
                            )
                        ),
                        options=options,
                        return_diagnostics=return_diagnostics,
                    )
                except (FloatingPointError, OverflowError) as error:
                    attempts.append(
                        {
                            "abundance_scale": abundance_scale,
                            "initialization": initialization,
                            "converged": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                except ValueError as error:
                    if not _is_retryable_numerical_value_error(error):
                        raise
                    attempts.append(
                        {
                            "abundance_scale": abundance_scale,
                            "initialization": initialization,
                            "converged": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                candidate = scaled_profile.layers[0]
                attempt = {
                    "abundance_scale": abundance_scale,
                    "initialization": initialization,
                    "converged": bool(candidate.converged),
                    "status": candidate.status,
                    "acceptance_tier": candidate.acceptance_tier,
                }
                candidate_diagnostics = candidate.diagnostics or {}
                lifecycle = candidate_diagnostics.get(
                    "fixed_support_v2", {}
                )
                if isinstance(lifecycle, Mapping):
                    attempt["lifecycle_outcome"] = lifecycle.get("outcome")
                budget_gate = candidate_diagnostics.get(
                    "full_condensate_budget_residual_gate", {}
                )
                if isinstance(budget_gate, Mapping):
                    attempt["budget_gate_accepted"] = budget_gate.get(
                        "accepted"
                    )
                    attempt["budget_gate_max_abs_relative_residual"] = (
                        budget_gate.get("max_abs_relative_residual")
                    )
                attempts.append(attempt)
                if candidate.converged:
                    caller_candidate = _rescale_layer_result(
                        candidate, abundance_scale
                    )
                    caller_candidate, projection = (
                        _remove_depleted_element_species(
                            setup=setup,
                            result=caller_candidate,
                            conserved_mask=conserved_mask,
                            inventory_target=current_inventory,
                        )
                    )
                    floorless_budget = _floorless_budget_certification(
                        setup=setup,
                        result=caller_candidate,
                        conserved_mask=conserved_mask,
                        inventory_target=current_inventory,
                        relative_tolerance=(
                            options.full_condensate_budget_relative_tolerance
                        ),
                    )
                    attempt["rainout_floorless_budget_accepted"] = (
                        floorless_budget["accepted"]
                    )
                    attempt[
                        "rainout_floorless_maximum_positive_relative_residual"
                    ] = floorless_budget[
                        "maximum_positive_relative_residual"
                    ]
                    attempt[
                        "rainout_zero_budget_maximum_absolute_reconstructed"
                    ] = floorless_budget[
                        "maximum_zero_absolute_reconstructed"
                    ]
                    attempt["rainout_floorless_relative_tolerance"] = (
                        floorless_budget["relative_tolerance"]
                    )
                    attempt["rainout_floorless_element_budget_target"] = (
                        floorless_budget["element_budget_target"]
                    )
                    attempt["rainout_floorless_element_budget_residual"] = (
                        floorless_budget["element_budget_residual"]
                    )
                    if bool(floorless_budget["accepted"]):
                        accepted_profile = scaled_profile
                        accepted_scale = abundance_scale
                        accepted_result = caller_candidate
                        accepted_projection = projection
                        accepted_floorless_budget = floorless_budget
                        break
                trace_report = _trace_capacity_acceptance_report(
                    setup=setup,
                    inventory=current_inventory,
                    inventory_sum=inventory_sum,
                    candidate=candidate,
                    abundance_scale=abundance_scale,
                    policy=policy,
                )
                if trace_report is not None:
                    trace_candidates.append(
                        (
                            float(trace_report["condensate_stationarity"]),
                            scaled_profile,
                            abundance_scale,
                            trace_report,
                        )
                    )
            if accepted_profile is not None:
                break
            if trace_candidates:
                trace_retry_scales += 1
                if trace_retry_scales >= policy.rainout_trace_exact_retry_scales:
                    break
        if accepted_profile is None and trace_candidates:
            (
                _stationarity,
                trace_profile,
                accepted_scale,
                trace_report,
            ) = min(trace_candidates, key=lambda item: item[0])
            trace_result = _accept_trace_capacity_candidate(
                trace_profile.layers[0],
                trace_report,
            )
            accepted_profile = replace(
                trace_profile,
                layers=(trace_result,),
            )
            accepted_result = _rescale_layer_result(
                trace_result, accepted_scale
            )
            accepted_result, accepted_projection = (
                _remove_depleted_element_species(
                    setup=setup,
                    result=accepted_result,
                    conserved_mask=conserved_mask,
                    inventory_target=current_inventory,
                )
            )
            accepted_floorless_budget = _floorless_budget_certification(
                setup=setup,
                result=accepted_result,
                conserved_mask=conserved_mask,
                inventory_target=current_inventory,
                relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
            )
            if not bool(accepted_floorless_budget["accepted"]):
                accepted_profile = None
                accepted_result = None
            attempts.append(
                {
                    "abundance_scale": accepted_scale,
                    "initialization": "trace_capacity_acceptance",
                    "converged": True,
                    "status": trace_result.status,
                    "acceptance_tier": trace_result.acceptance_tier,
                    "trace_capacity_report": dict(trace_report),
                }
            )
        if (
            accepted_profile is None
            or accepted_scale is None
            or accepted_result is None
            or accepted_projection is None
            or accepted_floorless_budget is None
        ):
            raise RuntimeError(
                "Rainout stopped after a non-converged layer at original "
                f"profile index {layer_index}; later upper layers were not "
                "evaluated. Element inventory: "
                f"{current_inventory.tolist()!r}. Attempts: {attempts!r}"
            )

        result = accepted_result
        result = _with_rainout_layer_diagnostics(
            result,
            layer_index=layer_index,
            abundance_scale=accepted_scale,
            preferred_abundance_scale=scales[0],
            previous_abundance_scale=previous_abundance_scale,
            working_inventory_total=(
                accepted_scale
                * float(np.sum(current_inventory[normalization_mask]))
            ),
            attempts=attempts,
            depleted_projection=accepted_projection,
            floorless_budget=accepted_floorless_budget,
        )
        propagation = _conservation_rainout_inventory(
            setup=setup,
            result=result,
            conserved_mask=conserved_mask,
            normalization_mask=normalization_mask,
            inventory_target=current_inventory,
            inventory_sum=inventory_sum,
            roundoff_multiplier=(
                policy.rainout_depletion_roundoff_multiplier
            ),
        )
        result_diagnostics = dict(result.diagnostics or {})
        rainout_diagnostics = dict(result_diagnostics["rainout"])
        rainout_diagnostics["propagation"] = {
            "schema": "exogibbs_rainout_conservation_propagation_v1",
            "source": (
                "b_current_minus_"
                "rainout_propagation_condensate_element_inventory"
            ),
            "gas_inventory_role": "independent_crosscheck_only",
            "raw_condensate_element_inventory": tuple(
                float(x)
                for x in propagation["raw_condensate_inventory"]
            ),
            "rainout_propagation_condensate_element_inventory": tuple(
                float(x) for x in propagation["condensate_inventory"]
            ),
            "condensate_element_inventory_alias_target": (
                "rainout_propagation_condensate_element_inventory"
            ),
            "suppressed_element_reintroduction": tuple(
                float(x)
                for x in (
                    propagation["gas_inventory"]
                    - propagation["propagation_gas_inventory"]
                )
            ),
            "no_condensate_removal": propagation[
                "no_condensate_removal"
            ],
            "normalization": propagation["normalization"],
            "conservation_inventory_sum": propagation[
                "conservation_sum"
            ],
            "depletion_snap_element_indices": tuple(
                np.flatnonzero(propagation["snap_mask"]).tolist()
            ),
            "depletion_snap_element_names": tuple(
                setup.elements[index]
                for index in np.flatnonzero(propagation["snap_mask"])
            ),
            "depletion_snap_amount": tuple(
                float(x) for x in propagation["snap_amount"]
            ),
            "depletion_snap_error_bound": tuple(
                float(x) for x in propagation["snap_error_bound"]
            ),
            "depletion_snap_error_source": (
                "reduced_propagation_crosscheck_residual_plus_roundoff"
            ),
            "gas_conservation_crosscheck_residual": tuple(
                float(x) for x in propagation["crosscheck_residual"]
            ),
            "reduced_propagation_crosscheck_residual": tuple(
                float(x)
                for x in propagation["propagation_crosscheck_residual"]
            ),
            "ignored_gas_species_indices": propagation[
                "ignored_gas_species_indices"
            ],
            "ignored_condensate_species_indices": propagation[
                "ignored_condensate_species_indices"
            ],
        }
        result_diagnostics["rainout"] = rainout_diagnostics
        result = replace(result, diagnostics=result_diagnostics)
        gas_by_layer[layer_index] = propagation["gas_inventory"]
        propagation_gas_by_layer[layer_index] = propagation[
            "propagation_gas_inventory"
        ]
        suppressed_reintroduction_by_layer[layer_index] = (
            propagation["gas_inventory"]
            - propagation["propagation_gas_inventory"]
        )
        condensate_by_layer[layer_index] = propagation[
            "condensate_inventory"
        ]
        raw_condensate_by_layer[layer_index] = propagation[
            "raw_condensate_inventory"
        ]
        conservation_by_layer[layer_index] = propagation[
            "conservation_inventory"
        ]
        crosscheck_by_layer[layer_index] = propagation[
            "crosscheck_residual"
        ]
        depletion_snap_by_layer[layer_index] = propagation["snap_amount"]
        depletion_bound_by_layer[layer_index] = propagation[
            "snap_error_bound"
        ]
        out_by_layer[layer_index] = propagation["next_inventory"]
        abundance_scale_by_layer[layer_index] = accepted_scale
        working_total_by_layer[layer_index] = (
            accepted_scale
            * float(np.sum(current_inventory[normalization_mask]))
        )
        layer_results[layer_index] = result
        layer_records[layer_index] = {
            "layer_index": layer_index,
            "abundance_scale": accepted_scale,
            "preferred_abundance_scale": scales[0],
            "previous_abundance_scale": previous_abundance_scale,
            "working_inventory_total": working_total_by_layer[layer_index],
            "attempts": tuple(attempts),
            "gas_inventory_sum": propagation["conservation_sum"],
            "depletion_snap_element_indices": tuple(
                np.flatnonzero(propagation["snap_mask"]).tolist()
            ),
        }
        current_inventory = propagation["next_inventory"]
        previous_solution = _gas_warm_start_for_next_layer(
            propagation["propagation_gas_amounts"],
            inventory_sum=inventory_sum,
            conservation_inventory_sum=propagation["conservation_sum"],
        )
        previous_abundance_scale = accepted_scale

    if any(result is None for result in layer_results):
        raise RuntimeError("Rainout profile ended with an unevaluated layer.")
    completed_layers = tuple(
        result for result in layer_results if result is not None
    )
    gas_ln_n = jnp.stack([result.gas_ln_n for result in completed_layers])
    gas_n = jnp.stack([result.gas_n for result in completed_layers])
    gas_x = jnp.stack([result.gas_x for result in completed_layers])
    gas_ntot = jnp.stack([result.gas_ntot for result in completed_layers])
    condensate_amounts = jnp.stack(
        [result.condensate_amounts for result in completed_layers]
    )
    target_array = jnp.asarray(target_by_layer, dtype=jnp.float64)
    gas_inventory_array = jnp.asarray(gas_by_layer, dtype=jnp.float64)
    propagation_gas_inventory_array = jnp.asarray(
        propagation_gas_by_layer, dtype=jnp.float64
    )
    suppressed_reintroduction_array = jnp.asarray(
        suppressed_reintroduction_by_layer, dtype=jnp.float64
    )
    condensate_inventory_array = jnp.asarray(
        condensate_by_layer, dtype=jnp.float64
    )
    raw_condensate_inventory_array = jnp.asarray(
        raw_condensate_by_layer, dtype=jnp.float64
    )
    conservation_inventory_array = jnp.asarray(
        conservation_by_layer, dtype=jnp.float64
    )
    crosscheck_array = jnp.asarray(crosscheck_by_layer, dtype=jnp.float64)
    depletion_snap_array = jnp.asarray(
        depletion_snap_by_layer, dtype=jnp.float64
    )
    depletion_bound_array = jnp.asarray(
        depletion_bound_by_layer, dtype=jnp.float64
    )
    out_array = jnp.asarray(out_by_layer, dtype=jnp.float64)
    abundance_scale_array = jnp.asarray(
        abundance_scale_by_layer,
        dtype=jnp.float64,
    )
    working_total_array = jnp.asarray(
        working_total_by_layer, dtype=jnp.float64
    )
    profile_diagnostics = None
    if return_diagnostics:
        profile_diagnostics = {
            "profile_schema": "exogibbs_condensate_rainout_profile_v1",
            "route": options.route,
            "preset": policy.name,
            "method": "scan_hot_from_bottom",
            "rainout": True,
            "input_order": "top_to_bottom",
            "scan_direction": "bottom_to_top",
            "processing_indices": processing_indices,
            "layer_count": n_layers,
            "layers": tuple(layer_records),
            "batched_array_aliases": {
                "condensate_element_inventory": (
                    "rainout_propagation_condensate_element_inventory"
                ),
            },
        }
    return CondensateEquilibriumProfileResult(
        layers=completed_layers,
        method="scan_hot_from_bottom",
        diagnostics=profile_diagnostics,
        batched_arrays={
            "gas_ln_n": gas_ln_n,
            "gas_n": gas_n,
            "gas_x": gas_x,
            "gas_ntot": gas_ntot,
            "condensate_amounts": condensate_amounts,
            "element_inventory_target": target_array,
            "gas_element_inventory": gas_inventory_array,
            "rainout_propagation_gas_element_inventory": (
                propagation_gas_inventory_array
            ),
            "rainout_suppressed_element_reintroduction": (
                suppressed_reintroduction_array
            ),
            "raw_condensate_element_inventory": (
                raw_condensate_inventory_array
            ),
            "rainout_propagation_condensate_element_inventory": (
                condensate_inventory_array
            ),
            # Backward-compatible alias for the original rainout field name.
            "condensate_element_inventory": condensate_inventory_array,
            "rainout_conservation_element_inventory": (
                conservation_inventory_array
            ),
            "rainout_gas_conservation_crosscheck_residual": (
                crosscheck_array
            ),
            "rainout_depletion_snap_amount": depletion_snap_array,
            "rainout_depletion_error_bound": depletion_bound_array,
            "rainout_element_inventory_out": out_array,
            "rainout_abundance_scale": abundance_scale_array,
            "rainout_working_inventory_total": working_total_array,
        },
        rainout=True,
        element_inventory_target=target_array,
        gas_element_inventory=gas_inventory_array,
        rainout_element_inventory_out=out_array,
        rainout_abundance_scale=abundance_scale_array,
    )


__all__ = ("run_rainout_profile",)
