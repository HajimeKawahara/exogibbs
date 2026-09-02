"""Dependent Python-level profile schedulers for condensate equilibrium."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate.initialization import (
    regauge_gas_only_warm_start,
    resolve_condensate_initial_guess,
)
from exogibbs.equilibrium.condensate.inventory_bridge import (
    interpolate_element_inventory,
    validate_equilibrium_point,
    validate_inventory_bridge_config,
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
    CondensateEquilibriumPoint,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
)
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction


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
    maximum_total: float,
) -> tuple[float, ...]:
    """Return one overflow-safe transport scale without normal upscaling.

    The lifecycle owns conversion to the canonical amount gauge.  Rainout
    therefore preserves the caller gauge unless its total exceeds the finite
    transport cap, in which case it applies one uniform downscale.
    """

    active = inventory[normalization_mask]
    total = float(np.sum(active))
    if not math.isfinite(maximum_total) or maximum_total <= 0.0:
        raise ValueError("maximum_total must be finite and positive.")
    if total <= maximum_total:
        return (1.0,)
    maximum_scale = maximum_total / total
    safe_maximum_scale = float(np.nextafter(maximum_scale, 0.0))
    if (
        not math.isfinite(safe_maximum_scale)
        or safe_maximum_scale <= 0.0
        or safe_maximum_scale * total > maximum_total
    ):
        raise ValueError("Unable to construct a bounded rainout gauge scale.")
    return (safe_maximum_scale,)


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
            else _lifecycle._transform_linear_amount_gauge_on_host(
                initial_guess.gas_ntot,
                scale,
                to_canonical=False,
            )
        ),
        condensate_amounts=(
            None
            if initial_guess.condensate_amounts is None
            else _lifecycle._transform_linear_amount_gauge_on_host(
                initial_guess.condensate_amounts,
                scale,
                to_canonical=False,
            )
        ),
        support_amounts=(
            None
            if initial_guess.support_amounts is None
            else tuple(float(value) * scale for value in initial_guess.support_amounts)
        ),
        barrier_epsilon=(
            None
            if initial_guess.barrier_epsilon is None
            else jnp.asarray(
                initial_guess.barrier_epsilon,
                dtype=jnp.float64,
            )
            + log_scale
        ),
        inventory_bridge_origin=(
            None
            if initial_guess.inventory_bridge_origin is None
            else replace(
                initial_guess.inventory_bridge_origin,
                element_inventory=(
                    _lifecycle._transform_linear_amount_gauge_on_host(
                        initial_guess.inventory_bridge_origin.element_inventory,
                        scale,
                        to_canonical=False,
                    )
                ),
            )
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
    return replace(
        result,
        gas_ln_n=jnp.asarray(result.gas_ln_n) - math.log(scale),
        gas_n=_lifecycle._transform_linear_amount_gauge_on_host(
            result.gas_n,
            scale,
            to_canonical=True,
        ),
        gas_ntot=_lifecycle._transform_linear_amount_gauge_on_host(
            result.gas_ntot,
            scale,
            to_canonical=True,
        ),
        condensate_amounts=(
            _lifecycle._transform_linear_amount_gauge_on_host(
                result.condensate_amounts,
                scale,
                to_canonical=True,
            )
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
        "schema": "exogibbs_condensate_rainout_layer_v2",
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
        "solver_diagnostics_gauge": "canonical_internal_amount_gauge",
        "budget_audit_gauge": (
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


@dataclass(frozen=True)
class _RainoutCandidateAssessment:
    attempt: Mapping[str, Any]
    accepted_result: CondensateEquilibriumResult | None
    depleted_projection: Mapping[str, Any] | None
    floorless_budget: Mapping[str, Any] | None


def _certify_rainout_candidate(
    *,
    setup: CondensateChemicalSetup,
    candidate: CondensateEquilibriumResult,
    initialization: str,
    abundance_scale: float,
    conserved_mask: np.ndarray,
    inventory_target: np.ndarray,
    relative_tolerance: float,
) -> _RainoutCandidateAssessment:
    """Apply the shared caller-gauge rainout gate to one solver candidate."""

    attempt: dict[str, Any] = {
        "abundance_scale": abundance_scale,
        "initialization": initialization,
        "converged": bool(candidate.converged),
        "status": candidate.status,
        "acceptance_tier": candidate.acceptance_tier,
        "support_indices": tuple(
            int(index)
            for index in np.asarray(
                jax.device_get(candidate.condensate_support_indices),
                dtype=np.int64,
            ).tolist()
        ),
        "support_names": tuple(candidate.condensate_support_names),
    }
    candidate_diagnostics = candidate.diagnostics or {}
    lifecycle = candidate_diagnostics.get("fixed_support_v2", {})
    if isinstance(lifecycle, Mapping):
        attempt["lifecycle_outcome"] = lifecycle.get("outcome")
    budget_gate = candidate_diagnostics.get(
        "full_condensate_budget_residual_gate", {}
    )
    if isinstance(budget_gate, Mapping):
        attempt["budget_gate_accepted"] = budget_gate.get("accepted")
        attempt["budget_gate_max_abs_relative_residual"] = budget_gate.get(
            "max_abs_relative_residual"
        )
    if not candidate.converged:
        return _RainoutCandidateAssessment(attempt, None, None, None)

    caller_candidate = _rescale_layer_result(candidate, abundance_scale)
    caller_candidate, projection = _remove_depleted_element_species(
        setup=setup,
        result=caller_candidate,
        conserved_mask=conserved_mask,
        inventory_target=inventory_target,
    )
    floorless_budget = _floorless_budget_certification(
        setup=setup,
        result=caller_candidate,
        conserved_mask=conserved_mask,
        inventory_target=inventory_target,
        relative_tolerance=relative_tolerance,
    )
    attempt["rainout_floorless_budget_accepted"] = floorless_budget[
        "accepted"
    ]
    attempt[
        "rainout_floorless_maximum_positive_relative_residual"
    ] = floorless_budget["maximum_positive_relative_residual"]
    attempt[
        "rainout_zero_budget_maximum_absolute_reconstructed"
    ] = floorless_budget["maximum_zero_absolute_reconstructed"]
    attempt["rainout_floorless_relative_tolerance"] = floorless_budget[
        "relative_tolerance"
    ]
    attempt["rainout_floorless_element_budget_target"] = floorless_budget[
        "element_budget_target"
    ]
    attempt["rainout_floorless_element_budget_residual"] = floorless_budget[
        "element_budget_residual"
    ]
    if not bool(floorless_budget["accepted"]):
        return _RainoutCandidateAssessment(attempt, None, None, None)
    return _RainoutCandidateAssessment(
        attempt,
        caller_candidate,
        projection,
        floorless_budget,
    )


def _run_rainout_solver_attempt(
    *,
    setup: CondensateChemicalSetup,
    temperature: float,
    pressure: float,
    inventory: np.ndarray,
    Pref: float,
    initial_guess: CondensateEquilibriumInit,
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    options: CondensateEquilibriumOptions,
    return_diagnostics: bool,
    lnphi_func: LogFugacityCoefficientFunction | None,
) -> CondensateEquilibriumProfileResult:
    """Run the existing lifecycle for exactly one rainout trial."""

    return _lifecycle._run_head_v2_profile(
        setup=setup,
        temperatures=np.asarray([temperature], dtype=np.float64),
        pressures=np.asarray([pressure], dtype=np.float64),
        b=jnp.asarray(inventory, dtype=jnp.float64),
        Pref=Pref,
        explicit_inits=(initial_guess,),
        initializer=None,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=options,
        return_diagnostics=return_diagnostics,
        lnphi_func=lnphi_func,
    )


def _bridge_trial_error_report(
    *,
    fraction: float,
    inventory: np.ndarray,
    error: Exception,
    stage: str = "lifecycle",
) -> Mapping[str, Any]:
    return {
        "fraction": fraction,
        "element_inventory": tuple(float(value) for value in inventory),
        "converged": False,
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
    }


def _run_inventory_bridge(
    *,
    setup: CondensateChemicalSetup,
    temperature: float,
    pressure: float,
    target_inventory: np.ndarray,
    initial_guess: CondensateEquilibriumInit,
    Pref: float,
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    options: CondensateEquilibriumOptions,
    return_diagnostics: bool,
    lnphi_func: LogFugacityCoefficientFunction | None,
    conserved_mask: np.ndarray,
    policy: FixedSupportV2ProductionPolicy,
) -> tuple[CondensateEquilibriumProfileResult | None, Mapping[str, Any]]:
    """Try bounded inventory anchors at the exact target thermodynamics."""

    origin = initial_guess.inventory_bridge_origin
    report: dict[str, Any] = {
        "schema": "exogibbs_condensate_inventory_bridge_v1",
        "path": "target_thermodynamics_inventory_bridge",
        "inventory_gauge": "rainout_lifecycle_caller_gauge",
        "converged": False,
        "maximum_lifecycle_solves": (
            policy.rainout_inventory_bridge.max_lifecycle_solves
        ),
        "trials": (),
    }
    if origin is None:
        report["termination_reason"] = "missing_origin"
        return None, report
    if initial_guess.gas_ln_n is None:
        report["termination_reason"] = "missing_gas_seed"
        return None, report
    try:
        validate_inventory_bridge_config(policy.rainout_inventory_bridge)
        origin_inventory = validate_equilibrium_point(
            origin,
            expected_inventory_shape=target_inventory.shape,
        )
    except (TypeError, ValueError) as error:
        report["termination_reason"] = "invalid_origin_or_policy"
        report["error"] = f"{type(error).__name__}: {error}"
        return None, report
    report["origin"] = {
        "temperature": float(origin.temperature),
        "pressure": float(origin.pressure),
        "element_inventory": tuple(float(value) for value in origin_inventory),
    }
    report["target"] = {
        "temperature": temperature,
        "pressure": pressure,
        "element_inventory": tuple(float(value) for value in target_inventory),
    }
    if np.array_equal(origin_inventory, target_inventory):
        report["termination_reason"] = "identical_inventories"
        return None, report

    trials: list[Mapping[str, Any]] = []
    lifecycle_calls = 0
    for fraction in policy.rainout_inventory_bridge.anchor_fractions:
        if lifecycle_calls >= policy.rainout_inventory_bridge.max_lifecycle_solves:
            break
        bridge_inventory = interpolate_element_inventory(
            origin_inventory,
            target_inventory,
            fraction,
        )
        try:
            bridge_init = regauge_gas_only_warm_start(
                setup,
                initial_guess.gas_ln_n,
                bridge_inventory,
            )
        except (TypeError, ValueError) as error:
            trials.append(
                _bridge_trial_error_report(
                    fraction=fraction,
                    inventory=bridge_inventory,
                    error=error,
                    stage="seed_preparation",
                )
            )
            continue
        try:
            bridge_profile = _run_rainout_solver_attempt(
                setup=setup,
                temperature=temperature,
                pressure=pressure,
                inventory=bridge_inventory,
                Pref=Pref,
                initial_guess=bridge_init,
                support_indices=support_indices,
                support_amounts_init=support_amounts_init,
                options=options,
                return_diagnostics=return_diagnostics,
                lnphi_func=lnphi_func,
            )
        except (FloatingPointError, OverflowError) as error:
            lifecycle_calls += 1
            trials.append(
                _bridge_trial_error_report(
                    fraction=fraction,
                    inventory=bridge_inventory,
                    error=error,
                )
            )
            continue
        except ValueError as error:
            if not _is_retryable_numerical_value_error(error):
                raise
            lifecycle_calls += 1
            trials.append(
                _bridge_trial_error_report(
                    fraction=fraction,
                    inventory=bridge_inventory,
                    error=error,
                )
            )
            continue
        lifecycle_calls += 1
        bridge_assessment = _certify_rainout_candidate(
            setup=setup,
            candidate=bridge_profile.layers[0],
            initialization="inventory_bridge_anchor",
            abundance_scale=1.0,
            conserved_mask=conserved_mask,
            inventory_target=bridge_inventory,
            relative_tolerance=(
                options.full_condensate_budget_relative_tolerance
            ),
        )
        bridge_trial = dict(bridge_assessment.attempt)
        bridge_trial["fraction"] = fraction
        bridge_trial["element_inventory"] = tuple(
            float(value) for value in bridge_inventory
        )
        bridge_trial["accepted_as_gas_seed"] = bool(
            bridge_assessment.accepted_result is not None
        )
        trials.append(bridge_trial)
        if bridge_assessment.accepted_result is None:
            continue
        if lifecycle_calls >= policy.rainout_inventory_bridge.max_lifecycle_solves:
            break

        try:
            target_init = regauge_gas_only_warm_start(
                setup,
                bridge_assessment.accepted_result.gas_ln_n,
                target_inventory,
            )
        except (TypeError, ValueError) as error:
            trials.append(
                _bridge_trial_error_report(
                    fraction=1.0,
                    inventory=target_inventory,
                    error=error,
                    stage="seed_preparation",
                )
            )
            continue
        try:
            target_profile = _run_rainout_solver_attempt(
                setup=setup,
                temperature=temperature,
                pressure=pressure,
                inventory=target_inventory,
                Pref=Pref,
                initial_guess=target_init,
                support_indices=support_indices,
                support_amounts_init=support_amounts_init,
                options=options,
                return_diagnostics=return_diagnostics,
                lnphi_func=lnphi_func,
            )
        except (FloatingPointError, OverflowError) as error:
            lifecycle_calls += 1
            trials.append(
                _bridge_trial_error_report(
                    fraction=1.0,
                    inventory=target_inventory,
                    error=error,
                )
            )
            continue
        except ValueError as error:
            if not _is_retryable_numerical_value_error(error):
                raise
            lifecycle_calls += 1
            trials.append(
                _bridge_trial_error_report(
                    fraction=1.0,
                    inventory=target_inventory,
                    error=error,
                )
            )
            continue
        lifecycle_calls += 1
        target_assessment = _certify_rainout_candidate(
            setup=setup,
            candidate=target_profile.layers[0],
            initialization="inventory_bridge_target_retry",
            abundance_scale=1.0,
            conserved_mask=conserved_mask,
            inventory_target=target_inventory,
            relative_tolerance=(
                options.full_condensate_budget_relative_tolerance
            ),
        )
        target_trial = dict(target_assessment.attempt)
        target_trial["fraction"] = 1.0
        target_trial["element_inventory"] = tuple(
            float(value) for value in target_inventory
        )
        target_trial["accepted_as_gas_seed"] = False
        trials.append(target_trial)
        if target_assessment.accepted_result is not None:
            report["converged"] = True
            report["termination_reason"] = "target_accepted"
            report["lifecycle_solves"] = lifecycle_calls
            report["trials"] = tuple(trials)
            return target_profile, report

    report["lifecycle_solves"] = lifecycle_calls
    report["trials"] = tuple(trials)
    if any(float(trial.get("fraction", -1.0)) == 1.0 for trial in trials):
        report["termination_reason"] = "target_retry_rejected"
    elif lifecycle_calls >= policy.rainout_inventory_bridge.max_lifecycle_solves:
        report["termination_reason"] = "maximum_lifecycle_solves"
    else:
        report["termination_reason"] = "anchor_rejected"
    return None, report


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
    lnphi_func: LogFugacityCoefficientFunction | None = None,
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
            maximum_total=policy.rainout_gauge_maximum_total,
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
        for abundance_scale in scales:
            working_inventory = (
                current_inventory
                if abundance_scale == 1.0
                else current_inventory * abundance_scale
            )
            working_support_amounts_init = (
                support_amounts_init
                if support_amounts_init is None or abundance_scale == 1.0
                else tuple(
                    float(value) * abundance_scale
                    for value in support_amounts_init
                )
            )
            for initialization, attempt_guess in _initialization_attempts(
                initial_guess
            ):
                scaled_attempt_guess = _scale_initial_guess(
                    attempt_guess,
                    abundance_scale,
                )
                scaled_profile = None
                try:
                    scaled_profile = _run_rainout_solver_attempt(
                        setup=setup,
                        temperature=float(temperatures[layer_index]),
                        pressure=float(pressures[layer_index]),
                        inventory=working_inventory,
                        Pref=Pref,
                        initial_guess=scaled_attempt_guess,
                        support_indices=support_indices,
                        support_amounts_init=working_support_amounts_init,
                        options=options,
                        return_diagnostics=return_diagnostics,
                        lnphi_func=lnphi_func,
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
                if scaled_profile is not None:
                    candidate = scaled_profile.layers[0]
                    assessment = _certify_rainout_candidate(
                        setup=setup,
                        candidate=candidate,
                        initialization=initialization,
                        abundance_scale=abundance_scale,
                        conserved_mask=conserved_mask,
                        inventory_target=current_inventory,
                        relative_tolerance=(
                            options.full_condensate_budget_relative_tolerance
                        ),
                    )
                    attempts.append(assessment.attempt)
                    if assessment.accepted_result is not None:
                        accepted_profile = scaled_profile
                        accepted_scale = abundance_scale
                        accepted_result = assessment.accepted_result
                        accepted_projection = assessment.depleted_projection
                        accepted_floorless_budget = assessment.floorless_budget
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
                                float(
                                    trace_report["condensate_stationarity"]
                                ),
                                scaled_profile,
                                abundance_scale,
                                trace_report,
                            )
                        )

                if (
                    initialization == "resolved"
                    and scaled_attempt_guess.inventory_bridge_origin is not None
                ):
                    bridge_profile, bridge_report = _run_inventory_bridge(
                        setup=setup,
                        temperature=float(temperatures[layer_index]),
                        pressure=float(pressures[layer_index]),
                        target_inventory=np.asarray(
                            working_inventory,
                            dtype=np.float64,
                        ),
                        initial_guess=scaled_attempt_guess,
                        Pref=Pref,
                        support_indices=support_indices,
                        support_amounts_init=working_support_amounts_init,
                        options=options,
                        return_diagnostics=return_diagnostics,
                        lnphi_func=lnphi_func,
                        conserved_mask=conserved_mask,
                        policy=policy,
                    )
                    bridge_attempt: dict[str, Any] = {
                        "abundance_scale": abundance_scale,
                        "initialization": "inventory_bridge",
                        "converged": False,
                        "inventory_bridge": dict(bridge_report),
                    }
                    if bridge_profile is not None:
                        bridge_assessment = _certify_rainout_candidate(
                            setup=setup,
                            candidate=bridge_profile.layers[0],
                            initialization="inventory_bridge",
                            abundance_scale=abundance_scale,
                            conserved_mask=conserved_mask,
                            inventory_target=current_inventory,
                            relative_tolerance=(
                                options.full_condensate_budget_relative_tolerance
                            ),
                        )
                        bridge_attempt.update(bridge_assessment.attempt)
                        bridge_attempt["initialization"] = "inventory_bridge"
                        bridge_attempt["inventory_bridge"] = dict(
                            bridge_report
                        )
                        bridge_attempt["converged"] = bool(
                            bridge_assessment.accepted_result is not None
                        )
                        if bridge_assessment.accepted_result is not None:
                            accepted_profile = bridge_profile
                            accepted_result = bridge_assessment.accepted_result
                            accepted_projection = (
                                bridge_assessment.depleted_projection
                            )
                            accepted_floorless_budget = (
                                bridge_assessment.floorless_budget
                            )
                    if (
                        int(bridge_report.get("lifecycle_solves", 0)) > 0
                        or bool(bridge_report.get("trials", ()))
                    ):
                        attempts.append(bridge_attempt)
                    if accepted_profile is not None:
                        accepted_scale = abundance_scale
                        break
            if accepted_profile is not None:
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
        previous_solution = replace(
            regauge_gas_only_warm_start(
                setup,
                result.gas_ln_n,
                current_inventory,
            ),
            inventory_bridge_origin=CondensateEquilibriumPoint(
                temperature=float(temperatures[layer_index]),
                pressure=float(pressures[layer_index]),
                element_inventory=jnp.asarray(
                    target_by_layer[layer_index].copy(),
                    dtype=jnp.float64,
                ),
            ),
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
