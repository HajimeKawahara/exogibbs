"""Post-solve acceptance and explicit numerical transformations."""

from __future__ import annotations

from typing import Any, Mapping
import weakref

import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    validate_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.types import (
    CONVERGED,
    DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
    NOT_CONVERGED,
    AcceptedCondensateState,
    Array,
)


_SETUP_NUMPY_FORMULA_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[Any], np.ndarray, np.ndarray],
] = {}


def setup_formula_matrices_numpy(
    setup: CondensateChemicalSetup,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached NumPy matrices for host-side acceptance work."""

    key = id(setup)
    cached = _SETUP_NUMPY_FORMULA_CACHE.get(key)
    if cached is not None:
        setup_ref, formula_matrix, formula_matrix_cond = cached
        if setup_ref() is setup:
            return formula_matrix, formula_matrix_cond
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    formula_matrix_cond = np.asarray(
        setup.formula_matrix_cond,
        dtype=np.float64,
    )

    def _drop_cache(
        _ref: weakref.ReferenceType[Any],
        *,
        cache_key: int = key,
    ) -> None:
        _SETUP_NUMPY_FORMULA_CACHE.pop(cache_key, None)

    _SETUP_NUMPY_FORMULA_CACHE[key] = (
        weakref.ref(setup, _drop_cache),
        formula_matrix,
        formula_matrix_cond,
    )
    return formula_matrix, formula_matrix_cond


def independent_kkt_passed(
    kkt: Mapping[str, float],
    *,
    stationarity_tolerance: float,
    budget_tolerance: float,
    complementarity_tolerance: float,
    total_density_tolerance: float,
) -> bool:
    """Apply the independent final KKT gate used by v2 validation."""

    return bool(
        kkt["gas_stationarity"] <= stationarity_tolerance
        and kkt["condensate_stationarity"] <= stationarity_tolerance
        and kkt["budget_scaled"] <= budget_tolerance
        and kkt["complementarity"] <= complementarity_tolerance
        and kkt["total_density_scaled"] <= total_density_tolerance
    )


def full_condensate_element_budget_residual_report(
    *,
    setup: CondensateChemicalSetup,
    gas_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
    relative_floor: float = DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
) -> dict[str, Any]:
    """Report the full gas-plus-condensate element-budget residual."""

    target = np.asarray(element_inventory_target, dtype=np.float64)
    if target.ndim != 1 or target.shape[0] != len(setup.elements):
        raise ValueError(
            "element_inventory_target must have one value per element."
        )
    gas_amounts = np.asarray(gas_n, dtype=np.float64)
    cond_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if (
        gas_amounts.ndim != 1
        or gas_amounts.shape[0] != len(setup.gas_species)
    ):
        raise ValueError("gas_n must have one value per gas species.")
    if (
        cond_amounts.ndim != 1
        or cond_amounts.shape[0] != len(setup.condensate_species)
    ):
        raise ValueError(
            "condensate_amounts must have one value per condensate species."
        )
    formula_matrix, formula_matrix_cond = setup_formula_matrices_numpy(setup)
    reconstructed = (
        formula_matrix @ gas_amounts
        + formula_matrix_cond @ cond_amounts
    )
    residual = reconstructed - target
    floor = float(relative_floor)
    denominator = np.maximum(np.abs(target), max(floor, 1.0e-300))
    signed_relative = residual / denominator
    absolute_relative = np.abs(signed_relative)
    gate_mask = np.asarray(
        tuple(
            str(element) not in {"e-", "electron"}
            for element in setup.elements
        ),
        dtype=bool,
    )
    gated_absolute_relative = np.where(gate_mask, absolute_relative, 0.0)
    finite = bool(
        np.all(np.isfinite(np.where(gate_mask, absolute_relative, 0.0)))
    )
    sanitized = np.where(
        np.isfinite(gated_absolute_relative),
        gated_absolute_relative,
        np.inf,
    )
    max_index = int(np.argmax(sanitized))
    max_abs_relative = float(gated_absolute_relative[max_index])
    tolerance = float(relative_tolerance)
    return {
        "gate_schema": (
            "exogibbs_full_condensate_element_budget_residual_gate_v1"
        ),
        "gate_name": "full_condensate_element_budget_residual",
        "accepted": bool(finite and max_abs_relative <= tolerance),
        "relative_tolerance": tolerance,
        "relative_floor": floor,
        "max_abs_relative_residual": max_abs_relative,
        "max_abs_relative_residual_element": setup.elements[max_index],
        "max_abs_relative_residual_element_index": max_index,
        "element_names": tuple(str(element) for element in setup.elements),
        "ignored_element_names": tuple(
            str(element)
            for element in setup.elements
            if str(element) in {"e-", "electron"}
        ),
        "element_budget_target": tuple(
            float(value) for value in target.tolist()
        ),
        "element_budget_reconstructed": tuple(
            float(value) for value in reconstructed.tolist()
        ),
        "element_budget_residual": tuple(
            float(value) for value in residual.tolist()
        ),
        "element_signed_relative_residual": tuple(
            float(value) for value in signed_relative.tolist()
        ),
        "element_abs_relative_residual": tuple(
            float(value) for value in absolute_relative.tolist()
        ),
        "element_relative_denominator": tuple(
            float(value) for value in denominator.tolist()
        ),
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def apply_full_condensate_budget_residual_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array | None,
    status: str,
    acceptance_tier: str,
    warning_messages: tuple[str, ...],
    metadata: dict[str, Any],
    enabled: bool,
    relative_tolerance: float,
    relative_floor: float = DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    """Apply the full-budget acceptance gate to a candidate result."""

    if element_inventory_target is None:
        return status, acceptance_tier, warning_messages, metadata
    report = full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
        relative_floor=relative_floor,
    )
    report["enabled"] = bool(enabled)
    metadata["full_condensate_budget_residual_gate"] = report
    if not enabled or report["accepted"] or status != CONVERGED:
        return status, acceptance_tier, warning_messages, metadata
    metadata.setdefault("pre_full_condensate_budget_gate_status", status)
    metadata.setdefault(
        "pre_full_condensate_budget_gate_acceptance_tier",
        acceptance_tier,
    )
    warnings = tuple(warning_messages) + (
        "The full condensate vector element-wise relative budget residual "
        "exceeded the accepted threshold.",
    )
    return (
        NOT_CONVERGED,
        "full_condensate_element_budget_residual_failed",
        warnings,
        metadata,
    )


def polish_gas_log_amounts_for_full_condensate_budget_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
    relative_floor: float = DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
    max_iterations: int = 16,
    max_abs_delta_q: float = 2.0,
) -> tuple[jnp.ndarray, Mapping[str, Any] | None]:
    """Return a diagnostic-only gas budget transform.

    Production acceptance deliberately does not call this helper because a
    gas-only amount correction cannot certify inactive-condensate closure.
    """

    q = np.asarray(gas_ln_n, dtype=np.float64).copy()
    condensates = np.asarray(condensate_amounts, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    if (
        q.ndim != 1
        or q.shape[0] != len(setup.gas_species)
        or condensates.ndim != 1
        or condensates.shape[0] != len(setup.condensate_species)
        or target.ndim != 1
        or target.shape[0] != len(setup.elements)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(condensates))
        or not np.all(np.isfinite(target))
    ):
        return jnp.asarray(gas_ln_n, dtype=jnp.float64), None

    ag, ac = setup_formula_matrices_numpy(setup)
    condensate_budget = ac @ condensates
    positive_target = target[target > 0.0]
    target_scale = (
        float(np.max(positive_target)) if positive_target.size else 1.0
    )
    floor = max(
        float(np.finfo(np.float64).tiny),
        1.0e-300 * target_scale,
    )
    row_weights = 1.0 / np.maximum(np.abs(target), floor)
    active_rows = np.asarray(
        [
            str(element) not in {"e-", "electron"}
            for element in setup.elements
        ],
        dtype=bool,
    )

    def gate_report(q_values: np.ndarray) -> dict[str, Any]:
        return full_condensate_element_budget_residual_report(
            setup=setup,
            gas_n=np.exp(q_values),
            condensate_amounts=condensates,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
            relative_floor=relative_floor,
        )

    initial_report = gate_report(q)
    if bool(initial_report["accepted"]):
        return jnp.asarray(q, dtype=jnp.float64), None

    accepted = False
    iteration_count = 0
    final_report = initial_report
    for iteration in range(int(max_iterations)):
        with np.errstate(over="ignore", invalid="ignore"):
            gas_n = np.exp(q)
        if not np.all(np.isfinite(gas_n)):
            break
        budget = ag @ gas_n + condensate_budget - target
        jacobian = ag * gas_n[None, :]
        matrix = (
            jacobian[active_rows, :]
            * row_weights[active_rows, None]
        )
        rhs = -budget[active_rows] * row_weights[active_rows]
        if matrix.size == 0:
            break
        delta_q, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if not np.all(np.isfinite(delta_q)):
            break
        norm_inf = (
            float(np.max(np.abs(delta_q))) if delta_q.size else 0.0
        )
        if norm_inf > max_abs_delta_q and norm_inf > 0.0:
            delta_q = delta_q * (max_abs_delta_q / norm_inf)
        trial_q = q + delta_q
        if not np.all(np.isfinite(trial_q)):
            break
        q = trial_q
        iteration_count = iteration + 1
        final_report = gate_report(q)
        accepted = bool(final_report["accepted"])
        if accepted:
            break

    polish_report = {
        "polish_schema": (
            "exogibbs_full_condensate_budget_gas_log_amount_polish_v1"
        ),
        "triggered": True,
        "accepted": bool(accepted),
        "iteration_count": iteration_count,
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": final_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    return jnp.asarray(q, dtype=jnp.float64), polish_report


def accept_condensate_result_state(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    condensate_amounts: Array,
    solver_success: bool,
    diagnostics: Mapping[str, Any] | None,
    element_inventory_target: Array | None,
    enable_full_condensate_budget_residual_gate: bool,
    full_condensate_budget_relative_tolerance: float,
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    ),
) -> AcceptedCondensateState:
    """Apply named post-solve transforms and acceptance gates."""

    validate_condensate_chemical_setup(setup)
    gas_ln_n_array = jnp.asarray(gas_ln_n, dtype=jnp.float64)
    if (
        gas_ln_n_array.ndim != 1
        or gas_ln_n_array.shape[0] != len(setup.gas_species)
    ):
        raise ValueError("gas_ln_n must have one value per gas species.")
    condensate_amounts_array = jnp.asarray(
        condensate_amounts,
        dtype=jnp.float64,
    )
    if (
        condensate_amounts_array.ndim != 1
        or condensate_amounts_array.shape[0]
        != len(setup.condensate_species)
    ):
        raise ValueError(
            "condensate_amounts must have one value per condensate species."
        )

    status = CONVERGED if solver_success else NOT_CONVERGED
    acceptance_tier = (
        "fixed_support_v2_accepted"
        if solver_success
        else "fixed_support_v2_solver_failed"
    )
    warning_messages: tuple[str, ...] = ()
    metadata: dict[str, Any] = dict(diagnostics or {})
    has_positive_condensate = bool(
        np.any(np.asarray(condensate_amounts_array, dtype=np.float64) > 0.0)
    )
    if (
        enable_full_condensate_budget_residual_gate
        and element_inventory_target is not None
        and status == CONVERGED
    ):
        metadata["full_condensate_budget_gas_log_amount_polish"] = {
            "polish_schema": (
                "exogibbs_full_condensate_budget_gas_log_amount_polish_v1"
            ),
            "triggered": False,
            "accepted": False,
            "skip_reason": (
                "positive_condensate_uses_zero_barrier_joint_kkt"
                if has_positive_condensate
                else "gas_only_budget_transform_disabled_requires_resolve"
            ),
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }

    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    (
        status,
        acceptance_tier,
        warning_messages,
        metadata,
    ) = apply_full_condensate_budget_residual_gate(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts_array,
        element_inventory_target=element_inventory_target,
        status=status,
        acceptance_tier=acceptance_tier,
        warning_messages=warning_messages,
        metadata=metadata,
        enabled=enable_full_condensate_budget_residual_gate,
        relative_tolerance=full_condensate_budget_relative_tolerance,
        relative_floor=full_condensate_budget_relative_floor,
    )
    if has_positive_condensate and status == CONVERGED:
        lifecycle = metadata.get("fixed_support_v2", {})
        exact_audit = (
            lifecycle.get("zero_barrier_active_support_polish")
            if isinstance(lifecycle, Mapping)
            else None
        )
        if not isinstance(exact_audit, Mapping) or not bool(
            exact_audit.get("accepted", False)
        ):
            metadata["pre_physical_condensate_kkt_audit_status"] = status
            status = NOT_CONVERGED
            acceptance_tier = "physical_condensate_kkt_audit_failed"
            warning_messages = tuple(warning_messages) + (
                "A positive-condensate state lacked an accepted zero-barrier "
                "physical KKT audit.",
            )
    return AcceptedCondensateState(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=condensate_amounts_array,
        status=status,
        acceptance_tier=acceptance_tier,
        warning_messages=warning_messages,
        diagnostics=metadata,
    )


def least_squares_element_potential(
    *,
    formula_matrix: Array,
    gas_ln_n: Array,
    gas_stationarity_source: Array,
) -> Array:
    """Recover an element potential from gas stationarity residuals."""

    ag = jnp.asarray(formula_matrix)
    q = jnp.asarray(gas_ln_n)
    source = jnp.asarray(gas_stationarity_source)
    if ag.ndim != 2:
        raise ValueError("formula_matrix must be two-dimensional.")
    if q.ndim != 1 or source.ndim != 1 or q.shape != source.shape:
        raise ValueError(
            "gas_ln_n and gas_stationarity_source must be same-length "
            "vectors."
        )
    if ag.shape[1] != q.shape[0]:
        raise ValueError(
            "formula_matrix column count must match gas_ln_n length."
        )
    return jnp.linalg.lstsq(ag.T, q + source, rcond=None)[0]


__all__ = (
    "accept_condensate_result_state",
    "apply_full_condensate_budget_residual_gate",
    "full_condensate_element_budget_residual_report",
    "independent_kkt_passed",
    "least_squares_element_potential",
    "polish_gas_log_amounts_for_full_condensate_budget_gate",
    "setup_formula_matrices_numpy",
)
