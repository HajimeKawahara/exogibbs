"""Production fixed-support v2 condensate equilibrium API.

The fixed-support solver owns one support solve. Support discovery and
expansion remain an API-level lifecycle outside that solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.condensates.fixed_support_v2_policy import (
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
)


Array = jax.Array
DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR = 1.0e-6
CondensateRoute = Literal["head_v2"]
CondensateFixedSupportV2Preset = Literal["validated_2026_07"]
CondensateProfileMethod = Literal["auto", "vmap_cold"]
CondensateProfileNativeActivitySupportPolicy = Literal[
    "topk_capacity",
    "fastchem_activity_all",
]
CondensateProfileFixedSupportSeedPolicy = Literal[
    "budget_preserving_fraction",
    "max_density",
]
CondensateProfileNativeActivitySource = Literal[
    "gas_only_full_budget",
    "initializer_gas",
]
CONDENSATE_HEAD_V2_ROUTE_VERSION = "v2.0"
CONDENSATE_HEAD_V2_ROUTE_NAME = "head_v2_fixed_support_lifecycle"
HEAD_ROUTE_V2 = "head_v2"
CONVERGED = "converged"
NOT_CONVERGED = "not_converged"
_SETUP_NUMPY_FORMULA_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[Any], np.ndarray, np.ndarray],
] = {}


@dataclass(frozen=True)
class CondensateChemicalSetup:
    """Gas and condensate thermochemistry bundle for condensate equilibrium."""

    gas_setup: ChemicalSetup
    condensate_setup: ChemicalSetup
    formula_matrix: Array
    formula_matrix_cond: Array
    gas_species: tuple[str, ...]
    condensate_species: tuple[str, ...]
    elements: tuple[str, ...]


@dataclass(frozen=True)
class CondensateEquilibriumOptions:
    """Options for the production fixed-support v2 route."""

    route: CondensateRoute = HEAD_ROUTE_V2
    fixed_support_v2_preset: CondensateFixedSupportV2Preset = (
        FIXED_SUPPORT_V2_VALIDATED_PRESET
    )
    profile_method: Optional[CondensateProfileMethod] = None
    return_diagnostics: bool = False
    enable_full_condensate_budget_residual_gate: bool = True
    full_condensate_budget_relative_tolerance: float = 1.0e-3
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    )


@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Result container for the production fixed-support v2 route."""

    gas_ln_n: Array
    gas_n: Array
    gas_x: Array
    gas_ntot: Array
    condensate_amounts: Array
    condensate_support_indices: Array
    condensate_support_names: tuple[str, ...]
    acceptance_tier: str
    selected_route: str
    status: str
    converged: bool
    diagnostics: Optional[Mapping[str, Any]] = None
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME


@dataclass(frozen=True)
class CondensateEquilibriumInit:
    """Optional condensate profile initial guess for one layer."""

    gas_ln_n: Optional[Array] = None
    gas_ntot: Optional[Array] = None
    condensate_amounts: Optional[Array] = None
    support_indices: Optional[Sequence[int]] = None
    support_amounts: Optional[Sequence[float]] = None
    element_potential: Optional[Array] = None
    rho: Optional[Array] = None
    barrier_epsilon: Optional[Array] = None


@dataclass(frozen=True)
class CondensateEquilibriumInitRequest:
    """Inputs available to a condensate profile initializer for one layer."""

    setup: CondensateChemicalSetup
    T: float
    P: float
    b: Array
    Pref: float = 1.0
    layer_index: Optional[int] = None
    user_init: Optional[CondensateEquilibriumInit] = None
    previous_solution: Optional[CondensateEquilibriumInit] = None


@runtime_checkable
class CondensateEquilibriumInitializer(Protocol):
    """Produce an optional condensate initial guess for one profile layer."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        ...


@dataclass(frozen=True)
class DefaultCondensateEquilibriumInitializer:
    """Use explicit per-layer init first, then the previous profile solution."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        if request.user_init is not None:
            return request.user_init
        if request.previous_solution is not None:
            return request.previous_solution
        return CondensateEquilibriumInit()


@dataclass(frozen=True)
class CondensateEquilibriumProfileResult:
    """Result container for a Python-level condensate profile solve."""

    layers: tuple[CondensateEquilibriumResult, ...]
    method: CondensateProfileMethod
    diagnostics: Optional[Mapping[str, Any]] = None
    batched_arrays: Optional[Mapping[str, Array]] = None


@dataclass(frozen=True)
class ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Reusable experimental fixed-support profile plan.

    This is an opt-in GPU-oriented surface for repeated profile evaluations
    with fixed condensate support. It intentionally does not change the default
    ``condensate_equilibrium_profile`` route.
    """

    setup: CondensateChemicalSetup
    buckets: Sequence[Any]
    formula_matrix: Array
    max_iter: int
    n_layers: int
    condensate_count: int
    bucket_layer_index_arrays: tuple[Array, ...] = ()
    temperatures: Optional[Array] = None


@dataclass(frozen=True)
class _HeadV2LayerState:
    """One host-owned fixed-support state between outer lifecycle rounds."""

    support_indices: tuple[int, ...]
    gas_ln_n: Array
    condensate_log_amounts: Array
    total_gas_log_amount: Array
    element_potential: Array


_ExperimentalProfileFixedSupportBatchPlan = (
    ExperimentalCondensateProfileFixedSupportBatchPlan
)
_DEFAULT_CONDENSATE_INITIALIZER = DefaultCondensateEquilibriumInitializer()


def validate_condensate_chemical_setup(setup: CondensateChemicalSetup) -> None:
    """Validate gas-condensate setup compatibility for HEAD route calls."""

    if not isinstance(setup.gas_setup, ChemicalSetup):
        raise TypeError("gas_setup must be a ChemicalSetup.")
    if not isinstance(setup.condensate_setup, ChemicalSetup):
        raise TypeError("condensate_setup must be a ChemicalSetup.")
    if setup.gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if setup.condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if tuple(setup.gas_setup.elements) != tuple(setup.condensate_setup.elements):
        raise ValueError("gas and condensate element orders must match.")
    formula_matrix = jnp.asarray(setup.formula_matrix)
    formula_matrix_cond = jnp.asarray(setup.formula_matrix_cond)
    if formula_matrix.ndim != 2:
        raise ValueError("formula_matrix must be a two-dimensional array.")
    if formula_matrix_cond.ndim != 2:
        raise ValueError("formula_matrix_cond must be a two-dimensional array.")
    if formula_matrix.shape[0] != formula_matrix_cond.shape[0]:
        raise ValueError("gas and condensate formula matrices must have the same element count.")
    if formula_matrix.shape[0] != len(setup.elements):
        raise ValueError("elements length must match formula matrix rows.")
    if formula_matrix.shape[1] != len(setup.gas_species):
        raise ValueError("gas_species length must match formula_matrix columns.")
    if formula_matrix_cond.shape[1] != len(setup.condensate_species):
        raise ValueError("condensate_species length must match formula_matrix_cond columns.")


def _formula_matrices_numpy(setup: CondensateChemicalSetup) -> tuple[np.ndarray, np.ndarray]:
    """Return cached NumPy formula matrices for Python-side reports/restoration."""

    key = id(setup)
    cached = _SETUP_NUMPY_FORMULA_CACHE.get(key)
    if cached is not None:
        setup_ref, formula_matrix, formula_matrix_cond = cached
        if setup_ref() is setup:
            return formula_matrix, formula_matrix_cond
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    formula_matrix_cond = np.asarray(setup.formula_matrix_cond, dtype=np.float64)

    def _drop_cache(_ref: weakref.ReferenceType[Any], *, cache_key: int = key) -> None:
        _SETUP_NUMPY_FORMULA_CACHE.pop(cache_key, None)

    _SETUP_NUMPY_FORMULA_CACHE[key] = (
        weakref.ref(setup, _drop_cache),
        formula_matrix,
        formula_matrix_cond,
    )
    return formula_matrix, formula_matrix_cond


def build_condensate_chemical_setup(
    *,
    gas_setup: ChemicalSetup,
    condensate_setup: ChemicalSetup,
) -> CondensateChemicalSetup:
    """Build and validate a gas-condensate chemical setup bundle."""

    if gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if gas_setup.species is None:
        raise ValueError("gas_setup.species is required for condensate equilibrium.")
    if condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if condensate_setup.species is None:
        raise ValueError("condensate_setup.species is required for condensate equilibrium.")
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix),
        formula_matrix_cond=jnp.asarray(condensate_setup.formula_matrix),
        gas_species=tuple(gas_setup.species),
        condensate_species=tuple(condensate_setup.species),
        elements=tuple(gas_setup.elements),
    )
    validate_condensate_chemical_setup(setup)
    return setup


def _ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    return jnp.log(jnp.asarray(pressure) / jnp.asarray(reference_pressure))


def _full_condensate_amounts(
    *,
    support_indices: Sequence[int],
    support_amounts: Array,
    condensate_count: int,
) -> Array:
    indices = np.asarray(tuple(int(index) for index in support_indices), dtype=np.int64)
    amounts = np.asarray(support_amounts)
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    if np.any(indices < 0) or np.any(indices >= condensate_count):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    full = np.zeros((condensate_count,), dtype=amounts.dtype)
    if indices.size:
        full[indices] = amounts
    return jnp.asarray(full)


def _merge_external_condensate_amounts(
    *,
    condensate_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None,
) -> Array:
    """Add externally budgeted condensates back to the public full vector."""

    amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if external_condensate_amounts is None:
        return jnp.asarray(amounts, dtype=jnp.float64)
    external = np.asarray(external_condensate_amounts, dtype=np.float64)
    if external.ndim != 1 or external.shape[0] != amounts.shape[0]:
        raise ValueError("external_condensate_amounts must match condensate_count.")
    return jnp.asarray(amounts + external, dtype=jnp.float64)


def _validate_options(options: CondensateEquilibriumOptions) -> None:
    """Validate the compact production-v2 option contract."""

    if options.route != HEAD_ROUTE_V2:
        raise ValueError(
            f"Unsupported condensate route {options.route!r}; "
            f"expected {HEAD_ROUTE_V2!r}."
        )
    if options.fixed_support_v2_preset != FIXED_SUPPORT_V2_VALIDATED_PRESET:
        raise ValueError(
            "fixed_support_v2_preset must be "
            f"{FIXED_SUPPORT_V2_VALIDATED_PRESET!r}."
        )
    if options.profile_method not in {None, "auto", "vmap_cold"}:
        raise ValueError(
            "profile_method must be None, 'auto', or 'vmap_cold'."
        )
    if not isinstance(options.return_diagnostics, bool):
        raise TypeError("return_diagnostics must be a bool.")
    if not isinstance(
        options.enable_full_condensate_budget_residual_gate,
        bool,
    ):
        raise TypeError(
            "enable_full_condensate_budget_residual_gate must be a bool."
        )
    if not math.isfinite(
        float(options.full_condensate_budget_relative_tolerance)
    ) or float(options.full_condensate_budget_relative_tolerance) <= 0.0:
        raise ValueError(
            "full_condensate_budget_relative_tolerance must be finite and "
            "positive."
        )
    if not math.isfinite(
        float(options.full_condensate_budget_relative_floor)
    ) or float(options.full_condensate_budget_relative_floor) <= 0.0:
        raise ValueError(
            "full_condensate_budget_relative_floor must be finite and "
            "positive."
        )


def _full_condensate_element_budget_residual_report(
    *,
    setup: CondensateChemicalSetup,
    gas_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
    relative_floor: float = DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
) -> dict[str, Any]:
    target = np.asarray(element_inventory_target, dtype=np.float64)
    if target.ndim != 1 or target.shape[0] != len(setup.elements):
        raise ValueError("element_inventory_target must have one value per element.")
    gas_amounts = np.asarray(gas_n, dtype=np.float64)
    cond_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if gas_amounts.ndim != 1 or gas_amounts.shape[0] != len(setup.gas_species):
        raise ValueError("gas_n must have one value per gas species.")
    if cond_amounts.ndim != 1 or cond_amounts.shape[0] != len(setup.condensate_species):
        raise ValueError("condensate_amounts must have one value per condensate species.")
    formula_matrix, formula_matrix_cond = _formula_matrices_numpy(setup)
    gas_budget = formula_matrix @ gas_amounts
    condensate_budget = formula_matrix_cond @ cond_amounts
    reconstructed = gas_budget + condensate_budget
    residual = reconstructed - target
    floor = float(relative_floor)
    denominator = np.maximum(np.abs(target), max(floor, 1.0e-300))
    signed_relative = residual / denominator
    absolute_relative = np.abs(signed_relative)
    gate_mask = np.asarray(
        tuple(str(element) not in {"e-", "electron"} for element in setup.elements),
        dtype=bool,
    )
    gated_absolute_relative = np.where(gate_mask, absolute_relative, 0.0)
    finite = bool(np.all(np.isfinite(np.where(gate_mask, absolute_relative, 0.0))))
    sanitized = np.where(
        np.isfinite(gated_absolute_relative),
        gated_absolute_relative,
        np.inf,
    )
    max_index = int(np.argmax(sanitized))
    max_abs_relative = float(gated_absolute_relative[max_index])
    tolerance = float(relative_tolerance)
    accepted = finite and max_abs_relative <= tolerance
    return {
        "gate_schema": "exogibbs_full_condensate_element_budget_residual_gate_v1",
        "gate_name": "full_condensate_element_budget_residual",
        "accepted": bool(accepted),
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
        "element_budget_target": tuple(float(value) for value in target.tolist()),
        "element_budget_reconstructed": tuple(float(value) for value in reconstructed.tolist()),
        "element_budget_residual": tuple(float(value) for value in residual.tolist()),
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


def _apply_full_condensate_budget_residual_gate(
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
    if element_inventory_target is None:
        return status, acceptance_tier, warning_messages, metadata
    report = _full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
        relative_floor=relative_floor,
    )
    metadata["full_condensate_budget_residual_gate"] = report
    if (
        not enabled
        or report["accepted"]
        or status != CONVERGED
    ):
        return status, acceptance_tier, warning_messages, metadata
    metadata.setdefault("pre_full_condensate_budget_gate_status", status)
    metadata.setdefault(
        "pre_full_condensate_budget_gate_acceptance_tier",
        acceptance_tier,
    )
    warnings = tuple(warning_messages) + (
        "The full condensate vector element-wise relative budget residual exceeded the accepted threshold.",
    )
    return (
        NOT_CONVERGED,
        "full_condensate_element_budget_residual_failed",
        warnings,
        metadata,
    )


def _polish_gas_log_amounts_for_full_condensate_budget_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
    max_iterations: int = 16,
    max_abs_delta_q: float = 2.0,
) -> tuple[jnp.ndarray, Mapping[str, Any] | None]:
    """Restore full element budget by minimally adjusting gas log amounts."""

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

    ag, ac = _formula_matrices_numpy(setup)
    condensate_budget = ac @ condensates
    positive_target = target[target > 0.0]
    target_scale = float(np.max(positive_target)) if positive_target.size else 1.0
    floor = max(float(np.finfo(np.float64).tiny), 1.0e-300 * target_scale)
    row_weights = 1.0 / np.maximum(np.abs(target), floor)
    ignored = [str(element) in {"e-", "electron"} for element in setup.elements]
    active_rows = np.asarray([not value for value in ignored], dtype=bool)

    def gate_report(q_values: np.ndarray) -> dict[str, Any]:
        return _full_condensate_element_budget_residual_report(
            setup=setup,
            gas_n=np.exp(q_values),
            condensate_amounts=condensates,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
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
        jac = ag * gas_n[None, :]
        matrix = jac[active_rows, :] * row_weights[active_rows, None]
        rhs = -budget[active_rows] * row_weights[active_rows]
        if matrix.size == 0:
            break
        delta_q, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if not np.all(np.isfinite(delta_q)):
            break
        norm_inf = float(np.max(np.abs(delta_q))) if delta_q.size else 0.0
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
        "polish_schema": "exogibbs_full_condensate_budget_gas_log_amount_polish_v1",
        "triggered": True,
        "accepted": bool(accepted),
        "iteration_count": iteration_count,
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": final_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    if accepted:
        return jnp.asarray(q, dtype=jnp.float64), polish_report
    return jnp.asarray(q, dtype=jnp.float64), polish_report


def _least_squares_element_potential(
    *,
    formula_matrix: Array,
    gas_ln_n: Array,
    gas_stationarity_source: Array,
) -> Array:
    ag = jnp.asarray(formula_matrix)
    q = jnp.asarray(gas_ln_n)
    source = jnp.asarray(gas_stationarity_source)
    if ag.ndim != 2:
        raise ValueError("formula_matrix must be two-dimensional.")
    if q.ndim != 1 or source.ndim != 1 or q.shape != source.shape:
        raise ValueError("gas_ln_n and gas_stationarity_source must be same-length vectors.")
    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match gas_ln_n length.")
    return jnp.linalg.lstsq(ag.T, q + source, rcond=None)[0]


def build_condensate_equilibrium_result_from_solver_payload(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    external_condensate_amounts: Sequence[float] | Array | None = None,
    selected_route: str,
    solver_success: bool,
    route: CondensateRoute = HEAD_ROUTE_V2,
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION,
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME,
    diagnostics: Optional[Mapping[str, Any]] = None,
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    ),
) -> CondensateEquilibriumResult:
    """Build a production-v2 result from explicit solver arrays."""

    validate_condensate_chemical_setup(setup)
    if route != HEAD_ROUTE_V2:
        raise ValueError("Only the head_v2 result route is supported.")
    gas_ln_n_array = jnp.asarray(gas_ln_n, dtype=jnp.float64)
    if (
        gas_ln_n_array.ndim != 1
        or gas_ln_n_array.shape[0] != len(setup.gas_species)
    ):
        raise ValueError("gas_ln_n must have one value per gas species.")
    support_amounts_array = jnp.asarray(
        support_amounts,
        dtype=jnp.float64,
    )
    condensate_amounts = _full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=support_amounts_array,
        condensate_count=len(setup.condensate_species),
    )
    condensate_amounts = _merge_external_condensate_amounts(
        condensate_amounts=condensate_amounts,
        external_condensate_amounts=external_condensate_amounts,
    )
    status = CONVERGED if solver_success else NOT_CONVERGED
    acceptance_tier = (
        "fixed_support_v2_accepted"
        if solver_success
        else "fixed_support_v2_solver_failed"
    )
    warning_messages: tuple[str, ...] = ()
    metadata: dict[str, Any] = dict(diagnostics or {})
    if (
        enable_full_condensate_budget_residual_gate
        and element_inventory_target is not None
        and status == CONVERGED
    ):
        polished_gas_ln_n, polish_report = (
            _polish_gas_log_amounts_for_full_condensate_budget_gate(
                setup=setup,
                gas_ln_n=gas_ln_n_array,
                condensate_amounts=condensate_amounts,
                element_inventory_target=element_inventory_target,
                relative_tolerance=(
                    full_condensate_budget_relative_tolerance
                ),
            )
        )
        if polish_report is not None:
            metadata["full_condensate_budget_gas_log_amount_polish"] = (
                polish_report
            )
            if bool(polish_report["accepted"]):
                gas_ln_n_array = polished_gas_ln_n
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    support_index_array = jnp.asarray(
        tuple(int(index) for index in support_indices),
        dtype=jnp.int32,
    )
    support_names = tuple(
        setup.condensate_species[int(index)]
        for index in support_index_array.tolist()
    )
    metadata.setdefault("route", route)
    metadata.setdefault("head_route_version", head_route_version)
    metadata.setdefault("head_route_name", head_route_name)
    metadata.setdefault("selected_route", selected_route)
    metadata.setdefault("acceptance_tier", acceptance_tier)
    metadata.setdefault("warning_messages", warning_messages)
    metadata.setdefault(
        "fastchem4_trace_public_runtime_constructor_inputs_used",
        False,
    )
    (
        status,
        acceptance_tier,
        warning_messages,
        metadata,
    ) = _apply_full_condensate_budget_residual_gate(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        status=status,
        acceptance_tier=acceptance_tier,
        warning_messages=warning_messages,
        metadata=metadata,
        enabled=enable_full_condensate_budget_residual_gate,
        relative_tolerance=full_condensate_budget_relative_tolerance,
        relative_floor=full_condensate_budget_relative_floor,
    )
    metadata["acceptance_tier"] = acceptance_tier
    metadata["warning_messages"] = warning_messages
    return CondensateEquilibriumResult(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=condensate_amounts,
        condensate_support_indices=support_index_array,
        condensate_support_names=support_names,
        acceptance_tier=acceptance_tier,
        selected_route=selected_route,
        status=status,
        converged=status == CONVERGED,
        diagnostics=metadata,
        head_route_version=head_route_version,
        head_route_name=head_route_name,
    )


def _build_empty_support_gas_result(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    diagnostics: Optional[Mapping[str, Any]],
    route: CondensateRoute = HEAD_ROUTE_V2,
    selected_route: str = "head_v2_gas_only_no_candidate",
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION,
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME,
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    ),
) -> CondensateEquilibriumResult:
    """Build a gas-only v2 lifecycle result."""

    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=(),
        support_amounts=(),
        selected_route=selected_route,
        solver_success=True,
        route=route,
        head_route_version=head_route_version,
        head_route_name=head_route_name,
        diagnostics=diagnostics,
        element_inventory_target=element_inventory_target,
        enable_full_condensate_budget_residual_gate=(
            enable_full_condensate_budget_residual_gate
        ),
        full_condensate_budget_relative_tolerance=(
            full_condensate_budget_relative_tolerance
        ),
        full_condensate_budget_relative_floor=(
            full_condensate_budget_relative_floor
        ),
    )


def _positive_support_amounts_for_warm_start(
    amounts: Sequence[float],
    *,
    min_seed_amount: float,
) -> tuple[float, ...]:
    floor = float(min_seed_amount)
    return tuple(
        float(value) if math.isfinite(float(value)) and float(value) > 0.0 else floor
        for value in amounts
    )


def _support_payload_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    min_seed_amount: float,
) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    """Return finite support payload from an optional profile initializer."""

    if init is None:
        return None
    if init.support_indices is not None:
        support_indices = tuple(int(index) for index in init.support_indices)
        if init.support_amounts is not None:
            support_amounts = _positive_support_amounts_for_warm_start(
                init.support_amounts,
                min_seed_amount=min_seed_amount,
            )
        elif init.condensate_amounts is not None:
            amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
            if amounts.ndim != 1 or amounts.shape[0] != len(setup.condensate_species):
                return None
            support_amounts = _positive_support_amounts_for_warm_start(
                (amounts[index] for index in support_indices),
                min_seed_amount=min_seed_amount,
            )
        else:
            return None
    elif init.condensate_amounts is not None:
        amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
        if amounts.ndim != 1 or amounts.shape[0] != len(setup.condensate_species):
            return None
        active = np.flatnonzero(np.isfinite(amounts) & (amounts > 0.0))
        support_indices = tuple(int(index) for index in active.tolist())
        support_amounts = _positive_support_amounts_for_warm_start(
            (amounts[index] for index in support_indices),
            min_seed_amount=min_seed_amount,
        )
    else:
        return None
    if len(support_indices) != len(support_amounts):
        return None
    if len(set(support_indices)) != len(support_indices):
        return None
    if any(index < 0 or index >= len(setup.condensate_species) for index in support_indices):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in support_amounts):
        return None
    return support_indices, support_amounts


def _solver_log_state_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    support_amounts_init: Sequence[float],
    source: str,
) -> Any | None:
    """Build a v2-owned prepared state from a profile initializer."""

    if init is None or init.gas_ln_n is None:
        return None
    gas_ln_n = jnp.asarray(init.gas_ln_n, dtype=jnp.float64)
    if gas_ln_n.ndim != 1 or gas_ln_n.shape[0] != len(setup.gas_species):
        return None
    if not bool(jnp.all(jnp.isfinite(gas_ln_n))):
        return None
    if init.gas_ntot is None:
        gas_ntot = jnp.sum(jnp.exp(gas_ln_n))
    else:
        gas_ntot = jnp.asarray(init.gas_ntot, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(gas_ntot))) or float(gas_ntot) <= 0.0:
        return None
    support_amounts = jnp.asarray(support_amounts_init, dtype=jnp.float64)
    if (
        support_amounts.ndim != 1
        or not bool(jnp.all(jnp.isfinite(support_amounts)))
        or not bool(jnp.all(support_amounts > 0.0))
    ):
        return None
    from exogibbs.optimize.fixed_support_v2_profile import (
        PreparedFixedSupportV2LayerState,
    )

    element_potential = None
    if init.element_potential is not None:
        element_potential = jnp.asarray(init.element_potential, dtype=jnp.float64)
        if (
            element_potential.ndim != 1
            or element_potential.shape[0] != len(setup.elements)
            or not bool(jnp.all(jnp.isfinite(element_potential)))
        ):
            element_potential = None
    rho = None
    if init.rho is not None:
        rho = jnp.asarray(init.rho, dtype=jnp.float64)
        if (
            rho.ndim != 1
            or rho.shape[0] not in {len(support_amounts_init), len(setup.condensate_species)}
            or not bool(jnp.all(jnp.isfinite(rho)))
        ):
            rho = None
    barrier_epsilon = None
    if init.barrier_epsilon is not None:
        barrier_epsilon = jnp.asarray(init.barrier_epsilon, dtype=jnp.float64)
        if barrier_epsilon.ndim != 0 or not bool(jnp.isfinite(barrier_epsilon)):
            barrier_epsilon = None
    del source
    return PreparedFixedSupportV2LayerState(
        ln_nk=gas_ln_n,
        ln_mk=jnp.log(jnp.maximum(support_amounts, 1.0e-300)),
        ln_ntot=jnp.log(jnp.asarray(gas_ntot, dtype=jnp.float64)),
        element_potential=element_potential,
        rho=rho,
        barrier_epsilon=barrier_epsilon,
    )


def _native_activity_expanded_profile_support_payload(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    support_indices: Sequence[int],
    support_policy: CondensateProfileNativeActivitySupportPolicy,
    support_topk: int,
    max_support_count: int,
    activity_threshold: float,
    activity_source_requested: CondensateProfileNativeActivitySource,
    seed_policy: CondensateProfileFixedSupportSeedPolicy,
    seed_fraction: float,
    max_seed_amount: float,
    min_seed_amount: float,
    activity_gas_ln_n: Sequence[float] | Array | None = None,
    activity_gas_ntot: Sequence[float] | Array | float | None = None,
    activity_gas_stationarity_source: Sequence[float] | Array | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], Mapping[str, Any]]:
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
    from exogibbs.condensates.support_selection_policy import (
        select_activity_driven_support_candidates,
    )

    base_support = tuple(dict.fromkeys(int(index) for index in support_indices))
    activity_source = str(activity_source_requested)
    gas_ln_n_for_activity = None
    gas_stationarity_source = None
    if activity_source == "initializer_gas" and activity_gas_ln_n is not None:
        candidate_ln_n = jnp.asarray(activity_gas_ln_n, dtype=jnp.float64)
        if (
            candidate_ln_n.ndim == 1
            and candidate_ln_n.shape[0] == len(setup.gas_species)
            and bool(jnp.all(jnp.isfinite(candidate_ln_n)))
        ):
            gas_ln_n_for_activity = candidate_ln_n
            if activity_gas_stationarity_source is not None:
                candidate_source = jnp.asarray(
                    activity_gas_stationarity_source,
                    dtype=jnp.float64,
                )
                if (
                    candidate_source.ndim == 1
                    and candidate_source.shape[0] == len(setup.gas_species)
                    and bool(jnp.all(jnp.isfinite(candidate_source)))
                ):
                    gas_stationarity_source = candidate_source
            if gas_stationarity_source is None:
                if activity_gas_ntot is None:
                    gas_ntot = jnp.sum(jnp.exp(candidate_ln_n))
                else:
                    gas_ntot = jnp.asarray(activity_gas_ntot, dtype=jnp.float64)
                gas_stationarity_source = setup.gas_setup.hvector_func(float(T)) + (
                    _ln_normalized_pressure(P, Pref)
                ) - jnp.log(jnp.clip(gas_ntot, 1.0e-300))
    if gas_ln_n_for_activity is None or gas_stationarity_source is None:
        activity_source = "gas_only_full_budget"
        gas_result = equilibrium(
            setup.gas_setup,
            float(T),
            float(P),
            jnp.asarray(b, dtype=jnp.float64),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        gas_ln_n_for_activity = jnp.asarray(gas_result.ln_n, dtype=jnp.float64)
        gas_stationarity_source = setup.gas_setup.hvector_func(float(T)) + (
            _ln_normalized_pressure(P, Pref)
        ) - jnp.log(jnp.asarray(gas_result.ntot, dtype=jnp.float64))
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=gas_ln_n_for_activity,
        gas_stationarity_source=gas_stationarity_source,
    )
    element_potential_provenance = (
        "exogibbs_profile_initializer_gas_state"
        if activity_source == "initializer_gas"
        else "exogibbs_native_gas_only_equilibrium"
    )
    activity_gauge_note = (
        "FastChem-style log_activity = A_cond.T @ element_potential "
        "- hvector_cond, with element_potential recovered from the selected "
        "ExoGibbs-native gas state."
    )
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=b,
        condensate_species_order=setup.condensate_species,
        hvector_cond=setup.condensate_setup.hvector_func(float(T)),
        element_potential=element_potential,
        max_positive_support_count=int(support_topk),
        activity_threshold=float(activity_threshold),
        existing_support_indices=base_support,
        temperature=float(T),
        condensate_temperature_validity_upper=setup.condensate_setup.metadata.get(
            "temperature_validity_upper"
        ),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_profile_budget",
            "hvector_cond": "exogibbs_condensate_thermochemistry",
            "element_potential": element_potential_provenance,
            "condensate_temperature_validity_upper": (
                "exogibbs_condensate_temperature_validity_metadata"
            ),
        },
    )
    if support_policy == "fastchem_activity_all":
        report = select_activity_driven_support_candidates(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=b,
            condensate_species_order=setup.condensate_species,
            hvector_cond=setup.condensate_setup.hvector_func(float(T)),
            element_potential=element_potential,
            max_positive_support_count=None,
            activity_threshold=float(activity_threshold),
            existing_support_indices=base_support,
            temperature=float(T),
            condensate_temperature_validity_upper=setup.condensate_setup.metadata.get(
                "temperature_validity_upper"
            ),
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_profile_budget",
                "hvector_cond": "exogibbs_condensate_thermochemistry",
                "element_potential": element_potential_provenance,
                "condensate_temperature_validity_upper": (
                    "exogibbs_condensate_temperature_validity_metadata"
                ),
                "activity_gauge": activity_gauge_note,
            },
        )
    additions = tuple(
        int(index)
        for index in report.positive_support_indices
        if int(index) not in set(base_support)
    )
    support_limit = max(
        int(max_support_count),
        len(base_support),
    )
    expanded_support = tuple(dict.fromkeys((*base_support, *additions)))[:support_limit]
    seeded_support = expanded_support
    if seeded_support:
        from exogibbs.condensates.initialization_policy import (
            recommend_budget_preserving_seed_amounts,
        )

        seed = recommend_budget_preserving_seed_amounts(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            condensate_species_order=setup.condensate_species,
            support_indices=seeded_support,
            seed_fraction=1.0 if seed_policy == "max_density" else seed_fraction,
            max_seed_amount=(
                1.0e300 if seed_policy == "max_density" else max_seed_amount
            ),
            min_seed_amount=min_seed_amount,
            preserve_budget_fraction=(seed_policy == "budget_preserving_fraction"),
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_profile_budget",
                "recommended_amounts": (
                    "derived_from_native_budget_capacity_with_shared_budget_fraction"
                    if seed_policy == "budget_preserving_fraction"
                    else "derived_from_native_budget_capacity_without_shared_budget_rescale"
                ),
            },
        )
        seeded_amounts = tuple(
            float(value) for value in seed.recommended_amounts
        )
    else:
        seeded_amounts = ()
    trace = {
        "policy": (
            "fastchem_style_activity_all_support_expansion"
            if support_policy == "fastchem_activity_all"
            else "native_gas_activity_curated_support_expansion"
        ),
        "profile_native_activity_support_policy": support_policy,
        "profile_native_activity_source": activity_source,
        "profile_native_activity_source_requested": activity_source_requested,
        "activity_threshold_semantics": (
            "FastChem selectActiveCondensates-compatible: log_activity >= "
            "threshold, where ExoGibbs computes log_activity as "
            "A_cond.T @ element_potential - hvector_cond."
        ),
        "fastchem4_public_values_used_as_constructor_inputs": False,
        "base_support_count": len(base_support),
        "max_support_count": support_limit,
        "profile_fixed_support_seed_policy": seed_policy,
        "seed_policy_semantics": (
            "budget_preserving_fraction keeps the existing batch seed behavior; "
            "max_density uses ExoGibbs-native condensate capacity without shared "
            "support rescaling, analogous to FastChem maxDensity initialization."
        ),
        "uncapped_positive_activity_count": len(report.positive_support_indices),
        "expanded_support_count": len(seeded_support),
        "added_support_count": max(0, len(seeded_support) - len(base_support)),
        "added_support_indices": tuple(
            index for index in seeded_support if index not in set(base_support)
        ),
        "added_support_names": tuple(
            setup.condensate_species[int(index)]
            for index in seeded_support
            if int(index) not in set(base_support)
        ),
        "selection_report": report.as_dict(),
    }
    return seeded_support, seeded_amounts, trace


def _head_v2_best_residual_element_potential(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    gas_ln_n: Array,
    total_gas_log_amount: Array,
    epsilon: float,
) -> Array:
    """Return the validated global best-residual multiplier initializer."""

    ag = jnp.asarray(setup.formula_matrix, dtype=jnp.float64)
    support = jnp.asarray(
        tuple(int(index) for index in support_indices), dtype=jnp.int32
    )
    ac = jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64)[:, support]
    q = jnp.asarray(gas_ln_n, dtype=jnp.float64)
    r = jnp.log(jnp.asarray(support_amounts, dtype=jnp.float64))
    qtot = jnp.asarray(total_gas_log_amount, dtype=jnp.float64)
    gamma = jnp.asarray(
        setup.gas_setup.hvector_func(float(T)), dtype=jnp.float64
    ) + _ln_normalized_pressure(P, Pref)
    hcond = jnp.asarray(
        setup.condensate_setup.hvector_func(float(T)), dtype=jnp.float64
    )[support]
    eta = jnp.ones_like(r)
    gas_matrix = ag.T
    gas_rhs = q + gamma - qtot
    joint_matrix = jnp.concatenate([ag.T, ac.T], axis=0)
    joint_rhs = jnp.concatenate([gas_rhs, hcond - eta])

    def damped_lstsq(matrix: Array, rhs: Array) -> Array:
        column_scale = jnp.maximum(
            jnp.linalg.norm(matrix, axis=0),
            jnp.asarray(1.0e-300, dtype=matrix.dtype),
        )
        scaled = matrix / column_scale[None, :]
        normal = scaled.T @ scaled
        normal_rhs = scaled.T @ rhs
        damping = jnp.maximum(
            jnp.asarray(1.0e-12, dtype=matrix.dtype)
            * jnp.mean(jnp.diag(normal)),
            jnp.asarray(1.0e-30, dtype=matrix.dtype),
        )
        solution = jnp.linalg.solve(
            normal + damping * jnp.eye(normal.shape[0], dtype=matrix.dtype),
            normal_rhs,
        )
        return jnp.nan_to_num(
            solution / column_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    candidates = jnp.stack(
        [
            jnp.zeros((ag.shape[0],), dtype=jnp.float64),
            jnp.linalg.lstsq(gas_matrix, gas_rhs, rcond=None)[0],
            jnp.linalg.lstsq(joint_matrix, joint_rhs, rcond=None)[0],
            damped_lstsq(gas_matrix, gas_rhs),
            damped_lstsq(joint_matrix, joint_rhs),
        ],
        axis=0,
    )

    def residual_norm(element_potential: Array) -> Array:
        values = jnp.concatenate(
            [
                q + gamma - qtot - ag.T @ element_potential,
                hcond - ac.T @ element_potential - eta,
                ag @ jnp.exp(q)
                + ac @ jnp.exp(r)
                - jnp.asarray(b, dtype=jnp.float64),
                r - jnp.asarray(epsilon, dtype=jnp.float64),
                jnp.asarray(
                    [jnp.sum(jnp.exp(q)) - jnp.exp(qtot)],
                    dtype=jnp.float64,
                ),
            ]
        )
        scale = jnp.max(
            jnp.abs(values),
            initial=jnp.asarray(0.0, dtype=values.dtype),
        )
        return jnp.where(
            scale == 0.0,
            0.0,
            scale * jnp.linalg.norm(values / scale),
        )

    residuals = jax.vmap(residual_norm)(candidates)
    return candidates[jnp.argmin(residuals)]


def _head_v2_initial_state(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    initial_guess: CondensateEquilibriumInit | None,
    first_epsilon: float,
) -> _HeadV2LayerState:
    """Build one v2 lifecycle state from gas equilibrium and support seeds."""

    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium

    support = tuple(int(index) for index in support_indices)
    amounts = jnp.asarray(support_amounts, dtype=jnp.float64)
    candidate_q = None if initial_guess is None else initial_guess.gas_ln_n
    if candidate_q is None:
        gas_result = equilibrium(
            setup.gas_setup,
            float(T),
            float(P),
            jnp.asarray(b, dtype=jnp.float64),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        q = jnp.asarray(gas_result.ln_n, dtype=jnp.float64)
        qtot = jnp.log(jnp.asarray(gas_result.ntot, dtype=jnp.float64))
    else:
        q = jnp.asarray(candidate_q, dtype=jnp.float64)
        if q.ndim != 1 or q.shape[0] != len(setup.gas_species):
            raise ValueError(
                "head_v2 initial gas_ln_n must have one value per gas species."
            )
        if initial_guess is not None and initial_guess.gas_ntot is not None:
            gas_ntot = jnp.asarray(initial_guess.gas_ntot, dtype=jnp.float64)
        else:
            gas_ntot = jnp.sum(jnp.exp(q))
        qtot = jnp.log(jnp.clip(gas_ntot, 1.0e-300))
    r = jnp.log(jnp.maximum(amounts, 1.0e-300))
    supplied_potential = (
        None if initial_guess is None else initial_guess.element_potential
    )
    if supplied_potential is None:
        element_potential = _head_v2_best_residual_element_potential(
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            b=b,
            support_indices=support,
            support_amounts=amounts,
            gas_ln_n=q,
            total_gas_log_amount=qtot,
            epsilon=first_epsilon,
        )
    else:
        element_potential = jnp.asarray(
            supplied_potential, dtype=jnp.float64
        )
        if (
            element_potential.ndim != 1
            or element_potential.shape[0] != len(setup.elements)
        ):
            raise ValueError(
                "head_v2 element_potential must have one value per element."
            )
    return _HeadV2LayerState(
        support_indices=support,
        gas_ln_n=q,
        condensate_log_amounts=r,
        total_gas_log_amount=qtot,
        element_potential=element_potential,
    )


def _head_v2_prepared_buckets(
    *,
    setup: CondensateChemicalSetup,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    b: Array,
    Pref: float,
    states: Sequence[_HeadV2LayerState],
) -> tuple[Any, ...]:
    """Group pending v2 lifecycle states by exact fixed support."""

    from exogibbs.optimize.fixed_support_v2_profile import (
        PreparedFixedSupportV2Bucket,
    )

    groups: dict[tuple[int, ...], list[int]] = {}
    for local_index, state in enumerate(states):
        groups.setdefault(state.support_indices, []).append(local_index)
    formula_matrix_cond = jnp.asarray(
        setup.formula_matrix_cond, dtype=jnp.float64
    )
    target = jnp.asarray(b, dtype=jnp.float64)
    buckets = []
    for support, local_indices in groups.items():
        support_array = jnp.asarray(support, dtype=jnp.int32)
        buckets.append(
            PreparedFixedSupportV2Bucket(
                support_indices=support,
                layer_indices=tuple(local_indices),
                formula_matrix_cond_active=formula_matrix_cond[:, support_array],
                ln_nk_init=jnp.stack(
                    [states[index].gas_ln_n for index in local_indices]
                ),
                ln_mk_init=jnp.stack(
                    [
                        states[index].condensate_log_amounts
                        for index in local_indices
                    ]
                ),
                ln_ntot_init=jnp.stack(
                    [
                        states[index].total_gas_log_amount
                        for index in local_indices
                    ]
                ),
                element_potential_init=jnp.stack(
                    [states[index].element_potential for index in local_indices]
                ),
                rho_init=None,
                barrier_epsilon_init=None,
                element_inventory_target=jnp.stack(
                    [target for _index in local_indices]
                ),
                hvector=jnp.stack(
                    [
                        jnp.asarray(
                            setup.gas_setup.hvector_func(
                                float(temperatures[index])
                            ),
                            dtype=jnp.float64,
                        )
                        for index in local_indices
                    ]
                ),
                hvector_cond_active=jnp.stack(
                    [
                        jnp.asarray(
                            setup.condensate_setup.hvector_func(
                                float(temperatures[index])
                            ),
                            dtype=jnp.float64,
                        )[support_array]
                        for index in local_indices
                    ]
                ),
                ln_normalized_pressure=jnp.stack(
                    [
                        _ln_normalized_pressure(
                            float(pressures[index]), Pref
                        )
                        for index in local_indices
                    ]
                ),
            )
        )
    return tuple(buckets)


def _head_v2_kkt_row(kkt_norms: Any, index: int) -> Mapping[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value))[index])
        for name, value in kkt_norms._asdict().items()
    }


def _head_v2_kkt_passed(
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


def _resolve_condensate_initial_guess(
    initializer: Optional[CondensateEquilibriumInitializer],
    request: CondensateEquilibriumInitRequest,
) -> CondensateEquilibriumInit:
    active_initializer = initializer or _DEFAULT_CONDENSATE_INITIALIZER
    return active_initializer(request)


def _run_head_v2_profile(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: CondensateEquilibriumInitializer | None,
    support_indices: Sequence[int] | None,
    support_amounts_init: Sequence[float] | None,
    options: CondensateEquilibriumOptions,
    return_diagnostics: bool,
) -> CondensateEquilibriumProfileResult:
    """Run the production v2 route and its external support lifecycle."""

    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
    from exogibbs.condensates.fixed_support_payload import (
        seed_fixed_support_payload,
    )
    from exogibbs.condensates.fixed_support_v2_policy import (
        fixed_support_v2_production_policy,
    )
    from exogibbs.optimize.fixed_support_v2.types import TerminalStatus
    from exogibbs.optimize.fixed_support_v2_profile import run_prepared_profile_v2

    policy = fixed_support_v2_production_policy(
        options.fixed_support_v2_preset
    )
    n_layers = int(temperatures.shape[0])
    records: list[dict[str, Any]] = [
        {"layer_index": index, "rounds": []} for index in range(n_layers)
    ]
    pending: dict[int, _HeadV2LayerState] = {}
    gas_only_layers: set[int] = set()
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                raise ValueError(
                    "head_v2 requires support_amounts_init with explicit "
                    "support_indices."
                )
            base_support = tuple(int(index) for index in support_indices)
            base_amounts = _positive_support_amounts_for_warm_start(
                support_amounts_init,
                min_seed_amount=policy.min_seed_amount,
            )
        else:
            payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=policy.min_seed_amount,
            )
            base_support = () if payload is None else payload[0]
            base_amounts = () if payload is None else payload[1]
        if len(base_support) != len(base_amounts):
            raise ValueError(
                "head_v2 support indices and initial amounts must have the "
                "same length."
            )
        if len(set(base_support)) != len(base_support):
            raise ValueError("head_v2 support indices must be unique.")
        if any(
            index < 0 or index >= len(setup.condensate_species)
            for index in base_support
        ):
            raise ValueError(
                "head_v2 support contains an out-of-range condensate index."
            )
        (
            initial_support,
            initial_amounts,
            support_trace,
        ) = _native_activity_expanded_profile_support_payload(
            setup=setup,
            T=float(temperatures[layer_index]),
            P=float(pressures[layer_index]),
            b=b,
            Pref=Pref,
            support_indices=base_support,
            support_policy="topk_capacity",
            support_topk=policy.initial_support_topk,
            max_support_count=policy.initial_support_limit,
            activity_threshold=0.0,
            activity_source_requested="gas_only_full_budget",
            seed_policy="budget_preserving_fraction",
            seed_fraction=policy.seed_fraction,
            max_seed_amount=policy.max_seed_amount,
            min_seed_amount=policy.min_seed_amount,
            activity_gas_ln_n=None,
            activity_gas_ntot=None,
            activity_gas_stationarity_source=None,
        )
        base_amount_by_index = {
            int(index): float(amount)
            for index, amount in zip(base_support, base_amounts)
        }
        initial_amounts = tuple(
            base_amount_by_index.get(int(index), float(amount))
            for index, amount in zip(initial_support, initial_amounts)
        )
        records[layer_index]["initial_support_indices"] = initial_support
        records[layer_index]["initial_support_count"] = len(initial_support)
        records[layer_index]["initial_support_selection"] = support_trace
        if not initial_support:
            records[layer_index]["outcome"] = "gas_only_no_candidate"
            gas_only_layers.add(layer_index)
            continue
        pending[layer_index] = _head_v2_initial_state(
            setup=setup,
            T=float(temperatures[layer_index]),
            P=float(pressures[layer_index]),
            b=b,
            Pref=Pref,
            support_indices=initial_support,
            support_amounts=initial_amounts,
            initial_guess=initial_guess,
            first_epsilon=policy.solver_config.continuation.epsilon_schedule[0],
        )

    last_outputs: dict[int, dict[str, Any]] = {}
    compilation_seconds = 0.0
    execution_seconds = 0.0
    diagnostic_seconds = 0.0
    backend = jax.default_backend()
    for round_index in range(policy.lifecycle_max_rounds):
        if not pending:
            break
        source_indices = tuple(sorted(pending))
        round_states = tuple(pending[index] for index in source_indices)
        round_temperatures = tuple(
            float(temperatures[index]) for index in source_indices
        )
        round_pressures = tuple(
            float(pressures[index]) for index in source_indices
        )
        buckets = _head_v2_prepared_buckets(
            setup=setup,
            temperatures=round_temperatures,
            pressures=round_pressures,
            b=b,
            Pref=Pref,
            states=round_states,
        )
        hcond_full = jnp.stack(
            [
                jnp.asarray(
                    setup.condensate_setup.hvector_func(temperature),
                    dtype=jnp.float64,
                )
                for temperature in round_temperatures
            ]
        )
        metadata = getattr(setup.condensate_setup, "metadata", {})
        validity_upper = metadata.get("temperature_validity_upper")
        if validity_upper is None:
            valid_mask = jnp.ones(
                (len(source_indices), len(setup.condensate_species)),
                dtype=bool,
            )
        else:
            upper = jnp.asarray(validity_upper, dtype=jnp.float64)
            if upper.shape != (len(setup.condensate_species),):
                raise ValueError(
                    "temperature_validity_upper must have one value per "
                    "condensate."
                )
            valid_mask = (
                jnp.asarray(round_temperatures, dtype=jnp.float64)[:, None]
                <= upper[None, :]
            )
        raw = run_prepared_profile_v2(
            buckets=buckets,
            formula_matrix=setup.formula_matrix,
            formula_matrix_cond_full=setup.formula_matrix_cond,
            condensate_standard_source_full=hcond_full,
            condensate_valid_mask=valid_mask,
            layer_count=len(source_indices),
            condensate_count=len(setup.condensate_species),
            config=policy.solver_config,
            budget_relative_floor=policy.budget_relative_floor,
            support_closure_tolerance=policy.support_closure_tolerance,
            include_terminal_diagnostics=return_diagnostics,
        )
        compilation_seconds += float(raw["compilation_seconds"])
        execution_seconds += float(raw["execution_seconds"])
        diagnostic_seconds += float(raw["diagnostic_seconds"])
        backend = str(raw["backend"])
        converged = np.asarray(
            jax.device_get(raw["fixed_support_converged"]), dtype=bool
        )
        closed = np.asarray(jax.device_get(raw["support_closed"]), dtype=bool)
        expansion = np.asarray(
            jax.device_get(raw["support_expansion_mask"]), dtype=bool
        )
        driving = np.asarray(
            jax.device_get(raw["inactive_condensate_driving"]),
            dtype=np.float64,
        )
        terminal = np.asarray(
            jax.device_get(raw["terminal_status"]), dtype=np.int64
        )
        next_pending: dict[int, _HeadV2LayerState] = {}
        for local_index, source_index in enumerate(source_indices):
            current = pending[source_index]
            candidate_indices = np.flatnonzero(expansion[local_index])
            ordered = tuple(
                int(index)
                for index in candidate_indices[
                    np.argsort(
                        driving[local_index, candidate_indices],
                        kind="stable",
                    )
                ]
            )
            additions = ordered[: policy.support_add_per_round]
            remaining_capacity = max(
                0, policy.support_limit - len(current.support_indices)
            )
            additions = additions[:remaining_capacity]
            expanded_support = tuple(
                dict.fromkeys((*current.support_indices, *additions))
            )
            terminal_code = int(terminal[local_index])
            independent_kkt = _head_v2_kkt_row(
                raw["final_kkt_norms"], local_index
            )
            tolerances = policy.solver_config.normal
            independent_kkt_passed = _head_v2_kkt_passed(
                independent_kkt,
                stationarity_tolerance=tolerances.stationarity_tolerance,
                budget_tolerance=tolerances.budget_tolerance,
                complementarity_tolerance=(
                    tolerances.complementarity_tolerance
                ),
                total_density_tolerance=(
                    tolerances.total_density_tolerance
                ),
            )
            final_state_values_finite = bool(
                np.asarray(
                    jax.device_get(
                        raw["final_state_values_finite"][local_index]
                    )
                )
            )
            round_record = {
                "round_index": round_index,
                "support_indices": current.support_indices,
                "support_count": len(current.support_indices),
                "fixed_support_converged": bool(converged[local_index]),
                "support_closed": bool(closed[local_index]),
                "terminal_status": terminal_code,
                "terminal_status_name": TerminalStatus(terminal_code).name,
                "positive_inactive_count": int(candidate_indices.size),
                "added_support_indices": additions,
                "independent_kkt": independent_kkt,
                "independent_kkt_passed": independent_kkt_passed,
                "final_state_values_finite": final_state_values_finite,
            }
            records[source_index]["rounds"].append(round_record)
            last_outputs[source_index] = {
                "raw": raw,
                "local_index": local_index,
                "support_indices": current.support_indices,
                "fixed_support_converged": bool(converged[local_index]),
                "support_closed": bool(closed[local_index]),
                "terminal_status": terminal_code,
                "independent_kkt": independent_kkt,
                "independent_kkt_passed": independent_kkt_passed,
                "final_state_values_finite": final_state_values_finite,
            }
            if not converged[local_index]:
                records[source_index]["outcome"] = "fixed_support_failed"
                continue
            if not independent_kkt_passed:
                records[source_index]["outcome"] = "independent_kkt_failed"
                continue
            if not final_state_values_finite:
                records[source_index]["outcome"] = "nonfinite_final_state"
                continue
            if closed[local_index]:
                records[source_index]["outcome"] = "closed"
                continue
            if not additions:
                records[source_index]["outcome"] = (
                    "open_at_support_or_round_limit"
                )
                continue
            seeded_support, seeded_amounts = seed_fixed_support_payload(
                setup=setup,
                element_inventory_target=b,
                support_indices=expanded_support,
                seed_fraction=policy.seed_fraction,
                max_seed_amount=policy.max_seed_amount,
                min_seed_amount=policy.min_seed_amount,
            )
            full_amounts = np.asarray(
                jax.device_get(raw["condensate_amounts"][local_index]),
                dtype=np.float64,
            )
            warm_amounts = tuple(
                float(full_amounts[index])
                if math.isfinite(float(full_amounts[index]))
                and float(full_amounts[index]) > 0.0
                else float(seed)
                for index, seed in zip(seeded_support, seeded_amounts)
            )
            next_pending[source_index] = _HeadV2LayerState(
                support_indices=seeded_support,
                gas_ln_n=jnp.asarray(
                    raw["gas_log_amounts"][local_index],
                    dtype=jnp.float64,
                ),
                condensate_log_amounts=jnp.log(
                    jnp.asarray(warm_amounts, dtype=jnp.float64)
                ),
                total_gas_log_amount=jnp.asarray(
                    raw["total_gas_log_amount"][local_index],
                    dtype=jnp.float64,
                ),
                element_potential=jnp.asarray(
                    raw["element_potential"][local_index],
                    dtype=jnp.float64,
                ),
            )
        pending = next_pending
    for source_index in pending:
        records[source_index]["outcome"] = "open_at_round_limit"

    layer_results: list[CondensateEquilibriumResult] = []
    for layer_index in range(n_layers):
        lifecycle_summary = {
            "schema": "exogibbs_head_v2_fixed_support_lifecycle_v1",
            "preset": policy.name,
            "outcome": records[layer_index].get(
                "outcome", "internal_missing_outcome"
            ),
            "rounds": tuple(records[layer_index]["rounds"]),
            "compilation_seconds_total": compilation_seconds,
            "execution_seconds_total": execution_seconds,
            "diagnostic_seconds_total": diagnostic_seconds,
            "backend": backend,
            "production_preset_promoted": True,
        }
        if layer_index in gas_only_layers:
            gas_result = equilibrium(
                setup.gas_setup,
                float(temperatures[layer_index]),
                float(pressures[layer_index]),
                jnp.asarray(b, dtype=jnp.float64),
                Pref=Pref,
                options=EquilibriumOptions(),
                return_diagnostics=False,
            )
            result = _build_empty_support_gas_result(
                setup=setup,
                gas_ln_n=gas_result.ln_n,
                diagnostics={"fixed_support_v2": lifecycle_summary},
                route=HEAD_ROUTE_V2,
                selected_route="head_v2_gas_only_no_candidate",
                head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
                head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
                element_inventory_target=b,
                enable_full_condensate_budget_residual_gate=(
                    options.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
                full_condensate_budget_relative_floor=(
                    options.full_condensate_budget_relative_floor
                ),
            )
        else:
            output = last_outputs[layer_index]
            raw = output["raw"]
            local_index = int(output["local_index"])
            support = output["support_indices"]
            full_amounts = jnp.asarray(
                raw["condensate_amounts"][local_index], dtype=jnp.float64
            )
            support_amounts = full_amounts[
                jnp.asarray(support, dtype=jnp.int32)
            ]
            lifecycle_summary.update(
                {
                    "terminal_status": output["terminal_status"],
                    "terminal_status_name": TerminalStatus(
                        output["terminal_status"]
                    ).name,
                    "fixed_support_converged": output[
                        "fixed_support_converged"
                    ],
                    "support_closed": output["support_closed"],
                    "independent_kkt": output["independent_kkt"],
                    "independent_kkt_passed": output[
                        "independent_kkt_passed"
                    ],
                    "final_state_values_finite": output[
                        "final_state_values_finite"
                    ],
                }
            )
            accepted = bool(
                output["fixed_support_converged"]
                and output["support_closed"]
                and output["independent_kkt_passed"]
                and output["final_state_values_finite"]
            )
            result = build_condensate_equilibrium_result_from_solver_payload(
                setup=setup,
                gas_ln_n=raw["gas_log_amounts"][local_index],
                support_indices=support,
                support_amounts=support_amounts,
                selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
                solver_success=accepted,
                route=HEAD_ROUTE_V2,
                head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
                head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
                diagnostics={"fixed_support_v2": lifecycle_summary},
                element_inventory_target=b,
                enable_full_condensate_budget_residual_gate=(
                    options.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
                full_condensate_budget_relative_floor=(
                    options.full_condensate_budget_relative_floor
                ),
            )
        layer_results.append(result)

    gas_ln_n = jnp.stack([result.gas_ln_n for result in layer_results])
    gas_n = jnp.stack([result.gas_n for result in layer_results])
    gas_ntot = jnp.stack([result.gas_ntot for result in layer_results])
    condensate_amounts = jnp.stack(
        [result.condensate_amounts for result in layer_results]
    )
    profile_diagnostics = None
    if return_diagnostics:
        profile_diagnostics = {
            "profile_schema": "exogibbs_condensate_equilibrium_profile_v2",
            "route": HEAD_ROUTE_V2,
            "preset": policy.name,
            "method": "vmap_cold",
            "layer_count": n_layers,
            "compilation_seconds": compilation_seconds,
            "execution_seconds": execution_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "backend": backend,
            "layers": tuple(records),
        }
    return CondensateEquilibriumProfileResult(
        layers=tuple(layer_results),
        method="vmap_cold",
        diagnostics=profile_diagnostics,
        batched_arrays={
            "gas_ln_n": gas_ln_n,
            "gas_n": gas_n,
            "gas_x": gas_n / jnp.clip(gas_ntot[:, None], 1.0e-300),
            "gas_ntot": gas_ntot,
            "condensate_amounts": condensate_amounts,
        },
    )


def condensate_equilibrium(
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[CondensateEquilibriumInit] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
) -> CondensateEquilibriumResult:
    """Compute one layer through the production fixed-support v2 route."""

    opts = options or CondensateEquilibriumOptions()
    validate_condensate_chemical_setup(setup)
    _validate_options(opts)
    profile = _run_head_v2_profile(
        setup=setup,
        temperatures=np.asarray([T], dtype=np.float64),
        pressures=np.asarray([P], dtype=np.float64),
        b=b,
        Pref=Pref,
        explicit_inits=(init,),
        initializer=None,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=opts,
        return_diagnostics=opts.return_diagnostics,
    )
    return profile.layers[0]


def condensate_equilibrium_profile(
    setup: CondensateChemicalSetup,
    T: Sequence[float] | Array,
    P: Sequence[float] | Array,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[Sequence[CondensateEquilibriumInit | None]] = None,
    initializer: Optional[CondensateEquilibriumInitializer] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
    method: Optional[CondensateProfileMethod] = None,
    return_diagnostics: bool = False,
) -> CondensateEquilibriumProfileResult:
    """Compute a 1D profile through the production v2 lifecycle."""

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    opts = options or CondensateEquilibriumOptions()
    _validate_options(opts)
    n_layers = int(temperatures.shape[0])
    if init is None:
        explicit_inits: tuple[CondensateEquilibriumInit | None, ...] = (
            None,
        ) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    requested_method = method if method is not None else opts.profile_method
    if requested_method not in {None, "auto", "vmap_cold"}:
        raise ValueError(
            "head_v2 currently supports profile method 'auto' or "
            "'vmap_cold'. Support lifecycle remains outside the "
            "fixed-support solver."
        )
    return _run_head_v2_profile(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        b=b,
        Pref=Pref,
        explicit_inits=explicit_inits,
        initializer=initializer,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=opts,
        return_diagnostics=return_diagnostics or opts.return_diagnostics,
    )


def _prepare_experimental_profile_fixed_support_batch_plan(
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
    max_iter: int,
    min_seed_amount: float,
) -> _ExperimentalProfileFixedSupportBatchPlan | None:
    n_layers = int(temperatures.shape[0])
    solver_inits = []
    support_by_layer = []
    states = []
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                return None
            support_payload = (
                tuple(int(index) for index in support_indices),
                _positive_support_amounts_for_warm_start(
                    support_amounts_init,
                    min_seed_amount=min_seed_amount,
                ),
            )
        else:
            support_payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=min_seed_amount,
            )
        if support_payload is None:
            return None
        layer_support_indices, layer_support_amounts = support_payload
        if len(layer_support_indices) == 0:
            return None
        solver_init = _solver_log_state_from_condensate_init(
            initial_guess,
            setup=setup,
            support_amounts_init=layer_support_amounts,
            source="exogibbs_experimental_profile_fixed_support_batch_plan",
        )
        if solver_init is None:
            return None
        solver_inits.append(solver_init)
        support_by_layer.append(layer_support_indices)
        states.append(
            ThermoState(
                temperature=float(temperatures[layer_index]),
                ln_normalized_pressure=_ln_normalized_pressure(
                    float(pressures[layer_index]),
                    Pref,
                ),
                element_vector=jnp.asarray(b, dtype=jnp.float64),
            )
        )
    from exogibbs.optimize.fixed_support_v2_profile import (
        prepare_fixed_support_v2_buckets,
    )

    temperature_array = jnp.asarray(temperatures, dtype=jnp.float64)
    hvector_by_layer = jnp.asarray(
        setup.gas_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_by_layer.ndim != 2 or hvector_by_layer.shape[0] != n_layers:
        hvector_by_layer = jnp.stack(
            [
                jnp.asarray(
                    setup.gas_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        )
    hvector_cond_by_layer = jnp.asarray(
        setup.condensate_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_cond_by_layer.ndim != 2 or hvector_cond_by_layer.shape[0] != n_layers:
        hvector_cond_by_layer = jnp.stack(
            [
                jnp.asarray(
                    setup.condensate_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        )
    buckets = prepare_fixed_support_v2_buckets(
        init_states=tuple(solver_inits),
        support_indices_by_layer=tuple(support_by_layer),
        formula_matrix_cond=jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64),
        element_inventory_target_by_layer=jnp.stack(
            [jnp.asarray(state.element_vector, dtype=jnp.float64) for state in states]
        ),
        hvector_by_layer=hvector_by_layer,
        hvector_cond_by_layer=hvector_cond_by_layer,
        ln_normalized_pressure_by_layer=jnp.stack(
            [
                jnp.asarray(
                    state.ln_normalized_pressure,
                    dtype=jnp.float64,
                )
                for state in states
            ]
        ),
    )
    return _ExperimentalProfileFixedSupportBatchPlan(
        setup=setup,
        buckets=buckets,
        formula_matrix=jnp.asarray(setup.formula_matrix, dtype=jnp.float64),
        max_iter=max_iter,
        n_layers=n_layers,
        condensate_count=len(setup.condensate_species),
        bucket_layer_index_arrays=tuple(
            jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
            for bucket in buckets
        ),
        temperatures=temperature_array,
    )


def prepare_experimental_profile_fixed_support_batch_plan(
    setup: CondensateChemicalSetup,
    T: Sequence[float] | Array,
    P: Sequence[float] | Array,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[Sequence[CondensateEquilibriumInit | None]] = None,
    initializer: Optional[CondensateEquilibriumInitializer] = None,
    max_iter: int = 100,
    min_seed_amount: float = 1.0e-300,
) -> ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Prepare a reusable experimental fixed-support batch profile plan.

    The returned plan can be run repeatedly with
    :func:`run_experimental_profile_fixed_support_v2_batch_plan` to avoid
    rebuilding support buckets and thermochemical vectors for every call.
    """

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    if isinstance(max_iter, bool) or int(max_iter) <= 0:
        raise ValueError("max_iter must be positive.")
    if not math.isfinite(float(min_seed_amount)) or float(
        min_seed_amount
    ) <= 0.0:
        raise ValueError("min_seed_amount must be finite and positive.")
    n_layers = int(temperatures.shape[0])
    if init is None:
        explicit_inits: tuple[CondensateEquilibriumInit | None, ...] = (
            None,
        ) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    plan = _prepare_experimental_profile_fixed_support_batch_plan(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        b=b,
        Pref=Pref,
        explicit_inits=explicit_inits,
        initializer=initializer,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        max_iter=int(max_iter),
        min_seed_amount=float(min_seed_amount),
    )
    if plan is None:
        raise ValueError(
            "experimental fixed-support batch plan requires non-empty explicit "
            "support and solver log-state initialization for every layer."
        )
    return plan


def run_experimental_profile_fixed_support_v2_batch_plan(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    *,
    element_inventory_target: Optional[Array] = None,
    config: Optional[Any] = None,
    budget_relative_floor: float = 1.0e-6,
    support_closure_tolerance: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Run an existing prepared profile plan through fixed-support v2.

    This route is explicitly opt-in. It reports fixed-support convergence and
    full inactive-condensate support closure separately, and does not alter the
    production preset.
    """

    from exogibbs.optimize.fixed_support_v2.types import FixedSupportV2Config
    from exogibbs.optimize.fixed_support_v2_profile import run_prepared_profile_v2

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if plan.temperatures is None:
        raise ValueError(
            "The prepared plan has no temperature array required by v2. "
            "Rebuild it with prepare_experimental_profile_fixed_support_batch_plan."
        )
    active_config = FixedSupportV2Config() if config is None else config
    if not isinstance(active_config, FixedSupportV2Config):
        raise TypeError("config must be a FixedSupportV2Config.")
    if config is None:
        # The integrated public v2 route includes the M2 restoration solver.
        # Component tests may still opt out explicitly with zero calls.
        active_config = replace(
            active_config,
            limits=replace(active_config.limits, max_restoration_calls=2),
        )
    active_config = replace(
        active_config,
        limits=replace(
            active_config.limits,
            max_normal_iterations=plan.max_iter,
        ),
    )

    buckets = plan.buckets
    if element_inventory_target is not None:
        target = jnp.asarray(element_inventory_target, dtype=jnp.float64)
        n_elements = int(plan.formula_matrix.shape[0])
        if target.ndim == 1:
            if target.shape[0] != n_elements:
                raise ValueError(
                    "element_inventory_target length must match elements."
                )
            target = jnp.broadcast_to(target, (plan.n_layers, n_elements))
        elif target.ndim == 2:
            if target.shape != (plan.n_layers, n_elements):
                raise ValueError(
                    "element_inventory_target must have shape "
                    f"({plan.n_layers}, {n_elements})."
                )
        else:
            raise ValueError(
                "element_inventory_target must be one- or two-dimensional."
            )
        buckets = tuple(
            replace(
                bucket,
                element_inventory_target=target[
                    jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
                ],
            )
            for bucket in plan.buckets
        )

    temperatures = jnp.asarray(plan.temperatures, dtype=jnp.float64)
    hcond_full = jnp.asarray(
        plan.setup.condensate_setup.hvector_func(temperatures),
        dtype=jnp.float64,
    )
    expected_shape = (plan.n_layers, plan.condensate_count)
    if hcond_full.shape != expected_shape:
        hcond_full = jax.vmap(plan.setup.condensate_setup.hvector_func)(
            temperatures
        )
    condensate_metadata = getattr(plan.setup.condensate_setup, "metadata", {})
    validity_upper = condensate_metadata.get("temperature_validity_upper")
    if validity_upper is None:
        condensate_valid_mask = jnp.ones(expected_shape, dtype=bool)
    else:
        upper = jnp.asarray(validity_upper, dtype=jnp.float64)
        if upper.shape != (plan.condensate_count,):
            raise ValueError(
                "temperature_validity_upper must have one value per condensate."
            )
        condensate_valid_mask = temperatures[:, None] <= upper[None, :]

    return run_prepared_profile_v2(
        buckets=buckets,
        formula_matrix=plan.formula_matrix,
        formula_matrix_cond_full=plan.setup.formula_matrix_cond,
        condensate_standard_source_full=hcond_full,
        condensate_valid_mask=condensate_valid_mask,
        layer_count=plan.n_layers,
        condensate_count=plan.condensate_count,
        config=active_config,
        budget_relative_floor=budget_relative_floor,
        support_closure_tolerance=support_closure_tolerance,
    )


__all__ = (
    "CONDENSATE_HEAD_V2_ROUTE_NAME",
    "CONDENSATE_HEAD_V2_ROUTE_VERSION",
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "HEAD_ROUTE_V2",
    "CondensateChemicalSetup",
    "CondensateEquilibriumInit",
    "CondensateEquilibriumInitRequest",
    "CondensateEquilibriumInitializer",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumProfileResult",
    "CondensateEquilibriumResult",
    "CondensateFixedSupportV2Preset",
    "CondensateProfileMethod",
    "DefaultCondensateEquilibriumInitializer",
    "ExperimentalCondensateProfileFixedSupportBatchPlan",
    "build_condensate_chemical_setup",
    "build_condensate_equilibrium_result_from_solver_payload",
    "condensate_equilibrium",
    "condensate_equilibrium_profile",
    "prepare_experimental_profile_fixed_support_batch_plan",
    "run_experimental_profile_fixed_support_v2_batch_plan",
    "validate_condensate_chemical_setup",
)
