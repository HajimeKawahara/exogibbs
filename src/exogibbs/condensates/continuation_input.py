"""Native condensate continuation input construction.

This module builds the explicit native state passed from the condensate
equilibrium lifecycle into the PD-IPM/R-GIE continuation machinery. It does not
call FastChem, does not read result artifacts, and does not run solver updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.optimize.pdipm_rgie_cond import (
    FORBIDDEN_PROVENANCE,
    PdipmRgieCondensateState,
    build_pdipm_rgie_condensate_state,
)


FORBIDDEN_PROVENANCE_TOKENS = tuple(sorted(FORBIDDEN_PROVENANCE))


@dataclass(frozen=True)
class CondensateContinuationInput:
    """Validated native input for condensate continuation."""

    input_schema: str
    state: PdipmRgieCondensateState
    support_indices: tuple[int, ...]
    support_species_names: tuple[str, ...] | None
    formula_matrix: tuple[tuple[float, ...], ...]
    formula_matrix_cond_active: tuple[tuple[float, ...], ...]
    element_inventory_target: tuple[float, ...]
    external_condensate_budget: tuple[float, ...]
    gas_stationarity_source: tuple[float, ...]
    condensate_standard_source: tuple[float, ...]
    gas_lambda_gauge_residual_l2: float
    gas_lambda_gauge_residual_max_abs: float
    inferred_rho_from_epsilon: bool
    dual_initialization_policy: str
    dual_push_floor: float | None
    dual_push_applied_count: int
    dual_push_max_log_delta: float
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    field_provenance: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_schema": self.input_schema,
            "state": self.state.as_dict(),
            "support_indices": self.support_indices,
            "support_species_names": self.support_species_names,
            "formula_matrix": self.formula_matrix,
            "formula_matrix_cond_active": self.formula_matrix_cond_active,
            "element_inventory_target": self.element_inventory_target,
            "external_condensate_budget": self.external_condensate_budget,
            "gas_stationarity_source": self.gas_stationarity_source,
            "condensate_standard_source": self.condensate_standard_source,
            "gas_lambda_gauge_residual_l2": self.gas_lambda_gauge_residual_l2,
            "gas_lambda_gauge_residual_max_abs": self.gas_lambda_gauge_residual_max_abs,
            "inferred_rho_from_epsilon": self.inferred_rho_from_epsilon,
            "dual_initialization_policy": self.dual_initialization_policy,
            "dual_push_floor": self.dual_push_floor,
            "dual_push_applied_count": self.dual_push_applied_count,
            "dual_push_max_log_delta": self.dual_push_max_log_delta,
            "diagnostic_only": self.diagnostic_only,
            "default_off": self.default_off,
            "production_behavior_change": self.production_behavior_change,
            "production_return_signature_change": self.production_return_signature_change,
            "preset_default_wiring_change": self.preset_default_wiring_change,
            "fastchem4_trace_public_runtime_constructor_inputs_used": (
                self.fastchem4_trace_public_runtime_constructor_inputs_used
            ),
            "field_provenance": self.field_provenance,
        }


def _validate_provenance(field_provenance: Mapping[str, str] | None) -> dict[str, str]:
    provenance = {} if field_provenance is None else dict(field_provenance)
    forbidden: list[str] = []
    for field_name, value in provenance.items():
        value_text = str(value)
        if any(token in value_text for token in FORBIDDEN_PROVENANCE_TOKENS):
            forbidden.append(f"{field_name}={value_text}")
    if forbidden:
        raise ValueError(f"field_provenance contains forbidden values: {forbidden}")
    return provenance


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_support_indices(values: Sequence[int], active_count: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("support_indices must be a one-dimensional vector.")
    if indices.shape[0] != active_count:
        raise ValueError("support_indices length must match ln_mk length.")
    if indices.size and np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("support_indices must be unique.")
    if np.any(indices < 0):
        raise ValueError("support_indices must be non-negative.")
    return indices


def _tuple_vector(array: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in array)


def _tuple_matrix(array: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in array)


def _rho_from_inputs(
    *,
    ln_mk: np.ndarray,
    rho: Sequence[float] | None,
    eta: Sequence[float] | None,
    epsilon: float | None,
    dual_push_floor: float | None,
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    if rho is not None:
        rho_array = _as_vector(rho, "rho")
        if rho_array.shape != ln_mk.shape:
            raise ValueError("rho length must match ln_mk length.")
        eta_array = np.exp(rho_array)
        return rho_array, eta_array, False, 0, 0.0
    if eta is not None:
        eta_array = _as_vector(eta, "eta")
        if eta_array.shape != ln_mk.shape:
            raise ValueError("eta length must match ln_mk length.")
        if np.any(eta_array <= 0.0):
            raise ValueError("eta must contain positive values.")
        return np.log(eta_array), eta_array, False, 0, 0.0
    if epsilon is None:
        raise ValueError("one of rho, eta, or epsilon must be provided.")
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value):
        raise ValueError("epsilon must be finite.")
    eta_array = np.exp(epsilon_value - ln_mk)
    applied_count = 0
    max_log_delta = 0.0
    if dual_push_floor is not None:
        floor = float(dual_push_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("dual_push_floor must be finite and positive when provided.")
        pushed_eta = np.maximum(eta_array, floor)
        pushed = pushed_eta > eta_array
        applied_count = int(np.count_nonzero(pushed))
        if applied_count:
            max_log_delta = float(
                np.max(np.log(pushed_eta[pushed]) - np.log(eta_array[pushed]))
            )
        eta_array = pushed_eta
    return np.log(eta_array), eta_array, True, applied_count, max_log_delta


def build_condensate_continuation_input(
    *,
    explicit_opt_in: bool,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    element_potential: Sequence[float],
    support_indices: Sequence[int],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    ln_ntot: float | None = None,
    rho: Sequence[float] | None = None,
    eta: Sequence[float] | None = None,
    epsilon: float | None = None,
    dual_initialization_policy: str = "centered_from_epsilon",
    dual_push_floor: float | None = None,
    condensate_species_names: Sequence[str] | None = None,
    field_provenance: Mapping[str, str] | None = None,
) -> CondensateContinuationInput:
    """Build validated native input for PD-IPM/R-GIE condensate continuation."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for condensate continuation input.")
    provenance = _validate_provenance(field_provenance)
    q = _as_vector(ln_nk, "ln_nk")
    r = _as_vector(ln_mk, "ln_mk")
    lam = _as_vector(element_potential, "element_potential")
    support = _as_support_indices(support_indices, r.shape[0])
    support_names: tuple[str, ...] | None = None
    if condensate_species_names is not None:
        names = tuple(str(name) for name in condensate_species_names)
        if support.size and int(np.max(support)) >= len(names):
            raise ValueError("condensate_species_names must cover support_indices.")
        support_names = tuple(names[int(index)] for index in support)
    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    cond_source = _as_vector(condensate_standard_source, "condensate_standard_source")

    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match ln_nk length.")
    if ag.shape[0] != lam.shape[0]:
        raise ValueError("formula_matrix row count must match element_potential length.")
    if ag.shape[0] != target.shape[0]:
        raise ValueError("formula_matrix row count must match element_inventory_target length.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element_inventory_target.")
    if ac.shape[0] != ag.shape[0]:
        raise ValueError("formula_matrix_cond_active row count must match formula_matrix.")
    if ac.shape[1] != r.shape[0]:
        raise ValueError("formula_matrix_cond_active column count must match ln_mk length.")
    if gas_source.shape[0] != q.shape[0]:
        raise ValueError("gas_stationarity_source length must match ln_nk length.")
    if cond_source.shape[0] != r.shape[0]:
        raise ValueError("condensate_standard_source length must match ln_mk length.")

    dual_policy = str(dual_initialization_policy)
    if dual_policy not in {"centered_from_epsilon", "ipopt_push_floor"}:
        raise ValueError(
            "dual_initialization_policy must be centered_from_epsilon or ipopt_push_floor."
        )
    effective_dual_floor = (
        None if dual_policy == "centered_from_epsilon" else dual_push_floor
    )
    rho_array, eta_array, inferred_rho, dual_push_count, dual_push_max_log_delta = (
        _rho_from_inputs(
            ln_mk=r,
            rho=rho,
            eta=eta,
            epsilon=epsilon,
            dual_push_floor=effective_dual_floor,
        )
    )
    state = build_pdipm_rgie_condensate_state(
        ln_nk=_tuple_vector(q),
        ln_mk=_tuple_vector(r),
        element_potential=_tuple_vector(lam),
        ln_ntot=ln_ntot,
        rho=_tuple_vector(rho_array),
        eta=_tuple_vector(eta_array),
        field_provenance={
            "ln_nk": provenance.get("ln_nk", "exogibbs_native"),
            "ln_mk": provenance.get("ln_mk", "exogibbs_native"),
            "element_potential": provenance.get("element_potential", "exogibbs_native"),
            "rho": provenance.get("rho", "exogibbs_native_derived"),
            "eta": provenance.get("eta", "exogibbs_native_derived"),
        },
    )
    gauge_residual = q + gas_source - ag.T @ lam
    return CondensateContinuationInput(
        input_schema="exogibbs_condensate_continuation_input_v1",
        state=state,
        support_indices=tuple(int(value) for value in support),
        support_species_names=support_names,
        formula_matrix=_tuple_matrix(ag),
        formula_matrix_cond_active=_tuple_matrix(ac),
        element_inventory_target=_tuple_vector(target),
        external_condensate_budget=_tuple_vector(external_budget),
        gas_stationarity_source=_tuple_vector(gas_source),
        condensate_standard_source=_tuple_vector(cond_source),
        gas_lambda_gauge_residual_l2=float(np.linalg.norm(gauge_residual)),
        gas_lambda_gauge_residual_max_abs=float(np.max(np.abs(gauge_residual)))
        if gauge_residual.size
        else 0.0,
        inferred_rho_from_epsilon=inferred_rho,
        dual_initialization_policy=dual_policy,
        dual_push_floor=None
        if effective_dual_floor is None
        else float(effective_dual_floor),
        dual_push_applied_count=dual_push_count,
        dual_push_max_log_delta=dual_push_max_log_delta,
        diagnostic_only=False,
        default_off=False,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        field_provenance=state.field_provenance,
    )


__all__ = (
    "CondensateContinuationInput",
    "build_condensate_continuation_input",
)
