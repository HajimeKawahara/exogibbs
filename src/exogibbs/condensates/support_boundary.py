"""Native condensate support-boundary construction.

This module converts explicit gas and condensate arrays into a validated
support boundary for the condensate lifecycle. It performs no solver updates
and does not consume FastChem or result-artifact values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.optimize.pdipm_rgie_cond import FORBIDDEN_PROVENANCE


FORBIDDEN_PROVENANCE_TOKENS = tuple(sorted(FORBIDDEN_PROVENANCE))


@dataclass(frozen=True)
class CondensateSupportBoundary:
    """Validated native support boundary for condensate lifecycle stages."""

    boundary_schema: str
    ln_nk: tuple[float, ...]
    ln_ntot: float
    support_indices: tuple[int, ...]
    support_amounts: tuple[float, ...]
    ln_mk: tuple[float, ...]
    full_condensate_amounts: tuple[float, ...]
    formula_matrix: tuple[tuple[float, ...], ...]
    formula_matrix_cond: tuple[tuple[float, ...], ...]
    formula_matrix_cond_active: tuple[tuple[float, ...], ...]
    element_inventory_target: tuple[float, ...]
    gas_element_inventory: tuple[float, ...]
    condensate_element_inventory: tuple[float, ...]
    total_element_inventory: tuple[float, ...]
    budget_residual: tuple[float, ...]
    budget_residual_l2: float
    budget_residual_max_abs: float
    amount_floor: float
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    field_provenance: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "boundary_schema": self.boundary_schema,
            "ln_nk": self.ln_nk,
            "ln_mk": self.ln_mk,
            "ln_ntot": self.ln_ntot,
            "support_indices": self.support_indices,
            "support_amounts": self.support_amounts,
            "full_condensate_amounts": self.full_condensate_amounts,
            "formula_matrix": self.formula_matrix,
            "formula_matrix_cond": self.formula_matrix_cond,
            "formula_matrix_cond_active": self.formula_matrix_cond_active,
            "element_inventory_target": self.element_inventory_target,
            "gas_element_inventory": self.gas_element_inventory,
            "condensate_element_inventory": self.condensate_element_inventory,
            "total_element_inventory": self.total_element_inventory,
            "budget_residual": self.budget_residual,
            "budget_residual_l2": self.budget_residual_l2,
            "budget_residual_max_abs": self.budget_residual_max_abs,
            "amount_floor": self.amount_floor,
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


def _as_support_indices(values: Sequence[int], condensate_count: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("support_indices must be a one-dimensional vector.")
    if indices.size and np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("support_indices must be unique.")
    if np.any(indices < 0) or np.any(indices >= condensate_count):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    return indices


def _tuple_vector(array: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in array)


def _tuple_matrix(array: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in array)


def _full_condensate_amounts(
    *,
    support_indices: np.ndarray,
    support_amounts: np.ndarray,
    condensate_count: int,
) -> np.ndarray:
    full = np.zeros((condensate_count,), dtype=np.float64)
    if support_indices.shape[0] != support_amounts.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    full[support_indices] = support_amounts
    return full


def build_condensate_support_boundary(
    *,
    explicit_opt_in: bool,
    ln_nk: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    amount_floor: float = 1.0e-300,
    field_provenance: Mapping[str, str] | None = None,
) -> CondensateSupportBoundary:
    """Build a native condensate support boundary from explicit arrays."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for condensate support boundaries.")
    provenance = _validate_provenance(field_provenance)
    q = _as_vector(ln_nk, "ln_nk")
    support_amount_array = _as_vector(support_amounts, "support_amounts")
    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    floor_value = float(amount_floor)
    if not np.isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError("amount_floor must be finite and positive.")
    if np.any(support_amount_array <= 0.0):
        raise ValueError("support_amounts must be positive.")
    support = _as_support_indices(support_indices, ac.shape[1])

    if support.shape[0] != support_amount_array.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match ln_nk length.")
    if ag.shape[0] != ac.shape[0]:
        raise ValueError("formula_matrix_cond row count must match formula_matrix.")
    if ag.shape[0] != target.shape[0]:
        raise ValueError("element_inventory_target length must match formula_matrix rows.")

    full_amounts = _full_condensate_amounts(
        support_indices=support,
        support_amounts=support_amount_array,
        condensate_count=ac.shape[1],
    )
    active_ac = ac[:, support] if support.size else np.zeros((ac.shape[0], 0), dtype=np.float64)
    ln_mk = np.log(np.maximum(support_amount_array, floor_value))
    gas_density = np.exp(np.clip(q, -745.0, 700.0))
    gas_inventory = ag @ gas_density
    condensate_inventory = ac @ full_amounts
    total_inventory = gas_inventory + condensate_inventory
    budget_residual = total_inventory - target
    ln_ntot = float(np.log(np.sum(gas_density)))
    if not np.isfinite(ln_ntot):
        raise ValueError("computed ln_ntot must be finite.")

    return CondensateSupportBoundary(
        boundary_schema="exogibbs_condensate_support_boundary_v1",
        ln_nk=_tuple_vector(q),
        ln_ntot=ln_ntot,
        support_indices=tuple(int(value) for value in support),
        support_amounts=_tuple_vector(support_amount_array),
        ln_mk=_tuple_vector(ln_mk),
        full_condensate_amounts=_tuple_vector(full_amounts),
        formula_matrix=_tuple_matrix(ag),
        formula_matrix_cond=_tuple_matrix(ac),
        formula_matrix_cond_active=_tuple_matrix(active_ac),
        element_inventory_target=_tuple_vector(target),
        gas_element_inventory=_tuple_vector(gas_inventory),
        condensate_element_inventory=_tuple_vector(condensate_inventory),
        total_element_inventory=_tuple_vector(total_inventory),
        budget_residual=_tuple_vector(budget_residual),
        budget_residual_l2=float(np.linalg.norm(budget_residual)),
        budget_residual_max_abs=float(np.max(np.abs(budget_residual)))
        if budget_residual.size
        else 0.0,
        amount_floor=floor_value,
        diagnostic_only=False,
        default_off=False,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        field_provenance={
            "ln_nk": provenance.get("ln_nk", "exogibbs_native"),
            "support_amounts": provenance.get("support_amounts", "exogibbs_native"),
            "support_indices": provenance.get("support_indices", "exogibbs_native"),
        },
    )


__all__ = (
    "CondensateSupportBoundary",
    "build_condensate_support_boundary",
)
