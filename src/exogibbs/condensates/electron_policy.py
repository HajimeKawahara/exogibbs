"""FastChem4-style diagnostic electron policy for gas log-density boundaries.

The helper separates the electron row from the normal neutral-element gas
boundary solve and reconstructs the electron density from a one-dimensional
charge-balance equation. It is explicit-import only, diagnostic, default-off,
and does not import FastChem4 or pyfastchem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import logsumexp

from exogibbs.condensates.native_bundle import (
    validate_native_bundle_provenance,
)


@dataclass(frozen=True)
class FastChem4StyleElectronPolicyReport:
    """Diagnostic report for the separated electron-row gas boundary solve."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    provenance: Mapping[str, str]
    solver_success: bool
    neutral_solver_success: bool
    charge_balance_success: bool
    neutral_total_nfev: int
    neutral_cost: float
    row_count: int
    neutral_element_count: int
    gas_species_count: int
    electron_row_index: int
    ion_species_count: int
    ln_species_density: tuple[float, ...]
    neutral_element_log_density: tuple[float, ...]
    electron_element_log_density: float
    element_inventory: tuple[float, ...]
    max_neutral_log_budget_residual: float
    max_neutral_absolute_budget_residual: float
    electron_charge_residual: float
    electron_charge_residual_abs: float
    electron_log_density: float
    electron_density: float
    cation_charge_inventory: float
    anion_plus_electron_charge_inventory: float
    charge_bracket_lower_value: float
    charge_bracket_upper_value: float
    finite: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _log_inventory_for_row(
    row: np.ndarray,
    ln_species: np.ndarray,
    floor_value: float,
) -> float:
    positive = row > 0.0
    if not np.any(positive):
        return math.log(float(floor_value))
    return float(logsumexp(ln_species[positive], b=row[positive]))


def _charge_balance_root(
    *,
    neutral_log_basis: np.ndarray,
    electron_row: np.ndarray,
    floor_value: float,
    electron_upper_log_density: float,
) -> tuple[float, bool, float, float]:
    active = electron_row != 0.0
    if not np.any(active):
        return math.log(float(floor_value)), True, 0.0, 0.0

    def charge_sum(y_electron: float) -> float:
        exponent = neutral_log_basis[active] + electron_row[active] * y_electron
        return float(
            np.sum(electron_row[active] * np.exp(np.clip(exponent, -745.0, 700.0)))
        )

    lower = math.log(float(floor_value))
    upper = float(electron_upper_log_density)
    lower_value = charge_sum(lower)
    upper_value = charge_sum(upper)
    if lower_value == 0.0:
        return lower, True, lower_value, upper_value
    if upper_value == 0.0:
        return upper, True, lower_value, upper_value
    if lower_value * upper_value < 0.0:
        root = float(brentq(charge_sum, lower, upper, maxiter=200, xtol=1.0e-12))
        return root, True, lower_value, upper_value
    root = lower if abs(lower_value) <= abs(upper_value) else upper
    return root, False, lower_value, upper_value


def solve_fastchem4_style_electron_policy_boundary(
    *,
    explicit_opt_in: bool,
    formula_matrix: Sequence[Sequence[float]],
    element_budget: Sequence[float],
    mass_action_constants: Sequence[float],
    electron_row_index: int = 0,
    provenance: Mapping[str, str] | None = None,
    max_nfev: int = 1000,
    floor_value: float = 1.0e-300,
    electron_upper_log_density: float = 80.0,
    neutral_xtol: float = 1.0e-10,
    neutral_ftol: float = 1.0e-10,
    neutral_gtol: float = 1.0e-10,
    success_log_tolerance: float = 1.0e-6,
    success_charge_tolerance: float = 1.0e-8,
) -> FastChem4StyleElectronPolicyReport:
    """Solve a diagnostic gas boundary with FastChem4-style electron handling."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for electron-policy diagnostics.")
    if int(max_nfev) <= 0:
        raise ValueError("max_nfev must be positive.")
    if float(floor_value) <= 0.0 or not math.isfinite(float(floor_value)):
        raise ValueError("floor_value must be positive and finite.")
    if float(electron_upper_log_density) <= math.log(float(floor_value)):
        raise ValueError("electron_upper_log_density must exceed log(floor_value).")
    if float(success_log_tolerance) < 0.0:
        raise ValueError("success_log_tolerance must be non-negative.")
    if float(success_charge_tolerance) < 0.0:
        raise ValueError("success_charge_tolerance must be non-negative.")

    field_provenance = validate_native_bundle_provenance(provenance)
    matrix = _as_matrix(formula_matrix, "formula_matrix")
    budget = _as_vector(element_budget, "element_budget")
    mac = _as_vector(mass_action_constants, "mass_action_constants")
    if matrix.shape[0] != budget.shape[0]:
        raise ValueError("element_budget length must match formula_matrix rows.")
    if matrix.shape[1] != mac.shape[0]:
        raise ValueError("mass_action_constants length must match formula_matrix columns.")
    charge_index = int(electron_row_index)
    if charge_index < 0 or charge_index >= matrix.shape[0]:
        raise ValueError("electron_row_index is out of range.")
    if np.any(budget < 0.0):
        raise ValueError("element_budget must be non-negative.")

    neutral_indices = np.asarray(
        [index for index in range(matrix.shape[0]) if index != charge_index],
        dtype=np.int64,
    )
    if neutral_indices.size == 0:
        raise ValueError("at least one neutral element row is required.")
    electron_row = matrix[charge_index]
    neutral_matrix = matrix[neutral_indices]
    neutral_budget = budget[neutral_indices]
    neutral_y0 = np.log(np.maximum(neutral_budget, float(floor_value)))

    def neutral_log_basis(y_neutral: np.ndarray) -> np.ndarray:
        return mac + neutral_matrix.T.dot(y_neutral)

    def residual(y_neutral: np.ndarray) -> np.ndarray:
        basis = neutral_log_basis(y_neutral)
        y_electron, _success, _lower, _upper = _charge_balance_root(
            neutral_log_basis=basis,
            electron_row=electron_row,
            floor_value=float(floor_value),
            electron_upper_log_density=float(electron_upper_log_density),
        )
        ln_species = basis + electron_row * y_electron
        return np.asarray(
            [
                _log_inventory_for_row(matrix[row_index], ln_species, float(floor_value))
                - math.log(max(float(neutral_budget[local_index]), float(floor_value)))
                for local_index, row_index in enumerate(neutral_indices)
            ],
            dtype=np.float64,
        )

    solution = least_squares(
        residual,
        neutral_y0,
        max_nfev=int(max_nfev),
        xtol=float(neutral_xtol),
        ftol=float(neutral_ftol),
        gtol=float(neutral_gtol),
        x_scale="jac",
    )
    y_neutral = np.asarray(solution.x, dtype=np.float64)
    basis = neutral_log_basis(y_neutral)
    y_electron, charge_success, bracket_lower, bracket_upper = _charge_balance_root(
        neutral_log_basis=basis,
        electron_row=electron_row,
        floor_value=float(floor_value),
        electron_upper_log_density=float(electron_upper_log_density),
    )
    ln_species = basis + electron_row * y_electron
    species_density = np.exp(np.clip(ln_species, -745.0, 700.0))
    inventory = matrix.dot(species_density)
    neutral_absolute_budget_residual = np.abs(inventory[neutral_indices] - neutral_budget)
    neutral_log_residual = residual(y_neutral)
    cation_mask = electron_row < 0.0
    anion_mask = electron_row > 0.0
    cation_charge = float(np.sum((-electron_row[cation_mask]) * species_density[cation_mask]))
    anion_plus_electron_charge = float(np.sum(electron_row[anion_mask] * species_density[anion_mask]))
    electron_charge_residual = float(inventory[charge_index])
    electron_charge_abs = abs(electron_charge_residual)
    max_neutral_log = float(np.max(np.abs(neutral_log_residual)))
    max_neutral_abs = float(np.max(neutral_absolute_budget_residual))
    finite = bool(
        np.all(np.isfinite(ln_species))
        and np.all(np.isfinite(y_neutral))
        and np.all(np.isfinite(inventory))
        and math.isfinite(float(y_electron))
        and math.isfinite(max_neutral_log)
        and math.isfinite(electron_charge_residual)
    )
    return FastChem4StyleElectronPolicyReport(
        report_schema="exogibbs_fastchem4_style_electron_policy_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        provenance=field_provenance,
        solver_success=bool(
            solution.success
            and charge_success
            and finite
            and max_neutral_log <= float(success_log_tolerance)
            and electron_charge_abs <= float(success_charge_tolerance)
        ),
        neutral_solver_success=bool(solution.success),
        charge_balance_success=bool(charge_success),
        neutral_total_nfev=int(solution.nfev),
        neutral_cost=float(solution.cost),
        row_count=int(matrix.shape[0]),
        neutral_element_count=int(neutral_indices.size),
        gas_species_count=int(matrix.shape[1]),
        electron_row_index=charge_index,
        ion_species_count=int(np.count_nonzero(electron_row) - int(electron_row[charge_index] != 0.0)),
        ln_species_density=tuple(float(value) for value in ln_species),
        neutral_element_log_density=tuple(float(value) for value in y_neutral),
        electron_element_log_density=float(y_electron),
        element_inventory=tuple(float(value) for value in inventory),
        max_neutral_log_budget_residual=max_neutral_log,
        max_neutral_absolute_budget_residual=max_neutral_abs,
        electron_charge_residual=electron_charge_residual,
        electron_charge_residual_abs=electron_charge_abs,
        electron_log_density=float(ln_species[charge_index]),
        electron_density=float(math.exp(min(float(ln_species[charge_index]), 700.0))),
        cation_charge_inventory=cation_charge,
        anion_plus_electron_charge_inventory=anion_plus_electron_charge,
        charge_bracket_lower_value=float(bracket_lower),
        charge_bracket_upper_value=float(bracket_upper),
        finite=finite,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "FastChem4StyleElectronPolicyReport",
    "solve_fastchem4_style_electron_policy_boundary",
)
