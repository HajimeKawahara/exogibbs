"""Explicit diagnostic depleted-budget gas refresh helpers.

The helpers in this module mirror the structural role of a condensate-stage
gas refresh without importing FastChem4, calling pyfastchem, or wiring any
default production path. They consume only ExoGibbs-native arrays and are
intended for explicit opt-in experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ThermoState
from exogibbs.condensates.native_bundle import (
    validate_native_bundle_provenance,
)
from exogibbs.condensates.electron_policy import (
    solve_fastchem4_style_electron_policy_boundary,
)
from exogibbs.optimize.minimize_cond import (
    CondensateEquilibriumInit,
    solve_gas_equilibrium_with_duals,
)


@dataclass(frozen=True)
class DepletedGasRefreshReport:
    """Report for an explicit depleted-budget gas refresh."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    provenance: Mapping[str, str]
    support_indices: tuple[int, ...]
    condensate_amounts: tuple[float, ...]
    condensate_inventory: tuple[float, ...]
    original_element_budget: tuple[float, ...]
    depleted_element_budget: tuple[float, ...]
    negative_depleted_budget_inf: float
    clamped_negative_budget_count: int
    gas_refresh_policy: str
    electron_policy_used: bool
    electron_policy_solver_success: bool
    electron_policy_charge_residual_abs: float
    electron_policy_max_neutral_log_budget_residual: float
    electron_policy_electron_density: float
    gas_solver_success: bool
    gas_solver_iterations: int
    gas_solver_final_residual: float
    ln_nk: tuple[float, ...]
    ln_ntot: float
    refreshed_gas_inventory: tuple[float, ...]
    depleted_budget_residual_inf: float
    finite: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_1d(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _active_condensate_amounts(
    *,
    formula_matrix_cond: np.ndarray,
    support_indices: np.ndarray,
    ln_mk: np.ndarray,
) -> np.ndarray:
    if ln_mk.shape[0] == formula_matrix_cond.shape[1]:
        return np.exp(ln_mk[support_indices])
    if ln_mk.shape[0] == support_indices.shape[0]:
        return np.exp(ln_mk)
    raise ValueError(
        "ln_mk must have either full condensate length or active support length."
    )


def build_depleted_gas_refresh_init(
    *,
    explicit_opt_in: bool,
    state: ThermoState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    hvector_func: Callable[[Any], Any],
    support_indices: Sequence[int],
    ln_mk: Sequence[float],
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 1000,
    floor_value: float = 0.0,
    gas_refresh_policy: str = "native_gas_solver",
    electron_row_index: int | None = None,
    mass_action_constants_func: Callable[[Any], Any] | None = None,
    field_provenance: Mapping[str, str] | None = None,
) -> tuple[CondensateEquilibriumInit, DepletedGasRefreshReport]:
    """Build a gas-refreshed initialization from a native depleted budget."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for depleted gas refresh.")
    provenance = validate_native_bundle_provenance(field_provenance)
    ag = np.asarray(formula_matrix, dtype=np.float64)
    ac = np.asarray(formula_matrix_cond, dtype=np.float64)
    budget = _as_1d("state.element_vector", np.asarray(state.element_vector, dtype=np.float64))
    indices = np.asarray(support_indices, dtype=np.int64)
    ln_m = _as_1d("ln_mk", ln_mk)
    if ag.ndim != 2:
        raise ValueError("formula_matrix must be two-dimensional.")
    if ac.ndim != 2:
        raise ValueError("formula_matrix_cond must be two-dimensional.")
    if ag.shape[0] != budget.shape[0] or ac.shape[0] != budget.shape[0]:
        raise ValueError("formula matrices must share the element budget row count.")
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if indices.size and (np.any(indices < 0) or np.any(indices >= ac.shape[1])):
        raise ValueError("support_indices contains an out-of-range condensate index.")
    if len(set(int(index) for index in indices.tolist())) != indices.shape[0]:
        raise ValueError("support_indices must not contain duplicates.")
    if not np.all(np.isfinite(ag)) or not np.all(np.isfinite(ac)):
        raise ValueError("formula matrices must contain finite values.")
    if float(floor_value) < 0.0 or not np.isfinite(float(floor_value)):
        raise ValueError("floor_value must be finite and non-negative.")
    if gas_refresh_policy not in {
        "native_gas_solver",
        "fastchem4_style_electron_policy",
    }:
        raise ValueError("gas_refresh_policy is not supported.")

    active_amounts = _active_condensate_amounts(
        formula_matrix_cond=ac,
        support_indices=indices,
        ln_mk=ln_m,
    )
    active_ac = ac[:, indices] if indices.size else ac[:, :0]
    condensate_inventory = active_ac.dot(active_amounts)
    raw_depleted = budget - condensate_inventory
    negative_part = np.minimum(raw_depleted, 0.0)
    negative_inf = float(np.max(np.abs(negative_part))) if negative_part.size else 0.0
    depleted = np.maximum(raw_depleted, float(floor_value))
    clamped_count = int(np.sum(raw_depleted < float(floor_value)))
    depleted_state = ThermoState(
        temperature=state.temperature,
        ln_normalized_pressure=state.ln_normalized_pressure,
        element_vector=jnp.asarray(depleted, dtype=jnp.float64),
    )
    electron_policy_used = gas_refresh_policy == "fastchem4_style_electron_policy"
    electron_policy_success = False
    electron_policy_charge_residual_abs = float("nan")
    electron_policy_max_neutral_log_budget_residual = float("nan")
    electron_policy_electron_density = float("nan")
    if electron_policy_used:
        if electron_row_index is None:
            raise ValueError(
                "electron_row_index is required for fastchem4_style_electron_policy."
            )
        mass_action_func = (
            hvector_func if mass_action_constants_func is None else mass_action_constants_func
        )
        electron_report = solve_fastchem4_style_electron_policy_boundary(
            explicit_opt_in=True,
            formula_matrix=ag,
            element_budget=depleted,
            mass_action_constants=np.asarray(
                mass_action_func(float(state.temperature)),
                dtype=np.float64,
            ),
            electron_row_index=int(electron_row_index),
            provenance={
                "formula_matrix": provenance.get(
                    "formula_matrix", "exogibbs_native_static_contract"
                ),
                "element_budget": "exogibbs_native_depleted_budget",
                "mass_action_constants": provenance.get(
                    "mass_action_constants",
                    "exogibbs_native_mass_action_constants_static_contract"
                    if mass_action_constants_func is not None
                    else "exogibbs_native_hvector_legacy_electron_policy_input",
                ),
            },
            max_nfev=gas_max_iter,
        )
        ln_nk = np.asarray(electron_report.ln_species_density, dtype=np.float64)
        ln_ntot = float(np.log(np.sum(np.exp(np.clip(ln_nk, -745.0, 700.0)))))
        gas_solver_success = bool(electron_report.solver_success)
        gas_solver_iterations = int(electron_report.neutral_total_nfev)
        gas_solver_final_residual = float(
            max(
                electron_report.max_neutral_log_budget_residual,
                electron_report.electron_charge_residual_abs,
            )
        )
        electron_policy_success = bool(electron_report.solver_success)
        electron_policy_charge_residual_abs = float(
            electron_report.electron_charge_residual_abs
        )
        electron_policy_max_neutral_log_budget_residual = float(
            electron_report.max_neutral_log_budget_residual
        )
        electron_policy_electron_density = float(electron_report.electron_density)
    else:
        gas_result = solve_gas_equilibrium_with_duals(
            depleted_state,
            jnp.asarray(ag, dtype=jnp.float64),
            hvector_func,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
        )
        ln_nk = np.asarray(gas_result["ln_nk"], dtype=np.float64)
        ln_ntot = float(gas_result["ln_ntot"])
        gas_solver_success = bool(gas_result["diagnostics"]["converged"])
        gas_solver_iterations = int(gas_result["diagnostics"]["n_iter"])
        gas_solver_final_residual = float(gas_result["diagnostics"]["final_residual"])
    refreshed_inventory = ag.dot(np.exp(np.clip(ln_nk, -745.0, 700.0)))
    budget_residual = refreshed_inventory - depleted
    residual_inf = (
        float(np.max(np.abs(budget_residual))) if budget_residual.size else 0.0
    )
    finite = bool(
        np.all(np.isfinite(depleted))
        and np.all(np.isfinite(ln_nk))
        and np.isfinite(ln_ntot)
        and np.all(np.isfinite(refreshed_inventory))
        and np.isfinite(residual_inf)
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.asarray(ln_nk, dtype=jnp.float64),
        ln_mk=jnp.asarray(ln_m, dtype=jnp.float64),
        ln_ntot=jnp.asarray(ln_ntot, dtype=jnp.float64),
        ln_nk_source_trace={
            "source": "exogibbs_native_depleted_budget_gas_refresh",
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        },
    )
    report = DepletedGasRefreshReport(
        report_schema="exogibbs_depleted_gas_refresh_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        provenance=provenance,
        support_indices=tuple(int(index) for index in indices.tolist()),
        condensate_amounts=tuple(float(value) for value in active_amounts.tolist()),
        condensate_inventory=tuple(float(value) for value in condensate_inventory.tolist()),
        original_element_budget=tuple(float(value) for value in budget.tolist()),
        depleted_element_budget=tuple(float(value) for value in depleted.tolist()),
        negative_depleted_budget_inf=negative_inf,
        clamped_negative_budget_count=clamped_count,
        gas_refresh_policy=gas_refresh_policy,
        electron_policy_used=electron_policy_used,
        electron_policy_solver_success=electron_policy_success,
        electron_policy_charge_residual_abs=electron_policy_charge_residual_abs,
        electron_policy_max_neutral_log_budget_residual=(
            electron_policy_max_neutral_log_budget_residual
        ),
        electron_policy_electron_density=electron_policy_electron_density,
        gas_solver_success=gas_solver_success,
        gas_solver_iterations=gas_solver_iterations,
        gas_solver_final_residual=gas_solver_final_residual,
        ln_nk=tuple(float(value) for value in ln_nk.tolist()),
        ln_ntot=ln_ntot,
        refreshed_gas_inventory=tuple(float(value) for value in refreshed_inventory.tolist()),
        depleted_budget_residual_inf=residual_inf,
        finite=finite,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )
    return init, report


__all__ = (
    "DepletedGasRefreshReport",
    "build_depleted_gas_refresh_init",
)
