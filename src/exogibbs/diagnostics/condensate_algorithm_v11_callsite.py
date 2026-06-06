"""Explicit diagnostic callsite wrapper for algorithm-v1.1 PD-IPM R-GIE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.diagnostics.condensate_thermo_valid_support import (
    ThermoValidSupportFilterReport,
    filter_thermo_valid_condensate_support,
)
from exogibbs.optimize.pdipm_rgie_cond import (
    PdipmRgieCondensateState,
    PdipmRgieReducedStepReport,
    build_pdipm_rgie_condensate_state,
    solve_pdipm_rgie_algorithm_v11_reduced_step,
)


@dataclass(frozen=True)
class AlgorithmV11ThermoValidCallsiteReport:
    """Report for a thermo-valid algorithm-v1.1 diagnostic callsite."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    original_support_count: int
    filtered_support_count: int
    removed_support_count: int
    filter_report: ThermoValidSupportFilterReport
    reduced_step_report: PdipmRgieReducedStepReport
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_report"] = self.filter_report.as_dict()
        payload["reduced_step_report"] = self.reduced_step_report.as_dict()
        return payload


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def run_algorithm_v11_thermo_valid_reduced_callsite(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    support_indices: Sequence[int],
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    epsilon: float,
    species_names: Sequence[str] | None = None,
    sentinel_abs_threshold: float = 1.0e10,
    alpha_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    qhat_regularization: float = 0.0,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_rho: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    require_budget_nonworsening: bool = False,
    field_provenance: Mapping[str, str] | None = None,
) -> AlgorithmV11ThermoValidCallsiteReport:
    """Filter thermo-invalid support and run one algorithm-v1.1 reduced step."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for algorithm-v1.1 callsites.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    if state.rho is None:
        raise ValueError("state.rho is required for algorithm-v1.1 callsites.")
    ln_mk = _as_vector(state.ln_mk, "state.ln_mk")
    rho = _as_vector(state.rho, "state.rho")
    eta = np.exp(rho)
    filtered = filter_thermo_valid_condensate_support(
        explicit_opt_in=True,
        support_indices=support_indices,
        condensate_standard_source=condensate_standard_source,
        formula_matrix_cond_active=formula_matrix_cond_active,
        ln_mk=ln_mk,
        rho=rho,
        eta=eta,
        species_names=species_names,
        sentinel_abs_threshold=sentinel_abs_threshold,
        field_provenance=field_provenance or state.field_provenance,
    )
    filtered_state = build_pdipm_rgie_condensate_state(
        ln_nk=state.ln_nk,
        ln_mk=filtered.ln_mk or (),
        element_potential=state.element_potential,
        ln_ntot=state.ln_ntot,
        rho=filtered.rho,
        eta=filtered.eta,
        field_provenance=state.field_provenance,
    )
    reduced_step = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=filtered_state,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=filtered.formula_matrix_cond_active or (),
        element_inventory_target=element_inventory_target,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=filtered.condensate_standard_source,
        epsilon=epsilon,
        alpha_candidates=alpha_candidates,
        qhat_regularization=qhat_regularization,
        max_abs_delta_q=max_abs_delta_q,
        max_abs_delta_r=max_abs_delta_r,
        max_abs_delta_rho=max_abs_delta_rho,
        max_abs_delta_lambda=max_abs_delta_lambda,
        require_budget_nonworsening=require_budget_nonworsening,
    )
    return AlgorithmV11ThermoValidCallsiteReport(
        report_schema="exogibbs_algorithm_v11_thermo_valid_callsite_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        original_support_count=filtered.report.original_support_count,
        filtered_support_count=filtered.report.filtered_support_count,
        removed_support_count=filtered.report.removed_support_count,
        filter_report=filtered.report,
        reduced_step_report=reduced_step,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "AlgorithmV11ThermoValidCallsiteReport",
    "run_algorithm_v11_thermo_valid_reduced_callsite",
)
