"""Source-convention-safe electron refresh checks for condensate lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ElectronRefreshReport:
    """Report for an electron refresh source/gauge compatibility check."""

    report_schema: str
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    accepted: bool
    classification: str
    charge_row_index: int
    sentinel_count: int
    gas_lambda_gauge_residual_l2: float
    gas_lambda_gauge_residual_max_abs: float
    max_gauge_residual: float
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def check_source_convention_safe_electron_refresh(
    *,
    explicit_opt_in: bool,
    ln_nk: Sequence[float],
    element_potential: Sequence[float],
    formula_matrix: Sequence[Sequence[float]],
    gas_stationarity_source: Sequence[float],
    charge_row_index: int = 0,
    max_gauge_residual: float = 1.0e-8,
    sentinel_abs_threshold: float = 1.0e10,
    metadata: Mapping[str, Any] | None = None,
) -> ElectronRefreshReport:
    """Check whether an electron refresh is in the RGIE source convention."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for electron refresh checks.")
    q = _as_vector(ln_nk, "ln_nk")
    lam = _as_vector(element_potential, "element_potential")
    ag = _as_matrix(formula_matrix, "formula_matrix")
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    row_index = int(charge_row_index)
    if row_index < 0 or row_index >= ag.shape[0]:
        raise ValueError("charge_row_index is out of range.")
    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match ln_nk length.")
    if ag.shape[0] != lam.shape[0]:
        raise ValueError("formula_matrix row count must match element_potential length.")
    if gas_source.shape[0] != q.shape[0]:
        raise ValueError("gas_stationarity_source length must match ln_nk length.")
    gauge_residual = q + gas_source - ag.T @ lam
    sentinel_count = int(np.sum(np.abs(gauge_residual) >= float(sentinel_abs_threshold)))
    max_abs = float(np.max(np.abs(gauge_residual))) if gauge_residual.size else 0.0
    l2 = float(np.linalg.norm(gauge_residual))
    limit = float(max_gauge_residual)
    if limit < 0.0:
        raise ValueError("max_gauge_residual must be non-negative.")
    accepted = sentinel_count == 0 and max_abs <= limit
    return ElectronRefreshReport(
        report_schema="exogibbs_source_convention_safe_electron_refresh_report_v1",
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        accepted=accepted,
        classification=(
            "electron_refresh_source_convention_compatible"
            if accepted
            else "electron_refresh_source_convention_mismatch"
        ),
        charge_row_index=row_index,
        sentinel_count=sentinel_count,
        gas_lambda_gauge_residual_l2=l2,
        gas_lambda_gauge_residual_max_abs=max_abs,
        max_gauge_residual=limit,
        metadata=dict(metadata or {}),
    )


__all__ = (
    "ElectronRefreshReport",
    "check_source_convention_safe_electron_refresh",
)
