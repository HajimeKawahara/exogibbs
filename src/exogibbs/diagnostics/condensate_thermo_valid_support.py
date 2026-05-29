"""Explicit diagnostic filtering for thermo-valid condensate support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.diagnostics.condensate_native_bundle import (
    validate_native_bundle_provenance,
)


@dataclass(frozen=True)
class ThermoValidSupportFilterReport:
    """Diagnostic report for thermo-valid support filtering."""

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
    valid_local_indices: tuple[int, ...]
    removed_local_indices: tuple[int, ...]
    valid_support_indices: tuple[int, ...]
    removed_support_indices: tuple[int, ...]
    removed_species_names: tuple[str, ...]
    sentinel_abs_threshold: float
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    field_provenance: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThermoValidSupportFilterResult:
    """Filtered arrays and accounting for thermo-valid support."""

    report: ThermoValidSupportFilterReport
    condensate_standard_source: tuple[float, ...]
    support_indices: tuple[int, ...]
    formula_matrix_cond_active: tuple[tuple[float, ...], ...] | None
    ln_mk: tuple[float, ...] | None
    rho: tuple[float, ...] | None
    eta: tuple[float, ...] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _optional_vector(
    values: Sequence[float] | None,
    name: str,
    expected_length: int,
) -> np.ndarray | None:
    if values is None:
        return None
    array = _as_vector(values, name)
    if array.shape[0] != expected_length:
        raise ValueError(f"{name} length must match condensate_standard_source.")
    return array


def _optional_matrix(
    values: Sequence[Sequence[float]] | None,
    name: str,
    expected_columns: int,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape[1] != expected_columns:
        raise ValueError(f"{name} column count must match condensate_standard_source.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def filter_thermo_valid_condensate_support(
    *,
    explicit_opt_in: bool,
    support_indices: Sequence[int],
    condensate_standard_source: Sequence[float],
    formula_matrix_cond_active: Sequence[Sequence[float]] | None = None,
    ln_mk: Sequence[float] | None = None,
    rho: Sequence[float] | None = None,
    eta: Sequence[float] | None = None,
    species_names: Sequence[str] | None = None,
    sentinel_abs_threshold: float = 1.0e10,
    field_provenance: Mapping[str, str] | None = None,
) -> ThermoValidSupportFilterResult:
    """Filter active support to condensates with finite, non-sentinel thermo sources."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for thermo-valid support filtering.")
    provenance = validate_native_bundle_provenance(field_provenance)
    support = tuple(int(index) for index in support_indices)
    if not support:
        raise ValueError("support_indices must not be empty.")
    if any(index < 0 for index in support):
        raise ValueError("support_indices must be non-negative.")
    source = _as_vector(condensate_standard_source, "condensate_standard_source")
    if source.shape[0] != len(support):
        raise ValueError("support_indices length must match condensate_standard_source.")
    threshold = float(sentinel_abs_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("sentinel_abs_threshold must be finite and positive.")

    matrix = _optional_matrix(
        formula_matrix_cond_active,
        "formula_matrix_cond_active",
        source.shape[0],
    )
    ln_mk_array = _optional_vector(ln_mk, "ln_mk", source.shape[0])
    rho_array = _optional_vector(rho, "rho", source.shape[0])
    eta_array = _optional_vector(eta, "eta", source.shape[0])

    valid_mask = np.abs(source) < threshold
    if not np.any(valid_mask):
        raise ValueError("thermo-valid support would be empty.")
    valid_local = tuple(int(index) for index in np.where(valid_mask)[0])
    removed_local = tuple(int(index) for index in np.where(~valid_mask)[0])
    valid_support = tuple(support[index] for index in valid_local)
    removed_support = tuple(support[index] for index in removed_local)
    removed_names: tuple[str, ...]
    if species_names is None:
        removed_names = tuple(str(index) for index in removed_support)
    else:
        removed_names = tuple(str(species_names[index]) for index in removed_support)

    report = ThermoValidSupportFilterReport(
        report_schema="exogibbs_thermo_valid_support_filter_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        original_support_count=len(support),
        filtered_support_count=len(valid_support),
        removed_support_count=len(removed_support),
        valid_local_indices=valid_local,
        removed_local_indices=removed_local,
        valid_support_indices=valid_support,
        removed_support_indices=removed_support,
        removed_species_names=removed_names,
        sentinel_abs_threshold=threshold,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        field_provenance=provenance,
    )
    return ThermoValidSupportFilterResult(
        report=report,
        condensate_standard_source=tuple(float(value) for value in source[valid_mask]),
        support_indices=valid_support,
        formula_matrix_cond_active=(
            tuple(tuple(float(value) for value in row) for row in matrix[:, valid_mask])
            if matrix is not None
            else None
        ),
        ln_mk=(
            tuple(float(value) for value in ln_mk_array[valid_mask])
            if ln_mk_array is not None
            else None
        ),
        rho=(
            tuple(float(value) for value in rho_array[valid_mask])
            if rho_array is not None
            else None
        ),
        eta=(
            tuple(float(value) for value in eta_array[valid_mask])
            if eta_array is not None
            else None
        ),
    )


__all__ = (
    "ThermoValidSupportFilterReport",
    "ThermoValidSupportFilterResult",
    "filter_thermo_valid_condensate_support",
)
