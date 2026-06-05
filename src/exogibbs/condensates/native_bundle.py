"""Default-off native condensate residual bundle diagnostics.

This module is intentionally explicit-import only. It does not import
FastChem4, call production solvers, modify production state, or connect to
presets/defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


FORBIDDEN_PROVENANCE_TOKENS = (
    "fastchem4_trace",
    "fastchem4_public",
    "fastchem4_runtime",
    "branch_replay",
    "reference_fit",
    "unknown_reference",
)


@dataclass(frozen=True)
class NativeCondensateResidualBundle:
    """Diagnostic native condensate residual bundle."""

    bundle_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool
    temperature: float
    pressure: float
    element_order: tuple[str, ...]
    gas_species_order: tuple[str, ...]
    condensate_species_order: tuple[str, ...]
    ln_nk: tuple[float, ...]
    ln_mk: tuple[float, ...]
    ln_ntot: float
    ntot: float
    nk: tuple[float, ...]
    condensate_amount: tuple[float, ...]
    Ag: tuple[tuple[float, ...], ...]
    Ac: tuple[tuple[float, ...], ...]
    gk: tuple[float, ...]
    standard_potential_T: tuple[float, ...]
    eta: tuple[float, ...]
    epsilon_or_nu: float
    element_inventory_target: tuple[float, ...]
    gas_element_inventory: tuple[float, ...]
    condensate_inventory: tuple[float, ...]
    total_element_inventory: tuple[float, ...]
    active_amount_floor: float
    active_set: tuple[str, ...]
    active_indices: tuple[int, ...]
    field_provenance: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(values: Sequence[float], name: str, expected: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if expected is not None and array.shape[0] != expected:
        raise ValueError(f"{name} length mismatch: got {array.shape[0]}, expected {expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape != shape:
        raise ValueError(f"{name} shape mismatch: got {array.shape}, expected {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_native_bundle_provenance(field_provenance: Mapping[str, str] | None) -> dict[str, str]:
    provenance = {str(key): str(value) for key, value in dict(field_provenance or {}).items()}
    for field, label in provenance.items():
        lowered = label.lower()
        if any(token in lowered for token in FORBIDDEN_PROVENANCE_TOKENS):
            raise ValueError(f"{field} provenance is forbidden for native bundle construction: {label}")
    return provenance


def build_native_condensate_residual_bundle(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gk: Sequence[float],
    hvector_cond: Sequence[float],
    eta: Sequence[float],
    epsilon_or_nu: float,
    element_order: Sequence[str],
    gas_species_order: Sequence[str],
    condensate_species_order: Sequence[str],
    temperature: float,
    pressure: float,
    active_amount_floor: float = 0.0,
    field_provenance: Mapping[str, str] | None = None,
) -> NativeCondensateResidualBundle:
    """Build a default-off diagnostic native condensate residual bundle."""

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    if float(pressure) <= 0.0:
        raise ValueError("pressure must be positive.")
    if float(active_amount_floor) < 0.0:
        raise ValueError("active_amount_floor must be non-negative.")

    provenance = validate_native_bundle_provenance(field_provenance)
    elements = tuple(str(item) for item in element_order)
    gas_species = tuple(str(item) for item in gas_species_order)
    condensates = tuple(str(item) for item in condensate_species_order)
    if not elements or not gas_species or not condensates:
        raise ValueError("element, gas species, and condensate species orders must be non-empty.")

    nelement = len(elements)
    ngas = len(gas_species)
    ncond = len(condensates)
    ln_nk_array = _as_vector(ln_nk, "ln_nk", ngas)
    ln_mk_array = _as_vector(ln_mk, "ln_mk", ncond)
    formula_gas = _as_matrix(formula_matrix, "formula_matrix", (nelement, ngas))
    formula_cond = _as_matrix(formula_matrix_cond, "formula_matrix_cond", (nelement, ncond))
    target = _as_vector(element_inventory_target, "element_inventory_target", nelement)
    gk_array = _as_vector(gk, "gk", ngas)
    hcond_array = _as_vector(hvector_cond, "hvector_cond", ncond)
    eta_array = _as_vector(eta, "eta", nelement)

    nk = np.exp(ln_nk_array)
    mk = np.exp(ln_mk_array)
    ntot = float(math.exp(float(ln_ntot)))
    gas_inventory = formula_gas @ nk
    condensate_inventory = formula_cond @ mk
    active_indices = tuple(int(index) for index, amount in enumerate(mk) if amount > active_amount_floor)
    active_set = tuple(condensates[index] for index in active_indices)

    field_provenance_out = {
        "ln_nk": provenance.get("ln_nk", "exogibbs_native"),
        "ln_mk": provenance.get("ln_mk", "exogibbs_native"),
        "eta": provenance.get("eta", "exogibbs_native_derived"),
        "gas_element_inventory": "derived_from_formula_matrix_exp_lnnk",
        "condensate_inventory": "derived_from_formula_matrix_cond_exp_lnmk",
        "condensate_amount": "derived_from_exp_lnmk",
        "active_set": "derived_from_native_condensate_amount",
    }

    return NativeCondensateResidualBundle(
        bundle_schema="exogibbs_native_condensate_residual_bundle_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
        temperature=float(temperature),
        pressure=float(pressure),
        element_order=elements,
        gas_species_order=gas_species,
        condensate_species_order=condensates,
        ln_nk=tuple(float(value) for value in ln_nk_array),
        ln_mk=tuple(float(value) for value in ln_mk_array),
        ln_ntot=float(ln_ntot),
        ntot=ntot,
        nk=tuple(float(value) for value in nk),
        condensate_amount=tuple(float(value) for value in mk),
        Ag=tuple(tuple(float(value) for value in row) for row in formula_gas),
        Ac=tuple(tuple(float(value) for value in row) for row in formula_cond),
        gk=tuple(float(value) for value in gk_array),
        standard_potential_T=tuple(float(value) for value in hcond_array),
        eta=tuple(float(value) for value in eta_array),
        epsilon_or_nu=float(epsilon_or_nu),
        element_inventory_target=tuple(float(value) for value in target),
        gas_element_inventory=tuple(float(value) for value in gas_inventory),
        condensate_inventory=tuple(float(value) for value in condensate_inventory),
        total_element_inventory=tuple(float(value) for value in gas_inventory + condensate_inventory),
        active_amount_floor=float(active_amount_floor),
        active_set=active_set,
        active_indices=active_indices,
        field_provenance=field_provenance_out,
    )
