"""Pure NumPy comparison metrics for ExoGibbs and FastChem results.

The helpers in this module deliberately operate on caller-provided arrays.
They do not construct a solver or use one solver's output as input to the
other solver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from exogibbs.utils.fastchem_parity import normalize_species_name


OccurrenceKey = tuple[str, int]


def occurrence_keys(names: Sequence[str]) -> tuple[OccurrenceKey, ...]:
    """Return normalized, duplicate-safe ``(name, occurrence)`` keys.

    Occurrences are zero-based and counted independently for each normalized
    name. For example, ``["Zn(s)", "Zn(s)"]`` maps to
    ``(("Zn(s)", 0), ("Zn(s)", 1))``.
    """

    counts: dict[str, int] = {}
    keys: list[OccurrenceKey] = []
    for raw_name in names:
        name = normalize_species_name(str(raw_name))
        occurrence = counts.get(name, 0)
        keys.append((name, occurrence))
        counts[name] = occurrence + 1
    return tuple(keys)


def align_species_values(
    target_names: Sequence[str],
    source_names: Sequence[str],
    source_values: Any,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Align the last axis of source values to a target species catalog.

    Simple FastChem/ExoGibbs aliases are normalized before matching.
    Duplicate names are matched by occurrence, so repeated catalog slots are
    never collapsed. Any leading dimensions in ``source_values`` are retained.
    Target slots missing from the source catalog receive ``fill_value``.
    """

    values = np.asarray(source_values)
    if values.ndim == 0:
        raise ValueError("source_values must have a species axis.")
    if values.shape[-1] != len(source_names):
        raise ValueError(
            "source_values last axis must match the source catalog length."
        )

    target_keys = occurrence_keys(target_names)
    source_index = {
        key: index for index, key in enumerate(occurrence_keys(source_names))
    }
    dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype)
    aligned = np.full(
        values.shape[:-1] + (len(target_keys),),
        fill_value,
        dtype=dtype,
    )
    for target_index, key in enumerate(target_keys):
        source_slot = source_index.get(key)
        if source_slot is not None:
            aligned[..., target_index] = values[..., source_slot]
    return aligned


def element_budget_metrics(
    *,
    gas_formula_matrix: Any,
    condensate_formula_matrix: Any,
    gas_amounts: Any,
    condensate_amounts: Any,
    target: Any,
    element_names: Sequence[str],
    relative_floor: float = 1.0e-12,
    excluded_names: Sequence[str] = ("e-", "electron"),
) -> dict[str, Any]:
    """Compute gas-plus-condensate elemental budget closure metrics.

    Formula matrices use the ExoGibbs convention ``(elements, species)``.
    Electron aliases are retained in per-element rows but excluded from
    aggregate closure metrics by default.
    """

    gas_matrix = np.asarray(gas_formula_matrix, dtype=np.float64)
    condensate_matrix = np.asarray(
        condensate_formula_matrix,
        dtype=np.float64,
    )
    gas = np.asarray(gas_amounts, dtype=np.float64)
    condensates = np.asarray(condensate_amounts, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    floor = float(relative_floor)

    if gas_matrix.ndim != 2:
        raise ValueError("gas_formula_matrix must be two-dimensional.")
    if condensate_matrix.ndim != 2:
        raise ValueError("condensate_formula_matrix must be two-dimensional.")
    if gas_matrix.shape[0] != len(element_names):
        raise ValueError(
            "gas_formula_matrix rows must match the element catalog length."
        )
    if condensate_matrix.shape[0] != len(element_names):
        raise ValueError(
            "condensate_formula_matrix rows must match the element catalog "
            "length."
        )
    if gas.ndim != 1 or gas.shape[0] != gas_matrix.shape[1]:
        raise ValueError(
            "gas_amounts must have one value per gas formula-matrix column."
        )
    if condensates.ndim != 1 or condensates.shape[0] != condensate_matrix.shape[1]:
        raise ValueError(
            "condensate_amounts must have one value per condensate "
            "formula-matrix column."
        )
    if target_array.ndim != 1 or target_array.shape[0] != len(element_names):
        raise ValueError("target must have one value per element.")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive.")

    reconstructed = gas_matrix @ gas + condensate_matrix @ condensates
    residual = reconstructed - target_array
    denominator = np.maximum(np.abs(target_array), floor)
    signed_relative = residual / denominator
    absolute_relative = np.abs(signed_relative)

    normalized_names = tuple(
        normalize_species_name(str(name)) for name in element_names
    )
    normalized_excluded = {
        normalize_species_name(str(name)) for name in excluded_names
    }
    included_mask = np.asarray(
        [name not in normalized_excluded for name in normalized_names],
        dtype=bool,
    )
    included_indices = np.flatnonzero(included_mask)

    if included_indices.size:
        sanitized = np.where(
            np.isfinite(absolute_relative[included_indices]),
            absolute_relative[included_indices],
            np.inf,
        )
        local_max_index = int(np.argmax(sanitized))
        max_index = int(included_indices[local_max_index])
        max_absolute_relative = float(absolute_relative[max_index])
        max_element: str | None = str(element_names[max_index])
        finite = bool(
            np.all(np.isfinite(absolute_relative[included_indices]))
        )
    else:
        max_index = -1
        max_absolute_relative = 0.0
        max_element = None
        finite = True

    rows = [
        {
            "name": str(element_names[index]),
            "normalized_name": normalized_names[index],
            "excluded": bool(not included_mask[index]),
            "target": float(target_array[index]),
            "reconstructed": float(reconstructed[index]),
            "residual": float(residual[index]),
            "relative_denominator": float(denominator[index]),
            "signed_relative_residual": float(signed_relative[index]),
            "absolute_relative_residual": float(absolute_relative[index]),
        }
        for index in range(len(element_names))
    ]
    return to_json_safe(
        {
            "relative_floor": floor,
            "finite": finite,
            "element_count": len(element_names),
            "included_element_count": int(np.count_nonzero(included_mask)),
            "excluded_element_names": [
                str(element_names[index])
                for index in np.flatnonzero(~included_mask)
            ],
            "max_abs_relative_residual": max_absolute_relative,
            "max_absolute_relative_residual": max_absolute_relative,
            "max_abs_relative_residual_element": max_element,
            "max_absolute_relative_residual_element": max_element,
            "max_abs_relative_residual_element_index": max_index,
            "max_absolute_relative_residual_element_index": max_index,
            "target": target_array,
            "reconstructed": reconstructed,
            "residual": residual,
            "relative_denominator": denominator,
            "signed_relative_residual": signed_relative,
            "absolute_relative_residual": absolute_relative,
            "rows": rows,
        }
    )


def gas_major_species_metrics(
    *,
    names: Sequence[str],
    left_values: Any,
    right_values: Any,
    threshold: float = 1.0e-8,
    ratio_floor: float = 1.0e-300,
    excluded_names: Sequence[str] = ("e-",),
    top_n: int = 10,
) -> dict[str, Any]:
    """Compare gas species that are major in either input vector.

    The major-species mask is the union of species meeting ``threshold`` on
    the left or right. Ratios are reported as left over right after applying
    ``ratio_floor`` to both magnitudes.
    """

    left, right, keys = _validate_pair(names, left_values, right_values)
    threshold_value = _validate_nonnegative_finite(threshold, "threshold")
    ratio_floor_value = _validate_positive_finite(
        ratio_floor,
        "ratio_floor",
    )
    top_count = _validate_top_n(top_n)

    excluded = {
        normalize_species_name(str(name)) for name in excluded_names
    }
    eligible = np.asarray([key[0] not in excluded for key in keys], dtype=bool)
    left_major = eligible & (np.abs(left) >= threshold_value)
    right_major = eligible & (np.abs(right) >= threshold_value)
    union = left_major | right_major
    intersection = left_major & right_major
    left_only = left_major & ~right_major
    right_only = right_major & ~left_major

    ratio, log10_ratio, absolute_log10_ratio = _ratio_arrays(
        left,
        right,
        ratio_floor_value,
    )
    absolute_difference = np.abs(left - right)
    union_indices = np.flatnonzero(union)
    ordered = _descending_indices(union_indices, absolute_log10_ratio)
    top_indices = ordered[:top_count]

    union_errors = absolute_log10_ratio[union]
    max_error, mean_error, rms_error = _error_summary(union_errors)
    union_count = int(np.count_nonzero(union))
    intersection_count = int(np.count_nonzero(intersection))

    return to_json_safe(
        {
            "threshold": threshold_value,
            "ratio_floor": ratio_floor_value,
            "excluded_names": sorted(excluded),
            "species_count": len(keys),
            "left_major_count": int(np.count_nonzero(left_major)),
            "right_major_count": int(np.count_nonzero(right_major)),
            "intersection_major_count": intersection_count,
            "major_species_count": union_count,
            "major_set_jaccard": _jaccard(intersection_count, union_count),
            "finite": bool(
                np.all(np.isfinite(left[eligible]))
                and np.all(np.isfinite(right[eligible]))
            ),
            "max_abs_log10_ratio": max_error,
            "max_absolute_log10_ratio": max_error,
            "mean_abs_log10_ratio": mean_error,
            "mean_absolute_log10_ratio": mean_error,
            "rms_log10_ratio": rms_error,
            "top_rows_sort": "absolute_log10_ratio_descending",
            "major_species": _records_for_mask(keys, union),
            "left_only_major_species": _records_for_mask(keys, left_only),
            "right_only_major_species": _records_for_mask(keys, right_only),
            "top_rows": [
                _amount_row(
                    keys,
                    int(index),
                    left,
                    right,
                    ratio,
                    log10_ratio,
                    absolute_log10_ratio,
                    absolute_difference,
                )
                for index in top_indices
            ],
        }
    )


def condensate_comparison_metrics(
    *,
    names: Sequence[str],
    left_values: Any,
    right_values: Any,
    active_floor: float = 0.0,
    ratio_floor: float = 1.0e-300,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compare condensate active sets and amounts without collapsing slots."""

    left, right, keys = _validate_pair(names, left_values, right_values)
    floor = _validate_nonnegative_finite(active_floor, "active_floor")
    ratio_floor_value = _validate_positive_finite(
        ratio_floor,
        "ratio_floor",
    )
    top_count = _validate_top_n(top_n)

    left_active = left > floor
    right_active = right > floor
    union = left_active | right_active
    intersection = left_active & right_active
    left_only = left_active & ~right_active
    right_only = right_active & ~left_active

    ratio, log10_ratio, absolute_log10_ratio = _ratio_arrays(
        left,
        right,
        ratio_floor_value,
    )
    absolute_difference = np.abs(left - right)
    union_indices = np.flatnonzero(union)
    ordered = _descending_indices(union_indices, absolute_log10_ratio)
    top_indices = ordered[:top_count]

    union_errors = absolute_log10_ratio[union]
    max_error, mean_error, rms_error = _error_summary(union_errors)
    union_count = int(np.count_nonzero(union))
    intersection_count = int(np.count_nonzero(intersection))

    return to_json_safe(
        {
            "active_floor": floor,
            "ratio_floor": ratio_floor_value,
            "slot_count": len(keys),
            "left_active_count": int(np.count_nonzero(left_active)),
            "right_active_count": int(np.count_nonzero(right_active)),
            "intersection_active_count": intersection_count,
            "union_active_count": union_count,
            "active_set_jaccard": _jaccard(
                intersection_count,
                union_count,
            ),
            "finite": bool(
                np.all(np.isfinite(left))
                and np.all(np.isfinite(right))
            ),
            "max_abs_log10_ratio": max_error,
            "max_absolute_log10_ratio": max_error,
            "mean_abs_log10_ratio": mean_error,
            "mean_absolute_log10_ratio": mean_error,
            "rms_log10_ratio": rms_error,
            "top_rows_sort": "absolute_log10_ratio_descending",
            "left_active": _records_for_mask(keys, left_active),
            "right_active": _records_for_mask(keys, right_active),
            "intersection_active": _records_for_mask(keys, intersection),
            "left_only_active": _records_for_mask(keys, left_only),
            "right_only_active": _records_for_mask(keys, right_only),
            "top_rows": [
                _amount_row(
                    keys,
                    int(index),
                    left,
                    right,
                    ratio,
                    log10_ratio,
                    absolute_log10_ratio,
                    absolute_difference,
                )
                for index in top_indices
            ],
        }
    )


def profile_phase_transitions(
    *,
    names: Sequence[str],
    amounts: Any,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Describe active-condensate changes between adjacent profile layers."""

    values = np.asarray(amounts, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("amounts must have shape (layers, condensate slots).")
    if values.shape[1] != len(names):
        raise ValueError(
            "amounts last axis must match the condensate catalog length."
        )
    threshold_value = _validate_nonnegative_finite(threshold, "threshold")
    keys = occurrence_keys(names)
    active = values > threshold_value

    adjacent: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for to_index in range(1, values.shape[0]):
        from_index = to_index - 1
        before = active[from_index]
        after = active[to_index]
        entered = ~before & after
        exited = before & ~after
        intersection_count = int(np.count_nonzero(before & after))
        union_count = int(np.count_nonzero(before | after))
        event = {
            "from_index": from_index,
            "to_index": to_index,
            "changed": bool(np.any(entered | exited)),
            "entered": _records_for_mask(keys, entered),
            "exited": _records_for_mask(keys, exited),
            "active_before": _records_for_mask(keys, before),
            "active_after": _records_for_mask(keys, after),
            "changed_slot_count": int(np.count_nonzero(entered | exited)),
            "active_set_jaccard": _jaccard(
                intersection_count,
                union_count,
            ),
        }
        adjacent.append(event)
        if event["changed"]:
            transitions.append(event)

    return to_json_safe(
        {
            "threshold": threshold_value,
            "layer_count": int(values.shape[0]),
            "slot_count": int(values.shape[1]),
            "adjacent_pair_count": max(int(values.shape[0]) - 1, 0),
            "transition_count": len(transitions),
            "transition_indices": [
                event["to_index"] for event in transitions
            ],
            "initial_active": (
                _records_for_mask(keys, active[0])
                if values.shape[0]
                else []
            ),
            "final_active": (
                _records_for_mask(keys, active[-1])
                if values.shape[0]
                else []
            ),
            "adjacent": adjacent,
            "transitions": transitions,
        }
    )


def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy values to strict JSON-compatible objects.

    Non-finite floating-point values become ``None`` so callers may use
    ``json.dumps(..., allow_nan=False)``.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {
            "real": to_json_safe(value.real),
            "imag": to_json_safe(value.imag),
        }
    if isinstance(value, Mapping):
        return {
            _json_key(key): to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            to_json_safe(item)
            for item in sorted(value, key=lambda item: repr(item))
        ]
    if hasattr(value, "__fspath__"):
        return str(value)
    return str(value)


def _validate_pair(
    names: Sequence[str],
    left_values: Any,
    right_values: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[OccurrenceKey, ...]]:
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if left.ndim != 1 or left.shape[0] != len(names):
        raise ValueError("left_values must have one value per catalog slot.")
    if right.ndim != 1 or right.shape[0] != len(names):
        raise ValueError("right_values must have one value per catalog slot.")
    return left, right, occurrence_keys(names)


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _validate_positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _validate_top_n(top_n: int) -> int:
    result = int(top_n)
    if result != top_n or result < 0:
        raise ValueError("top_n must be a nonnegative integer.")
    return result


def _ratio_arrays(
    left: np.ndarray,
    right: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_magnitude = np.maximum(np.abs(left), floor)
    right_magnitude = np.maximum(np.abs(right), floor)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = left_magnitude / right_magnitude
        log10_ratio = np.log10(left_magnitude) - np.log10(right_magnitude)
    return ratio, log10_ratio, np.abs(log10_ratio)


def _descending_indices(
    indices: np.ndarray,
    score: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return indices
    sortable_score = np.nan_to_num(
        score[indices],
        nan=-np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    order = np.argsort(-sortable_score, kind="stable")
    return indices[order]


def _error_summary(errors: np.ndarray) -> tuple[float, float, float]:
    if errors.size == 0:
        return 0.0, 0.0, 0.0
    if not np.all(np.isfinite(errors)):
        return math.inf, math.inf, math.inf
    return (
        float(np.max(errors)),
        float(np.mean(errors)),
        float(np.sqrt(np.mean(np.square(errors)))),
    )


def _jaccard(intersection_count: int, union_count: int) -> float:
    if union_count == 0:
        return 1.0
    return float(intersection_count / union_count)


def _slot_record(
    keys: tuple[OccurrenceKey, ...],
    index: int,
) -> dict[str, Any]:
    name, occurrence = keys[index]
    return {
        "name": name,
        "occurrence": occurrence,
        "slot": index,
        "occurrence_key": [name, occurrence],
    }


def _records_for_mask(
    keys: tuple[OccurrenceKey, ...],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        _slot_record(keys, int(index))
        for index in np.flatnonzero(mask)
    ]


def _amount_row(
    keys: tuple[OccurrenceKey, ...],
    index: int,
    left: np.ndarray,
    right: np.ndarray,
    ratio: np.ndarray,
    log10_ratio: np.ndarray,
    absolute_log10_ratio: np.ndarray,
    absolute_difference: np.ndarray,
) -> dict[str, Any]:
    row = _slot_record(keys, index)
    row.update(
        {
            "left": float(left[index]),
            "right": float(right[index]),
            "absolute_difference": float(absolute_difference[index]),
            "ratio_left_to_right": float(ratio[index]),
            "log10_ratio_left_to_right": float(log10_ratio[index]),
            "absolute_log10_ratio": float(
                absolute_log10_ratio[index]
            ),
        }
    )
    return row


def _json_key(key: Any) -> str:
    safe_key = to_json_safe(key)
    if isinstance(safe_key, str):
        return safe_key
    if isinstance(safe_key, (bool, int, float)) or safe_key is None:
        return str(safe_key)
    return repr(safe_key)


__all__ = (
    "OccurrenceKey",
    "align_species_values",
    "condensate_comparison_metrics",
    "element_budget_metrics",
    "gas_major_species_metrics",
    "occurrence_keys",
    "profile_phase_transitions",
    "to_json_safe",
)
