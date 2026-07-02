#!/usr/bin/env python
"""Compare ExoGibbs profile auto batch/rescue against FastChem4 outputs.

This is a volatile comparison runner. It intentionally keeps FastChem4 values
out of ExoGibbs constructor inputs and uses them only as a post-solve reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from jax import config
import jax.numpy as jnp

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    condensate_equilibrium_profile,
    _least_squares_element_potential,
)
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.api.potential import gibbs_energies
from exogibbs.condensates.inactive_driving import evaluate_inactive_condensate_driving
from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    CuratedProfileDefinition,
    element_budget_for_profile,
    support_payload_for_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.utils.fastchem_parity import normalize_species_name

try:
    import pyfastchem
except ImportError as exc:  # pragma: no cover - environment dependent.
    raise SystemExit(
        "pyfastchem is required for this volatile comparison runner."
    ) from exc


BOLTZMANN_CGS = 1.380649e-16
DEFAULT_OUTPUT_DIR = Path("volatiles_artifacts/fastchem4_vmap_cold_rescue_compare")
AMOUNT_FLOOR = 1.0e-300


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _normal_name_map(names: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, name in enumerate(names):
        out.setdefault(normalize_species_name(str(name)), index)
    return out


def _positive_names(
    names: Iterable[str],
    values: np.ndarray,
    *,
    floor: float,
) -> set[str]:
    return {
        normalize_species_name(name)
        for name, value in zip(names, values)
        if np.isfinite(value) and float(value) > floor
    }


def _budget_report(setup: Any, gas_n: np.ndarray, cond_n: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    reconstructed = (
        np.asarray(setup.formula_matrix, dtype=np.float64) @ np.asarray(gas_n, dtype=np.float64)
        + np.asarray(setup.formula_matrix_cond, dtype=np.float64) @ np.asarray(cond_n, dtype=np.float64)
    )
    residual = reconstructed - target
    relative = residual / np.maximum(np.abs(target), AMOUNT_FLOOR)
    mask = np.asarray([str(element) not in {"e-", "electron"} for element in setup.elements], dtype=bool)
    masked_relative = np.where(mask, np.abs(relative), 0.0)
    max_index = int(np.argmax(np.where(np.isfinite(masked_relative), masked_relative, np.inf)))
    return {
        "max_abs_relative_residual": float(masked_relative[max_index]),
        "max_abs_relative_residual_element": str(setup.elements[max_index]),
        "relative_l2": float(np.linalg.norm(masked_relative)),
        "absolute_l2": float(np.linalg.norm(np.where(mask, residual, 0.0))),
    }


def _safe_ln_amounts(values: np.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.asarray(np.maximum(values, AMOUNT_FLOOR), dtype=jnp.float64))


def _gibbs_over_rt(
    setup: Any,
    temperature: float,
    pressure: float,
    gas_n: np.ndarray,
    cond_n: np.ndarray,
) -> float:
    value = gibbs_energies(
        temperatures=jnp.asarray([float(temperature)], dtype=jnp.float64),
        pressures=jnp.asarray([float(pressure)], dtype=jnp.float64),
        chem_gas=setup.gas_setup,
        ln_ngas=_safe_ln_amounts(gas_n)[None, :],
        chem_cond=setup.condensate_setup,
        ln_ncond=_safe_ln_amounts(cond_n)[None, :],
        nomalize=True,
    )[0]
    return float(jax.device_get(value))


def _inactive_driving_report(
    setup: Any,
    temperature: float,
    pressure: float,
    gas_n: np.ndarray,
    cond_n: np.ndarray,
) -> dict[str, Any]:
    gas_ln_n = _safe_ln_amounts(gas_n)
    gas_stationarity_source = setup.gas_setup.hvector_func(float(temperature)) + math.log(
        float(pressure)
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=gas_ln_n,
        gas_stationarity_source=gas_stationarity_source,
    )
    report = evaluate_inactive_condensate_driving(
        formula_matrix_cond=setup.formula_matrix_cond,
        condensate_species_order=setup.condensate_species,
        condensate_amounts=cond_n,
        hvector_cond=setup.condensate_setup.hvector_func(float(temperature)),
        element_potential=element_potential,
        temperature=float(temperature),
        condensate_temperature_validity_upper=setup.condensate_setup.metadata.get(
            "temperature_validity_upper"
        ),
        active_floor=1.0e-50,
        activity_threshold=0.0,
    ).as_dict()
    return {
        "all_max_positive_inactive_driving": report["all_condensates"][
            "max_positive_inactive_driving"
        ],
        "all_positive_inactive_count": report["all_condensates"][
            "positive_inactive_count"
        ],
        "temperature_valid_max_positive_inactive_driving": report[
            "temperature_valid_condensates"
        ]["max_positive_inactive_driving"],
        "temperature_valid_positive_inactive_count": report[
            "temperature_valid_condensates"
        ]["positive_inactive_count"],
    }


def _aligned_fastchem_amounts(
    *,
    setup: Any,
    fastchem: dict[str, Any],
    layer_index: int,
    target: np.ndarray,
) -> dict[str, Any]:
    gas = np.zeros((len(setup.gas_species),), dtype=np.float64)
    cond = np.zeros((len(setup.condensate_species),), dtype=np.float64)
    fc_gas_map = _normal_name_map(fastchem["gas_names"])
    fc_cond_map = _normal_name_map(fastchem["condensate_names"])
    fc_gas_n = fastchem["gas_number_densities"][layer_index]
    fc_cond_n = fastchem["condensate_number_densities"][layer_index]
    for index, name in enumerate(setup.gas_species):
        fc_index = fc_gas_map.get(normalize_species_name(name))
        if fc_index is not None:
            gas[index] = float(fc_gas_n[fc_index])
    for index, name in enumerate(setup.condensate_species):
        fc_index = fc_cond_map.get(normalize_species_name(name))
        if fc_index is not None:
            cond[index] = float(fc_cond_n[fc_index])
    reconstructed = (
        np.asarray(setup.formula_matrix, dtype=np.float64) @ gas
        + np.asarray(setup.formula_matrix_cond, dtype=np.float64) @ cond
    )
    mask = np.asarray(
        [
            str(element) not in {"e-", "electron"}
            and float(target[index]) > 0.0
            and np.isfinite(reconstructed[index])
            and reconstructed[index] > 0.0
            for index, element in enumerate(setup.elements)
        ],
        dtype=bool,
    )
    if not np.any(mask):
        scale = 0.0
    else:
        denom = float(np.dot(reconstructed[mask], reconstructed[mask]))
        scale = 0.0 if denom <= 0.0 else float(np.dot(reconstructed[mask], target[mask]) / denom)
    return {
        "gas_n": gas * scale,
        "condensate_amounts": cond * scale,
        "scale_to_exogibbs_budget": scale,
        "unscaled_budget": reconstructed,
    }


def _log10_ratio_metrics(
    *,
    exo_names: tuple[str, ...],
    exo_values: np.ndarray,
    fc_names: tuple[str, ...],
    fc_values: np.ndarray,
    floor: float,
    top_n: int,
) -> dict[str, Any]:
    exo_map = _normal_name_map(exo_names)
    fc_map = _normal_name_map(fc_names)
    common = sorted(set(exo_map).intersection(fc_map))
    rows = []
    for name in common:
        exo_value = float(exo_values[exo_map[name]])
        fc_value = float(fc_values[fc_map[name]])
        if not (
            np.isfinite(exo_value)
            and np.isfinite(fc_value)
            and exo_value > floor
            and fc_value > floor
        ):
            continue
        log10_ratio = math.log10(exo_value) - math.log10(fc_value)
        rows.append(
            {
                "species": name,
                "exogibbs": exo_value,
                "fastchem4": fc_value,
                "log10_ratio_exogibbs_over_fastchem4": log10_ratio,
                "abs_log10_ratio": abs(log10_ratio),
            }
        )
    rows.sort(key=lambda row: row["abs_log10_ratio"], reverse=True)
    abs_values = [float(row["abs_log10_ratio"]) for row in rows]
    return {
        "common_species": len(common),
        "finite_positive_compared": len(rows),
        "max_abs_log10_ratio": rows[0]["abs_log10_ratio"] if rows else None,
        "p95_abs_log10_ratio": (
            float(np.percentile(abs_values, 95.0)) if abs_values else None
        ),
        "mean_abs_log10_ratio": float(np.mean(abs_values)) if abs_values else None,
        "top_outliers": rows[:top_n],
    }


def _threshold_key(value: float) -> str:
    return f"{float(value):.0e}"


def _floor_sweep_log10_ratio_metrics(
    *,
    exo_names: tuple[str, ...],
    exo_values: np.ndarray,
    fc_names: tuple[str, ...],
    fc_values: np.ndarray,
    floors: Sequence[float],
    top_n: int,
) -> dict[str, Any]:
    return {
        _threshold_key(floor): _log10_ratio_metrics(
            exo_names=exo_names,
            exo_values=exo_values,
            fc_names=fc_names,
            fc_values=fc_values,
            floor=float(floor),
            top_n=top_n,
        )
        for floor in floors
    }


def _major_species_log10_ratio_metrics(
    *,
    exo_names: tuple[str, ...],
    exo_values: np.ndarray,
    fc_names: tuple[str, ...],
    fc_values: np.ndarray,
    abundance_threshold: float,
    ratio_floor: float,
    selection_mode: str,
    top_n: int,
) -> dict[str, Any]:
    """Compare gas species selected by union or overlap abundance thresholds."""

    if selection_mode not in {"union", "overlap"}:
        raise ValueError("selection_mode must be 'union' or 'overlap'.")
    exo_map = _normal_name_map(exo_names)
    fc_map = _normal_name_map(fc_names)
    common = sorted(set(exo_map).intersection(fc_map))
    rows = []
    major_count = 0
    for name in common:
        exo_value = float(exo_values[exo_map[name]])
        fc_value = float(fc_values[fc_map[name]])
        if not (np.isfinite(exo_value) and np.isfinite(fc_value)):
            continue
        if selection_mode == "union":
            selected = max(exo_value, fc_value) >= float(abundance_threshold)
        else:
            selected = min(exo_value, fc_value) >= float(abundance_threshold)
        if not selected:
            continue
        major_count += 1
        exo_for_ratio = max(exo_value, float(ratio_floor))
        fc_for_ratio = max(fc_value, float(ratio_floor))
        log10_ratio = math.log10(exo_for_ratio) - math.log10(fc_for_ratio)
        rows.append(
            {
                "species": name,
                "exogibbs": exo_value,
                "fastchem4": fc_value,
                "abundance_threshold": float(abundance_threshold),
                "ratio_floor": float(ratio_floor),
                "selection_mode": selection_mode,
                "floor_clipped": bool(
                    exo_value <= float(ratio_floor) or fc_value <= float(ratio_floor)
                ),
                "log10_ratio_exogibbs_over_fastchem4": log10_ratio,
                "abs_log10_ratio": abs(log10_ratio),
            }
        )
    rows.sort(key=lambda row: row["abs_log10_ratio"], reverse=True)
    abs_values = [float(row["abs_log10_ratio"]) for row in rows]
    return {
        "common_species": len(common),
        "major_species": major_count,
        "finite_compared": len(rows),
        "abundance_threshold": float(abundance_threshold),
        "ratio_floor": float(ratio_floor),
        "selection_mode": selection_mode,
        "max_abs_log10_ratio": rows[0]["abs_log10_ratio"] if rows else None,
        "p95_abs_log10_ratio": (
            float(np.percentile(abs_values, 95.0)) if abs_values else None
        ),
        "mean_abs_log10_ratio": float(np.mean(abs_values)) if abs_values else None,
        "top_outliers": rows[:top_n],
    }


def _major_species_sweep_log10_ratio_metrics(
    *,
    exo_names: tuple[str, ...],
    exo_values: np.ndarray,
    fc_names: tuple[str, ...],
    fc_values: np.ndarray,
    abundance_thresholds: Sequence[float],
    ratio_floor: float,
    selection_mode: str,
    top_n: int,
) -> dict[str, Any]:
    return {
        _threshold_key(threshold): _major_species_log10_ratio_metrics(
            exo_names=exo_names,
            exo_values=exo_values,
            fc_names=fc_names,
            fc_values=fc_values,
            abundance_threshold=float(threshold),
            ratio_floor=float(ratio_floor),
            selection_mode=selection_mode,
            top_n=top_n,
        )
        for threshold in abundance_thresholds
    }


def _max_metric_by_key(
    summary: dict[str, float | None],
    metrics: Mapping[str, Mapping[str, Any]],
    metric_name: str,
) -> None:
    for key, row in metrics.items():
        value = row.get(metric_name)
        if value is None:
            continue
        value = float(value)
        old = summary.get(key)
        summary[key] = value if old is None else max(float(old), value)


def _format_optional_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}g}"


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _make_fastchem(args: argparse.Namespace):
    root = Path(args.fastchem_root)
    fastchem = pyfastchem.FastChem(
        str(root / args.fastchem_element_file),
        str(root / args.fastchem_gas_logk),
        str(root / args.fastchem_cond_logk),
        int(args.fastchem_verbosity),
    )
    return fastchem


def _set_fastchem_profile_abundances(
    fastchem: Any,
    definition: CuratedProfileDefinition,
) -> None:
    if definition.carbon_to_oxygen_ratio is None:
        return
    abundances = np.asarray(fastchem.getElementAbundances(), dtype=np.float64)
    index_c = fastchem.getElementIndex("C")
    index_o = fastchem.getElementIndex("O")
    if index_c == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
        raise ValueError("FastChem4 element C is unavailable.")
    if index_o == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
        raise ValueError("FastChem4 element O is unavailable.")
    abundances[index_c] = float(definition.carbon_to_oxygen_ratio) * abundances[index_o]
    fastchem.setElementAbundances(abundances)


def _run_fastchem_profile(
    args: argparse.Namespace,
    definition: CuratedProfileDefinition,
) -> dict[str, Any]:
    fastchem = _make_fastchem(args)
    _set_fastchem_profile_abundances(fastchem, definition)

    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()
    temperatures = np.asarray(definition.temperatures, dtype=np.float64)
    pressures = np.asarray(definition.pressures, dtype=np.float64)
    input_data.temperature = temperatures
    input_data.pressure = pressures
    if args.fastchem_condensation == "equilibrium":
        input_data.equilibrium_condensation = True
        input_data.rainout_condensation = False
    elif args.fastchem_condensation == "rainout":
        input_data.equilibrium_condensation = False
        input_data.rainout_condensation = True
    elif args.fastchem_condensation == "none":
        input_data.equilibrium_condensation = False
        input_data.rainout_condensation = False
    else:
        raise ValueError(f"Unknown FastChem condensation mode: {args.fastchem_condensation}")

    flag = fastchem.calcDensities(input_data, output_data)
    gas_names = tuple(
        str(fastchem.getGasSpeciesSymbol(index))
        for index in range(fastchem.getGasSpeciesNumber())
    )
    cond_names = tuple(
        str(fastchem.getCondSpeciesSymbol(index))
        for index in range(fastchem.getCondSpeciesNumber())
    )
    number_densities = np.asarray(output_data.number_densities, dtype=np.float64)
    number_densities_cond = np.asarray(
        output_data.number_densities_cond,
        dtype=np.float64,
    )
    gas_number_density = pressures * 1.0e6 / (BOLTZMANN_CGS * temperatures)
    gas_x = number_densities / gas_number_density[:, None]
    return {
        "flag": int(flag),
        "flag_message": str(pyfastchem.FASTCHEM_MSG[flag]),
        "gas_names": gas_names,
        "condensate_names": cond_names,
        "gas_number_densities": number_densities,
        "gas_x": gas_x,
        "condensate_number_densities": number_densities_cond,
        "element_conserved": np.asarray(output_data.element_conserved, dtype=np.int32),
        "nb_iterations": np.asarray(output_data.nb_iterations, dtype=np.int32),
        "nb_chemistry_iterations": np.asarray(
            output_data.nb_chemistry_iterations,
            dtype=np.int32,
        ),
        "nb_cond_iterations": np.asarray(output_data.nb_cond_iterations, dtype=np.int32),
        "total_element_density": np.asarray(
            output_data.total_element_density,
            dtype=np.float64,
        ),
    }


def _run_exogibbs_profile(
    args: argparse.Namespace,
    setup: Any,
    definition: CuratedProfileDefinition,
) -> Any:
    budget = element_budget_for_profile(setup, definition)
    support_indices, support_amounts_init = support_payload_for_profile(
        setup,
        definition,
        budget,
    )
    explicit_inits = None
    if args.explicit_gas_init and support_indices:
        initial_states = []
        for temperature, pressure in zip(definition.temperatures, definition.pressures):
            gas_result = equilibrium(
                setup.gas_setup,
                float(temperature),
                float(pressure),
                budget,
                Pref=1.0,
                options=EquilibriumOptions(),
            )
            initial_states.append(
                CondensateEquilibriumInit(
                    gas_ln_n=gas_result.ln_n,
                    gas_ntot=gas_result.ntot,
                    support_indices=support_indices,
                    support_amounts=support_amounts_init,
                )
            )
        explicit_inits = tuple(initial_states)
    options = CondensateEquilibriumOptions(
        profile_method=args.exogibbs_method,
        profile_warm_start_support_policy="explicit_payload",
        return_diagnostics=True,
        allow_caveat_tiers=True,
        max_inner_iterations=args.max_inner_iterations,
        enable_head_route_warm_start=True,
        enable_depleted_gas_refresh=True,
        enable_full_condensate_budget_residual_gate=not args.disable_budget_gate,
        enable_profile_native_activity_support_expansion=(
            args.enable_native_activity_support_expansion
        ),
        profile_native_activity_support_topk=args.native_activity_support_topk,
        profile_native_activity_max_support_count=args.native_activity_max_support_count,
        profile_native_activity_threshold=args.native_activity_threshold,
        seed_fraction=args.seed_fraction,
        max_seed_amount=args.max_seed_amount,
    )
    return condensate_equilibrium_profile(
        setup,
        jnp.asarray(definition.temperatures, dtype=jnp.float64),
        jnp.asarray(definition.pressures, dtype=jnp.float64),
        budget,
        init=explicit_inits,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=options,
        method=args.exogibbs_method,
        return_diagnostics=True,
    )


def _summarize_exogibbs_profile(result: Any) -> dict[str, Any]:
    route_counts = Counter(str(layer.selected_route) for layer in result.layers)
    status_counts = Counter(str(layer.status) for layer in result.layers)
    profile_batch = {}
    if isinstance(result.diagnostics, dict):
        profile_batch = dict(result.diagnostics.get("experimental_profile_fixed_support_batch", {}))
    fallback_rescue = profile_batch.get("fallback_rescue", {})
    batched_arrays = result.batched_arrays or {}
    fallback_required = batched_arrays.get("fallback_required")
    fallback_required_count = None
    if fallback_required is not None:
        fallback_required_count = int(np.sum(np.asarray(jax.device_get(fallback_required))))
    support_expansion_added_count = 0
    support_expansion_layer_count = 0
    support_expansion_max_support_count = None
    if isinstance(result.diagnostics, dict):
        for report in result.diagnostics.get("layers", ()):
            expansion = report.get("support_expansion") if isinstance(report, dict) else None
            if not isinstance(expansion, dict):
                continue
            support_expansion_layer_count += 1
            support_expansion_added_count += int(expansion.get("added_support_count", 0))
            count = expansion.get("expanded_support_count")
            if count is not None:
                support_expansion_max_support_count = max(
                    int(count),
                    0 if support_expansion_max_support_count is None else support_expansion_max_support_count,
                )
    return {
        "method": str(result.method),
        "route_counts": dict(route_counts),
        "status_counts": dict(status_counts),
        "converged_count": int(sum(bool(layer.converged) for layer in result.layers)),
        "batch_route": profile_batch.get("route"),
        "batch_accepted": profile_batch.get("accepted"),
        "fallback_required_count": fallback_required_count,
        "fallback_rescue_mode": (
            fallback_rescue.get("mode") if isinstance(fallback_rescue, dict) else None
        ),
        "fallback_rescue_replaced_count": (
            fallback_rescue.get("replaced_count")
            if isinstance(fallback_rescue, dict)
            else None
        ),
        "support_expansion_layer_count": support_expansion_layer_count,
        "support_expansion_added_count": support_expansion_added_count,
        "support_expansion_max_support_count": support_expansion_max_support_count,
    }


def _compare_family(
    args: argparse.Namespace,
    setup: Any,
    definition: CuratedProfileDefinition,
) -> dict[str, Any]:
    fastchem = _run_fastchem_profile(args, definition)
    exogibbs = _run_exogibbs_profile(args, setup, definition)
    target_budget = np.asarray(element_budget_for_profile(setup, definition), dtype=np.float64)

    layers = []
    max_abs_log10_ratio = None
    floor_sweep_family_max: dict[str, float | None] = {
        _threshold_key(floor): None for floor in args.gas_floor_sweep
    }
    major_gas_family_max: dict[str, float | None] = {
        _threshold_key(threshold): None for threshold in args.major_gas_thresholds
    }
    major_overlap_gas_family_max: dict[str, float | None] = {
        _threshold_key(threshold): None for threshold in args.major_gas_thresholds
    }
    jaccards = []
    for layer_index, layer in enumerate(exogibbs.layers):
        exo_gas_x = np.asarray(jax.device_get(layer.gas_x), dtype=np.float64)
        gas_metrics = _log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            floor=args.gas_floor,
            top_n=args.top_outliers,
        )
        gas_floor_sweep = _floor_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            floors=args.gas_floor_sweep,
            top_n=args.top_outliers,
        )
        major_gas_metrics = _major_species_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            abundance_thresholds=args.major_gas_thresholds,
            ratio_floor=args.major_gas_ratio_floor,
            selection_mode="union",
            top_n=args.top_outliers,
        )
        major_overlap_gas_metrics = _major_species_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            abundance_thresholds=args.major_gas_thresholds,
            ratio_floor=args.major_gas_ratio_floor,
            selection_mode="overlap",
            top_n=args.top_outliers,
        )
        if gas_metrics["max_abs_log10_ratio"] is not None:
            max_abs_log10_ratio = max(
                max_abs_log10_ratio or 0.0,
                float(gas_metrics["max_abs_log10_ratio"]),
            )
        _max_metric_by_key(
            floor_sweep_family_max,
            gas_floor_sweep,
            "max_abs_log10_ratio",
        )
        _max_metric_by_key(
            major_gas_family_max,
            major_gas_metrics,
            "max_abs_log10_ratio",
        )
        _max_metric_by_key(
            major_overlap_gas_family_max,
            major_overlap_gas_metrics,
            "max_abs_log10_ratio",
        )
        exo_active = _positive_names(
            tuple(setup.condensate_species),
            np.asarray(jax.device_get(layer.condensate_amounts), dtype=np.float64),
            floor=args.exogibbs_condensate_floor,
        )
        fc_active = _positive_names(
            fastchem["condensate_names"],
            fastchem["condensate_number_densities"][layer_index],
            floor=args.fastchem_condensate_floor,
        )
        jaccard = _jaccard(exo_active, fc_active)
        if jaccard is not None:
            jaccards.append(jaccard)
        exo_gas_n = np.asarray(jax.device_get(layer.gas_n), dtype=np.float64)
        exo_cond_n = np.asarray(
            jax.device_get(layer.condensate_amounts),
            dtype=np.float64,
        )
        fc_scaled = _aligned_fastchem_amounts(
            setup=setup,
            fastchem=fastchem,
            layer_index=layer_index,
            target=target_budget,
        )
        exo_gibbs = _gibbs_over_rt(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            exo_gas_n,
            exo_cond_n,
        )
        fc_gibbs = _gibbs_over_rt(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
        )
        exo_budget = _budget_report(setup, exo_gas_n, exo_cond_n, target_budget)
        fc_budget = _budget_report(
            setup,
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
            target_budget,
        )
        exo_inactive = _inactive_driving_report(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            exo_gas_n,
            exo_cond_n,
        )
        fc_inactive = _inactive_driving_report(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
        )
        layers.append(
            {
                "layer_index": layer_index,
                "temperature": float(definition.temperatures[layer_index]),
                "pressure": float(definition.pressures[layer_index]),
                "exogibbs_status": str(layer.status),
                "exogibbs_converged": bool(layer.converged),
                "exogibbs_route": str(layer.selected_route),
                "fastchem4_element_conserved_all": bool(
                    np.all(fastchem["element_conserved"][layer_index])
                ),
                "gas_log10_ratio": gas_metrics,
                "gas_floor_sweep_log10_ratio": gas_floor_sweep,
                "major_gas_log10_ratio": major_gas_metrics,
                "major_overlap_gas_log10_ratio": major_overlap_gas_metrics,
                "active_condensate_jaccard": jaccard,
                "exogibbs_active_condensates": sorted(exo_active),
                "fastchem4_active_condensates": sorted(fc_active),
                "gibbs_over_rt": {
                    "exogibbs": exo_gibbs,
                    "fastchem4_scaled": fc_gibbs,
                    "delta_exogibbs_minus_fastchem4_scaled": exo_gibbs - fc_gibbs,
                },
                "budget": {
                    "exogibbs": exo_budget,
                    "fastchem4_scaled": fc_budget,
                    "fastchem4_scale_to_exogibbs_budget": fc_scaled[
                        "scale_to_exogibbs_budget"
                    ],
                },
                "inactive_driving": {
                    "exogibbs": exo_inactive,
                    "fastchem4_scaled": fc_inactive,
                },
            }
        )
    deltas = [
        layer["gibbs_over_rt"]["delta_exogibbs_minus_fastchem4_scaled"]
        for layer in layers
        if math.isfinite(layer["gibbs_over_rt"]["delta_exogibbs_minus_fastchem4_scaled"])
    ]
    exo_budget_residuals = [
        layer["budget"]["exogibbs"]["max_abs_relative_residual"] for layer in layers
    ]
    fc_budget_residuals = [
        layer["budget"]["fastchem4_scaled"]["max_abs_relative_residual"]
        for layer in layers
    ]
    exo_valid_inactive = [
        layer["inactive_driving"]["exogibbs"][
            "temperature_valid_max_positive_inactive_driving"
        ]
        for layer in layers
    ]
    fc_valid_inactive = [
        layer["inactive_driving"]["fastchem4_scaled"][
            "temperature_valid_max_positive_inactive_driving"
        ]
        for layer in layers
    ]
    return {
        "family": definition.family,
        "rows": len(definition.temperatures),
        "exogibbs": _summarize_exogibbs_profile(exogibbs),
        "fastchem4": {
            "flag": fastchem["flag"],
            "flag_message": fastchem["flag_message"],
            "element_conserved_counts": {
                str(k): int(v)
                for k, v in Counter(np.ravel(fastchem["element_conserved"]).tolist()).items()
            },
            "nb_iterations_max": int(np.max(fastchem["nb_iterations"])),
            "nb_cond_iterations_max": int(np.max(fastchem["nb_cond_iterations"])),
        },
        "comparison": {
            "max_abs_log10_gas_ratio": max_abs_log10_ratio,
            "gas_floor_sweep_max_abs_log10_ratio": floor_sweep_family_max,
            "major_gas_max_abs_log10_ratio": major_gas_family_max,
            "major_overlap_gas_max_abs_log10_ratio": major_overlap_gas_family_max,
            "active_condensate_jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
            "active_condensate_jaccard_min": float(np.min(jaccards)) if jaccards else None,
            "gibbs_over_rt": {
                "finite_rows": len(deltas),
                "exogibbs_lower_rows": int(sum(delta < 0.0 for delta in deltas)),
                "fastchem4_scaled_lower_rows": int(sum(delta > 0.0 for delta in deltas)),
                "tie_rows": int(sum(delta == 0.0 for delta in deltas)),
                "max_abs_delta_exogibbs_minus_fastchem4_scaled": (
                    float(np.max(np.abs(deltas))) if deltas else None
                ),
                "mean_delta_exogibbs_minus_fastchem4_scaled": (
                    float(np.mean(deltas)) if deltas else None
                ),
            },
            "budget": {
                "exogibbs_max_relative_residual": float(np.max(exo_budget_residuals)),
                "fastchem4_scaled_max_relative_residual": float(
                    np.max(fc_budget_residuals)
                ),
            },
            "inactive_driving": {
                "exogibbs_temperature_valid_max": float(np.max(exo_valid_inactive)),
                "fastchem4_scaled_temperature_valid_max": float(
                    np.max(fc_valid_inactive)
                ),
            },
        },
        "layers": layers,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FastChem4 vs ExoGibbs vmap_cold/rescue comparison",
        "",
        f"- timestamp UTC: `{report['timestamp_utc']}`",
        f"- JAX backend: `{report['environment']['jax_default_backend']}`",
        f"- ExoGibbs method: `{report['config']['exogibbs_method']}`",
        f"- FastChem4 condensation: `{report['config']['fastchem_condensation']}`",
        f"- FastChem4 values used as ExoGibbs constructor inputs: `{report['provenance']['fastchem4_constructor_inputs_used']}`",
        "",
        "| family | rows | route | raw gas dex | floor 1e-20 gas dex | major overlap >=1e-8 gas dex | Jaccard mean/min | Exo lower G/RT | FC lower G/RT | max abs dG/RT | Exo max budget rel | FC max budget rel | Exo valid inactive max |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in report["families"]:
        exo = family["exogibbs"]
        comp = family["comparison"]
        jmean = comp["active_condensate_jaccard_mean"]
        jmin = comp["active_condensate_jaccard_min"]
        jtext = (
            "n/a"
            if jmean is None
            else f"{jmean:.3g}/{jmin:.3g}"
        )
        max_ratio = comp["max_abs_log10_gas_ratio"]
        max_text = "n/a" if max_ratio is None else f"{max_ratio:.4g}"
        floor_sweep = comp.get("gas_floor_sweep_max_abs_log10_ratio", {})
        major_overlap_gas = comp.get("major_overlap_gas_max_abs_log10_ratio", {})
        floor_1e20_text = _format_optional_float(floor_sweep.get("1e-20"))
        major_1e8_text = _format_optional_float(major_overlap_gas.get("1e-08"))
        gibbs = comp["gibbs_over_rt"]
        budget = comp["budget"]
        inactive = comp["inactive_driving"]
        lines.append(
            "| {family} | {rows} | {route} | {max_ratio} | {floor_1e20} | "
            "{major_1e8} | {jtext} | {exo_lower} | {fc_lower} | "
            "{max_dg} | {exo_budget:.3g} | "
            "{fc_budget:.3g} | {exo_inactive:.3g} |".format(
                family=family["family"],
                rows=family["rows"],
                route=exo["batch_route"],
                max_ratio=max_text,
                floor_1e20=floor_1e20_text,
                major_1e8=major_1e8_text,
                jtext=jtext,
                exo_lower=gibbs["exogibbs_lower_rows"],
                fc_lower=gibbs["fastchem4_scaled_lower_rows"],
                max_dg=(
                    "n/a"
                    if gibbs["max_abs_delta_exogibbs_minus_fastchem4_scaled"] is None
                    else f"{gibbs['max_abs_delta_exogibbs_minus_fastchem4_scaled']:.3g}"
                ),
                exo_budget=budget["exogibbs_max_relative_residual"],
                fc_budget=budget["fastchem4_scaled_max_relative_residual"],
                exo_inactive=inactive["exogibbs_temperature_valid_max"],
            )
        )
    lines.extend(
        [
            "",
            "## Gas Metric Floor Sweep",
            "",
            "| family | 1e-300 | 1e-30 | 1e-20 | union >=1e-8 | overlap >=1e-12 | overlap >=1e-10 | overlap >=1e-8 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in report["families"]:
        comp = family["comparison"]
        floor_sweep = comp.get("gas_floor_sweep_max_abs_log10_ratio", {})
        major_gas = comp.get("major_gas_max_abs_log10_ratio", {})
        major_overlap_gas = comp.get("major_overlap_gas_max_abs_log10_ratio", {})
        lines.append(
            "| {family} | {f300} | {f30} | {f20} | {u8} | {o12} | {o10} | {o8} |".format(
                family=family["family"],
                f300=_format_optional_float(floor_sweep.get("1e-300")),
                f30=_format_optional_float(floor_sweep.get("1e-30")),
                f20=_format_optional_float(floor_sweep.get("1e-20")),
                u8=_format_optional_float(major_gas.get("1e-08")),
                o12=_format_optional_float(major_overlap_gas.get("1e-12")),
                o10=_format_optional_float(major_overlap_gas.get("1e-10")),
                o8=_format_optional_float(major_overlap_gas.get("1e-08")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This runner uses FastChem4 only as a comparison target.",
            "- ExoGibbs inputs come from native FastChem4 preset files and curated support payloads.",
            "- Raw gas comparison is on common positive species above the configured floor.",
            "- The floor sweep suppresses species below larger floors on both sides.",
            "- The major union gas metric compares species whose ExoGibbs or FastChem4 abundance exceeds the stated threshold, with small values clipped only for the log ratio.",
            "- The major overlap gas metric compares only species above the threshold in both outputs; this is the most trace-insensitive gas-abundance score in this report.",
            "- Condensate comparison is active-list Jaccard, not amount matching.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        help="Curated families to run, or 'all'.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="comparison")
    parser.add_argument("--fastchem-root", default="FastChem")
    parser.add_argument("--fastchem-element-file", default="input/element_abundances/asplund_2021.dat")
    parser.add_argument("--fastchem-gas-logk", default="input/logK/logK_wo_ions.dat")
    parser.add_argument("--fastchem-cond-logk", default="input/logK/logK_condensates.dat")
    parser.add_argument(
        "--fastchem-condensation",
        choices=("equilibrium", "rainout", "none"),
        default="equilibrium",
    )
    parser.add_argument("--fastchem-verbosity", type=int, default=0)
    parser.add_argument(
        "--exogibbs-element-file",
        default="FastChem4/element_abundances/asplund_2021.dat",
    )
    parser.add_argument(
        "--exogibbs-method",
        choices=("auto", "vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom"),
        default="auto",
    )
    parser.add_argument("--max-inner-iterations", type=int, default=100)
    parser.add_argument(
        "--no-explicit-gas-init",
        dest="explicit_gas_init",
        action="store_false",
        help="Disable native gas-only per-layer init for fixed-support batch planning.",
    )
    parser.set_defaults(explicit_gas_init=True)
    parser.add_argument("--disable-budget-gate", action="store_true")
    parser.add_argument("--enable-native-activity-support-expansion", action="store_true")
    parser.add_argument("--native-activity-support-topk", type=int, default=8)
    parser.add_argument("--native-activity-max-support-count", type=int, default=16)
    parser.add_argument("--native-activity-threshold", type=float, default=0.0)
    parser.add_argument("--seed-fraction", type=float, default=0.8)
    parser.add_argument("--max-seed-amount", type=float, default=1.0)
    parser.add_argument("--gas-floor", type=float, default=1.0e-300)
    parser.add_argument(
        "--gas-floor-sweep",
        nargs="+",
        type=float,
        default=(1.0e-300, 1.0e-30, 1.0e-20),
        help="Gas abundance floors for trace-insensitive max log-ratio summaries.",
    )
    parser.add_argument(
        "--major-gas-thresholds",
        nargs="+",
        type=float,
        default=(1.0e-12, 1.0e-10, 1.0e-8),
        help=(
            "Compare gas species whose ExoGibbs or FastChem4 abundance exceeds "
            "one of these thresholds."
        ),
    )
    parser.add_argument(
        "--major-gas-ratio-floor",
        type=float,
        default=1.0e-300,
        help="Small-value clipping floor used only inside major-species log ratios.",
    )
    parser.add_argument("--exogibbs-condensate-floor", type=float, default=0.0)
    parser.add_argument("--fastchem-condensate-floor", type=float, default=0.0)
    parser.add_argument("--top-outliers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.update("jax_enable_x64", True)
    family_names = (
        tuple(FRESH_CURATED_PROFILES)
        if args.families == ["all"]
        else tuple(args.families)
    )
    unknown = [name for name in family_names if name not in FRESH_CURATED_PROFILES]
    if unknown:
        raise ValueError(f"Unknown curated families: {unknown}")

    setup = condensate_chemical_setup(
        species_defalt_elements=False,
        element_file=args.exogibbs_element_file,
        silent=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = []
    for family_name in family_names:
        definition = FRESH_CURATED_PROFILES[family_name]
        print(f"running {family_name} ({len(definition.temperatures)} layers)")
        families.append(_compare_family(args, setup, definition))

    report = {
        "schema": "exogibbs_volatile_fastchem4_vmap_cold_rescue_comparison_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "jax_version": jax.__version__,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "config": {
            "families": family_names,
            "fastchem_root": str(args.fastchem_root),
            "fastchem_element_file": args.fastchem_element_file,
            "fastchem_gas_logk": args.fastchem_gas_logk,
            "fastchem_cond_logk": args.fastchem_cond_logk,
            "fastchem_condensation": args.fastchem_condensation,
            "exogibbs_element_file": args.exogibbs_element_file,
            "exogibbs_method": args.exogibbs_method,
            "max_inner_iterations": args.max_inner_iterations,
            "explicit_gas_init": args.explicit_gas_init,
            "enable_native_activity_support_expansion": (
                args.enable_native_activity_support_expansion
            ),
            "native_activity_support_topk": args.native_activity_support_topk,
            "native_activity_max_support_count": args.native_activity_max_support_count,
            "native_activity_threshold": args.native_activity_threshold,
            "seed_fraction": args.seed_fraction,
            "max_seed_amount": args.max_seed_amount,
            "gas_floor": args.gas_floor,
            "gas_floor_sweep": args.gas_floor_sweep,
            "major_gas_thresholds": args.major_gas_thresholds,
            "major_gas_ratio_floor": args.major_gas_ratio_floor,
            "exogibbs_condensate_floor": args.exogibbs_condensate_floor,
            "fastchem_condensate_floor": args.fastchem_condensate_floor,
        },
        "provenance": {
            "fastchem4_constructor_inputs_used": False,
            "fastchem4_exact_replay_target": False,
            "fastchem4_is_comparison_target_only": True,
            "volatile_artifact": True,
        },
        "families": families,
    }
    json_path = args.output_dir / f"{args.output_prefix}.json"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(_jsonable(report), md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
