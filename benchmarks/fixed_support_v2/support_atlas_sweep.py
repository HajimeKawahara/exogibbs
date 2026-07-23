#!/usr/bin/env python
"""Build a fixed-support atlas for curated condensate cases.

This volatile GPU-oriented runner separates two questions:

1. Which explicit fixed supports produce good PD-IPM results?
2. Which native support-selection rule could reproduce those supports?

It intentionally writes one JSON per family/variant so long atlas runs can be
resumed cheaply.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from jax import config
import jax.numpy as jnp

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    build_condensate_equilibrium_result_from_solver_payload,
    prepare_experimental_profile_fixed_support_batch_plan,
    run_experimental_profile_fixed_support_batch_plan,
    _least_squares_element_potential,
)
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    CuratedProfileDefinition,
    element_budget_for_profile,
    support_payload_for_profile,
)
from exogibbs.condensates.fixed_support_payload import (
    condensate_capacity,
    seed_fixed_support_payload,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.utils.fastchem_parity import normalize_species_name


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    ROOT
    / "benchmarks"
    / "fastchem4"
    / "fastchem4_vmap_cold_rescue_compare.py"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "fixed_support_v2_support_atlas"
AMOUNT_FLOOR = 1.0e-300


def _load_benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_exogibbs_fastchem4_compare",
        BENCHMARK_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class SupportVariant:
    label: str
    group: str
    policy: str
    supports: tuple[tuple[int, ...], ...]
    metadata: Mapping[str, Any]


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


def _device_array_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _device_array_tree(item) for key, item in value.items()}
    return np.asarray(jax.device_get(value))


def _slice_batch_tree(value: Any, subset_index: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _slice_batch_tree(item, subset_index)
            for key, item in value.items()
        }
    array = np.asarray(value)
    sliced = array if array.shape == () else array[subset_index]
    return _jsonable(sliced)


def _slice_batch_scalar(value: Any, subset_index: int) -> Any:
    array = np.asarray(value)
    sliced = array[subset_index]
    if np.asarray(sliced).ndim > 0:
        return _jsonable(sliced)
    if np.issubdtype(array.dtype, np.integer):
        return int(sliced)
    if np.issubdtype(array.dtype, np.bool_):
        return bool(sliced)
    return float(sliced)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _solver_checkpoint_path(
    output_dir: Path,
    family: str,
    variant: str,
) -> Path:
    return output_dir / (
        f"{_safe_label(family)}__{_safe_label(variant)}__solver_checkpoint.pkl"
    )


def _write_solver_checkpoint(
    *,
    path: Path,
    family: str,
    variant: str,
    arrays: Mapping[str, Any],
) -> None:
    payload = {
        "schema": "exogibbs_fixed_support_solver_checkpoint_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "variant": variant,
        "arrays": _device_array_tree(arrays),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _safe_label(value: str) -> str:
    return (
        value.replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _threshold_label(value: float) -> str:
    return _safe_label(f"{float(value):.0e}")


def _subset_profile_definition(
    definition: CuratedProfileDefinition,
    layer_indices: Sequence[int] | None,
) -> tuple[CuratedProfileDefinition, tuple[int, ...]]:
    source_indices = tuple(range(len(definition.temperatures)))
    if layer_indices is None:
        return definition, source_indices
    source_indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
    invalid = [
        index
        for index in source_indices
        if index < 0 or index >= len(definition.temperatures)
    ]
    if invalid:
        raise ValueError(f"Invalid layer indices for {definition.family}: {invalid}")
    return (
        replace(
            definition,
            temperatures=tuple(
                definition.temperatures[index] for index in source_indices
            ),
            pressures=tuple(definition.pressures[index] for index in source_indices),
        ),
        source_indices,
    )


def _normal_name_map(names: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, name in enumerate(names):
        out.setdefault(normalize_species_name(str(name)), index)
    return out


def _validity_upper(setup: Any) -> np.ndarray:
    upper = setup.condensate_setup.metadata.get("temperature_validity_upper")
    if upper is None:
        return np.full((len(setup.condensate_species),), np.inf, dtype=np.float64)
    return np.asarray(upper, dtype=np.float64)


def _gas_only_activity_fields(
    *,
    setup: Any,
    temperature: float,
    pressure: float,
    budget: Sequence[float],
) -> dict[str, Any]:
    gas = equilibrium(
        setup.gas_setup,
        float(temperature),
        float(pressure),
        jnp.asarray(budget, dtype=jnp.float64),
        Pref=1.0,
        options=EquilibriumOptions(),
        return_diagnostics=False,
    )
    stationarity_source = (
        setup.gas_setup.hvector_func(float(temperature))
        + math.log(float(pressure))
        - jnp.log(jnp.asarray(gas.ntot, dtype=jnp.float64))
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=jnp.asarray(gas.ln_n, dtype=jnp.float64),
        gas_stationarity_source=stationarity_source,
    )
    hcond = np.asarray(
        setup.condensate_setup.hvector_func(float(temperature)),
        dtype=np.float64,
    )
    driving = np.asarray(
        jax.device_get(
            jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64).T
            @ element_potential
            - hcond
        ),
        dtype=np.float64,
    )
    capacities = np.asarray(
        [
            condensate_capacity(setup.formula_matrix_cond, budget, index)
            for index in range(len(setup.condensate_species))
        ],
        dtype=np.float64,
    )
    valid = float(temperature) <= _validity_upper(setup)
    usable = valid & np.isfinite(capacities) & (capacities > 0.0)
    return {
        "gas_ln_n": gas.ln_n,
        "gas_ntot": gas.ntot,
        "stationarity_source": stationarity_source,
        "driving": driving,
        "capacities": capacities,
        "temperature_valid": valid,
        "usable": usable,
    }


def _ordered_activity_capacity(fields: Mapping[str, Any], threshold: float) -> tuple[int, ...]:
    driving = np.asarray(fields["driving"], dtype=np.float64)
    capacities = np.asarray(fields["capacities"], dtype=np.float64)
    usable = np.asarray(fields["usable"], dtype=bool)
    rows = []
    for index, score in enumerate(driving):
        if not usable[index] or score <= float(threshold):
            continue
        rows.append((float(capacities[index]), float(score), int(index)))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(index for _, _, index in rows)


def _ordered_activity_driving(
    fields: Mapping[str, Any],
    *,
    threshold: float | None,
) -> tuple[int, ...]:
    driving = np.asarray(fields["driving"], dtype=np.float64)
    usable = np.asarray(fields["usable"], dtype=bool)
    rows = []
    for index, score in enumerate(driving):
        if not usable[index]:
            continue
        if threshold is not None and score <= float(threshold):
            continue
        rows.append((float(score), int(index)))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return tuple(index for _, index in rows)


def _ordered_capacity(fields: Mapping[str, Any]) -> tuple[int, ...]:
    capacities = np.asarray(fields["capacities"], dtype=np.float64)
    usable = np.asarray(fields["usable"], dtype=bool)
    rows = [
        (float(capacity), int(index))
        for index, capacity in enumerate(capacities)
        if usable[index]
    ]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return tuple(index for _, index in rows)


def _dedup_cap(indices: Sequence[int], cap: int | None) -> tuple[int, ...]:
    out = tuple(dict.fromkeys(int(index) for index in indices))
    if cap is None:
        return out
    return out[: int(cap)]


def _oracle_fastchem_supports(
    *,
    setup: Any,
    fastchem: Mapping[str, Any],
    floor: float,
) -> tuple[tuple[int, ...], ...]:
    exo_map = _normal_name_map(setup.condensate_species)
    fc_cond_map = _normal_name_map(fastchem["condensate_names"])
    fc_to_exo = {
        fc_index: exo_map[name]
        for name, fc_index in fc_cond_map.items()
        if name in exo_map
    }
    supports = []
    for layer_amounts in fastchem["condensate_number_densities"]:
        indices = [
            int(fc_to_exo[index])
            for index, value in enumerate(layer_amounts)
            if index in fc_to_exo and np.isfinite(value) and float(value) > float(floor)
        ]
        supports.append(tuple(dict.fromkeys(indices)))
    return tuple(supports)


def _build_support_variants(
    *,
    args: argparse.Namespace,
    setup: Any,
    definition: CuratedProfileDefinition,
    fastchem: Mapping[str, Any],
) -> tuple[SupportVariant, ...]:
    budget = np.asarray(element_budget_for_profile(setup, definition), dtype=np.float64)
    base_support, _ = support_payload_for_profile(setup, definition, budget)
    fields_by_layer = tuple(
        _gas_only_activity_fields(
            setup=setup,
            temperature=float(temperature),
            pressure=float(pressure),
            budget=budget,
        )
        for temperature, pressure in zip(definition.temperatures, definition.pressures)
    )
    variants: list[SupportVariant] = []

    include_core_policies = set(str(value) for value in args.include_core_policies)
    exclude_core_policies = set(str(value) for value in args.exclude_core_policies)

    def add(label: str, group: str, policy: str, supports: Sequence[Sequence[int]], **meta: Any) -> None:
        if args.variant_groups != ["all"] and group not in set(args.variant_groups):
            return
        if group == "core":
            if include_core_policies and policy not in include_core_policies:
                return
            if policy in exclude_core_policies:
                return
        variants.append(
            SupportVariant(
                label=label,
                group=group,
                policy=policy,
                supports=tuple(tuple(int(index) for index in support) for support in supports),
                metadata=meta,
            )
        )

    add(
        "curated_base",
        "core",
        "curated_profile_support",
        tuple(base_support for _ in definition.temperatures),
        support_count=len(base_support),
    )

    for threshold in args.activity_thresholds:
        ordered = tuple(
            _ordered_activity_capacity(fields, float(threshold))
            for fields in fields_by_layer
        )
        for cap in args.caps:
            add(
                f"activity_capacity_t{_threshold_label(threshold)}_cap{cap}",
                "core",
                "gas_only_activity_positive_capacity_priority",
                tuple(_dedup_cap(indices, cap) for indices in ordered),
                threshold=float(threshold),
                cap=int(cap),
            )

    for cap in args.caps:
        add(
            f"activity_driving_top_cap{cap}",
            "core",
            "gas_only_activity_driving_top_cap",
            tuple(
                _dedup_cap(
                    _ordered_activity_driving(fields, threshold=None),
                    cap,
                )
                for fields in fields_by_layer
            ),
            cap=int(cap),
        )
        add(
            f"capacity_top_cap{cap}",
            "core",
            "temperature_valid_capacity_top_cap",
            tuple(_dedup_cap(_ordered_capacity(fields), cap) for fields in fields_by_layer),
            cap=int(cap),
        )

    oracle = _oracle_fastchem_supports(
        setup=setup,
        fastchem=fastchem,
        floor=float(args.fastchem_condensate_floor),
    )
    add(
        "oracle_fastchem_active",
        "oracle",
        "fastchem4_active_condensates_diagnostic_only",
        oracle,
        fastchem4_values_used=True,
        floor=float(args.fastchem_condensate_floor),
    )
    for cap in args.oracle_fill_caps:
        filled = []
        for support, fields in zip(oracle, fields_by_layer):
            tail = _ordered_activity_driving(fields, threshold=None)
            filled.append(_dedup_cap(tuple(support) + tuple(tail), cap))
        add(
            f"oracle_active_plus_driving_cap{cap}",
            "oracle",
            "fastchem4_active_plus_gas_activity_diagnostic_only",
            tuple(filled),
            fastchem4_values_used=True,
            cap=int(cap),
        )

    rng = np.random.default_rng(int(args.random_seed))
    for pool_size in args.random_pool_sizes:
        pools = tuple(
            _dedup_cap(_ordered_activity_driving(fields, threshold=None), int(pool_size))
            for fields in fields_by_layer
        )
        for cap in args.random_caps:
            if int(cap) > int(pool_size):
                continue
            for sample_index in range(int(args.random_samples)):
                sampled = []
                for pool in pools:
                    if len(pool) <= int(cap):
                        sampled.append(pool)
                        continue
                    choice = rng.choice(np.asarray(pool, dtype=np.int64), size=int(cap), replace=False)
                    sampled.append(tuple(int(index) for index in sorted(choice.tolist())))
                add(
                    f"random_driving_pool{pool_size}_cap{cap}_s{sample_index}",
                    "random",
                    "random_subset_of_gas_activity_driving_pool",
                    tuple(sampled),
                    pool_size=int(pool_size),
                    cap=int(cap),
                    sample_index=int(sample_index),
                    random_seed=int(args.random_seed),
                )
    return tuple(variants)


def _run_profile_for_variant(
    *,
    args: argparse.Namespace,
    setup: Any,
    definition: CuratedProfileDefinition,
    variant: SupportVariant,
) -> Any:
    budget = np.asarray(element_budget_for_profile(setup, definition), dtype=np.float64)
    inits = []
    support_diagnostics = []
    for layer_index, (temperature, pressure, support) in enumerate(
        zip(definition.temperatures, definition.pressures, variant.supports)
    ):
        gas = equilibrium(
            setup.gas_setup,
            float(temperature),
            float(pressure),
            budget,
            Pref=1.0,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        support_indices, support_amounts = seed_fixed_support_payload(
            setup=setup,
            element_inventory_target=budget,
            support_indices=support,
            seed_fraction=float(args.seed_fraction),
            max_seed_amount=float(args.max_seed_amount),
            min_seed_amount=AMOUNT_FLOOR,
        )
        inits.append(
            CondensateEquilibriumInit(
                gas_ln_n=gas.ln_n,
                gas_ntot=gas.ntot,
                support_indices=support_indices,
                support_amounts=support_amounts,
            )
        )
        support_diagnostics.append(
            {
                "layer_index": int(layer_index),
                "support_count": int(len(support_indices)),
                "support_indices": tuple(int(index) for index in support_indices),
                "support_species": tuple(
                    str(setup.condensate_species[int(index)])
                    for index in support_indices
                ),
            }
        )
    options = CondensateEquilibriumOptions(
        profile_method=args.exogibbs_method,
        profile_warm_start_support_policy="explicit_payload",
        return_diagnostics=True,
        allow_caveat_tiers=True,
        max_inner_iterations=int(args.max_inner_iterations),
        enable_head_route_warm_start=True,
        enable_depleted_gas_refresh=True,
        enable_full_condensate_budget_residual_gate=True,
        enable_experimental_profile_fixed_support_batch=True,
        enable_experimental_profile_fixed_support_fallback_rescue=False,
        enable_profile_native_activity_support_expansion=False,
        profile_fixed_support_seed_policy="budget_preserving_fraction",
        experimental_profile_fixed_support_rescue_residual_tolerance_multiplier=(
            float(args.fixed_support_rescue_residual_tolerance_multiplier)
        ),
        seed_fraction=float(args.seed_fraction),
        max_seed_amount=float(args.max_seed_amount),
    )
    plan = prepare_experimental_profile_fixed_support_batch_plan(
        setup,
        jnp.asarray(
            [
                definition.temperatures[index]
                for index, init in enumerate(inits)
                if init.support_indices
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [
                definition.pressures[index]
                for index, init in enumerate(inits)
                if init.support_indices
            ],
            dtype=jnp.float64,
        ),
        budget,
        init=tuple(init for init in inits if init.support_indices),
        support_indices=None,
        support_amounts_init=None,
        options=options,
    ) if any(init.support_indices for init in inits) else None
    arrays = None
    nonempty_original_indices = tuple(
        index for index, init in enumerate(inits) if init.support_indices
    )
    support_by_subset: list[tuple[int, ...] | None] = [None] * len(nonempty_original_indices)
    if plan is not None:
        arrays = run_experimental_profile_fixed_support_batch_plan(
            plan,
            rho_initialization="unit_activity",
            lambda_initialization="best_residual",
            residual_tolerance_multiplier=float(args.residual_tolerance_multiplier),
        )
        checkpoint_path = _solver_checkpoint_path(
            args.output_dir,
            definition.family,
            variant.label,
        )
        _write_solver_checkpoint(
            path=checkpoint_path,
            family=definition.family,
            variant=variant.label,
            arrays=arrays,
        )
        print(f"solver checkpoint: {checkpoint_path}")
        for bucket in plan.buckets:
            support_tuple = tuple(int(index) for index in bucket.support_indices)
            for subset_layer_index in bucket.layer_indices:
                support_by_subset[int(subset_layer_index)] = support_tuple
        gas_ln_n_batch = jnp.asarray(arrays["gas_ln_n"], dtype=jnp.float64)
        condensate_amounts_batch = jnp.asarray(
            arrays["condensate_amounts"],
            dtype=jnp.float64,
        )
        converged = np.asarray(jax.device_get(arrays["converged"]), dtype=bool)
        final_residual = np.asarray(
            jax.device_get(arrays["final_residual"]),
            dtype=np.float64,
        )
        n_iter = np.asarray(jax.device_get(arrays["n_iter"]), dtype=np.int64)
        residual_components_batch = {
            key: np.asarray(jax.device_get(value), dtype=np.float64)
            for key, value in arrays.get("residual_components", {}).items()
        }
        step_diagnostics_batch = {
            key: _device_array_tree(value)
            for key, value in arrays.get("step_diagnostics", {}).items()
        }
        continuation_diagnostics_batch = {
            key: np.asarray(jax.device_get(value))
            for key, value in arrays.get("continuation_diagnostics", {}).items()
        }
        epsilon_schedule = tuple(float(value) for value in arrays.get("epsilon_schedule", ()))
        lambda_candidate_labels = tuple(
            str(value) for value in arrays.get("lambda_candidate_labels", ())
        )
        stop_reason_labels = tuple(
            str(value) for value in arrays.get("stop_reason_labels", ())
        )
        residual_component_labels = tuple(
            str(value) for value in arrays.get("residual_component_labels", ())
        )
    else:
        gas_ln_n_batch = None
        condensate_amounts_batch = None
        converged = np.zeros((0,), dtype=bool)
        final_residual = np.zeros((0,), dtype=np.float64)
        n_iter = np.zeros((0,), dtype=np.int64)
        residual_components_batch = {}
        step_diagnostics_batch = {}
        continuation_diagnostics_batch = {}
        epsilon_schedule = ()
        lambda_candidate_labels = ()
        stop_reason_labels = ()
        residual_component_labels = ()
    subset_index_by_original = {
        original_index: subset_index
        for subset_index, original_index in enumerate(nonempty_original_indices)
    }
    layers = []
    for layer_index, support_diag in enumerate(support_diagnostics):
        subset_index = subset_index_by_original.get(layer_index)
        if subset_index is None:
            support_tuple = ()
            support_amounts = jnp.asarray((), dtype=jnp.float64)
            gas_ln_n = jnp.asarray(inits[layer_index].gas_ln_n, dtype=jnp.float64)
            solver_success = True
            layer_n_iter = 0
            layer_final_residual = 0.0
            route_name = "support_atlas_empty_support_gas_only"
            schema = "exogibbs_support_atlas_empty_support_gas_only_v1"
        else:
            support_tuple = support_by_subset[subset_index]
            if support_tuple is None:
                raise RuntimeError(f"missing support metadata for layer {layer_index}")
            support_index_array = jnp.asarray(support_tuple, dtype=jnp.int32)
            support_amounts = condensate_amounts_batch[subset_index][support_index_array]
            gas_ln_n = gas_ln_n_batch[subset_index]
            solver_success = bool(converged[subset_index])
            layer_n_iter = int(n_iter[subset_index])
            layer_final_residual = float(final_residual[subset_index])
            route_name = "support_atlas_fixed_support_batch"
            schema = "exogibbs_support_atlas_fixed_support_batch_v1"
        residual_components = (
            {
                key: float(values[subset_index])
                for key, values in residual_components_batch.items()
            }
            if subset_index is not None
            else {}
        )
        step_diagnostics = {}
        if subset_index is not None:
            for key, values in step_diagnostics_batch.items():
                if isinstance(values, dict):
                    step_diagnostics[key] = _slice_batch_tree(values, subset_index)
                else:
                    step_diagnostics[key] = _slice_batch_scalar(values, subset_index)
            lambda_index = step_diagnostics.get("lambda_selection_index")
            if (
                lambda_index is not None
                and 0 <= int(lambda_index) < len(lambda_candidate_labels)
            ):
                step_diagnostics["lambda_selection_label"] = (
                    lambda_candidate_labels[int(lambda_index)]
                )
            stop_reason_code = step_diagnostics.get("stop_reason_code")
            if (
                stop_reason_code is not None
                and 0 <= int(stop_reason_code) < len(stop_reason_labels)
            ):
                step_diagnostics["stop_reason_label"] = (
                    stop_reason_labels[int(stop_reason_code)]
                )
            component_index = step_diagnostics.get(
                "dominant_residual_component_index"
            )
            if (
                component_index is not None
                and 0 <= int(component_index) < len(residual_component_labels)
            ):
                step_diagnostics["dominant_residual_component_label"] = (
                    residual_component_labels[int(component_index)]
                )
            if "restoration_entry_residual_vector" in step_diagnostics:
                step_diagnostics["restoration_residual_diagnostic_labels"] = (
                    "full",
                    "gas",
                    "condensate_stationarity",
                    "budget_relative_max",
                    "complementarity",
                    "total_density",
                )
                restoration_type = step_diagnostics.get(
                    "restoration_first_normal_selected_type"
                )
                restoration_type_labels = (
                    "normal",
                    "stationarity_restoration",
                    "soc",
                    "not_recorded",
                )
                if restoration_type is not None and 0 <= int(
                    restoration_type
                ) < len(restoration_type_labels):
                    step_diagnostics[
                        "restoration_first_normal_selected_type_label"
                    ] = restoration_type_labels[int(restoration_type)]
        continuation_diagnostics = {"epsilon_schedule": epsilon_schedule, "stages": []}
        if subset_index is not None and continuation_diagnostics_batch:
            stage_count = len(epsilon_schedule)
            for stage_index in range(stage_count):
                stage = {}
                for key, values in continuation_diagnostics_batch.items():
                    value = values[subset_index, stage_index]
                    if np.asarray(value).ndim > 0:
                        stage[key] = _jsonable(value)
                    elif np.issubdtype(values.dtype, np.integer):
                        stage[key] = int(value)
                    elif np.issubdtype(values.dtype, np.bool_):
                        stage[key] = bool(value)
                    else:
                        stage[key] = float(value)
                stop_reason_code = stage.get("stop_reason_code")
                if (
                    stop_reason_code is not None
                    and 0 <= int(stop_reason_code) < len(stop_reason_labels)
                ):
                    stage["stop_reason_label"] = stop_reason_labels[
                        int(stop_reason_code)
                    ]
                component_index = stage.get("dominant_residual_component_index")
                if (
                    component_index is not None
                    and 0 <= int(component_index) < len(residual_component_labels)
                ):
                    stage["dominant_residual_component_label"] = (
                        residual_component_labels[int(component_index)]
                    )
                continuation_diagnostics["stages"].append(stage)
        diagnostics = {
            "experimental_profile_fixed_support_batch": {
                "schema": schema,
                "enabled": True,
                "layer_index": int(layer_index),
                "solver_success": solver_success,
                "max_iter": int(args.max_inner_iterations),
                "n_iter": layer_n_iter,
                "final_residual": layer_final_residual,
                "support_indices": support_tuple,
                "fallback_rescue_replaced": False,
                "support_atlas_no_route_fallback": True,
                "empty_support_gas_only": subset_index is None,
                "residual_components": residual_components,
                "step_diagnostics": step_diagnostics,
                "continuation_diagnostics": continuation_diagnostics,
            }
        }
        diagnostics["support_atlas_variant"] = {
            "label": variant.label,
            "group": variant.group,
            "policy": variant.policy,
            "metadata": dict(variant.metadata),
            "support": support_diag,
        }
        result_layer = build_condensate_equilibrium_result_from_solver_payload(
            setup=setup,
            gas_ln_n=gas_ln_n,
            support_indices=support_tuple,
            support_amounts=support_amounts,
            selected_route=route_name,
            metric_status=None,
            solver_success=solver_success,
            allow_caveat_tiers=True,
            diagnostics=diagnostics,
            element_inventory_target=budget,
            enable_full_condensate_budget_residual_gate=True,
            full_condensate_budget_relative_tolerance=1.0e-3,
        )
        layers.append(result_layer)
    diagnostics = {
        "profile_schema": "exogibbs_support_atlas_fixed_support_profile_v1",
        "method": "vmap_cold",
        "layer_count": len(layers),
        "experimental_profile_fixed_support_batch": {
            "schema": "exogibbs_support_atlas_fixed_support_batch_profile_v1",
            "accepted": bool(all(layer.converged for layer in layers)),
            "route": "support_atlas_fixed_support_batch",
            "fallback_rescue": None,
            "converged_count": int(np.count_nonzero(converged)),
            "nonempty_support_layer_count": int(len(nonempty_original_indices)),
            "empty_support_layer_count": int(len(inits) - len(nonempty_original_indices)),
        },
    }
    diagnostics["support_atlas_variant"] = {
        "label": variant.label,
        "group": variant.group,
        "policy": variant.policy,
        "metadata": dict(variant.metadata),
        "supports": tuple(support_diagnostics),
    }
    return SimpleNamespace(
        layers=tuple(layers),
        method="vmap_cold",
        diagnostics=diagnostics,
        batched_arrays=arrays if arrays is not None else {},
    )


def _compare_family_with_result(
    *,
    bench: Any,
    metric_args: argparse.Namespace,
    setup: Any,
    definition: CuratedProfileDefinition,
    fastchem: Mapping[str, Any],
    exogibbs: Any,
) -> dict[str, Any]:
    target_budget = np.asarray(element_budget_for_profile(setup, definition), dtype=np.float64)
    layers = []
    max_abs_log10_ratio = None
    floor_sweep_family_max = {
        bench._threshold_key(floor): None for floor in metric_args.gas_floor_sweep
    }
    major_gas_family_max = {
        bench._threshold_key(threshold): None for threshold in metric_args.major_gas_thresholds
    }
    major_overlap_gas_family_max = {
        bench._threshold_key(threshold): None for threshold in metric_args.major_gas_thresholds
    }
    jaccards = []
    for layer_index, layer in enumerate(exogibbs.layers):
        exo_gas_x = np.asarray(jax.device_get(layer.gas_x), dtype=np.float64)
        gas_metrics = bench._log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            floor=metric_args.gas_floor,
            top_n=metric_args.top_outliers,
        )
        gas_floor_sweep = bench._floor_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            floors=metric_args.gas_floor_sweep,
            top_n=metric_args.top_outliers,
        )
        major_gas_metrics = bench._major_species_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            abundance_thresholds=metric_args.major_gas_thresholds,
            ratio_floor=metric_args.major_gas_ratio_floor,
            selection_mode="union",
            top_n=metric_args.top_outliers,
        )
        major_overlap_gas_metrics = bench._major_species_sweep_log10_ratio_metrics(
            exo_names=tuple(setup.gas_species),
            exo_values=exo_gas_x,
            fc_names=fastchem["gas_names"],
            fc_values=fastchem["gas_x"][layer_index],
            abundance_thresholds=metric_args.major_gas_thresholds,
            ratio_floor=metric_args.major_gas_ratio_floor,
            selection_mode="overlap",
            top_n=metric_args.top_outliers,
        )
        if gas_metrics["max_abs_log10_ratio"] is not None:
            max_abs_log10_ratio = max(
                max_abs_log10_ratio or 0.0,
                float(gas_metrics["max_abs_log10_ratio"]),
            )
        bench._max_metric_by_key(floor_sweep_family_max, gas_floor_sweep, "max_abs_log10_ratio")
        bench._max_metric_by_key(major_gas_family_max, major_gas_metrics, "max_abs_log10_ratio")
        bench._max_metric_by_key(
            major_overlap_gas_family_max,
            major_overlap_gas_metrics,
            "max_abs_log10_ratio",
        )
        exo_active = bench._positive_names(
            tuple(setup.condensate_species),
            np.asarray(jax.device_get(layer.condensate_amounts), dtype=np.float64),
            floor=metric_args.exogibbs_condensate_floor,
        )
        fc_active = bench._positive_names(
            fastchem["condensate_names"],
            fastchem["condensate_number_densities"][layer_index],
            floor=metric_args.fastchem_condensate_floor,
        )
        jaccard = bench._jaccard(exo_active, fc_active)
        if jaccard is not None:
            jaccards.append(jaccard)
        exo_gas_n = np.asarray(jax.device_get(layer.gas_n), dtype=np.float64)
        exo_cond_n = np.asarray(jax.device_get(layer.condensate_amounts), dtype=np.float64)
        fc_scaled = bench._aligned_fastchem_amounts(
            setup=setup,
            fastchem=fastchem,
            layer_index=layer_index,
            target=target_budget,
        )
        exo_gibbs = bench._gibbs_over_rt(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            exo_gas_n,
            exo_cond_n,
        )
        fc_gibbs = bench._gibbs_over_rt(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
        )
        exo_budget = bench._budget_report(setup, exo_gas_n, exo_cond_n, target_budget)
        fc_budget = bench._budget_report(
            setup,
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
            target_budget,
        )
        exo_inactive = bench._inactive_driving_report(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            exo_gas_n,
            exo_cond_n,
        )
        fc_inactive = bench._inactive_driving_report(
            setup,
            float(definition.temperatures[layer_index]),
            float(definition.pressures[layer_index]),
            fc_scaled["gas_n"],
            fc_scaled["condensate_amounts"],
        )
        layer_payload = {
            "layer_index": int(layer_index),
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
            "exogibbs_diagnostics": _jsonable(layer.diagnostics or {}),
        }
        layers.append(layer_payload)
    deltas = [
        layer["gibbs_over_rt"]["delta_exogibbs_minus_fastchem4_scaled"]
        for layer in layers
        if math.isfinite(layer["gibbs_over_rt"]["delta_exogibbs_minus_fastchem4_scaled"])
    ]
    return {
        "family": definition.family,
        "rows": len(definition.temperatures),
        "exogibbs": bench._summarize_exogibbs_profile(exogibbs),
        "fastchem4": {
            "flag": fastchem["flag"],
            "flag_message": fastchem["flag_message"],
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
                "exogibbs_max_relative_residual": float(
                    np.max(
                        [
                            layer["budget"]["exogibbs"]["max_abs_relative_residual"]
                            for layer in layers
                        ]
                    )
                ),
                "fastchem4_scaled_max_relative_residual": float(
                    np.max(
                        [
                            layer["budget"]["fastchem4_scaled"][
                                "max_abs_relative_residual"
                            ]
                            for layer in layers
                        ]
                    )
                ),
            },
            "inactive_driving": {
                "exogibbs_temperature_valid_max": float(
                    np.max(
                        [
                            layer["inactive_driving"]["exogibbs"][
                                "temperature_valid_max_positive_inactive_driving"
                            ]
                            for layer in layers
                        ]
                    )
                ),
                "fastchem4_scaled_temperature_valid_max": float(
                    np.max(
                        [
                            layer["inactive_driving"]["fastchem4_scaled"][
                                "temperature_valid_max_positive_inactive_driving"
                            ]
                            for layer in layers
                        ]
                    )
                ),
            },
        },
        "layers": layers,
    }


def _metric_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        gas_floor=args.gas_floor,
        gas_floor_sweep=tuple(args.gas_floor_sweep),
        major_gas_thresholds=tuple(args.major_gas_thresholds),
        major_gas_ratio_floor=args.major_gas_ratio_floor,
        exogibbs_condensate_floor=args.exogibbs_condensate_floor,
        fastchem_condensate_floor=args.fastchem_condensate_floor,
        top_outliers=args.top_outliers,
    )


def _write_one(
    *,
    args: argparse.Namespace,
    path: Path,
    family: Mapping[str, Any],
    variant: SupportVariant,
) -> None:
    report = {
        "schema": "exogibbs_support_atlas_variant_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "jax_version": jax.__version__,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "config": {
            "family": family["family"],
            "variant": variant.label,
            "variant_group": variant.group,
            "variant_policy": variant.policy,
            "variant_metadata": dict(variant.metadata),
            "exogibbs_method": args.exogibbs_method,
            "max_inner_iterations": args.max_inner_iterations,
            "residual_tolerance_multiplier": args.residual_tolerance_multiplier,
            "fixed_support_rescue_residual_tolerance_multiplier": (
                args.fixed_support_rescue_residual_tolerance_multiplier
            ),
            "seed_fraction": args.seed_fraction,
            "max_seed_amount": args.max_seed_amount,
            "gas_floor_sweep": tuple(args.gas_floor_sweep),
            "major_gas_thresholds": tuple(args.major_gas_thresholds),
        },
        "provenance": {
            "support_atlas": True,
            "fastchem4_constructor_inputs_used": False,
            "fastchem4_exact_replay_target": False,
            "fastchem4_used_for_oracle_support": bool(
                variant.metadata.get("fastchem4_values_used", False)
            ),
        },
        "families": [family],
    }
    _atomic_write_text(
        path,
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
    )


def _summarize_output_dir(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {
            "index.json",
            "index_core.json",
            "index_oracle.json",
            "index_random.json",
            "index_stress.json",
            "summary.json",
            "best_supports.json",
        }:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "families" not in data:
            continue
        family = data["families"][0]
        comp = family["comparison"]
        rows.append(
            {
                "path": str(path),
                "family": family["family"],
                "variant": data["config"]["variant"],
                "group": data["config"]["variant_group"],
                "policy": data["config"]["variant_policy"],
                "major_overlap_1e8": comp["major_overlap_gas_max_abs_log10_ratio"].get(
                    "1e-08"
                ),
                "major_union_1e8": comp["major_gas_max_abs_log10_ratio"].get("1e-08"),
                "budget": comp["budget"]["exogibbs_max_relative_residual"],
                "inactive": comp["inactive_driving"]["exogibbs_temperature_valid_max"],
                "jaccard_mean": comp["active_condensate_jaccard_mean"],
                "converged": family["exogibbs"]["converged_count"],
                "rows": family["rows"],
                "support_max": family["exogibbs"].get("support_expansion_max_support_count"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["family"],
            float("inf") if row["major_overlap_1e8"] is None else row["major_overlap_1e8"],
            float("inf") if row["budget"] is None else row["budget"],
            float("inf") if row["inactive"] is None else row["inactive"],
        )
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable({"rows": rows}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Support Atlas Summary",
        "",
        "| family | variant | group | major overlap >=1e-8 dex | budget | inactive | converged | support max |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[: int(500)]:
        major = row["major_overlap_1e8"]
        lines.append(
            "| {family} | {variant} | {group} | {major} | {budget:.3g} | {inactive:.3g} | {conv}/{rows} | {support} |".format(
                family=row["family"],
                variant=row["variant"],
                group=row["group"],
                major="n/a" if major is None else f"{float(major):.4g}",
                budget=float(row["budget"]),
                inactive=float(row["inactive"]),
                conv=int(row["converged"]),
                rows=int(row["rows"]),
                support=row["support_max"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        type=int,
        default=None,
        help="Run only these zero-based source profile layers.",
    )
    parser.add_argument("--variant-groups", nargs="+", default=["core"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fastchem-root", type=Path, default=Path("FastChem"))
    parser.add_argument("--fastchem-element-file", default="input/element_abundances/asplund_2021.dat")
    parser.add_argument("--fastchem-gas-logk", default="input/logK/logK.dat")
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
        choices=("vmap_cold", "auto"),
        default="vmap_cold",
        help=(
            "Use vmap_cold by default so atlas variants measure the requested "
            "fixed support without auto fallback-rescue replacement."
        ),
    )
    parser.add_argument("--max-inner-iterations", type=int, default=100)
    parser.add_argument(
        "--residual-tolerance-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the fixed-support PD-IPM convergence tolerance. "
            "Use 1.0 for strict atlas solves; rescue tolerance is controlled "
            "separately."
        ),
    )
    parser.add_argument(
        "--fixed-support-rescue-residual-tolerance-multiplier",
        type=float,
        default=1.0e9,
    )
    parser.add_argument("--seed-fraction", type=float, default=0.8)
    parser.add_argument("--max-seed-amount", type=float, default=1.0)
    parser.add_argument("--caps", nargs="+", type=int, default=(8, 16, 24, 32, 48, 64, 96, 128))
    parser.add_argument(
        "--activity-thresholds",
        nargs="+",
        type=float,
        default=(0.0, -0.1, -1.0, -3.0),
    )
    parser.add_argument("--oracle-fill-caps", nargs="+", type=int, default=(16, 24, 32, 48, 64, 96, 128))
    parser.add_argument("--random-pool-sizes", nargs="+", type=int, default=(32, 64, 128))
    parser.add_argument("--random-caps", nargs="+", type=int, default=(16, 24, 32, 48, 64))
    parser.add_argument("--random-samples", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument(
        "--include-core-policies",
        nargs="+",
        default=(),
        help=(
            "If provided, generate only these core variant policies. Useful for "
            "broad screening runs that should avoid expensive stress policies."
        ),
    )
    parser.add_argument(
        "--exclude-core-policies",
        nargs="+",
        default=(),
        help=(
            "Core variant policies to skip. The variant labels remain available "
            "when not excluded, preserving existing diagnostic runs."
        ),
    )
    parser.add_argument("--gas-floor", type=float, default=1.0e-300)
    parser.add_argument("--gas-floor-sweep", nargs="+", type=float, default=(1.0e-300, 1.0e-30, 1.0e-20))
    parser.add_argument("--major-gas-thresholds", nargs="+", type=float, default=(1.0e-12, 1.0e-10, 1.0e-8))
    parser.add_argument("--major-gas-ratio-floor", type=float, default=1.0e-300)
    parser.add_argument("--exogibbs-condensate-floor", type=float, default=0.0)
    parser.add_argument("--fastchem-condensate-floor", type=float, default=0.0)
    parser.add_argument("--top-outliers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.update("jax_enable_x64", True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize_output_dir(args.output_dir)
        return
    bench = _load_benchmark_module()
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
    metric_args = _metric_args(args)
    index_rows = []
    for family_name in family_names:
        definition = FRESH_CURATED_PROFILES[family_name]
        definition, source_layer_indices = _subset_profile_definition(
            definition, args.layer_indices
        )
        print(f"== family {family_name} ==")
        fastchem_args = argparse.Namespace(
            fastchem_root=args.fastchem_root,
            fastchem_element_file=args.fastchem_element_file,
            fastchem_gas_logk=args.fastchem_gas_logk,
            fastchem_cond_logk=args.fastchem_cond_logk,
            fastchem_verbosity=args.fastchem_verbosity,
            fastchem_condensation=args.fastchem_condensation,
        )
        fastchem = bench._run_fastchem_profile(fastchem_args, definition)
        variants = _build_support_variants(
            args=args,
            setup=setup,
            definition=definition,
            fastchem=fastchem,
        )
        print(f"variants: {len(variants)}")
        for variant_index, variant in enumerate(variants):
            filename = f"{_safe_label(family_name)}__{_safe_label(variant.label)}.json"
            path = args.output_dir / filename
            if path.exists() and not args.force:
                print(f"skip {variant_index + 1}/{len(variants)} {variant.label}")
                index_rows.append({"family": family_name, "variant": variant.label, "path": str(path), "skipped": True})
                continue
            print(f"run {variant_index + 1}/{len(variants)} {variant.label}")
            result = _run_profile_for_variant(
                args=args,
                setup=setup,
                definition=definition,
                variant=variant,
            )
            family = _compare_family_with_result(
                bench=bench,
                metric_args=metric_args,
                setup=setup,
                definition=definition,
                fastchem=fastchem,
                exogibbs=result,
            )
            family["support_atlas_variant"] = {
                "label": variant.label,
                "group": variant.group,
                "policy": variant.policy,
                "metadata": dict(variant.metadata),
                "support_counts": [
                    int(len(support)) for support in variant.supports
                ],
                "source_layer_indices": source_layer_indices,
            }
            _write_one(args=args, path=path, family=family, variant=variant)
            checkpoint_path = _solver_checkpoint_path(
                args.output_dir,
                family_name,
                variant.label,
            )
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            index_rows.append({"family": family_name, "variant": variant.label, "path": str(path), "skipped": False})
    (args.output_dir / "index.json").write_text(
        json.dumps(_jsonable({"rows": index_rows}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _summarize_output_dir(args.output_dir)


if __name__ == "__main__":
    main()
