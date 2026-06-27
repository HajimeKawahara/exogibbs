"""Benchmark the opt-in API profile fixed-support batch route."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _bootstrap_jax_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "cuda", "gpu", "default"),
        default="cpu",
    )
    args, _ = parser.parse_known_args()
    platform = str(args.jax_platform)
    if platform != "default":
        os.environ["JAX_PLATFORMS"] = platform
        os.environ["JAX_PLATFORM_NAME"] = platform
    return platform


REQUESTED_JAX_PLATFORM = _bootstrap_jax_platform()
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "volatiles_code") not in sys.path:
    sys.path.insert(0, str(ROOT / "volatiles_code"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmark_pdipm_fixed_support_core import (  # noqa: E402
    _block_tree,
    _build_fixed_support_inputs,
    _jax_runtime_report,
    _json_safe,
)
from exogibbs.api.condensate_equilibrium import (  # noqa: E402
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    ExperimentalCondensateProfileFixedSupportPruneRescueCache,
    condensate_equilibrium,
    condensate_equilibrium_profile,
    prepare_experimental_profile_fixed_support_batch_plan,
    prepare_experimental_profile_fixed_support_prune_rescue_plan,
    run_experimental_profile_fixed_support_batch_plan,
    run_experimental_profile_fixed_support_batch_plan_with_cached_fallback_rescue,
    run_experimental_profile_fixed_support_batch_plan_with_prepared_fallback_rescue,
    run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue,
    run_experimental_profile_fixed_support_batch_plan_many,
    run_experimental_profile_fixed_support_batch_plan_many_with_cached_fallback_rescue,
    run_experimental_profile_fixed_support_batch_plan_many_with_prepared_fallback_rescue,
    run_experimental_profile_fixed_support_batch_plan_many_with_fallback_rescue,
)
from exogibbs.optimize import minimize_cond as condopt  # noqa: E402
from exogibbs.condensates.curated_profiles import FRESH_CURATED_PROFILES  # noqa: E402
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402


def _lifecycle_final_state(result: Any) -> Mapping[str, Any] | None:
    diagnostics = result.diagnostics or {}
    lifecycle = diagnostics.get("head_route_lifecycle")
    if not isinstance(lifecycle, Mapping):
        return None
    primary = lifecycle.get("primary_execution_report")
    if not isinstance(primary, Mapping):
        return None
    continuation = primary.get("continuation_report")
    if not isinstance(continuation, Mapping):
        return None
    final_state = continuation.get("final_state")
    return final_state if isinstance(final_state, Mapping) else None


def _block_profile_output(value: Any, mode: str) -> None:
    if mode == "batched" and getattr(value, "batched_arrays", None) is not None:
        _block_tree(value.batched_arrays)
        return
    _block_tree(value)


def _time(
    fn: Any,
    *,
    warmup: int,
    repeat: int,
    block_output: str,
) -> tuple[Any, dict[str, Any]]:
    first_start = time.perf_counter()
    first = fn()
    _block_profile_output(first, block_output)
    first_elapsed = time.perf_counter() - first_start
    for _ in range(warmup):
        _block_profile_output(fn(), block_output)
    times = []
    last = first
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn()
        _block_profile_output(last, block_output)
        times.append(time.perf_counter() - start)
    return last, {
        "first_call_seconds": first_elapsed,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "warm_call_seconds": times,
        "warm_median_seconds": statistics.median(times),
        "warm_min_seconds": min(times),
        "warm_max_seconds": max(times),
    }


def _family_inputs(setup: Any, family: str) -> tuple[list[Any], list[dict[str, Any]]]:
    definition = FRESH_CURATED_PROFILES[family]
    inputs = []
    skipped = []
    for layer_index in range(len(definition.temperatures)):
        try:
            inputs.append(
                _build_fixed_support_inputs(
                    setup=setup,
                    family=family,
                    layer_index=layer_index,
                    support_mode="explicit_payload",
                )
            )
        except ValueError as exc:
            skipped.append(
                {
                    "family": family,
                    "layer": int(layer_index),
                    "reason": str(exc),
                }
            )
    return inputs, skipped


def _activity_outer_profile_args(
    setup: Any,
    family: str,
    *,
    iterations: int,
    budget_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    definition = FRESH_CURATED_PROFILES[family]
    temperatures = np.asarray(definition.temperatures, dtype=np.float64)
    pressures = np.asarray(definition.pressures, dtype=np.float64)
    b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    layer_inits = []
    kept_temperatures = []
    kept_pressures = []
    skipped = []
    support_sizes = []
    selected_routes = []
    support_layers = []
    start = time.perf_counter()
    for layer_index, (temperature, pressure) in enumerate(
        zip(temperatures, pressures)
    ):
        try:
            result = condensate_equilibrium(
                setup,
                float(temperature),
                float(pressure),
                b,
                options=CondensateEquilibriumOptions(
                    max_inner_iterations=int(iterations),
                    enable_full_condensate_budget_residual_gate=budget_gate,
                    return_diagnostics=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - benchmark records skipped layers.
            skipped.append(
                {
                    "family": family,
                    "layer": int(layer_index),
                    "reason": f"activity_outer_failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        support_indices = tuple(
            int(value) for value in np.asarray(result.condensate_support_indices)
        )
        support_amounts = tuple(
            float(value)
            for value in np.asarray(result.condensate_amounts, dtype=np.float64)[
                list(support_indices)
            ]
        )
        final_state = _lifecycle_final_state(result)
        element_potential = None
        rho = None
        barrier_epsilon = None
        gas_stationarity_source = None
        if final_state is not None:
            lifecycle_payload = (result.diagnostics or {}).get(
                "head_route_lifecycle",
                {},
            )
            continuation_input = lifecycle_payload.get("continuation_input", {})
            if "gas_stationarity_source" in continuation_input:
                gas_stationarity_source = jnp.asarray(
                    continuation_input["gas_stationarity_source"],
                    dtype=jnp.float64,
                )
            if "element_potential" in final_state:
                element_potential = jnp.asarray(
                    final_state["element_potential"],
                    dtype=jnp.float64,
                )
            if "rho" in final_state:
                rho = jnp.asarray(final_state["rho"], dtype=jnp.float64)
                barrier_epsilon = jnp.asarray(
                    (
                        (result.diagnostics or {})
                        .get("head_route_lifecycle", {})
                        .get("primary_execution_report", {})
                        .get("continuation_report", {})
                        .get("final_epsilon", -10.0)
                    ),
                    dtype=jnp.float64,
                )
        if len(support_indices) == 0:
            skipped.append(
                {
                    "family": family,
                    "layer": int(layer_index),
                    "reason": "activity_outer_empty_support",
                }
            )
            continue
        gas_ln_n_array = np.asarray(jax.device_get(result.gas_ln_n), dtype=np.float64)
        gas_ntot_value = float(np.asarray(jax.device_get(result.gas_ntot)))
        support_amounts_array = np.asarray(support_amounts, dtype=np.float64)
        layer_summary = {
            "layer_index": int(layer_index),
            "selected_route": str(result.selected_route),
            "support_count": int(len(support_indices)),
            "support_indices": [int(value) for value in support_indices],
            "gas_ln_n_min": float(np.min(gas_ln_n_array)),
            "gas_ln_n_max": float(np.max(gas_ln_n_array)),
            "gas_ntot": gas_ntot_value,
            "support_amount_min": float(np.min(support_amounts_array)),
            "support_amount_max": float(np.max(support_amounts_array)),
            "support_amount_sum": float(np.sum(support_amounts_array)),
            "has_element_potential": element_potential is not None,
            "has_rho": rho is not None,
            "has_barrier_epsilon": barrier_epsilon is not None,
            "has_gas_stationarity_source": gas_stationarity_source is not None,
        }
        if element_potential is not None:
            element_potential_array = np.asarray(
                jax.device_get(element_potential),
                dtype=np.float64,
            )
            layer_summary.update(
                {
                    "element_potential_min": float(np.min(element_potential_array)),
                    "element_potential_max": float(np.max(element_potential_array)),
                }
            )
        if rho is not None:
            rho_array = np.asarray(jax.device_get(rho), dtype=np.float64)
            layer_summary.update(
                {
                    "rho_min": float(np.min(rho_array)),
                    "rho_max": float(np.max(rho_array)),
                }
            )
        if barrier_epsilon is not None:
            layer_summary["barrier_epsilon"] = float(
                np.asarray(jax.device_get(barrier_epsilon))
            )
        if gas_stationarity_source is not None:
            gas_source_array = np.asarray(
                jax.device_get(gas_stationarity_source),
                dtype=np.float64,
            )
            layer_summary.update(
                {
                    "gas_stationarity_source_min": float(np.min(gas_source_array)),
                    "gas_stationarity_source_max": float(np.max(gas_source_array)),
                    "gas_stationarity_source_l2": float(
                        np.linalg.norm(gas_source_array)
                    ),
                }
            )
        layer_inits.append(
            CondensateEquilibriumInit(
                gas_ln_n=jnp.asarray(result.gas_ln_n, dtype=jnp.float64),
                gas_ntot=jnp.asarray(result.gas_ntot, dtype=jnp.float64),
                support_indices=support_indices,
                support_amounts=support_amounts,
                element_potential=element_potential,
                rho=rho,
                barrier_epsilon=barrier_epsilon,
                gas_stationarity_source=gas_stationarity_source,
            )
        )
        kept_temperatures.append(float(temperature))
        kept_pressures.append(float(pressure))
        support_sizes.append(len(support_indices))
        selected_routes.append(result.selected_route)
        support_layers.append(layer_summary)
    if not layer_inits:
        raise ValueError(f"activity_outer produced no non-empty support for {family}")
    metadata = {
        "support_source": "activity_outer",
        "support_prepare_seconds": time.perf_counter() - start,
        "support_size_min": int(min(support_sizes)),
        "support_size_median": float(statistics.median(support_sizes)),
        "support_size_max": int(max(support_sizes)),
        "selected_routes": selected_routes,
        "support_layers": support_layers,
    }
    return (
        {
            "T": jnp.asarray(kept_temperatures, dtype=jnp.float64),
            "P": jnp.asarray(kept_pressures, dtype=jnp.float64),
            "b": b,
            "init": tuple(layer_inits),
        },
        skipped,
        metadata,
    )


def _profile_args(inputs: list[Any]) -> dict[str, Any]:
    first = inputs[0]
    return {
        "T": jnp.asarray([item.temperature for item in inputs], dtype=jnp.float64),
        "P": jnp.asarray([item.pressure for item in inputs], dtype=jnp.float64),
        "b": jnp.asarray(first.element_inventory_target, dtype=jnp.float64),
        "init": tuple(
            CondensateEquilibriumInit(
                gas_ln_n=jnp.asarray(item.q0, dtype=jnp.float64),
                gas_ntot=jnp.exp(jnp.asarray(item.qtot0, dtype=jnp.float64)),
                support_indices=tuple(int(v) for v in item.support_indices),
                support_amounts=tuple(float(v) for v in np.exp(item.r0).tolist()),
            )
            for item in inputs
        ),
    }


def _amount_map(init: CondensateEquilibriumInit) -> dict[int, float]:
    if init.support_indices is None or init.support_amounts is None:
        return {}
    return {
        int(index): float(amount)
        for index, amount in zip(init.support_indices, init.support_amounts)
    }


def _make_candidate_init(
    init: CondensateEquilibriumInit,
    support_indices: tuple[int, ...],
    support_amounts: tuple[float, ...],
    *,
    keep_activity_duals: bool,
) -> CondensateEquilibriumInit:
    return CondensateEquilibriumInit(
        gas_ln_n=init.gas_ln_n,
        gas_ntot=init.gas_ntot,
        support_indices=support_indices,
        support_amounts=support_amounts,
        element_potential=init.element_potential,
        rho=init.rho if keep_activity_duals else None,
        barrier_epsilon=init.barrier_epsilon,
        gas_stationarity_source=init.gas_stationarity_source,
    )


def _parse_prune_relative_floors(value: str) -> tuple[float, ...]:
    floors = tuple(
        float(item)
        for item in value.replace(";", ",").split(",")
        if item.strip()
    )
    if not floors:
        raise ValueError("--support-candidate-prune-floors must not be empty")
    if any(not np.isfinite(floor) or floor <= 0.0 for floor in floors):
        raise ValueError("--support-candidate-prune-floors must be positive finite values")
    return tuple(sorted(set(floors)))


def _support_candidate_profile_args(
    profile_args: Mapping[str, Any],
    *,
    mode: str,
    prune_relative_floors: tuple[float, ...],
    include_neighbor_union: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode == "current":
        n_layers = len(profile_args["init"])
        return (
            dict(profile_args),
            {
                "mode": "current",
                "original_layer_count": int(n_layers),
                "expanded_layer_count": int(n_layers),
                "candidate_count_by_layer": [1 for _ in range(n_layers)],
                "expanded_to_original_layer": [int(index) for index in range(n_layers)],
                "candidate_labels": ["current" for _ in range(n_layers)],
                "candidate_support_counts": [
                    len(tuple(init.support_indices or ()))
                    for init in profile_args["init"]
                ],
                "prune_relative_floors": [],
                "include_neighbor_union": False,
            },
        )
    if mode != "current_prune_neighbor":
        raise ValueError(f"unknown support candidate mode: {mode}")

    temperatures = list(np.asarray(profile_args["T"], dtype=np.float64))
    pressures = list(np.asarray(profile_args["P"], dtype=np.float64))
    inits = list(profile_args["init"])
    expanded_temperatures: list[float] = []
    expanded_pressures: list[float] = []
    expanded_inits: list[CondensateEquilibriumInit] = []
    expanded_to_original: list[int] = []
    candidate_labels: list[str] = []
    candidate_support_counts: list[int] = []
    candidate_count_by_layer: list[int] = []

    for layer_index, init in enumerate(inits):
        base_support = tuple(int(index) for index in tuple(init.support_indices or ()))
        base_amounts = tuple(float(amount) for amount in tuple(init.support_amounts or ()))
        if not base_support or not base_amounts:
            continue
        variants: list[tuple[str, tuple[int, ...], tuple[float, ...], bool]] = [
            ("current", base_support, base_amounts, True),
        ]
        amount_by_index = _amount_map(init)
        max_amount = max(base_amounts)
        for relative_floor in prune_relative_floors:
            prune_floor = max(1.0e-12, relative_floor * max_amount)
            pruned = tuple(
                index
                for index, amount in zip(base_support, base_amounts)
                if amount >= prune_floor
            )
            if pruned and pruned != base_support:
                variants.append(
                    (
                        f"prune_amount_ge_{relative_floor:g}_max",
                        pruned,
                        tuple(amount_by_index[index] for index in pruned),
                        False,
                    )
                )

        if include_neighbor_union:
            neighbor_amounts: dict[int, float] = {}
            for neighbor_index in (layer_index - 1, layer_index + 1):
                if neighbor_index < 0 or neighbor_index >= len(inits):
                    continue
                neighbor_amounts.update(_amount_map(inits[neighbor_index]))
            union_support = tuple(sorted(set(base_support).union(neighbor_amounts)))
            if union_support != base_support:
                fallback_amount = max(min(base_amounts), 1.0e-30)
                union_amounts = tuple(
                    amount_by_index.get(
                        index,
                        max(neighbor_amounts.get(index, fallback_amount), 1.0e-30),
                    )
                    for index in union_support
                )
                variants.append(("neighbor_union", union_support, union_amounts, False))

        seen: set[tuple[int, ...]] = set()
        kept_count = 0
        for label, support, amounts, keep_duals in variants:
            if support in seen:
                continue
            seen.add(support)
            expanded_temperatures.append(float(temperatures[layer_index]))
            expanded_pressures.append(float(pressures[layer_index]))
            expanded_inits.append(
                _make_candidate_init(
                    init,
                    support,
                    amounts,
                    keep_activity_duals=keep_duals,
                )
            )
            expanded_to_original.append(int(layer_index))
            candidate_labels.append(label)
            candidate_support_counts.append(len(support))
            kept_count += 1
        candidate_count_by_layer.append(kept_count)

    return (
        {
            "T": jnp.asarray(expanded_temperatures, dtype=jnp.float64),
            "P": jnp.asarray(expanded_pressures, dtype=jnp.float64),
            "b": profile_args["b"],
            "init": tuple(expanded_inits),
        },
        {
            "mode": mode,
            "original_layer_count": len(inits),
            "expanded_layer_count": len(expanded_inits),
            "candidate_count_by_layer": candidate_count_by_layer,
            "expanded_to_original_layer": expanded_to_original,
            "candidate_labels": candidate_labels,
            "candidate_support_counts": candidate_support_counts,
            "prune_relative_floors": list(prune_relative_floors),
            "include_neighbor_union": bool(include_neighbor_union),
        },
    )


def _subset_profile_args(
    profile_args: Mapping[str, Any],
    layer_indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "T": jnp.asarray(
            np.asarray(profile_args["T"], dtype=np.float64)[list(layer_indices)],
            dtype=jnp.float64,
        ),
        "P": jnp.asarray(
            np.asarray(profile_args["P"], dtype=np.float64)[list(layer_indices)],
            dtype=jnp.float64,
        ),
        "b": profile_args["b"],
        "init": tuple(profile_args["init"][index] for index in layer_indices),
    }


def _select_support_candidate_outputs(
    result: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    original_count = int(candidate_metadata["original_layer_count"])
    expanded_to_original = np.asarray(
        candidate_metadata["expanded_to_original_layer"],
        dtype=np.int64,
    )
    labels = np.asarray(candidate_metadata["candidate_labels"], dtype=object)
    support_counts = np.asarray(
        candidate_metadata["candidate_support_counts"],
        dtype=np.int64,
    )
    final_residual = np.asarray(jax.device_get(result["final_residual"]), dtype=np.float64)
    converged = np.asarray(jax.device_get(result["converged"]), dtype=bool)
    if final_residual.ndim == 1:
        eval_count = 1
        residual_view = final_residual[None, :]
        converged_view = converged[None, :]
    else:
        eval_count = int(final_residual.shape[0])
        residual_view = final_residual
        converged_view = converged

    best_expanded = np.zeros((eval_count, original_count), dtype=np.int64)
    selected_labels = np.empty((eval_count, original_count), dtype=object)
    selected_support_counts = np.zeros((eval_count, original_count), dtype=np.int64)
    for eval_index in range(eval_count):
        for original_index in range(original_count):
            candidates = np.where(expanded_to_original == original_index)[0]
            if candidates.size == 0:
                raise ValueError("support candidate metadata has an empty layer")
            candidate_converged = converged_view[eval_index, candidates]
            candidate_residual = residual_view[eval_index, candidates]
            score = np.where(candidate_converged, candidate_residual, np.inf)
            if not np.isfinite(score).any():
                score = np.where(np.isfinite(candidate_residual), candidate_residual, np.inf)
            selected = int(candidates[int(np.argmin(score))])
            best_expanded[eval_index, original_index] = selected
            selected_labels[eval_index, original_index] = labels[selected]
            selected_support_counts[eval_index, original_index] = support_counts[selected]

    def gather_layer_array(value: Any) -> Any:
        array = np.asarray(jax.device_get(value))
        if array.ndim == 0:
            return array
        if final_residual.ndim == 1:
            if array.shape[0] != expanded_to_original.shape[0]:
                return array
            return array[best_expanded[0]]
        if array.shape[0] != eval_count or array.shape[1] != expanded_to_original.shape[0]:
            return array
        trailing = array.shape[2:]
        gathered = np.empty((eval_count, original_count) + trailing, dtype=array.dtype)
        for eval_index in range(eval_count):
            gathered[eval_index] = array[eval_index, best_expanded[eval_index]]
        return gathered

    selected = {
        key: gather_layer_array(value)
        for key, value in result.items()
        if key not in {"residual_components", "step_diagnostics"}
    }
    selected["residual_components"] = {
        key: gather_layer_array(value)
        for key, value in result.get("residual_components", {}).items()
    }
    selected["step_diagnostics"] = {
        key: gather_layer_array(value)
        for key, value in result.get("step_diagnostics", {}).items()
    }
    selected["support_candidate_selection"] = {
        "selected_expanded_layer_index": (
            best_expanded[0] if final_residual.ndim == 1 else best_expanded
        ),
        "selected_candidate_label": (
            selected_labels[0].tolist()
            if final_residual.ndim == 1
            else selected_labels.tolist()
        ),
        "selected_support_count": (
            selected_support_counts[0]
            if final_residual.ndim == 1
            else selected_support_counts
        ),
    }
    return selected


def _maybe_select_support_candidate_outputs(
    result: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if candidate_metadata.get("mode") == "current":
        return result
    return _select_support_candidate_outputs(result, candidate_metadata)


def _fallback_layer_indices(result: Mapping[str, Any]) -> tuple[int, ...]:
    fallback_required = np.asarray(
        jax.device_get(result.get("fallback_required", ~result["converged"])),
        dtype=bool,
    )
    if fallback_required.ndim == 1:
        indices = np.where(fallback_required)[0]
    else:
        indices = np.where(np.any(fallback_required, axis=0))[0]
    return tuple(int(index) for index in indices.tolist())


def _merge_fallback_rescue_outputs(
    default_result: Mapping[str, Any],
    rescue_result: Mapping[str, Any],
    rescue_original_indices: tuple[int, ...],
) -> dict[str, Any]:
    default_residual = np.asarray(
        jax.device_get(default_result["final_residual"]),
        dtype=np.float64,
    )
    default_fallback = np.asarray(
        jax.device_get(default_result.get("fallback_required", ~default_result["converged"])),
        dtype=bool,
    )
    rescue_residual = np.asarray(
        jax.device_get(rescue_result["final_residual"]),
        dtype=np.float64,
    )
    rescue_converged = np.asarray(
        jax.device_get(rescue_result["converged"]),
        dtype=bool,
    )
    if default_residual.ndim == 1:
        eval_count = 1
        default_residual_view = default_residual[None, :]
        default_fallback_view = default_fallback[None, :]
        rescue_residual_view = rescue_residual[None, :]
        rescue_converged_view = rescue_converged[None, :]
    else:
        eval_count = int(default_residual.shape[0])
        default_residual_view = default_residual
        default_fallback_view = default_fallback
        rescue_residual_view = rescue_residual
        rescue_converged_view = rescue_converged
    n_layers = int(default_residual_view.shape[1])
    replace_mask = np.zeros((eval_count, n_layers), dtype=bool)
    rescue_by_original = {
        int(original_index): rescue_index
        for rescue_index, original_index in enumerate(rescue_original_indices)
    }
    for original_index, rescue_index in rescue_by_original.items():
        default_values = default_residual_view[:, original_index]
        rescue_values = rescue_residual_view[:, rescue_index]
        finite_improvement = np.isfinite(rescue_values) & (
            ~np.isfinite(default_values) | (rescue_values < default_values)
        )
        replace_mask[:, original_index] = (
            default_fallback_view[:, original_index]
            & (rescue_converged_view[:, rescue_index] | finite_improvement)
        )

    def merge_layer_array(
        default_value: Any,
        rescue_value: Any,
        *,
        force_replace: np.ndarray | None = None,
    ) -> Any:
        default_array = np.asarray(jax.device_get(default_value))
        rescue_array = np.asarray(jax.device_get(rescue_value))
        mask = replace_mask if force_replace is None else force_replace
        if default_residual.ndim == 1:
            if default_array.ndim == 0 or default_array.shape[0] != n_layers:
                return default_array
            merged = default_array.copy()
            for original_index, rescue_index in rescue_by_original.items():
                if mask[0, original_index]:
                    merged[original_index] = rescue_array[rescue_index]
            return merged
        if (
            default_array.ndim < 2
            or default_array.shape[0] != eval_count
            or default_array.shape[1] != n_layers
        ):
            return default_array
        merged = default_array.copy()
        for original_index, rescue_index in rescue_by_original.items():
            rows = mask[:, original_index]
            if np.any(rows):
                merged[rows, original_index] = rescue_array[rows, rescue_index]
        return merged

    merged = {
        key: merge_layer_array(value, rescue_result[key])
        for key, value in default_result.items()
        if (
            key in rescue_result
            and key not in {"residual_components", "step_diagnostics"}
        )
    }
    merged["residual_components"] = {
        key: merge_layer_array(value, rescue_result["residual_components"][key])
        for key, value in default_result.get("residual_components", {}).items()
        if key in rescue_result.get("residual_components", {})
    }
    merged["step_diagnostics"] = {
        key: merge_layer_array(value, rescue_result["step_diagnostics"][key])
        for key, value in default_result.get("step_diagnostics", {}).items()
        if key in rescue_result.get("step_diagnostics", {})
    }

    rescue_selection = rescue_result.get("support_candidate_selection", {})
    rescue_labels = rescue_selection.get("selected_candidate_label")
    rescue_support_counts = rescue_selection.get("selected_support_count")
    selected_labels = np.full((eval_count, n_layers), None, dtype=object)
    selected_support_counts = np.zeros((eval_count, n_layers), dtype=np.int64)
    if rescue_labels is not None and rescue_support_counts is not None:
        rescue_label_array = np.asarray(rescue_labels, dtype=object)
        rescue_support_array = np.asarray(jax.device_get(rescue_support_counts), dtype=np.int64)
        if default_residual.ndim == 1:
            rescue_label_array = rescue_label_array[None, :]
            rescue_support_array = rescue_support_array[None, :]
        for original_index, rescue_index in rescue_by_original.items():
            rows = replace_mask[:, original_index]
            if np.any(rows):
                selected_labels[rows, original_index] = rescue_label_array[rows, rescue_index]
                selected_support_counts[rows, original_index] = rescue_support_array[
                    rows,
                    rescue_index,
                ]
    merged["support_candidate_selection"] = {
        "fallback_rescue_original_layer_indices": list(rescue_original_indices),
        "fallback_rescue_replaced_count": int(np.count_nonzero(replace_mask)),
        "selected_candidate_label": (
            selected_labels[0].tolist()
            if default_residual.ndim == 1
            else selected_labels.tolist()
        ),
        "selected_support_count": (
            selected_support_counts[0]
            if default_residual.ndim == 1
            else selected_support_counts
        ),
        "replaced_by_fallback_rescue": (
            replace_mask[0] if default_residual.ndim == 1 else replace_mask
        ),
    }
    return merged


def _combine_timing(
    base: Mapping[str, Any],
    rescue: Mapping[str, Any] | None,
) -> dict[str, Any]:
    combined = dict(base)
    if rescue is None:
        combined["fallback_rescue_warm_median_seconds"] = 0.0
        return combined
    base_calls = list(base.get("warm_call_seconds", ()))
    rescue_calls = list(rescue.get("warm_call_seconds", ()))
    if len(base_calls) == len(rescue_calls):
        combined_calls = [
            float(base_value) + float(rescue_value)
            for base_value, rescue_value in zip(base_calls, rescue_calls)
        ]
    else:
        combined_calls = base_calls
    combined["first_call_seconds"] = float(base["first_call_seconds"]) + float(
        rescue["first_call_seconds"]
    )
    combined["warm_call_seconds"] = combined_calls
    combined["warm_median_seconds"] = float(base["warm_median_seconds"]) + float(
        rescue["warm_median_seconds"]
    )
    combined["warm_min_seconds"] = float(base["warm_min_seconds"]) + float(
        rescue["warm_min_seconds"]
    )
    combined["warm_max_seconds"] = float(base["warm_max_seconds"]) + float(
        rescue["warm_max_seconds"]
    )
    combined["fallback_rescue_warm_median_seconds"] = float(
        rescue["warm_median_seconds"]
    )
    return combined


def _run_profile(
    setup: Any,
    args: dict[str, Any],
    *,
    iterations: int,
    experimental: bool,
    return_diagnostics: bool,
    budget_gate: bool,
) -> Any:
    return condensate_equilibrium_profile(
        setup,
        args["T"],
        args["P"],
        args["b"],
        init=args["init"],
        options=CondensateEquilibriumOptions(
            profile_method="vmap_cold",
            profile_warm_start_support_policy="explicit_payload",
            enable_experimental_profile_fixed_support_batch=experimental,
            max_inner_iterations=int(iterations),
            enable_full_condensate_budget_residual_gate=budget_gate,
        ),
        return_diagnostics=return_diagnostics,
    )


def _run_plan_bucket_core(
    plan: Any,
    *,
    rho_initialization: str,
    lambda_initialization: str,
    residual_tolerance_multiplier: float,
) -> Any:
    return condopt._run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets(
        buckets=plan.buckets,
        formula_matrix=plan.formula_matrix,
        epsilon=-10.0,
        max_iter=plan.max_iter,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=float(residual_tolerance_multiplier),
    )


def _profile_delta(experimental_result: Any, baseline_result: Any) -> dict[str, Any]:
    max_gas_ln = 0.0
    max_cond = 0.0
    max_gas_n = 0.0
    for exp_layer, base_layer in zip(
        experimental_result.layers,
        baseline_result.layers,
    ):
        exp_gas_ln = np.asarray(jax.device_get(exp_layer.gas_ln_n), dtype=np.float64)
        base_gas_ln = np.asarray(jax.device_get(base_layer.gas_ln_n), dtype=np.float64)
        exp_cond = np.asarray(
            jax.device_get(exp_layer.condensate_amounts),
            dtype=np.float64,
        )
        base_cond = np.asarray(
            jax.device_get(base_layer.condensate_amounts),
            dtype=np.float64,
        )
        max_gas_ln = max(max_gas_ln, float(np.max(np.abs(exp_gas_ln - base_gas_ln))))
        max_cond = max(max_cond, float(np.max(np.abs(exp_cond - base_cond))))
        max_gas_n = max(
            max_gas_n,
            float(
                np.max(
                    np.abs(
                        np.exp(np.clip(exp_gas_ln, -745.0, 709.0))
                        - np.exp(np.clip(base_gas_ln, -745.0, 709.0))
                    )
                )
            ),
        )
    return {
        "max_gas_ln_n_abs_delta": max_gas_ln,
        "max_gas_n_abs_delta": max_gas_n,
        "max_condensate_amount_abs_delta": max_cond,
    }


def _profile_delta_from_arrays(
    experimental_arrays: Mapping[str, Any],
    baseline_result: Any,
) -> dict[str, Any]:
    exp_gas_ln_all = np.asarray(
        jax.device_get(experimental_arrays["gas_ln_n"]),
        dtype=np.float64,
    )
    exp_cond_all = np.asarray(
        jax.device_get(experimental_arrays["condensate_amounts"]),
        dtype=np.float64,
    )
    if exp_gas_ln_all.ndim == 3:
        exp_gas_ln_all = exp_gas_ln_all[0]
        exp_cond_all = exp_cond_all[0]
    max_gas_ln = 0.0
    max_cond = 0.0
    max_gas_n = 0.0
    for layer_index, base_layer in enumerate(baseline_result.layers):
        exp_gas_ln = exp_gas_ln_all[layer_index]
        exp_cond = exp_cond_all[layer_index]
        base_gas_ln = np.asarray(jax.device_get(base_layer.gas_ln_n), dtype=np.float64)
        base_cond = np.asarray(
            jax.device_get(base_layer.condensate_amounts),
            dtype=np.float64,
        )
        max_gas_ln = max(max_gas_ln, float(np.max(np.abs(exp_gas_ln - base_gas_ln))))
        max_cond = max(max_cond, float(np.max(np.abs(exp_cond - base_cond))))
        max_gas_n = max(
            max_gas_n,
            float(
                np.max(
                    np.abs(
                        np.exp(np.clip(exp_gas_ln, -745.0, 709.0))
                        - np.exp(np.clip(base_gas_ln, -745.0, 709.0))
                    )
                )
            ),
        )
    return {
        "max_gas_ln_n_abs_delta": max_gas_ln,
        "max_gas_n_abs_delta": max_gas_n,
        "max_condensate_amount_abs_delta": max_cond,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "cuda", "gpu", "default"),
        default=REQUESTED_JAX_PLATFORM,
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--return-diagnostics", action="store_true")
    parser.add_argument("--disable-budget-gate", action="store_true")
    parser.add_argument(
        "--support-source",
        choices=("explicit_payload", "activity_outer"),
        default="explicit_payload",
    )
    parser.add_argument(
        "--rho-initialization",
        choices=("unit_activity", "complementarity", "provided"),
        default="unit_activity",
    )
    parser.add_argument(
        "--lambda-initialization",
        choices=("gas_lstsq", "gas_cond_lstsq", "provided", "best_residual"),
        default="best_residual",
    )
    parser.add_argument("--residual-tolerance-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--block-output",
        choices=("layers", "batched"),
        default="layers",
    )
    parser.add_argument("--prepared-plan", action="store_true")
    parser.add_argument(
        "--element-inventory-scale",
        type=float,
        default=None,
        help="When using --prepared-plan, rerun the plan with b scaled by this factor.",
    )
    parser.add_argument(
        "--element-inventory-batch-size",
        type=int,
        default=1,
        help="When using --prepared-plan, evaluate this many b vectors together.",
    )
    parser.add_argument(
        "--element-inventory-batch-mode",
        choices=("scaled", "repeat"),
        default="scaled",
        help=(
            "When using --element-inventory-batch-size > 1, either perturb b "
            "across the batch or repeat the exact same b for saturation timing."
        ),
    )
    parser.add_argument(
        "--support-candidate-mode",
        choices=(
            "current",
            "current_prune_neighbor",
            "fallback_rescue_prune_neighbor",
            "api_fallback_rescue_prune",
            "api_prepared_fallback_rescue_prune",
            "api_cached_fallback_rescue_prune",
        ),
        default="current",
        help=(
            "When using --prepared-plan, optionally evaluate multiple support "
            "candidates per original layer and select the best residual result."
        ),
    )
    parser.add_argument(
        "--support-candidate-prune-floors",
        default="1e-9,1e-7,1e-6,1e-5,1e-4,1e-3",
        help=(
            "Comma-separated relative support amount floors for prune candidates. "
            "Each value is multiplied by the current layer's max support amount."
        ),
    )
    parser.add_argument(
        "--disable-support-candidate-neighbor-union",
        action="store_true",
        help="Disable the neighbor-union support candidate.",
    )
    parser.add_argument("--print-jax-devices", action="store_true")
    parser.add_argument(
        "--output",
        default="volatiles_artifacts/pdipm_api_profile_fixed_support_batch_benchmark.json",
    )
    args = parser.parse_args()
    runtime = _jax_runtime_report(str(args.jax_platform))
    if args.print_jax_devices:
        print(json.dumps(runtime, sort_keys=True), flush=True)

    setup = condensate_chemical_setup(silent=True)
    families = tuple(args.families or sorted(FRESH_CURATED_PROFILES))
    return_diagnostics = bool(args.return_diagnostics)
    budget_gate = not bool(args.disable_budget_gate)
    support_candidate_prune_floors = _parse_prune_relative_floors(
        str(args.support_candidate_prune_floors)
    )
    include_support_candidate_neighbor_union = not bool(
        args.disable_support_candidate_neighbor_union
    )
    rows = []
    skipped = []
    for family in families:
        support_metadata: dict[str, Any] = {"support_source": str(args.support_source)}
        if args.support_source == "activity_outer":
            try:
                profile_args, family_skipped, support_metadata = (
                    _activity_outer_profile_args(
                        setup,
                        family,
                        iterations=int(args.iterations),
                        budget_gate=budget_gate,
                    )
                )
            except ValueError as exc:
                skipped.append(
                    {
                        "family": family,
                        "layer": None,
                        "reason": str(exc),
                    }
                )
                continue
            inputs = list(profile_args["init"])
            skipped.extend(family_skipped)
        else:
            inputs, family_skipped = _family_inputs(setup, family)
            skipped.extend(family_skipped)
            if not inputs:
                continue
            profile_args = _profile_args(inputs)
        execution_profile_args = profile_args
        candidate_metadata = {
            "mode": "current",
            "original_layer_count": len(inputs),
            "expanded_layer_count": len(inputs),
            "candidate_count_by_layer": [1 for _ in inputs],
            "expanded_to_original_layer": [int(index) for index in range(len(inputs))],
            "candidate_labels": ["current" for _ in inputs],
            "candidate_support_counts": [
                len(tuple(init.support_indices or ()))
                for init in profile_args["init"]
            ],
        }
        support_candidate_mode = str(args.support_candidate_mode)
        if support_candidate_mode == "current_prune_neighbor":
            if not args.prepared_plan:
                raise ValueError("--support-candidate-mode requires --prepared-plan")
            execution_profile_args, candidate_metadata = _support_candidate_profile_args(
                profile_args,
                mode=support_candidate_mode,
                prune_relative_floors=support_candidate_prune_floors,
                include_neighbor_union=include_support_candidate_neighbor_union,
            )
            support_metadata["support_candidate_metadata"] = candidate_metadata
        elif support_candidate_mode in (
            "fallback_rescue_prune_neighbor",
            "api_fallback_rescue_prune",
            "api_prepared_fallback_rescue_prune",
            "api_cached_fallback_rescue_prune",
        ):
            if not args.prepared_plan:
                raise ValueError("--support-candidate-mode requires --prepared-plan")
        plan_prepare_seconds = None
        rescue_plan_prepare_seconds = None
        rescue_candidate_metadata = None
        rescue_timing = None
        fallback_rescue_layer_indices: tuple[int, ...] = ()
        prepared_bucket_core_timing = None
        if args.prepared_plan:
            plan_options = CondensateEquilibriumOptions(
                profile_method="vmap_cold",
                profile_warm_start_support_policy="explicit_payload",
                enable_experimental_profile_fixed_support_batch=True,
                max_inner_iterations=int(args.iterations),
                enable_full_condensate_budget_residual_gate=budget_gate,
            )
            start = time.perf_counter()
            plan = prepare_experimental_profile_fixed_support_batch_plan(
                setup,
                np.asarray(execution_profile_args["T"], dtype=np.float64),
                np.asarray(execution_profile_args["P"], dtype=np.float64),
                execution_profile_args["b"],
                Pref=1.0,
                init=execution_profile_args["init"],
                initializer=None,
                support_indices=None,
                support_amounts_init=None,
                options=plan_options,
            )
            _block_tree(plan.buckets)
            plan_prepare_seconds = time.perf_counter() - start
            api_rescue_cache = None
            if support_candidate_mode == "api_cached_fallback_rescue_prune":
                api_rescue_cache = (
                    ExperimentalCondensateProfileFixedSupportPruneRescueCache()
                )
            _bucket_core_last, prepared_bucket_core_timing = _time(
                lambda plan=plan: _run_plan_bucket_core(
                    plan,
                    rho_initialization=str(args.rho_initialization),
                    lambda_initialization=str(args.lambda_initialization),
                    residual_tolerance_multiplier=float(
                        args.residual_tolerance_multiplier
                    ),
                ),
                warmup=int(args.warmup),
                repeat=int(args.repeat),
                block_output="layers",
            )
            batched_targets = None
            if int(args.element_inventory_batch_size) > 1:
                base_target = jnp.asarray(profile_args["b"], dtype=jnp.float64)
                if str(args.element_inventory_batch_mode) == "repeat":
                    scale0 = (
                        1.0
                        if args.element_inventory_scale is None
                        else float(args.element_inventory_scale)
                    )
                    batched_targets = jnp.broadcast_to(
                        scale0 * base_target[None, :],
                        (int(args.element_inventory_batch_size), base_target.shape[0]),
                    )
                else:
                    scale0 = (
                        1.0
                        if args.element_inventory_scale is None
                        else float(args.element_inventory_scale)
                    )
                    scales = jnp.asarray(
                        [
                            scale0 * (1.0 + 1.0e-4 * index)
                            for index in range(int(args.element_inventory_batch_size))
                        ],
                        dtype=jnp.float64,
                    )
                    batched_targets = scales[:, None] * base_target
                prepared_api_rescue = None
                if support_candidate_mode == "api_prepared_fallback_rescue_prune":
                    probe_arrays = run_experimental_profile_fixed_support_batch_plan_many(
                        plan,
                        batched_targets,
                        rho_initialization=str(args.rho_initialization),
                        lambda_initialization=str(args.lambda_initialization),
                        residual_tolerance_multiplier=float(
                            args.residual_tolerance_multiplier
                        ),
                    )
                    _block_tree(probe_arrays)
                    fallback_rescue_layer_indices = _fallback_layer_indices(
                        probe_arrays
                    )
                    start = time.perf_counter()
                    prepared_api_rescue = (
                        prepare_experimental_profile_fixed_support_prune_rescue_plan(
                            plan,
                            fallback_rescue_layer_indices,
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                    )
                    if prepared_api_rescue.rescue_plan is not None:
                        _block_tree(prepared_api_rescue.rescue_plan.buckets)
                    rescue_plan_prepare_seconds = time.perf_counter() - start
                    rescue_candidate_metadata = prepared_api_rescue.metadata
                experimental_last, experimental_timing = _time(
                    lambda plan=plan, batched_targets=batched_targets, prepared_api_rescue=prepared_api_rescue, api_rescue_cache=api_rescue_cache: (
                        run_experimental_profile_fixed_support_batch_plan_many_with_fallback_rescue(
                            plan,
                            batched_targets,
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                        if support_candidate_mode == "api_fallback_rescue_prune"
                        else run_experimental_profile_fixed_support_batch_plan_many_with_cached_fallback_rescue(
                            plan,
                            api_rescue_cache,
                            batched_targets,
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                        if support_candidate_mode == "api_cached_fallback_rescue_prune"
                        else run_experimental_profile_fixed_support_batch_plan_many_with_prepared_fallback_rescue(
                            plan,
                            prepared_api_rescue,
                            batched_targets,
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                        )
                        if support_candidate_mode
                        == "api_prepared_fallback_rescue_prune"
                        else _maybe_select_support_candidate_outputs(
                            run_experimental_profile_fixed_support_batch_plan_many(
                                plan,
                                batched_targets,
                                rho_initialization=str(args.rho_initialization),
                                lambda_initialization=str(args.lambda_initialization),
                                residual_tolerance_multiplier=float(
                                    args.residual_tolerance_multiplier
                                ),
                            ),
                            candidate_metadata,
                        )
                    ),
                    warmup=int(args.warmup),
                    repeat=int(args.repeat),
                    block_output="layers",
                )
                if support_candidate_mode == "fallback_rescue_prune_neighbor":
                    fallback_rescue_layer_indices = _fallback_layer_indices(
                        experimental_last
                    )
                    if fallback_rescue_layer_indices:
                        rescue_profile_args = _subset_profile_args(
                            profile_args,
                            fallback_rescue_layer_indices,
                        )
                        (
                            rescue_execution_profile_args,
                            rescue_candidate_metadata,
                        ) = _support_candidate_profile_args(
                            rescue_profile_args,
                            mode="current_prune_neighbor",
                            prune_relative_floors=support_candidate_prune_floors,
                            include_neighbor_union=(
                                include_support_candidate_neighbor_union
                            ),
                        )
                        start = time.perf_counter()
                        rescue_plan = prepare_experimental_profile_fixed_support_batch_plan(
                            setup,
                            np.asarray(
                                rescue_execution_profile_args["T"],
                                dtype=np.float64,
                            ),
                            np.asarray(
                                rescue_execution_profile_args["P"],
                                dtype=np.float64,
                            ),
                            rescue_execution_profile_args["b"],
                            Pref=1.0,
                            init=rescue_execution_profile_args["init"],
                            initializer=None,
                            support_indices=None,
                            support_amounts_init=None,
                            options=plan_options,
                        )
                        _block_tree(rescue_plan.buckets)
                        rescue_plan_prepare_seconds = time.perf_counter() - start
                        rescue_last, rescue_timing = _time(
                            lambda rescue_plan=rescue_plan, batched_targets=batched_targets: (
                                _select_support_candidate_outputs(
                                    run_experimental_profile_fixed_support_batch_plan_many(
                                        rescue_plan,
                                        batched_targets,
                                        rho_initialization=str(args.rho_initialization),
                                        lambda_initialization=str(
                                            args.lambda_initialization
                                        ),
                                        residual_tolerance_multiplier=float(
                                            args.residual_tolerance_multiplier
                                        ),
                                    ),
                                    rescue_candidate_metadata,
                                )
                            ),
                            warmup=int(args.warmup),
                            repeat=int(args.repeat),
                            block_output="layers",
                        )
                        experimental_last = _merge_fallback_rescue_outputs(
                            experimental_last,
                            rescue_last,
                            fallback_rescue_layer_indices,
                        )
                        experimental_timing = _combine_timing(
                            experimental_timing,
                            rescue_timing,
                        )
                    else:
                        experimental_timing = _combine_timing(
                            experimental_timing,
                            None,
                        )
            else:
                prepared_api_rescue = None
                if support_candidate_mode == "api_prepared_fallback_rescue_prune":
                    probe_arrays = run_experimental_profile_fixed_support_batch_plan(
                        plan,
                        element_inventory_target=(
                            None
                            if args.element_inventory_scale is None
                            else profile_args["b"] * float(args.element_inventory_scale)
                        ),
                        rho_initialization=str(args.rho_initialization),
                        lambda_initialization=str(args.lambda_initialization),
                        residual_tolerance_multiplier=float(
                            args.residual_tolerance_multiplier
                        ),
                    )
                    _block_tree(probe_arrays)
                    fallback_rescue_layer_indices = _fallback_layer_indices(
                        probe_arrays
                    )
                    start = time.perf_counter()
                    prepared_api_rescue = (
                        prepare_experimental_profile_fixed_support_prune_rescue_plan(
                            plan,
                            fallback_rescue_layer_indices,
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                    )
                    if prepared_api_rescue.rescue_plan is not None:
                        _block_tree(prepared_api_rescue.rescue_plan.buckets)
                    rescue_plan_prepare_seconds = time.perf_counter() - start
                    rescue_candidate_metadata = prepared_api_rescue.metadata
                experimental_last, experimental_timing = _time(
                    lambda plan=plan, profile_args=profile_args, prepared_api_rescue=prepared_api_rescue, api_rescue_cache=api_rescue_cache: (
                        run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue(
                            plan,
                            element_inventory_target=(
                                None
                                if args.element_inventory_scale is None
                                else profile_args["b"]
                                * float(args.element_inventory_scale)
                            ),
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                        if support_candidate_mode == "api_fallback_rescue_prune"
                        else run_experimental_profile_fixed_support_batch_plan_with_cached_fallback_rescue(
                            plan,
                            api_rescue_cache,
                            element_inventory_target=(
                                None
                                if args.element_inventory_scale is None
                                else profile_args["b"] * float(args.element_inventory_scale)
                            ),
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                            prune_relative_floors=support_candidate_prune_floors,
                        )
                        if support_candidate_mode == "api_cached_fallback_rescue_prune"
                        else run_experimental_profile_fixed_support_batch_plan_with_prepared_fallback_rescue(
                            plan,
                            prepared_api_rescue,
                            element_inventory_target=(
                                None
                                if args.element_inventory_scale is None
                                else profile_args["b"] * float(args.element_inventory_scale)
                            ),
                            rho_initialization=str(args.rho_initialization),
                            lambda_initialization=str(args.lambda_initialization),
                            residual_tolerance_multiplier=float(
                                args.residual_tolerance_multiplier
                            ),
                        )
                        if support_candidate_mode
                        == "api_prepared_fallback_rescue_prune"
                        else _maybe_select_support_candidate_outputs(
                            run_experimental_profile_fixed_support_batch_plan(
                                plan,
                                element_inventory_target=(
                                    None
                                    if args.element_inventory_scale is None
                                    else profile_args["b"]
                                    * float(args.element_inventory_scale)
                                ),
                                rho_initialization=str(args.rho_initialization),
                                lambda_initialization=str(args.lambda_initialization),
                                residual_tolerance_multiplier=float(
                                    args.residual_tolerance_multiplier
                                ),
                            ),
                            candidate_metadata,
                        )
                    ),
                    warmup=int(args.warmup),
                    repeat=int(args.repeat),
                    block_output="layers",
                )
                if support_candidate_mode == "fallback_rescue_prune_neighbor":
                    fallback_rescue_layer_indices = _fallback_layer_indices(
                        experimental_last
                    )
                    if fallback_rescue_layer_indices:
                        rescue_profile_args = _subset_profile_args(
                            profile_args,
                            fallback_rescue_layer_indices,
                        )
                        (
                            rescue_execution_profile_args,
                            rescue_candidate_metadata,
                        ) = _support_candidate_profile_args(
                            rescue_profile_args,
                            mode="current_prune_neighbor",
                            prune_relative_floors=support_candidate_prune_floors,
                            include_neighbor_union=(
                                include_support_candidate_neighbor_union
                            ),
                        )
                        start = time.perf_counter()
                        rescue_plan = prepare_experimental_profile_fixed_support_batch_plan(
                            setup,
                            np.asarray(
                                rescue_execution_profile_args["T"],
                                dtype=np.float64,
                            ),
                            np.asarray(
                                rescue_execution_profile_args["P"],
                                dtype=np.float64,
                            ),
                            rescue_execution_profile_args["b"],
                            Pref=1.0,
                            init=rescue_execution_profile_args["init"],
                            initializer=None,
                            support_indices=None,
                            support_amounts_init=None,
                            options=plan_options,
                        )
                        _block_tree(rescue_plan.buckets)
                        rescue_plan_prepare_seconds = time.perf_counter() - start
                        rescue_last, rescue_timing = _time(
                            lambda rescue_plan=rescue_plan, profile_args=profile_args: (
                                _select_support_candidate_outputs(
                                    run_experimental_profile_fixed_support_batch_plan(
                                        rescue_plan,
                                        element_inventory_target=(
                                            None
                                            if args.element_inventory_scale is None
                                            else profile_args["b"]
                                            * float(args.element_inventory_scale)
                                        ),
                                        rho_initialization=str(args.rho_initialization),
                                        lambda_initialization=str(
                                            args.lambda_initialization
                                        ),
                                        residual_tolerance_multiplier=float(
                                            args.residual_tolerance_multiplier
                                        ),
                                    ),
                                    rescue_candidate_metadata,
                                )
                            ),
                            warmup=int(args.warmup),
                            repeat=int(args.repeat),
                            block_output="layers",
                        )
                        experimental_last = _merge_fallback_rescue_outputs(
                            experimental_last,
                            rescue_last,
                            fallback_rescue_layer_indices,
                        )
                        experimental_timing = _combine_timing(
                            experimental_timing,
                            rescue_timing,
                        )
                    else:
                        experimental_timing = _combine_timing(
                            experimental_timing,
                            None,
                        )
        else:
            experimental_last, experimental_timing = _time(
                lambda profile_args=profile_args: _run_profile(
                    setup,
                    profile_args,
                    iterations=int(args.iterations),
                    experimental=True,
                    return_diagnostics=return_diagnostics,
                    budget_gate=budget_gate,
                ),
                warmup=int(args.warmup),
                repeat=int(args.repeat),
                block_output=str(args.block_output),
            )
        baseline_timing = None
        delta = None
        if not args.skip_baseline:
            baseline_last, baseline_timing = _time(
                lambda profile_args=profile_args: _run_profile(
                    setup,
                    profile_args,
                    iterations=int(args.iterations),
                    experimental=False,
                    return_diagnostics=return_diagnostics,
                    budget_gate=budget_gate,
                ),
                warmup=0,
                repeat=max(1, min(2, int(args.repeat))),
                block_output=str(args.block_output),
            )
            delta = (
                _profile_delta_from_arrays(experimental_last, baseline_last)
                if args.prepared_plan
                else _profile_delta(experimental_last, baseline_last)
            )
        if args.prepared_plan:
            exp_diag = {}
            converged = np.asarray(jax.device_get(experimental_last["converged"]))
            fallback_required = np.asarray(
                jax.device_get(
                    experimental_last.get(
                        "fallback_required",
                        ~experimental_last["converged"],
                    )
                )
            )
            final_residual = np.asarray(
                jax.device_get(experimental_last["final_residual"]),
                dtype=np.float64,
            )
            n_iter = np.asarray(
                jax.device_get(experimental_last["n_iter"]),
                dtype=np.int64,
            )
            residual_components = experimental_last.get("residual_components", {})
            residual_component_arrays = {
                str(name): np.asarray(jax.device_get(values), dtype=np.float64)
                for name, values in residual_components.items()
            }
            step_diagnostics = experimental_last.get("step_diagnostics", {})
            step_diagnostic_arrays = {
                str(name): np.asarray(jax.device_get(values))
                for name, values in step_diagnostics.items()
            }
            support_candidate_selection = experimental_last.get(
                "support_candidate_selection",
                {},
            )
            if not support_candidate_selection:
                support_candidate_selection = experimental_last.get(
                    "fallback_rescue",
                    {},
                )
            selected_candidate_labels = support_candidate_selection.get(
                "selected_candidate_label"
            )
            selected_support_counts = support_candidate_selection.get(
                "selected_support_count"
            )
            if selected_support_counts is not None:
                selected_support_counts = np.asarray(
                    jax.device_get(selected_support_counts),
                    dtype=np.int64,
                )

            def selected_label_at(*indices: int) -> str | None:
                if selected_candidate_labels is None:
                    return None
                value = selected_candidate_labels
                for index in indices:
                    value = value[index]
                return None if value is None else str(value)

            def selected_support_count_at(*indices: int) -> int | None:
                if selected_support_counts is None or selected_label_at(*indices) is None:
                    return None
                return int(selected_support_counts[indices])

            converged_count = int(np.count_nonzero(converged))
            fallback_required_count = int(np.count_nonzero(fallback_required))
            fast_path_layer_count = len(inputs) * int(args.element_inventory_batch_size)
            layer_indices = list(range(len(inputs)))
            if final_residual.ndim == 2:
                per_layer = []
                for eval_index in range(final_residual.shape[0]):
                    per_layer.extend(
                        {
                            "eval_index": int(eval_index),
                            "layer_index": int(layer_index),
                            "converged": bool(converged[eval_index, layer_index]),
                            "fallback_required": bool(
                                fallback_required[eval_index, layer_index]
                            ),
                            "selected_support_candidate": selected_label_at(
                                eval_index,
                                layer_index,
                            ),
                            "selected_support_count": selected_support_count_at(
                                eval_index,
                                layer_index,
                            ),
                            "final_residual": float(
                                final_residual[eval_index, layer_index]
                            ),
                            "residual_components": {
                                name: float(values[eval_index, layer_index])
                                for name, values in residual_component_arrays.items()
                            },
                            "step_diagnostics": {
                                name: (
                                    float(values[eval_index, layer_index])
                                    if np.issubdtype(values.dtype, np.floating)
                                    else int(values[eval_index, layer_index])
                                )
                                for name, values in step_diagnostic_arrays.items()
                            },
                            "n_iter": int(n_iter[eval_index, layer_index]),
                        }
                        for layer_index in layer_indices
                    )
                per_layer_summary = per_layer
            else:
                per_layer_summary = [
                    {
                        "layer_index": int(layer_index),
                        "converged": bool(converged[layer_index]),
                        "fallback_required": bool(fallback_required[layer_index]),
                        "selected_support_candidate": selected_label_at(layer_index),
                        "selected_support_count": selected_support_count_at(layer_index),
                        "final_residual": float(final_residual[layer_index]),
                        "residual_components": {
                            name: float(values[layer_index])
                            for name, values in residual_component_arrays.items()
                        },
                        "step_diagnostics": {
                            name: (
                                float(values[layer_index])
                                if np.issubdtype(values.dtype, np.floating)
                                else int(values[layer_index])
                            )
                            for name, values in step_diagnostic_arrays.items()
                        },
                        "n_iter": int(n_iter[layer_index]),
                    }
                    for layer_index in layer_indices
                ]
            finite_residual = final_residual[np.isfinite(final_residual)]
            residual_summary = {
                "finite_final_residual_count": int(finite_residual.size),
                "nonfinite_final_residual_count": int(
                    final_residual.size - finite_residual.size
                ),
                "min_finite_final_residual": None
                if finite_residual.size == 0
                else float(np.min(finite_residual)),
                "median_finite_final_residual": None
                if finite_residual.size == 0
                else float(np.median(finite_residual)),
                "max_finite_final_residual": None
                if finite_residual.size == 0
                else float(np.max(finite_residual)),
                "min_n_iter": int(np.min(n_iter)),
                "median_n_iter": float(np.median(n_iter)),
                "max_n_iter": int(np.max(n_iter)),
            }
        else:
            exp_diag = experimental_last.diagnostics or {}
            converged_count = sum(bool(layer.converged) for layer in experimental_last.layers)
            fallback_required_count = sum(
                not bool(layer.converged) for layer in experimental_last.layers
            )
            fast_path_layer_count = sum(
                layer.selected_route == "experimental_profile_fixed_support_batch"
                or bool(
                    (layer.diagnostics or {})
                    .get("experimental_profile_fixed_support_batch", {})
                    .get("enabled", False)
                )
                for layer in experimental_last.layers
            )
            residual_summary = None
            per_layer_summary = None
        api_fallback_rescue = (
            experimental_last.get("fallback_rescue") if args.prepared_plan else None
        )
        fallback_rescue_layer_indices_for_row = list(fallback_rescue_layer_indices)
        rescue_candidate_expanded_layer_count = 0
        if rescue_candidate_metadata is not None:
            rescue_candidate_expanded_layer_count = int(
                rescue_candidate_metadata["expanded_layer_count"]
            )
        elif api_fallback_rescue is not None:
            rescue_candidate_expanded_layer_count = int(
                api_fallback_rescue.get("expanded_layer_count", 0)
            )
            fallback_rescue_layer_indices_for_row = list(
                api_fallback_rescue.get("fallback_layer_indices", ())
            )
        total_executed_candidate_layer_count = (
            int(candidate_metadata["expanded_layer_count"])
            + rescue_candidate_expanded_layer_count
        )
        fallback_rescue_cache = None
        if api_rescue_cache is not None:
            fallback_rescue_cache = {
                "prepare_count": int(api_rescue_cache.prepare_count),
                "hit_count": int(api_rescue_cache.hit_count),
                "cached_plan_count": int(len(api_rescue_cache.plans)),
            }
        row = {
            "family": family,
            "layer_count": len(inputs),
            "support_metadata": support_metadata,
            "support_candidate_metadata": candidate_metadata,
            "api_fallback_rescue": api_fallback_rescue,
            "support_candidate_expanded_layer_count": int(
                candidate_metadata["expanded_layer_count"]
            ),
            "fallback_rescue_layer_indices": fallback_rescue_layer_indices_for_row,
            "fallback_rescue_candidate_metadata": rescue_candidate_metadata,
            "fallback_rescue_candidate_expanded_layer_count": (
                rescue_candidate_expanded_layer_count
            ),
            "fallback_rescue_cache": fallback_rescue_cache,
            "total_executed_candidate_layer_count": total_executed_candidate_layer_count,
            "rho_initialization": str(args.rho_initialization),
            "lambda_initialization": str(args.lambda_initialization),
            "lambda_candidate_labels": tuple(
                str(value)
                for value in experimental_last.get(
                    "lambda_candidate_labels",
                    condopt.FIXED_SUPPORT_BATCH_LAMBDA_CANDIDATE_LABELS,
                )
            )
            if args.prepared_plan
            else None,
            "residual_tolerance_multiplier": float(
                args.residual_tolerance_multiplier
            ),
            "element_inventory_batch_size": int(args.element_inventory_batch_size),
            "element_inventory_batch_mode": str(args.element_inventory_batch_mode),
            "experimental_converged_count": converged_count,
            "experimental_fallback_required_count": fallback_required_count,
            "experimental_fast_path_layer_count": fast_path_layer_count,
            "experimental_residual_summary": residual_summary,
            "experimental_per_layer": per_layer_summary,
            "prepared_plan": bool(args.prepared_plan),
            "plan_prepare_seconds": plan_prepare_seconds,
            "fallback_rescue_plan_prepare_seconds": rescue_plan_prepare_seconds,
            "prepared_bucket_core": prepared_bucket_core_timing,
            "skipped_layers": family_skipped,
            "experimental": experimental_timing,
            "fallback_rescue_experimental": rescue_timing,
            "baseline": baseline_timing,
            "speedup_vs_baseline": None
            if baseline_timing is None
            else (
                baseline_timing["warm_median_seconds"]
                / experimental_timing["warm_median_seconds"]
            ),
            "experimental_profile_trace": exp_diag.get(
                "experimental_profile_fixed_support_batch"
            ),
            "delta_experimental_minus_baseline": delta,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "family": family,
                    "layers": len(inputs),
                    "experimental_converged_count": converged_count,
                    "experimental_fallback_required_count": fallback_required_count,
                    "experimental_fast_path_layer_count": fast_path_layer_count,
                    "support_candidate_expanded_layer_count": int(
                        candidate_metadata["expanded_layer_count"]
                    ),
                    "fallback_rescue_layer_indices": (
                        fallback_rescue_layer_indices_for_row
                    ),
                    "fallback_rescue_candidate_expanded_layer_count": (
                        rescue_candidate_expanded_layer_count
                    ),
                    "fallback_rescue_cache": fallback_rescue_cache,
                    "total_executed_candidate_layer_count": (
                        total_executed_candidate_layer_count
                    ),
                    "experimental_residual_summary": residual_summary,
                    "experimental_nonconverged_layers": None
                    if per_layer_summary is None
                    else [
                        item
                        for item in per_layer_summary
                        if not bool(item["converged"])
                    ],
                    "experimental_warm_median_seconds": experimental_timing[
                        "warm_median_seconds"
                    ],
                    "prepared_bucket_core_warm_median_seconds": None
                    if prepared_bucket_core_timing is None
                    else prepared_bucket_core_timing["warm_median_seconds"],
                    "baseline_warm_median_seconds": None
                    if baseline_timing is None
                    else baseline_timing["warm_median_seconds"],
                    "speedup_vs_baseline": row["speedup_vs_baseline"],
                    **({} if delta is None else delta),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_experimental = sum(row["experimental"]["warm_median_seconds"] for row in rows)
    total_layers = sum(row["layer_count"] for row in rows)
    total_converged = sum(row["experimental_converged_count"] for row in rows)
    total_fallback_required = sum(
        row["experimental_fallback_required_count"] for row in rows
    )
    total_fast_path_layers = sum(
        row["experimental_fast_path_layer_count"] for row in rows
    )
    total_candidate_expanded_layers = sum(
        row["support_candidate_expanded_layer_count"] for row in rows
    )
    total_rescue_expanded_layers = sum(
        row["fallback_rescue_candidate_expanded_layer_count"] for row in rows
    )
    total_executed_candidate_layers = sum(
        row["total_executed_candidate_layer_count"] for row in rows
    )
    total_rescue_plan_prepare = sum(
        0.0
        if row["fallback_rescue_plan_prepare_seconds"] is None
        else float(row["fallback_rescue_plan_prepare_seconds"])
        for row in rows
    )
    total_rescue_cache_prepare_count = sum(
        0
        if row["fallback_rescue_cache"] is None
        else int(row["fallback_rescue_cache"]["prepare_count"])
        for row in rows
    )
    total_rescue_cache_hit_count = sum(
        0
        if row["fallback_rescue_cache"] is None
        else int(row["fallback_rescue_cache"]["hit_count"])
        for row in rows
    )
    total_plan_prepare = (
        None
        if not args.prepared_plan
        else sum(float(row["plan_prepare_seconds"]) for row in rows)
    )
    total_prepared_bucket_core = (
        None
        if not args.prepared_plan
        else sum(
            float(row["prepared_bucket_core"]["warm_median_seconds"]) for row in rows
        )
    )
    baseline_rows = [row for row in rows if row["baseline"] is not None]
    total_baseline = (
        None
        if len(baseline_rows) != len(rows)
        else sum(row["baseline"]["warm_median_seconds"] for row in baseline_rows)
    )
    delta_rows = [
        row for row in rows if row["delta_experimental_minus_baseline"] is not None
    ]
    payload = {
        "schema": "exogibbs_pdipm_api_profile_fixed_support_batch_benchmark_v1",
        "jax_runtime": runtime,
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "skip_baseline": bool(args.skip_baseline),
        "return_diagnostics": return_diagnostics,
        "budget_gate": budget_gate,
        "block_output": str(args.block_output),
        "prepared_plan": bool(args.prepared_plan),
        "element_inventory_scale": args.element_inventory_scale,
        "element_inventory_batch_size": int(args.element_inventory_batch_size),
        "element_inventory_batch_mode": str(args.element_inventory_batch_mode),
        "support_candidate_mode": str(args.support_candidate_mode),
        "support_candidate_prune_floors": list(support_candidate_prune_floors),
        "support_candidate_neighbor_union": bool(
            include_support_candidate_neighbor_union
        ),
        "residual_tolerance_multiplier": float(args.residual_tolerance_multiplier),
        "skipped_layers": skipped,
        "summary": {
            "family_count": len(rows),
            "layer_count": total_layers,
            "evaluation_count": total_layers * int(args.element_inventory_batch_size),
            "experimental_converged_count": total_converged,
            "experimental_fallback_required_count": total_fallback_required,
            "experimental_converged_fraction": None
            if total_layers == 0
            else total_converged
            / (total_layers * int(args.element_inventory_batch_size)),
            "experimental_fallback_free_fraction": None
            if total_layers == 0
            else 1.0
            - total_fallback_required
            / (total_layers * int(args.element_inventory_batch_size)),
            "experimental_fast_path_layer_count": total_fast_path_layers,
            "support_candidate_expanded_layer_count": total_candidate_expanded_layers,
            "support_candidate_evaluation_count": (
                total_candidate_expanded_layers
                * int(args.element_inventory_batch_size)
            ),
            "fallback_rescue_candidate_expanded_layer_count": (
                total_rescue_expanded_layers
            ),
            "fallback_rescue_candidate_evaluation_count": (
                total_rescue_expanded_layers
                * int(args.element_inventory_batch_size)
            ),
            "total_executed_candidate_layer_count": total_executed_candidate_layers,
            "total_executed_candidate_evaluation_count": (
                total_executed_candidate_layers
                * int(args.element_inventory_batch_size)
            ),
            "total_experimental_warm_median_seconds": total_experimental,
            "experimental_warm_median_seconds_per_layer": None
            if total_layers == 0
            else total_experimental / total_layers,
            "experimental_warm_median_seconds_per_evaluation": None
            if total_layers == 0
            else total_experimental
            / (total_layers * int(args.element_inventory_batch_size)),
            "experimental_layers_per_second": None
            if total_experimental == 0.0
            else total_layers / total_experimental,
            "experimental_evaluations_per_second": None
            if total_experimental == 0.0
            else (total_layers * int(args.element_inventory_batch_size))
            / total_experimental,
            "total_plan_prepare_seconds": total_plan_prepare,
            "total_fallback_rescue_plan_prepare_seconds": (
                total_rescue_plan_prepare
            ),
            "fallback_rescue_cache_prepare_count": (
                total_rescue_cache_prepare_count
            ),
            "fallback_rescue_cache_hit_count": total_rescue_cache_hit_count,
            "plan_prepare_seconds_per_layer": None
            if total_layers == 0 or total_plan_prepare is None
            else total_plan_prepare / total_layers,
            "total_prepared_bucket_core_warm_median_seconds": total_prepared_bucket_core,
            "prepared_bucket_core_warm_median_seconds_per_layer": None
            if total_layers == 0 or total_prepared_bucket_core is None
            else total_prepared_bucket_core / total_layers,
            "prepared_bucket_core_layers_per_second": None
            if total_prepared_bucket_core in (None, 0.0)
            else total_layers / total_prepared_bucket_core,
            "total_baseline_warm_median_seconds": total_baseline,
            "speedup_vs_baseline": None
            if total_baseline is None
            else total_baseline / total_experimental,
            "max_gas_ln_n_abs_delta": None
            if not delta_rows
            else max(
                row["delta_experimental_minus_baseline"]["max_gas_ln_n_abs_delta"]
                for row in delta_rows
            ),
            "max_gas_n_abs_delta": None
            if not delta_rows
            else max(
                row["delta_experimental_minus_baseline"]["max_gas_n_abs_delta"]
                for row in delta_rows
            ),
            "max_condensate_amount_abs_delta": None
            if not delta_rows
            else max(
                row["delta_experimental_minus_baseline"][
                    "max_condensate_amount_abs_delta"
                ]
                for row in delta_rows
            ),
        },
        "rows": rows,
    }
    output = ROOT / str(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
