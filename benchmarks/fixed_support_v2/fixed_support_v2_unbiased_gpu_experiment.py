#!/usr/bin/env python
"""Run an unbiased fixed-support v1/v2 GPU verification.

The artifact keeps three questions separate:

1. exact-state v1/v2 comparisons on small controls;
2. exact-state v1/v2 comparisons on historically hard, large supports;
3. v2 support lifecycle runs (solve, inactive check, expand, re-solve).

The first two lanes use the same prepared plan and explicitly preserve
``q, r, lambda, rho, qtot, epsilon`` in both solvers.  The lifecycle lane is
not folded into the solver-only score.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from jax import config as jax_config
import jax.numpy as jnp

import exogibbs
EXPECTED_EXOGIBBS_ROOT = (SOURCE_ROOT / "exogibbs").resolve()
IMPORTED_EXOGIBBS_ROOT = Path(exogibbs.__file__).resolve().parent
if IMPORTED_EXOGIBBS_ROOT != EXPECTED_EXOGIBBS_ROOT:
    raise RuntimeError(
        "Imported exogibbs from outside this repository: "
        f"{IMPORTED_EXOGIBBS_ROOT} != {EXPECTED_EXOGIBBS_ROOT}"
    )

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    _least_squares_element_potential,
    prepare_experimental_profile_fixed_support_batch_plan,
    run_experimental_profile_fixed_support_batch_plan,
    run_experimental_profile_fixed_support_v2_batch_plan,
)
from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    element_budget_for_profile,
    support_payload_for_profile,
)
from exogibbs.condensates.fixed_support_payload import seed_fixed_support_payload
from exogibbs.condensates.support_selection_policy import (
    select_activity_driven_support_candidates,
)
from exogibbs.optimize.fixed_support_v2.types import (
    ContinuationConfig,
    FixedSupportV2Config,
    SolverLimitConfig,
    TerminalStatus,
)
from exogibbs.optimize.fixed_support_v2_profile import (
    _prepared_original_state_batch,
    _prepared_problem_batch,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

from support_atlas_sweep import (
    _dedup_cap,
    _gas_only_activity_fields,
    _ordered_activity_capacity,
    _ordered_activity_driving,
    _ordered_capacity,
)


MATRIX_PATH = Path(__file__).with_name("fixed_support_v2_gpu_matrix.json")
DEFAULT_OUTPUT_DIR = ROOT / "results" / "fixed_support_v2_unbiased_gpu"
DEFAULT_SCHEDULE = (-11.0, -13.0, -15.0, -17.0)


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _host(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _host(item) for key, item in value._asdict().items()}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    array = np.asarray(jax.device_get(value))
    if array.shape == ():
        item = array.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if np.issubdtype(array.dtype, np.floating):
        clean = array.astype(object)
        clean[~np.isfinite(array)] = None
        return clean.tolist()
    return array.tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_host(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD") or None,
        "worktree_dirty": bool(run("git", "status", "--porcelain")),
    }


def _baseline_integrity(matrix: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for declared in matrix["frozen_v1_baseline"]["artifacts"]:
        path = ROOT / declared["path"]
        actual = _sha256(path) if path.exists() else None
        rows.append(
            {
                **declared,
                "exists": path.exists(),
                "actual_sha256": actual,
                "matches": actual == declared["sha256"],
            }
        )
    return {
        "matrix_path": str(MATRIX_PATH.relative_to(ROOT)),
        "matrix_sha256": _sha256(MATRIX_PATH),
        "artifacts": rows,
        "passed": all(row["matches"] for row in rows),
    }


def _source_integrity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        MATRIX_PATH,
        Path(__file__).with_name("run_fixed_support_v2_unbiased_gpu.csh"),
        Path(__file__).with_name("run_fixed_support_v2_reaudit_gpu.csh"),
        Path(__file__).with_name("run_fixed_support_v2_corrected_gpu.csh"),
        Path(__file__).with_name(
            "run_fixed_support_v2_water128_gpu100.csh"
        ),
        Path(__file__).with_name(
            "run_fixed_support_v2_final_solver_matrix_gpu.csh"
        ),
        ROOT / "benchmarks/fixed_support_v2/support_atlas_sweep.py",
        ROOT / "src/exogibbs/api/condensate_equilibrium.py",
        ROOT / "src/exogibbs/optimize/fixed_support_batch.py",
        ROOT / "src/exogibbs/optimize/minimize_cond.py",
        ROOT / "src/exogibbs/optimize/fixed_support_v2_profile.py",
        *sorted((ROOT / "src/exogibbs/optimize/fixed_support_v2").glob("*.py")),
    )
    return {
        "algorithm": "sha256",
        "files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in paths
            if path.exists()
        },
        **_git_metadata(),
    }


def _prior_validation_integrity() -> dict[str, Any]:
    declared = (
        (
            "corrected_support_lifecycle",
            ROOT / "results/fixed_support_v2_corrected_gpu/summary.json",
        ),
        (
            "water128_restoration_limit",
            ROOT / "results/fixed_support_v2_water128_gpu100/summary.json",
        ),
    )
    artifacts = [
        {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else None,
        }
        for label, path in declared
    ]
    return {
        "artifacts": artifacts,
        "passed": all(row["exists"] for row in artifacts),
    }


def _gas_initial_state(
    *,
    setup: Any,
    definition: Any,
    source_index: int,
    budget: jax.Array,
    support: Sequence[int],
    first_epsilon: float,
    seed_fraction: float,
    max_seed_amount: float,
    center_bound_multipliers: bool = False,
) -> CondensateEquilibriumInit:
    temperature = float(definition.temperatures[source_index])
    pressure = float(definition.pressures[source_index])
    fields = _gas_only_activity_fields(
        setup=setup,
        temperature=temperature,
        pressure=pressure,
        budget=budget,
    )
    support_indices, support_amounts = seed_fixed_support_payload(
        setup=setup,
        element_inventory_target=budget,
        support_indices=support,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
        min_seed_amount=1.0e-300,
    )
    gas_ln_n = jnp.asarray(fields["gas_ln_n"], dtype=jnp.float64)
    gas_ntot = jnp.asarray(fields["gas_ntot"], dtype=jnp.float64)
    stationarity_source = jnp.asarray(
        fields["stationarity_source"], dtype=jnp.float64
    )
    element_potential = _historical_best_residual_lambda(
        setup=setup,
        budget=budget,
        support=support_indices,
        support_amounts=support_amounts,
        temperature=temperature,
        gas_ln_n=gas_ln_n,
        gas_ntot=gas_ntot,
        stationarity_source=stationarity_source,
        epsilon=float(first_epsilon),
    )
    support_log_amounts = jnp.log(
        jnp.asarray(support_amounts, dtype=jnp.float64)
    )
    rho = (
        jnp.asarray(first_epsilon, dtype=jnp.float64) - support_log_amounts
        if center_bound_multipliers
        else jnp.zeros((len(support_indices),), dtype=jnp.float64)
    )
    return CondensateEquilibriumInit(
        gas_ln_n=gas_ln_n,
        gas_ntot=gas_ntot,
        support_indices=support_indices,
        support_amounts=support_amounts,
        element_potential=element_potential,
        rho=rho,
        barrier_epsilon=jnp.asarray(first_epsilon, dtype=jnp.float64),
    )


def _historical_best_residual_lambda(
    *,
    setup: Any,
    budget: jax.Array,
    support: Sequence[int],
    support_amounts: Sequence[float],
    temperature: float,
    gas_ln_n: jax.Array,
    gas_ntot: jax.Array,
    stationarity_source: jax.Array,
    epsilon: float,
) -> jax.Array:
    """Reproduce v1 ``best_residual`` once, then freeze the chosen lambda."""

    ag = jnp.asarray(setup.formula_matrix, dtype=jnp.float64)
    support_array = jnp.asarray(support, dtype=jnp.int32)
    ac = jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64)[
        :, support_array
    ]
    q = jnp.asarray(gas_ln_n, dtype=jnp.float64)
    r = jnp.log(jnp.asarray(support_amounts, dtype=jnp.float64))
    qtot = jnp.log(jnp.asarray(gas_ntot, dtype=jnp.float64))
    source = jnp.asarray(stationarity_source, dtype=jnp.float64)
    hcond = jnp.asarray(
        setup.condensate_setup.hvector_func(float(temperature)),
        dtype=jnp.float64,
    )[support_array]
    eta = jnp.ones_like(r)
    gas_matrix = ag.T
    gas_rhs = q + source
    joint_matrix = jnp.concatenate([ag.T, ac.T], axis=0)
    joint_rhs = jnp.concatenate([gas_rhs, hcond - eta])

    def damped_lstsq(matrix, rhs):
        column_scale = jnp.maximum(
            jnp.linalg.norm(matrix, axis=0),
            jnp.asarray(1.0e-300, dtype=matrix.dtype),
        )
        scaled = matrix / column_scale[None, :]
        normal = scaled.T @ scaled
        normal_rhs = scaled.T @ rhs
        damping = jnp.maximum(
            jnp.asarray(1.0e-12, dtype=matrix.dtype)
            * jnp.mean(jnp.diag(normal)),
            jnp.asarray(1.0e-30, dtype=matrix.dtype),
        )
        solution = jnp.linalg.solve(
            normal + damping * jnp.eye(normal.shape[0], dtype=matrix.dtype),
            normal_rhs,
        )
        return jnp.nan_to_num(
            solution / column_scale, nan=0.0, posinf=0.0, neginf=0.0
        )

    candidates = jnp.stack(
        [
            jnp.zeros((ag.shape[0],), dtype=jnp.float64),
            jnp.linalg.lstsq(gas_matrix, gas_rhs, rcond=None)[0],
            jnp.linalg.lstsq(joint_matrix, joint_rhs, rcond=None)[0],
            damped_lstsq(gas_matrix, gas_rhs),
            damped_lstsq(joint_matrix, joint_rhs),
        ],
        axis=0,
    )

    def residual_norm(lambda_):
        blocks = (
            q + source - ag.T @ lambda_,
            hcond - ac.T @ lambda_ - eta,
            ag @ jnp.exp(q)
            + ac @ jnp.exp(r)
            - jnp.asarray(budget, dtype=jnp.float64),
            r - jnp.asarray(epsilon, dtype=jnp.float64),
            jnp.asarray([jnp.sum(jnp.exp(q)) - jnp.exp(qtot)]),
        )
        values = jnp.concatenate(blocks)
        scale = jnp.max(
            jnp.abs(values), initial=jnp.asarray(0.0, dtype=values.dtype)
        )
        return jnp.where(
            scale == 0.0, 0.0, scale * jnp.linalg.norm(values / scale)
        )

    residuals = jax.vmap(residual_norm)(candidates)
    return candidates[jnp.argmin(residuals)]


def _warm_lifecycle_state(
    *,
    setup: Any,
    definition: Any,
    source_index: int,
    budget: jax.Array,
    support: Sequence[int],
    result: Mapping[str, Any],
    local_index: int,
    first_epsilon: float,
    seed_fraction: float,
    max_seed_amount: float,
) -> CondensateEquilibriumInit:
    support_indices, seeded_amounts = seed_fixed_support_payload(
        setup=setup,
        element_inventory_target=budget,
        support_indices=support,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
        min_seed_amount=1.0e-300,
    )
    full_amounts = np.asarray(
        jax.device_get(result["condensate_amounts"][local_index]),
        dtype=np.float64,
    )
    support_amounts = tuple(
        float(full_amounts[index])
        if math.isfinite(float(full_amounts[index]))
        and float(full_amounts[index]) > 0.0
        else float(seed)
        for index, seed in zip(support_indices, seeded_amounts)
    )
    q = jnp.asarray(result["gas_log_amounts"][local_index], dtype=jnp.float64)
    qtot = jnp.asarray(
        result["total_gas_log_amount"][local_index], dtype=jnp.float64
    )
    return CondensateEquilibriumInit(
        gas_ln_n=q,
        gas_ntot=jnp.exp(qtot),
        support_indices=support_indices,
        support_amounts=support_amounts,
        element_potential=jnp.asarray(
            result["element_potential"][local_index], dtype=jnp.float64
        ),
        rho=(
            jnp.asarray(first_epsilon, dtype=jnp.float64)
            - jnp.log(jnp.asarray(support_amounts, dtype=jnp.float64))
        ),
        barrier_epsilon=jnp.asarray(first_epsilon, dtype=jnp.float64),
    )


def _prepare_plan(
    *,
    setup: Any,
    definition: Any,
    source_indices: Sequence[int],
    inits: Sequence[CondensateEquilibriumInit],
    budget: jax.Array,
    max_normal_iterations: int,
    seed_fraction: float,
    max_seed_amount: float,
) -> Any:
    options = CondensateEquilibriumOptions(
        profile_method="vmap_cold",
        profile_warm_start_support_policy="explicit_payload",
        return_diagnostics=True,
        allow_caveat_tiers=True,
        max_inner_iterations=int(max_normal_iterations),
        enable_experimental_profile_fixed_support_batch=True,
        enable_experimental_profile_fixed_support_fallback_rescue=False,
        enable_profile_native_activity_support_expansion=False,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
    )
    return prepare_experimental_profile_fixed_support_batch_plan(
        setup,
        jnp.asarray(
            [definition.temperatures[index] for index in source_indices],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [definition.pressures[index] for index in source_indices],
            dtype=jnp.float64,
        ),
        budget,
        init=tuple(inits),
        options=options,
    )


def _support_for_case(
    *,
    setup: Any,
    definition: Any,
    source_index: int,
    budget: jax.Array,
    case: Mapping[str, Any],
) -> tuple[int, ...]:
    policy = str(case["policy"])
    if policy == "curated":
        support, _ = support_payload_for_profile(setup, definition, budget)
        return tuple(int(index) for index in support)
    fields = _gas_only_activity_fields(
        setup=setup,
        temperature=float(definition.temperatures[source_index]),
        pressure=float(definition.pressures[source_index]),
        budget=budget,
    )
    if policy == "activity_capacity":
        ordered = _ordered_activity_capacity(fields, float(case["threshold"]))
    elif policy == "activity_driving":
        ordered = _ordered_activity_driving(fields, threshold=None)
    elif policy == "capacity":
        ordered = _ordered_capacity(fields)
    else:
        raise ValueError(f"Unknown support policy: {policy}")
    return _dedup_cap(ordered, int(case["cap"]))


def _array_digest_update(digest: Any, label: str, value: Any) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())


def _audit_prepared_initial_state(plan: Any, config: FixedSupportV2Config) -> dict[str, Any]:
    maxima = {
        "q": 0.0,
        "r": 0.0,
        "lambda": 0.0,
        "rho": 0.0,
        "qtot": 0.0,
        "epsilon": 0.0,
        "gamma": 0.0,
    }
    digest = hashlib.sha256()
    for bucket_index, bucket in enumerate(plan.buckets):
        problems = _prepared_problem_batch(
            bucket,
            plan.formula_matrix,
            budget_relative_floor=1.0e-6,
        )
        state = _prepared_original_state_batch(bucket, problems, config)
        first_epsilon = jnp.asarray(
            config.continuation.epsilon_schedule[0],
            dtype=jnp.asarray(bucket.ln_mk_init).dtype,
        )
        if config.continuation.initial_state_policy == "center":
            expected_rho = first_epsilon - jnp.asarray(bucket.ln_mk_init)
            expected_epsilon = jnp.full(
                (jnp.asarray(bucket.ln_mk_init).shape[0],),
                first_epsilon,
            )
        else:
            expected_rho = bucket.rho_init
            expected_epsilon = bucket.barrier_epsilon_init
        expected = {
            "q": bucket.ln_nk_init,
            "r": bucket.ln_mk_init,
            "lambda": bucket.element_potential_init,
            "rho": expected_rho,
            "qtot": bucket.ln_ntot_init,
            "epsilon": expected_epsilon,
            "gamma": (
                jnp.asarray(bucket.hvector)
                + jnp.asarray(bucket.ln_normalized_pressure)[:, None]
            ),
        }
        actual = {
            "q": state.q,
            "r": state.r,
            "lambda": state.lambda_,
            "rho": state.rho,
            "qtot": state.qtot,
            "epsilon": state.epsilon,
            "gamma": problems.gamma,
        }
        for name in maxima:
            if expected[name] is None:
                raise ValueError(f"Prepared exact-state field is absent: {name}")
            difference = np.asarray(
                jax.device_get(actual[name] - expected[name]), dtype=np.float64
            )
            maxima[name] = max(
                maxima[name], float(np.max(np.abs(difference), initial=0.0))
            )
            _array_digest_update(
                digest, f"bucket{bucket_index}:{name}", expected[name]
            )
        _array_digest_update(
            digest, f"bucket{bucket_index}:support", bucket.support_indices
        )
    return {
        "initialization_contract": (
            "exact provided state"
            if config.continuation.initial_state_policy == "provided"
            else "barrier-centered bound multipliers"
        ),
        "v2_initial_state_policy": config.continuation.initial_state_policy,
        "gas_source_policy": (
            "canonical thermochemical gamma; no legacy external source override"
        ),
        "max_abs_v2_adapter_difference_from_plan": maxima,
        "prepared_state_sha256": digest.hexdigest(),
        "passed": all(value == 0.0 for value in maxima.values()),
    }


def _time_v1(plan: Any, residual_tolerance_multiplier: float) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    first = run_experimental_profile_fixed_support_batch_plan(
        plan,
        rho_initialization="provided",
        lambda_initialization="provided",
        residual_tolerance_multiplier=float(residual_tolerance_multiplier),
    )
    _block_until_ready(first)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    warm = run_experimental_profile_fixed_support_batch_plan(
        plan,
        rho_initialization="provided",
        lambda_initialization="provided",
        residual_tolerance_multiplier=float(residual_tolerance_multiplier),
    )
    _block_until_ready(warm)
    warm_seconds = time.perf_counter() - start
    return warm, {
        "first_call_seconds": first_seconds,
        "warm_execution_seconds": warm_seconds,
        "estimated_compilation_seconds": max(first_seconds - warm_seconds, 0.0),
        "note": "v1 compile time is estimated as first minus warm execution.",
    }


def _time_v2(
    plan: Any,
    config: FixedSupportV2Config,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    first = run_experimental_profile_fixed_support_v2_batch_plan(
        plan,
        config=config,
        budget_relative_floor=float(args.budget_relative_floor),
        support_closure_tolerance=float(args.support_closure_tolerance),
    )
    _block_until_ready(first)
    warm = run_experimental_profile_fixed_support_v2_batch_plan(
        plan,
        config=config,
        budget_relative_floor=float(args.budget_relative_floor),
        support_closure_tolerance=float(args.support_closure_tolerance),
    )
    _block_until_ready(warm)
    return warm, {
        "compilation_seconds": float(first["compilation_seconds"]),
        "first_execution_seconds": float(first["execution_seconds"]),
        "warm_compile_lookup_seconds": float(warm["compilation_seconds"]),
        "warm_execution_seconds": float(warm["execution_seconds"]),
        "first_diagnostic_seconds": float(first["diagnostic_seconds"]),
        "warm_diagnostic_seconds": float(warm["diagnostic_seconds"]),
    }


def _kkt_row(result: Mapping[str, Any], index: int) -> dict[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value))[index])
        for name, value in result["final_kkt_norms"]._asdict().items()
    }


def _batched_row(value: Any, index: int) -> Any:
    """Select one leading batch row while preserving diagnostic structure."""

    if isinstance(value, Mapping):
        return {
            str(key): _batched_row(item, index) for key, item in value.items()
        }
    if hasattr(value, "_asdict"):
        return {
            str(key): _batched_row(item, index)
            for key, item in value._asdict().items()
        }
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return _host(array)
    return _host(array[index])


def _bucket_diagnostics_for_layer(
    result: Mapping[str, Any], layer_index: int
) -> dict[str, Any]:
    """Extract one profile-local layer from its common-support bucket report."""

    for report in result["bucket_reports"]:
        layers = tuple(int(value) for value in report["layer_indices"])
        if layer_index not in layers:
            continue
        row = layers.index(layer_index)
        names = (
            "terminal_status",
            "completed_stage_count",
            "stage_statuses",
            "stage_normal_iteration_counts",
            "stage_restoration_call_counts",
            "stage_restoration_accepted_iteration_counts",
            "stage_last_return_diagnostics",
            "stage_soc_attempt_counts",
            "stage_soc_accepted_counts",
            "final_kkt_norms",
            "terminal_restoration_diagnostics",
            "terminal_normal_diagnostics",
            "last_return_diagnostics",
        )
        return {
            "support_indices": list(report["support_indices"]),
            "bucket_layer_indices": list(layers),
            **{
                name: _batched_row(report[name], row)
                for name in names
            },
        }
    raise RuntimeError(
        f"No v2 bucket report contains local layer {layer_index}."
    )


def _run_solver_case(
    *,
    setup: Any,
    matrix_case: Mapping[str, Any],
    config: FixedSupportV2Config,
    args: argparse.Namespace,
) -> dict[str, Any]:
    definition = FRESH_CURATED_PROFILES[str(matrix_case["family"])]
    source_index = int(matrix_case["layer"])
    budget = jnp.asarray(
        element_budget_for_profile(setup, definition), dtype=jnp.float64
    )
    support = _support_for_case(
        setup=setup,
        definition=definition,
        source_index=source_index,
        budget=budget,
        case=matrix_case,
    )
    expected_count = matrix_case.get("expected_support_count")
    if expected_count is not None and len(support) != int(expected_count):
        raise RuntimeError(
            f"{matrix_case['label']}: support drifted from frozen count "
            f"{expected_count} to {len(support)}."
        )
    if not support:
        raise RuntimeError(f"{matrix_case['label']}: fixed support is empty.")
    init = _gas_initial_state(
        setup=setup,
        definition=definition,
        source_index=source_index,
        budget=budget,
        support=support,
        first_epsilon=float(args.epsilon_schedule[0]),
        seed_fraction=float(args.seed_fraction),
        max_seed_amount=float(args.max_seed_amount),
    )
    plan = _prepare_plan(
        setup=setup,
        definition=definition,
        source_indices=(source_index,),
        inits=(init,),
        budget=budget,
        max_normal_iterations=int(args.max_normal_iterations),
        seed_fraction=float(args.seed_fraction),
        max_seed_amount=float(args.max_seed_amount),
    )
    audit = _audit_prepared_initial_state(plan, config)
    if not audit["passed"]:
        raise RuntimeError(f"{matrix_case['label']}: exact-state audit failed.")
    v1, v1_timing = _time_v1(plan, args.v1_residual_tolerance_multiplier)
    v2, v2_timing = _time_v2(plan, config, args)
    v1_converged = bool(np.asarray(jax.device_get(v1["converged"]))[0])
    v2_converged = bool(
        np.asarray(jax.device_get(v2["fixed_support_converged"]))[0]
    )
    terminal = int(np.asarray(jax.device_get(v2["terminal_status"]))[0])
    kkt = _kkt_row(v2, 0)
    tolerances = config.normal
    independent_kkt = bool(
        kkt["gas_stationarity"] <= tolerances.stationarity_tolerance
        and kkt["condensate_stationarity"]
        <= tolerances.stationarity_tolerance
        and kkt["budget_scaled"] <= tolerances.budget_tolerance
        and kkt["complementarity"] <= tolerances.complementarity_tolerance
        and kkt["total_density_scaled"]
        <= tolerances.total_density_tolerance
    )
    return {
        **dict(matrix_case),
        "lane": "small_control" if len(support) <= 8 else "large_support_stress",
        "temperature": float(definition.temperatures[source_index]),
        "pressure": float(definition.pressures[source_index]),
        "support_count": len(support),
        "support_indices": list(support),
        "support_species": [str(setup.condensate_species[index]) for index in support],
        "initial_state_contract": audit,
        "v1": {
            "converged": v1_converged,
            "final_residual": float(
                np.asarray(jax.device_get(v1["final_residual"]))[0]
            ),
            "timing": v1_timing,
        },
        "v2": {
            "converged": v2_converged,
            "terminal_status": terminal,
            "terminal_status_name": TerminalStatus(terminal).name,
            "completed_stage_count": int(
                np.asarray(jax.device_get(v2["completed_stage_count"]))[0]
            ),
            "support_closed": bool(
                np.asarray(jax.device_get(v2["support_closed"]))[0]
            ),
            "independent_kkt": kkt,
            "independent_kkt_passed": independent_kkt,
            "final_state_values_finite": bool(
                np.asarray(jax.device_get(v2["final_state_values_finite"]))[0]
            ),
            "timing": v2_timing,
            "solver_diagnostics": _bucket_diagnostics_for_layer(v2, 0),
        },
        "historical_v1_status_reproduced": (
            None
            if "historical_v1_converged" not in matrix_case
            else v1_converged == bool(matrix_case["historical_v1_converged"])
        ),
    }


def _native_initial_support(
    *,
    setup: Any,
    definition: Any,
    source_index: int,
    budget: jax.Array,
    args: argparse.Namespace,
) -> tuple[int, ...]:
    base, _ = support_payload_for_profile(setup, definition, budget)
    temperature = float(definition.temperatures[source_index])
    pressure = float(definition.pressures[source_index])
    fields = _gas_only_activity_fields(
        setup=setup,
        temperature=temperature,
        pressure=pressure,
        budget=budget,
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=jnp.asarray(fields["gas_ln_n"], dtype=jnp.float64),
        gas_stationarity_source=jnp.asarray(
            fields["stationarity_source"], dtype=jnp.float64
        ),
    )
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=budget,
        condensate_species_order=setup.condensate_species,
        hvector_cond=setup.condensate_setup.hvector_func(temperature),
        element_potential=element_potential,
        max_positive_support_count=int(args.lifecycle_initial_topk),
        activity_threshold=float(args.lifecycle_activity_threshold),
        existing_support_indices=base,
        temperature=temperature,
        condensate_temperature_validity_upper=(
            setup.condensate_setup.metadata.get("temperature_validity_upper")
        ),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_profile_budget",
            "hvector_cond": "exogibbs_condensate_thermochemistry",
            "element_potential": "exogibbs_native_gas_only_equilibrium",
            "condensate_temperature_validity_upper": (
                "exogibbs_condensate_temperature_validity_metadata"
            ),
        },
    )
    base_set = set(int(index) for index in base)
    additions = tuple(
        int(index)
        for index in report.positive_support_indices
        if int(index) not in base_set
    )
    support_limit = max(int(args.lifecycle_initial_max_support), len(base))
    return tuple(dict.fromkeys((*base, *additions)))[:support_limit]


def _run_lifecycle_family(
    *,
    setup: Any,
    family: str,
    config: FixedSupportV2Config,
    args: argparse.Namespace,
) -> dict[str, Any]:
    definition = FRESH_CURATED_PROFILES[family]
    budget = jnp.asarray(
        element_budget_for_profile(setup, definition), dtype=jnp.float64
    )
    layer_records = {
        index: {
            "source_layer_index": index,
            "temperature": float(definition.temperatures[index]),
            "pressure": float(definition.pressures[index]),
            "rounds": [],
        }
        for index in range(len(definition.temperatures))
    }
    pending: dict[int, tuple[tuple[int, ...], CondensateEquilibriumInit]] = {}
    for source_index in layer_records:
        support = _native_initial_support(
            setup=setup,
            definition=definition,
            source_index=source_index,
            budget=budget,
            args=args,
        )
        layer_records[source_index]["initial_support_count"] = len(support)
        layer_records[source_index]["initial_support_indices"] = list(support)
        if not support:
            layer_records[source_index]["outcome"] = "gas_only_no_candidate"
            continue
        pending[source_index] = (
            support,
            _gas_initial_state(
                setup=setup,
                definition=definition,
                source_index=source_index,
                budget=budget,
                support=support,
                first_epsilon=float(args.epsilon_schedule[0]),
                seed_fraction=float(args.seed_fraction),
                max_seed_amount=float(args.max_seed_amount),
                center_bound_multipliers=True,
            ),
        )

    total_compilation = 0.0
    total_execution = 0.0
    total_diagnostics = 0.0
    for round_index in range(int(args.lifecycle_max_rounds)):
        if not pending:
            break
        source_indices = tuple(sorted(pending))
        plan = _prepare_plan(
            setup=setup,
            definition=definition,
            source_indices=source_indices,
            inits=tuple(pending[index][1] for index in source_indices),
            budget=budget,
            max_normal_iterations=int(args.max_normal_iterations),
            seed_fraction=float(args.seed_fraction),
            max_seed_amount=float(args.max_seed_amount),
        )
        audit = _audit_prepared_initial_state(plan, config)
        if not audit["passed"]:
            raise RuntimeError(f"{family}: lifecycle exact-state audit failed.")
        result = run_experimental_profile_fixed_support_v2_batch_plan(
            plan,
            config=config,
            budget_relative_floor=float(args.budget_relative_floor),
            support_closure_tolerance=float(args.support_closure_tolerance),
        )
        _block_until_ready(result)
        total_compilation += float(result["compilation_seconds"])
        total_execution += float(result["execution_seconds"])
        total_diagnostics += float(result["diagnostic_seconds"])
        converged = np.asarray(
            jax.device_get(result["fixed_support_converged"]), dtype=bool
        )
        closed = np.asarray(jax.device_get(result["support_closed"]), dtype=bool)
        terminal = np.asarray(
            jax.device_get(result["terminal_status"]), dtype=np.int64
        )
        driving = np.asarray(
            jax.device_get(result["inactive_condensate_driving"]),
            dtype=np.float64,
        )
        expansion = np.asarray(
            jax.device_get(result["support_expansion_mask"]), dtype=bool
        )
        next_pending = {}
        for local_index, source_index in enumerate(source_indices):
            support = pending[source_index][0]
            candidates = np.flatnonzero(expansion[local_index])
            ordered = tuple(
                int(index)
                for index in candidates[
                    np.argsort(driving[local_index, candidates], kind="stable")
                ]
            )
            additions = ordered[: int(args.lifecycle_add_per_round)]
            max_new = max(
                0, int(args.lifecycle_max_support) - len(support)
            )
            additions = additions[:max_new]
            expanded = tuple(dict.fromkeys((*support, *additions)))
            kkt = _kkt_row(result, local_index)
            layer_records[source_index]["rounds"].append(
                {
                    "round_index": round_index,
                    "support_count": len(support),
                    "support_indices": list(support),
                    "fixed_support_converged": bool(converged[local_index]),
                    "support_closed": bool(closed[local_index]),
                    "terminal_status_name": TerminalStatus(
                        int(terminal[local_index])
                    ).name,
                    "positive_inactive_count": int(candidates.size),
                    "max_positive_inactive_driving": float(
                        np.max(-driving[local_index, candidates], initial=0.0)
                    ),
                    "added_support_indices": list(additions),
                    "independent_kkt": kkt,
                    "prepared_state_sha256": audit["prepared_state_sha256"],
                    "solver_diagnostics": _bucket_diagnostics_for_layer(
                        result, local_index
                    ),
                }
            )
            if not converged[local_index]:
                layer_records[source_index]["outcome"] = "fixed_support_failed"
            elif closed[local_index]:
                layer_records[source_index]["outcome"] = "closed"
            elif not additions:
                layer_records[source_index]["outcome"] = (
                    "open_at_support_or_round_limit"
                )
            else:
                next_pending[source_index] = (
                    expanded,
                    _warm_lifecycle_state(
                        setup=setup,
                        definition=definition,
                        source_index=source_index,
                        budget=budget,
                        support=expanded,
                        result=result,
                        local_index=local_index,
                        first_epsilon=float(args.epsilon_schedule[0]),
                        seed_fraction=float(args.seed_fraction),
                        max_seed_amount=float(args.max_seed_amount),
                    ),
                )
        pending = next_pending

    for source_index in pending:
        layer_records[source_index]["outcome"] = "open_at_round_limit"
    rows = [layer_records[index] for index in sorted(layer_records)]
    outcomes = {}
    for row in rows:
        outcome = row.get("outcome", "open_at_round_limit")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "family": family,
        "layer_count": len(rows),
        "policy": {
            "initial": "public native gas-only activity expansion",
            "initial_topk": int(args.lifecycle_initial_topk),
            "initial_max_support": int(args.lifecycle_initial_max_support),
            "outer_operation": "solve -> temperature-valid inactive KKT check -> expand -> restart schedule",
            "add_per_round": int(args.lifecycle_add_per_round),
            "max_support": int(args.lifecycle_max_support),
            "max_rounds": int(args.lifecycle_max_rounds),
            "support_change_restart": (
                "preserve q and lambda; seed new support; center every bound "
                "multiplier with rho=epsilon-r; restart at first epsilon"
            ),
        },
        "outcome_counts": outcomes,
        "all_non_gas_only_layers_closed": all(
            row.get("outcome") in {"closed", "gas_only_no_candidate"}
            for row in rows
        ),
        "timing": {
            "compilation_seconds": total_compilation,
            "execution_seconds": total_execution,
            "diagnostic_seconds": total_diagnostics,
        },
        "layers": rows,
    }


def _aggregate_solver_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_lane = {}
    for lane in ("small_control", "large_support_stress"):
        rows = [case for case in cases if case["lane"] == lane]
        by_lane[lane] = {
            "case_count": len(rows),
            "v1_converged": sum(bool(row["v1"]["converged"]) for row in rows),
            "v2_converged": sum(bool(row["v2"]["converged"]) for row in rows),
            "v2_independent_kkt_passed": sum(
                bool(row["v2"]["independent_kkt_passed"]) for row in rows
            ),
        }
    historical = [
        row["historical_v1_status_reproduced"]
        for row in cases
        if row["historical_v1_status_reproduced"] is not None
    ]
    return {
        "by_lane": by_lane,
        "all_exact_state_contracts_passed": all(
            row["initial_state_contract"]["passed"] for row in cases
        ),
        "frozen_historical_v1_statuses_reproduced": bool(historical)
        and all(historical),
        "promotion_ready": False,
        "promotion_note": (
            "This is a verification artifact. Promotion additionally requires "
            "all design gates, including accepted-iterate/restoration traces."
        ),
    }


def _write_markdown(path: Path, artifact: Mapping[str, Any]) -> None:
    lines = [
        "# Fixed-support v2 unbiased GPU verification",
        "",
        f"Backend: `{artifact['environment']['backend']}`",
        f"Frozen baseline integrity: `{artifact['baseline_integrity']['passed']}`",
        f"Schedule: `{artifact['config']['epsilon_schedule']}`",
        "",
        "## Exact-state solver matrix",
        "",
        "| lane | case | support | v1 | v2 | v2 status | KKT pass |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in artifact["solver_matrix"]:
        lines.append(
            f"| {row['lane']} | {row['label']} | {row['support_count']} | "
            f"{row['v1']['converged']} | {row['v2']['converged']} | "
            f"{row['v2']['terminal_status_name']} | "
            f"{row['v2']['independent_kkt_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Support lifecycle (separate gate)",
            "",
            "| family | layers | outcomes | all closed/gas-only |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in artifact["support_lifecycle"]:
        lines.append(
            f"| {row['family']} | {row['layer_count']} | "
            f"`{row['outcome_counts']}` | "
            f"{row['all_non_gas_only_layers_closed']} |"
        )
    lines.extend(
        [
            "",
            "Small-control, large-support, and lifecycle results are intentionally "
            "not combined into one success rate.",
            "",
        ]
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lanes", nargs="+", choices=("solver", "lifecycle"), default=("solver", "lifecycle")
    )
    parser.add_argument("--cases", nargs="+", default=("all",))
    parser.add_argument("--lifecycle-families", nargs="+", default=("manifest",))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--element-file", default="FastChem4/element_abundances/asplund_2021.dat"
    )
    parser.add_argument(
        "--epsilon-schedule", nargs="+", type=float, default=DEFAULT_SCHEDULE
    )
    parser.add_argument("--max-normal-iterations", type=int, default=1000)
    parser.add_argument("--max-line-search-trials", type=int, default=20)
    parser.add_argument("--max-restoration-calls", type=int, default=2)
    parser.add_argument("--max-restoration-iterations", type=int, default=100)
    parser.add_argument(
        "--max-restoration-line-search-trials", type=int, default=20
    )
    parser.add_argument("--v1-residual-tolerance-multiplier", type=float, default=2.0)
    parser.add_argument("--seed-fraction", type=float, default=0.8)
    parser.add_argument("--max-seed-amount", type=float, default=1.0)
    parser.add_argument("--budget-relative-floor", type=float, default=1.0e-6)
    parser.add_argument("--support-closure-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--lifecycle-initial-topk", type=int, default=8)
    parser.add_argument("--lifecycle-initial-max-support", type=int, default=16)
    parser.add_argument("--lifecycle-activity-threshold", type=float, default=0.0)
    parser.add_argument("--lifecycle-add-per-round", type=int, default=8)
    parser.add_argument("--lifecycle-max-support", type=int, default=128)
    parser.add_argument("--lifecycle-max-rounds", type=int, default=4)
    parser.add_argument("--allow-baseline-mismatch", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    args.epsilon_schedule = tuple(float(value) for value in args.epsilon_schedule)
    if not args.epsilon_schedule or any(
        current >= previous
        for previous, current in zip(
            args.epsilon_schedule, args.epsilon_schedule[1:]
        )
    ):
        raise ValueError("epsilon_schedule must be strictly decreasing.")
    positive = {
        "max_normal_iterations": args.max_normal_iterations,
        "max_line_search_trials": args.max_line_search_trials,
        "max_restoration_iterations": args.max_restoration_iterations,
        "max_restoration_line_search_trials": args.max_restoration_line_search_trials,
        "lifecycle_initial_topk": args.lifecycle_initial_topk,
        "lifecycle_initial_max_support": args.lifecycle_initial_max_support,
        "lifecycle_add_per_round": args.lifecycle_add_per_round,
        "lifecycle_max_support": args.lifecycle_max_support,
        "lifecycle_max_rounds": args.lifecycle_max_rounds,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid or args.max_restoration_calls < 0:
        raise ValueError(f"Invalid positive options: {invalid}")
    if args.lifecycle_initial_max_support > args.lifecycle_max_support:
        raise ValueError(
            "lifecycle_initial_max_support may not exceed lifecycle_max_support."
        )
    if args.budget_relative_floor <= 0.0:
        raise ValueError("budget_relative_floor must be positive.")
    if args.support_closure_tolerance < 0.0:
        raise ValueError("support_closure_tolerance must be non-negative.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    jax_config.update("jax_enable_x64", True)
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    integrity = _baseline_integrity(matrix)
    if not integrity["passed"] and not args.allow_baseline_mismatch:
        raise RuntimeError(
            "Frozen v1 baseline artifact integrity failed. Use "
            "--allow-baseline-mismatch only for a deliberate diagnostic run."
        )
    os.environ["EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON_SCHEDULE"] = ",".join(
        str(value) for value in args.epsilon_schedule
    )
    os.environ["EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON"] = str(
        args.epsilon_schedule[-1]
    )
    os.environ["EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_RECENTER"] = "rho"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup = condensate_chemical_setup(
        species_defalt_elements=False,
        element_file=args.element_file,
        silent=True,
    )
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=args.epsilon_schedule,
            initial_state_policy="provided",
        ),
        limits=SolverLimitConfig(
            max_normal_iterations=int(args.max_normal_iterations),
            max_line_search_trials=int(args.max_line_search_trials),
            max_restoration_calls=int(args.max_restoration_calls),
            max_restoration_iterations=int(args.max_restoration_iterations),
            max_restoration_line_search_trials=int(
                args.max_restoration_line_search_trials
            ),
        ),
    )
    lifecycle_config = replace(
        config,
        continuation=replace(
            config.continuation,
            initial_state_policy="center",
        ),
    )
    devices = jax.devices()
    artifact = {
        "schema": "exogibbs_fixed_support_v2_unbiased_gpu_experiment_v2",
        "production_preset_promoted": False,
        "algorithm_reaudit": {
            "canonical_problem": (
                "gamma=hgas+log(P/Pref); current qtot remains in the gas "
                "stationarity residual"
            ),
            "normal_globalization": (
                "one R-GIE Newton family; ordered parallel alpha ladder with "
                "sequential first-acceptable selection"
            ),
            "soc_globalization": (
                "M5 method-0 recurrence with separate alpha_test, alpha_soc, "
                "alpha_y, and alpha_dual; its first-order amount limiter is "
                "not applied to ordinary normal steps"
            ),
            "restoration": (
                "persistent elastic amount-space NLP; stable Schur elimination "
                "for trace amounts; audited scale-aware interior push on entry; "
                "barrier-merit Armijo with separate constraint nonincrease"
            ),
            "controller": (
                "NORMAL/SOC/RESTORATION ownership is phase-separated and every "
                "failure has one terminal status"
            ),
            "continuation": (
                "a stricter epsilon stage starts only after convergence of the "
                "previous complete fixed-epsilon solve"
            ),
            "support_lifecycle": (
                "support changes only outside continuation after inactive-KKT "
                "and temperature-validity checks"
            ),
            "terminal_restoration_replay_exported": True,
            "terminal_normal_replay_exported": True,
            "stage_restoration_accepted_iteration_counts_exported": True,
            "stage_restoration_return_diagnostics_exported": True,
        },
        "environment": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "devices": [
                {
                    "id": int(device.id),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                }
                for device in devices
            ],
        },
        "config": {
            "epsilon_schedule": list(args.epsilon_schedule),
            "max_normal_iterations": int(args.max_normal_iterations),
            "max_line_search_trials": int(args.max_line_search_trials),
            "max_restoration_calls": int(args.max_restoration_calls),
            "max_restoration_iterations": int(args.max_restoration_iterations),
            "max_restoration_line_search_trials": int(
                args.max_restoration_line_search_trials
            ),
            "v1_rho_initialization": "provided",
            "v1_lambda_initialization": "provided",
            "v1_continuation_recenter": "rho",
            "v2_initial_state_policy": "provided",
            "v2_continuation_recenter": "rho",
            "lifecycle_v2_initial_state_policy": "center",
            "lifecycle_support_restart_rho": "epsilon-r",
            "restoration_interior_push_fraction": (
                config.restoration.interior_push_fraction
            ),
            "restoration_entry_injection_tolerance": (
                config.restoration.representation_floor_injection_tolerance
            ),
            "v1_residual_tolerance_multiplier": float(
                args.v1_residual_tolerance_multiplier
            ),
            "budget_relative_floor": float(args.budget_relative_floor),
            "support_closure_tolerance": float(args.support_closure_tolerance),
        },
        "baseline_integrity": integrity,
        "prior_validation_integrity": _prior_validation_integrity(),
        "source_integrity": _source_integrity(),
        "solver_matrix": [],
        "support_lifecycle": [],
        "preflight_only": bool(args.preflight_only),
    }

    declared_cases = list(matrix["solver_matrix"])
    selected_cases = (
        declared_cases
        if args.cases == ["all"] or tuple(args.cases) == ("all",)
        else [case for case in declared_cases if case["label"] in set(args.cases)]
    )
    unknown_cases = set(args.cases) - {case["label"] for case in declared_cases} - {"all"}
    if unknown_cases:
        raise ValueError(f"Unknown matrix cases: {sorted(unknown_cases)}")
    lifecycle_families = (
        tuple(matrix["lifecycle_families"])
        if args.lifecycle_families == ["manifest"]
        or tuple(args.lifecycle_families) == ("manifest",)
        else tuple(args.lifecycle_families)
    )
    unknown_families = set(lifecycle_families) - set(FRESH_CURATED_PROFILES)
    if unknown_families:
        raise ValueError(f"Unknown lifecycle families: {sorted(unknown_families)}")
    if args.preflight_only:
        for case in selected_cases:
            definition = FRESH_CURATED_PROFILES[case["family"]]
            budget = jnp.asarray(
                element_budget_for_profile(setup, definition), dtype=jnp.float64
            )
            source_index = int(case["layer"])
            support = _support_for_case(
                setup=setup,
                definition=definition,
                source_index=source_index,
                budget=budget,
                case=case,
            )
            init = _gas_initial_state(
                setup=setup,
                definition=definition,
                source_index=source_index,
                budget=budget,
                support=support,
                first_epsilon=float(args.epsilon_schedule[0]),
                seed_fraction=float(args.seed_fraction),
                max_seed_amount=float(args.max_seed_amount),
            )
            plan = _prepare_plan(
                setup=setup,
                definition=definition,
                source_indices=(source_index,),
                inits=(init,),
                budget=budget,
                max_normal_iterations=int(args.max_normal_iterations),
                seed_fraction=float(args.seed_fraction),
                max_seed_amount=float(args.max_seed_amount),
            )
            state_audit = _audit_prepared_initial_state(plan, config)
            count_passed = case.get("expected_support_count") in {
                None,
                len(support),
            }
            artifact["solver_matrix"].append(
                {
                    "label": case["label"],
                    "support_count": len(support),
                    "expected_support_count": case.get("expected_support_count"),
                    "initial_state_contract": state_audit,
                    "passed": count_passed and state_audit["passed"],
                }
            )
        for family in lifecycle_families:
            definition = FRESH_CURATED_PROFILES[family]
            budget = jnp.asarray(
                element_budget_for_profile(setup, definition), dtype=jnp.float64
            )
            source_indices = []
            inits = []
            support_counts = []
            for source_index in range(len(definition.temperatures)):
                support = _native_initial_support(
                    setup=setup,
                    definition=definition,
                    source_index=source_index,
                    budget=budget,
                    args=args,
                )
                support_counts.append(len(support))
                if not support:
                    continue
                source_indices.append(source_index)
                inits.append(
                    _gas_initial_state(
                        setup=setup,
                        definition=definition,
                        source_index=source_index,
                        budget=budget,
                        support=support,
                        first_epsilon=float(args.epsilon_schedule[0]),
                        seed_fraction=float(args.seed_fraction),
                        max_seed_amount=float(args.max_seed_amount),
                        center_bound_multipliers=True,
                    )
                )
            if source_indices:
                plan = _prepare_plan(
                    setup=setup,
                    definition=definition,
                    source_indices=tuple(source_indices),
                    inits=tuple(inits),
                    budget=budget,
                    max_normal_iterations=int(args.max_normal_iterations),
                    seed_fraction=float(args.seed_fraction),
                    max_seed_amount=float(args.max_seed_amount),
                )
                state_audit = _audit_prepared_initial_state(
                    plan, lifecycle_config
                )
                passed = bool(state_audit["passed"])
            else:
                state_audit = None
                passed = True
            artifact["support_lifecycle"].append(
                {
                    "family": family,
                    "layer_count": len(definition.temperatures),
                    "active_layer_count": len(source_indices),
                    "initial_support_counts": support_counts,
                    "initial_state_policy": "center",
                    "initial_state_contract": state_audit,
                    "passed": passed,
                }
            )
        artifact["preflight_passed"] = all(
            row["passed"] for row in artifact["solver_matrix"]
        ) and all(row["passed"] for row in artifact["support_lifecycle"])
        output = args.output_dir / "preflight.json"
        _write_json(output, artifact)
        print(f"wrote {output}")
        if not artifact["preflight_passed"]:
            raise RuntimeError("Fixed-support v2 preflight contract failed.")
        return

    if "solver" in args.lanes:
        for case in selected_cases:
            print(
                f"== solver matrix: {case['label']} ({case['policy']}) ==",
                flush=True,
            )
            artifact["solver_matrix"].append(
                _run_solver_case(
                    setup=setup,
                    matrix_case=case,
                    config=config,
                    args=args,
                )
            )
            artifact["solver_aggregate"] = _aggregate_solver_cases(
                artifact["solver_matrix"]
            )
            _write_json(args.output_dir / "summary.json", artifact)

    if "lifecycle" in args.lanes:
        for family in lifecycle_families:
            print(f"== support lifecycle: {family} ==", flush=True)
            artifact["support_lifecycle"].append(
                _run_lifecycle_family(
                    setup=setup,
                    family=family,
                    config=lifecycle_config,
                    args=args,
                )
            )
            _write_json(args.output_dir / "summary.json", artifact)

    if artifact["solver_matrix"]:
        artifact["solver_aggregate"] = _aggregate_solver_cases(
            artifact["solver_matrix"]
        )
    artifact["lifecycle_aggregate"] = {
        "family_count": len(artifact["support_lifecycle"]),
        "all_families_closed": bool(artifact["support_lifecycle"])
        and all(
            row["all_non_gas_only_layers_closed"]
            for row in artifact["support_lifecycle"]
        ),
    }
    _write_json(args.output_dir / "summary.json", artifact)
    _write_markdown(args.output_dir / "summary.md", artifact)
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
