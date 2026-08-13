"""Measure repeated warm L-dwarf production-profile evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform as system_platform
import resource
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.documented_examples.worker import _checkout_package_path
from benchmarks.documented_examples.worker import _jax_environment
from benchmarks.documented_examples.worker import _revision
from benchmarks.documented_examples.worker import _source_provenance
from benchmarks.documented_examples.worker import _timestamp
from benchmarks.documented_examples.worker import _tree_fingerprint
from benchmarks.documented_examples.worker import _write_json
from benchmarks.documented_examples.worker import configure_jax_environment
from benchmarks.documented_examples.worker import OPTIMIZATION_MODES


SCHEMA = "exogibbs_ldwarf_repeated_profile_benchmark_v1"
DEFAULT_EVALUATION_COUNT = 10
DEFAULT_SEED = 0
PRESSURE_OFFSET_LIMIT_DEX = 0.005
TEMPERATURE_OFFSET_LIMIT_K = 10.0
TEMPERATURE_TILT_LIMIT_K = 5.0
SOURCE_SCRIPTS = (
    "examples/comparisons/comparison_with_fastchem4_condensates.py",
)


def generate_conditions(
    base_temperatures: Sequence[float],
    base_pressures: Sequence[float],
    *,
    evaluation_count: int = DEFAULT_EVALUATION_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate deterministic, smooth, non-extreme T-P perturbations."""

    temperatures = np.asarray(base_temperatures, dtype=np.float64)
    pressures = np.asarray(base_pressures, dtype=np.float64)
    if evaluation_count <= 0:
        raise ValueError("evaluation_count must be positive.")
    if (
        temperatures.ndim != 1
        or pressures.ndim != 1
        or temperatures.shape != pressures.shape
        or temperatures.size == 0
    ):
        raise ValueError("Base T and P must be non-empty equal-length vectors.")
    if not np.all(np.isfinite(temperatures)) or not np.all(
        np.isfinite(pressures) & (pressures > 0.0)
    ):
        raise ValueError("Base T and P must be finite and P must be positive.")

    generator = np.random.default_rng(seed)
    unit_offsets = generator.uniform(-1.0, 1.0, size=(evaluation_count, 3))
    pressure_offsets = PRESSURE_OFFSET_LIMIT_DEX * unit_offsets[:, 0]
    temperature_offsets = TEMPERATURE_OFFSET_LIMIT_K * unit_offsets[:, 1]
    temperature_tilts = TEMPERATURE_TILT_LIMIT_K * unit_offsets[:, 2]
    coordinate = np.linspace(-1.0, 1.0, temperatures.size)
    generated_pressures = pressures[None, :] * np.power(
        10.0, pressure_offsets[:, None]
    )
    generated_temperatures = (
        temperatures[None, :]
        + temperature_offsets[:, None]
        + temperature_tilts[:, None] * coordinate[None, :]
    )
    if not np.all(
        np.isfinite(generated_temperatures) & (generated_temperatures > 0.0)
    ) or not np.all(
        np.isfinite(generated_pressures) & (generated_pressures > 0.0)
    ):
        raise RuntimeError("Generated T-P profiles must be finite and positive.")
    if not np.all(np.diff(generated_pressures, axis=1) > 0.0):
        raise RuntimeError("Generated pressure profiles are not monotonic.")
    if not np.all(np.diff(generated_temperatures, axis=1) > 0.0):
        raise RuntimeError("Generated temperature profiles are not monotonic.")

    fingerprint_payload = {
        "seed": seed,
        "pressure_offset_limit_dex": PRESSURE_OFFSET_LIMIT_DEX,
        "temperature_offset_limit_k": TEMPERATURE_OFFSET_LIMIT_K,
        "temperature_tilt_limit_k": TEMPERATURE_TILT_LIMIT_K,
        "temperature_k": generated_temperatures.tolist(),
        "pressure_bar": generated_pressures.tolist(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        **fingerprint_payload,
        "sha256": fingerprint,
        "pressure_offset_dex": pressure_offsets.tolist(),
        "temperature_offset_k": temperature_offsets.tolist(),
        "temperature_tilt_k": temperature_tilts.tolist(),
        "temperature_k": generated_temperatures,
        "pressure_bar": generated_pressures,
    }


def steady_statistics(
    wall_seconds: Sequence[float],
) -> dict[str, float | int]:
    """Summarize directly measured post-cold evaluation latency."""

    values = np.asarray(wall_seconds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("wall_seconds must contain at least one value.")
    if not np.all(np.isfinite(values) & (values >= 0.0)):
        raise ValueError("wall_seconds must be finite and non-negative.")
    total = float(np.sum(values))
    return {
        "evaluation_count": int(values.size),
        "total_wall_seconds": total,
        "mean_wall_seconds_per_evaluation": float(np.mean(values)),
        "median_wall_seconds_per_evaluation": float(np.median(values)),
        "p95_wall_seconds_per_evaluation": float(np.percentile(values, 95.0)),
        "minimum_wall_seconds_per_evaluation": float(np.min(values)),
        "maximum_wall_seconds_per_evaluation": float(np.max(values)),
        "standard_deviation_seconds": float(np.std(values)),
    }


def _budget_per_evaluation(
    budget: Mapping[str, Any], evaluation_count: int
) -> dict[str, float]:
    names = (
        "wall_seconds",
        "pdipm_call_count",
        "pdipm_bucket_count",
        "pdipm_wall_seconds",
        "pdipm_compilation_seconds",
        "pdipm_execution_seconds",
        "pdipm_diagnostic_seconds",
        "pdipm_diagnostic_compilation_seconds",
        "pdipm_diagnostic_execution_seconds",
        "pdipm_internal_orchestration_wall_seconds",
        "zero_barrier_call_count",
        "zero_barrier_host_wall_seconds",
        "zero_barrier_function_evaluations",
        "outside_pdipm_and_zero_barrier_wall_seconds",
    )
    return {
        name: float(budget[name]) / evaluation_count
        for name in names
    }


def _support_signature(profile: Any) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(str(name) for name in layer.condensate_support_names)
        for layer in profile.layers
    )


def _support_sha256(signature: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _profile_validation(profile: Any) -> dict[str, Any]:
    from benchmarks.documented_examples.workloads import _profile_physical_audit
    from benchmarks.documented_examples.workloads import _profile_summary

    summary = _profile_summary(profile)
    audit = _profile_physical_audit(profile)
    signature = _support_signature(profile)
    accepted = bool(
        summary["all_layers_converged"]
        and audit["all_layers_finite_and_physically_accepted"]
    )
    return {
        "accepted": accepted,
        "summary": summary,
        "physical_audit": audit,
        "support_signature": signature,
        "support_sha256": _support_sha256(signature),
    }


def _shape_signatures(timing: Mapping[str, Any]) -> tuple[str, ...]:
    pdipm = timing.get("pdipm", {})
    if not isinstance(pdipm, Mapping):
        return ()
    records = pdipm.get("shape_compilation", ())
    return tuple(
        sorted(
            str(record["signature"])
            for record in records
            if isinstance(record, Mapping) and "signature" in record
        )
    )


def _pdipm_support_signature(
    calls: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], ...]:
    """Return the real per-layer supports used by each PD-IPM call."""

    def bucket_entries(
        bucket: Mapping[str, Any],
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        layer_indices = tuple(
            int(value)
            for value in bucket.get(
                "source_layer_indices",
                bucket.get("layer_indices", ()),
            )
        )
        supports_by_layer = tuple(
            tuple(int(value) for value in support)
            for support in bucket.get("support_indices_by_layer", ())
        )
        if supports_by_layer:
            if len(supports_by_layer) != len(layer_indices):
                raise ValueError(
                    "PD-IPM support metadata must have one row per layer."
                )
            return tuple(
                ((layer_index,), support)
                for layer_index, support in zip(
                    layer_indices, supports_by_layer
                )
            )
        return (
            (
                layer_indices,
                tuple(
                    int(value)
                    for value in bucket.get("support_indices", ())
                ),
            ),
        )

    return tuple(
        tuple(
            sorted(
                entry
                for bucket in call.get("buckets", ())
                for entry in bucket_entries(bucket)
            )
        )
        for call in calls
    )


def _new_shape_signatures(
    cold_timing: Mapping[str, Any], steady_timing: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(_shape_signatures(steady_timing))
            - set(_shape_signatures(cold_timing))
        )
    )


def _result_status(
    *,
    all_evaluations_accepted: bool,
    timing_attribution_consistent: bool,
    no_new_pdipm_executable_shapes_after_cold: bool,
) -> str:
    if not all_evaluations_accepted:
        return "fail_validation"
    if not timing_attribution_consistent:
        return "fail_timing"
    if not no_new_pdipm_executable_shapes_after_cold:
        return "fail_recompilation"
    return "pass"


def _run_benchmark(
    *, evaluation_count: int, seed: int, smoke_layers: int | None
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from benchmarks.documented_examples.instrumentation import TimingCollector
    from examples.comparisons import (
        comparison_with_fastchem4_condensates as example,
    )
    from exogibbs.api.condensate import CondensateEquilibriumOptions

    base_temperatures, base_pressures = example._profile_conditions("l-dwarf")
    if smoke_layers is not None:
        base_temperatures = base_temperatures[:smoke_layers]
        base_pressures = base_pressures[:smoke_layers]
    conditions = generate_conditions(
        base_temperatures,
        base_pressures,
        evaluation_count=evaluation_count,
        seed=seed,
    )

    setup_started = time.perf_counter()
    setup = example.condensate_chemical_setup(
        gas_path="FastChem4/logK/logK_wo_ions.dat",
        condensate_path="FastChem4/logK/logK_condensates.dat",
        species_default_elements=False,
        element_file="FastChem4/element_abundances/asplund_2021.dat",
        silent=True,
    )
    aligned_abundance = example.build_aligned_abundance_vector(
        setup.elements,
        source="fastchem_file",
        normalize=True,
        element_file=example.ELEMENT_FILE,
    )
    budget = jnp.asarray(aligned_abundance.vector, dtype=jnp.float64)
    setup_seconds = time.perf_counter() - setup_started
    options = CondensateEquilibriumOptions(return_diagnostics=False)

    def solve(temperature: Any, pressure: Any) -> Any:
        profile = example.solve_condensate_profile(
            setup,
            T=jnp.asarray(temperature, dtype=jnp.float64),
            P=jnp.asarray(pressure, dtype=jnp.float64),
            b=budget,
            options=options,
            return_diagnostics=False,
        )
        support_arrays = tuple(
            layer.condensate_support_indices for layer in profile.layers
        )
        jax.block_until_ready((profile.batched_arrays, support_arrays))
        return profile

    cold_collector = TimingCollector()
    cold_started = time.perf_counter()
    with cold_collector:
        with cold_collector.phase("cold_nominal", category="solver"):
            cold_profile = solve(base_temperatures, base_pressures)
    cold_validation = _profile_validation(cold_profile)
    cold_workload_seconds = time.perf_counter() - cold_started
    cold_timing = cold_collector.summary(
        workload_wall_seconds=cold_workload_seconds
    )
    cold_shapes = set(_shape_signatures(cold_timing))
    cold_pdipm_support = _pdipm_support_signature(
        cold_timing["pdipm"]["calls"]
    )

    steady_collector = TimingCollector()
    validations = []
    steady_started = time.perf_counter()
    with steady_collector:
        for index in range(evaluation_count):
            print(
                f"STEADY {index + 1}/{evaluation_count}",
                flush=True,
            )
            phase = f"evaluation_{index + 1:03d}"
            with steady_collector.phase(phase, category="solver"):
                profile = solve(
                    conditions["temperature_k"][index],
                    conditions["pressure_bar"][index],
                )
            validations.append(_profile_validation(profile))
    steady_workload_seconds = time.perf_counter() - steady_started
    steady_timing = steady_collector.summary(
        workload_wall_seconds=steady_workload_seconds
    )
    phase_budgets = {
        str(record["name"]): record
        for record in steady_timing["phase_budgets"]
    }
    evaluation_records = []
    evaluation_wall = []
    cold_support = cold_validation["support_signature"]
    seen_shapes = set(cold_shapes)
    for index, validation in enumerate(validations):
        phase = f"evaluation_{index + 1:03d}"
        phase_budget = phase_budgets[phase]
        phase_calls = tuple(
            call
            for call in steady_timing["pdipm"]["calls"]
            if call.get("phase") == phase
        )
        phase_shapes = tuple(
            sorted(
                {
                    str(bucket["signature"])
                    for call in phase_calls
                    for bucket in call.get("buckets", ())
                }
            )
        )
        phase_pdipm_support = _pdipm_support_signature(phase_calls)
        new_phase_shapes = tuple(sorted(set(phase_shapes) - seen_shapes))
        seen_shapes.update(phase_shapes)
        changed_support_layers = tuple(
            layer_index
            for layer_index, (cold_layer, current_layer) in enumerate(
                zip(cold_support, validation["support_signature"])
            )
            if cold_layer != current_layer
        )
        wall_seconds = float(phase_budget["wall_seconds"])
        evaluation_wall.append(wall_seconds)
        evaluation_records.append(
            {
                "evaluation": index + 1,
                "temperature_offset_k": conditions["temperature_offset_k"][
                    index
                ],
                "temperature_tilt_k": conditions["temperature_tilt_k"][index],
                "pressure_offset_dex": conditions["pressure_offset_dex"][index],
                "minimum_temperature_k": float(
                    np.min(conditions["temperature_k"][index])
                ),
                "maximum_temperature_k": float(
                    np.max(conditions["temperature_k"][index])
                ),
                "minimum_pressure_bar": float(
                    np.min(conditions["pressure_bar"][index])
                ),
                "maximum_pressure_bar": float(
                    np.max(conditions["pressure_bar"][index])
                ),
                "wall_seconds": wall_seconds,
                "validation": validation,
                "executable_shape_signatures": phase_shapes,
                "pdipm_support_signature": phase_pdipm_support,
                "pdipm_support_identity_preserved": (
                    phase_pdipm_support == cold_pdipm_support
                ),
                "new_executable_shape_signatures_after_cold": (
                    new_phase_shapes
                ),
                "changed_support_layer_indices": changed_support_layers,
                "budget": phase_budget,
            }
        )

    new_shapes = _new_shape_signatures(cold_timing, steady_timing)
    final_support_preserved = all(
        validation["support_signature"]
        == cold_validation["support_signature"]
        for validation in validations
    )
    pdipm_support_preserved = all(
        record["pdipm_support_identity_preserved"]
        for record in evaluation_records
    )
    all_accepted = cold_validation["accepted"] and all(
        validation["accepted"] for validation in validations
    )
    statistics = steady_statistics(evaluation_wall)
    cold_compilation_boundary = float(
        cold_timing["solver_budget"]["pdipm_compilation_seconds"]
        + cold_timing["solver_budget"][
            "pdipm_diagnostic_compilation_seconds"
        ]
    )
    steady_compilation_boundary = float(
        steady_timing["solver_budget"]["pdipm_compilation_seconds"]
        + steady_timing["solver_budget"][
            "pdipm_diagnostic_compilation_seconds"
        ]
    )
    timing_valid = bool(
        cold_timing["timing_attribution_consistent"]
        and steady_timing["timing_attribution_consistent"]
    )
    result_status = _result_status(
        all_evaluations_accepted=all_accepted,
        timing_attribution_consistent=timing_valid,
        no_new_pdipm_executable_shapes_after_cold=not new_shapes,
    )
    late_shape_evaluations = tuple(
        record["evaluation"]
        for record in evaluation_records
        if record["new_executable_shape_signatures_after_cold"]
    )
    changed_support_evaluations = tuple(
        record["evaluation"]
        for record in evaluation_records
        if record["changed_support_layer_indices"]
    )
    changed_pdipm_support_evaluations = tuple(
        record["evaluation"]
        for record in evaluation_records
        if not record["pdipm_support_identity_preserved"]
    )
    steady_budget = steady_timing["solver_budget"]
    return {
        "status": result_status,
        "scope": {
            "kind": "smoke" if smoke_layers is not None else "full",
            "scenario": "nuts_like_production_forward",
            "nuts_representative": False,
            "includes": (
                "full-catalog production support lifecycle",
                "fixed-support PD-IPM",
                "host zero-barrier refinement",
                "result construction and synchronization",
            ),
            "excludes": (
                "independent gas-only comparison profile",
                "reverse-mode gradient",
                "spectral likelihood",
                "NUTS transition logic",
            ),
        },
        "configuration": {
            "evaluation_count": evaluation_count,
            "seed": seed,
            "layer_count": len(base_temperatures),
            "smoke_layers": smoke_layers,
            "return_diagnostics": False,
        },
        "conditions": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in conditions.items()
        },
        "setup": {"wall_seconds": setup_seconds},
        "cold": {
            "validation": cold_validation,
            "timing": cold_timing,
            "budget": cold_timing["solver_budget"],
            "pdipm_compilation_boundary_seconds": (
                cold_compilation_boundary
            ),
            "shape_signatures": tuple(sorted(cold_shapes)),
            "pdipm_support_signature": cold_pdipm_support,
        },
        "steady": {
            **statistics,
            "workload_wall_seconds_including_validation": (
                steady_workload_seconds
            ),
            "budget_total": steady_budget,
            "budget_per_evaluation": _budget_per_evaluation(
                steady_budget, evaluation_count
            ),
            "pdipm_compilation_boundary_seconds": (
                steady_compilation_boundary
            ),
            "new_pdipm_executable_shape_count_after_cold": len(new_shapes),
            "new_pdipm_executable_shape_signatures": new_shapes,
            "late_shape_evaluations": late_shape_evaluations,
            "no_new_pdipm_executable_shapes_after_cold": not new_shapes,
            "final_support_identity_preserved": final_support_preserved,
            "changed_final_support_evaluations": (
                changed_support_evaluations
            ),
            "pdipm_support_identity_preserved": pdipm_support_preserved,
            "changed_pdipm_support_evaluations": (
                changed_pdipm_support_evaluations
            ),
            "nuts_fixed_support_assumption_met": (
                final_support_preserved and pdipm_support_preserved
            ),
            "timing": steady_timing,
            "evaluations": tuple(evaluation_records),
        },
        "validation": {
            "all_evaluations_accepted": all_accepted,
            "timing_attribution_consistent": timing_valid,
            "no_new_pdipm_executable_shapes_after_cold": not new_shapes,
            "final_support_identity_preserved": final_support_preserved,
            "pdipm_support_identity_preserved": pdipm_support_preserved,
            "failed_evaluations": tuple(
                record["evaluation"]
                for record in evaluation_records
                if not record["validation"]["accepted"]
            ),
        },
        "catalog": {
            "element_count": len(setup.elements),
            "gas_species_count": len(setup.gas_species),
            "condensate_species_count": len(setup.condensate_species),
        },
    }


def _evaluation_row(record: Mapping[str, Any]) -> dict[str, Any]:
    budget = record["budget"]
    return {
        "evaluation": record["evaluation"],
        "temperature_offset_k": record["temperature_offset_k"],
        "temperature_tilt_k": record["temperature_tilt_k"],
        "pressure_offset_dex": record["pressure_offset_dex"],
        "minimum_temperature_k": record["minimum_temperature_k"],
        "maximum_temperature_k": record["maximum_temperature_k"],
        "minimum_pressure_bar": record["minimum_pressure_bar"],
        "maximum_pressure_bar": record["maximum_pressure_bar"],
        "wall_seconds": record["wall_seconds"],
        "accepted": record["validation"]["accepted"],
        "support_sha256": record["validation"]["support_sha256"],
        "pdipm_support_identity_preserved": record[
            "pdipm_support_identity_preserved"
        ],
        "executable_shape_signatures": ";".join(
            record["executable_shape_signatures"]
        ),
        "new_executable_shape_signatures_after_cold": ";".join(
            record["new_executable_shape_signatures_after_cold"]
        ),
        "changed_support_layer_indices": ";".join(
            str(index) for index in record["changed_support_layer_indices"]
        ),
        "pdipm_call_count": budget["pdipm_call_count"],
        "pdipm_bucket_count": budget["pdipm_bucket_count"],
        "pdipm_wall_seconds": budget["pdipm_wall_seconds"],
        "pdipm_compilation_seconds": budget["pdipm_compilation_seconds"],
        "pdipm_execution_seconds": budget["pdipm_execution_seconds"],
        "pdipm_diagnostic_seconds": budget["pdipm_diagnostic_seconds"],
        "pdipm_diagnostic_compilation_seconds": budget[
            "pdipm_diagnostic_compilation_seconds"
        ],
        "pdipm_diagnostic_execution_seconds": budget[
            "pdipm_diagnostic_execution_seconds"
        ],
        "pdipm_internal_orchestration_wall_seconds": budget[
            "pdipm_internal_orchestration_wall_seconds"
        ],
        "zero_barrier_call_count": budget["zero_barrier_call_count"],
        "zero_barrier_host_wall_seconds": budget[
            "zero_barrier_host_wall_seconds"
        ],
        "zero_barrier_function_evaluations": budget[
            "zero_barrier_function_evaluations"
        ],
        "outside_pdipm_and_zero_barrier_wall_seconds": budget[
            "outside_pdipm_and_zero_barrier_wall_seconds"
        ],
        "attribution_delta_seconds": budget["attribution_delta_seconds"],
        "attribution_consistent": budget["attribution_consistent"],
    }


def _write_evaluations_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [_evaluation_row(record) for record in records]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    cold = payload["cold"]
    steady = payload["steady"]
    per_evaluation = steady["budget_per_evaluation"]
    summary_rows = (
        ("evaluation_count", steady["evaluation_count"]),
        ("total_wall_seconds", steady["total_wall_seconds"]),
        (
            "mean_wall_seconds_per_evaluation",
            steady["mean_wall_seconds_per_evaluation"],
        ),
        (
            "median_wall_seconds_per_evaluation",
            steady["median_wall_seconds_per_evaluation"],
        ),
        (
            "p95_wall_seconds_per_evaluation",
            steady["p95_wall_seconds_per_evaluation"],
        ),
        (
            "new_pdipm_executable_shape_count_after_cold",
            steady["new_pdipm_executable_shape_count_after_cold"],
        ),
        (
            "pdipm_calls_per_evaluation",
            per_evaluation["pdipm_call_count"],
        ),
        (
            "pdipm_buckets_per_evaluation",
            per_evaluation["pdipm_bucket_count"],
        ),
        (
            "no_new_pdipm_executable_shapes_after_cold",
            steady["no_new_pdipm_executable_shapes_after_cold"],
        ),
        (
            "final_support_identity_preserved",
            steady["final_support_identity_preserved"],
        ),
        (
            "pdipm_support_identity_preserved",
            steady["pdipm_support_identity_preserved"],
        ),
    )
    budget_fields = (
        "pdipm_compilation_seconds",
        "pdipm_execution_seconds",
        "pdipm_diagnostic_compilation_seconds",
        "pdipm_diagnostic_execution_seconds",
        "pdipm_internal_orchestration_wall_seconds",
        "zero_barrier_host_wall_seconds",
        "outside_pdipm_and_zero_barrier_wall_seconds",
    )
    lines = [
        "# Repeated L-dwarf production-profile benchmark",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Cold solver wall: {cold['timing']['solver_phase_wall_seconds']} s",
        "Cold explicit PD-IPM compile boundary: "
        f"{cold['pdipm_compilation_boundary_seconds']} s",
        f"Cold PD-IPM calls: {cold['budget']['pdipm_call_count']}",
        f"Cold PD-IPM buckets: {cold['budget']['pdipm_bucket_count']}",
        "",
        "| Steady metric | Value |",
        "| --- | ---: |",
        *(f"| {name} | {value} |" for name, value in summary_rows),
        "",
        "| Additive wall budget per evaluation | Seconds |",
        "| --- | ---: |",
        *(f"| {name} | {per_evaluation[name]} |" for name in budget_fields),
        "",
        "This is a synchronized production forward benchmark, not a "
        "value-and-gradient or NUTS transition benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--optimization",
        choices=OPTIMIZATION_MODES,
        default="default",
    )
    parser.add_argument(
        "--evaluations", type=int, default=DEFAULT_EVALUATION_COUNT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-layers", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.evaluations <= 0:
        raise SystemExit("--evaluations must be positive")
    if args.smoke_layers is not None and args.smoke_layers <= 0:
        raise SystemExit("--smoke-layers must be positive")

    configure_jax_environment(args.platform, args.optimization)
    repository_root = Path(__file__).resolve().parents[2]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": _timestamp(),
        "execution": {
            "requested_platform": args.platform,
            "optimization_mode": args.optimization,
        },
        "revision": _revision(repository_root),
        "solver_source_tree": _tree_fingerprint(
            repository_root / "src" / "exogibbs"
        ),
        "source_provenance": _source_provenance(
            repository_root, SOURCE_SCRIPTS
        ),
    }
    try:
        environment = _jax_environment(args.platform, args.optimization)
        environment["exogibbs_package_root"] = _checkout_package_path(
            repository_root
        )
        payload["environment"] = environment
    except NotImplementedError as error:
        payload.update(
            {
                "status": "unsupported",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 2
    except Exception as error:
        payload.update(
            {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 2
    else:
        try:
            result = _run_benchmark(
                evaluation_count=args.evaluations,
                seed=args.seed,
                smoke_layers=args.smoke_layers,
            )
            payload.update(result)
            return_code = 0 if payload["status"] == "pass" else 1
        except Exception as error:
            payload.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            return_code = 1

    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_resident_set_size_kb = float(usage.ru_maxrss)
    if system_platform.system() == "Darwin":
        maximum_resident_set_size_kb /= 1024.0
    payload["resources"] = {
        "maximum_resident_set_size_kb": maximum_resident_set_size_kb
    }
    payload["finished_at_utc"] = _timestamp()
    if payload["status"] == "pass":
        csv_path = args.output.with_name(f"{args.output.stem}.evaluations.csv")
        markdown_path = args.output.with_name(f"{args.output.stem}.md")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_evaluations_csv(csv_path, payload["steady"]["evaluations"])
        _write_markdown(markdown_path, payload)
        payload["artifacts"] = {
            "evaluations_csv": str(csv_path),
            "summary_markdown": str(markdown_path),
        }
    _write_json(args.output, payload)
    print(f"{payload['status'].upper()} {args.output}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
