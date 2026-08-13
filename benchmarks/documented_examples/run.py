"""Run the documented example benchmark matrix in isolated processes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.documented_examples.instrumentation import SOLVER_BUDGET_SCHEMA
from benchmarks.documented_examples.manifest import CASES, CASES_BY_ID
from benchmarks.documented_examples.worker import OPTIMIZATION_MODES
from benchmarks.documented_examples.worker import SCHEMA as WORKER_SCHEMA


SCHEMA = "exogibbs_documented_example_benchmark_suite_v2"
PHASE_BUDGET_FIELDS = (
    "case_id",
    "platform",
    "optimization",
    "repetition",
    "scope",
    "status",
    "phase_name",
    "phase_category",
    "phase_invocation_count",
    "phase_wall_seconds",
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
    "attribution_delta_seconds",
    "attribution_consistent",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "missing_result",
            "error": f"{type(error).__name__}: {error}",
        }
    return payload if isinstance(payload, dict) else {"status": "invalid_result"}


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _budget_violations(
    budget: Mapping[str, Any], *, label: str
) -> list[str]:
    violations = []
    if budget.get("schema") != SOLVER_BUDGET_SCHEMA:
        violations.append(f"pass result has invalid {label} schema")
    wall_seconds = budget.get("wall_seconds")
    tolerance = 1.0e-9 * max(
        float(wall_seconds) if _is_finite_number(wall_seconds) else 0.0,
        1.0,
    )
    for name in (
        "wall_seconds",
        "pdipm_wall_seconds",
        "pdipm_compilation_seconds",
        "pdipm_execution_seconds",
        "pdipm_diagnostic_seconds",
        "pdipm_diagnostic_compilation_seconds",
        "pdipm_diagnostic_execution_seconds",
        "pdipm_internal_orchestration_wall_seconds",
        "zero_barrier_host_wall_seconds",
        "outside_pdipm_and_zero_barrier_wall_seconds",
        "attribution_delta_seconds",
    ):
        value = budget.get(name)
        if not _is_finite_number(value) or (
            name != "attribution_delta_seconds" and float(value) < -tolerance
        ):
            violations.append(f"pass result has invalid {label} metric {name}")
    for name in (
        "pdipm_call_count",
        "pdipm_bucket_count",
        "zero_barrier_call_count",
        "zero_barrier_function_evaluations",
    ):
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            violations.append(f"pass result has invalid {label} count {name}")
    if budget.get("attribution_consistent") is not True:
        violations.append(f"pass result has inconsistent {label} attribution")
    values = {
        name: float(budget[name])
        for name in (
            "wall_seconds",
            "pdipm_wall_seconds",
            "pdipm_compilation_seconds",
            "pdipm_execution_seconds",
            "pdipm_diagnostic_seconds",
            "pdipm_diagnostic_compilation_seconds",
            "pdipm_diagnostic_execution_seconds",
            "pdipm_internal_orchestration_wall_seconds",
            "zero_barrier_host_wall_seconds",
            "outside_pdipm_and_zero_barrier_wall_seconds",
            "attribution_delta_seconds",
        )
        if _is_finite_number(budget.get(name))
    }
    if len(values) == 11:
        equations = (
            (
                values["pdipm_wall_seconds"],
                values["pdipm_compilation_seconds"]
                + values["pdipm_execution_seconds"]
                + values["pdipm_diagnostic_seconds"]
                + values["pdipm_internal_orchestration_wall_seconds"],
                "PD-IPM partition",
            ),
            (
                values["pdipm_diagnostic_seconds"],
                values["pdipm_diagnostic_compilation_seconds"]
                + values["pdipm_diagnostic_execution_seconds"],
                "diagnostic partition",
            ),
            (
                values["wall_seconds"],
                values["pdipm_wall_seconds"]
                + values["zero_barrier_host_wall_seconds"]
                + values[
                    "outside_pdipm_and_zero_barrier_wall_seconds"
                ],
                "wall partition",
            ),
        )
        for actual, expected, equation in equations:
            if not math.isclose(actual, expected, abs_tol=tolerance):
                violations.append(
                    f"pass result has inconsistent {label} {equation}"
                )
        if abs(values["attribution_delta_seconds"]) > tolerance:
            violations.append(
                f"pass result has nonzero {label} attribution delta"
            )
    return violations


def _validate_worker_result(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    platform: str,
    optimization: str,
    repetition: int,
    smoke_layers: int | None,
) -> tuple[str, ...]:
    """Return contract violations for one isolated worker result."""

    expected_scope = "full" if smoke_layers is None else "smoke"
    checks = (
        ("schema", payload.get("schema"), WORKER_SCHEMA),
        ("case_id", _nested(payload, "case", "case_id"), case_id),
        (
            "platform",
            _nested(payload, "execution", "requested_platform"),
            platform,
        ),
        (
            "optimization",
            _nested(payload, "execution", "optimization_mode"),
            optimization,
        ),
        (
            "repetition",
            _nested(payload, "execution", "repetition"),
            repetition,
        ),
        ("scope", _nested(payload, "scope", "kind"), expected_scope),
        (
            "smoke_layers",
            _nested(payload, "scope", "smoke_layers"),
            smoke_layers,
        ),
    )
    violations = [
        f"{name}: {actual!r} != {expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if payload.get("status") == "pass":
        validation = payload.get("validation")
        timing = payload.get("timing")
        if not isinstance(validation, Mapping):
            violations.append("pass result has no validation mapping")
        else:
            if validation.get("all_layers_converged") is not True:
                violations.append("pass result did not accept every layer")
            output_count = validation.get("output_layer_count")
            if (
                isinstance(output_count, bool)
                or not isinstance(output_count, int)
                or output_count <= 0
            ):
                violations.append("pass result has invalid output_layer_count")
            elif smoke_layers is None:
                expected_count = CASES_BY_ID[case_id].expected_output_layer_count(
                    smoke_layers
                )
                if output_count != expected_count:
                    violations.append(
                        "full result output_layer_count: "
                        f"{output_count!r} != {expected_count!r}"
                    )
            else:
                expected_count = CASES_BY_ID[case_id].expected_output_layer_count(
                    smoke_layers
                )
                if output_count != expected_count:
                    violations.append(
                        "smoke result output_layer_count: "
                        f"{output_count!r} != {expected_count!r}"
                    )
        if not isinstance(timing, Mapping):
            violations.append("pass result has no timing mapping")
        else:
            metric_paths = (
                ("workload_wall_seconds",),
                ("setup_phase_wall_seconds",),
                ("solver_phase_wall_seconds",),
                ("unphased_workload_wall_seconds",),
                ("pdipm", "compilation_seconds"),
                ("pdipm", "execution_seconds"),
                ("pdipm", "diagnostic_seconds"),
                ("pdipm", "diagnostic_compilation_seconds"),
                ("pdipm", "diagnostic_execution_seconds"),
                ("pdipm", "wall_seconds"),
                ("zero_barrier", "host_wall_seconds"),
            )
            for path in metric_paths:
                value = _nested(timing, *path)
                if (
                    not _is_finite_number(value)
                    or float(value) < 0.0
                ):
                    violations.append(
                        "pass result has invalid timing metric "
                        + ".".join(path)
                    )
            for path in (
                ("pdipm", "call_count"),
                ("pdipm", "bucket_count"),
                ("zero_barrier", "call_count"),
            ):
                value = _nested(timing, *path)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    violations.append(
                        "pass result has invalid timing count "
                        + ".".join(path)
                    )
            solver_budget = timing.get("solver_budget")
            if not isinstance(solver_budget, Mapping):
                violations.append("pass result has no solver_budget mapping")
            else:
                violations.extend(
                    _budget_violations(solver_budget, label="solver_budget")
                )
            phase_budgets = timing.get("phase_budgets")
            if not isinstance(phase_budgets, (list, tuple)) or not phase_budgets:
                violations.append("pass result has no phase_budgets")
            else:
                for index, budget in enumerate(phase_budgets):
                    if not isinstance(budget, Mapping):
                        violations.append(
                            f"pass result has invalid phase_budget[{index}]"
                        )
                        continue
                    label = f"phase_budget[{index}]"
                    violations.extend(_budget_violations(budget, label=label))
                    if not isinstance(budget.get("name"), str):
                        violations.append(f"pass result has invalid {label} name")
                    if budget.get("category") not in ("setup", "solver"):
                        violations.append(
                            f"pass result has invalid {label} category"
                        )
                    invocation_count = budget.get("invocation_count")
                    if (
                        isinstance(invocation_count, bool)
                        or not isinstance(invocation_count, int)
                        or invocation_count <= 0
                    ):
                        violations.append(
                            f"pass result has invalid {label} invocation_count"
                        )
                valid_phase_budgets = [
                    budget
                    for budget in phase_budgets
                    if isinstance(budget, Mapping)
                ]
                actual_phases = {
                    (budget.get("name"), budget.get("category"))
                    for budget in valid_phase_budgets
                }
                expected_phases = set(CASES_BY_ID[case_id].expected_phases)
                if actual_phases != expected_phases:
                    violations.append(
                        "pass result phase set differs from manifest"
                    )
                solver_phase_budgets = [
                    budget
                    for budget in valid_phase_budgets
                    if budget.get("category") == "solver"
                ]
                if isinstance(solver_budget, Mapping):
                    for name in (
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
                    ):
                        phase_total = sum(
                            float(budget.get(name, 0.0))
                            for budget in solver_phase_budgets
                            if _is_finite_number(budget.get(name))
                        )
                        expected_total = solver_budget.get(name)
                        if _is_finite_number(expected_total) and not math.isclose(
                            phase_total,
                            float(expected_total),
                            abs_tol=1.0e-9 * max(abs(phase_total), 1.0),
                        ):
                            violations.append(
                                "pass result solver phase total differs from "
                                f"solver_budget.{name}"
                            )
                for category, timing_name in (
                    ("setup", "setup_phase_wall_seconds"),
                    ("solver", "solver_phase_wall_seconds"),
                ):
                    phase_total = sum(
                        float(budget["wall_seconds"])
                        for budget in valid_phase_budgets
                        if budget.get("category") == category
                        and _is_finite_number(budget.get("wall_seconds"))
                    )
                    expected_total = timing.get(timing_name)
                    if _is_finite_number(expected_total) and not math.isclose(
                        phase_total,
                        float(expected_total),
                        abs_tol=1.0e-9 * max(abs(phase_total), 1.0),
                    ):
                        violations.append(
                            f"pass result has inconsistent {category} phase total"
                        )
                global_component_paths = (
                    ("pdipm_call_count", ("pdipm", "call_count")),
                    ("pdipm_bucket_count", ("pdipm", "bucket_count")),
                    ("pdipm_wall_seconds", ("pdipm", "wall_seconds")),
                    (
                        "pdipm_compilation_seconds",
                        ("pdipm", "compilation_seconds"),
                    ),
                    (
                        "pdipm_execution_seconds",
                        ("pdipm", "execution_seconds"),
                    ),
                    (
                        "pdipm_diagnostic_seconds",
                        ("pdipm", "diagnostic_seconds"),
                    ),
                    (
                        "pdipm_diagnostic_compilation_seconds",
                        ("pdipm", "diagnostic_compilation_seconds"),
                    ),
                    (
                        "pdipm_diagnostic_execution_seconds",
                        ("pdipm", "diagnostic_execution_seconds"),
                    ),
                    (
                        "zero_barrier_call_count",
                        ("zero_barrier", "call_count"),
                    ),
                    (
                        "zero_barrier_host_wall_seconds",
                        ("zero_barrier", "host_wall_seconds"),
                    ),
                    (
                        "zero_barrier_function_evaluations",
                        ("zero_barrier", "function_evaluations"),
                    ),
                )
                for budget_name, path in global_component_paths:
                    phase_total = sum(
                        float(budget.get(budget_name, 0.0))
                        for budget in valid_phase_budgets
                        if _is_finite_number(budget.get(budget_name))
                    )
                    global_total = _nested(timing, *path)
                    if _is_finite_number(global_total) and not math.isclose(
                        phase_total,
                        float(global_total),
                        abs_tol=1.0e-9 * max(abs(phase_total), 1.0),
                    ):
                        violations.append(
                            "pass result phase total differs from "
                            + ".".join(path)
                        )
            if isinstance(solver_budget, Mapping) and _is_finite_number(
                solver_budget.get("wall_seconds")
            ) and _is_finite_number(timing.get("solver_phase_wall_seconds")):
                if not math.isclose(
                    float(solver_budget["wall_seconds"]),
                    float(timing["solver_phase_wall_seconds"]),
                    abs_tol=1.0e-9
                    * max(float(timing["solver_phase_wall_seconds"]), 1.0),
                ):
                    violations.append(
                        "pass result solver_budget wall differs from solver phase"
                    )
            if isinstance(solver_budget, Mapping):
                other_solver = timing.get(
                    "other_solver_and_orchestration_wall_seconds"
                )
                internal = solver_budget.get(
                    "pdipm_internal_orchestration_wall_seconds"
                )
                outside = solver_budget.get(
                    "outside_pdipm_and_zero_barrier_wall_seconds"
                )
                if all(
                    _is_finite_number(value)
                    for value in (other_solver, internal, outside)
                ) and not math.isclose(
                    float(other_solver),
                    float(internal) + float(outside),
                    abs_tol=1.0e-9 * max(abs(float(other_solver)), 1.0),
                ):
                    violations.append(
                        "pass result has inconsistent compatibility residual"
                    )
            workload_components = (
                timing.get("setup_phase_wall_seconds"),
                timing.get("solver_phase_wall_seconds"),
                timing.get("unphased_workload_wall_seconds"),
            )
            workload_wall = timing.get("workload_wall_seconds")
            if _is_finite_number(workload_wall) and all(
                _is_finite_number(value) for value in workload_components
            ):
                component_sum = sum(float(value) for value in workload_components)
                if not math.isclose(
                    float(workload_wall),
                    component_sum,
                    abs_tol=1.0e-9 * max(float(workload_wall), 1.0),
                ):
                    violations.append(
                        "pass result has inconsistent workload phase total"
                    )
            if timing.get("timing_attribution_consistent") is not True:
                violations.append("pass result has inconsistent timing attribution")
        if not isinstance(payload.get("environment"), Mapping):
            violations.append("pass result has no environment mapping")
    return tuple(violations)


def _nested(mapping: Mapping[str, Any], *names: str) -> Any:
    value: Any = mapping
    for name in names:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def summary_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return one compact tabular record from a worker result."""

    timing = payload.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    return {
        "case_id": _nested(payload, "case", "case_id"),
        "platform": _nested(payload, "execution", "requested_platform"),
        "optimization": _nested(
            payload, "execution", "optimization_mode"
        ),
        "repetition": _nested(payload, "execution", "repetition"),
        "scope": _nested(payload, "scope", "kind"),
        "status": payload.get("status"),
        "worker_return_code": payload.get("worker_return_code"),
        "output_layer_count": _nested(
            payload, "validation", "output_layer_count"
        ),
        "workload_wall_seconds": timing.get("workload_wall_seconds"),
        "solver_phase_wall_seconds": timing.get("solver_phase_wall_seconds"),
        "pdipm_compile_seconds": _nested(
            timing, "solver_budget", "pdipm_compilation_seconds"
        ),
        "pdipm_execute_seconds": _nested(
            timing, "solver_budget", "pdipm_execution_seconds"
        ),
        "pdipm_diagnostic_seconds": _nested(
            timing, "solver_budget", "pdipm_diagnostic_seconds"
        ),
        "pdipm_first_shape_compilation_seconds": _nested(
            timing, "pdipm", "first_shape_compilation_seconds"
        ),
        "pdipm_repeated_shape_compilation_seconds": _nested(
            timing, "pdipm", "repeated_shape_compilation_seconds"
        ),
        "pdipm_executable_shape_count": _nested(
            timing, "pdipm", "executable_shape_count"
        ),
        "zero_barrier_host_wall_seconds": _nested(
            timing, "solver_budget", "zero_barrier_host_wall_seconds"
        ),
        "zero_barrier_call_count": _nested(
            timing, "solver_budget", "zero_barrier_call_count"
        ),
        "zero_barrier_function_evaluations": _nested(
            timing, "solver_budget", "zero_barrier_function_evaluations"
        ),
        "setup_phase_wall_seconds": timing.get("setup_phase_wall_seconds"),
        "unphased_workload_wall_seconds": timing.get(
            "unphased_workload_wall_seconds"
        ),
        "pdipm_call_count": _nested(
            timing, "solver_budget", "pdipm_call_count"
        ),
        "pdipm_bucket_count": _nested(
            timing, "solver_budget", "pdipm_bucket_count"
        ),
        "pdipm_wall_seconds": _nested(
            timing, "solver_budget", "pdipm_wall_seconds"
        ),
        "pdipm_diagnostic_compilation_seconds": _nested(
            timing,
            "solver_budget",
            "pdipm_diagnostic_compilation_seconds",
        ),
        "pdipm_diagnostic_execution_seconds": _nested(
            timing,
            "solver_budget",
            "pdipm_diagnostic_execution_seconds",
        ),
        "pdipm_internal_orchestration_wall_seconds": _nested(
            timing,
            "solver_budget",
            "pdipm_internal_orchestration_wall_seconds",
        ),
        "outside_pdipm_and_zero_barrier_wall_seconds": _nested(
            timing,
            "solver_budget",
            "outside_pdipm_and_zero_barrier_wall_seconds",
        ),
        "other_solver_and_orchestration_wall_seconds": timing.get(
            "other_solver_and_orchestration_wall_seconds"
        ),
        "timing_attribution_consistent": timing.get(
            "timing_attribution_consistent"
        ),
        "timing_attribution_delta_seconds": _nested(
            timing, "solver_budget", "attribution_delta_seconds"
        ),
        "maximum_resident_set_size_kb": _nested(
            payload, "resources", "maximum_resident_set_size_kb"
        ),
        "result_file": payload.get("result_file"),
        "log_file": payload.get("log_file"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_row({}).keys())
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def phase_budget_rows(
    payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return long-form phase budgets for every worker result."""

    rows = []
    for payload in payloads:
        timing = payload.get("timing", {})
        if not isinstance(timing, Mapping):
            continue
        budgets = timing.get("phase_budgets", ())
        if not isinstance(budgets, (list, tuple)):
            continue
        for budget in budgets:
            if not isinstance(budget, Mapping):
                continue
            rows.append(
                {
                    "case_id": _nested(payload, "case", "case_id"),
                    "platform": _nested(
                        payload, "execution", "requested_platform"
                    ),
                    "optimization": _nested(
                        payload, "execution", "optimization_mode"
                    ),
                    "repetition": _nested(
                        payload, "execution", "repetition"
                    ),
                    "scope": _nested(payload, "scope", "kind"),
                    "status": payload.get("status"),
                    "phase_name": budget.get("name"),
                    "phase_category": budget.get("category"),
                    "phase_invocation_count": budget.get("invocation_count"),
                    "phase_wall_seconds": budget.get("wall_seconds"),
                    "pdipm_call_count": budget.get("pdipm_call_count"),
                    "pdipm_bucket_count": budget.get("pdipm_bucket_count"),
                    "pdipm_wall_seconds": budget.get("pdipm_wall_seconds"),
                    "pdipm_compilation_seconds": budget.get(
                        "pdipm_compilation_seconds"
                    ),
                    "pdipm_execution_seconds": budget.get(
                        "pdipm_execution_seconds"
                    ),
                    "pdipm_diagnostic_seconds": budget.get(
                        "pdipm_diagnostic_seconds"
                    ),
                    "pdipm_diagnostic_compilation_seconds": budget.get(
                        "pdipm_diagnostic_compilation_seconds"
                    ),
                    "pdipm_diagnostic_execution_seconds": budget.get(
                        "pdipm_diagnostic_execution_seconds"
                    ),
                    "pdipm_internal_orchestration_wall_seconds": budget.get(
                        "pdipm_internal_orchestration_wall_seconds"
                    ),
                    "zero_barrier_call_count": budget.get(
                        "zero_barrier_call_count"
                    ),
                    "zero_barrier_host_wall_seconds": budget.get(
                        "zero_barrier_host_wall_seconds"
                    ),
                    "zero_barrier_function_evaluations": budget.get(
                        "zero_barrier_function_evaluations"
                    ),
                    "outside_pdipm_and_zero_barrier_wall_seconds": budget.get(
                        "outside_pdipm_and_zero_barrier_wall_seconds"
                    ),
                    "attribution_delta_seconds": budget.get(
                        "attribution_delta_seconds"
                    ),
                    "attribution_consistent": budget.get(
                        "attribution_consistent"
                    ),
                }
            )
    return rows


def _write_phase_budget_csv(
    path: Path, payloads: Sequence[Mapping[str, Any]]
) -> None:
    rows = phase_budget_rows(payloads)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PHASE_BUDGET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    overview_fields = (
        "case_id",
        "platform",
        "optimization",
        "repetition",
        "scope",
        "status",
        "output_layer_count",
        "workload_wall_seconds",
        "setup_phase_wall_seconds",
        "solver_phase_wall_seconds",
        "unphased_workload_wall_seconds",
        "pdipm_call_count",
        "pdipm_bucket_count",
        "zero_barrier_call_count",
        "zero_barrier_function_evaluations",
    )
    budget_fields = (
        "case_id",
        "platform",
        "optimization",
        "repetition",
        "pdipm_compile_seconds",
        "pdipm_execute_seconds",
        "pdipm_diagnostic_compilation_seconds",
        "pdipm_diagnostic_execution_seconds",
        "pdipm_internal_orchestration_wall_seconds",
        "zero_barrier_host_wall_seconds",
        "outside_pdipm_and_zero_barrier_wall_seconds",
        "timing_attribution_consistent",
    )

    def table(fields: Sequence[str]) -> list[str]:
        table_lines = [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
        ]
        for row in rows:
            table_lines.append(
                "| "
                + " | ".join(
                    "" if row.get(field) is None else str(row[field])
                    for field in fields
                )
                + " |"
            )
        return table_lines

    lines = [
        "# Documented example benchmark summary",
        "",
        "## Workload overview",
        "",
        *table(overview_fields),
        "",
        "## Solver wall-time budget",
        "",
        *table(budget_fields),
        "",
        "Per-phase budgets are in `phase_budgets.csv` and each worker JSON.",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES_BY_ID),
        help="Case to run; repeat this option. The default is every case.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        choices=("cpu", "gpu"),
        help="Platform to run; repeat this option. The default is CPU and GPU.",
    )
    parser.add_argument(
        "--optimization",
        action="append",
        choices=OPTIMIZATION_MODES,
        help="Optimization mode; repeat this option. The default is both.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--smoke-layers", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Default: results/documented_example_benchmarks/TIMESTAMP",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed or unavailable worker.",
    )
    return parser


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def benchmark_jobs(
    cases: Sequence[str],
    platforms: Sequence[str],
    optimizations: Sequence[str],
    repeat: int,
) -> tuple[tuple[str, str, str, int], ...]:
    """Return the deterministic Cartesian benchmark matrix."""

    return tuple(
        (case_id, platform, optimization, repetition)
        for case_id in cases
        for platform in platforms
        for optimization in optimizations
        for repetition in range(1, repeat + 1)
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.smoke_layers is not None and args.smoke_layers <= 0:
        raise SystemExit("--smoke-layers must be positive")

    cases = _ordered_unique(args.case or (case.case_id for case in CASES))
    platforms = _ordered_unique(args.platform or ("cpu", "gpu"))
    optimizations = _ordered_unique(
        args.optimization or OPTIMIZATION_MODES
    )
    output_directory = args.output_directory or (
        Path("results") / "documented_example_benchmarks" / _slug_timestamp()
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    suite: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": _timestamp(),
        "selection": {
            "cases": cases,
            "platforms": platforms,
            "optimizations": optimizations,
            "repeat": args.repeat,
            "smoke_layers": args.smoke_layers,
        },
        "results": [],
    }
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, suite)

    repository_root = Path(__file__).resolve().parents[2]
    child_environment = os.environ.copy()
    import_paths = (
        str(repository_root / "src"),
        str(repository_root),
    )
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    if inherited_pythonpath:
        import_paths += (inherited_pythonpath,)
    child_environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    result_payloads = []
    stop = False
    jobs = benchmark_jobs(cases, platforms, optimizations, args.repeat)
    for case_id, platform, optimization, repetition in jobs:
        stem = (
            f"{case_id}__{platform}__{optimization}"
            f"__repeat{repetition}"
        )
        result_path = output_directory / f"{stem}.json"
        log_path = output_directory / f"{stem}.log"
        if result_path.exists():
            result_path.unlink()
        command = (
            sys.executable,
            "-m",
            "benchmarks.documented_examples.worker",
            "--case",
            case_id,
            "--platform",
            platform,
            "--optimization",
            optimization,
            "--output",
            str(result_path.resolve()),
            "--repetition",
            str(repetition),
        )
        if args.smoke_layers is not None:
            command += ("--smoke-layers", str(args.smoke_layers))
        print(f"RUN {' '.join(command)}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                env=child_environment,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        payload = _read_result(result_path)
        if not isinstance(payload.get("case"), Mapping):
            payload["case"] = {"case_id": case_id}
        if not isinstance(payload.get("execution"), Mapping):
            payload["execution"] = {
                "requested_platform": platform,
                "optimization_mode": optimization,
                "repetition": repetition,
            }
        if not isinstance(payload.get("scope"), Mapping):
            payload["scope"] = {
                "kind": "full" if args.smoke_layers is None else "smoke",
                "smoke_layers": args.smoke_layers,
            }
        violations = _validate_worker_result(
            payload,
            case_id=case_id,
            platform=platform,
            optimization=optimization,
            repetition=repetition,
            smoke_layers=args.smoke_layers,
        )
        if violations:
            payload["reported_status"] = payload.get("status")
            payload["status"] = "invalid_result"
            payload["contract_violations"] = violations
        if completed.returncode != 0 and payload.get("status") == "pass":
            payload["reported_status"] = "pass"
            payload["status"] = "worker_error"
            payload["contract_violations"] = (
                f"worker return code was {completed.returncode}",
            )
        payload["worker_return_code"] = completed.returncode
        payload["result_file"] = str(result_path)
        payload["log_file"] = str(log_path)
        result_payloads.append(payload)
        suite["results"] = result_payloads
        _write_json(summary_path, suite)
        print(
            f"{payload.get('status', 'unknown').upper()} "
            f"{case_id} {platform} {optimization}: {log_path}",
            flush=True,
        )
        if (
            completed.returncode != 0 or payload.get("status") != "pass"
        ) and args.fail_fast:
            stop = True
        if stop:
            break

    rows = [summary_row(payload) for payload in result_payloads]
    _write_csv(output_directory / "summary.csv", rows)
    _write_phase_budget_csv(
        output_directory / "phase_budgets.csv", result_payloads
    )
    _write_markdown(output_directory / "summary.md", rows)
    passed_results = [
        payload.get("status") == "pass"
        and payload.get("worker_return_code") == 0
        for payload in result_payloads
    ]
    suite.update(
        {
            "status": (
                "pass"
                if passed_results and all(passed_results)
                else "incomplete"
            ),
            "finished_at_utc": _timestamp(),
            "result_count": len(result_payloads),
            "passed_result_count": sum(passed_results),
            "results": result_payloads,
        }
    )
    _write_json(summary_path, suite)
    print(f"SUMMARY {summary_path}", flush=True)
    return 0 if suite["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
