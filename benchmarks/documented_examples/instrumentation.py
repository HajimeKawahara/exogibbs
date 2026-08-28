"""Non-invasive timing hooks for documented condensate workloads."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import math
import time
from typing import Any, Iterator, Mapping


SOLVER_BUDGET_SCHEMA = "exogibbs_documented_example_solver_budget_v1"
_ATTRIBUTION_ABSOLUTE_TOLERANCE_SECONDS = 1.0e-9


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _float(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name, 0.0)
    return float(value) if value is not None else 0.0


def _diagnostic_timings(result: Mapping[str, Any]) -> tuple[float, float]:
    """Return additive diagnostic compile and execution timings."""

    total = _float(result, "diagnostic_seconds")
    compilation = result.get("diagnostic_compilation_seconds")
    execution = result.get("diagnostic_execution_seconds")
    if compilation is None and execution is None:
        return 0.0, total
    if compilation is None:
        execution_seconds = float(execution)
        return total - execution_seconds, execution_seconds
    if execution is None:
        compilation_seconds = float(compilation)
        return compilation_seconds, total - compilation_seconds
    return float(compilation), float(execution)


def _wall_budget(
    *,
    wall_seconds: float,
    pdipm_calls: list[dict[str, Any]],
    zero_barrier_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Partition one wall interval without adding synchronization points."""

    pdipm_wall = sum(float(call.get("wall_seconds", 0.0)) for call in pdipm_calls)
    compilation = sum(
        float(call.get("compilation_seconds", 0.0)) for call in pdipm_calls
    )
    execution = sum(
        float(call.get("execution_seconds", 0.0)) for call in pdipm_calls
    )
    diagnostic = sum(
        float(call.get("diagnostic_seconds", 0.0)) for call in pdipm_calls
    )
    diagnostic_compilation = sum(
        float(call.get("diagnostic_compilation_seconds", 0.0))
        for call in pdipm_calls
    )
    diagnostic_execution = sum(
        float(call.get("diagnostic_execution_seconds", 0.0))
        for call in pdipm_calls
    )
    pdipm_internal = pdipm_wall - compilation - execution - diagnostic
    zero_barrier_wall = sum(
        float(call.get("host_wall_seconds", 0.0))
        for call in zero_barrier_calls
    )
    zero_barrier_function_evaluations = sum(
        int(call["function_evaluations"])
        for call in zero_barrier_calls
        if call.get("function_evaluations") is not None
    )
    outside_instrumented_calls = wall_seconds - pdipm_wall - zero_barrier_wall
    leaf_component_sum = (
        compilation
        + execution
        + diagnostic_compilation
        + diagnostic_execution
        + pdipm_internal
        + zero_barrier_wall
        + outside_instrumented_calls
    )
    attribution_delta = wall_seconds - leaf_component_sum
    scale = max(abs(wall_seconds), abs(leaf_component_sum), 1.0)
    tolerance = _ATTRIBUTION_ABSOLUTE_TOLERANCE_SECONDS * scale
    values = (
        wall_seconds,
        pdipm_wall,
        compilation,
        execution,
        diagnostic,
        diagnostic_compilation,
        diagnostic_execution,
        pdipm_internal,
        zero_barrier_wall,
        outside_instrumented_calls,
        attribution_delta,
    )
    consistent = (
        all(math.isfinite(value) for value in values)
        and pdipm_internal >= -tolerance
        and outside_instrumented_calls >= -tolerance
        and abs(
            diagnostic - diagnostic_compilation - diagnostic_execution
        )
        <= tolerance
        and abs(attribution_delta) <= tolerance
    )
    return {
        "schema": SOLVER_BUDGET_SCHEMA,
        "wall_seconds": wall_seconds,
        "pdipm_call_count": len(pdipm_calls),
        "pdipm_bucket_count": sum(
            len(call.get("buckets", ())) for call in pdipm_calls
        ),
        "pdipm_wall_seconds": pdipm_wall,
        "pdipm_compilation_seconds": compilation,
        "pdipm_execution_seconds": execution,
        "pdipm_diagnostic_seconds": diagnostic,
        "pdipm_diagnostic_compilation_seconds": diagnostic_compilation,
        "pdipm_diagnostic_execution_seconds": diagnostic_execution,
        "pdipm_internal_orchestration_wall_seconds": pdipm_internal,
        "zero_barrier_call_count": len(zero_barrier_calls),
        "zero_barrier_host_wall_seconds": zero_barrier_wall,
        "zero_barrier_function_evaluations": (
            zero_barrier_function_evaluations
        ),
        "outside_pdipm_and_zero_barrier_wall_seconds": (
            outside_instrumented_calls
        ),
        "attribution_delta_seconds": attribution_delta,
        "attribution_consistent": consistent,
    }


class TimingCollector:
    """Collect solver-component timings without changing production code."""

    def __init__(self) -> None:
        self.phases: list[dict[str, Any]] = []
        self.pdipm_calls: list[dict[str, Any]] = []
        self.zero_barrier_calls: list[dict[str, Any]] = []
        self._phase_stack: list[str] = []
        self._phase_categories: list[str] = []
        self._original_pdipm: Any = None
        self._original_zero_barrier: Any = None
        self._batch_module: Any = None
        self._zero_barrier_module: Any = None

    @property
    def current_phase(self) -> str | None:
        """Return the innermost active benchmark phase."""

        return self._phase_stack[-1] if self._phase_stack else None

    @property
    def current_category(self) -> str | None:
        """Return the category of the innermost active phase."""

        return self._phase_categories[-1] if self._phase_categories else None

    def __enter__(self) -> "TimingCollector":
        from exogibbs.equilibrium.condensate.fixed_support import batch
        from exogibbs.equilibrium.condensate.fixed_support import zero_barrier

        self._batch_module = batch
        self._zero_barrier_module = zero_barrier
        self._original_pdipm = batch.run_fixed_support_profile
        self._original_zero_barrier = (
            zero_barrier.polish_zero_barrier_active_support
        )
        batch.run_fixed_support_profile = self._timed_pdipm
        zero_barrier.polish_zero_barrier_active_support = (
            self._timed_zero_barrier
        )
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._batch_module is not None:
            self._batch_module.run_fixed_support_profile = self._original_pdipm
        if self._zero_barrier_module is not None:
            self._zero_barrier_module.polish_zero_barrier_active_support = (
                self._original_zero_barrier
            )

    @contextmanager
    def phase(self, name: str, *, category: str) -> Iterator[None]:
        """Measure one named setup or solver phase."""

        self._phase_stack.append(name)
        self._phase_categories.append(category)
        started = time.perf_counter()
        status = "pass"
        error_message = None
        try:
            yield
        except Exception as error:
            status = "error"
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            wall_seconds = time.perf_counter() - started
            self._phase_stack.pop()
            self._phase_categories.pop()
            self.phases.append(
                {
                    "name": name,
                    "category": category,
                    "wall_seconds": wall_seconds,
                    "status": status,
                    "error": error_message,
                }
            )

    def _timed_pdipm(self, *args: Any, **kwargs: Any) -> Any:
        phase = self.current_phase
        category = self.current_category
        started = time.perf_counter()
        try:
            result = self._original_pdipm(*args, **kwargs)
        except Exception as error:
            self.pdipm_calls.append(
                {
                    "phase": phase,
                    "phase_category": category,
                    "wall_seconds": time.perf_counter() - started,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "buckets": (),
                }
            )
            raise

        wall_seconds = time.perf_counter() - started
        formula_matrix = kwargs.get("formula_matrix")
        if formula_matrix is None and len(args) >= 2:
            formula_matrix = args[1]
        formula_shape = _shape(formula_matrix)
        include_diagnostics = bool(
            kwargs.get("include_terminal_diagnostics", True)
        )
        diagnostic_compilation, diagnostic_execution = _diagnostic_timings(
            result
        )
        config = kwargs.get("config")
        config_fingerprint = hashlib.sha256(
            repr(config).encode("utf-8")
        ).hexdigest()[:12]
        bucket_records = []
        for bucket_index, report in enumerate(result.get("bucket_reports", ())):
            support_indices = tuple(
                int(item) for item in report.get("support_indices", ())
            )
            support_indices_by_layer = tuple(
                tuple(int(item) for item in support)
                for support in report.get("support_indices_by_layer", ())
            )
            layer_indices = tuple(
                int(item) for item in report.get("layer_indices", ())
            )
            source_layer_indices = tuple(
                int(item)
                for item in report.get(
                    "source_layer_indices",
                    layer_indices,
                )
            )
            support_capacity = int(
                report.get("support_capacity", len(support_indices))
            )
            batch_capacity = int(
                report.get("batch_capacity", len(layer_indices))
            )
            element_count = formula_shape[0] if formula_shape else None
            gas_count = formula_shape[1] if formula_shape else None
            signature = (
                f"backend={result.get('backend', 'unknown')},"
                f"dtype={getattr(formula_matrix, 'dtype', None)},"
                f"config={config_fingerprint},"
                f"elements={element_count},gas={gas_count},"
                f"support={support_capacity},batch={batch_capacity},"
                f"diagnostics={int(include_diagnostics)}"
            )
            bucket_diagnostic_compilation, bucket_diagnostic_execution = (
                _diagnostic_timings(report)
            )
            bucket_records.append(
                {
                    "bucket_index": bucket_index,
                    "signature": signature,
                    "support_count": support_capacity,
                    "batch_size": batch_capacity,
                    "valid_batch_size": int(
                        report.get("valid_batch_size", len(layer_indices))
                    ),
                    "support_indices": support_indices,
                    "support_indices_by_layer": support_indices_by_layer,
                    "layer_indices": layer_indices,
                    "source_layer_indices": source_layer_indices,
                    "compilation_seconds": _float(
                        report, "compilation_seconds"
                    ),
                    "execution_seconds": _float(report, "execution_seconds"),
                    "diagnostic_compilation_seconds": (
                        bucket_diagnostic_compilation
                    ),
                    "diagnostic_execution_seconds": (
                        bucket_diagnostic_execution
                    ),
                }
            )
        self.pdipm_calls.append(
            {
                "phase": phase,
                "phase_category": category,
                "wall_seconds": wall_seconds,
                "status": "pass",
                "error": None,
                "compilation_seconds": _float(result, "compilation_seconds"),
                "execution_seconds": _float(result, "execution_seconds"),
                "diagnostic_seconds": _float(result, "diagnostic_seconds"),
                "diagnostic_compilation_seconds": diagnostic_compilation,
                "diagnostic_execution_seconds": diagnostic_execution,
                "backend": str(result.get("backend", "unknown")),
                "buckets": tuple(bucket_records),
            }
        )
        return result

    def _timed_zero_barrier(self, *args: Any, **kwargs: Any) -> Any:
        phase = self.current_phase
        category = self.current_category
        started = time.perf_counter()
        try:
            result = self._original_zero_barrier(*args, **kwargs)
        except Exception as error:
            self.zero_barrier_calls.append(
                {
                    "phase": phase,
                    "phase_category": category,
                    "host_wall_seconds": (
                        time.perf_counter() - started
                    ),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "function_evaluations": None,
                }
            )
            raise

        report = result.report if isinstance(result.report, Mapping) else {}
        closure = report.get("exact_active_set_closure", {})
        if not isinstance(closure, Mapping):
            closure = {}
        function_evaluations = closure.get(
            "cumulative_function_evaluations",
            report.get("function_evaluations"),
        )
        portfolio = report.get(
            "normalized_gas_reduced_initializer_portfolio", {}
        )
        if not isinstance(portfolio, Mapping):
            portfolio = {}
        closure_rounds = closure.get("rounds", ())
        if not isinstance(closure_rounds, (tuple, list)):
            closure_rounds = ()
        selected_initializers = tuple(
            round_report.get("selected_normalized_initializer")
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        regularized_attempt_count = sum(
            int(
                round_report.get(
                    "regularized_normalized_initializer_attempt_count",
                    bool(
                        round_report.get(
                            "regularized_normalized_initializer_attempted",
                            False,
                        )
                    ),
                )
            )
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        regularized_function_evaluations = sum(
            int(
                round_report.get(
                    "regularized_normalized_initializer_function_evaluations",
                    0,
                )
            )
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        unregularized_attempt_count = sum(
            int(
                round_report.get(
                    "unregularized_normalized_initializer_attempt_count",
                    0,
                )
            )
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        unregularized_function_evaluations = sum(
            int(
                round_report.get(
                    "unregularized_normalized_initializer_function_evaluations",
                    0,
                )
            )
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        raw_retry_count = sum(
            int(
                round_report.get(
                    "raw_normalized_initializer_retry_count",
                    bool(
                        round_report.get(
                            "raw_normalized_initializer_retry_attempted",
                            False,
                        )
                    ),
                )
            )
            for round_report in closure_rounds
            if isinstance(round_report, Mapping)
        )
        if not closure_rounds:
            selected = portfolio.get("selected_initializer")
            selected_initializers = (() if selected is None else (selected,))
            regularized_attempt_count = int(
                bool(portfolio.get("regularized_attempted", False))
            )
            regularized_function_evaluations = sum(
                int(attempt.get("function_evaluations", 0))
                for attempt in portfolio.get("attempts", ())
                if attempt.get("initializer") == "capacity_regularized"
            )
            unregularized_attempt_count = int(
                bool(portfolio.get("unregularized_attempted", False))
            )
            unregularized_function_evaluations = sum(
                int(attempt.get("function_evaluations", 0))
                for attempt in portfolio.get("attempts", ())
                if attempt.get("initializer") == "unregularized"
            )
            raw_retry_count = int(
                bool(portfolio.get("raw_retry_attempted", False))
            )
        initial_support = tuple(
            int(index) for index in kwargs.get("support_indices", ())
        )
        self.zero_barrier_calls.append(
            {
                "phase": phase,
                "phase_category": category,
                "host_wall_seconds": (
                    time.perf_counter() - started
                ),
                "execution_location": "host_cpu",
                "status": "pass",
                "error": None,
                "accepted": bool(result.accepted),
                "initial_support_count": len(initial_support),
                "initial_support_indices": initial_support,
                "final_support_count": len(result.support_indices),
                "final_support_indices": tuple(
                    int(index) for index in result.support_indices
                ),
                "function_evaluations": (
                    None
                    if function_evaluations is None
                    else int(function_evaluations)
                ),
                "active_set_round_count": (
                    None
                    if closure.get("round_count") is None
                    else int(closure["round_count"])
                ),
                "termination_reason": closure.get("termination_reason"),
                "selected_normalized_initializers": selected_initializers,
                "regularized_normalized_initializer_attempt_count": (
                    regularized_attempt_count
                ),
                "regularized_normalized_initializer_function_evaluations": (
                    regularized_function_evaluations
                ),
                "unregularized_normalized_initializer_attempt_count": (
                    unregularized_attempt_count
                ),
                "unregularized_normalized_initializer_function_evaluations": (
                    unregularized_function_evaluations
                ),
                "raw_normalized_initializer_retry_count": raw_retry_count,
            }
        )
        return result

    def summary(self, *, workload_wall_seconds: float) -> dict[str, Any]:
        """Return the stable timing summary and detailed call records."""

        solver_wall_seconds = sum(
            float(phase["wall_seconds"])
            for phase in self.phases
            if phase["category"] == "solver"
        )
        setup_wall_seconds = sum(
            float(phase["wall_seconds"])
            for phase in self.phases
            if phase["category"] == "setup"
        )
        pdipm_compilation = sum(
            float(call.get("compilation_seconds", 0.0))
            for call in self.pdipm_calls
        )
        pdipm_execution = sum(
            float(call.get("execution_seconds", 0.0))
            for call in self.pdipm_calls
        )
        pdipm_diagnostics = sum(
            float(call.get("diagnostic_seconds", 0.0))
            for call in self.pdipm_calls
        )
        pdipm_diagnostic_compilation = sum(
            float(call.get("diagnostic_compilation_seconds", 0.0))
            for call in self.pdipm_calls
        )
        pdipm_diagnostic_execution = sum(
            float(call.get("diagnostic_execution_seconds", 0.0))
            for call in self.pdipm_calls
        )
        pdipm_wall = sum(
            float(call.get("wall_seconds", 0.0))
            for call in self.pdipm_calls
        )
        zero_barrier_wall = sum(
            float(call["host_wall_seconds"])
            for call in self.zero_barrier_calls
        )

        shape_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in self.pdipm_calls:
            for bucket in call.get("buckets", ()):
                shape_groups[str(bucket["signature"])].append(bucket)
        shape_compilation = []
        first_shape_compilation = 0.0
        repeated_shape_compilation = 0.0
        for signature, buckets in shape_groups.items():
            first = float(buckets[0]["compilation_seconds"])
            repeated = sum(
                float(bucket["compilation_seconds"])
                for bucket in buckets[1:]
            )
            first_shape_compilation += first
            repeated_shape_compilation += repeated
            shape_compilation.append(
                {
                    "signature": signature,
                    "invocation_count": len(buckets),
                    "first_compilation_seconds": first,
                    "repeated_compilation_seconds": repeated,
                }
            )

        solver_pdipm_calls = [
            call
            for call in self.pdipm_calls
            if call.get("phase_category") == "solver"
        ]
        solver_zero_barrier_calls = [
            call
            for call in self.zero_barrier_calls
            if call.get("phase_category") == "solver"
        ]
        solver_budget = _wall_budget(
            wall_seconds=solver_wall_seconds,
            pdipm_calls=solver_pdipm_calls,
            zero_barrier_calls=solver_zero_barrier_calls,
        )
        phase_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for phase in self.phases:
            key = (str(phase["name"]), str(phase["category"]))
            phase_groups.setdefault(key, []).append(phase)
        phase_budgets = []
        for (name, category), phases in phase_groups.items():
            phase_pdipm_calls = [
                call
                for call in self.pdipm_calls
                if call.get("phase") == name
                and call.get("phase_category") == category
            ]
            phase_zero_barrier_calls = [
                call
                for call in self.zero_barrier_calls
                if call.get("phase") == name
                and call.get("phase_category") == category
            ]
            budget = _wall_budget(
                wall_seconds=sum(
                    float(phase["wall_seconds"]) for phase in phases
                ),
                pdipm_calls=phase_pdipm_calls,
                zero_barrier_calls=phase_zero_barrier_calls,
            )
            phase_budgets.append(
                {
                    "name": name,
                    "category": category,
                    "invocation_count": len(phases),
                    **budget,
                }
            )
        function_evaluations = [
            int(call["function_evaluations"])
            for call in self.zero_barrier_calls
            if call.get("function_evaluations") is not None
        ]
        other_solver_wall = (
            solver_budget["pdipm_internal_orchestration_wall_seconds"]
            + solver_budget[
                "outside_pdipm_and_zero_barrier_wall_seconds"
            ]
        )
        unphased_wall = (
            float(workload_wall_seconds)
            - setup_wall_seconds
            - solver_wall_seconds
        )
        timing_consistent = (
            solver_budget["attribution_consistent"]
            and all(
                budget["attribution_consistent"] for budget in phase_budgets
            )
            and unphased_wall
            >= -_ATTRIBUTION_ABSOLUTE_TOLERANCE_SECONDS
        )
        return {
            "workload_wall_seconds": float(workload_wall_seconds),
            "setup_phase_wall_seconds": setup_wall_seconds,
            "solver_phase_wall_seconds": solver_wall_seconds,
            "unphased_workload_wall_seconds": unphased_wall,
            "pdipm": {
                "call_count": len(self.pdipm_calls),
                "bucket_count": sum(
                    len(call.get("buckets", ())) for call in self.pdipm_calls
                ),
                "compilation_seconds": pdipm_compilation,
                "execution_seconds": pdipm_execution,
                "diagnostic_seconds": pdipm_diagnostics,
                "diagnostic_compilation_seconds": (
                    pdipm_diagnostic_compilation
                ),
                "diagnostic_execution_seconds": pdipm_diagnostic_execution,
                "wall_seconds": pdipm_wall,
                "first_shape_compilation_seconds": first_shape_compilation,
                "repeated_shape_compilation_seconds": (
                    repeated_shape_compilation
                ),
                "executable_shape_count": len(shape_groups),
                "shape_compilation": tuple(shape_compilation),
                "calls": tuple(self.pdipm_calls),
            },
            "zero_barrier": {
                "execution_location": "host_cpu",
                "call_count": len(self.zero_barrier_calls),
                "host_wall_seconds": zero_barrier_wall,
                "function_evaluations": sum(function_evaluations),
                "calls": tuple(self.zero_barrier_calls),
            },
            "other_solver_and_orchestration_wall_seconds": other_solver_wall,
            "solver_budget": solver_budget,
            "phase_budgets": tuple(phase_budgets),
            "timing_attribution_consistent": timing_consistent,
            "phases": tuple(self.phases),
        }


__all__ = ("SOLVER_BUDGET_SCHEMA", "TimingCollector")
