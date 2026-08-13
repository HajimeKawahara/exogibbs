from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.documented_examples import instrumentation
from benchmarks.documented_examples import workloads
from benchmarks.documented_examples.instrumentation import SOLVER_BUDGET_SCHEMA
from benchmarks.documented_examples.instrumentation import TimingCollector
from benchmarks.documented_examples.manifest import CASES
from benchmarks.documented_examples.manifest import example_documents_from_index
from benchmarks.documented_examples.run import benchmark_jobs
from benchmarks.documented_examples.run import _validate_worker_result
from benchmarks.documented_examples.run import summary_row
from benchmarks.documented_examples import run as suite_runner
from benchmarks.documented_examples.worker import _jax_environment
from benchmarks.documented_examples.worker import configure_jax_environment


def test_manifest_covers_every_documented_example() -> None:
    documents = example_documents_from_index(
        REPOSITORY_ROOT / "documents" / "index.rst"
    )

    assert documents == tuple(case.document for case in CASES)
    assert len({case.case_id for case in CASES}) == len(CASES)
    assert len({case.workload for case in CASES}) == len(CASES)
    assert set(case.workload for case in CASES) == set(workloads.WORKLOADS)
    for case in CASES:
        assert case.full_output_layer_count > 0
        assert case.output_rows_per_condition > 0
        assert (
            case.full_output_layer_count % case.output_rows_per_condition == 0
        )
        assert case.expected_output_layer_count(1) == (
            case.output_rows_per_condition
        )
        assert len(set(case.expected_phases)) == len(case.expected_phases)
        assert all(
            category in ("setup", "solver")
            for _, category in case.expected_phases
        )
        assert case.source_scripts
        document = REPOSITORY_ROOT / "documents" / f"{case.document}.rst"
        assert document.is_file()
        document_source = document.read_text()
        assert all(
            (REPOSITORY_ROOT / source).is_file()
            for source in case.source_scripts
        )
        assert all(
            Path(source).name in document_source
            for source in case.source_scripts
        )
        assert all(
            (REPOSITORY_ROOT / artifact).is_file()
            for artifact in case.input_artifacts
        )
        assert all(
            Path(artifact).name in document_source
            for artifact in case.input_artifacts
        )

    assert sum(len(case.expected_phases) for case in CASES) == 14


@pytest.mark.parametrize(
    ("platform", "optimization", "jax_platform", "disabled"),
    [
        ("cpu", "default", "cpu", "false"),
        ("gpu", "default", "cuda", "false"),
        (
            "cpu",
            "disable_most_optimizations",
            "cpu",
            "true",
        ),
    ],
)
def test_jax_environment_is_explicit_before_worker_imports(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    optimization: str,
    jax_platform: str,
    disabled: str,
) -> None:
    monkeypatch.setenv("JAX_DISABLE_MOST_OPTIMIZATIONS", "inherited")

    configure_jax_environment(platform, optimization)

    assert __import__("os").environ["JAX_PLATFORMS"] == jax_platform
    assert __import__("os").environ["JAX_PLATFORM_NAME"] == jax_platform
    assert __import__("os").environ["JAX_ENABLE_X64"] == "1"
    assert __import__("os").environ["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert __import__("os").environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert (
        __import__("os").environ["JAX_DISABLE_MOST_OPTIMIZATIONS"]
        == disabled
    )


def test_worker_module_does_not_import_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import benchmarks.documented_examples.worker; "
                "raise SystemExit('jax' in sys.modules)"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert completed.returncode == 0


@pytest.mark.parametrize(
    ("values", "exception"),
    [
        (
            {
                "jax_enable_x64": True,
                "jax_enable_compilation_cache": False,
            },
            NotImplementedError,
        ),
        (
            {
                "jax_disable_most_optimizations": False,
                "jax_enable_x64": True,
                "jax_enable_compilation_cache": False,
            },
            RuntimeError,
        ),
        (
            {
                "jax_disable_most_optimizations": True,
                "jax_enable_x64": False,
                "jax_enable_compilation_cache": False,
            },
            RuntimeError,
        ),
        (
            {
                "jax_disable_most_optimizations": True,
                "jax_enable_x64": True,
                "jax_enable_compilation_cache": True,
            },
            RuntimeError,
        ),
    ],
)
def test_jax_environment_rejects_unsupported_or_ineffective_config(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, object],
    exception: type[Exception],
) -> None:
    fake_jax = SimpleNamespace(
        __version__="test",
        config=SimpleNamespace(values=values),
        devices=lambda: (
            SimpleNamespace(id=0, platform="cpu", device_kind="cpu"),
        ),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(
        sys.modules,
        "jaxlib",
        SimpleNamespace(__version__="test"),
    )

    with pytest.raises(exception):
        _jax_environment("cpu", "disable_most_optimizations")


def test_default_benchmark_matrix_covers_every_case_and_mode() -> None:
    case_ids = tuple(case.case_id for case in CASES)
    jobs = benchmark_jobs(
        case_ids,
        ("cpu", "gpu"),
        ("default", "disable_most_optimizations"),
        repeat=2,
    )

    assert len(jobs) == len(CASES) * 2 * 2 * 2
    assert set(jobs) == {
        (case_id, platform, optimization, repetition)
        for case_id in case_ids
        for platform in ("cpu", "gpu")
        for optimization in ("default", "disable_most_optimizations")
        for repetition in (1, 2)
    }


def test_timing_collector_aggregates_and_restores_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exogibbs.equilibrium.condensate.fixed_support import batch
    from exogibbs.equilibrium.condensate.fixed_support import zero_barrier

    pdipm_result = {
        "backend": "cpu",
        "compilation_seconds": 0.2,
        "execution_seconds": 0.3,
        "diagnostic_seconds": 0.1,
        "diagnostic_compilation_seconds": 0.06,
        "diagnostic_execution_seconds": 0.04,
        "bucket_reports": (
            {
                "support_indices": (2, 3),
                "support_indices_by_layer": ((2,), (3,)),
                "layer_indices": (0, 1),
                "source_layer_indices": (4, 7),
                "support_capacity": 8,
                "batch_capacity": 4,
                "valid_batch_size": 2,
                "compilation_seconds": 0.2,
                "execution_seconds": 0.3,
                "diagnostic_compilation_seconds": 0.06,
                "diagnostic_execution_seconds": 0.04,
            },
        ),
    }
    zero_result = SimpleNamespace(
        accepted=True,
        support_indices=(2,),
        report={
            "exact_active_set_closure": {
                "cumulative_function_evaluations": 7,
                "round_count": 2,
                "termination_reason": "accepted",
            }
        },
    )

    repeated_pdipm_result = {
        **pdipm_result,
        "compilation_seconds": 0.03,
        "bucket_reports": (
            {
                **pdipm_result["bucket_reports"][0],
                "compilation_seconds": 0.03,
            },
        ),
    }
    pdipm_results = iter((pdipm_result, repeated_pdipm_result))

    def fake_pdipm(*_args: object, **_kwargs: object) -> dict:
        return next(pdipm_results)

    def fake_zero_barrier(*_args: object, **_kwargs: object) -> object:
        return zero_result

    monkeypatch.setattr(batch, "run_fixed_support_profile", fake_pdipm)
    monkeypatch.setattr(
        zero_barrier,
        "polish_zero_barrier_active_support",
        fake_zero_barrier,
    )
    ticks = iter(float(index) for index in range(20))
    monkeypatch.setattr(instrumentation.time, "perf_counter", lambda: next(ticks))

    formula = np.zeros((3, 5), dtype=np.float64)
    with TimingCollector() as collector:
        with collector.phase("setup", category="setup"):
            pass
        with collector.phase("solve", category="solver"):
            assert collector._batch_module.run_fixed_support_profile(
                buckets=(), formula_matrix=formula
            ) is pdipm_result
            assert collector._batch_module.run_fixed_support_profile(
                buckets=(), formula_matrix=formula
            ) is repeated_pdipm_result
            assert collector._zero_barrier_module.polish_zero_barrier_active_support(
                support_indices=(2,)
            ) is zero_result

    assert batch.run_fixed_support_profile is fake_pdipm
    assert (
        zero_barrier.polish_zero_barrier_active_support
        is fake_zero_barrier
    )
    summary = collector.summary(workload_wall_seconds=10.0)
    assert summary["setup_phase_wall_seconds"] == 1.0
    assert summary["solver_phase_wall_seconds"] == 7.0
    assert summary["pdipm"]["compilation_seconds"] == pytest.approx(0.23)
    assert summary["pdipm"]["execution_seconds"] == pytest.approx(0.6)
    assert summary["pdipm"]["diagnostic_seconds"] == pytest.approx(0.2)
    assert summary["pdipm"]["diagnostic_compilation_seconds"] == pytest.approx(
        0.12
    )
    assert summary["pdipm"]["diagnostic_execution_seconds"] == pytest.approx(
        0.08
    )
    assert summary["pdipm"]["wall_seconds"] == 2.0
    assert summary["pdipm"]["executable_shape_count"] == 1
    assert summary["pdipm"]["first_shape_compilation_seconds"] == 0.2
    assert summary["pdipm"]["repeated_shape_compilation_seconds"] == 0.03
    assert summary["pdipm"]["shape_compilation"][0]["invocation_count"] == 2
    assert summary["pdipm"]["shape_compilation"][0]["signature"] == (
        "backend=cpu,dtype=float64,config=dc937b598926,elements=3,gas=5,"
        "support=8,batch=4,diagnostics=1"
    )
    bucket_record = summary["pdipm"]["calls"][0]["buckets"][0]
    assert bucket_record["support_indices_by_layer"] == ((2,), (3,))
    assert bucket_record["source_layer_indices"] == (4, 7)
    assert bucket_record["valid_batch_size"] == 2
    assert summary["zero_barrier"]["host_wall_seconds"] == 1.0
    assert summary["zero_barrier"]["function_evaluations"] == 7
    assert summary["zero_barrier"]["calls"][0]["active_set_round_count"] == 2
    solver_budget = summary["solver_budget"]
    assert solver_budget["schema"] == SOLVER_BUDGET_SCHEMA
    assert solver_budget["wall_seconds"] == 7.0
    assert solver_budget["pdipm_call_count"] == 2
    assert solver_budget["pdipm_bucket_count"] == 2
    assert solver_budget["pdipm_wall_seconds"] == 2.0
    assert solver_budget["pdipm_internal_orchestration_wall_seconds"] == (
        pytest.approx(0.97)
    )
    assert solver_budget["zero_barrier_host_wall_seconds"] == 1.0
    assert solver_budget[
        "outside_pdipm_and_zero_barrier_wall_seconds"
    ] == pytest.approx(4.0)
    assert solver_budget["attribution_consistent"] is True
    assert summary["other_solver_and_orchestration_wall_seconds"] == (
        pytest.approx(4.97)
    )
    phase_budgets = {
        budget["name"]: budget for budget in summary["phase_budgets"]
    }
    assert phase_budgets["setup"][
        "outside_pdipm_and_zero_barrier_wall_seconds"
    ] == 1.0
    assert phase_budgets["solve"]["pdipm_call_count"] == 2
    assert phase_budgets["solve"]["zero_barrier_call_count"] == 1
    assert summary["timing_attribution_consistent"] is True


def test_phase_budget_attributes_gas_only_work_to_the_phase_residual() -> None:
    collector = TimingCollector()
    collector.phases = [
        {
            "name": "solve_condensate_profile",
            "category": "solver",
            "wall_seconds": 3.0,
            "status": "pass",
            "error": None,
        },
        {
            "name": "solve_gas_only_profile",
            "category": "solver",
            "wall_seconds": 2.0,
            "status": "pass",
            "error": None,
        },
    ]
    collector.pdipm_calls = [
        {
            "phase": "solve_condensate_profile",
            "phase_category": "solver",
            "wall_seconds": 1.5,
            "compilation_seconds": 0.5,
            "execution_seconds": 0.5,
            "diagnostic_seconds": 0.0,
            "diagnostic_compilation_seconds": 0.0,
            "diagnostic_execution_seconds": 0.0,
            "buckets": ({"signature": "one", "compilation_seconds": 0.5},),
        }
    ]

    summary = collector.summary(workload_wall_seconds=5.0)
    budgets = {budget["name"]: budget for budget in summary["phase_budgets"]}

    gas_budget = budgets["solve_gas_only_profile"]
    assert gas_budget["pdipm_call_count"] == 0
    assert gas_budget["zero_barrier_call_count"] == 0
    assert gas_budget[
        "outside_pdipm_and_zero_barrier_wall_seconds"
    ] == 2.0
    assert gas_budget["attribution_consistent"] is True


def test_run_case_fails_closed_but_allows_smoke_layer_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[0]
    result = {
        "all_layers_converged": True,
        "output_layer_count": 2,
    }
    monkeypatch.setitem(
        workloads.WORKLOADS,
        case.workload,
        lambda _collector, _smoke_layers: result,
    )

    with pytest.raises(RuntimeError, match="expected"):
        workloads.run_case(case, TimingCollector(), smoke_layers=None)
    assert workloads.run_case(
        case,
        TimingCollector(),
        smoke_layers=1,
    ) is result

    result["output_layer_count"] = 1
    with pytest.raises(RuntimeError, match="expected"):
        workloads.run_case(case, TimingCollector(), smoke_layers=1)

    result["output_layer_count"] = 2
    result["all_layers_converged"] = False
    with pytest.raises(RuntimeError, match="converged"):
        workloads.run_case(case, TimingCollector(), smoke_layers=1)

    result.update(all_layers_converged=True, output_layer_count=0)
    with pytest.raises(RuntimeError, match="no output"):
        workloads.run_case(case, TimingCollector(), smoke_layers=1)


@pytest.mark.parametrize(
    (
        "condensate_amount",
        "gas_amount",
        "caller_audit",
        "status",
        "finite",
        "accepted",
    ),
    [
        (0.0, 1.0, None, "converged", True, True),
        (0.5, 1.0, True, "converged", True, True),
        (0.5, 1.0, None, "converged", True, False),
        (0.5, 1.0, False, "converged", True, False),
        (-0.5, 1.0, None, "converged", True, False),
        (0.0, -1.0, None, "converged", True, False),
        (0.0, 1.0, None, "not_converged", True, False),
        (0.0, 1.0, None, "converged", False, False),
    ],
)
def test_profile_physical_audit_fails_closed(
    condensate_amount: float,
    gas_amount: float,
    caller_audit: bool | None,
    status: str,
    finite: bool,
    accepted: bool,
) -> None:
    lifecycle = {}
    if caller_audit is not None:
        lifecycle["caller_gauge_zero_barrier_kkt"] = {
            "accepted": caller_audit
        }
    gas_n = [gas_amount] if finite else [np.nan]
    layer = SimpleNamespace(
        acceptance_tier="fixed_support_v2_accepted",
        condensate_amounts=np.asarray([condensate_amount]),
        gas_n=np.asarray(gas_n),
        converged=status == "converged",
        status=status,
        diagnostics={"fixed_support_v2": lifecycle},
    )

    report = workloads._profile_physical_audit(
        SimpleNamespace(layers=(layer,))
    )

    assert report["all_layers_finite_and_physically_accepted"] is accepted
    assert report["failed_layer_indices"] == (() if accepted else (0,))


@pytest.mark.parametrize(
    ("gas_x", "accepted"),
    [
        ([[0.25, 0.75]], True),
        ([[0.25, -0.75]], False),
        ([[0.0, 0.0]], False),
        ([[np.nan, 1.0]], False),
        ([0.25, 0.75], False),
    ],
)
def test_gas_profile_physical_audit_fails_closed(
    gas_x: list,
    accepted: bool,
) -> None:
    report = workloads._gas_profile_physical_audit(
        gas_x,
        converged=[True],
        species_count=2,
    )

    assert report["all_layers_finite_nonnegative_and_nonempty"] is accepted


def test_summary_row_flattens_stable_metrics() -> None:
    payload = {
        "case": {"case_id": "case"},
        "scope": {"kind": "full"},
        "execution": {
            "requested_platform": "cpu",
            "optimization_mode": "default",
            "repetition": 1,
        },
        "status": "pass",
        "validation": {"output_layer_count": 4},
        "timing": {
            "workload_wall_seconds": 5.0,
            "setup_phase_wall_seconds": 0.5,
            "solver_phase_wall_seconds": 4.0,
            "unphased_workload_wall_seconds": 0.5,
            "pdipm": {
                "compilation_seconds": 2.0,
                "execution_seconds": 1.0,
                "diagnostic_seconds": 0.5,
                "diagnostic_compilation_seconds": 0.3,
                "diagnostic_execution_seconds": 0.2,
                "wall_seconds": 3.75,
                "first_shape_compilation_seconds": 1.5,
                "repeated_shape_compilation_seconds": 0.5,
                "executable_shape_count": 2,
            },
            "zero_barrier": {
                "host_wall_seconds": 0.25,
                "call_count": 2,
                "function_evaluations": 0,
            },
            "solver_budget": {
                "pdipm_call_count": 3,
                "pdipm_bucket_count": 4,
                "pdipm_wall_seconds": 3.75,
                "pdipm_compilation_seconds": 2.0,
                "pdipm_execution_seconds": 1.0,
                "pdipm_diagnostic_seconds": 0.5,
                "pdipm_diagnostic_compilation_seconds": 0.3,
                "pdipm_diagnostic_execution_seconds": 0.2,
                "pdipm_internal_orchestration_wall_seconds": 0.25,
                "zero_barrier_call_count": 2,
                "zero_barrier_function_evaluations": 0,
                "zero_barrier_host_wall_seconds": 0.25,
                "outside_pdipm_and_zero_barrier_wall_seconds": 0.0,
            },
            "other_solver_and_orchestration_wall_seconds": 0.25,
            "timing_attribution_consistent": True,
        },
    }

    row = summary_row(payload)

    assert row["case_id"] == "case"
    assert row["pdipm_compile_seconds"] == 2.0
    assert row["pdipm_execute_seconds"] == 1.0
    assert row["pdipm_repeated_shape_compilation_seconds"] == 0.5
    assert row["pdipm_executable_shape_count"] == 2
    assert row["zero_barrier_host_wall_seconds"] == 0.25
    assert row["pdipm_call_count"] == 3
    assert row["pdipm_bucket_count"] == 4
    assert row["pdipm_diagnostic_compilation_seconds"] == 0.3
    assert row["pdipm_internal_orchestration_wall_seconds"] == 0.25


def test_worker_contract_recomputes_budget_identities() -> None:
    case_id = "fe_fes_rainout_demo"
    timing = {
        "workload_wall_seconds": 5.0,
        "setup_phase_wall_seconds": 0.5,
        "solver_phase_wall_seconds": 4.0,
        "unphased_workload_wall_seconds": 0.5,
        "pdipm": {
            "call_count": 1,
            "bucket_count": 1,
            "compilation_seconds": 2.0,
            "execution_seconds": 1.0,
            "diagnostic_seconds": 0.5,
            "diagnostic_compilation_seconds": 0.3,
            "diagnostic_execution_seconds": 0.2,
            "wall_seconds": 3.75,
        },
        "zero_barrier": {
            "call_count": 0,
            "host_wall_seconds": 0.0,
            "function_evaluations": 0,
        },
        "solver_budget": {
            "schema": SOLVER_BUDGET_SCHEMA,
            "wall_seconds": 4.0,
            "pdipm_call_count": 1,
            "pdipm_bucket_count": 1,
            "pdipm_wall_seconds": 3.75,
            "pdipm_compilation_seconds": 2.0,
            "pdipm_execution_seconds": 1.0,
            "pdipm_diagnostic_seconds": 0.5,
            "pdipm_diagnostic_compilation_seconds": 0.3,
            "pdipm_diagnostic_execution_seconds": 0.2,
            "pdipm_internal_orchestration_wall_seconds": 0.25,
            "zero_barrier_call_count": 0,
            "zero_barrier_function_evaluations": 0,
            "zero_barrier_host_wall_seconds": 0.0,
            "outside_pdipm_and_zero_barrier_wall_seconds": 0.25,
            "attribution_delta_seconds": 0.0,
            "attribution_consistent": True,
        },
        "phase_budgets": [
            {
                "name": "solve_local",
                "category": "solver",
                "invocation_count": 1,
                "schema": SOLVER_BUDGET_SCHEMA,
                "wall_seconds": 4.0,
                "pdipm_call_count": 1,
                "pdipm_bucket_count": 1,
                "pdipm_wall_seconds": 3.75,
                "pdipm_compilation_seconds": 2.0,
                "pdipm_execution_seconds": 1.0,
                "pdipm_diagnostic_seconds": 0.5,
                "pdipm_diagnostic_compilation_seconds": 0.3,
                "pdipm_diagnostic_execution_seconds": 0.2,
                "pdipm_internal_orchestration_wall_seconds": 0.25,
                "zero_barrier_call_count": 0,
                "zero_barrier_function_evaluations": 0,
                "zero_barrier_host_wall_seconds": 0.0,
                "outside_pdipm_and_zero_barrier_wall_seconds": 0.25,
                "attribution_delta_seconds": 0.0,
                "attribution_consistent": True,
            },
            {
                "name": "solve_rainout",
                "category": "solver",
                "invocation_count": 1,
                "schema": SOLVER_BUDGET_SCHEMA,
                "wall_seconds": 0.0,
                "pdipm_call_count": 0,
                "pdipm_bucket_count": 0,
                "pdipm_wall_seconds": 0.0,
                "pdipm_compilation_seconds": 0.0,
                "pdipm_execution_seconds": 0.0,
                "pdipm_diagnostic_seconds": 0.0,
                "pdipm_diagnostic_compilation_seconds": 0.0,
                "pdipm_diagnostic_execution_seconds": 0.0,
                "pdipm_internal_orchestration_wall_seconds": 0.0,
                "zero_barrier_call_count": 0,
                "zero_barrier_function_evaluations": 0,
                "zero_barrier_host_wall_seconds": 0.0,
                "outside_pdipm_and_zero_barrier_wall_seconds": 0.0,
                "attribution_delta_seconds": 0.0,
                "attribution_consistent": True,
            },
            {
                "name": "build_reduced_setup",
                "category": "setup",
                "invocation_count": 1,
                "schema": SOLVER_BUDGET_SCHEMA,
                "wall_seconds": 0.5,
                "pdipm_call_count": 0,
                "pdipm_bucket_count": 0,
                "pdipm_wall_seconds": 0.0,
                "pdipm_compilation_seconds": 0.0,
                "pdipm_execution_seconds": 0.0,
                "pdipm_diagnostic_seconds": 0.0,
                "pdipm_diagnostic_compilation_seconds": 0.0,
                "pdipm_diagnostic_execution_seconds": 0.0,
                "pdipm_internal_orchestration_wall_seconds": 0.0,
                "zero_barrier_call_count": 0,
                "zero_barrier_function_evaluations": 0,
                "zero_barrier_host_wall_seconds": 0.0,
                "outside_pdipm_and_zero_barrier_wall_seconds": 0.5,
                "attribution_delta_seconds": 0.0,
                "attribution_consistent": True,
            },
        ],
        "timing_attribution_consistent": True,
    }
    payload = {
        "schema": "exogibbs_documented_example_benchmark_v2",
        "status": "pass",
        "case": {"case_id": case_id},
        "execution": {
            "requested_platform": "cpu",
            "optimization_mode": "default",
            "repetition": 1,
        },
        "scope": {"kind": "smoke", "smoke_layers": 1},
        "validation": {
            "all_layers_converged": True,
            "output_layer_count": 2,
        },
        "timing": timing,
        "environment": {},
    }

    assert not _validate_worker_result(
        payload,
        case_id=case_id,
        platform="cpu",
        optimization="default",
        repetition=1,
        smoke_layers=1,
    )
    timing["solver_budget"][
        "outside_pdipm_and_zero_barrier_wall_seconds"
    ] = 9.0

    violations = _validate_worker_result(
        payload,
        case_id=case_id,
        platform="cpu",
        optimization="default",
        repetition=1,
        smoke_layers=1,
    )

    assert any("wall partition" in violation for violation in violations)


@pytest.mark.parametrize(
    (
        "worker_status",
        "worker_return_code",
        "reported_case_id",
        "result_status",
        "suite_status",
        "main_code",
    ),
    [
        ("pass", 0, "fe_fes_rainout_demo", "pass", "pass", 0),
        ("pass", 1, "fe_fes_rainout_demo", "worker_error", "incomplete", 1),
        (
            "unavailable",
            2,
            "fe_fes_rainout_demo",
            "unavailable",
            "incomplete",
            1,
        ),
        ("pass", 0, "wrong_case", "invalid_result", "incomplete", 1),
    ],
)
def test_suite_writes_summary_and_fails_closed_on_worker_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_status: str,
    worker_return_code: int,
    reported_case_id: str,
    result_status: str,
    suite_status: str,
    main_code: int,
) -> None:
    output_directory = tmp_path / "results"

    def fake_run(command: tuple[str, ...], **_kwargs: object) -> object:
        child_environment = _kwargs["env"]
        python_paths = child_environment["PYTHONPATH"].split(
            __import__("os").pathsep
        )
        assert python_paths[:2] == [
            str(REPOSITORY_ROOT / "src"),
            str(REPOSITORY_ROOT),
        ]
        output_index = command.index("--output") + 1
        output = Path(command[output_index])
        payload = {
            "schema": "exogibbs_documented_example_benchmark_v2",
            "status": worker_status,
            "case": {"case_id": reported_case_id},
            "scope": {"kind": "smoke"},
            "execution": {
                "requested_platform": "cpu",
                "optimization_mode": "default",
                "repetition": 1,
            },
            "validation": {"output_layer_count": 2},
            "timing": {
                "workload_wall_seconds": 5.0,
                "setup_phase_wall_seconds": 0.5,
                "solver_phase_wall_seconds": 4.0,
                "unphased_workload_wall_seconds": 0.5,
                "pdipm": {
                    "call_count": 3,
                    "bucket_count": 4,
                    "compilation_seconds": 2.0,
                    "execution_seconds": 1.0,
                    "diagnostic_seconds": 0.5,
                    "diagnostic_compilation_seconds": 0.3,
                    "diagnostic_execution_seconds": 0.2,
                    "wall_seconds": 3.75,
                    "first_shape_compilation_seconds": 1.5,
                    "repeated_shape_compilation_seconds": 0.5,
                    "executable_shape_count": 2,
                },
                "zero_barrier": {
                    "host_wall_seconds": 0.25,
                    "call_count": 2,
                    "function_evaluations": 0,
                },
                "other_solver_and_orchestration_wall_seconds": 0.25,
                "solver_budget": {
                    "schema": SOLVER_BUDGET_SCHEMA,
                    "wall_seconds": 4.0,
                    "pdipm_call_count": 3,
                    "pdipm_bucket_count": 4,
                    "pdipm_wall_seconds": 3.75,
                    "pdipm_compilation_seconds": 2.0,
                    "pdipm_execution_seconds": 1.0,
                    "pdipm_diagnostic_seconds": 0.5,
                    "pdipm_diagnostic_compilation_seconds": 0.3,
                    "pdipm_diagnostic_execution_seconds": 0.2,
                    "pdipm_internal_orchestration_wall_seconds": 0.25,
                    "zero_barrier_call_count": 2,
                    "zero_barrier_function_evaluations": 0,
                    "zero_barrier_host_wall_seconds": 0.25,
                    "outside_pdipm_and_zero_barrier_wall_seconds": 0.0,
                    "attribution_delta_seconds": 0.0,
                    "attribution_consistent": True,
                },
                "phase_budgets": [
                    {
                        "name": "solve_local",
                        "category": "solver",
                        "invocation_count": 1,
                        "schema": SOLVER_BUDGET_SCHEMA,
                        "wall_seconds": 4.0,
                        "pdipm_call_count": 3,
                        "pdipm_bucket_count": 4,
                        "pdipm_wall_seconds": 3.75,
                        "pdipm_compilation_seconds": 2.0,
                        "pdipm_execution_seconds": 1.0,
                        "pdipm_diagnostic_seconds": 0.5,
                        "pdipm_diagnostic_compilation_seconds": 0.3,
                        "pdipm_diagnostic_execution_seconds": 0.2,
                        "pdipm_internal_orchestration_wall_seconds": 0.25,
                        "zero_barrier_call_count": 2,
                        "zero_barrier_function_evaluations": 0,
                        "zero_barrier_host_wall_seconds": 0.25,
                        "outside_pdipm_and_zero_barrier_wall_seconds": 0.0,
                        "attribution_delta_seconds": 0.0,
                        "attribution_consistent": True,
                    },
                    {
                        "name": "solve_rainout",
                        "category": "solver",
                        "invocation_count": 1,
                        "schema": SOLVER_BUDGET_SCHEMA,
                        "wall_seconds": 0.0,
                        "pdipm_call_count": 0,
                        "pdipm_bucket_count": 0,
                        "pdipm_wall_seconds": 0.0,
                        "pdipm_compilation_seconds": 0.0,
                        "pdipm_execution_seconds": 0.0,
                        "pdipm_diagnostic_seconds": 0.0,
                        "pdipm_diagnostic_compilation_seconds": 0.0,
                        "pdipm_diagnostic_execution_seconds": 0.0,
                        "pdipm_internal_orchestration_wall_seconds": 0.0,
                        "zero_barrier_call_count": 0,
                        "zero_barrier_function_evaluations": 0,
                        "zero_barrier_host_wall_seconds": 0.0,
                        "outside_pdipm_and_zero_barrier_wall_seconds": 0.0,
                        "attribution_delta_seconds": 0.0,
                        "attribution_consistent": True,
                    },
                    {
                        "name": "build_reduced_setup",
                        "category": "setup",
                        "invocation_count": 1,
                        "schema": SOLVER_BUDGET_SCHEMA,
                        "wall_seconds": 0.5,
                        "pdipm_call_count": 0,
                        "pdipm_bucket_count": 0,
                        "pdipm_wall_seconds": 0.0,
                        "pdipm_compilation_seconds": 0.0,
                        "pdipm_execution_seconds": 0.0,
                        "pdipm_diagnostic_seconds": 0.0,
                        "pdipm_diagnostic_compilation_seconds": 0.0,
                        "pdipm_diagnostic_execution_seconds": 0.0,
                        "pdipm_internal_orchestration_wall_seconds": 0.0,
                        "zero_barrier_call_count": 0,
                        "zero_barrier_function_evaluations": 0,
                        "zero_barrier_host_wall_seconds": 0.0,
                        "outside_pdipm_and_zero_barrier_wall_seconds": 0.5,
                        "attribution_delta_seconds": 0.0,
                        "attribution_consistent": True,
                    },
                ],
                "timing_attribution_consistent": True,
            },
            "resources": {"maximum_resident_set_size_kb": 100},
            "environment": {"backend": "cpu"},
        }
        payload["scope"]["smoke_layers"] = 1
        payload["validation"]["all_layers_converged"] = True
        output.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, worker_return_code)

    monkeypatch.setattr(suite_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--case",
            "fe_fes_rainout_demo",
            "--platform",
            "cpu",
            "--optimization",
            "default",
            "--smoke-layers",
            "1",
            "--output-directory",
            str(output_directory),
        ],
    )

    assert suite_runner.main() == main_code
    summary = json.loads((output_directory / "summary.json").read_text())
    assert summary["schema"] == (
        "exogibbs_documented_example_benchmark_suite_v2"
    )
    assert summary["status"] == suite_status
    assert summary["result_count"] == 1
    assert summary["passed_result_count"] == int(
        result_status == "pass" and worker_return_code == 0
    )
    assert summary["results"][0]["worker_return_code"] == worker_return_code
    assert summary["results"][0]["status"] == result_status
    if result_status == "invalid_result":
        assert summary["results"][0]["reported_status"] == worker_status
        assert summary["results"][0]["contract_violations"]

    with (output_directory / "summary.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    required_fields = {
        "case_id",
        "platform",
        "optimization",
        "status",
        "worker_return_code",
        "workload_wall_seconds",
        "setup_phase_wall_seconds",
        "solver_phase_wall_seconds",
        "unphased_workload_wall_seconds",
        "pdipm_compile_seconds",
        "pdipm_execute_seconds",
        "pdipm_diagnostic_seconds",
        "pdipm_diagnostic_compilation_seconds",
        "pdipm_diagnostic_execution_seconds",
        "pdipm_repeated_shape_compilation_seconds",
        "pdipm_executable_shape_count",
        "pdipm_call_count",
        "pdipm_bucket_count",
        "pdipm_wall_seconds",
        "pdipm_internal_orchestration_wall_seconds",
        "outside_pdipm_and_zero_barrier_wall_seconds",
        "zero_barrier_host_wall_seconds",
        "zero_barrier_call_count",
        "zero_barrier_function_evaluations",
        "timing_attribution_consistent",
        "result_file",
        "log_file",
    }
    assert required_fields <= set(rows[0])
    assert rows[0]["pdipm_compile_seconds"] == "2.0"
    assert rows[0]["pdipm_call_count"] == "3"

    with (output_directory / "phase_budgets.csv").open(
        newline=""
    ) as input_file:
        phase_rows = list(csv.DictReader(input_file))
    assert phase_rows[0]["phase_name"] == "solve_local"
    assert phase_rows[0]["pdipm_call_count"] == "3"
    assert phase_rows[0]["pdipm_diagnostic_seconds"] == "0.5"
    assert phase_rows[0]["attribution_delta_seconds"] == "0.0"
    assert phase_rows[0][
        "outside_pdipm_and_zero_barrier_wall_seconds"
    ] == "0.0"
    markdown = (output_directory / "summary.md").read_text()
    assert "## Workload overview" in markdown
    assert "## Solver wall-time budget" in markdown
    assert "pdipm_call_count" in markdown
    assert "pdipm_internal_orchestration_wall_seconds" in markdown
    assert "Per-phase budgets are in `phase_budgets.csv`" in markdown
