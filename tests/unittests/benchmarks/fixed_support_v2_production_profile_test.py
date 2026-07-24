import runpy
from pathlib import Path


_module = runpy.run_path(
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "fixed_support_v2"
    / "production_profile_gpu_gate.py"
)
_evaluate_budgets = _module["_evaluate_budgets"]
_parse_gpu_compute_processes = _module["_parse_gpu_compute_processes"]
_preflight_report = _module["_preflight_report"]
DEFAULT_FAMILIES = _module["DEFAULT_FAMILIES"]


def test_production_profile_preflight_requires_promoted_v2_default():
    report = _preflight_report(DEFAULT_FAMILIES)
    launcher = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "fixed_support_v2"
        / "run_fixed_support_v2_production_profile_gpu.csh"
    ).read_text(encoding="utf-8")

    assert report["passed"]
    assert report["checks"]["public_default_is_promoted_head_v2"]
    assert report["checks"]["validated_schedule_is_exact"]
    assert report["checks"]["validated_support_limit_is_exact"]
    assert report["checks"]["approved_runtime_budget_is_exact"]
    assert "env JAX_PLATFORMS=cpu python $RUNNER" in launcher
    assert "\nJAX_PLATFORMS=cpu python $RUNNER" not in launcher


def test_production_profile_budget_requires_all_reviewed_limits():
    cold = [{"compilation_seconds": 3.0, "wall_seconds": 5.0}]
    warm = [{"execution_seconds": 0.5, "wall_seconds": 1.0}]
    incomplete = {
        "cold_compilation_seconds": 4.0,
        "cold_wall_seconds": 6.0,
        "warm_execution_seconds": None,
        "warm_wall_seconds": 2.0,
    }

    report = _evaluate_budgets(cold, warm, incomplete)

    assert not report["approved_limits_supplied"]
    assert not report["passed"]
    assert report["checks"]["warm_execution_seconds"] is None


def test_production_profile_budget_applies_maximum_family_time():
    cold = [
        {"compilation_seconds": 3.0, "wall_seconds": 5.0},
        {"compilation_seconds": 7.0, "wall_seconds": 9.0},
    ]
    warm = [
        {"execution_seconds": 0.5, "wall_seconds": 1.0},
        {"execution_seconds": 1.5, "wall_seconds": 2.0},
    ]
    limits = {
        "cold_compilation_seconds": 8.0,
        "cold_wall_seconds": 10.0,
        "warm_execution_seconds": 2.0,
        "warm_wall_seconds": 3.0,
    }

    report = _evaluate_budgets(cold, warm, limits)

    assert report["observed_seconds"]["cold_compilation_seconds"] == 7.0
    assert report["observed_seconds"]["warm_execution_seconds"] == 1.5
    assert report["passed"]


def test_gpu_compute_process_parser_preserves_pid_and_name():
    output = "123, python\n456, /usr/bin/python worker.py\n"

    assert _parse_gpu_compute_processes(output) == [
        {"pid": 123, "process_name": "python"},
        {"pid": 456, "process_name": "/usr/bin/python worker.py"},
    ]
