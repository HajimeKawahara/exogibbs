from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.documented_examples.ldwarf_repeated import (  # noqa: E402
    DEFAULT_EVALUATION_COUNT,
    PRESSURE_OFFSET_LIMIT_DEX,
    TEMPERATURE_OFFSET_LIMIT_K,
    TEMPERATURE_TILT_LIMIT_K,
    _new_shape_signatures,
    _pdipm_support_signature,
    _result_status,
    _write_markdown,
    generate_conditions,
    steady_statistics,
)


def test_repeated_conditions_are_deterministic_smooth_and_bounded() -> None:
    base_temperature = np.linspace(1100.0, 2600.0, 13)
    base_pressure = np.logspace(-4.0, 2.0, 13)

    first = generate_conditions(base_temperature, base_pressure)
    second = generate_conditions(base_temperature, base_pressure)

    assert DEFAULT_EVALUATION_COUNT == 10
    assert first["sha256"] == second["sha256"]
    assert first["temperature_k"] == pytest.approx(second["temperature_k"])
    assert first["pressure_bar"] == pytest.approx(second["pressure_bar"])
    assert first["temperature_k"].shape == (10, 13)
    assert first["pressure_bar"].shape == (10, 13)
    assert first["temperature_k"].dtype == np.float64
    assert first["pressure_bar"].dtype == np.float64
    assert np.all(np.diff(first["temperature_k"], axis=1) > 0.0)
    assert np.all(np.diff(first["pressure_bar"], axis=1) > 0.0)
    assert max(abs(value) for value in first["pressure_offset_dex"]) <= (
        PRESSURE_OFFSET_LIMIT_DEX
    )
    assert max(abs(value) for value in first["temperature_offset_k"]) <= (
        TEMPERATURE_OFFSET_LIMIT_K
    )
    assert max(abs(value) for value in first["temperature_tilt_k"]) <= (
        TEMPERATURE_TILT_LIMIT_K
    )


def test_steady_statistics_exclude_cold_call() -> None:
    statistics = steady_statistics(
        tuple(float(value) for value in range(1, 11))
    )

    assert statistics["evaluation_count"] == 10
    assert statistics["total_wall_seconds"] == 55.0
    assert statistics["mean_wall_seconds_per_evaluation"] == 5.5
    assert statistics["median_wall_seconds_per_evaluation"] == 5.5
    assert statistics["p95_wall_seconds_per_evaluation"] == pytest.approx(9.55)
    assert statistics["minimum_wall_seconds_per_evaluation"] == 1.0
    assert statistics["maximum_wall_seconds_per_evaluation"] == 10.0


def test_pdipm_support_signature_detects_identity_not_only_shape() -> None:
    first = (
        {
            "buckets": (
                {"layer_indices": (0, 2), "support_indices": (1, 4)},
                {"layer_indices": (1,), "support_indices": (3,)},
            )
        },
    )
    reordered = (
        {
            "buckets": (
                {"layer_indices": (1,), "support_indices": (3,)},
                {"layer_indices": (0, 2), "support_indices": (1, 4)},
            )
        },
    )
    same_shape_different_support = (
        {
            "buckets": (
                {"layer_indices": (0, 2), "support_indices": (1, 5)},
                {"layer_indices": (1,), "support_indices": (3,)},
            )
        },
    )

    signature = _pdipm_support_signature(first)

    assert signature == _pdipm_support_signature(reordered)
    assert signature != _pdipm_support_signature(same_shape_different_support)


def test_pdipm_support_signature_uses_real_support_for_each_padded_row() -> None:
    first = (
        {
            "buckets": (
                {
                    "layer_indices": (0, 1),
                    "source_layer_indices": (2, 7),
                    "support_indices": (2, 3, 4),
                    "support_indices_by_layer": ((2,), (3, 4)),
                    "support_capacity": 128,
                    "batch_capacity": 13,
                },
            )
        },
    )
    changed = (
        {
            "buckets": (
                {
                    "layer_indices": (0, 1),
                    "source_layer_indices": (2, 7),
                    "support_indices": (2, 3, 4),
                    "support_indices_by_layer": ((3,), (2, 4)),
                    "support_capacity": 128,
                    "batch_capacity": 13,
                },
            )
        },
    )

    assert _pdipm_support_signature(first) == (
        (((2,), (2,)), ((7,), (3, 4))),
    )
    assert _pdipm_support_signature(first) != _pdipm_support_signature(changed)


def test_new_shape_and_validation_fail_closed() -> None:
    cold = {
        "pdipm": {"shape_compilation": ({"signature": "shape-a"},)}
    }
    cached = {
        "pdipm": {"shape_compilation": ({"signature": "shape-a"},)}
    }
    changed = {
        "pdipm": {
            "shape_compilation": (
                {"signature": "shape-a"},
                {"signature": "shape-b"},
            )
        }
    }

    assert _new_shape_signatures(cold, cached) == ()
    assert _new_shape_signatures(cold, changed) == ("shape-b",)
    assert (
        _result_status(
            all_evaluations_accepted=False,
            timing_attribution_consistent=True,
            no_new_pdipm_executable_shapes_after_cold=True,
        )
        == "fail_validation"
    )
    assert (
        _result_status(
            all_evaluations_accepted=True,
            timing_attribution_consistent=False,
            no_new_pdipm_executable_shapes_after_cold=True,
        )
        == "fail_timing"
    )
    assert (
        _result_status(
            all_evaluations_accepted=True,
            timing_attribution_consistent=True,
            no_new_pdipm_executable_shapes_after_cold=False,
        )
        == "fail_recompilation"
    )


def test_markdown_reports_cold_and_steady_pdipm_calls(tmp_path: Path) -> None:
    output = tmp_path / "summary.md"
    per_evaluation = {
        "pdipm_call_count": 1.0,
        "pdipm_bucket_count": 3.0,
        "pdipm_compilation_seconds": 0.01,
        "pdipm_execution_seconds": 1.0,
        "pdipm_diagnostic_compilation_seconds": 0.0,
        "pdipm_diagnostic_execution_seconds": 0.0,
        "pdipm_internal_orchestration_wall_seconds": 0.1,
        "zero_barrier_host_wall_seconds": 0.2,
        "outside_pdipm_and_zero_barrier_wall_seconds": 0.3,
    }
    payload = {
        "status": "pass",
        "cold": {
            "timing": {"solver_phase_wall_seconds": 10.0},
            "pdipm_compilation_boundary_seconds": 8.0,
            "budget": {"pdipm_call_count": 1, "pdipm_bucket_count": 3},
        },
        "steady": {
            "evaluation_count": 10,
            "total_wall_seconds": 16.0,
            "mean_wall_seconds_per_evaluation": 1.6,
            "median_wall_seconds_per_evaluation": 1.5,
            "p95_wall_seconds_per_evaluation": 2.0,
            "new_pdipm_executable_shape_count_after_cold": 0,
            "no_new_pdipm_executable_shapes_after_cold": True,
            "final_support_identity_preserved": True,
            "pdipm_support_identity_preserved": True,
            "budget_per_evaluation": per_evaluation,
        },
    }

    _write_markdown(output, payload)
    text = output.read_text()

    assert "Cold PD-IPM calls: 1" in text
    assert "| pdipm_calls_per_evaluation | 1.0 |" in text
    assert "| pdipm_buckets_per_evaluation | 3.0 |" in text


def test_repeated_module_does_not_import_jax_before_configuration() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import benchmarks.documented_examples.ldwarf_repeated; "
                "raise SystemExit('jax' in sys.modules)"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert completed.returncode == 0


def test_gpu_sequence_runs_only_exogibbs_benchmarks() -> None:
    script = (
        REPOSITORY_ROOT
        / "benchmarks"
        / "documented_examples"
        / "run_all_gpu.csh"
    )
    text = script.read_text()

    assert script.stat().st_mode & 0o111
    assert "benchmarks.documented_examples.run" in text
    assert "benchmarks.documented_examples.ldwarf_repeated" in text
    assert "rocky_raccoon_boundary_corpus_test.py" in text
    assert "foreach COMPILE_LIGHT ( false true )" in text
    assert text.index("warm-boundary correctness gate") < text.index(
        '"$PYTHON_COMMAND" -m benchmarks.documented_examples.run'
    )
    assert text.count("--evaluations 10") == 2
    assert "--optimization default" in text
    assert "--optimization disable_most_optimizations" in text
    assert "exojax" not in text.lower()
