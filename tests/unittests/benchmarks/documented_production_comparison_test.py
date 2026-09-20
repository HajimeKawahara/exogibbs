"""Documented module commands require complete independent comparison output."""

import json

import pytest

from benchmarks.documented_examples import acceptance, run_all


@pytest.mark.parametrize("failure", [None, "preflight", "comparison", "partial"])
def test_production_comparison_rejects_exit_zero_without_semantic_success(tmp_path, failure):
    job = run_all.Job("fastchem4_production_comparison", "comparison.py")
    preflight = {"passed": failure != "preflight"}
    report = {"preflight": preflight, "summary": {"comparison_completed": failure != "comparison"},
              "input_contract": {"temperature_K": [1800.0, 1600.0, 1400.0, 1200.0],
                                 "pressure_bar": [0.1] * 4}, "layers": [{}, {}, {}, {}]}
    if failure == "partial":
        report["layers"].pop()
    (tmp_path / "comparison.json").write_text(json.dumps(report))
    (tmp_path / "comparison.preflight.json").write_text(json.dumps(preflight))
    errors = acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)
    assert bool(errors) is (failure is not None)


def test_module_coverage_cannot_hide_a_new_documented_benchmark(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "example.rst").write_text("python -m benchmarks.new_comparison\n")
    japanese = tmp_path / "doc_ExoGibbs"
    japanese.mkdir()
    (japanese / "main_ja.tex").write_text("Manual")
    script = tmp_path / "benchmarks/new_comparison.py"
    script.parent.mkdir()
    script.write_text("print('comparison')\n")
    assert not acceptance.coverage_report(tmp_path, japanese, (), {})["accepted"]
    jobs = (run_all.Job("comparison", "benchmarks/new_comparison.py"),)
    report = acceptance.coverage_report(tmp_path, japanese, jobs, {})
    assert report["accepted"]
    assert report["documented_commands"][0]["module"] == "benchmarks.new_comparison"


def test_production_job_uses_all_documented_points_and_the_selected_backend():
    job = next(job for job in run_all.build_jobs() if job.name == "fastchem4_production_comparison")
    points = [job.arguments[index + 1] for index, argument in enumerate(job.arguments) if argument == "--point"]
    assert points == ["1800,0.1", "1600,0.1", "1400,0.1", "1200,0.1"]
    assert job.arguments[job.arguments.index("--jax-platform") + 1] == "{platform}"
    assert "fastchem_source" in job.resources
    assert "comparison.preflight.json" in job.artifacts
    assert "--preflight-only" not in job.arguments
