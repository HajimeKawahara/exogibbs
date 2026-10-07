"""Coverage and failure-reporting contracts for every documented example."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from benchmarks.documented_examples import run_all


def test_every_public_script_and_all_expensive_configurations_are_covered():
    jobs = run_all.build_jobs()
    assert len({job.name for job in jobs}) == len(jobs)
    public = {
        str(path.relative_to(run_all.ROOT))
        for path in (run_all.ROOT / "examples").rglob("*.py")
        if any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
               for node in ast.parse(path.read_text()).body)
    }
    covered = {job.script for job in jobs} | set(run_all.ALIASES)
    curated = [job for job in jobs if job.script.endswith("check_curated.py")]
    covered.update(str(Path(job.arguments[job.arguments.index("--script") + 1]).relative_to(run_all.ROOT))
                   for job in curated)
    assert any(job.script.endswith("check_exoeos.py") for job in jobs)
    covered.add("examples/plot_exoeos_pure_fugacity.py")
    assert public <= covered, f"Unexecuted entry points: {public - covered}"
    assert len(curated) == 10
    assert sum(job.script.startswith("examples/comparisons/") for job in jobs) == 13
    assert {alias for alias in run_all.ALIASES.values()} <= {job.name for job in jobs}
    assert {job.arguments for job in jobs if job.script.endswith("/native.py")} == {
        ("--case", f"{kind}_{temperature}")
        for kind in ("source_full", "completed_dry") for temperature in (2350, 3000)
    }
    assert {job.arguments[1] for job in jobs if job.script.endswith("/gas_exchange.py")} == {
        "source_full_2350", "source_full_3000"
    }
    assert {"metal_independent_reference", "metal_source_extraction", "metal_thermochemistry"} <= {
        job.name for job in jobs
    }
    assert "examples/metal_silicate/local.py" not in public
    assert "examples/metal_silicate/source.py" not in public
    retrievals = [job for job in jobs if job.name.startswith("retrieval_")]
    assert len(retrievals) == 4
    assert all("--quick" not in job.arguments for job in retrievals)
    assert all("posterior_samples.npz" in job.artifacts for job in retrievals)
    assert {job.name for job in jobs if job.name.startswith("inline_")} == {
        "inline_preset_ykb4_gas", "inline_preset_fastchem_gas", "inline_preset_fastchem4_gas",
        "inline_preset_fastchem4_condensate", "inline_preset_fastchem4_profile",
        "inline_condensate_profile", "inline_solubility", "inline_magma_gas_interface",
        "inline_ideal_solution_adapter", "inline_ja_magma_gas_interface",
    }


@pytest.mark.parametrize("quick", [False, True])
def test_jobs_run_independently_and_continue_after_failures(tmp_path, monkeypatch, quick):
    jobs = (
        run_all.Job("numerical_failure", "failure.py"),
        run_all.Job("unavailable", "missing.py", resources=("fastchem",)),
        run_all.Job("missing_plot", "plot.py", artifacts=("plot.png",)),
        run_all.Job("missing_report", "report.py"),
        run_all.Job("retrieval_success", "retrieval.py", artifacts=("posterior_samples.npz",)),
    )
    calls = []

    def fake_process(command, **kwargs):
        calls.append(command)
        report_path = Path(command[command.index("--worker-report") + 1])
        name = report_path.parent.name
        if name != "missing_report":
            report_path.write_text(json.dumps({"status": "FAIL" if name == "numerical_failure" else "PASS"}))
        if name == "retrieval_success":
            np.savez(report_path.parent / "posterior_samples.npz", parameter=np.zeros(1000))
            (report_path.parent / "run_status.json").write_text(json.dumps({"mode": "sampling", "state": "complete"}))
            (report_path.parent / "run_summary.json").write_text(json.dumps({
                "sampling_diagnostics": {"passed": True}, "run_config": {"num_warmup": 500, "num_samples": 1000},
            }))
        return SimpleNamespace(returncode=1 if name == "numerical_failure" else 0)

    monkeypatch.setattr(run_all.subprocess, "run", fake_process)
    output = tmp_path / "run"
    assert run_all.run_jobs(jobs, output=output, resources={"fastchem": ""},
                            environment={}, platform="gpu", retrieval_quick=quick) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert [item["status"] for item in summary["jobs"]] == ["FAIL", "UNAVAILABLE", "FAIL", "FAIL", "PASS"]
    assert len(calls) == 4
    assert all(command[:3] == [sys.executable, "-m", "benchmarks.documented_examples.run_all"] for command in calls)
    assert len({command[command.index("--worker-report") + 1] for command in calls}) == 4
    assert ("--quick" in calls[-1]) is quick
    assert all("--quick" not in command for command in calls[:-1])
    with pytest.raises(FileExistsError):
        run_all.run_jobs(jobs, output=output, resources={}, environment={}, platform="gpu")
    assert len(calls) == 4  # Existing PASS records must not replace a new execution.


def test_co_resource_requires_states_transitions_partition_and_definition(tmp_path):
    database = tmp_path / "12C-16O" / "Li2015"
    database.mkdir(parents=True)
    job = run_all.Job("retrieval", "retrieval.py", resources=("co",))
    values = {"co": str(database)}
    assert len(run_all.resource_errors(job, values)) == 4
    for suffix in (".def", ".pf", ".states.bz2"):
        (database / ("12C-16O__Li2015" + suffix)).write_bytes(b"local input")
    assert len(run_all.resource_errors(job, values)) == 1
    (database / "12C-16O__Li2015.trans.bz2").write_bytes(b"local input")
    assert run_all.resource_errors(job, values) == []


@pytest.mark.parametrize("backend", ["cpu", "initialization_error"])
def test_worker_records_gpu_backend_failure_before_running_script(tmp_path, monkeypatch, backend):
    def default_backend():
        if backend == "initialization_error":
            raise RuntimeError("GPU initialization failed")
        return backend

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(default_backend=default_backend, devices=lambda: ["cpu:0"]))
    script = tmp_path / "must_not_execute.py"
    marker = tmp_path / "ran"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    report = tmp_path / "worker.json"
    assert run_all.execute_worker(str(script), (), report, "gpu") == 1
    result = json.loads(report.read_text())
    assert result["status"] == "FAIL"
    assert "error" in result
    assert not marker.exists()


def test_actual_dry_run_does_not_import_jax_or_execute_jobs(tmp_path):
    output = tmp_path / "must_not_create"
    code = (
        "import sys; from benchmarks.documented_examples.run_all import main; "
        f"assert main(['--dry-run', '--output-directory', {str(output)!r}]) == 0; "
        "assert 'jax' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=run_all.ROOT,
                               env=dict(os.environ), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert len(report["jobs"]) == len(run_all.build_jobs())
    assert not output.exists()
