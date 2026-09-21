"""Fresh numerical acceptance cannot be replaced by a successful process exit."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.documented_examples import acceptance, run_all
from benchmarks.documented_examples.check_inline import extract_blocks


@pytest.mark.parametrize("failure", ["preflight", "short", "nonfinite", "diagnostics", "missing"])
def test_incomplete_retrieval_cannot_pass(tmp_path, failure):
    job = run_all.Job("retrieval_test", "example.py")
    (tmp_path / "run_status.json").write_text(json.dumps({
        "mode": "preflight" if failure == "preflight" else "sampling", "state": "complete",
    }))
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "sampling_diagnostics": {"passed": failure != "diagnostics"},
        "run_config": {"num_warmup": 500, "num_samples": 8 if failure == "short" else 1000},
    }))
    if failure != "missing":
        samples = np.zeros(8 if failure == "short" else 1000)
        if failure == "nonfinite":
            samples[0] = np.nan
        np.savez(tmp_path / "posterior_samples.npz", parameter=samples)
    assert acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)


def test_archive_balance_only_is_not_fresh_native_acceptance(tmp_path):
    job = run_all.Job("metal_archive_revalidation", "revalidate_archive.py")
    report = {"cases": {name: {"chemistry": "not_run"} for name in ("melts_present", "melts_absent")}}
    path = tmp_path / "revalidated.json"
    path.write_text(json.dumps(report))
    assert len(acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)) == 2
    for state in report["cases"].values():
        state["chemistry"] = "recomputed"
        state["accepted"] = True
    path.write_text(json.dumps(report))
    assert not acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)


@pytest.mark.parametrize("failure", ["summary", "missing_case", "unresolved", "metal_floor"])
def test_metal_reference_successful_exit_cannot_hide_failed_controls(tmp_path, failure):
    job = run_all.Job("metal_phase_selection", "run_metal_selection.py")
    source = run_all.ROOT / "results/m2_phase_selection/reference_20260920.json"
    report = json.loads(source.read_text())
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(report))
    assert not acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)
    if failure == "summary":
        report["numerical_reference_acceptance"]["unshifted_coexistence_recovered"] = False
    elif failure == "missing_case":
        report["cases"].pop(1)
    elif failure == "unresolved":
        report["cases"][2]["selection"]["status"] = "metal_absent"
    else:
        report["cases"][3]["selection"]["metal_amount_mol"] = 1e-30
    path.write_text(json.dumps(report))
    assert acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)


def test_coverage_detects_new_public_and_unavailable_japanese_commands(tmp_path):
    example = tmp_path / "examples/new.py"
    example.parent.mkdir()
    example.write_text("if __name__ == '__main__':\n    print('new')\n")
    docs = tmp_path / "japanese"
    (docs / "ja").mkdir(parents=True)
    (docs / "main_ja.tex").write_text("Japanese manual")
    (docs / "ja/example.tex").write_text("python examples/missing.py\n")
    report = acceptance.coverage_report(tmp_path, docs, (), {})
    assert not report["accepted"]
    assert any("new.py" in error for error in report["errors"])
    assert any("missing.py" in error for error in report["errors"])


def test_source_fingerprint_changes_for_uncommitted_content(tmp_path):
    path = tmp_path / "example.py"
    path.write_text("print(1)\n")
    first = acceptance._fingerprint([path], tmp_path)
    path.write_text("print(2)\n")
    assert first["sha256"] != acceptance._fingerprint([path], tmp_path)["sha256"]


def test_japanese_block_extraction_preserves_actual_code(tmp_path):
    path = tmp_path / "example.tex"
    path.write_text("before\\begin{verbatim}\nprint('original')\n\\end{verbatim}after")
    assert extract_blocks(path) == ("print('original')\n",)


def test_native_jobs_require_all_external_resources():
    jobs = {job.name: job for job in run_all.build_jobs()}
    required = {"exoeos", "exoeos_checkout", "melts_runtime", "melts_python"}
    for name in ("metal_common_gibbs", "metal_melts_present", "metal_melts_absent", "metal_archive_revalidation", "metal_standards_audit"):
        assert required <= set(jobs[name].resources)
        assert run_all.resource_errors(jobs[name], {})
    assert "bse_inventory" in jobs["metal_common_gibbs"].resources
    assert "bse_inventory" in jobs["metal_standards_audit"].resources


@pytest.mark.parametrize("failure", ["temperature", "reaction", "native", "calibration"])
def test_standards_diagnostic_requires_fresh_evidence_and_retains_scientific_gates(tmp_path, failure):
    job = run_all.Job("metal_standards_audit", "m2_standards_audit.py")
    report = {"numerical_audit_completed": True,
              "gas_standard_comparisons": [{"temperature_K": value, "independent_reaction_rank": 4}
                                           for value in (2173.15, 2350.)],
              "native_standard_comparison": {"native_evaluations": 1,
                                              "standard_comparisons": [{"native_mu0_rt": -1.}],
                                              "cross_phase_standards_accepted": False},
              "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"}}
    path = tmp_path / "standards_audit.json"
    path.write_text(json.dumps(report))
    assert not acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)
    if failure == "temperature":
        report["gas_standard_comparisons"].pop()
    elif failure == "reaction":
        report["gas_standard_comparisons"][0]["independent_reaction_rank"] = 3
    elif failure == "native":
        report["native_standard_comparison"]["native_evaluations"] = 0
    else:
        report["scientific_acceptance"]["M2_A"] = "accepted"
    path.write_text(json.dumps(report))
    assert acceptance.artifact_errors(job, tmp_path, retrieval_quick=False)


def test_quick_or_partial_run_cannot_claim_full_acceptance(tmp_path, monkeypatch):
    def successful_process(command, **kwargs):
        report = Path(command[command.index("--worker-report") + 1])
        report.write_text('{"status":"PASS"}')
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(run_all.subprocess, "run", successful_process)
    for quick, full in ((True, True), (False, False)):
        output = tmp_path / f"{quick}_{full}"
        assert run_all.run_jobs((run_all.Job("example", "example.py"),), output=output,
                                resources={}, environment={}, platform="gpu",
                                retrieval_quick=quick, full_selection=full) == 0
        assert json.loads((output / "summary.json").read_text())["full_acceptance"] is False
