"""Run all documented executable examples, including external comparisons and NUTS.

Each job runs in a fresh process with its own artifacts and log. Missing inputs
and numerical failures remain failures in the final report; no PASS is reused.
Use --dry-run to inspect commands without importing JAX or executing examples.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
ALIASES = {
    "examples/comparisons/comparison_with_fastchem.py": "fastchem4_gas",
    "examples/comparisons/comparison_with_fastchem_extended.py": "fastchem4_gas",
    "examples/comparisons/comparison_with_fastchem_cond.py": "fastchem4_validation",
    "results/subneptune_taxonomy/20260911/verify.py": "metal_archive_revalidation",
}


@dataclass(frozen=True)
class Job:
    """One real example entry point and its required output artifacts."""

    name: str
    script: str
    arguments: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


def build_jobs() -> tuple[Job, ...]:
    """Cover public example scripts; aliases share their canonical execution."""
    jobs = []
    comparison = "examples/comparisons/"
    for name, script, flag, resources in (
        ("fastchem4_gas", "comparison_with_fastchem4_gas", "--output", ("fastchem",)),
        ("fastchem_initializer", "comparison_with_fastchem_initializer", "--output", ("fastchem",)),
        ("hsystem", "comparison_with_hsystem", "--output", ()),
        ("hcosystem", "comparison_with_hcosystem", "--output", ()),
        ("ykcode", "comparison_with_ykcode", "--output", ()),
        ("visscher_2006_morley_2012", "comparison_with_visscher_2006_na2s_morley_2012_kcl", "--figure", ("fastchem",)),
        ("visscher_2010", "comparison_with_visscher_2010_forsterite_enstatite", "--figure", ()),
        ("fe_fes_rainout", "demo_fe_fes_rainout", "--figure", ()),
        ("rocky_raccoon_trace_mg", "demo_rocky_raccoon_trace_mg", "--output", ()),
    ):
        artifact = "audit.json" if name == "rocky_raccoon_trace_mg" else "comparison.png"
        arguments = (flag, "{output}/" + artifact)
        if "fastchem" in resources:
            arguments += ("--fastchem-executable", "{fastchem}")
        jobs.append(Job(name, comparison + script + ".py", arguments, resources, (artifact,)))
    for profile in ("validation", "l-dwarf"):
        jobs.append(Job(
            "fastchem4_" + profile.replace("-", "_"),
            comparison + "comparison_with_fastchem4_condensates.py",
            ("--profile", profile, "--fastchem-executable", "{fastchem}", "--output", "{output}/comparison.png"),
            ("fastchem",), ("comparison.png",),
        ))
    jobs.append(Job(
        "fastchem4_production_comparison", "benchmarks/fastchem4/run_production_comparison.py",
        ("--fastchem-executable", "{fastchem}", "--fastchem-source-root", "{fastchem_source}",
         "--fastchem-version-label", "4.0.3 (ae67cbd)", "--jax-platform", "{platform}",
         "--point", "1800,0.1", "--point", "1600,0.1", "--point", "1400,0.1", "--point", "1200,0.1",
         "--output", "{output}/comparison.json"),
        ("fastchem", "fastchem_source"), ("comparison.json", "comparison.md", "comparison.preflight.json"),
    ))
    for name, script, archive in (
        ("ito_one_grid", "comparison_with_ito_2025.py", "--checkpoint"),
        ("ito_rainout", "comparison_with_ito_2025_rainout.py", "--archive"),
    ):
        arguments = (
            "--fastchem-executable", "{fastchem}", "--input", "{ito}",
            "--figure", "{output}/comparison.png", "--table", "{output}/layers.csv",
            "--summary", "{output}/comparison.json", archive, "{output}/layers.npz",
        )
        if name == "ito_one_grid":
            arguments += ("--no-resume",)
        jobs.append(Job(name, comparison + script, arguments, ("fastchem", "ito"),
                        ("comparison.png", "layers.csv", "comparison.json", "layers.npz")))
    for script in sorted((ROOT / "examples/condensates_curated_demo").glob("demo_*.py")):
        jobs.append(Job(
            script.stem, "benchmarks/documented_examples/check_curated.py",
            ("--script", str(script), "--output-directory", "{output}"),
            artifacts=("audit.json", script.stem + ".png"),
        ))
    jobs.append(Job(
        "exoeos_pure_fugacity", "benchmarks/documented_examples/check_exoeos.py",
        ("--output-directory", "{output}"), ("exoeos",),
        ("exoeos_audit.json", "exoeos_pure_fugacity.png"),
    ))
    metal = "examples/metal_silicate/"
    jobs.append(Job("metal_thermochemistry", metal + "reference.py"))
    jobs.append(Job(
        "metal_independent_reference", metal + "generate_equilibrium_reference.py",
        ("--output", "{output}/equilibrium_reference.json"), artifacts=("equilibrium_reference.json",),
    ))
    jobs.append(Job(
        "metal_source_extraction", metal + "extract_reference.py",
        ("--gce-checkout", "{gce}", "--check"), ("gce",),
    ))
    for kind in ("source_full", "completed_dry"):
        for temperature in (2350, 3000):
            case = f"{kind}_{temperature}"
            jobs.append(Job(
                "metal_" + case, metal + "native.py", ("--case", case),
                ("exoeos",) if kind == "completed_dry" else (),
            ))
    for temperature in (2350, 3000):
        jobs.append(Job(
            f"metal_gas_exchange_{temperature}", metal + "gas_exchange.py",
            ("--case", f"source_full_{temperature}", "--output", "{output}/equilibrium.json"),
            artifacts=("equilibrium.json",),
        ))
    for scale in (1, 0):
        jobs.append(Job("metal_hydrogen_" + str(scale), metal + "hydrogen.py",
                        ("--hydrogen-scale", str(scale)), ("exoeos",)))
    for name in ("sulfur_source", "sulfide", "cns_inventory"):
        jobs.append(Job("metal_" + name, metal + name + ".py"))
    jobs.append(Job(
        "metal_sulfur_extraction", metal + "extract_sulfur_reference.py",
        ("{gce}", "--output", "{output}/sulfur_reference.json"), ("gce",),
        ("sulfur_reference.json",),
    ))
    jobs.append(Job(
        "metal_m1_chemistry", metal + "m1_chemistry.py",
        ("--pressure-bar", "100", "--point", "1400", "0.1", "--output", "{output}/chemistry.json"),
        artifacts=("chemistry.json",),
    ))
    jobs.append(Job(
        "metal_m1_boundary", metal + "m1_boundary_validation.py",
        ("--output", "{output}/boundary.json"), artifacts=("boundary.json",),
    ))
    native_arguments = ("--exoeos-checkout", "{exoeos_checkout}", "--runtime", "{melts_runtime}",
                        "--python", "{melts_python}", "--output", "{output}/equilibrium.json")
    native_resources = ("exoeos", "exoeos_checkout", "melts_runtime", "melts_python")
    for branch in ("present", "absent"):
        jobs.append(Job(
            "metal_melts_" + branch, metal + "run_melts_reference.py",
            native_arguments + (("--metal-absent",) if branch == "absent" else ()),
            native_resources, ("equilibrium.json",),
        ))
    jobs.append(Job(
        "metal_common_gibbs", metal + "run_bse_common_gibbs.py",
        native_arguments + ("--inventory", "{bse_inventory}", "--metal-absent"),
        native_resources + ("bse_inventory",), ("equilibrium.json",),
    ))
    jobs.append(Job(
        "metal_phase_selection", metal + "run_metal_selection.py",
        ("--exoeos-checkout", "{exoeos_checkout}", "--output", "{output}/reference.json"),
        ("exoeos", "exoeos_checkout"), ("reference.json",),
    ))
    jobs.append(Job(
        "metal_bse_phase_selection", metal + "run_metal_selection.py",
        ("--exoeos-checkout", "{exoeos_checkout}", "--bse-inventory", "{bse_inventory}",
         "--runtime", "{melts_runtime}", "--python", "{melts_python}",
         "--gas-model", "m1_shared", "--output", "{output}/selection.json"),
        native_resources + ("bse_inventory",), ("selection.json",),
    ))
    jobs.append(Job(
        "metal_standards_audit", metal + "m2_standards_audit.py",
        ("--inventory", "{bse_inventory}", "--exoeos-checkout", "{exoeos_checkout}",
         "--runtime", "{melts_runtime}", "--python", "{melts_python}",
         "--output", "{output}/standards_audit.json"),
        native_resources + ("bse_inventory",), ("standards_audit.json",),
    ))
    jobs.append(Job(
        "metal_m2_contact", metal + "run_m2_contact.py",
        ("--inventory", "{bse_inventory}", "--exoeos-checkout", "{exoeos_checkout}",
         "--runtime", "{melts_runtime}", "--python", "{melts_python}",
         "--output", "{output}/contact.json"),
        native_resources + ("bse_inventory",), ("contact.json",),
    ))
    for metal_mode in ("suppressed", "select"):
        jobs.append(Job(
            "metal_m2_retained_" + metal_mode, metal + "run_m2_contact.py",
            ("--inventory", "{bse_inventory}", "--exoeos-checkout", "{exoeos_checkout}",
             "--runtime", "{melts_runtime}", "--python", "{melts_python}",
             "--gas-model", "m1_retained", "--metal-mode", metal_mode,
             "--output", "{output}/contact.json"),
            native_resources + ("bse_inventory",), ("contact.json",),
        ))
    jobs.append(Job(
        "metal_m2_host_stability", metal + "run_m2_host_stability.py",
        native_arguments + ("--source-json", str(ROOT / "results/m2_expanded_contact/20260924/contact.json")),
        native_resources, ("equilibrium.json",),
    ))
    jobs.append(Job(
        "metal_m2_omitted_gas", metal + "run_m2_omitted_gas.py",
        ("--source", str(ROOT / "results/m2_expanded_contact/20260924/contact.json"),
         "--mole-fraction-target", "1e-8", "--output", "{output}/screen.json"),
        artifacts=("screen.json",),
    ))
    jobs.append(Job(
        "metal_archive_revalidation", metal + "revalidate_archive.py",
        ("--archive", str(ROOT / "results/subneptune_taxonomy/20260911"),
         "--exoeos-checkout", "{exoeos_checkout}", "--runtime", "{melts_runtime}",
         "--worker-python", "{melts_python}", "--output", "{output}/revalidated.json"),
        native_resources, ("revalidated.json",),
    ))
    for case in ("gas_no_grid", "gas_grid", "condensate_fixed_support", "condensate_grid"):
        jobs.append(Job(
            "retrieval_" + case, f"examples/retrievals/exojax_nuts_{case}.py",
            ("--co-database", "{co}", "--output-dir", "{output}", "--no-progress-bar"),
            ("co",), ("run_status.json", "run_summary.json", "posterior_samples.npz"),
        ))
    from benchmarks.documented_examples.check_inline import CASES

    for case in CASES:
        jobs.append(Job(
            "inline_" + case, "benchmarks/documented_examples/check_inline.py",
            ("--case", case, "--output-directory", "{output}"),
            ("exoeos",) if case == "ideal_solution_adapter" else (), ("inline_audit.json",),
        ))
    jobs.append(Job(
        "inline_ja_magma_gas_interface", "benchmarks/documented_examples/check_inline.py",
        ("--case", "ja_magma_gas_interface", "--output-directory", "{output}"),
        ("japanese_docs",), ("inline_audit.json",),
    ))
    return tuple(jobs)


def resource_errors(job: Job, values: dict[str, str]) -> list[str]:
    """Check local inputs before launching a potentially long calculation."""
    errors = []
    for name in job.resources:
        value = values.get(name, "")
        if not value:
            errors.append(f"Missing {name} input; supply its command-line option.")
            continue
        path = Path(value)
        if name in {"fastchem", "melts_python"} and (not path.is_file() or not os.access(path, os.X_OK)):
            errors.append(f"{name} executable is unavailable: {path}")
        elif name in {"ito", "bse_inventory"} and not path.is_file():
            errors.append(f"{name} file is unavailable: {path}")
        elif name not in {"fastchem", "melts_python", "ito", "bse_inventory"} and not path.is_dir():
            errors.append(f"{name} directory is unavailable: {path}")
        elif name == "exoeos" and not (path / "exoeos/__init__.py").is_file():
            errors.append(f"ExoEOS source is unavailable: {path}")
        elif name == "melts_runtime" and not (path / "libalphamelts.so").is_file():
            errors.append(f"MELTS native library is unavailable: {path}")
        elif name == "japanese_docs" and not (path / "ja/magma_gas_interface.tex").is_file():
            errors.append(f"Japanese documentation checkout is unavailable: {path}")
        elif name == "co":
            prefix = f"{path.parent.name}__{path.name}"
            for pattern in (prefix + ".def", prefix + ".pf", prefix + ".states*", prefix + ".trans*"):
                if not any(path.glob(pattern)):
                    errors.append(f"CO database is missing {pattern}")
    return errors


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute_worker(script: str, arguments: Sequence[str], report_path: Path, platform: str) -> int:
    """Execute an actual script after verifying the requested JAX backend."""
    started = time.monotonic()
    report: dict[str, Any] = {"status": "FAIL", "requested_platform": platform}
    code = 1
    try:
        import jax

        backend = jax.default_backend()
        report.update(backend=backend, devices=[str(device) for device in jax.devices()])
        print(f"JAX backend: {backend}; devices: {report['devices']}", flush=True)
        if backend != platform:
            raise RuntimeError(f"Requested {platform}, but JAX initialized {backend}.")
        from benchmarks.documented_examples.worker import _checkout_package_path

        report["exogibbs_source"] = _checkout_package_path(ROOT)
        report["jax_version"] = jax.__version__
        report["float64"] = bool(jax.config.x64_enabled)
        if not report["float64"]:
            raise RuntimeError("Documentation acceptance requires float64.")
        path = (ROOT / script).resolve()
        sys.path.insert(0, str(path.parent))
        sys.argv = [str(path), *arguments]
        try:
            runpy.run_path(str(path), run_name="__main__")
            code = 0
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
            if code:
                report["error"] = str(error)
        if jax.default_backend() != platform:
            raise RuntimeError("The example changed the requested JAX backend.")
        provider = sys.modules.get("exoeos")
        if provider is not None:
            actual = Path(provider.__file__).resolve()
            expected = Path(os.environ["EXOGIBBS_EXOEOS_SOURCE"]) / "exoeos/__init__.py"
            report["exoeos_source"] = str(actual)
            if actual != expected.resolve():
                raise RuntimeError(f"ExoEOS import differs from the recorded checkout: {actual}.")
        report["numerical_environment"] = {key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_DISABLE_MOST_OPTIMIZATIONS", "XLA_FLAGS",
        )}
        report["status"] = "PASS" if code == 0 else "FAIL"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        code = 1
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["exit_code"] = code
        write_json(report_path, report)
    return code


def run_jobs(
    jobs: Sequence[Job], *, output: Path, resources: dict[str, str],
    environment: dict[str, str], platform: str, retrieval_quick: bool = False,
    source_provenance: Optional[dict] = None, coverage: Optional[dict] = None,
    full_selection: bool = True,
) -> int:
    """Run every selected job and retain failures without silently skipping jobs."""
    output.mkdir(parents=True, exist_ok=False)
    records = [{"name": job.name, "status": "PENDING"} for job in jobs]
    summary: dict[str, Any] = {
        "status": "RUNNING", "platform": platform, "retrieval_quick": retrieval_quick,
        "aliases": ALIASES, "jobs": records, "provenance": source_provenance,
        "coverage": coverage, "full_selection": full_selection,
        "full_acceptance": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", summary)
    for index, (job, record) in enumerate(zip(jobs, records), 1):
        directory = output / job.name
        directory.mkdir()
        values = {**resources, "output": str(directory), "platform": platform}
        arguments = [argument.format_map(values) for argument in job.arguments]
        if job.name.startswith("retrieval_") and retrieval_quick:
            arguments.append("--quick")
        errors = resource_errors(job, resources)
        command = [
            sys.executable, "-m", "benchmarks.documented_examples.run_all",
            "--platform", platform, "--worker-report", str(directory / "worker.json"),
            "--worker", job.script, *arguments,
        ]
        record.update(command=command, log=str(directory / "run.log"))
        print(f"[{index}/{len(jobs)}] {job.name}", flush=True)
        if errors:
            record.update(status="UNAVAILABLE", errors=errors)
        else:
            record["status"] = "RUNNING"
            write_json(output / "summary.json", summary)
            with (directory / "run.log").open("w") as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    exit_code = completed.returncode
                except OSError as error:
                    log.write(f"Worker launch failed: {error}\n")
                    exit_code = 1
            worker_path = directory / "worker.json"
            try:
                worker = json.loads(worker_path.read_text())
            except (OSError, ValueError):
                worker = {"status": "FAIL", "error": "Worker report was not produced."}
            missing = [name for name in job.artifacts
                       if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
            from benchmarks.documented_examples.acceptance import artifact_errors

            audit_errors = artifact_errors(job, directory, retrieval_quick=retrieval_quick)
            passed = exit_code == 0 and worker.get("status") == "PASS" and not missing and not audit_errors
            record.update(status="PASS" if passed else "FAIL", exit_code=exit_code,
                          worker=worker, missing_artifacts=missing, artifact_errors=audit_errors)
        print(f"  {record['status']}: {record.get('errors', record['log'])}", flush=True)
        write_json(output / "summary.json", summary)
    passed = bool(records) and all(item["status"] == "PASS" for item in records)
    passed = passed and (coverage is None or coverage["accepted"])
    summary["status"] = ("SMOKE_PASS" if retrieval_quick else "PASS") if passed else "FAIL"
    summary["full_acceptance"] = bool(passed and full_selection and not retrieval_quick)
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print(f"{summary['status']}: {output / 'summary.json'}", flush=True)
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--fastchem-executable", default=os.environ.get("EXOGIBBS_FASTCHEM_EXECUTABLE"))
    parser.add_argument("--fastchem-source-root", default=os.environ.get("EXOGIBBS_FASTCHEM_SOURCE_ROOT", ROOT / "FastChem"))
    parser.add_argument("--co-database", default=os.environ.get("EXOJAX_CO_DATABASE"))
    parser.add_argument("--gce-checkout", default=os.environ.get("EXOGIBBS_GCE_CHECKOUT"))
    parser.add_argument("--ito-input", default=os.environ.get("EXOGIBBS_ITO_INPUT", str(ROOT / "external_data/Ito_2025.xlsx")))
    parser.add_argument("--exoeos-source", type=Path, default=os.environ.get("EXOGIBBS_EXOEOS_SOURCE", ROOT.parent / "exoeos/src"))
    parser.add_argument("--melts-runtime", default=os.environ.get("EXOGIBBS_MELTS_RUNTIME"))
    parser.add_argument("--melts-python", default=os.environ.get("EXOGIBBS_MELTS_PYTHON"))
    parser.add_argument("--bse-inventory", default=os.environ.get("EXOGIBBS_BSE_INVENTORY"))
    parser.add_argument("--japanese-docs", type=Path, default=os.environ.get("EXOGIBBS_JAPANESE_DOCS", ROOT / "doc_ExoGibbs"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--retrieval-quick", action="store_true", help="Use each tutorial's quick NUTS settings; other cases remain full size.")
    parser.add_argument("--dry-run", action="store_true", help="List selected jobs and arguments; execute nothing.")
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return execute_worker(args.worker[0], args.worker[1:], args.worker_report, args.platform)
    jobs = build_jobs()
    from benchmarks.documented_examples.acceptance import coverage_report, provenance

    coverage = coverage_report(ROOT, args.japanese_docs.resolve(), jobs, ALIASES)
    if args.case:
        unknown = set(args.case) - {job.name for job in jobs}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        jobs = tuple(job for job in jobs if job.name in args.case)
    if args.dry_run:
        print(json.dumps({"jobs": [asdict(job) for job in jobs], "aliases": ALIASES,
                          "coverage": coverage}, indent=2))
        return 0
    resources = {name: str(Path(value).expanduser().absolute()) if value else ""
                 for name, value in (("fastchem", args.fastchem_executable), ("co", args.co_database),
                                     ("fastchem_source", args.fastchem_source_root),
                                     ("gce", args.gce_checkout), ("ito", args.ito_input),
                                     ("exoeos", args.exoeos_source),
                                     ("exoeos_checkout", args.exoeos_source.resolve().parent),
                                     ("melts_runtime", args.melts_runtime), ("melts_python", args.melts_python),
                                     ("bse_inventory", args.bse_inventory), ("japanese_docs", args.japanese_docs))}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_directory or ROOT / "results/all_documented_examples" / f"{stamp}_{os.getpid()}").resolve()
    environment = dict(os.environ)
    paths = [str(ROOT / "src"), str(ROOT)]
    if args.exoeos_source.is_dir():
        paths.append(str(args.exoeos_source.resolve()))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment.update(
        PYTHONPATH=os.pathsep.join(paths), JAX_PLATFORMS="cuda" if args.platform == "gpu" else "cpu",
        JAX_PLATFORM_NAME="cuda" if args.platform == "gpu" else "cpu", JAX_ENABLE_X64="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg", PYTHONHASHSEED="0",
        NUMBA_DISABLE_JIT="1",
        EXOGIBBS_JAPANESE_DOCS=resources["japanese_docs"],
        EXOGIBBS_EXOEOS_SOURCE=resources["exoeos"],
        JAX_ENABLE_COMPILATION_CACHE="false",
    )
    return run_jobs(jobs, output=output, resources=resources, environment=environment,
                    platform=args.platform, retrieval_quick=args.retrieval_quick,
                    source_provenance=provenance(ROOT, resources), coverage=coverage,
                    full_selection=not args.case)


if __name__ == "__main__":
    raise SystemExit(main())
