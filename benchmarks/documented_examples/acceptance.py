"""Coverage, provenance, and artifact contracts for fresh documentation runs."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence


def _is_generated_document(path: Path, root: Path) -> bool:
    """Identify Sphinx outputs configured by documents/conf.py."""
    parts = path.relative_to(root).parts
    return (len(parts) > 1 and parts[0] == "documents"
            and parts[1] in {"_build", "examples", "exogibbs", "backreferences", "sg_execution_times.rst"})


def covered_scripts(jobs: Sequence[Any], aliases: dict[str, str], root: Path) -> set[str]:
    """Resolve audited wrappers to the original public example entry points."""
    scripts = {job.script for job in jobs} | set(aliases)
    for job in jobs:
        if job.script.endswith("check_curated.py"):
            scripts.add(str(Path(job.arguments[job.arguments.index("--script") + 1]).relative_to(root)))
        elif job.script.endswith("check_exoeos.py"):
            scripts.add("examples/plot_exoeos_pure_fugacity.py")
    return scripts


def coverage_report(root: Path, japanese_docs: Path, jobs: Sequence[Any], aliases: dict[str, str]) -> dict:
    """Fail visibly when a public or documented executable lacks a fresh job."""
    scripts = covered_scripts(jobs, aliases, root)
    public = set()
    for path in (root / "examples").rglob("*.py"):
        if any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
               for node in ast.parse(path.read_text()).body):
            public.add(str(path.relative_to(root)))
    documents = sorted(path for path in (root / "documents").rglob("*.rst")
                       if not _is_generated_document(path, root))
    documents += sorted((root / "examples/metal_silicate").glob("*.md"))
    documents += sorted((japanese_docs / "ja").glob("*.tex"))
    commands = []
    for path in documents:
        for match in re.finditer(r"\bpython(?:3)?\s+((?:examples|results)/[\w/.-]+\.py)", path.read_text()):
            script = match.group(1)
            commands.append({"document": str(path), "script": script, "covered": script in scripts,
                             "exists": (root / script).is_file()})
        for match in re.finditer(r"\bpython(?:3)?\s+-m\s+(benchmarks\.[\w.]+)", path.read_text()):
            module = match.group(1)
            script = module.replace(".", "/") + ".py"
            commands.append({"document": str(path), "module": module, "script": script,
                             "covered": script in scripts, "exists": (root / script).is_file()})
    errors = [f"Uncovered public script: {name}" for name in sorted(public - scripts)]
    errors += [f"Uncovered or unavailable documented script: {item['script']} ({item['document']})"
               for item in commands if not item["covered"] or not item["exists"]]
    if not (japanese_docs / "main_ja.tex").is_file():
        errors.append(f"Japanese documentation checkout is unavailable: {japanese_docs}")
    return {"public_scripts": sorted(public), "documented_commands": commands,
            "errors": errors, "accepted": not errors}


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _fingerprint(paths: Sequence[Path], root: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        count += 1
    return {"file_count": count, "sha256": digest.hexdigest()}


def provenance(root: Path, resources: dict[str, str]) -> dict:
    """Record actual source contents, including uncommitted files and providers."""
    records = {}
    for label, directory, suffixes in (
        ("exogibbs", root, {".py", ".json", ".rst", ".csh", ".md", ".dat", ".txt", ".npz", ".nc"}),
        ("japanese_docs", Path(resources["japanese_docs"]), {".tex", ".bib"}),
        ("exoeos", Path(resources["exoeos_checkout"]), {".py", ".json"}),
    ):
        paths = []
        roots = [directory / name for name in ("src", "examples", "benchmarks", "documents", "tests/unittests/examples/data")] if label == "exogibbs" else [directory]
        for source_root in roots:
            if source_root.is_dir():
                paths.extend(path for path in source_root.rglob("*")
                             if path.is_file() and path.suffix in suffixes
                             and not {".git", "__pycache__"}.intersection(path.parts)
                             and (label != "exogibbs" or not _is_generated_document(path, directory)))
        status = _git(directory, "status", "--porcelain") if directory.is_dir() else ""
        records[label] = {"path": str(directory), "commit": _git(directory, "rev-parse", "HEAD") if directory.is_dir() else None,
                          "dirty": bool(status), "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
                          **_fingerprint(paths, directory)}
    records["resources"] = {}
    for name, value in resources.items():
        path = Path(value) if value else None
        record = {"path": value, "exists": bool(path and path.exists())}
        if path and path.is_file():
            record.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        elif path and path.is_dir():
            record["commit"] = _git(path, "rev-parse", "HEAD") or None
            if name in {"co", "melts_runtime", "gce", "fastchem_source"}:
                inputs = [item for item in path.rglob("*") if item.is_file()
                          and not {".git", "__pycache__"}.intersection(item.parts)]
                record["contents"] = _fingerprint(inputs, path)
        records["resources"][name] = record
    return records


def artifact_errors(job: Any, directory: Path, *, retrieval_quick: bool) -> list[str]:
    """Require actual sampling and reject optional native-chemistry omissions."""
    errors = []
    try:
        if job.name == "fastchem4_production_comparison":
            report = json.loads((directory / "comparison.json").read_text())
            preflight = json.loads((directory / "comparison.preflight.json").read_text())
            if preflight.get("passed") is not True or report.get("preflight", {}).get("passed") is not True:
                errors.append("Pinned FastChem source/data preflight did not pass.")
            if report.get("summary", {}).get("comparison_completed") is not True:
                errors.append("Production comparison did not converge with finite comparison metrics.")
            contract = report.get("input_contract", {})
            if (contract.get("temperature_K") != [1800.0, 1600.0, 1400.0, 1200.0]
                    or contract.get("pressure_bar") != [0.1] * 4 or len(report.get("layers", ())) != 4):
                errors.append("The documented four-point production comparison is incomplete.")
        elif job.name.startswith("retrieval_"):
            status = json.loads((directory / "run_status.json").read_text())
            summary = json.loads((directory / "run_summary.json").read_text())
            if status.get("mode") != "sampling" or status.get("state") != "complete":
                errors.append("NUTS sampling did not complete.")
            if summary.get("sampling_diagnostics", {}).get("passed") is not True:
                errors.append("NUTS posterior diagnostics failed or are unavailable.")
            config = summary.get("run_config", {})
            if not retrieval_quick and (config.get("num_warmup", 0) < 500 or config.get("num_samples", 0) < 1000):
                errors.append("Full NUTS requires at least 500 warmup steps and 1000 samples.")
            import numpy as np

            with np.load(directory / "posterior_samples.npz", allow_pickle=False) as samples:
                if not samples.files or any(
                    samples[name].ndim == 0 or samples[name].shape[0] != config.get("num_samples")
                    or not np.all(np.isfinite(samples[name])) for name in samples.files
                ):
                    errors.append("Posterior samples are incomplete or nonfinite.")
        elif job.name == "metal_phase_selection":
            report = json.loads((directory / "reference.json").read_text())
            accepted = report.get("numerical_reference_acceptance", {})
            if any(accepted.get(name) is not True for name in (
                "unshifted_coexistence_recovered", "both_metal_presence_and_absence_certified",
            )):
                errors.append("Mixed-metal numerical reference acceptance did not pass.")
            cases = report.get("cases", ())
            actual = [(case.get("metal_standard_shift_rt"), case.get("selection", {}).get("status"))
                      for case in cases]
            if actual != [(0., "metal_present"), (.1, "metal_present"),
                          (4.62, "unresolved"), (4.63, "metal_absent")]:
                errors.append("The documented four-point metal-selection reference is incomplete or changed.")
            if any(case.get("selection", {}).get("metal_amount_mol") != 0.
                   for case in cases if case.get("selection", {}).get("status") == "metal_absent"):
                errors.append("An accepted absent metal phase must have exactly zero amount.")
        elif job.name == "metal_archive_revalidation":
            report = json.loads((directory / "revalidated.json").read_text())
            for name in ("melts_present", "melts_absent"):
                state = report.get("cases", {}).get(name, {})
                if state.get("chemistry") != "recomputed" or state.get("accepted") is not True:
                    errors.append(f"Fresh native chemistry was not evaluated: {name}.")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"Invalid acceptance artifact: {error}")
    return errors
