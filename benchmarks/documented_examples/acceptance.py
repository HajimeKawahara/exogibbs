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
                          (4.62, "metal_present"), (4.63, "metal_absent")]:
                errors.append("The documented four-point metal-selection reference is incomplete or changed.")
            if any(case.get("selection", {}).get("metal_amount_mol") != 0.
                   for case in cases if case.get("selection", {}).get("status") == "metal_absent"):
                errors.append("An accepted absent metal phase must have exactly zero amount.")
            for case in cases:
                selection = case.get("selection", {})
                if selection.get("status") != "metal_present":
                    continue
                result = selection.get("result", {})
                if result.get("accepted") is not True or not result.get("composition_constraints"):
                    errors.append("Present-metal controls require fresh constrained KKT audits.")
        elif job.name == "metal_bse_phase_selection":
            import numpy as np

            report = json.loads((directory / "selection.json").read_text())
            cases = report.get("cases", ())
            if len(cases) != 1:
                errors.append("The fresh BSE phase-selection control is incomplete.")
            else:
                case = cases[0]
                selection = case.get("selection", {})
                result = selection.get("result") or {}
                insertion = selection.get("insertion") or {}
                composition = np.asarray(selection.get("metal_composition"), dtype=float)
                lower = np.asarray(report.get("metal_domain", {}).get("lower", ()))
                upper = np.asarray(report.get("metal_domain", {}).get("upper", ()))
                amounts = np.asarray(result.get("component_amounts_mol", ()))
                residual = np.asarray(result.get("constrained_kkt_residual_rt", ()), dtype=float)
                if (result.get("accepted") is not True or insertion.get("minimum_certified") is not True
                        or selection.get("metal_amount_mol", 0.) <= 0
                        or lower.ndim != 1 or not lower.size or upper.shape != lower.shape
                        or composition.shape != lower.shape or not np.all(np.isfinite(composition))
                        or abs(composition.sum() - 1.) >= 1e-10
                        or np.any(composition < lower - 1e-10) or np.any(composition > upper + 1e-10)
                        or residual.shape != amounts.shape or not amounts.size
                        or not np.all(np.isfinite(residual[amounts > 0]))
                        or np.max(np.abs(residual[amounts > 0]), initial=0.) >= 1e-8):
                    errors.append("The constrained BSE local minimum did not pass numerical acceptance.")
                if (case.get("metadata", {}).get("numerical_execution", {}).get("native_melt_calls", 0) < 1
                        or case.get("independent_audit", {}).get("maximum_element_error", float("inf")) >= 1e-9):
                    errors.append("The BSE control requires fresh native evaluations and independent atom reconstruction.")
                if (selection.get("status") != "unresolved"
                        or selection.get("reasons") != ["Host global stability is not established."]
                        or report.get("scientific_acceptance") != {"M2_A": "pending", "M2_B": "pending"}):
                    errors.append("Local BSE closure cannot establish missing global host stability or calibration.")
        elif job.name in {"metal_m2_retained_suppressed", "metal_m2_retained_select"}:
            import numpy as np

            report = json.loads((directory / "contact.json").read_text())
            catalogs = report.get("catalogs", ())
            contact = catalogs[0].get("expanded_contact", {}) if len(catalogs) == 1 else {}
            gas = contact.get("gas_contact", {})
            residual = np.asarray(gas.get("log_pressure_residual", ()), dtype=float)
            cloud = np.asarray(contact.get("cloud_element_relative_residual", ()), dtype=float)
            if (report.get("numerical_diagnostics_completed") is not True
                    or report.get("matched_catalog") != "expanded_condensed"
                    or report.get("source_result", {}).get("accepted") is not True
                    or report.get("source_internal_result", {}).get("accepted") is not True
                    or report.get("source_metadata", {}).get("numerical_execution", {}).get("native_melt_calls", 0) < 1
                    or contact.get("accepted") is not True
                    or contact.get("common_standards", {}).get("accepted") is not True
                    or contact.get("source_atmosphere_audit", {}).get("accepted") is not True
                    or contact.get("upper_atmosphere_audit", {}).get("accepted") is not True
                    or gas.get("accepted") is not True
                    or residual.shape != (35,) or not np.all(np.isfinite(residual))
                    or np.max(np.abs(residual), initial=0.) >= 1e-8
                    or cloud.shape != (7,) or not np.all(np.isfinite(cloud))
                    or np.max(np.abs(cloud), initial=0.) >= 1e-9
                    or len(report.get("record", {}).get("retained_condensate_components", {})) != 26):
                errors.append("Finite retained-atmosphere contact requires all gases, clouds and fresh source audits.")
            if (report.get("scientific_acceptance") != {"M2_A": "pending", "M2_B": "pending", "M2_C": "pending"}
                    or (job.name.endswith("select") and report.get("metal_selection", {}).get("status") != "unresolved")):
                errors.append("Local expanded contact cannot establish missing material or global host stability.")
        elif job.name == "metal_m2_contact":
            report = json.loads((directory / "contact.json").read_text())
            if (report.get("numerical_diagnostics_completed") is not True
                    or report.get("shared_contact_accepted") is not True
                    or report.get("source_result", {}).get("accepted") is not True
                    or report.get("source_metadata", {}).get("numerical_execution", {}).get("native_melt_calls", 0) < 1):
                errors.append("Fresh native common-gas contact control did not pass.")
            catalogs = report.get("catalogs", ())
            if ([item.get("catalog") for item in catalogs] != ["shared_gas", "expanded_gas", "expanded_condensed"]
                    or any(item.get("parcel", {}).get("accepted") is not True for item in catalogs)
                    or report.get("common_standards", {}).get("accepted") is not True
                    or report.get("source_gas_audit", {}).get("accepted") is not True
                    or not catalogs or catalogs[0].get("contact", {}).get("accepted") is not True):
                errors.append("Matched-gas acceptance and expanded-catalog diagnostics are incomplete.")
            if report.get("scientific_acceptance") != {"M2_A": "pending", "M2_B": "pending", "M2_C": "pending"}:
                errors.append("The contact control does not establish material or global acceptance.")
        elif job.name == "metal_standards_audit":
            report = json.loads((directory / "standards_audit.json").read_text())
            comparisons = report.get("gas_standard_comparisons", ())
            if (report.get("numerical_audit_completed") is not True
                    or [item.get("temperature_K") for item in comparisons] != [2173.15, 2350.]
                    or any(item.get("independent_reaction_rank") != 4 for item in comparisons)):
                errors.append("Independent gas-standard audits are incomplete.")
            native = report.get("native_standard_comparison") or {}
            if native.get("native_evaluations") != 1 or not native.get("standard_comparisons"):
                errors.append("Fresh native MELTS standards were not evaluated.")
            if (report.get("scientific_acceptance") != {"M2_A": "pending", "M2_B": "pending"}
                    or native.get("cross_phase_standards_accepted") is not False):
                errors.append("The diagnostic audit cannot establish calibrated common standards.")
        elif job.name == "metal_archive_revalidation":
            report = json.loads((directory / "revalidated.json").read_text())
            for name in ("melts_present", "melts_absent"):
                state = report.get("cases", {}).get(name, {})
                if state.get("chemistry") != "recomputed" or state.get("accepted") is not True:
                    errors.append(f"Fresh native chemistry was not evaluated: {name}.")
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"Invalid acceptance artifact: {error}")
    return errors
