"""Run one documented example benchmark in an isolated JAX process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform as system_platform
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from benchmarks.documented_examples.manifest import CASES_BY_ID, get_case


SCHEMA = "exogibbs_documented_example_benchmark_v2"
OPTIMIZATION_MODES = ("default", "disable_most_optimizations")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def configure_jax_environment(platform: str, optimization_mode: str) -> None:
    """Set JAX options before importing JAX or any example module."""

    jax_platform = "cpu" if platform == "cpu" else "cuda"
    os.environ["JAX_PLATFORMS"] = jax_platform
    os.environ["JAX_PLATFORM_NAME"] = jax_platform
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_DISABLE_MOST_OPTIMIZATIONS"] = (
        "true" if optimization_mode == "disable_most_optimizations" else "false"
    )


def _revision(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            arguments,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("git", "status", "--short")
    diff = run("git", "diff", "--binary", "--", "src", "benchmarks", "examples")
    return {
        "commit": run("git", "rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_source_diff_sha256": hashlib.sha256(
            diff.encode("utf-8")
        ).hexdigest(),
    }


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    """Fingerprint every Python source used by the installed solver tree."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*.py")):
        payload = path.read_bytes()
        relative = path.relative_to(root)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        file_count += 1
        total_bytes += len(payload)
    return {
        "root": str(root),
        "python_file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _source_provenance(
    repository_root: Path,
    source_scripts: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Fingerprint example inputs and benchmark sources used by this worker."""

    benchmark_directory = Path(__file__).resolve().parent
    paths = [repository_root / source for source in source_scripts]
    paths.extend(sorted(benchmark_directory.glob("*.py")))
    paths.append(benchmark_directory / "README.md")
    records = []
    for path in dict.fromkeys(item.resolve() for item in paths):
        payload = path.read_bytes()
        try:
            relative = path.relative_to(repository_root.resolve())
        except ValueError:
            relative = path
        records.append(
            {
                "path": str(relative),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return tuple(records)


def _checkout_package_path(repository_root: Path) -> str:
    """Fail if this worker imported ExoGibbs from outside the checkout."""

    import exogibbs

    package_file = getattr(exogibbs, "__file__", None)
    if package_file is None:
        raise RuntimeError("The imported exogibbs package has no source path.")
    package_root = Path(package_file).resolve().parent
    expected_root = (repository_root / "src" / "exogibbs").resolve()
    if package_root != expected_root:
        raise RuntimeError(
            "Benchmark imported exogibbs from outside this checkout: "
            f"actual={package_root}, expected={expected_root}."
        )
    return str(package_root)


def _jax_environment(
    requested_platform: str,
    optimization_mode: str,
) -> dict[str, Any]:
    import jax
    import jaxlib

    config_names = (
        "jax_disable_most_optimizations",
        "jax_optimization_level",
        "jax_exec_time_optimization_effort",
        "jax_memory_fitting_level",
        "jax_memory_fitting_effort",
        "jax_enable_compilation_cache",
        "jax_enable_x64",
    )
    config_values = {
        name: jax.config.values[name]
        for name in config_names
        if name in jax.config.values
    }
    option_supported = "jax_disable_most_optimizations" in config_values
    requested_value = optimization_mode == "disable_most_optimizations"
    effective_value = config_values.get("jax_disable_most_optimizations")
    if requested_value and not option_supported:
        raise NotImplementedError(
            "This JAX version has no jax_disable_most_optimizations option."
        )
    if effective_value is not requested_value:
        raise RuntimeError(
            "jax_disable_most_optimizations did not take the requested value: "
            f"requested={requested_value!r}, effective={effective_value!r}."
        )
    if config_values.get("jax_enable_x64") is not True:
        raise RuntimeError("JAX x64 mode is not enabled.")
    if config_values.get("jax_enable_compilation_cache") is not False:
        raise RuntimeError("JAX persistent compilation cache is not disabled.")

    devices = jax.devices()
    backend = jax.default_backend()
    expected_backend = "cpu" if requested_platform == "cpu" else "gpu"
    if backend != expected_backend:
        raise RuntimeError(
            f"Requested {requested_platform!r}, but JAX selected {backend!r}."
        )
    return {
        "python_version": system_platform.python_version(),
        "python_executable": sys.executable,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "requested_platform": requested_platform,
        "backend": backend,
        "devices": tuple(
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
            }
            for device in devices
        ),
        "optimization": {
            "mode": optimization_mode,
            "option": "jax_disable_most_optimizations",
            "supported": option_supported,
            "requested_value": requested_value,
            "effective_value": effective_value,
        },
        "jax_config": config_values,
        "process_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "JAX_ENABLE_X64",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_DISABLE_MOST_OPTIMIZATIONS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
            )
        },
        "host": {
            "hostname": system_platform.node(),
            "system": system_platform.system(),
            "machine": system_platform.machine(),
            "processor": system_platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES_BY_ID), required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--optimization",
        choices=OPTIMIZATION_MODES,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-layers",
        type=int,
        default=None,
        help="Run only the first N output conditions; never use for baselines.",
    )
    parser.add_argument("--repetition", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke_layers is not None and args.smoke_layers <= 0:
        raise SystemExit("--smoke-layers must be positive")
    if args.repetition <= 0:
        raise SystemExit("--repetition must be positive")

    configure_jax_environment(args.platform, args.optimization)
    repository_root = Path(__file__).resolve().parents[2]
    case = get_case(args.case)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": _timestamp(),
        "case": case.to_dict(),
        "scope": {
            "kind": "full" if args.smoke_layers is None else "smoke",
            "smoke_layers": args.smoke_layers,
            "measured_work": (
                "ExoGibbs solver calls at the documented conditions; "
                "external solvers, plotting, and artifact writing are excluded"
            ),
        },
        "execution": {
            "requested_platform": args.platform,
            "optimization_mode": args.optimization,
            "repetition": args.repetition,
            "pid": os.getpid(),
        },
        "revision": _revision(repository_root),
        "solver_source_tree": _tree_fingerprint(
            repository_root / "src" / "exogibbs"
        ),
        "source_provenance": _source_provenance(
            repository_root,
            case.source_scripts + case.input_artifacts,
        ),
    }

    try:
        environment = _jax_environment(args.platform, args.optimization)
        environment["exogibbs_package_root"] = _checkout_package_path(
            repository_root
        )
        payload["environment"] = environment
    except NotImplementedError as error:
        payload.update(
            {
                "status": "unsupported",
                "finished_at_utc": _timestamp(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(args.output, payload)
        print(f"UNSUPPORTED {case.case_id}: {error}", flush=True)
        return 2
    except Exception as error:
        payload.update(
            {
                "status": "unavailable",
                "finished_at_utc": _timestamp(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(args.output, payload)
        print(f"UNAVAILABLE {case.case_id}: {error}", flush=True)
        return 2

    from benchmarks.documented_examples.instrumentation import TimingCollector
    from benchmarks.documented_examples.workloads import run_case

    print(
        f"RUN {case.case_id} platform={args.platform} "
        f"optimization={args.optimization} repetition={args.repetition}",
        flush=True,
    )
    collector = TimingCollector()
    workload_started = time.perf_counter()
    try:
        with collector:
            validation = run_case(case, collector, args.smoke_layers)
        workload_wall_seconds = time.perf_counter() - workload_started
        payload.update(
            {
                "status": "pass",
                "finished_at_utc": _timestamp(),
                "validation": validation,
                "timing": collector.summary(
                    workload_wall_seconds=workload_wall_seconds
                ),
            }
        )
        return_code = 0
    except Exception as error:
        workload_wall_seconds = time.perf_counter() - workload_started
        payload.update(
            {
                "status": "error",
                "finished_at_utc": _timestamp(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "timing": collector.summary(
                    workload_wall_seconds=workload_wall_seconds
                ),
            }
        )
        return_code = 1

    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_resident_set_size_kb = float(usage.ru_maxrss)
    if system_platform.system() == "Darwin":
        maximum_resident_set_size_kb /= 1024.0
    payload["resources"] = {
        "maximum_resident_set_size_kb": maximum_resident_set_size_kb
    }
    _write_json(args.output, payload)
    print(
        f"{payload['status'].upper()} {case.case_id}: {args.output}",
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
