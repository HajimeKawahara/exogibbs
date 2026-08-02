#!/usr/bin/env python
"""Measure and gate the promoted public fixed-support v2 product route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from jax import config as jax_config
import jax.numpy as jnp

import exogibbs
EXPECTED_EXOGIBBS_ROOT = (SOURCE_ROOT / "exogibbs").resolve()
IMPORTED_EXOGIBBS_ROOT = Path(exogibbs.__file__).resolve().parent
if IMPORTED_EXOGIBBS_ROOT != EXPECTED_EXOGIBBS_ROOT:
    raise RuntimeError(
        "Imported exogibbs from outside this repository: "
        f"{IMPORTED_EXOGIBBS_ROOT} != {EXPECTED_EXOGIBBS_ROOT}"
    )

from exogibbs.api import (
    CondensateEquilibriumOptions,
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
    HEAD_ROUTE_V2,
    condensate_equilibrium_profile,
)
from exogibbs.api.condensate_equilibrium import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
)
from benchmarks.fixed_support_v2.curated_profiles import (
    FRESH_CURATED_PROFILES,
    element_budget_for_profile,
    support_payload_for_profile,
)
from exogibbs.equilibrium.condensate.policy import (
    fixed_support_v2_production_policy,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


DEFAULT_OUTPUT_DIR = ROOT / "results" / "fixed_support_v2_production_profile"
DEFAULT_FAMILIES = (
    "solar_highT_no_condensate_gas_regression",
    "solar_silicate_first_condensation",
    "solar_water_condensation",
    "lowT_strong_condensation_budget_stress",
    "near_phase_boundary_support_sensitivity",
)
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("curated_profiles.py"),
    Path(__file__).with_name("run_fixed_support_v2_production_profile_gpu.csh"),
    ROOT / "src/exogibbs/api/condensate.py",
    ROOT / "src/exogibbs/equilibrium/condensate/lifecycle.py",
    ROOT / "src/exogibbs/equilibrium/condensate/policy.py",
    ROOT / "src/exogibbs/equilibrium/condensate/support.py",
    ROOT / "src/exogibbs/equilibrium/condensate/fixed_support/batch.py",
)


def _host(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _host(item) for key, item in value._asdict().items()}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    array = np.asarray(jax.device_get(value))
    if array.shape == ():
        item = array.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    return array.tolist()


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_clean": status == "",
        "worktree_status": status,
    }


def _parse_gpu_compute_processes(output: str) -> list[dict[str, Any]]:
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid_text, process_name = line.split(",", maxsplit=1)
        processes.append(
            {
                "pid": int(pid_text.strip()),
                "process_name": process_name.strip(),
            }
        )
    return processes


def _external_gpu_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not inspect concurrent GPU compute processes: "
            f"{completed.stderr.strip()}"
        )
    current_pid = os.getpid()
    return [
        process
        for process in _parse_gpu_compute_processes(completed.stdout)
        if process["pid"] != current_pid
    ]


def _source_integrity() -> dict[str, Any]:
    missing = [
        str(path.relative_to(ROOT)) for path in SOURCE_PATHS if not path.exists()
    ]
    return {
        "algorithm": "sha256",
        "missing": missing,
        "files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in SOURCE_PATHS
            if path.exists()
        },
        **_git_metadata(),
    }


def _preflight_report(families: Sequence[str]) -> dict[str, Any]:
    policy = fixed_support_v2_production_policy(
        FIXED_SUPPORT_V2_VALIDATED_PRESET
    )
    default_options = CondensateEquilibriumOptions()
    unknown = sorted(set(families) - set(FRESH_CURATED_PROFILES))
    source = _source_integrity()
    checks = {
        "public_default_is_promoted_head_v2": (
            default_options.route == HEAD_ROUTE_V2
        ),
        "candidate_route_is_head_v2": HEAD_ROUTE_V2 == "head_v2",
        "validated_preset_is_named": (
            default_options.fixed_support_v2_preset
            == FIXED_SUPPORT_V2_VALIDATED_PRESET
        ),
        "validated_schedule_is_exact": (
            policy.solver_config.continuation.epsilon_schedule
            == (-11.0, -13.0, -15.0, -17.0)
        ),
        "validated_normal_limit_is_exact": (
            policy.solver_config.limits.max_normal_iterations == 1000
        ),
        "validated_support_limit_is_exact": policy.support_limit == 128,
        "approved_runtime_budget_is_named": (
            policy.runtime_budget_name == "a100_40gb_2026_07"
        ),
        "approved_runtime_budget_is_exact": (
            policy.max_cold_compilation_seconds == 900.0
            and policy.max_cold_wall_seconds == 960.0
            and policy.max_warm_execution_seconds == 20.0
            and policy.max_warm_wall_seconds == 25.0
        ),
        "all_families_known": not unknown,
        "all_sources_present": not source["missing"],
    }
    return {
        "schema": "exogibbs_fixed_support_v2_production_preflight_v1",
        "checks": checks,
        "unknown_families": unknown,
        "source_integrity": source,
        "passed": all(checks.values()),
    }


def _layer_report(layer: Any) -> dict[str, Any]:
    diagnostics = layer.diagnostics or {}
    lifecycle = diagnostics.get("fixed_support_v2", {})
    outcome = lifecycle.get("outcome")
    active = outcome != "gas_only_no_candidate"
    lifecycle_passed = (
        bool(lifecycle.get("fixed_support_converged"))
        and bool(lifecycle.get("support_closed"))
        and bool(lifecycle.get("independent_kkt_passed"))
        and bool(lifecycle.get("final_state_values_finite"))
        if active
        else outcome == "gas_only_no_candidate"
    )
    passed = (
        bool(layer.converged)
        and layer.head_route_version == CONDENSATE_HEAD_V2_ROUTE_VERSION
        and layer.head_route_name == CONDENSATE_HEAD_V2_ROUTE_NAME
        and lifecycle.get("production_preset_promoted") is True
        and lifecycle_passed
    )
    return {
        "status": layer.status,
        "converged": bool(layer.converged),
        "selected_route": layer.selected_route,
        "head_route_version": layer.head_route_version,
        "head_route_name": layer.head_route_name,
        "support_count": int(np.asarray(layer.condensate_support_indices).size),
        "support_names": list(layer.condensate_support_names),
        "lifecycle_outcome": outcome,
        "fixed_support_converged": lifecycle.get("fixed_support_converged"),
        "support_closed": lifecycle.get("support_closed"),
        "independent_kkt_passed": lifecycle.get(
            "independent_kkt_passed"
        ),
        "terminal_status_name": lifecycle.get("terminal_status_name"),
        "final_state_values_finite": lifecycle.get(
            "final_state_values_finite"
        ),
        "production_preset_promoted": lifecycle.get(
            "production_preset_promoted"
        ),
        "rounds": lifecycle.get("rounds", ()),
        "passed": passed,
    }


def _run_family(setup: Any, family: str, phase: str) -> dict[str, Any]:
    definition = FRESH_CURATED_PROFILES[family]
    budget = jnp.asarray(
        element_budget_for_profile(setup, definition), dtype=jnp.float64
    )
    support, amounts = support_payload_for_profile(setup, definition, budget)
    options = CondensateEquilibriumOptions(
        fixed_support_v2_preset=FIXED_SUPPORT_V2_VALIDATED_PRESET,
    )
    started = time.perf_counter()
    result = condensate_equilibrium_profile(
        setup,
        definition.temperatures,
        definition.pressures,
        budget,
        support_indices=support,
        support_amounts_init=amounts,
        options=options,
        method="vmap_cold",
    )
    _block_until_ready(result.batched_arrays)
    wall_seconds = time.perf_counter() - started
    profile = result.diagnostics or {}
    layers = [_layer_report(layer) for layer in result.layers]
    lifecycle_timing = (
        result.layers[0].diagnostics.get("fixed_support_v2", {})
        if result.layers and result.layers[0].diagnostics
        else {}
    )
    return {
        "family": family,
        "phase": phase,
        "layer_count": len(layers),
        "initial_support_count": len(support),
        "backend": profile.get("backend", lifecycle_timing.get("backend")),
        "compilation_seconds": float(
            lifecycle_timing.get("compilation_seconds_total", 0.0)
        ),
        "execution_seconds": float(
            lifecycle_timing.get("execution_seconds_total", 0.0)
        ),
        "diagnostic_seconds": float(
            lifecycle_timing.get("diagnostic_seconds_total", 0.0)
        ),
        "wall_seconds": wall_seconds,
        "layers": layers,
        "passed": all(layer["passed"] for layer in layers),
    }


def _maximum(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return max((float(row[field]) for row in rows), default=0.0)


def _timing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    fields = (
        "compilation_seconds",
        "execution_seconds",
        "diagnostic_seconds",
        "wall_seconds",
    )
    return {
        **{
            f"total_{field}": sum(float(row[field]) for row in rows)
            for field in fields
        },
        **{
            f"maximum_family_{field}": _maximum(rows, field)
            for field in fields
        },
    }


def _evaluate_budgets(
    cold_rows: Sequence[Mapping[str, Any]],
    warm_rows: Sequence[Mapping[str, Any]],
    limits: Mapping[str, float | None],
) -> dict[str, Any]:
    observed = {
        "cold_compilation_seconds": _maximum(
            cold_rows, "compilation_seconds"
        ),
        "cold_wall_seconds": _maximum(cold_rows, "wall_seconds"),
        "warm_execution_seconds": _maximum(warm_rows, "execution_seconds"),
        "warm_wall_seconds": _maximum(warm_rows, "wall_seconds"),
    }
    approved = all(value is not None for value in limits.values())
    checks = {
        name: (
            None
            if limits[name] is None
            else observed[name] <= float(limits[name])
        )
        for name in observed
    }
    return {
        "scope": "maximum over one curated public profile family",
        "approved_limits_supplied": approved,
        "limits_seconds": dict(limits),
        "observed_seconds": observed,
        "checks": checks,
        "passed": approved and all(value is True for value in checks.values()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_host(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    rows = [
        "# Fixed-support v2 production-profile GPU gate",
        "",
        f"- Correctness passed: `{payload['correctness_passed']}`",
        f"- Exclusive GPU measurement: "
        f"`{payload['environment']['exclusive_gpu_measurement']}`",
        f"- Runtime limits supplied: "
        f"`{payload['runtime_budget']['approved_limits_supplied']}`",
        f"- Runtime budget passed: `{payload['runtime_budget']['passed']}`",
        f"- Production-profile gate passed: "
        f"`{payload['production_profile_gate_passed']}`",
        f"- Promotion authorized: `{payload['promotion_authorized']}`",
        "",
        "| Phase | Family | Layers | Compile (s) | Execute (s) | Wall (s) | Passed |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for phase in ("cold", "warm"):
        for row in payload[phase]:
            rows.append(
                f"| {phase} | `{row['family']}` | {row['layer_count']} | "
                f"{row['compilation_seconds']:.6f} | "
                f"{row['execution_seconds']:.6f} | "
                f"{row['wall_seconds']:.6f} | `{row['passed']}` |"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument(
        "--element-file",
        default="FastChem4/element_abundances/asplund_2021.dat",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-approved-budgets", action="store_true")
    parser.add_argument("--max-cold-compilation-seconds", type=float)
    parser.add_argument("--max-cold-wall-seconds", type=float)
    parser.add_argument("--max-warm-execution-seconds", type=float)
    parser.add_argument("--max-warm-wall-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax_config.update("jax_enable_x64", True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_report(args.families)
    _write_json(args.output_dir / "production_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Production-profile input preflight failed.")
    if args.preflight_only:
        print(
            f"wrote {args.output_dir / 'production_preflight.json'}",
            flush=True,
        )
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("The production-profile runtime gate requires a GPU.")

    setup = condensate_chemical_setup(
        species_defalt_elements=False,
        element_file=args.element_file,
        silent=True,
    )
    cold = []
    warm = []
    gpu_process_samples = []

    def sample_gpu_processes(label: str) -> None:
        gpu_process_samples.append(
            {
                "label": label,
                "external_compute_processes": (
                    _external_gpu_compute_processes()
                ),
            }
        )

    sample_gpu_processes("before_measurement")
    for family in args.families:
        jax.clear_caches()
        sample_gpu_processes(f"before:{family}:cold")
        cold.append(_run_family(setup, family, "cold"))
        sample_gpu_processes(f"after:{family}:cold")
        sample_gpu_processes(f"before:{family}:warm")
        warm.append(_run_family(setup, family, "warm"))
        sample_gpu_processes(f"after:{family}:warm")
    sample_gpu_processes("after_measurement")
    policy = fixed_support_v2_production_policy(
        FIXED_SUPPORT_V2_VALIDATED_PRESET
    )
    limits = {
        "cold_compilation_seconds": (
            args.max_cold_compilation_seconds
            if args.max_cold_compilation_seconds is not None
            else policy.max_cold_compilation_seconds
        ),
        "cold_wall_seconds": (
            args.max_cold_wall_seconds
            if args.max_cold_wall_seconds is not None
            else policy.max_cold_wall_seconds
        ),
        "warm_execution_seconds": (
            args.max_warm_execution_seconds
            if args.max_warm_execution_seconds is not None
            else policy.max_warm_execution_seconds
        ),
        "warm_wall_seconds": (
            args.max_warm_wall_seconds
            if args.max_warm_wall_seconds is not None
            else policy.max_warm_wall_seconds
        ),
    }
    runtime_budget = _evaluate_budgets(cold, warm, limits)
    correctness_passed = all(row["passed"] for row in (*cold, *warm))
    environment = {
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_version": jax.__version__,
        "exclusive_gpu_measurement": not any(
            sample["external_compute_processes"]
            for sample in gpu_process_samples
        ),
        "gpu_process_samples": gpu_process_samples,
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
            }
            for device in jax.devices()
        ],
    }
    production_profile_gate_passed = bool(
        correctness_passed
        and environment["backend"] == "gpu"
        and environment["jax_enable_x64"]
        and environment["exclusive_gpu_measurement"]
        and runtime_budget["passed"]
    )
    payload = {
        "schema": "exogibbs_fixed_support_v2_production_profile_gpu_gate_v1",
        "production_preset_promoted": True,
        "route": HEAD_ROUTE_V2,
        "route_version": CONDENSATE_HEAD_V2_ROUTE_VERSION,
        "preset": FIXED_SUPPORT_V2_VALIDATED_PRESET,
        "families": list(args.families),
        "environment": environment,
        "source_integrity": _source_integrity(),
        "cache_measurement_policy": (
            "Clear JAX in-memory caches before each family, then run the same "
            "family immediately once cold and once warm."
        ),
        "terminal_diagnostics_enabled": False,
        "cold": cold,
        "warm": warm,
        "cold_timing": _timing_summary(cold),
        "warm_timing": _timing_summary(warm),
        "runtime_budget": runtime_budget,
        "runtime_budget_policy": policy.runtime_budget_name,
        "correctness_passed": correctness_passed,
        "production_profile_gate_passed": production_profile_gate_passed,
        "promotion_authorized": production_profile_gate_passed,
        "explicit_promotion_decision_required": False,
    }
    _write_json(args.output_dir / "summary.json", payload)
    _write_markdown(args.output_dir / "summary.md", payload)
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    print(f"wrote {args.output_dir / 'summary.md'}", flush=True)
    if not correctness_passed:
        raise RuntimeError("The public head_v2 correctness gate failed.")
    if not environment["exclusive_gpu_measurement"]:
        raise RuntimeError(
            "External GPU compute activity invalidated the runtime measurement."
        )
    if args.require_approved_budgets and not runtime_budget["passed"]:
        raise RuntimeError("Approved production runtime budgets did not pass.")


if __name__ == "__main__":
    main()
