"""Smoke benchmark for condensate_equilibrium_profile auto rescue routing."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

def _bootstrap_jax_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "cuda", "gpu", "default"),
        default="cpu",
    )
    args, _ = parser.parse_known_args()
    platform = str(args.jax_platform)
    if platform != "default":
        os.environ["JAX_PLATFORMS"] = platform
        os.environ["JAX_PLATFORM_NAME"] = platform
    return platform


REQUESTED_JAX_PLATFORM = _bootstrap_jax_platform()
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "volatiles_code") not in sys.path:
    sys.path.insert(0, str(ROOT / "volatiles_code"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmark_pdipm_api_profile_fixed_support_batch import (  # noqa: E402
    _activity_outer_profile_args,
)
from benchmark_pdipm_fixed_support_core import (  # noqa: E402
    _block_tree,
    _jax_runtime_report,
    _json_safe,
)
from exogibbs.api.condensate_equilibrium import (  # noqa: E402
    CondensateEquilibriumOptions,
    condensate_equilibrium_profile,
)
from exogibbs.condensates.curated_profiles import FRESH_CURATED_PROFILES  # noqa: E402
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402


def _block_profile_result(result: Any) -> None:
    if getattr(result, "batched_arrays", None) is not None:
        _block_tree(result.batched_arrays)
        return
    for layer in getattr(result, "layers", ()):
        _block_tree(
            (
                layer.gas_ln_n,
                layer.gas_n,
                layer.gas_x,
                layer.gas_ntot,
                layer.condensate_amounts,
                layer.condensate_support_indices,
            )
        )


def _time(fn: Any, *, warmup: int, repeat: int) -> tuple[Any, dict[str, Any]]:
    first_start = time.perf_counter()
    first = fn()
    _block_profile_result(first)
    first_elapsed = time.perf_counter() - first_start
    for _ in range(warmup):
        _block_profile_result(fn())
    times = []
    last = first
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn()
        _block_profile_result(last)
        times.append(time.perf_counter() - start)
    return last, {
        "first_call_seconds": first_elapsed,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "warm_call_seconds": times,
        "warm_median_seconds": statistics.median(times),
        "warm_min_seconds": min(times),
        "warm_max_seconds": max(times),
    }


def _fallback_rescue_summary(profile_result: Any) -> dict[str, Any]:
    diagnostics = profile_result.diagnostics or {}
    profile_diag = diagnostics.get("experimental_profile_fixed_support_batch", {})
    rescue = profile_diag.get("fallback_rescue")
    if not isinstance(rescue, dict):
        return {
            "route": profile_diag.get("route"),
            "rescue_mode": None,
            "fallback_layer_indices": [],
            "expanded_layer_count": 0,
            "replaced_count": 0,
        }
    return {
        "route": profile_diag.get("route"),
        "rescue_mode": rescue.get("mode"),
        "fallback_layer_indices": list(rescue.get("fallback_layer_indices", ())),
        "expanded_layer_count": int(rescue.get("expanded_layer_count", 0)),
        "replaced_count": int(rescue.get("replaced_count", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-platform", default=REQUESTED_JAX_PLATFORM)
    parser.add_argument("--print-jax-devices", action="store_true")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "carbon_rich_graphite_window",
            "solar_silicate_first_condensation",
        ],
    )
    parser.add_argument("--disable-budget-gate", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.print_jax_devices:
        print(json.dumps(_jax_runtime_report(str(args.jax_platform)), sort_keys=True))
    setup = condensate_chemical_setup()
    budget_gate = not bool(args.disable_budget_gate)
    rows = []
    skipped = []
    for family in args.families:
        if family not in FRESH_CURATED_PROFILES:
            raise ValueError(f"unknown family: {family}")
        try:
            profile_args, family_skipped, support_metadata = (
                _activity_outer_profile_args(
                    setup,
                    family,
                    iterations=int(args.iterations),
                    budget_gate=budget_gate,
                )
            )
        except ValueError as exc:
            skipped.append({"family": family, "reason": str(exc)})
            continue
        skipped.extend(family_skipped)
        base_b = jnp.asarray(profile_args["b"], dtype=jnp.float64)
        options = CondensateEquilibriumOptions(
            max_inner_iterations=int(args.iterations),
            enable_full_condensate_budget_residual_gate=budget_gate,
        )

        def run_profile(*, diagnostics: bool = False) -> Any:
            return condensate_equilibrium_profile(
                setup,
                profile_args["T"],
                profile_args["P"],
                base_b,
                init=profile_args["init"],
                options=options,
                return_diagnostics=diagnostics,
            )

        diagnostic_result = run_profile(diagnostics=True)
        _block_profile_result(diagnostic_result)
        route_summary = _fallback_rescue_summary(diagnostic_result)
        last, timing = _time(
            lambda: run_profile(diagnostics=False),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )
        del last
        layers = len(profile_args["init"])
        eval_count = layers
        row = {
            "family": family,
            "layers": layers,
            "support_metadata": support_metadata,
            "auto_route": route_summary,
            "timing": timing,
            "evaluations_per_second": (
                None
                if timing["warm_median_seconds"] == 0.0
                else eval_count / timing["warm_median_seconds"]
            ),
            "seconds_per_layer": (
                None
                if layers == 0
                else timing["warm_median_seconds"] / layers
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "family": family,
                    "layers": layers,
                    "auto_route": route_summary,
                    "warm_median_seconds": timing["warm_median_seconds"],
                    "evaluations_per_second": row["evaluations_per_second"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_time = sum(float(row["timing"]["warm_median_seconds"]) for row in rows)
    total_layers = sum(int(row["layers"]) for row in rows)
    payload = {
        "schema": "exogibbs_condensate_profile_auto_rescue_smoke_v1",
        "jax_runtime": _jax_runtime_report(str(args.jax_platform)),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "budget_gate": budget_gate,
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "family_count": len(rows),
            "layer_count": total_layers,
            "total_warm_median_seconds": total_time,
            "layers_per_second": None
            if total_time == 0.0
            else total_layers / total_time,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
