"""Execute a curated plotting demo and reject concealed numerical failures."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any
from unittest.mock import patch

import numpy as np


def validate_layer(setup: Any, budget: Any, result: Any) -> dict[str, Any]:
    """Independently check the returned state in the caller's amount gauge."""

    if result is None:
        return {"accepted": False, "status": "missing"}
    gas = np.asarray(result.gas_n, dtype=float)
    fractions = np.asarray(result.gas_x, dtype=float)
    gas_total = float(result.gas_ntot)
    condensed = np.asarray(result.condensate_amounts, dtype=float)
    target = np.asarray(budget, dtype=float)
    inventory = np.asarray(setup.formula_matrix) @ gas
    inventory += np.asarray(setup.formula_matrix_cond) @ condensed
    elements = np.asarray([name not in {"e-", "electron"} for name in setup.elements])
    # Match the public full-budget gate's tolerance and inventory-relative floor.
    scale = np.sum(target[elements & (target > 0)])
    floor = max(1.0e-6 * scale, np.nextafter(0.0, 1.0))
    relative_error = np.abs(inventory - target) / np.maximum(np.abs(target), floor)
    budget_error = float(np.max(relative_error[elements]))
    checks = {
        "converged": bool(result.converged),
        "accepted_status": result.status in {"converged", "converged_with_caveat"},
        "finite_nonnegative_state": all(
            bool(np.all(np.isfinite(values)) and np.all(values >= 0))
            for values in (gas, fractions, condensed)
        ),
        "normalized_gas": bool(np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-7)),
        "consistent_gas_amounts": bool(
            np.isfinite(gas_total) and gas_total > 0
            and np.isclose(gas.sum(), gas_total, rtol=1e-7, atol=0)
            and np.allclose(gas / gas_total, fractions, rtol=1e-7, atol=1e-14)
        ),
        "element_budget": bool(scale > 0 and np.isfinite(budget_error) and budget_error <= 1.0e-3),
    }
    return {
        "accepted": all(checks.values()), "status": result.status,
        "acceptance_tier": result.acceptance_tier,
        "max_relative_element_error": budget_error,
        "lifecycle_outcome": (result.diagnostics or {}).get("fixed_support_v2", {}).get("outcome"),
        "checks": checks,
    }


def run_curated(script: Path, output_directory: Path) -> dict[str, Any]:
    """Call the real demo main, preserving plots while auditing every layer."""

    script = script.resolve()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import jax
    from matplotlib.figure import Figure

    backend = jax.default_backend()
    requested_gpu = any(
        value in {"gpu", "cuda", "rocm"}
        for name in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
        for value in os.environ.get(name, "").split(",")
    )
    if requested_gpu and backend != "gpu":
        raise RuntimeError(f"GPU was requested, but JAX selected {backend!r}")
    sys.path.insert(0, str(script.parent))
    report: dict[str, Any] = {
        "script": str(script), "backend": backend, "layers": [],
        "errors": [], "expected_layers": None, "plots": [],
    }
    original_savefig = Figure.savefig

    def savefig(figure: Any, filename: Any, *args: Any, **kwargs: Any) -> Any:
        destination = output_directory / Path(filename).name
        value = original_savefig(figure, destination, *args, **kwargs)
        report["plots"].append(str(destination))
        return value

    try:
        common = importlib.import_module("_curated_demo_common")
        namespace = runpy.run_path(str(script))
        main = namespace["main"]
        if "_run_profile" in namespace:
            original_run = namespace["_run_profile"]

            def audited_run(setup: Any) -> Any:
                rows = original_run(setup)
                report["expected_layers"] = len(namespace["PRESSURES_BAR"])
                budget = setup.gas_setup.element_vector_reference
                report["layers"] = [validate_layer(setup, budget, result) for _, result, _ in rows]
                report["errors"] = [error for _, _, error in rows if error is not None]
                return rows

            runner_patch = patch.dict(main.__globals__, {"_run_profile": audited_run})
        else:
            original_run = common.run_fresh_curated_profile

            def audited_run(setup: Any, definition: Any) -> Any:
                results, gas_rows, errors = original_run(setup, definition)
                report["expected_layers"] = len(definition.temperatures)
                budget = common.element_budget_for_profile(setup, definition)
                report["layers"] = [validate_layer(setup, budget, result) for result in results]
                report["errors"] = list(errors)
                if len(definition.pressures) != len(results) or len(gas_rows) != len(results):
                    report["errors"].append("Incomplete pressure or plotted-gas profile")
                return results, gas_rows, errors

            runner_patch = patch.object(common, "run_fresh_curated_profile", audited_run)
        output_path = lambda path: output_directory / Path(path).with_suffix(".png").name
        with runner_patch, patch.object(Figure, "savefig", savefig), patch.dict(
            main.__globals__, {"curated_output_path": output_path}
        ):
            main()
    except Exception as error:  # A plotting script can otherwise conceal solver failures.
        report["errors"].append(f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        sys.path.remove(str(script.parent))
    report["accepted"] = bool(
        not report["errors"] and report["layers"]
        and len(report["layers"]) == report["expected_layers"]
        and all(layer["accepted"] for layer in report["layers"])
        and report["plots"]
        and all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in report["plots"])
    )
    (output_directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_curated(arguments.script, arguments.output_directory)
    print(json.dumps({"accepted": report["accepted"], "layers": len(report["layers"])}))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
