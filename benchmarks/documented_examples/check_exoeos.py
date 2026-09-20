"""Run the optional fugacity gallery example with independent physical checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "plot_exoeos_pure_fugacity.py"
)


def _load_example() -> Any:
    spec = importlib.util.spec_from_file_location("audited_exoeos_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EXAMPLE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_result(setup, result, temperature, pressure, inventory, **options) -> dict:
    """Check the actual solve, including chemical stationarity in its gas basis."""
    amounts, fractions = np.asarray(result.n), np.asarray(result.x)
    formula, budget = np.asarray(setup.formula_matrix), np.asarray(inventory)
    finite = bool(
        np.all(np.isfinite(amounts)) and np.all(amounts > 0)
        and np.all(np.isfinite(fractions)) and np.all(fractions > 0)
    )
    correction = options.get("lnphi_func")
    errors = np.full(3, np.inf)
    if finite:
        # No trace floor: every element in this illustrative C-H-O case is present.
        errors[0] = np.max(np.abs(formula @ amounts / budget - 1))
        errors[1] = max(
            abs(float(np.sum(fractions)) - 1),
            float(np.max(np.abs(fractions - amounts / np.sum(amounts)))),
        )
        chemical_potential = (
            np.asarray(setup.hvector_func(temperature)) + np.log(fractions)
            + np.log(pressure / options.get("Pref", 1.0))
        )
        if correction is not None:
            chemical_potential += np.asarray(correction(temperature, pressure, None))
        if np.all(np.isfinite(chemical_potential)):
            multipliers = np.linalg.lstsq(formula.T, chemical_potential, rcond=None)[0]
            errors[2] = np.max(np.abs(chemical_potential - formula.T @ multipliers))
    return {
        "mode": "ideal" if correction is None else "pure_fugacity",
        "pressure_bar": float(pressure),
        "accepted": bool(finite and np.all(errors <= (1e-9, 1e-12, 1e-8))),
        **dict(zip(
            ("maximum_relative_element_error", "normalization_error", "stationarity_error"),
            [float(value) if np.isfinite(value) else None for value in errors],
        )),
    }


def run(output_directory: Path) -> dict[str, Any]:
    """Execute the original gallery main and audit every returned equilibrium."""
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = output_directory / "exoeos_pure_fugacity.png"
    figure.unlink(missing_ok=True)
    audit: dict[str, Any] = {"accepted": False, "layers": [], "error": None}
    original_argv = sys.argv
    try:
        example = _load_example()
        original_solve = example.solve

        def audited_solve(setup, temperature, pressure, inventory, **options):
            result = original_solve(setup, temperature, pressure, inventory, **options)
            audit["layers"].append(
                _audit_result(setup, result, temperature, pressure, inventory, **options)
            )
            return result

        example.solve = audited_solve
        sys.argv = [str(EXAMPLE_PATH), "--output", str(figure)]
        example.main()
        audit["accepted"] = bool(
            figure.is_file() and figure.stat().st_size > 0
            and len(audit["layers"]) == 2 * len(example.PRESSURES_BAR)
            and all(layer["accepted"] for layer in audit["layers"])
        )
        if not audit["accepted"]:
            audit["error"] = "Missing figure, incomplete profile, or failed physical audit."
    except Exception as error:
        audit["error"] = f"{type(error).__name__}: {error}"
    finally:
        sys.argv = original_argv
        (output_directory / "exoeos_audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if not run(args.output_directory)["accepted"]:
        raise SystemExit("The ExoEOS gallery example failed its physical audit.")


if __name__ == "__main__":
    main()
