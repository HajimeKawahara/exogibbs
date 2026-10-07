"""Execute complete documentation examples and audit their computed results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "preset_ykb4_gas": {"document": "documents/presets/ykb4.rst", "blocks": (0,), "kind": "gas"},
    "preset_fastchem_gas": {"document": "documents/presets/fastchem.rst", "blocks": (0,), "kind": "gas"},
    "preset_fastchem4_gas": {"document": "documents/presets/fastchem4.rst", "blocks": (0,), "kind": "gas"},
    "preset_fastchem4_condensate": {"document": "documents/presets/fastchem4.rst", "blocks": (1,), "kind": "condensate"},
    "preset_fastchem4_profile": {"document": "documents/presets/fastchem4.rst", "blocks": (2,), "kind": "profile"},
    "condensate_profile": {"document": "documents/condensate_profile.rst", "blocks": (3, 4), "kind": "profile"},
    "solubility": {"document": "documents/solubility.rst", "blocks": (0,), "kind": "solubility"},
    "magma_gas_interface": {"document": "documents/magma_gas_interface.rst", "blocks": (0,), "kind": "magma"},
    "ideal_solution_adapter": {"document": "documents/metal_silicate_reference.rst", "blocks": (0,), "kind": "adapter"},
}
JAPANESE_CASES = {
    "ja_magma_gas_interface": {"document": "ja/magma_gas_interface.tex", "blocks": (0,), "kind": "legacy_magma"},
}
ANCHORS = {
    "gas": "result = solve(", "condensate": "result = solve(",
    "profile": "profile = solve_profile(", "solubility": "x_h2 = h2_hirschmann2012(",
    "magma": "result = magma_gas.solve(", "adapter": "values = lngamma(",
    "legacy_magma": "state = solve_magma_atmosphere_interface(",
}


def extract_blocks(path: Path) -> tuple[str, ...]:
    """Extract the indented Python code blocks in one checked-in RST source."""
    if path.suffix == ".tex":
        return tuple(match.group(1).strip() + "\n" for match in re.finditer(
            r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", path.read_text(encoding="utf-8"), re.DOTALL,
        ))
    return tuple(
        textwrap.dedent(match.group().split("\n", 1)[1]).strip() + "\n"
        for match in re.finditer(
            r"^\.\. code-block:: python\n(?:\n|[ \t]+.*\n)*",
            path.read_text(encoding="utf-8"), re.MULTILINE,
        )
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _budget_report(reconstructed, target) -> dict[str, float]:
    target, reconstructed = np.asarray(target), np.asarray(reconstructed)
    _require(bool(np.all(np.isfinite(reconstructed))), "Non-finite element inventory.")
    positive = target > 0
    relative = float(np.max(np.abs(reconstructed[positive] / target[positive] - 1), initial=0))
    zero = float(np.max(np.abs(reconstructed[target == 0]), initial=0))
    return {"maximum_relative_element_error": relative, "maximum_zero_element_error": zero}


def _budget_audit(reconstructed, target, tolerance: float) -> dict[str, float]:
    report = _budget_report(reconstructed, target)
    relative, zero = report.values()
    _require(relative <= tolerance, f"Positive element budget error {relative:g} exceeds {tolerance:g}.")
    _require(zero <= 1e-10 * float(np.sum(np.abs(target))), f"Zero element budget error {zero:g}.")
    return report


def _amount_audit(amounts, fractions) -> None:
    amounts, fractions = np.asarray(amounts), np.asarray(fractions)
    _require(bool(np.all(np.isfinite(amounts)) and np.all(amounts >= 0)), "Invalid gas amounts.")
    _require(bool(np.all(np.isfinite(fractions)) and np.all(fractions >= 0)), "Invalid gas fractions.")
    _require(abs(float(np.sum(fractions)) - 1) <= 1e-8, "Gas fractions are not normalized.")
    _require(bool(np.allclose(fractions, amounts / np.sum(amounts), rtol=1e-8, atol=0)), "Gas amounts and fractions disagree.")


def _condensate_audit(setup, result, budget) -> dict[str, Any]:
    from benchmarks.documented_examples.check_curated import validate_layer

    _require(bool(result.converged), f"Condensate solve failed: {result.status}.")
    audit = validate_layer(setup, budget, result)
    _require(audit["accepted"], f"Condensate physical audit failed: {audit['checks']}.")
    reconstructed = np.asarray(setup.formula_matrix) @ np.asarray(result.gas_n)
    reconstructed += np.asarray(setup.formula_matrix_cond) @ np.asarray(result.condensate_amounts)
    charge = np.asarray([name in {"e-", "electron"} for name in setup.elements])
    charge_error = float(np.max(np.abs((reconstructed - np.asarray(budget))[charge]), initial=0))
    # The converged lifecycle result already includes its charge/KKT gate;
    # the independent full-budget gate excludes charge, which we report apart.
    return {
        **audit, **_budget_report(reconstructed, budget),
        "maximum_absolute_charge_error": charge_error,
    }


def _gas_audit(setup, result, temperature, pressure, budget) -> dict[str, Any]:
    from exogibbs.api.gas import EquilibriumOptions
    from exogibbs.equilibrium.gas.solve import _effective_equilibrium_tolerance

    _amount_audit(result.n, result.x)
    amounts, formula = np.asarray(result.n), np.asarray(setup.formula_matrix)
    chemical_potential = (
        np.asarray(setup.hvector_func(temperature)) + np.asarray(result.ln_n)
        - np.log(float(result.ntot)) + np.log(pressure)
    )
    _require(bool(np.all(np.isfinite(chemical_potential))), "Non-finite gas chemical potential.")
    # compute_residuals_with_at_pi uses an absolute L2 amount residual, not
    # element-relative or unweighted log errors. The public result omits the
    # final multipliers and independent total; project its returned state in
    # that same amount norm, retaining raw trace-relative errors as diagnostics.
    weighted_formula = amounts[:, None] * formula.T
    weighted_mu = amounts * chemical_potential
    multipliers = np.linalg.lstsq(weighted_formula, weighted_mu, rcond=None)[0]
    stationarity = weighted_formula @ multipliers - weighted_mu
    element_residual = formula @ amounts - np.asarray(budget)
    residual = float(np.linalg.norm(np.concatenate((stationarity, element_residual))))
    tolerance = _effective_equilibrium_tolerance(EquilibriumOptions().epsilon_crit, amounts.dtype)
    raw_multipliers = np.linalg.lstsq(formula.T, chemical_potential, rcond=None)[0]
    raw_stationarity = float(np.max(np.abs(chemical_potential - formula.T @ raw_multipliers)))
    return {
        **_budget_report(formula @ amounts, budget),
        "accepted": bool(np.isfinite(residual) and residual <= tolerance),
        "projected_amount_residual": residual,
        "effective_epsilon_crit": tolerance,
        "absolute_element_residual_norm": float(np.linalg.norm(element_residual)),
        "amount_weighted_stationarity_norm": float(np.linalg.norm(stationarity)),
        "unweighted_log_stationarity_error": raw_stationarity,
        "criterion": "Independent projected absolute amount residual; final internal multipliers and total are not exposed.",
    }


def audit_result(kind: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Certify each complete block without changing its solver or input values."""
    if kind == "gas":
        return _gas_audit(
            namespace["setup"], namespace["result"], namespace["T"], namespace["P"], namespace["b"],
        )
    if kind == "condensate":
        return _condensate_audit(namespace["setup"], namespace["result"], namespace["b"])
    if kind == "profile":
        profile, setup = namespace["profile"], namespace["setup"]
        _require(bool(profile.layers), "Empty condensate profile.")
        targets = profile.element_inventory_target
        _require(not profile.rainout or targets is not None, "Rainout per-layer inventories are missing.")
        fallback = setup.gas_setup.element_vector_reference
        return {
            "rainout": bool(profile.rainout),
            "layers": [
                _condensate_audit(setup, layer, targets[index] if targets is not None else fallback)
                for index, layer in enumerate(profile.layers)
            ],
        }
    if kind == "magma":
        result = namespace["result"]
        _require(bool(result.diagnostics.converged), "Magma-gas interface did not converge.")
        _require(bool(result.diagnostics.inner_converged), "Inner gas solve did not converge.")
        _require(bool(result.diagnostics.outer_converged), "Outer magma solve did not converge.")
        gas = result.gas.equilibrium
        _amount_audit(gas.n, gas.x)
        values = np.asarray(result.model_state.melt_volatile_mole_ratios)
        _require(bool(np.all(np.isfinite(values)) and np.all(values > 0)), "Invalid melt volatile ratios.")
        return _budget_audit(
            np.asarray(namespace["problem"].setup.formula_matrix) @ np.asarray(gas.n),
            result.element_abundances, 1e-8,
        )
    if kind == "legacy_magma":
        state = namespace["state"]
        _require(bool(state.diagnostics.converged), "Japanese magma-gas example did not converge.")
        _require(bool(state.diagnostics.inner_converged and state.diagnostics.outer_converged),
                 "Japanese magma-gas example has incomplete inner/outer closure.")
        _amount_audit(np.exp(np.asarray(state.gas_ln_n)), state.gas_mole_fractions)
        return _budget_audit(
            np.asarray(namespace["chemistry"].setup.formula_matrix) @ np.exp(np.asarray(state.gas_ln_n)),
            state.element_abundances, 1e-8,
        )
    if kind == "solubility":
        value = float(namespace["x_h2"])
        _require(np.isfinite(value) and 0 < value < 1, "Invalid H2 solubility.")
        return {"x_h2": value}
    if kind == "adapter":
        values = np.asarray(namespace["values"])
        _require(values.shape == (3,) and bool(np.all(values == 0)), "Ideal solution activity coefficients must vanish.")
        return {"lngamma": values.tolist()}
    raise ValueError(f"Unknown inline example kind: {kind}.")


def run(case_name: str, output_directory: Path) -> dict[str, Any]:
    """Run and record the exact selected source blocks in their document order."""
    case = {**CASES, **JAPANESE_CASES}[case_name]
    output_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"case": case_name, **case, "accepted": False, "checks": [], "error": None}
    try:
        root = (Path(os.environ.get("EXOGIBBS_JAPANESE_DOCS", REPOSITORY_ROOT / "doc_ExoGibbs"))
                if case_name in JAPANESE_CASES else REPOSITORY_ROOT)
        blocks = extract_blocks(root / case["document"])
        namespace: dict[str, Any] = {"__name__": "__documented_example__"}
        for index in case["blocks"]:
            source = blocks[index]
            _require(ANCHORS[case["kind"]] in source, "The documented block changed; update its explicit manifest.")
            check = {
                "block_index": index, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "accepted": False,
            }
            report["checks"].append(check)
            exec(compile(source, case["document"], "exec"), namespace)
            audit = audit_result(case["kind"], namespace)
            check.update(audit)
            _require(audit.get("accepted", True), "The documented result failed its physical acceptance criterion.")
            check["accepted"] = True
        report["accepted"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (output_directory / "inline_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, *JAPANESE_CASES), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if not run(args.case, args.output_directory)["accepted"]:
        raise SystemExit("The inline documentation example failed its audit.")


if __name__ == "__main__":
    main()
