"""Compare public ExoGibbs condensation with pinned native GGChem functions.

The input is a generated, sampled thermochemical export, not a new fit. Native
abundances are used only after an independent ExoGibbs cold solve for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.api import condensate
from exogibbs.thermo.models import ChemicalSetup


def sampled_setup(data: dict, phase: str) -> ChemicalSetup:
    """Evaluate exactly exported knots; reject interpolation and extrapolation."""
    order = np.argsort(data["temperature_K"])
    temperatures = jnp.asarray(data["temperature_K"])[order]
    potential = (-np.asarray(data["gas_lnK_bar"]) if phase == "gas"
                 else np.asarray(data["condensate_hvector_RT"]))
    potential = jnp.asarray(potential[order])

    def hvector(t):
        index = jnp.clip(jnp.searchsorted(temperatures, t), 0, len(temperatures) - 1)
        return jnp.where(t == temperatures[index], potential[index], jnp.nan)

    return ChemicalSetup(
        jnp.asarray(data[f"{phase}_formula_matrix"], dtype=float), jax.jit(hvector),
        tuple(data["elements"]), tuple(data[f"{phase}_species"]),
        jnp.asarray(data["budget"]), {"sampled_only": True, "standard_pressure_bar": 1.})


def primitive_checks(ln_n, amounts, matrix, phase_matrix, budget, ln_k, hcond,
                     pressure, atoms) -> dict:
    """Audit log-space mass action and full trace-element inventories."""
    logx = ln_n - logsumexp(ln_n)
    logp = logx + np.log(pressure)
    mu = logp[atoms]
    affinity = phase_matrix.T @ mu - hcond
    active = amounts > 0
    return {
        "element_relative_residual": float(np.max(np.abs(
            matrix @ np.exp(ln_n) + phase_matrix @ amounts - budget) / budget)),
        "mass_action_max_abs_ln": float(np.max(np.abs(logp - mu @ matrix - ln_k))),
        "active_affinity_max_abs_ln": float(np.max(np.abs(affinity[active]), initial=0.)),
        "inactive_affinity_positive_max_ln": float(np.max(affinity[~active], initial=0.)),
        "nonnegative_condensates": bool(np.all(amounts >= 0)),
        "all_log_amounts_finite": bool(np.isfinite(ln_n).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/phosphorus_cold_validation/native7_phase_inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phosphorus_cold_validation/condensate")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    data = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    gas, phases = sampled_setup(data, "gas"), sampled_setup(data, "condensate")
    bundle = condensate.build_condensate_chemical_setup(gas_setup=gas, condensate_setup=phases)
    matrix, phase_matrix, budget = map(np.asarray, (
        gas.formula_matrix, phases.formula_matrix, gas.element_vector_reference))
    atoms = [data["gas_species"].index(name) for name in data["elements"]]
    if not np.array_equal(matrix[:, atoms], np.eye(len(atoms))):
        raise ValueError("Expected neutral atomic reference species")
    rows = []
    receipt = {
        "scope": "Same native functions, 1156 gases and 22 pure condensates; independent cold starts; no rainout",
        "input_scope": data["scope"], "source_revision": data["source_revision"],
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "temperature_K": data["temperature_K"], "pressure_bar": data["pressure_bar"],
        "elements": data["elements"], "budget": data["budget"],
        "gas_species": data["gas_species"], "condensate_species": data["condensate_species"],
        "gas_formula_matrix": matrix.tolist(), "condensate_formula_matrix": phase_matrix.tolist(),
        "gates": {"relative_element": 1e-6, "mass_action_ln": 1e-8,
                  "phase_affinity_ln": 1e-8, "delta_logx": 2e-5,
                  "delta_condensate_element_fraction": 2e-5},
        "trace_reference": "Native printed trace densities are floored. Full logx is reconstructed from native atomic partial pressures and exported lnK; independently printed logx is also compared above logx=-600.",
        "states": rows,
    }
    for i, temperature in enumerate(data["temperature_K"]):
        start = time.monotonic()
        result = condensate.solve(bundle, temperature, data["pressure_bar"], gas.element_vector_reference,
                                  options=condensate.CondensateEquilibriumOptions(return_diagnostics=True))
        ln_n, amounts = np.asarray(result.gas_ln_n), np.asarray(result.condensate_amounts)
        logx = ln_n - logsumexp(ln_n)
        ln_k = np.asarray(data["gas_lnK_bar"])[i]
        checks = primitive_checks(ln_n, amounts, matrix, phase_matrix, budget, ln_k,
                                 np.asarray(data["condensate_hvector_RT"])[i],
                                 data["pressure_bar"], atoms)
        native_printed = np.asarray(data["native_condensed_gas_logx"])[i]
        native_logx = ln_k + (native_printed[atoms] + np.log(data["pressure_bar"])) @ matrix - np.log(data["pressure_bar"])
        above_floor = native_printed > -600.
        reference_amounts = np.asarray(data["native_condensed_amounts_per_H"])[i]
        delta_phase = float(np.max(np.abs(phase_matrix * (amounts - reference_amounts)[None, :]) / budget[:, None]))
        delta_logx = float(np.max(np.abs(logx - native_logx)))
        delta_printed = float(np.max(np.abs(logx[above_floor] - native_printed[above_floor])))
        passed = bool(result.converged and checks["element_relative_residual"] < 1e-6
                      and checks["mass_action_max_abs_ln"] < 1e-8
                      and checks["active_affinity_max_abs_ln"] < 1e-8
                      and checks["inactive_affinity_positive_max_ln"] < 1e-8
                      and checks["nonnegative_condensates"] and checks["all_log_amounts_finite"]
                      and delta_logx < 2e-5 and delta_printed < 2e-5 and delta_phase < 2e-5)
        rows.append({"temperature_K": temperature, "passed": passed,
                     "solver_status": result.status, "acceptance_tier": result.acceptance_tier,
                     "seconds": time.monotonic() - start, **checks,
                     "delta_logx_reconstructed_max_abs": delta_logx,
                     "delta_logx_printed_max_abs": delta_printed,
                     "printed_comparison_species_count": int(above_floor.sum()),
                     "delta_condensate_element_fraction_max_abs": delta_phase,
                     "gas_logx": logx.tolist(), "gas_log_amounts": ln_n.tolist(),
                     "condensate_amounts_per_H": amounts.tolist(),
                     "native_gas_logx_reconstructed": native_logx.tolist(),
                     "native_condensate_amounts_per_H": reference_amounts.tolist()})
        receipt["states_passed"] = sum(row["passed"] for row in rows)
        (args.output / "comparison_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{temperature:.6f} K: pass={passed}, delta_ln_x={delta_logx:.3g}, "
              f"element={checks['element_relative_residual']:.3g}, {result.condensate_support_names}", flush=True)
    if not all(row["passed"] for row in rows):
        raise SystemExit("One or more declared comparison gates failed; inspect the receipt")


if __name__ == "__main__":
    main()
