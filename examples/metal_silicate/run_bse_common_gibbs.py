"""Conditional common-energy minimization from an exported absolute BSE ledger.
============================================================================

The external MELTS runtime and ExoEOS checkout must be selected explicitly.
This does not calibrate cross-phase standards or establish BSE liquid stability.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from common_gibbs import PhaseEvaluationError, minimize_gibbs
from full_potential import PhaseState, ideal_phase
from hydrogen import _checkout_provenance, dissolved_h2_standard_rt, hirschmann2012_ln_solubility
from local import build_problem
from melts_coupled import COMMON_R, load_melts_evaluator, make_melts_h2_phase, provider_ledger
from reference import load_reference
from run_melts_reference import reduced_gas_standards_rt
from source import make_source_standard_potentials_rt


ELEMENTS = ("O", "Mg", "Si", "Fe", "Al", "Ca", "Na", "K", "Ti", "Cr", "P", "H", "He")


def source_standards_rt(temperature_k: float, pressure_bar: float = 1.) -> tuple[dict, dict]:
    """Return the recorded source/Burcat standards and formulas on common R.

    The 2350 K Shomate branch remains fixed. This helper does not align
    these data with MELTS or the M1 upper atmosphere.
    """
    source = load_reference()
    source_names = [name for values in source["phases"].values() for name in values]
    # The selected source branch is deliberately fixed and recorded. It is a
    # formal extrapolation at 2173.15 K, not a newly aligned MELTS standard.
    standard = np.asarray(make_source_standard_potentials_rt(source, source["cases"][0])(temperature_k, pressure_bar))
    standard = standard * source["source"]["gas_constant_J_mol_K"] / COMMON_R
    standards = dict(zip(source_names, standard))
    formulas = dict(source["component_formulas"])
    extra_record = json.loads(Path(__file__).with_name("reduced_gas_reference.json").read_text())
    extra = reduced_gas_standards_rt(temperature_k)
    for name, value in extra.items():
        standards[name + "_gas"] = value
        formulas[name + "_gas"] = extra_record["species"][name]["formula"]
    return standards, formulas


def build_bse_problem(
    inventory_path: Path, exoeos_checkout: Path, runtime: Path, python_executable: str,
    *, temperature_k: float = 2173.15, pressure_bar: float = 1.,
) -> tuple[dict, np.ndarray, dict, np.ndarray, dict]:
    """Return record, absolute budgets, callbacks, initial ledger and metadata.

    Only the caller-selected JSON ledger is consumed, with no ExoInventory
    import. The canonical initial ledger contains dry BSE plus H2/He gas and
    exactly zero metal; optimization guesses cannot change its atomic budget.
    MELTS is evaluated at a fixed 100 g dry-rock amount scale, then its
    extensive energy is converted back to the full physical mol basis.
    """
    import exoeos
    from exoeos import MaFeSiOHLiquid, total_solution_gibbs_RT, total_solution_state

    checkout = Path(exoeos_checkout).resolve()
    if Path(exoeos.__file__).resolve().parent != checkout / "src" / "exoeos":
        raise RuntimeError("Set PYTHONPATH to the selected ExoEOS checkout before importing it.")
    input_path = Path(inventory_path).resolve()
    inventory = json.loads(input_path.read_text())
    if tuple(inventory["elements"]) != ELEMENTS:
        raise ValueError("The exported BSE ledger must use the declared thirteen-element basis.")
    if inventory["preexisting_core_mass_kg"] != 0 or any(inventory["zero_element_amounts_mol"].values()):
        raise ValueError("This BSE case requires zero preexisting core and C=N=S=0.")
    budget = np.asarray(inventory["total_element_amounts_mol"], dtype=float)
    rock = np.asarray(inventory["rock_element_amounts_mol"], dtype=float)
    if (budget.shape != (13,) or rock.shape != (13,) or not np.all(np.isfinite(budget))
            or np.any(budget < 0) or not np.all(np.isfinite(rock)) or np.any(rock < 0)
            or not np.array_equal(budget[:11], rock[:11]) or np.any(rock[11:] != 0)):
        raise ValueError("Dry rock and independent H/He inventories are inconsistent.")
    evaluator = load_melts_evaluator(checkout)
    oxide = dict(zip(inventory["oxide_order"], inventory["oxide_amounts_mol"]))
    if len(oxide) != len(inventory["oxide_order"]) or set(k.lower() for k in oxide) - set(evaluator.OXIDES):
        raise ValueError("Oxide ledger contains duplicate or unsupported oxide names.")
    oxide = {name.lower(): value for name, value in oxide.items()}
    physical_host = np.linalg.solve(evaluator.NU.T, [oxide.get(name, 0.) for name in evaluator.OXIDES])
    if not np.all(np.isfinite(physical_host)) or np.any(physical_host < 0):
        raise ValueError("BSE lies outside the nonnegative MELTS component cone.")
    host_totals = dict(zip(evaluator.ELEMENTS, evaluator.FORMULA_MATRIX.T @ physical_host))
    if not np.allclose([host_totals.get(name, 0.) for name in ELEMENTS], rock, rtol=1e-12, atol=0):
        raise ValueError("MELTS endmembers do not reconstruct the exported dry-rock atoms.")
    if not np.isfinite(inventory["dry_rock_mass_kg"]) or inventory["dry_rock_mass_kg"] <= 0:
        raise ValueError("Dry rock mass must be positive and finite.")
    amount_scale = .1 / inventory["dry_rock_mass_kg"]
    # Keep every supported MELTS component, including initially zero ferric
    # iron and native water. Their formation consumes the finite O/H ledger.
    allowed_elements = {name for name, value in zip(ELEMENTS, budget) if value > 0}
    host_indices = [i for i, formula in enumerate(evaluator.FORMULA_MATRIX)
                    if all(value == 0 or name in allowed_elements for name, value in zip(evaluator.ELEMENTS, formula))]
    host_names = [evaluator.COMPONENTS[i] for i in host_indices]
    source = load_reference()
    standards, formulas = source_standards_rt(temperature_k, pressure_bar)
    extra = reduced_gas_standards_rt(temperature_k)
    gas_names = [name for name in source["phases"]["gas"] + [n + "_gas" for n in extra]
                 if set(formulas[name]) <= allowed_elements]
    metal_names = [name for name in source["phases"]["metal"] if set(formulas[name]) <= allowed_elements]
    melt_names = [name + "_melts" for name in host_names]
    for index, name in zip(host_indices, melt_names):
        formulas[name] = {element: float(value) for element, value in zip(evaluator.ELEMENTS, evaluator.FORMULA_MATRIX[index]) if value}
    if budget[ELEMENTS.index("H")] == 0:
        raise ValueError("This wet BSE provider case requires positive H; use an explicit dry branch for H=0.")
    melt_names.append("H2_dissolved")
    formulas["H2_dissolved"] = {"H": 2}
    phases = {"silicate": melt_names, "metal": metal_names, "gas": gas_names}
    names = [name for phase in phases.values() for name in phase]
    matrix = np.array([[formulas[name].get(e, 0) for name in names] for e in ELEMENTS])
    initial = np.zeros(len(names))
    initial[:len(host_indices)] = physical_host[host_indices]
    initial[names.index("H2_gas")] = budget[ELEMENTS.index("H")] / 2
    if "He_gas" in names:
        initial[names.index("He_gas")] = budget[ELEMENTS.index("He")]
    if not np.allclose(matrix @ initial, budget, rtol=1e-12, atol=0):
        raise ValueError("Canonical initial components add or omit atoms from the input ledger.")
    record = {"elements": list(ELEMENTS), "phases": phases,
              "component_formulas": {name: formulas[name] for name in names}, "reactions": []}
    h2_standard = float(dissolved_h2_standard_rt(standards["H2_gas"], hirschmann2012_ln_solubility(pressure_bar)))
    scaled_melt = make_melts_h2_phase(evaluator, host_names, lambda t, p: h2_standard,
                                      runtime=runtime, python_executable=python_executable)
    execution = {"native_melt_calls": 0, "last_failed_melt_state": None}
    def melt(t, p, n):
        execution["native_melt_calls"] += 1
        try:
            state = scaled_melt(t, p, amount_scale * n)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            execution["last_failed_melt_state"] = {
                "temperature_K": t, "pressure_bar": p,
                "component_order": melt_names,
                "component_amounts_mol": np.asarray(n).tolist(),
                "native_scaled_component_amounts_mol": (amount_scale * n).tolist(),
            }
            raise PhaseEvaluationError(str(error)) from error
        return PhaseState(state.mu_rt, state.gibbs_rt / amount_scale)
    model = MaFeSiOHLiquid()
    full_metal_names = source["phases"]["metal"]
    metal_indices = np.array([full_metal_names.index(name) for name in metal_names])
    metal_standard = np.array([standards[name] for name in full_metal_names]) + np.asarray(model.standard_state_shift_RT(temperature_k))
    metal_evaluator = jax.jit(lambda n: total_solution_state(model, temperature_k, pressure_bar * 1e5, n, metal_standard))
    def alloy(t, p, n):
        if t != temperature_k or p != pressure_bar:
            raise ValueError("Rebuild callbacks after changing the reference temperature or pressure.")
        expanded = np.zeros(4)
        expanded[metal_indices] = n
        if expanded.sum() > 0:
            try:
                model.validate_state(t, p * 1e5, expanded / expanded.sum())
            except ValueError as error:
                raise PhaseEvaluationError(str(error)) from error
        state = metal_evaluator(expanded)
        return PhaseState(np.asarray(state.mu_RT)[metal_indices], float(state.gibbs_RT))
    if len(metal_indices) == 4:
        alloy_derivative = jax.jit(jax.value_and_grad(
            lambda n: total_solution_gibbs_RT(model, temperature_k, pressure_bar * 1e5, n, metal_standard)))
        alloy.energy_value_and_grad_rt = lambda t, p, n: alloy_derivative(n)
    gas_standard = np.array([standards[name] for name in gas_names])
    callbacks = {"silicate": melt, "metal": alloy,
                 "gas": ideal_phase(lambda t, p: gas_standard, gas=True)}
    metadata = {
        "model_id": "bse_melts_ma_fe_si_o_h_common_gibbs_conditional_v1",
        "evidence_level": "conditional_numerical_mechanism_only",
        "numerical_execution": execution,
        "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"},
        "input": {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "source_provenance": inventory["provenance"], "dry_rock_mass_kg": inventory["dry_rock_mass_kg"],
                  "native_amount_scale": amount_scale, "element_amounts_mol": budget.tolist(), "elements": list(ELEMENTS)},
        "host_ledger": provider_ledger(evaluator, host_names),
        "standards": {"source_temperature_branch_K": source["cases"][0]["T_K"],
                      "evaluation_temperature_K": temperature_k,
                      "policy": "Fixed source branch extrapolation; common R conversion, native alloy shifts, and one H2 solubility pressure term. Cross-phase alignment is unverified."},
        "missing_acceptance": ["aligned MELTS/alloy/gas/upper-atmosphere standards", "revised host-specific H2 calibration",
                               "alloy H interactions and pressure response", "BSE liquid stability and omitted transfer bounds",
                               "global nonconvex phase stability"],
        "provenance": {"exogibbs": _checkout_provenance(Path(__file__).resolve().parents[2]),
                       "exoeos": _checkout_provenance(checkout), "jax_version": jax.__version__, "numpy_version": np.__version__,
                       "file_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("run_bse_common_gibbs.py", "common_gibbs.py", "full_potential.py", "melts_coupled.py", "hydrogen.py", "source.py", "reference.json", "reduced_gas_reference.json")}},
    }
    return record, budget, callbacks, initial, metadata


def json_value(value: Any) -> Any:
    """Represent unavailable endpoint diagnostics as JSON null, never NaN."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--temperature", type=float, default=2173.15)
    parser.add_argument("--pressure", type=float, default=1.)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--metal-absent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record, budget, callbacks, initial, metadata = build_bse_problem(
        args.inventory, args.exoeos_checkout, args.runtime, args.python,
        temperature_k=args.temperature, pressure_bar=args.pressure,
    )
    phases = ("silicate", "gas") if args.metal_absent else tuple(record["phases"])
    problem = build_problem(record, budget, lambda t, p: np.zeros(initial.size), phases=phases)
    report = {**metadata, "temperature_K": args.temperature, "pressure_bar": args.pressure,
              "record": record, "active_phases": list(phases), "canonical_initial_component_amounts_mol": initial.tolist(),
              "solver": {"name": "constrained_scalar_gibbs", "maxiter": args.maxiter}, "command": [sys.executable, *sys.argv]}
    try:
        result = minimize_gibbs(problem, args.temperature, args.pressure, budget,
                               {name: callbacks[name] for name in phases}, maxiter=args.maxiter)
        report["result"] = asdict(result)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        report["result"] = {"accepted": False, "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
    if not report["result"]["accepted"]:
        raise SystemExit("BSE common-energy case is unresolved; see the saved diagnostics.")


if __name__ == "__main__":
    main()
