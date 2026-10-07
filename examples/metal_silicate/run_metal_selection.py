"""Mixed-Ma metal selection controls
==================================

Archive mixed-alloy reference states or a conditional BSE attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np
import scipy

from full_potential import PhaseState, ideal_phase
from hydrogen import _checkout_provenance
from phase_selection import select_metal_phase
from run_bse_common_gibbs import build_bse_problem, json_value


TEMPERATURE_K, PRESSURE_BAR = 2173.15, 1.
METAL_LOWER = np.array([.86, 0., 0., 0.])
METAL_UPPER = np.array([1., .08, .02, .04])
METAL_REFERENCE_X = np.array([.945, .025, .01, .02])


def formula_matrix(record):
    names = [name for components in record["phases"].values() for name in components]
    return np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                     for element in record["elements"]])


def reference_problem(model, metal_standard_shift_rt):
    """Construct known coexistence using real Ma mixing and synthetic standards."""
    from exoeos import solution_state, total_solution_gibbs_RT, total_solution_state

    phases = {"silicate": ["SiO2_l", "FeO_l", "H2O_l", "H2_l"],
              "metal": ["Fe_m", "Si_m", "O_m", "H_m"],
              "gas": ["H2_g", "He_g", "H2O_g", "SiO_g"]}
    formulas = {"SiO2": {"Si": 1, "O": 2}, "FeO": {"Fe": 1, "O": 1},
                "H2O": {"H": 2, "O": 1}, "H2": {"H": 2},
                "Fe": {"Fe": 1}, "Si": {"Si": 1}, "O": {"O": 1}, "H": {"H": 1},
                "He": {"He": 1}, "SiO": {"Si": 1, "O": 1}}
    record = {"elements": ["H", "He", "O", "Si", "Fe", "C"], "phases": phases,
              "component_formulas": {name: formulas[name.split("_")[0]]
                                     for group in phases.values() for name in group}, "reactions": []}
    target = np.r_[np.array([1., .5, .03, .05]), .2 * METAL_REFERENCE_X,
                   np.array([.3, .1, .02, .002])]
    standards, callbacks, start = {}, {}, 0
    for phase, names in phases.items():
        amounts = target[start:start + len(names)]
        x = amounts / amounts.sum()
        standards[phase] = -np.log(x)
        if phase == "metal":
            excess = solution_state(model, TEMPERATURE_K, PRESSURE_BAR * 1e5, x)
            standards[phase] += metal_standard_shift_rt - np.asarray(excess.lngamma)
        else:
            callbacks[phase] = ideal_phase(lambda t, p, mu0=standards[phase]: mu0, gas=phase == "gas")
        start += len(names)
    evaluate = jax.jit(lambda n: total_solution_state(
        model, TEMPERATURE_K, PRESSURE_BAR * 1e5, n, standards["metal"]))
    value_gradient = jax.jit(jax.value_and_grad(lambda n: total_solution_gibbs_RT(
        model, TEMPERATURE_K, PRESSURE_BAR * 1e5, n, standards["metal"])))

    def metal(t, p, amounts):
        if t != TEMPERATURE_K or p != PRESSURE_BAR:
            raise ValueError("Rebuild the fixed-condition reference after changing T/P.")
        state = evaluate(amounts)
        return PhaseState(np.asarray(state.mu_RT), float(state.gibbs_RT))

    metal.energy_value_and_grad_rt = lambda t, p, n: value_gradient(n)
    callbacks["metal"] = metal
    budget = formula_matrix(record) @ target
    return record, budget, callbacks, target, standards


def source_receipt(checkout):
    directory = Path(__file__).resolve().parent
    local = ("run_metal_selection.py", "phase_selection.py", "common_gibbs.py", "full_potential.py",
             "local.py", "run_bse_common_gibbs.py", "hydrogen.py")
    provider = ("ma_fe_si_o.py", "ma_fe_si_o_h.py", "ma_interval.py", "solution_gibbs.py",
                "gibbs_excess.py", "state.py", "_arrays.py")
    files = {"exogibbs/examples/metal_silicate/" + name: directory / name for name in local}
    files.update({"exoeos/src/exoeos/" + name: checkout / "src/exoeos" / name for name in provider})
    return {"exogibbs": _checkout_provenance(directory.parents[1]),
            "exoeos": _checkout_provenance(checkout),
            "file_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
            "jax_version": jax.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64), "command": [sys.executable, *sys.argv]}


def selection_record(record, selection):
    """Restore all phase rows, including exact-zero metal, for JSON consumers."""
    result = asdict(selection)
    if selection.result is not None:
        amounts = selection.result.component_amounts_mol
        matrix = formula_matrix(record)
        phase_amounts, phase_atoms, start = [], [], 0
        for names in record["phases"].values():
            section = slice(start, start + len(names))
            phase_amounts.append(float(amounts[section].sum()))
            phase_atoms.append(matrix[:, section] @ amounts[section])
            start += len(names)
        result["result"]["phase_order"] = list(record["phases"])
        result["result"]["phase_amounts_mol"] = phase_amounts
        result["result"]["phase_element_amounts_mol"] = phase_atoms
    return result


def independent_audit(record, budget, selection, callbacks, expected=None):
    """Reconstruct atoms and energy directly from the returned component vector."""
    if selection.result is None:
        return {"available": False}
    n = selection.result.component_amounts_mol
    reconstructed = formula_matrix(record) @ n
    positive = budget > 0
    error = np.zeros_like(budget)
    error[positive] = (reconstructed[positive] - budget[positive]) / budget[positive]
    error[~positive] = reconstructed[~positive]
    energies, phase_mu, start = {}, {}, 0
    for phase, names in record["phases"].items():
        amounts = n[start:start + len(names)]
        state = callbacks[phase](TEMPERATURE_K, PRESSURE_BAR, amounts)
        energies[phase] = state.gibbs_rt
        phase_mu[phase] = state.mu_rt
        start += len(names)
    audit = {"available": True, "reconstructed_element_amounts_mol": reconstructed,
             "element_relative_error": error, "maximum_element_error": float(np.max(np.abs(error))),
             "fresh_phase_gibbs_rt": energies, "fresh_phase_mu_rt": phase_mu,
             "energy_difference_rt": float(sum(energies.values()) - selection.result.gibbs_rt)}
    if expected is not None:
        audit["known_coexistence_component_amounts_mol"] = expected
        audit["maximum_known_coexistence_relative_error"] = float(np.max(np.abs((n - expected) / expected)))
    if audit["maximum_element_error"] > 1e-9:
        raise ValueError("Independent returned-amount audit failed.")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--bse-inventory", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--gas-model", choices=("source", "m1_shared", "m1_expanded"), default="m1_shared")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve prior reference records.")
    if args.bse_inventory and (args.runtime is None or args.python is None):
        parser.error("The BSE route requires --runtime and --python.")
    if not jax.config.jax_enable_x64:
        parser.error("Set JAX_ENABLE_X64=1 for the recorded numerical tolerances.")
    import exoeos
    from exoeos.ma_interval import ma_alloy_curvature_lower_bound

    checkout = args.exoeos_checkout.resolve()
    if Path(exoeos.__file__).resolve().parent != checkout / "src/exoeos":
        parser.error("Set PYTHONPATH to the selected ExoEOS checkout's src directory.")
    model = exoeos.MaFeSiOHLiquid()
    curvature = ma_alloy_curvature_lower_bound(model, TEMPERATURE_K, METAL_LOWER, METAL_UPPER)
    report = {"model_id": "m2_mixed_ma_phase_selection_controls_v1",
              "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"},
              "temperature_K": TEMPERATURE_K, "pressure_bar": PRESSURE_BAR,
              "metal_domain": {"component_order": list(model.components), "lower": METAL_LOWER,
                               "upper": METAL_UPPER, "curvature_lower_bound_rt": curvature,
                               "meaning": "Restricted mathematical control, not experimental calibration."},
              "provenance": source_receipt(checkout), "cases": []}
    if args.bse_inventory:
        record, budget, callbacks, initial, metadata = build_bse_problem(
            args.bse_inventory, checkout, args.runtime, args.python,
            temperature_k=TEMPERATURE_K, pressure_bar=PRESSURE_BAR, gas_model=args.gas_model)
        selection = select_metal_phase(record, budget, TEMPERATURE_K, PRESSURE_BAR, callbacks,
                                       METAL_LOWER, METAL_UPPER,
                                       convex_phase_bounds={"metal": curvature, "gas": 0.}, maxiter=args.maxiter)
        report["model_id"] = "m2_bse_phase_selection_conditional_v1"
        report["cases"].append({"record": record, "element_amounts_mol": budget,
                                "canonical_initial_amounts_mol": initial, "metadata": metadata,
                                "selection": selection_record(record, selection),
                                "independent_audit": independent_audit(record, budget, selection, callbacks)})
        if selection.status != "unresolved":
            raise ValueError("BSE cannot be certified without a global liquid stability bound.")
    else:
        for shift in (0., .1, 4.62, 4.63):
            record, budget, callbacks, target, standards = reference_problem(model, shift)
            selection = select_metal_phase(record, budget, TEMPERATURE_K, PRESSURE_BAR, callbacks,
                                           METAL_LOWER, METAL_UPPER,
                                           convex_phase_bounds={"silicate": 0., "metal": curvature, "gas": 0.},
                                           maxiter=args.maxiter)
            audit = independent_audit(record, budget, selection, callbacks, target if shift == 0 else None)
            report["cases"].append({"metal_standard_shift_rt": shift, "record": record,
                                    "element_amounts_mol": budget, "standard_potentials_rt": standards,
                                    "selection": selection_record(record, selection), "independent_audit": audit})
        report["reference_definition"] = {
            "unshifted_metal_composition": METAL_REFERENCE_X,
            "known_unshifted_metal_amount_mol": .2,
            "known_unshifted_elemental_potentials_rt": [0.] * len(record["elements"]),
            "standards": "Manufactured so every component potential is zero at known three-phase coexistence; metal shifts add the same dimensionless standard offset to all four atoms.",
            "scope": "Real Ma excess mixing, ideal synthetic host/gas and synthetic absolute standards. Parameter offsets are not physical hydrogen-series boundaries or BSE predictions."}
        report["insertion_sign_bracket"] = {
            "metal_standard_shift_rt": [4.62, 4.63],
            "meaning": "Candidate loss of stability of the metal-free branch only; an unresolved metal-bearing branch is not a certified coexistence boundary."}
        report["numerical_reference_acceptance"] = {
            "unshifted_coexistence_recovered": bool(report["cases"][0]["selection"]["status"] == "metal_present"
                and report["cases"][0]["independent_audit"].get("maximum_known_coexistence_relative_error", np.inf) < 1e-6),
            "both_metal_presence_and_absence_certified": {case["selection"]["status"] for case in report["cases"]}
                >= {"metal_present", "metal_absent"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
