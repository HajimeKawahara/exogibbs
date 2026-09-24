"""Fresh finite-BSE contact with common gas reactions and catalog diagnostics.
===========================================================================

The matched gas control tests the common reference independently of additional
upper species and clouds. Metal is explicitly suppressed in this first contact
control; neither phase stability nor physical material calibration is claimed.
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

from common_gibbs import minimize_gibbs
from local import build_problem
from m1_chemistry import audit_parcel, build_setups, solve_parcel
from m2_common_gas import SHARED_SPECIES, UPPER_SPECIES, build_common_gas_setup
from m2_standards_audit import audit_contact, audit_shared_standards
from run_bse_common_gibbs import build_bse_problem, json_value


def diagnose_contact(record, budget, source_result, callbacks, temperature_k, pressure_bar):
    """Recount primitive source amounts, then solve three independent parcels."""
    shared = build_common_gas_setup()
    _, expanded = build_setups()
    names = [name for phase in record["phases"].values() for name in phase]
    formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                        for element in record["elements"]])
    amounts = np.asarray(source_result["component_amounts_mol"], dtype=float)
    b = np.asarray(budget, dtype=float)
    if (source_result.get("accepted") is not True or amounts.shape != (len(names),)
            or b.shape != (len(record["elements"]),) or not np.all(np.isfinite(amounts))
            or np.any(amounts < 0) or not np.all(np.isfinite(b)) or np.any(b <= 0)
            or np.max(np.abs(formula @ amounts / b - 1.)) > 1e-9):
        raise ValueError("The fresh source must independently preserve its positive atomic ledger.")
    columns = [names.index(name) for name in record["phases"]["gas"]]
    shared_columns = [names.index(name + "_gas") for name in SHARED_SPECIES]
    if set(columns) != set(shared_columns):
        raise ValueError("The contact control requires exactly the eleven shared source gases.")
    rows = [record["elements"].index(element) for element in shared.elements]
    excluded = [i for i in range(len(b)) if i not in rows]
    if (np.any(formula[np.ix_(excluded, columns)] != 0)
            or not np.array_equal(formula[np.ix_(rows, shared_columns)], shared.formula_matrix)):
        raise ValueError("The lower and upper shared gas formulas must agree exactly.")
    gas = amounts[shared_columns]
    gas_budget = formula[np.ix_(rows, shared_columns)] @ gas
    lower_pressure = pressure_bar * gas / gas.sum()
    state = callbacks["gas"](temperature_k, pressure_bar, amounts[columns])
    lower_standard = np.asarray(state.mu_rt) - np.log(amounts[columns] / amounts[columns].sum()) - np.log(pressure_bar)
    lower_standard = lower_standard[[columns.index(column) for column in shared_columns]]
    standards = audit_shared_standards(
        shared.formula_matrix, lower_standard, shared.hvector_func(temperature_k),
        species=SHARED_SPECIES, elements=shared.elements,
    )
    source_gas_audit = audit_parcel(shared, temperature_k, pressure_bar, gas_budget, gas)
    reports = []
    for label, setup in (("shared_gas", shared), ("expanded_gas", expanded.gas_setup),
                         ("expanded_condensed", expanded)):
        try:
            parcel = solve_parcel(setup, temperature_k, pressure_bar, gas_budget)
            # Reconstruct partial pressures directly from component amounts.
            upper_amounts = np.asarray(parcel["gas_amounts_mol"])
            indices = [parcel["gas_species"].index(name) for name in UPPER_SPECIES]
            contact = audit_contact(lower_pressure, pressure_bar * upper_amounts[indices] / upper_amounts.sum(),
                                    species=SHARED_SPECIES)
            reports.append({"catalog": label, "parcel": parcel, "contact": contact})
        except (ValueError, RuntimeError, FloatingPointError) as error:
            reports.append({"catalog": label, "error_type": type(error).__name__, "error": str(error)})
    control = reports[0]
    accepted = (standards["accepted"] and source_gas_audit["accepted"]
                and control.get("parcel", {}).get("accepted") is True
                and control.get("contact", {}).get("accepted") is True)
    return {"common_standards": standards, "source_gas_audit": source_gas_audit,
            "shared_contact_accepted": bool(accepted), "catalogs": reports,
            "numerical_diagnostics_completed": bool(accepted and all(
                report.get("parcel", {}).get("accepted") is True for report in reports)),
            "interpretation": "Expanded catalog differences are measured model changes, not failures of the matched gas control."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--temperature", type=float, default=2173.15)
    parser.add_argument("--pressure", type=float, default=1.)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=1 for the declared contact tolerances.")
    if (not all(np.isfinite(value) and value > 0 for value in (args.temperature, args.pressure))
            or args.maxiter < 1 or args.output.exists()):
        parser.error("Require positive finite T/P, maxiter >= 1, and a new output path.")
    report = {"model_id": "m2_common_gas_contact_control_v1", "command": [sys.executable, *sys.argv],
              "temperature_K": args.temperature, "pressure_bar": args.pressure,
              "metal_status": "suppressed_for_contact_control",
              "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending", "M2_C": "pending"},
              "numerical_diagnostics_completed": False}
    try:
        record, budget, callbacks, initial, metadata = build_bse_problem(
            args.inventory, args.exoeos_checkout, args.runtime, args.python,
            temperature_k=args.temperature, pressure_bar=args.pressure, gas_model="m1_shared",
        )
        report.update(source_metadata=metadata, record=record, element_amounts_mol=budget.tolist(),
                      canonical_initial_component_amounts_mol=initial.tolist())
        metadata["provenance"]["file_sha256"]["run_m2_contact.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        problem = build_problem(record, budget, lambda t, p: np.zeros(len(initial)), phases=("silicate", "gas"))
        result = minimize_gibbs(problem, args.temperature, args.pressure, budget,
                                {name: callbacks[name] for name in problem.phases}, maxiter=args.maxiter)
        report["source_result"] = asdict(result)
        report.update(diagnose_contact(record, budget, report["source_result"], callbacks, args.temperature, args.pressure))
    except (ValueError, RuntimeError, FloatingPointError) as error:
        report["failure"] = {"error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
    if not report["numerical_diagnostics_completed"]:
        raise SystemExit("Contact control is unresolved; inspect the preserved diagnostics.")


if __name__ == "__main__":
    main()
