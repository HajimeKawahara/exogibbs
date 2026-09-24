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
from m2_atmosphere import audit_atmosphere, make_atmosphere_phase
from m2_common_gas import build_common_gas_setup, source_gas_names
from m2_standards_audit import audit_contact, audit_shared_standards
from run_bse_common_gibbs import build_bse_problem, json_value


def audit_expanded_contact(record, source_result, callbacks, temperature_k, pressure_bar, basal):
    """Audit primitive source/basal gases, clouds, standards and atom transfer."""
    _, upper = build_setups()
    gas_map = record.get("gas_species_aliases", {})
    cloud_map = record.get("retained_condensate_components", {})
    cloud_names = [name for phase, names in record["phases"].items()
                   if phase not in ("silicate", "metal", "gas") for name in names]
    if (set(gas_map) != set(record["phases"]["gas"])
            or set(gas_map.values()) != set(upper.gas_species) or len(gas_map) != len(upper.gas_species)
            or set(cloud_map) != set(cloud_names) or set(cloud_map.values()) != set(upper.condensate_species)
            or len(cloud_map) != len(upper.condensate_species)):
        raise ValueError("Contact requires every expanded gas and retained condensate exactly once.")
    names = [name for group in record["phases"].values() for name in group]
    n = np.asarray(source_result["component_amounts_mol"], dtype=float)
    if n.shape != (len(names),) or np.any(n < 0) or not np.all(np.isfinite(n)):
        raise ValueError("Source component amounts must be finite and nonnegative.")
    source_gas = [next(name for name, alias in gas_map.items() if alias == species) for species in upper.gas_species]
    source_cloud = [next(name for name, alias in cloud_map.items() if alias == species) for species in upper.condensate_species]
    for source_names, setup in ((source_gas, upper.gas_setup), (source_cloud, upper.condensate_setup)):
        formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in source_names]
                            for element in record["elements"]])
        expected = np.array([np.asarray(setup.formula_matrix)[setup.elements.index(element)]
                             if element in setup.elements else np.zeros(len(source_names))
                             for element in record["elements"]])
        if not np.array_equal(formula, expected):
            raise ValueError("Source gas/cloud formulas must match the complete atmospheric catalog.")
    gas = n[[names.index(name) for name in source_gas]]
    cloud = n[[names.index(name) for name in source_cloud]]
    gas_atoms = np.asarray(upper.gas_setup.formula_matrix) @ gas
    cloud_atoms = np.asarray(upper.condensate_setup.formula_matrix) @ cloud
    budget = gas_atoms + cloud_atoms
    if (tuple(basal["elements"]) != upper.gas_setup.elements
            or tuple(basal["gas_species"]) != upper.gas_species
            or tuple(basal["condensate_species"]) != upper.condensate_species
            or basal["T_K"] != temperature_k or basal["P_bar"] != pressure_bar):
        raise ValueError("Basal catalog and temperature/pressure must match the finite source.")
    source_audit = audit_atmosphere(upper, temperature_k, pressure_bar, budget, gas, cloud)
    upper_audit = audit_atmosphere(upper, temperature_k, pressure_bar, budget,
                                   basal["gas_amounts_mol"], basal["condensate_amounts_mol"])
    gas_contact = audit_contact(source_audit["partial_pressures_bar"], upper_audit["partial_pressures_bar"],
                                species=upper.gas_species)
    # Recover gas standards at a positive test composition, so exact-zero
    # source elements do not turn a declared standard into -inf - (-inf).
    uniform = np.ones(len(record["phases"]["gas"]))
    gas_mu = np.asarray(callbacks["gas"](temperature_k, pressure_bar, uniform).mu_rt)
    gas_standard = gas_mu + np.log(len(uniform)) - np.log(pressure_bar)
    lower = list(gas_standard[[record["phases"]["gas"].index(name) for name in source_gas]])
    cloud_standard = {}
    for phase, phase_names in record["phases"].items():
        if phase not in ("silicate", "metal", "gas"):
            state = callbacks[phase](temperature_k, pressure_bar, np.zeros(len(phase_names)))
            cloud_standard.update(zip(phase_names, np.asarray(state.mu_rt)))
    eligible = np.asarray(source_audit["condensate_temperature_eligible"], dtype=bool)
    lower.extend(cloud_standard[name] for name, valid in zip(source_cloud, eligible) if valid)
    matrix = np.column_stack((upper.gas_setup.formula_matrix,
                              np.asarray(upper.condensate_setup.formula_matrix)[:, eligible]))
    raw = np.r_[upper.gas_setup.hvector_func(temperature_k),
                np.asarray(upper.condensate_setup.hvector_func(temperature_k))[eligible]]
    standard_audit = audit_shared_standards(matrix, lower, raw,
        species=list(upper.gas_species) + [name for name, valid in zip(upper.condensate_species, eligible) if valid],
        elements=upper.gas_setup.elements)
    positive = budget > 0
    atom_residual = np.asarray(upper_audit["relative_element_residual"])
    cloud_residual = np.asarray(upper_audit["cloud_element_amounts_mol"]) - cloud_atoms
    cloud_residual[positive] /= budget[positive]
    accepted = (source_result.get("accepted") is True and source_audit["accepted"]
                and upper_audit["accepted"] and gas_contact["accepted"] and standard_audit["accepted"]
                and np.max(np.abs(cloud_residual), initial=0.) < 1e-9)
    return {"accepted": bool(accepted), "gas_contact": gas_contact, "common_standards": standard_audit,
            "source_atmosphere_audit": source_audit, "upper_atmosphere_audit": upper_audit,
            "relative_element_residual": atom_residual.tolist(),
            "cloud_element_relative_residual": cloud_residual.tolist(),
            "scope": "Local retained-atmosphere contact; material calibration and global planetary closure remain separate."}


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
            or np.any(amounts < 0) or not np.all(np.isfinite(b)) or np.any(b < 0)
            or len(set(names)) != len(names) or not np.all(np.isfinite(formula)) or np.any(formula < 0)):
        raise ValueError("The fresh source must independently preserve its nonnegative atomic ledger.")
    reconstructed = formula @ amounts
    if (np.any(reconstructed[b == 0] != 0)
            or np.max(np.abs(reconstructed[b > 0] / b[b > 0] - 1.), initial=0.) > 1e-9):
        raise ValueError("The fresh source must independently preserve its nonnegative atomic ledger.")
    if "retained_condensate_components" in record:
        atmospheric = [name for phase, group in record["phases"].items()
                       if phase not in ("silicate", "metal") for name in group]
        positions = [names.index(name) for name in atmospheric]
        rows = [record["elements"].index(element) for element in expanded.gas_setup.elements]
        atmospheric_budget = formula[np.ix_(rows, positions)] @ amounts[positions]
        parcel = make_atmosphere_phase(expanded, np.zeros(7)).parcel(temperature_k, pressure_bar, atmospheric_budget)
        contact = audit_expanded_contact(record, source_result, callbacks, temperature_k, pressure_bar, parcel)
        return {"common_standards": contact["common_standards"],
                "source_gas_audit": contact["source_atmosphere_audit"],
                "matched_catalog": "expanded_condensed", "matched_contact_accepted": contact["accepted"],
                "shared_contact_accepted": contact["accepted"],
                "catalogs": [{"catalog": "expanded_condensed", "parcel": parcel,
                              "contact": contact["gas_contact"], "expanded_contact": contact}],
                "numerical_diagnostics_completed": contact["accepted"],
                "interpretation": "All 35 gases and 26 retained condensates participate in the finite source and contact audits."}
    columns = [names.index(name) for name in record["phases"]["gas"]]
    aliases = record.get("gas_species_aliases", dict(zip(source_gas_names(shared), shared.species)))
    if set(aliases) != set(record["phases"]["gas"]) or len(set(aliases.values())) != len(aliases):
        raise ValueError("Source gas aliases must map every component exactly once.")
    if set(aliases.values()) not in (set(shared.species), set(expanded.gas_species)):
        raise ValueError("The source gas catalog must match the eleven or 35 species control.")
    source_setup = expanded.gas_setup if set(aliases.values()) == set(expanded.gas_species) else shared
    source_names = [next(name for name, alias in aliases.items() if alias == species)
                    for species in source_setup.species]
    source_columns = [names.index(name) for name in source_names]
    if set(columns) != set(source_columns):
        raise ValueError("The source gas catalog must match the eleven or 35 species control.")
    rows = [record["elements"].index(element) for element in shared.elements]
    excluded = [i for i in range(len(b)) if i not in rows]
    if (np.any(formula[np.ix_(excluded, columns)] != 0)
            or not np.array_equal(formula[np.ix_(rows, source_columns)], source_setup.formula_matrix)):
        raise ValueError("The lower and upper shared gas formulas must agree exactly.")
    gas = amounts[source_columns]
    gas_budget = formula[np.ix_(rows, source_columns)] @ gas
    lower_pressure = pressure_bar * gas / gas.sum()
    state = callbacks["gas"](temperature_k, pressure_bar, amounts[columns])
    lower_standard = np.asarray(state.mu_rt) - np.log(amounts[columns] / amounts[columns].sum()) - np.log(pressure_bar)
    lower_standard = lower_standard[[columns.index(column) for column in source_columns]]
    standards = audit_shared_standards(
        source_setup.formula_matrix, lower_standard, source_setup.hvector_func(temperature_k),
        species=source_setup.species, elements=source_setup.elements,
    )
    source_gas_audit = audit_parcel(source_setup, temperature_k, pressure_bar, gas_budget, gas)
    reports = []
    for label, setup in (("shared_gas", shared), ("expanded_gas", expanded.gas_setup),
                         ("expanded_condensed", expanded)):
        try:
            parcel = solve_parcel(setup, temperature_k, pressure_bar, gas_budget)
            # Reconstruct partial pressures directly from component amounts.
            upper_amounts = np.asarray(parcel["gas_amounts_mol"])
            compared = [name for name in source_setup.species if name in parcel["gas_species"]]
            indices = [parcel["gas_species"].index(name) for name in compared]
            lower_indices = [source_setup.species.index(name) for name in compared]
            contact = audit_contact(lower_pressure[lower_indices], pressure_bar * upper_amounts[indices] / upper_amounts.sum(),
                                    species=compared)
            reports.append({"catalog": label, "parcel": parcel, "contact": contact})
        except (ValueError, RuntimeError, FloatingPointError) as error:
            reports.append({"catalog": label, "error_type": type(error).__name__, "error": str(error)})
    matched_catalog = "expanded_gas" if source_setup is expanded.gas_setup else "shared_gas"
    control = next(report for report in reports if report["catalog"] == matched_catalog)
    accepted = (standards["accepted"] and source_gas_audit["accepted"]
                and control.get("parcel", {}).get("accepted") is True
                and control.get("contact", {}).get("accepted") is True)
    return {"common_standards": standards, "source_gas_audit": source_gas_audit,
            "shared_contact_accepted": bool(accepted), "matched_catalog": matched_catalog,
            "matched_contact_accepted": bool(accepted), "catalogs": reports,
            "numerical_diagnostics_completed": bool(accepted and all(
                report.get("parcel", {}).get("accepted") is True for report in reports)),
            "interpretation": "Matched catalog acceptance audits every source gas; other catalog changes remain separate diagnostics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--temperature", type=float, default=2173.15)
    parser.add_argument("--pressure", type=float, default=1.)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--gas-model", choices=("m1_shared", "m1_expanded", "m1_retained"), default="m1_shared")
    parser.add_argument("--metal-mode", choices=("suppressed", "select"), default="suppressed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=1 for the declared contact tolerances.")
    if (not all(np.isfinite(value) and value > 0 for value in (args.temperature, args.pressure))
            or args.maxiter < 1 or args.output.exists()):
        parser.error("Require positive finite T/P, maxiter >= 1, and a new output path.")
    report = {"model_id": "m2_common_gas_contact_control_v1", "command": [sys.executable, *sys.argv],
              "temperature_K": args.temperature, "pressure_bar": args.pressure,
              "metal_status": "suppressed_for_contact_control" if args.metal_mode == "suppressed" else "selected_local_state",
              "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending", "M2_C": "pending"},
              "numerical_diagnostics_completed": False}
    try:
        if args.gas_model == "m1_retained":
            from m2_expanded_source import build_expanded_bse_problem, unpack_expanded_source
            record, budget, callbacks, initial, metadata = build_expanded_bse_problem(
                args.inventory, args.exoeos_checkout, args.runtime, args.python,
                temperature_k=args.temperature, pressure_bar=args.pressure)
        else:
            record, budget, callbacks, initial, metadata = build_bse_problem(
                args.inventory, args.exoeos_checkout, args.runtime, args.python,
                temperature_k=args.temperature, pressure_bar=args.pressure, gas_model=args.gas_model)
        report.update(source_metadata=metadata, record=record, element_amounts_mol=budget.tolist(),
                      canonical_initial_component_amounts_mol=initial.tolist())
        metadata["provenance"]["file_sha256"]["run_m2_contact.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.metal_mode == "select":
            from exoeos import MaFeSiOHLiquid
            from exoeos.ma_interval import ma_alloy_curvature_lower_bound
            from phase_selection import select_metal_phase
            from run_metal_selection import METAL_LOWER, METAL_UPPER
            curvature = ma_alloy_curvature_lower_bound(MaFeSiOHLiquid(), args.temperature, METAL_LOWER, METAL_UPPER)
            selection = select_metal_phase(record, budget, args.temperature, args.pressure, callbacks,
                METAL_LOWER, METAL_UPPER, convex_phase_bounds={"metal": curvature}, maxiter=args.maxiter)
            report["metal_selection"] = asdict(selection)
            if selection.result is None:
                raise ValueError("Metal selection did not return an accepted local state.")
            result = selection.result
        else:
            phases = tuple(phase for phase in record["phases"] if phase != "metal")
            problem = build_problem(record, budget, lambda t, p: np.zeros(len(initial)), phases=phases)
            result = minimize_gibbs(problem, args.temperature, args.pressure, budget,
                                    {name: callbacks[name] for name in problem.phases}, maxiter=args.maxiter)
        report["source_result"] = asdict(result)
        if args.gas_model == "m1_retained":
            report["source_internal_record"] = record
            report["source_internal_result"] = asdict(result)
            record, public, callbacks, parcel = unpack_expanded_source(
                record, asdict(result), callbacks, args.temperature, args.pressure)
            report.update(record=record, source_result=public, source_atmosphere=parcel)
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
