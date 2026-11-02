"""Audit C3 energy, thermal functions and source conventions offline."""

from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs


DATA = ROOT / "tests/data/validation/c3_source_review/20261102"
TABLES = {"C3": "C-138.txt", "C": "C-003.txt"}
STOICHIOMETRY = {"C3": 1, "C": -3}
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
JANAF_R = 8.31441
NASA_R = 8.31451


def archived_sources() -> list:
    """Verify downloaded/extracted source bytes before numerical use."""
    manifest = json.loads((DATA / "sources.json").read_text())
    paths = []
    for source in manifest["downloads"]:
        if source["archived"]:
            path = DATA / source["file"]
            if file_hash(path) != source["sha256"]:
                raise ValueError("Source hash mismatch: " + source["file"])
            paths.append(path)
    return paths


def nasa_headers() -> dict:
    """Read original thermal anchors without extrapolating polynomials to 0 K."""
    records = {}
    for block in (DATA / "nasa_selected_records.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        records[lines[0].split()[0]] = {
            "Hf298_J_mol": float(lines[1].split()[-1]),
            "molar_mass_g_mol": float(lines[1].split()[-2]),
            "H298_minus_H0_J_mol": float(lines[2].split()[-1]),
            "intervals_K": [[float(value) for value in re.findall(r"\d+\.\d{3}", line)[:2]]
                            for line in lines[2::3]],
            "coefficients": [float(value.replace("D", "E"))
                             for value in re.findall(r"[+-]?\d+\.\d+D[+-]\d+", block)],
        }
    return records


def fastchem_record() -> tuple:
    """Extract the unchanged five-coefficient current C3 fit."""
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("C3 "))
    coefficients = [float(value) for value in lines[index + 1].split()]
    if len(coefficients) != 5:
        raise ValueError("Expected five FastChem C3 coefficients")
    return coefficients, lines[index]


def fit_properties(coefficients: list, temperature: float, gas_constant: float) -> tuple:
    """Return fitted reaction ln K, H, S and Cp from analytic derivatives."""
    a, b, c, d, e = coefficients
    t, r = temperature, gas_constant
    ln_k = a/t + b*math.log(t) + c + d*t + e*t*t
    h = r*(-a + b*t + d*t*t + 2*e*t**3)
    return ln_k, h, r*ln_k+h/t, r*(b+2*d*t+6*e*t*t)


def janaf_properties(table: dict, temperature: float) -> tuple:
    """Return H, S and Cp using the explicit 298.15 K table anchor."""
    node = table[temperature]
    return (table[298.15][4]+node[3])*1000, node[1], node[0]


def reference_audit(tables: dict) -> dict:
    """Check graphite-reference closure and printed/online agreement."""
    graphite = read_table(DATA / "C-002.txt")
    molecule = tables["C3"]
    rows = []
    for t in sorted(molecule.keys() & graphite.keys()):
        if t <= 0:
            continue
        m, c = molecule[t], graphite[t]
        hf = molecule[298.15][4]+m[3]-3*c[3]
        gf = hf-t*(m[1]-3*c[1])/1000
        rows.append({"temperature_K": t, "formation_H_error_kJ_mol": m[4]-hf,
                     "formation_G_error_kJ_mol": m[5]-gf})
    printed = json.loads((DATA / "printed_c3_nodes.json").read_text())
    for row in printed["rows"]:
        expected = [row[name] for name in ("cp_J_mol_K", "entropy_J_mol_K",
                    "enthalpy_increment_kJ_mol", "formation_enthalpy_kJ_mol", "formation_gibbs_kJ_mol")]
        node = molecule[row["temperature_K"]]
        if expected != [node[i] for i in (0, 1, 3, 4, 5)]:
            raise ValueError("Printed and online C3 table nodes differ")
    return {"reference": "3 C(graphite, ref) -> C3(g)", "samples": rows,
            "printed_online_matching_nodes": len(printed["rows"]),
            "max_abs_formation_H_error_kJ_mol": max(abs(row["formation_H_error_kJ_mol"]) for row in rows),
            "max_abs_formation_G_error_kJ_mol": max(abs(row["formation_G_error_kJ_mol"]) for row in rows)}


def partition_heat_capacity(partition: dict, temperature: float, step: float, gas_constant: float) -> float:
    """Evaluate Cp from five-point derivatives of ln Q at exact saved nodes."""
    t, h = temperature, step
    a, b, c, d, e = [math.log(partition[t+i*h]) for i in (-2, -1, 0, 1, 2)]
    first = (a-8*b+8*d-e)/(12*h)
    second = (-a+16*b-30*c+16*d-e)/(12*h*h)
    return gas_constant*(2.5+2*t*first+t*t*second)


def summarize_states(path: Path) -> dict:
    """Reduce an optional full AtLast state file to independently checkable moments."""
    import numpy as np

    with bz2.open(path, "rt") as stream:
        energy, weight, angular, bending, calculated = np.loadtxt(stream, usecols=(1, 2, 3, 9, 17), unpack=True)
    if not np.array_equal(weight, 2*angular+1):
        raise ValueError("AtLast state weights do not equal 2J+1")
    c2 = 6.62607015e-34*299792458*100/1.380649e-23
    rows = []
    for t in (1., 300., 500., 1000., 1500., 2000., 3000., 4000.):
        z = c2*energy/t
        terms = weight*np.exp(-z)
        q = float(terms.sum())
        rows.append({"temperature_K": t, "Q": q,
                     "Q_using_auxiliary_calculated_energies": float(np.sum(weight*np.exp(-c2*calculated/t))),
                     "mean_E_over_kT": float(terms@z/q),
                     "mean_squared_E_over_kT": float(terms@(z*z)/q),
                     "fraction_E_above_15000_cm-1": float(terms[energy > 15000].sum()/q),
                     "fraction_J_above_130": float(terms[angular > 130].sum()/q),
                     "fraction_l2_above_10": float(terms[bending > 10].sum()/q)})
    return {"schema_version": "1", "source_url": "https://www.exomol.com/db/C3/12C3/AtLast/12C3__AtLast.states.bz2",
            "source_file_sha256": file_hash(path), "implementation_sha256": file_hash(Path(__file__)),
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv], "python": platform.python_version(), "numpy": np.__version__,
            "columns": "Energy column 2 in cm^-1, total weight column 3, J column 4, |l2| column 10; moments use final energies. Auxiliary calculated energies (column 18) provide a separate Q cross-check.",
            "state_count": int(energy.size), "all_weights_equal_2J_plus_1": True,
            "minimum_energy_cm-1": float(energy.min()), "maximum_energy_cm-1": float(energy.max()),
            "maximum_J": float(angular.max()), "maximum_l2": float(bending.max()),
            "second_radiation_constant_cm_K": c2,
            "interpretation": "Moments of the downloaded finite ground-electronic-state list. Coverage fractions describe included states near declared bounds, not a measured missing-state tail or physical completeness.",
            "samples": rows}


def partition_comparison(tables: dict, inputs: dict, headers: dict) -> dict:
    """Compare internal Q with the 12C3 ground-state AtLast state sum."""
    exomol = {float(row[0]): float(row[1]) for line in (DATA / "exomol_atlast_12C3.pf").read_text().splitlines()
              if len(row := line.split()) == 2}
    kb, planck, avogadro = 1.380649e-23, 6.62607015e-34, 6.02214076e23
    mass = headers["C3"]["molar_mass_g_mol"]*.001/avogadro
    nasa_h0 = headers["C3"]["Hf298_J_mol"]-headers["C3"]["H298_minus_H0_J_mol"]
    janaf_h0 = (tables["C3"][298.15][4]+tables["C3"][0][3])*1000
    rows, caloric = [], []
    for t in (300., 500., 1000., 1500., 2000., 3000., 4000.):
        jh, js, _ = janaf_properties(tables["C3"], t)
        nh, ns, _ = nasa_properties(inputs["nasa9_records"]["C3"], t, NASA_R)
        translation = (2*math.pi*mass*kb*t/planck**2)**1.5*kb*t/inputs["standard_pressure_Pa"]
        jq = math.exp((janaf_h0-jh+t*js)/(JANAF_R*t))/translation
        nq = math.exp((nasa_h0-nh+t*ns)/(NASA_R*t))/translation
        q = exomol[t]
        rows.append({"temperature_K": t, "AtLast_Q": q, "JANAF_inferred_Q": jq,
                     "NASA_inferred_Q": nq, "AtLast_over_JANAF": q/jq, "AtLast_over_NASA": q/nq})
        if t >= 1500:
            estimates = {str(step): partition_heat_capacity(exomol, t, step, inputs["gas_constant_J_mol_K"])
                         for step in (10, 20, 40)}
            caloric.append({"temperature_K": t, "AtLast_Cp_J_mol_K_by_step_K": estimates,
                            "step_spread_J_mol_K": max(estimates.values())-min(estimates.values()),
                            "JANAF_Cp_J_mol_K": tables["C3"][t][0],
                            "NASA_Cp_J_mol_K": nasa_properties(inputs["nasa9_records"]["C3"], t, inputs["gas_constant_J_mol_K"])[2]})
    state_sum = json.loads((DATA / "state_sum_checks.json").read_text())
    state_checks = []
    for row in state_sum["samples"]:
        t = row["temperature_K"]
        state_checks.append({"temperature_K": t, "state_sum_over_published_Q": row["Q"]/exomol[t],
                             "auxiliary_calculated_sum_over_published_Q": row["Q_using_auxiliary_calculated_energies"]/exomol[t],
                             "state_sum_Cp_J_mol_K": inputs["gas_constant_J_mol_K"]*(
                                 2.5+row["mean_squared_E_over_kT"]-row["mean_E_over_kT"]**2)})
    return {"formula": "Qint = exp((H0-H(T)+T*S(T))/(R*T)) / ((2*pi*m*kB*T/h^2)^(3/2)*kB*T/p0)",
            "JANAF_gas_constant_J_mol_K": JANAF_R, "NASA_gas_constant_J_mol_K": NASA_R,
            "molar_mass_g_mol": headers["C3"]["molar_mass_g_mol"], "nuclear_spin_divisor": 1,
            "convention": "12C has spin zero; AtLast's spin/parity restrictions are already included. Use source R for Q inference and the explicit legacy molecular mass for both source comparisons.",
            "scope": "The AtLast ground-electronic-state sum is a lower limit to an electronically complete partition function; state-sum numerical convergence does not certify high-temperature completeness. Isotopic mass conventions differ slightly.",
            "definition_file_discrepancy": "The archived .def says Sigma+/Sigma- weights 6, conflicting with the original paper and actual 12C3 state weights 2J+1. No division by 6 is applied. Ground-state weight 1 and the low-temperature Q support the spin-zero convention.",
            "partition_file_discrepancy": "The direct sum of all downloaded state weights/energies does not reproduce the published .pf at sampled temperatures. Using auxiliary calculated energies does not remove this discrepancy. Its cause is unresolved; this is not a certified replacement thermal model.",
            "independent_state_sum": state_checks,
            "caloric_diagnostic": {"formula": "Cp/R = 5/2 + 2*T*d(ln Q)/dT + T^2*d2(ln Q)/dT2",
                                   "method": "Five-point differences at exact 1 K nodes, steps 10, 20 and 40 K; spread measures differentiation stability, not physical uncertainty.",
                                   "gas_constant_J_mol_K": inputs["gas_constant_J_mol_K"], "samples": caloric},
            "samples": rows}


def energy_anchors(tables: dict, headers: dict) -> dict:
    """Balance independent source energies against their own C atoms."""
    atct = json.loads((DATA / "atct_evidence.json").read_text())
    species = atct["species"]
    rows = {}
    for label, key in (("0", "Hf0_kJ_mol"), ("298", "Hf298_kJ_mol")):
        atct_reaction = (species["C3_linear"][key]-3*species["C_atom"][key])*1000
        if label == "0":
            janaf = sum(count*tables[name][0][4]*1000 for name, count in STOICHIOMETRY.items())
            nasa = sum(count*(headers[name]["Hf298_J_mol"]-headers[name]["H298_minus_H0_J_mol"])
                       for name, count in STOICHIOMETRY.items())
        else:
            janaf = sum(count*tables[name][298.15][4]*1000 for name, count in STOICHIOMETRY.items())
            nasa = sum(count*headers[name]["Hf298_J_mol"] for name, count in STOICHIOMETRY.items())
        rows[label] = {"temperature_K": 0 if label == "0" else 298.15,
                       "JANAF_reaction_H_J_mol": janaf, "NASA_header_reaction_H_J_mol": nasa,
                       "ATcT_reaction_H_J_mol": atct_reaction,
                       "ATcT_minus_JANAF_reaction_H_J_mol": atct_reaction-janaf,
                       "ATcT_minus_NASA_reaction_H_J_mol": atct_reaction-nasa}
    return {"reaction": "3 C(g) -> linear C3(g)", "ATcT_version": atct["version"],
            "interpretation": "Balanced source energy anchors. 0 K atomization is the negative reaction energy; it is not a single C-C bond dissociation energy. NASA headers, not a polynomial extrapolation, supply 0 K increments.",
            "uncertainty": "ATcT estimated 95 percent limits and the reported C3/C correlation are preserved in atct_evidence.json. The rounded correlation is not a full covariance record; no independent-error combination or source-difference uncertainty distribution is assigned.",
            "reported_C3_C_correlation": atct["correlation"]["C3_linear_C_atom"],
            "anchors": rows}


def historical_entropy_comparison(tables: dict, inputs: dict, headers: dict) -> dict:
    """Keep conflicting historical entropy temperatures and conventions explicit."""
    evidence = json.loads((DATA / "historical_entropy_evidence.json").read_text())
    q = {float(row[0]): float(row[1]) for line in (DATA / "exomol_atlast_12C3.pf").read_text().splitlines()
         if len(row := line.split()) == 2}
    kb, planck, avogadro = 1.380649e-23, 6.62607015e-34, 6.02214076e23
    mass, r = headers["C3"]["molar_mass_g_mol"]*.001/avogadro, inputs["gas_constant_J_mol_K"]
    shift = r*math.log(101325/inputs["standard_pressure_Pa"])
    groups = {}
    for group in ("samples", "primary_abstract_samples"):
        rows = []
        samples = evidence["samples"] if group == "samples" else evidence["primary_abstract"]["samples"]
        for sample in samples:
            t, step = sample["temperature_K"], 20
            a, b, c, d = [math.log(q[t+i*step]) for i in (-2, -1, 1, 2)]
            derivative = (a-8*b+8*c-d)/(12*step)
            translation = (2*math.pi*mass*kb*t/planck**2)**1.5*kb*t/inputs["standard_pressure_Pa"]
            quoted = sample["entropy_cal_mol_K"]*evidence["calorie_J"]
            rows.append({"temperature_K": t, "quoted_entropy_J_mol_K": quoted,
                         "if_quoted_at_1atm_entropy_at_1bar_J_mol_K": quoted+shift,
                         "JANAF_entropy_J_mol_K": tables["C3"][t][1],
                         "NASA_entropy_J_mol_K": nasa_properties(inputs["nasa9_records"]["C3"], t, r)[1],
                         "AtLast_published_Q_entropy_J_mol_K": r*(math.log(q[t]*translation)+2.5+t*derivative)})
        groups[group] = rows
    return {"calorie_J": evidence["calorie_J"], "conditional_1atm_to_1bar_shift_J_mol_K": shift,
            "AtLast_convention": "Published Q, a 20 K five-point first derivative, common source mass and baseline R; the separate .pf/state-sum discrepancy is unresolved.",
            "interpretation": "Jones quotes 2400/2700 K; the authors' ASMS abstract states 2400/2600 K for the same entropy numbers. Neither the journal article nor a resolution of this temperature discrepancy is available. The 1 atm conversion is conditional. These linked records are not independent measurements or validation of an adopted thermal model.",
            **groups}


def review() -> dict:
    """Separate reaction energy, thermal functions, atomic terms and fit errors."""
    archived_sources()
    inputs = load_inputs()
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    headers = nasa_headers()
    for name, source in headers.items():
        record = inputs["nasa9_records"][name]
        expected = [value for interval in record["intervals"] for value in interval["coefficients"]]
        if source["coefficients"] != expected or source["intervals_K"] != [row["bounds_K"] for row in record["intervals"]]:
            raise ValueError("Original NASA record does not match the frozen baseline")
    coefficients, fit_label = fastchem_record()
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    r = inputs["gas_constant_J_mol_K"]
    anchor_checks = []
    for name, header in headers.items():
        record = inputs["nasa9_records"][name]
        original = nasa_properties(record, 298.15, NASA_R)[0]
        common = nasa_properties(record, 298.15, r)[0]
        anchor_checks.append({"species": name, "header_H298_J_mol": header["Hf298_J_mol"],
                              "source_R_polynomial_H298_J_mol": original,
                              "common_R_polynomial_H298_J_mol": common,
                              "source_R_fit_residual_J_mol": original-header["Hf298_J_mol"],
                              "gas_constant_conversion_J_mol": common-original})
    rows = []
    for t in inputs["temperatures_K"]:
        properties = {}
        components = dict.fromkeys(("anchor", "enthalpy_increment", "entropy_at_298", "entropy_increment"), 0.)
        janaf_ln_k, raw_ln_k, nasa_ln_k = 0., 0., 0.
        for name, count in STOICHIOMETRY.items():
            jh, js, jcp = janaf_properties(tables[name], t)
            jh0, js0, _ = janaf_properties(tables[name], 298.15)
            nh, ns, ncp = nasa_properties(inputs["nasa9_records"][name], t, r)
            nh0, ns0, _ = nasa_properties(inputs["nasa9_records"][name], 298.15, r)
            delta = {"anchor": jh0-nh0, "enthalpy_increment": (jh-jh0)-(nh-nh0),
                     "entropy_at_298": -t*(js0-ns0), "entropy_increment": -t*((js-js0)-(ns-ns0))}
            for key, value in delta.items():
                components[key] += count*value
            properties[name] = {"janaf_H298_J_mol": jh0, "nasa_H298_J_mol": nh0,
                                "janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": js, "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": jcp, "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh-nh-t*(js-ns)}
            janaf_ln_k -= count*(jh-t*js)/(r*t)
            nasa_ln_k -= count*(nh-t*ns)/(r*t)
            raw_ln_k -= count*tables[name][t][5]*1000/(r*t)
        old = next(row for row in baseline["samples"] if row["species_id"] == "C3" and row["temperature_K"] == t)
        for key, value in (("janaf", janaf_ln_k), ("nasa", nasa_ln_k)):
            if not math.isclose(value, old["ln_Kp"][key], abs_tol=2e-11, rel_tol=0):
                raise ValueError("Fresh tables do not reproduce the C3 baseline")
        curated = old["ln_Kp"]["curated"]
        fitted = None if curated is None else fit_properties(coefficients, t, r)[0]
        if curated is not None and not math.isclose(fitted, curated, abs_tol=2e-11, rel_tol=0):
            raise ValueError("Current C3 fit differs from the archived curated row")
        delta_g = sum(components.values())
        rows.append({"temperature_K": t, "properties": properties,
                     "reaction_delta_G_components_J_mol": components, "reaction_delta_G_J_mol": delta_g,
                     "molecular_delta_G_J_mol": properties["C3"]["delta_G_J_mol"],
                     "atomic_delta_G_J_mol": delta_g-properties["C3"]["delta_G_J_mol"],
                     "janaf_over_nasa": math.exp(janaf_ln_k-nasa_ln_k),
                     "energy_aligned_janaf_over_nasa": math.exp(-(delta_g-components["anchor"])/(r*t)),
                     "curated_over_nasa": None if curated is None else math.exp(curated-nasa_ln_k),
                     "janaf_internal_difference_dex": (raw_ln_k-janaf_ln_k)/math.log(10),
                     "ln_Kp": {"janaf_HS": janaf_ln_k, "janaf_formation_columns": raw_ln_k,
                               "nasa": nasa_ln_k, "fastchem_fit": fitted, "curated": curated},
                     "baseline_closure_ln_K": -delta_g/(r*t)-(old["ln_Kp"]["janaf"]-old["ln_Kp"]["nasa"])})
    fit_checks = []
    for t in sorted(set.intersection(*(set(table) for table in tables.values()))):
        if not 1500 <= t <= 4000:
            continue
        jh, js, jcp = [sum(count*janaf_properties(tables[name], t)[i] for name, count in STOICHIOMETRY.items()) for i in range(3)]
        fitted, fh, fs, fcp = fit_properties(coefficients, t, r)
        fit_checks.append({"temperature_K": t, "fit_minus_JANAF_ln_K": fitted+(jh-t*js)/(r*t),
                           "fit_minus_JANAF_reaction_H_J_mol": fh-jh,
                           "fit_minus_JANAF_reaction_S_J_mol_K": fs-js,
                           "fit_minus_JANAF_reaction_Cp_J_mol_K": fcp-jcp,
                           "JANAF_molecular_Cp_J_mol_K": tables["C3"][t][0],
                           "NASA_molecular_Cp_J_mol_K": nasa_properties(inputs["nasa9_records"]["C3"], t, r)[2]})
    joins = []
    for name in STOICHIOMETRY:
        record = inputs["nasa9_records"][name]
        for low, high in zip(record["intervals"], record["intervals"][1:]):
            t = low["bounds_K"][1]
            sides = [nasa_properties({"intervals": [interval]}, t, r) for interval in (low, high)]
            joins.append({"species": name, "temperature_K": t,
                          **{key: sides[1][i]-sides[0][i] for i, key in enumerate(
                              ("high_minus_low_H_J_mol", "high_minus_low_S_J_mol_K", "high_minus_low_Cp_J_mol_K"))}})
    return {"schema_version": "1", "reaction": "3 C(g) -> C3(g)",
            "standard_pressure_Pa": inputs["standard_pressure_Pa"], "gas_constant_J_mol_K": r,
            "reference_temperature_K": 298.15, "delta_convention": "JANAF minus NASA, molecule minus atoms",
            "energy_alignment": "Diagnostic reaction H298 shift only, not a model adoption",
            "NASA_interval_join_checks": joins, "NASA_298_anchor_checks": anchor_checks,
            "source_intervals": {name: row["intervals_K"] for name, row in headers.items()},
            "formation_reference_audit": reference_audit(tables), "energy_anchors": energy_anchors(tables, headers),
            "historical_entropy_comparison": historical_entropy_comparison(tables, inputs, headers),
            "fastchem_fit": {"source_label": fit_label, "coefficients": coefficients, "diagnostic_range_K": [1500, 4000],
                             "max_abs_JANAF_residual_ln_K": max(abs(row["fit_minus_JANAF_ln_K"]) for row in fit_checks),
                             "samples": fit_checks},
            "partition_comparison": partition_comparison(tables, inputs, headers), "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--states-file", type=Path)
    parser.add_argument("--states-output", type=Path)
    args = parser.parse_args()
    if args.states_file or args.states_output:
        if not (args.states_file and args.states_output) or args.output or args.receipt:
            parser.error("Use --states-file and --states-output together, separately from source replay")
        args.states_output.parent.mkdir(parents=True, exist_ok=True)
        args.states_output.write_text(json.dumps(summarize_states(args.states_file), indent=2, allow_nan=False)+"\n")
        return
    if args.receipt and args.output is None:
        parser.error("--receipt requires --output")
    result = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end="")
    if args.receipt:
        selected = [Path(__file__), ROOT / "mceg/reviews/c2h_source_review.py", ROOT / "mceg/reviews/gas_source_comparison.py",
                    FASTCHEM, BASELINE / "inputs.json", BASELINE / "comparison.json",
                    DATA / "sources.json", DATA / "literature.json", *archived_sources()]
        receipt = {"schema_version": "1", "study": "c3-source-review-20261102", "execution_kind": "fresh_offline_execution",
                   "executed_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                   "python": platform.python_version(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "implementation_identity": "Selected file hashes include uncommitted files; HEAD alone does not identify this execution.",
                   "selected_input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in selected},
                   "output_sha256": {str(args.output.resolve().relative_to(ROOT)): file_hash(args.output)},
                   "scope": "C3 energy and thermal-source audit, not a model adoption or equilibrium execution", "result": "passed"}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
