"""Audit PH2 source versions and elemental references using offline evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs


DATA = ROOT / "tests/data/validation/ph2_source_review/20261102"
TABLES = {"PH2": "H-079.txt", "P": "P-008.txt", "H": "H-001.txt",
          "H2": "H-050.txt"}
STOICHIOMETRY = {"PH2": 1, "P": -1, "H": -2}
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"


def archived_sources() -> list:
    """Verify the source manifest and return its locally archived files."""
    sources = json.loads((DATA / "sources.json").read_text())
    paths = []
    for source in sources["downloads"]:
        if source["archived"]:
            path = DATA / source["file"]
            if file_hash(path) != source["sha256"]:
                raise ValueError("Source hash mismatch: " + source["file"])
            paths.append(path)
    return paths


def phosphorus_reference() -> dict:
    """Read regular P(ref) nodes; transition rows have two phase-specific values."""
    return {float(row[0]): [float(value.replace("INFINITE", "inf")) for value in row[1:]]
            for line in (DATA / "P-001.txt").read_text().splitlines()[2:]
            if len(row := line.split()) == 8 and row[-3:] == ["0.", "0.", "0."]}


def fastchem_record() -> tuple:
    """Extract the unchanged five-coefficient PH2 fit and its source label."""
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("H2P1 "))
    coefficients = [float(value) for value in lines[index+1].split()]
    if len(coefficients) != 5:
        raise ValueError("Expected five FastChem PH2 coefficients")
    return coefficients, lines[index]


def fit_ln_k(coefficients: list, temperature: float) -> float:
    """Evaluate FastChem's dimensionless atom-to-molecule ln Kp fit."""
    a, b, c, d, e = coefficients
    t = temperature
    return a/t + b*math.log(t) + c + d*t + e*t*t


def formation_reference_audit(tables: dict, reference: dict, printed: dict) -> dict:
    """Test elemental-reference signatures without assigning a historical cause."""
    molecule, hydrogen = tables["PH2"], tables["H2"]
    online_anchor = molecule[298.15][4]
    printed_anchor = next(row["formation_enthalpy_kJ_mol"] for row in printed["rows"]
                          if row["temperature_K"] == 298.15)
    rows = []
    for t in sorted(molecule.keys() & hydrogen.keys()):
        if t <= 0:
            continue
        m, h, p = molecule[t], hydrogen[t], reference.get(t)
        wrong_h = online_anchor + m[3] - 1.5*h[3]
        wrong_g = wrong_h - t*(m[1]-1.5*h[1])/1000
        correct_h = None if p is None else online_anchor + m[3] - h[3] - p[3]
        correct_g = None if p is None else correct_h - t*(m[1]-h[1]-p[1])/1000
        rows.append({"temperature_K": t,
                     "online_Hf_kJ_mol": m[4], "online_Gf_kJ_mol": m[5],
                     "Hf_with_1p5_H2_reference_kJ_mol": wrong_h,
                     "Gf_with_1p5_H2_reference_kJ_mol": wrong_g,
                     "online_minus_1p5_H2_Hf_kJ_mol": m[4]-wrong_h,
                     "online_minus_1p5_H2_Gf_kJ_mol": m[5]-wrong_g,
                     "Hf_with_P_ref_plus_H2_kJ_mol": correct_h,
                     "Gf_with_P_ref_plus_H2_kJ_mol": correct_g})
    printed_checks = []
    for row in printed["rows"]:
        t = row["temperature_K"]
        m, h, p = molecule[t], hydrogen[t], reference[t]
        if (row["cp_J_mol_K"], row["entropy_J_mol_K"], row["enthalpy_increment_kJ_mol"]) != (m[0], m[1], m[3]):
            raise ValueError("Printed and online PH2 thermal columns differ")
        correct_h = printed_anchor + m[3] - h[3] - p[3]
        correct_g = correct_h - t*(m[1]-h[1]-p[1])/1000
        printed_checks.append({"temperature_K": t,
                               "printed_minus_P_ref_plus_H2_Hf_kJ_mol": row["formation_enthalpy_kJ_mol"]-correct_h,
                               "printed_minus_P_ref_plus_H2_Gf_kJ_mol": row["formation_gibbs_kJ_mol"]-correct_g})
    atomic_checks = []
    for t in (298.15, 500, 1000, 1500, 3000):
        for name, elemental, count in (("P", reference[t], 1), ("H", hydrogen[t], 0.5)):
            atom = tables[name][t]
            hf = tables[name][298.15][4]+atom[3]-count*elemental[3]
            gf = hf-t*(atom[1]-count*elemental[1])/1000
            atomic_checks.append({"species": name, "temperature_K": t,
                                  "online_minus_elemental_reference_Hf_kJ_mol": atom[4]-hf,
                                  "online_minus_elemental_reference_Gf_kJ_mol": atom[5]-gf})
    return {"interpretation": "Online H-079 formation columns reproduce a 1.5 H2 reference instead of P(ref) + H2. This numerical signature does not establish how the source acquired the discrepancy.",
            "phosphorus_reference": "P-001 JANAF reference table, including its phase transitions; not P(g)",
            "reference_signature_formula": "Hf = Hf298 + dH(PH2) - 1.5*dH(H2); Gf = Hf - T*(S(PH2)-1.5*S(H2))/1000, in kJ/mol",
            "missing_P_ref_nodes": "Correct-reference values are null where P-001 has no exact node; no phase interpolation is performed.",
            "online_anchor_kJ_mol": online_anchor,
            "printed_1998_anchor_kJ_mol": printed_anchor,
            "online_reference_signature_samples": rows,
            "max_abs_online_minus_1p5_H2_Hf_kJ_mol": max(abs(row["online_minus_1p5_H2_Hf_kJ_mol"]) for row in rows),
            "max_abs_online_minus_1p5_H2_Gf_kJ_mol": max(abs(row["online_minus_1p5_H2_Gf_kJ_mol"]) for row in rows),
            "printed_1998_reference_checks": printed_checks,
            "atomic_reference_checks": atomic_checks}


def review() -> dict:
    """Separate source versions, reaction energy, thermal and atomic effects."""
    archived_sources()
    inputs = load_inputs()
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    reference = phosphorus_reference()
    printed = json.loads((DATA / "printed_ph2_nodes.json").read_text())
    reference_audit = formation_reference_audit(tables, reference, printed)
    anchor_shift = 1000*(reference_audit["printed_1998_anchor_kJ_mol"]-
                         reference_audit["online_anchor_kJ_mol"])
    coefficients, fit_label = fastchem_record()
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    r = inputs["gas_constant_J_mol_K"]
    rows = []
    for t in inputs["temperatures_K"]:
        properties = {}
        components = dict.fromkeys(("anchor", "enthalpy_increment", "entropy_at_298",
                                    "entropy_increment"), 0.0)
        online_ln_k, raw_ln_k, nasa_ln_k = 0.0, 0.0, 0.0
        for name, count in STOICHIOMETRY.items():
            node, at_298 = tables[name][t], tables[name][298.15]
            nh, ns, ncp = nasa_properties(inputs["nasa9_records"][name], t, r)
            nh0, ns0, _ = nasa_properties(inputs["nasa9_records"][name], 298.15, r)
            jh0, js0 = at_298[4]*1000, at_298[1]
            jh, js = jh0+node[3]*1000, node[1]
            delta = {"anchor": jh0-nh0,
                     "enthalpy_increment": (jh-jh0)-(nh-nh0),
                     "entropy_at_298": -t*(js0-ns0),
                     "entropy_increment": -t*((js-js0)-(ns-ns0))}
            for key, value in delta.items():
                components[key] += count*value
            properties[name] = {"online_janaf_H298_J_mol": jh0, "nasa_H298_J_mol": nh0,
                                "online_janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": js, "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": node[0], "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh-nh-t*(js-ns)}
            online_ln_k -= count*(jh-t*js)/(r*t)
            nasa_ln_k -= count*(nh-t*ns)/(r*t)
            raw_ln_k -= count*node[5]*1000/(r*t)
        delta_g = sum(components.values())
        old = next(row for row in baseline["samples"]
                   if row["species_id"] == "H2P1" and row["temperature_K"] == t)
        for key, value in (("janaf", online_ln_k), ("nasa", nasa_ln_k)):
            if not math.isclose(value, old["ln_Kp"][key], abs_tol=2e-11, rel_tol=0):
                raise ValueError("Fresh tables do not reproduce the PH2 baseline")
        fitted = fit_ln_k(coefficients, t)
        curated = old["ln_Kp"]["curated"]
        if curated is not None and not math.isclose(fitted, curated, abs_tol=2e-11, rel_tol=0):
            raise ValueError("PH2 FastChem fit no longer matches the archived curated row")
        printed_ln_k = online_ln_k-anchor_shift/(r*t)
        rows.append({"temperature_K": t, "properties": properties,
                     "reaction_delta_G_components_J_mol": components,
                     "reaction_delta_G_J_mol": delta_g,
                     "printed_1998_reaction_delta_G_J_mol": delta_g+anchor_shift,
                     "molecular_delta_G_J_mol": properties["PH2"]["delta_G_J_mol"],
                     "atomic_delta_G_J_mol": delta_g-properties["PH2"]["delta_G_J_mol"],
                     "janaf_over_nasa": math.exp(online_ln_k-nasa_ln_k),
                     "printed_1998_janaf_over_nasa": math.exp(printed_ln_k-nasa_ln_k),
                     "energy_aligned_janaf_over_nasa": math.exp(-(delta_g-components["anchor"])/(r*t)),
                     "curated_over_nasa": None if curated is None else math.exp(curated-nasa_ln_k),
                     "fastchem_minus_printed_1998_HS_ln_K": fitted-printed_ln_k,
                     "janaf_internal_difference_dex": (raw_ln_k-online_ln_k)/math.log(10),
                     "ln_Kp": {"online_janaf_HS": online_ln_k,
                               "online_janaf_formation_columns": raw_ln_k,
                               "printed_1998_janaf_HS": printed_ln_k,
                               "nasa": nasa_ln_k, "fastchem_fit": fitted, "curated": curated},
                     "baseline_closure_ln_K": -delta_g/(r*t)-(old["ln_Kp"]["janaf"]-old["ln_Kp"]["nasa"])})
    fit_checks = []
    for t in sorted(tables["PH2"].keys() & tables["P"].keys() & tables["H"].keys()):
        if not 1500 <= t <= 4000:
            continue
        online = -sum(count*((tables[name][298.15][4]+tables[name][t][3])*1000-
                             t*tables[name][t][1]) for name, count in STOICHIOMETRY.items())/(r*t)
        printed_ln_k = online-anchor_shift/(r*t)
        fit_checks.append({"temperature_K": t,
                           "fastchem_minus_printed_1998_HS_ln_K": fit_ln_k(coefficients, t)-printed_ln_k})
    return {"schema_version": "1", "reaction": "P(g) + 2 H(g) -> PH2(g)",
            "standard_pressure_Pa": inputs["standard_pressure_Pa"],
            "gas_constant_J_mol_K": r, "reference_temperature_K": 298.15,
            "delta_convention": "Online JANAF minus raw NASA, molecule minus atoms, unless explicitly marked printed_1998",
            "energy_alignment": "Diagnostic reaction H298 shift only; no change to NASA data or curated model",
            "formation_reference_audit": reference_audit,
            "fastchem_fit": {"source_label": fit_label, "coefficients": coefficients,
                             "diagnostic_range_K": [1500, 4000],
                             "interpretation": "Agreement with reconstructed printed-1998 JANAF does not establish the historical fit-generation procedure or the physical accuracy of that source.",
                             "max_abs_printed_1998_HS_residual_ln_K": max(abs(row["fastchem_minus_printed_1998_HS_ln_K"]) for row in fit_checks),
                             "samples": fit_checks},
            "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end="")
    if args.receipt:
        if args.output is None:
            parser.error("--receipt requires --output")
        selected = [Path(__file__), ROOT / "mceg/reviews/c2h_source_review.py",
                    ROOT / "mceg/reviews/gas_source_comparison.py", FASTCHEM,
                    BASELINE / "inputs.json", BASELINE / "comparison.json",
                    DATA / "printed_ph2_nodes.json", DATA / "P-001.txt",
                    DATA / "nasa_selected_records.txt",
                    DATA / "sources.json", DATA / "literature.json",
                    *(DATA / name for name in TABLES.values()), *archived_sources()]
        receipt = {"schema_version": "1", "study": "ph2-source-review-20261102",
                   "execution_kind": "fresh_offline_execution",
                   "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "command": [sys.executable, *sys.argv], "python": platform.python_version(),
                   "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "implementation_identity": "Selected file hashes include uncommitted files; HEAD alone does not identify this execution.",
                   "selected_input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in selected},
                   "output_sha256": {str(args.output.resolve().relative_to(ROOT)): file_hash(args.output)},
                   "scope": "PH2 source/version/reference audit; no model adoption or equilibrium execution",
                   "result": "passed"}
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
