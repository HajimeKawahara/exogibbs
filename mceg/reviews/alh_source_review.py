"""Audit AlH energy, thermal functions and source conventions offline."""

from __future__ import annotations

import argparse
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


DATA = ROOT / "tests/data/validation/alh_source_review/20261102"
TABLES = {"ALH": "Al-056.txt", "AL": "Al-005.txt", "H": "H-001.txt"}
STOICHIOMETRY = {"ALH": 1, "AL": -1, "H": -1}
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
MAJOR = ROOT / "examples/comparisons/data/major_molecule_reference.json"
ELECTRONVOLT_J_MOL = 1.602176634e-19 * 6.02214076e23


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
    """Read original thermal anchors and interval bounds without extrapolation."""
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
    """Extract the unchanged five-coefficient current AlH fit."""
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Al1H1 "))
    coefficients = [float(value) for value in lines[index + 1].split()]
    if len(coefficients) != 5:
        raise ValueError("Expected five FastChem AlH coefficients")
    return coefficients, lines[index]


def fit_properties(coefficients: list, temperature: float, gas_constant: float) -> tuple:
    """Return fitted reaction ln K, H, S and Cp from analytic derivatives."""
    a, b, c, d, e = coefficients
    t, r = temperature, gas_constant
    ln_k = a/t + b*math.log(t) + c + d*t + e*t*t
    h = r*(-a + b*t + d*t*t + 2*e*t**3)
    s = r*ln_k + h/t
    cp = r*(b + 2*d*t + 6*e*t*t)
    return ln_k, h, s, cp


def janaf_properties(table: dict, temperature: float) -> tuple:
    """Return H, S and Cp using the explicit 298.15 K table anchor."""
    node = table[temperature]
    return (table[298.15][4] + node[3])*1000, node[1], node[0]


def reference_audit(tables: dict) -> dict:
    """Check elemental-reference closure and printed/online agreement."""
    aluminum = {float(row[0]): [float(value.replace("INFINITE", "inf")) for value in row[1:]]
                for line in (DATA / "Al-001.txt").read_text().splitlines()[2:]
                if len(row := line.split()) == 8 and row[-3:] == ["0.", "0.", "0."]}
    hydrogen = read_table(DATA / "H-050.txt")
    rows = []
    molecule = tables["ALH"]
    for t in sorted(molecule.keys() & aluminum.keys() & hydrogen.keys()):
        if t <= 0:
            continue
        m, a, h = molecule[t], aluminum[t], hydrogen[t]
        hf = molecule[298.15][4] + m[3] - a[3] - .5*h[3]
        gf = hf-t*(m[1]-a[1]-.5*h[1])/1000
        rows.append({"temperature_K": t, "formation_H_error_kJ_mol": m[4]-hf,
                     "formation_G_error_kJ_mol": m[5]-gf})
    printed = json.loads((DATA / "printed_alh_nodes.json").read_text())
    for row in printed["rows"]:
        expected = [row[name] for name in ("cp_J_mol_K", "entropy_J_mol_K",
                    "enthalpy_increment_kJ_mol", "formation_enthalpy_kJ_mol", "formation_gibbs_kJ_mol")]
        node = molecule[row["temperature_K"]]
        if expected != [node[i] for i in (0, 1, 3, 4, 5)]:
            raise ValueError("Printed and online AlH table nodes differ")
    return {"reference": "Al(ref) + 0.5 H2(ref) -> AlH(g), respecting reference-phase transitions",
            "printed_online_matching_nodes": len(printed["rows"]), "samples": rows,
            "max_abs_formation_H_error_kJ_mol": max(abs(row["formation_H_error_kJ_mol"]) for row in rows),
            "max_abs_formation_G_error_kJ_mol": max(abs(row["formation_G_error_kJ_mol"]) for row in rows)}


def partition_heat_capacity(partition: dict, temperature: float, step: float, gas_constant: float) -> float:
    """Evaluate Cp from five-point derivatives of ln Q at exact saved nodes."""
    t, h = temperature, step
    minus2, minus, middle, plus, plus2 = [math.log(partition[t+i*h]) for i in (-2, -1, 0, 1, 2)]
    first = (minus2-8*minus+8*plus-plus2)/(12*h)
    second = (-minus2+16*minus-30*middle+16*plus-plus2)/(12*h*h)
    return gas_constant*(2.5+2*t*first+t*t*second)


def partition_comparison(tables: dict, inputs: dict, headers: dict, source_r: float) -> dict:
    """Compare molecular internal Q after removing ExoMol nuclear-spin weight."""
    exomol = {float(row[0]): float(row[1]) for line in (DATA / "exomol_aloha_27AlH.pf").read_text().splitlines()
              if len(row := line.split()) == 2}
    kb, planck, avogadro = 1.380649e-23, 6.62607015e-34, 6.02214076e23
    mass = headers["ALH"]["molar_mass_g_mol"]*.001/avogadro
    nasa_h0 = headers["ALH"]["Hf298_J_mol"]-headers["ALH"]["H298_minus_H0_J_mol"]
    janaf_h0 = (tables["ALH"][298.15][4]+tables["ALH"][0][3])*1000
    janaf_r = 8.31441
    rows, caloric = [], []
    for t in (300., 500., 1000., 1500., 2000., 3000., 4000.):
        jh, js, _ = janaf_properties(tables["ALH"], t)
        nh, ns, _ = nasa_properties(inputs["nasa9_records"]["ALH"], t, source_r)
        translation = (2*math.pi*mass*kb*t/planck**2)**1.5*kb*t/inputs["standard_pressure_Pa"]
        jq = math.exp((janaf_h0-jh+t*js)/(janaf_r*t))/translation
        nq = math.exp((nasa_h0-nh+t*ns)/(source_r*t))/translation
        q = exomol[t]/12
        rows.append({"temperature_K": t, "ExoMol_Q_including_nuclear_spin": exomol[t],
                     "ExoMol_Q_without_nuclear_spin": q, "JANAF_inferred_Q": jq,
                     "NASA_inferred_Q": nq, "ExoMol_over_JANAF": q/jq, "ExoMol_over_NASA": q/nq})
        if t >= 1500:
            estimates = {str(step): partition_heat_capacity(exomol, t, step, inputs["gas_constant_J_mol_K"])
                         for step in (10, 20, 40)}
            caloric.append({"temperature_K": t, "ExoMol_Cp_J_mol_K_by_step_K": estimates,
                            "step_spread_J_mol_K": max(estimates.values())-min(estimates.values()),
                            "JANAF_Cp_J_mol_K": tables["ALH"][t][0],
                            "NASA_Cp_J_mol_K": nasa_properties(inputs["nasa9_records"]["ALH"], t, inputs["gas_constant_J_mol_K"])[2]})
    return {"formula": "Qint = exp((H0-H(T)+T*S(T))/(R*T)) / ((2*pi*m*kB*T/h^2)^(3/2)*kB*T/p0)",
            "JANAF_gas_constant_J_mol_K": janaf_r, "NASA_gas_constant_J_mol_K": source_r,
            "convention": "Use the original JANAF 1973-CODATA R=8.31441 and NASA R=8.31451 when inferring Q; the reaction comparison and ExoMol Cp conversion use the frozen baseline R.",
            "molar_mass_g_mol": headers["ALH"]["molar_mass_g_mol"],
            "nuclear_spin_divisor": 12,
            "scope": "AlOHa 27Al-1H X/A bound and quasi-bound internal Q without nuclear spin, versus inferred source Q at matched standard pressure and legacy molecular mass. Its continuum is separate. Additional electronic states and full physical completeness are not established by this diagnostic.",
            "caloric_diagnostic": {"formula": "Cp/R = 5/2 + 2*T*d(ln Q)/dT + T^2*d2(ln Q)/dT2",
                                   "method": "Five-point central differences at exact 1 K table nodes, steps 10, 20 and 40 K; spread measures numerical differentiation sensitivity, not physical uncertainty.",
                                   "gas_constant_J_mol_K": inputs["gas_constant_J_mol_K"], "samples": caloric},
            "samples": rows}


def spectroscopic_dissociation() -> dict:
    """Convert the printed De without repeating its fine-structure offset."""
    evidence = json.loads((DATA / "dissociation_evidence.json").read_text())
    primary = evidence["baltayan_nedelec1979"]
    constants = evidence["zero_point_diagnostic"]
    zpe_cm = constants["omega_e_cm-1"]/2 - constants["omega_ex_e_cm-1"]/4
    zpe_ev = zpe_cm*100*6.62607015e-34*299792458/1.602176634e-19
    szajna = evidence["szajna2011_zero_point_diagnostic"]
    coefficients = szajna["signed_Yk0_cm-1"]
    conversion = 100*6.62607015e-34*299792458/1.602176634e-19
    five_term_cm = math.fsum(value*.5**k for k, value in enumerate(coefficients, 1))
    two_term_cm = coefficients[0]/2 + coefficients[1]/4
    term_checks = []
    for v, observed in enumerate(szajna["X_v0_relative_terms_cm-1"]):
        calculated = math.fsum(value*((v+.5)**k-.5**k)
                               for k, value in enumerate(coefficients, 1))
        term_checks.append({"v": v, "printed_term_cm-1": observed,
                            "calculated_term_cm-1": calculated,
                            "residual_cm-1": calculated-observed})
    return {"printed_De_eV": primary["X_De_eV"], "two_term_ZPE_eV": zpe_ev,
            "diagnostic_D0_eV": primary["X_De_eV"]-zpe_ev,
            "additional_fine_structure_correction_eV": 0.,
            "interpretation": "The original lower X asymptote already includes the stated fine-structure offset. Preserve the earlier rounded two-term diagnostic; the supplied Szajna coefficients give the separate five-term check below. Neither is a new threshold determination or uncertainty interval.",
            "szajna2011_five_term": {
                "ZPE_cm-1": five_term_cm, "ZPE_eV": five_term_cm*conversion,
                "diagnostic_D0_eV": primary["X_De_eV"]-five_term_cm*conversion,
                "higher_than_second_order_ZPE_eV": (five_term_cm-two_term_cm)*conversion,
                "change_from_previous_rounded_two_term_D0_eV": zpe_ev-five_term_cm*conversion,
                "term_checks": term_checks,
                "interpretation": "Evaluate sum(Yk0*(v+1/2)^k), k=1..5, with no Y00/potential-zero correction. Y50 was fixed in the paper; coefficient covariance and its uncertainty were not supplied. Low-v term agreement verifies transcription, not an absolute potential zero, high-v completeness or calibrated D0 uncertainty."}}


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
    major = json.loads(MAJOR.read_text())
    r = inputs["gas_constant_J_mol_K"]
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
        old = next(row for row in baseline["samples"] if row["species_id"] == "Al1H1" and row["temperature_K"] == t)
        for key, value in (("janaf", janaf_ln_k), ("nasa", nasa_ln_k)):
            if not math.isclose(value, old["ln_Kp"][key], abs_tol=2e-11, rel_tol=0):
                raise ValueError("Fresh tables do not reproduce the AlH baseline")
        fitted = fit_properties(coefficients, t, r)[0]
        curated = old["ln_Kp"]["curated"]
        if curated is not None and not math.isclose(fitted, curated, abs_tol=2e-11, rel_tol=0):
            raise ValueError("Current AlH fit differs from the archived curated row")
        delta_g = sum(components.values())
        rows.append({"temperature_K": t, "properties": properties,
                     "reaction_delta_G_components_J_mol": components, "reaction_delta_G_J_mol": delta_g,
                     "molecular_delta_G_J_mol": properties["ALH"]["delta_G_J_mol"],
                     "atomic_delta_G_J_mol": delta_g-properties["ALH"]["delta_G_J_mol"],
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
        nh, ns, ncp = nasa_properties(inputs["nasa9_records"]["ALH"], t, r)
        fit_checks.append({"temperature_K": t, "fit_minus_JANAF_ln_K": fitted+(jh-t*js)/(r*t),
                           "fit_minus_JANAF_reaction_H_J_mol": fh-jh,
                           "fit_minus_JANAF_reaction_S_J_mol_K": fs-js,
                           "fit_minus_JANAF_reaction_Cp_J_mol_K": fcp-jcp,
                           "JANAF_molecular_Cp_J_mol_K": tables["ALH"][t][0], "NASA_molecular_Cp_J_mol_K": ncp})
    nasa_d0 = -sum(count*(headers[name]["Hf298_J_mol"]-headers[name]["H298_minus_H0_J_mol"]) for name, count in STOICHIOMETRY.items())
    janaf_d0 = -sum(count*tables[name][0][4]*1000 for name, count in STOICHIOMETRY.items())
    joins = []
    for name in STOICHIOMETRY:
        record = inputs["nasa9_records"][name]
        sides = [nasa_properties({"intervals": [interval]}, 1000, r) for interval in record["intervals"][:2]]
        joins.append({"species": name, "temperature_K": 1000,
                      **{key: sides[1][i]-sides[0][i] for i, key in enumerate(
                          ("high_minus_low_H_J_mol", "high_minus_low_S_J_mol_K", "high_minus_low_Cp_J_mol_K"))}})
    return {"schema_version": "1", "reaction": "Al(g) + H(g) -> AlH(g)",
            "standard_pressure_Pa": inputs["standard_pressure_Pa"], "gas_constant_J_mol_K": r,
            "reference_temperature_K": 298.15, "delta_convention": "JANAF minus NASA, molecule minus atoms",
            "source_intervals": {"official_NASA_ALH": headers["ALH"]["intervals_K"],
                                 "earlier_major_fixture_ALH": [row["bounds_K"] for row in major["nasa9_records"]["AlH"]["intervals"]],
                                 "note": "The official report and frozen source baseline start at 200 K; the older major-molecule fixture states 300 K for the same low-temperature coefficients. No records are overwritten or extrapolated."},
            "energy_alignment": "Diagnostic reaction H298 shift only, not a new model adoption",
            "NASA_interval_join_checks": joins,
            "formation_reference_audit": reference_audit(tables),
            "spectroscopic_dissociation": spectroscopic_dissociation(),
            "dissociation_anchors": {"interpretation": "Balanced 0 K atomization from source thermal anchors, not extrapolated polynomial values or a new measurement. Compare to experiments only after checking De/D0 and atomic asymptotes.",
                                     "JANAF_D0_J_mol": janaf_d0, "JANAF_D0_eV": janaf_d0/ELECTRONVOLT_J_MOL,
                                     "NASA_header_D0_J_mol": nasa_d0, "NASA_header_D0_eV": nasa_d0/ELECTRONVOLT_J_MOL,
                                     "Barklem_Collet_selected_D0_eV": major["barklem_collet"]["AlH"]["dissociation"]["dissociation_eV"],
                                     "Barklem_Collet_quoted_uncertainty_eV": major["barklem_collet"]["AlH"]["dissociation"]["uncertainty_eV"]},
            "fastchem_fit": {"source_label": fit_label, "coefficients": coefficients, "diagnostic_range_K": [1500, 4000],
                             "max_abs_JANAF_residual_ln_K": max(abs(row["fit_minus_JANAF_ln_K"]) for row in fit_checks),
                             "samples": fit_checks},
            "partition_comparison": partition_comparison(tables, inputs, headers, major["nasa9_gas_constant_J_mol_K"]),
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
        selected = [Path(__file__), ROOT / "mceg/reviews/c2h_source_review.py", ROOT / "mceg/reviews/gas_source_comparison.py",
                    FASTCHEM, MAJOR, BASELINE / "inputs.json", BASELINE / "comparison.json",
                    DATA / "sources.json", DATA / "literature.json", *archived_sources()]
        receipt = {"schema_version": "1", "study": "alh-source-review-20261102", "execution_kind": "fresh_offline_execution",
                   "executed_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                   "python": platform.python_version(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "implementation_identity": "Selected file hashes include uncommitted files; HEAD alone does not identify this execution.",
                   "selected_input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in selected},
                   "output_sha256": {str(args.output.resolve().relative_to(ROOT)): file_hash(args.output)},
                   "scope": "AlH energy and thermal-source audit, not a model adoption or equilibrium execution", "result": "passed"}
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
