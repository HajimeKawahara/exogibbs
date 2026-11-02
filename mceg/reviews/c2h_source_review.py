"""Decompose the C2H source difference using pinned, offline source tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs


DATA = ROOT / "tests/data/validation/c2h_source_review/20261102"
TABLES = {"C2H": "C-124.txt", "C": "C-003.txt", "H": "H-001.txt"}
STOICHIOMETRY = {"C2H": 1, "C": -2, "H": -1}


def read_table(path: Path) -> dict:
    """Read exact JANAF nodes, retaining the source's stated units."""
    return {float(row[0]): [float(value.replace("INFINITE", "inf")) for value in row[1:]]
            for line in path.read_text().splitlines()[2:]
            if len(row := line.split()) == 8}


def nasa_properties(record: dict, temperature: float, gas_constant: float) -> tuple:
    """Return H (J/mol), S and Cp (J/mol/K); use the baseline interval rule."""
    interval = next((row for row in record["intervals"]
                     if row["bounds_K"][0] <= temperature <= row["bounds_K"][1]), None)
    if interval is None:
        raise ValueError("Temperature outside NASA intervals")
    a, b, c, d, e, f, g, h, s = interval["coefficients"]
    t = temperature
    entropy = gas_constant * (-a / (2*t*t) - b/t + c*math.log(t) + d*t
                              + e*t*t/2 + f*t**3/3 + g*t**4/4 + s)
    enthalpy = gas_constant * (-a/t + b*math.log(t) + c*t + d*t*t/2
                               + e*t**3/3 + f*t**4/4 + g*t**5/5 + h)
    cp = gas_constant * (a/t**2 + b/t + c + d*t + e*t*t + f*t**3 + g*t**4)
    return enthalpy, entropy, cp


def review() -> dict:
    """Separate reaction energy, thermal increments, entropy and atomic effects."""
    inputs = load_inputs()
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"] and file_hash(DATA / source["file"]) != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    r = inputs["gas_constant_J_mol_K"]
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    rows = []
    for t in inputs["temperatures_K"]:
        properties = {}
        components = dict.fromkeys(("anchor", "enthalpy_increment", "entropy_at_298",
                                    "entropy_increment"), 0.0)
        for name, coefficient in STOICHIOMETRY.items():
            node, reference = tables[name][t], tables[name][298.15]
            nh, ns, ncp = nasa_properties(inputs["nasa9_records"][name], t, r)
            nh0, ns0, _ = nasa_properties(inputs["nasa9_records"][name], 298.15, r)
            jh0, js0 = reference[4]*1000, reference[1]
            jh, js = jh0 + node[3]*1000, node[1]
            delta = {"anchor": jh0-nh0,
                     "enthalpy_increment": (jh-jh0)-(nh-nh0),
                     "entropy_at_298": -t*(js0-ns0),
                     "entropy_increment": -t*((js-js0)-(ns-ns0))}
            for key, value in delta.items():
                components[key] += coefficient*value
            properties[name] = {"janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": js, "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": node[0], "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh-nh-t*(js-ns)}
        delta_g = sum(components.values())
        ln_ratio = -delta_g/(r*t)
        old = next(row for row in baseline["samples"]
                   if row["species_id"] == "C2H1" and row["temperature_K"] == t)
        expected = old["ln_Kp"]["janaf"]-old["ln_Kp"]["nasa"]
        if not math.isclose(ln_ratio, expected, abs_tol=2e-11, rel_tol=0):
            raise ValueError("Fresh official tables do not reproduce the baseline")
        curated = old["ln_Kp"]["curated"]
        rows.append({"temperature_K": t, "properties": properties,
                     "reaction_delta_G_components_J_mol": components,
                     "reaction_delta_G_J_mol": delta_g,
                     "molecular_delta_G_J_mol": properties["C2H"]["delta_G_J_mol"],
                     "atomic_delta_G_J_mol": delta_g-properties["C2H"]["delta_G_J_mol"],
                     "janaf_over_nasa": math.exp(ln_ratio),
                     "energy_aligned_janaf_over_nasa": math.exp(-(delta_g-components["anchor"])/(r*t)),
                     "curated_over_nasa": None if curated is None else math.exp(curated-old["ln_Kp"]["nasa"]),
                     "baseline_closure_ln_K": ln_ratio-expected})
    return {"reaction": "2 C(g) + H(g) -> CCH(g)", "standard_pressure_Pa": 100000,
            "delta_convention": "JANAF minus NASA, molecule minus atoms",
            "gas_constant_J_mol_K": r, "reference_temperature_K": 298.15,
            "energy_alignment": "Diagnostic reaction H298 shift only; not an adopted model",
            "equilibrium_impact": "not_run", "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
