"""Replay PO2 source differences and an existing phosphorus-budget footprint.

All calculations use frozen inputs. Energy alignment is a diagnostic only;
this script does not provide new production thermochemistry or solve equilibrium.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs


DATA = ROOT / "tests/data/validation/po2_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
TABLES = {"PO2": "O-032.txt", "P": "P-008.txt", "O": "O-001.txt"}
STOICHIOMETRY = {"PO2": 1, "P": -1, "O": -2}


def curated_ln_k(temperature: float) -> float:
    """Evaluate the inherited PO2 atom-formation fit within its declared domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated review range")
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("O2P1 "))
    a, b, c, d, e = map(float, lines[index + 1].split())
    t = temperature
    return a / t + b * math.log(t) + c + d * t + e * t*t


def footprint() -> dict:
    """Read archived solved endpoints; fractions use the conserved P budget."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("O2P1")
        phosphorus = grid["elements"].tolist().index("P")
        shares = (grid["formula_matrix"][phosphorus, column]
                  * np.exp(grid["ln_n"][:, column]) / grid["budget"][:, phosphorus])
        case = int(np.argmax(shares))
        return {
            "source": str(GRID.relative_to(ROOT)), "source_sha256": file_hash(GRID),
            "model": "curated-c2h-v1", "cases": len(shares),
            "species_count": len(grid["species"]),
            "all_cases_converged": bool(np.all(grid["converged"])),
            "maximum_phosphorus_share": {
                "case_index": case, "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]),
                "mole_fraction": float(np.exp(grid["ln_x"][case, column])),
                "phosphorus_fraction": float(shares[case]),
            },
            "interpretation": "Existing finite ideal-gas footprint, not a substitution response or an observation.",
        }


def review() -> dict:
    """Balance source-specific atoms before separating energy and thermal terms."""
    inputs = load_inputs()
    literature = json.loads((DATA / "literature.json").read_text())
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    r = inputs["gas_constant_J_mol_K"]
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    rows = []
    for t in (500, 1000, 1500, 2000, 3000, 4000):
        properties = {}
        components = dict.fromkeys(("anchor", "enthalpy_increment", "entropy_at_298",
                                    "entropy_increment"), 0.0)
        janaf_reaction_g = nasa_reaction_g = 0.0
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
            janaf_reaction_g += coefficient*(jh-t*js)
            nasa_reaction_g += coefficient*(nh-t*ns)
            properties[name] = {"janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": js, "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": node[0], "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh-nh-t*(js-ns)}
        delta_g = sum(components.values())
        ln_ratio = -delta_g/(r*t)
        old = next((row for row in baseline["samples"]
                    if row["species_id"] == "O2P1" and row["temperature_K"] == t), None)
        closure = None
        if old is not None:
            closure = ln_ratio-(old["ln_Kp"]["janaf"]-old["ln_Kp"]["nasa"])
            if abs(closure) > 2e-11:
                raise ValueError("Fresh official tables do not reproduce the baseline")
        current = curated_ln_k(t) if 1500 <= t <= 4000 else None
        rows.append({
            "temperature_K": t, "properties": properties,
            "reaction_delta_G_components_J_mol": components,
            "reaction_delta_G_J_mol": delta_g,
            "molecular_delta_G_J_mol": properties["PO2"]["delta_G_J_mol"],
            "atomic_delta_G_J_mol": delta_g-properties["PO2"]["delta_G_J_mol"],
            "janaf_ln_K": -janaf_reaction_g/(r*t), "nasa_ln_K": -nasa_reaction_g/(r*t),
            "curated_ln_K": current, "janaf_over_nasa": math.exp(ln_ratio),
            "H298_aligned_janaf_over_nasa": math.exp(-(delta_g-components["anchor"])/(r*t)),
            "curated_over_nasa": None if current is None else math.exp(current+nasa_reaction_g/(r*t)),
            "baseline_closure_ln_K": closure,
        })
    fit_rows = []
    for t in range(1500, 4001, 100):
        jg = sum(coefficient*(tables[name][298.15][4]*1000
                             + tables[name][t][3]*1000-t*tables[name][t][1])
                 for name, coefficient in STOICHIOMETRY.items())
        fit_rows.append({"temperature_K": t, "curated_minus_janaf_ln_K": curated_ln_k(t)+jg/(r*t)})
    # Hf(0) uses the elemental zero at 0 K; H(0) below uses the 298 K gauge.
    # Subtract atoms within each source before comparing total atomization energies.
    h0 = {"janaf": sum(c*(tables[n][298.15][4]+tables[n][0][3])*1000
                       for n, c in STOICHIOMETRY.items()),
          "nasa_headers": sum(c*(literature["nasa_headers"][n]["hf298_J_mol"]
                                  - literature["nasa_headers"][n]["H298_minus_H0_J_mol"])
                              for n, c in STOICHIOMETRY.items())}
    return {
        "reaction": "P(g) + 2 O(g) -> PO2(g)", "standard_pressure_Pa": 100000,
        "delta_convention": "JANAF minus NASA; molecule minus source-specific atoms",
        "gas_constant_J_mol_K": r, "reference_temperature_K": 298.15,
        "source_samples": rows, "fit_samples": fit_rows,
        "max_abs_curated_minus_janaf_ln_K": max(abs(x["curated_minus_janaf_ln_K"]) for x in fit_rows),
        "balanced_H0_reaction_J_mol": h0, "footprint": footprint(),
        "energy_alignment": "Diagnostic reaction H298 shift only; not an adopted model",
        "equilibrium_substitution": "not_run", "production_change": "none",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text, end="")
