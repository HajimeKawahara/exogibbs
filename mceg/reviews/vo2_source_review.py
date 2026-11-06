"""Audit gaseous VO2 sources and replay its archived vanadium footprint offline."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs

DATA = ROOT / "tests/data/validation/vo2_source_review/20261102"
TABLES = {"VO2": "O-076.txt", "V": "V-005.txt", "O": "O-001.txt"}
STOICHIOMETRY = {"VO2": 1, "V": -1, "O": -2}
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
FIT = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"


def electronic_partition(levels_K: list, weights: list, temperature: float,
                         gas_constant: float) -> dict:
    """Return electronic-only Q and caloric moments, without rovibrational terms."""
    terms = [g * math.exp(-energy / temperature)
             for energy, g in zip(levels_K, weights)]
    q = sum(terms)
    mean = sum(w * e for w, e in zip(terms, levels_K)) / q
    variance = sum(w * (e - mean)**2 for w, e in zip(terms, levels_K)) / q
    return {"Q_el": q, "H_el_J_mol": gas_constant * mean,
            "S_el_J_mol_K": gas_constant * (math.log(q) + mean / temperature),
            "Cp_el_J_mol_K": gas_constant * variance / temperature**2}


def historical_energy_audit(data: dict, janaf: dict, euo_shift: float = 0.0) -> dict:
    """Replay the 1983 selected-energy algebra, retaining its shared EuO anchor."""
    h1, h2, h3 = [row["selected_H0_kJ_mol"] for row in data["reactions"]]
    euo = data["EuO_D0_kJ_mol"] + euo_shift
    vo = euo + h1
    thermal = []
    h298 = data["table_IV_VO2"][0][2]
    for t, phi, h0 in data["table_IV_VO2"]:
        thermal.append({"temperature_K": t,
                        "printed_convention_S_J_mol_K": phi + 1000*h0/t,
                        "H_minus_H298_kJ_mol": h0 - h298,
                        "delta_H_increment_vs_janaf_kJ_mol": h0 - h298 - janaf[t][3]})
    return {"VO_D0_kJ_mol": vo,
            "VO2_D0_from_reaction2_kJ_mol": 2*vo + h2,
            "VO2_D0_from_reaction3_kJ_mol": euo + vo + h3,
            "reaction_cycle_residual_kJ_mol": h3 - h1 - h2,
            "EuO_anchor_shift_kJ_mol": euo_shift,
            "shared_reference": "Both VO2 routes contain 2 D0(EuO); no independent-energy averaging or revised uncertainty is justified here.",
            "thermal_nodes": thermal,
            "thermal_scope": "Historical Table IV enthalpy increments versus JANAF; entropy retains the printed pressure convention. No pressure-normalized Q ratio, updated equilibrium reduction or new model."}


def footprint() -> dict:
    """Recompute the maximum V share from the existing 75-case gas-only archive."""
    with np.load(GRID, allow_pickle=False) as grid:
        species, elements = grid["species"].tolist(), grid["elements"].tolist()
        v, vo2 = elements.index("V"), species.index("O2V1")
        n = np.exp(grid["ln_n"])
        if not bool(np.all(grid["converged"])):
            raise ValueError("Archived grid contains unconverged cases")
        fractions = n[:, vo2] * grid["formula_matrix"][v, vo2] / grid["budget"][:, v]
        closure = np.abs(n @ grid["formula_matrix"][v] / grid["budget"][:, v] - 1)
        case = int(np.argmax(fractions))
        allocation = [{"species": species[i],
                       "V_fraction": float(n[case, i] * coefficient / grid["budget"][case, v])}
                      for i, coefficient in enumerate(grid["formula_matrix"][v]) if coefficient > 0]
        return {"source": str(GRID.relative_to(ROOT)), "sha256": file_hash(GRID),
                "dataset": "curated-c2h-v1", "cases": len(n), "species_count": len(species),
                "all_cases_converged": True,
                "max_V_relative_residual": float(np.max(closure)),
                "case_index": case, "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]),
                "VO2_mole_fraction": float(np.exp(grid["ln_x"][case, vo2])),
                "VO2_V_fraction": float(fractions[case]),
                "vanadium_allocation": sorted(allocation, key=lambda row: -row["V_fraction"]),
                "scope": "Archived gas-only importance, not new solves, a source-substitution response or atmospheric validation; condensation omitted."}


def review() -> dict:
    """Compare balanced source properties, current fit and isolated level models."""
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"] and file_hash(DATA / source["file"]) != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    inputs, literature = load_inputs(), json.loads((DATA / "literature_inputs.json").read_text())
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    r = inputs["gas_constant_J_mol_K"]
    official = (DATA / "nasa_selected_records.txt").read_text().splitlines()
    for name in TABLES:
        start = next(i for i, line in enumerate(official) if line[:18].strip() == name)
        for i, interval in enumerate(inputs["nasa9_records"][name]["intervals"]):
            text = " ".join(official[start + 3 + 3*i:start + 5 + 3*i])
            coefficients = [float(value.replace("D", "E")) for value in
                            re.findall(r"[-+]?\d+\.\d+D[-+]\d+", text)]
            if coefficients != interval["coefficients"]:
                raise ValueError("Official NASA coefficients differ: " + name)
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    lines = FIT.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("O2V1 "))
    a, b, c, d, e = map(float, lines[index + 1].split())
    samples, fit_errors = [], []
    for t in [500, 1000, 1500, 3000, 4000]:
        components = dict.fromkeys(("H298_anchor", "enthalpy_increment", "entropy_at_298",
                                    "entropy_increment"), 0.0)
        properties, jg = {}, 0.0
        for name, coefficient in STOICHIOMETRY.items():
            node, reference = tables[name][t], tables[name][298.15]
            nh, ns, ncp = nasa_properties(inputs["nasa9_records"][name], t, r)
            nh0, ns0, _ = nasa_properties(inputs["nasa9_records"][name], 298.15, r)
            jh0, js0 = reference[4] * 1000, reference[1]
            jh, js = jh0 + node[3] * 1000, node[1]
            delta = {"H298_anchor": jh0 - nh0,
                     "enthalpy_increment": node[3] * 1000 - (nh - nh0),
                     "entropy_at_298": -t * (js0 - ns0),
                     "entropy_increment": -t * ((js - js0) - (ns - ns0))}
            for key, value in delta.items():
                components[key] += coefficient * value
            jg += coefficient * (jh - t * js)
            properties[name] = {"janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": js, "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": node[0], "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh - nh - t * (js - ns)}
        delta_g = sum(components.values())
        ln_ratio = -delta_g / (r * t)
        old = next((row for row in baseline["samples"]
                    if row["species_id"] == "O2V1" and row["temperature_K"] == t), None)
        closure = None
        if old:
            closure = ln_ratio - (old["ln_Kp"]["janaf"] - old["ln_Kp"]["nasa"])
            if not math.isclose(closure, 0, abs_tol=2e-11):
                raise ValueError("Official JANAF tables differ from the frozen baseline")
        fit_ln_k = None if t < 1500 else a/t + b*math.log(t) + c + d*t + e*t*t
        samples.append({"temperature_K": t, "properties": properties,
                        "reaction_delta_G_components_J_mol": components,
                        "reaction_delta_G_J_mol": delta_g,
                        "atomic_delta_G_J_mol": delta_g - properties["VO2"]["delta_G_J_mol"],
                        "janaf_over_nasa": math.exp(ln_ratio),
                        "energy_aligned_janaf_over_nasa": math.exp(-(delta_g-components["H298_anchor"])/(r*t)),
                        "curated_over_nasa": None if fit_ln_k is None else math.exp(fit_ln_k+jg/(r*t)+ln_ratio),
                        "baseline_closure_ln_K": closure})
    for t in range(1500, 4001, 100):
        jg = sum(k * (tables[s][298.15][4]*1000 + tables[s][t][3]*1000 - t*tables[s][t][1])
                 for s, k in STOICHIOMETRY.items())
        fit_errors.append(abs(a/t + b*math.log(t) + c + d*t + e*t*t + jg/(r*t)))
    janaf = literature["janaf_evaluation"]
    printed_errors = [abs(tables["VO2"][float(t)][i] - value)
                      for t, values in janaf["printed_nodes"].items()
                      for i, value in zip([0, 1, 3, 4], values)]
    constants = literature["constants"]
    rrho_closure = []
    for t in [298.15, 500, 1000, 1500, 3000, 4000]:
        # Nonlinear ideal-gas translation+rotation contribute 4R; a constant
        # ground electronic weight contributes no heat capacity.
        modes = [v * constants["second_radiation_constant_cm_K"] / t
                 for v in janaf["vibrations_cm-1"]]
        cp = r * (4 + sum(x*x * math.exp(x) / math.expm1(x)**2 for x in modes))
        rrho_closure.append({"temperature_K": t, "rrho_Cp_J_mol_K": cp,
                             "janaf_Cp_J_mol_K": tables["VO2"][t][0],
                             "delta_Cp_J_mol_K": cp - tables["VO2"][t][0]})
    old_levels = [v * constants["second_radiation_constant_cm_K"] for v in janaf["electronic_levels_cm-1"]]
    modern = literature["kim2014"]
    new_levels = [v * constants["electronvolt_J_mol"] / r for v in modern["electronic_levels_eV"]]
    electronic = []
    for t in [1500, 3000, 4000]:
        old = electronic_partition(old_levels, janaf["electronic_weights"], t, r)
        new = electronic_partition(new_levels, modern["electronic_weights"], t, r)
        electronic.append({"temperature_K": t, "janaf_list": old, "kim2014_list": new,
                           "kim_over_janaf_Q_el": new["Q_el"] / old["Q_el"]})
    return {"reaction": "V(g) + 2 O(g) -> VO2(g)", "standard_pressure_Pa": 100000,
            "delta_convention": "JANAF minus NASA, molecule minus atoms",
            "gas_constant_J_mol_K": r, "reference_temperature_K": 298.15,
            "reference_caution": "NASA H298 decomposition uses the frozen local record extended to 200 K; official V and VO2 minima are 300 K. Official header retained separately, not an endorsement of extrapolation.",
            "samples": samples, "janaf_total_atomization_D0_kJ_mol": -sum(k*tables[s][0][4] for s,k in STOICHIOMETRY.items()),
            "printed_node_max_abs_difference": max(printed_errors),
            "current_fit_max_abs_ln_K_error_1500_4000": max(fit_errors),
            "janaf_ground_only_rrho_closure": rrho_closure,
            "balducci1983_selected_energy_audit": historical_energy_audit(literature["balducci1983"], tables["VO2"]),
            "electronic_only_diagnostic": electronic,
            "electronic_scope": "Isolated electronic-level sums only; identical state rovibrational factors assumed; no full partition model, isotope normalization or adoption.",
            "footprint": footprint(), "equilibrium_source_substitution": "not_run",
            "adoption": "none; current coefficients retained operationally pending consistent energy and thermal reassessment"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
