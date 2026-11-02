"""Audit methane thermal sources and partition sums using frozen offline inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs, nasa_g


DATA = ROOT / "tests/data/validation/ch4_source_review/20261102"
MAJOR = ROOT / "examples/comparisons/data/major_molecule_reference.json"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
STOICHIOMETRY = {"CH4": 1, "C": -1, "H": -4}
R = 8.31446261815324
NASA_R = 8.31451


def internal_partition(temperature: float, thermal_g_rt: float, mass_u: float) -> float:
    """Remove ideal-gas translation at 1 bar from (G-H0)/(RT)."""
    kt = 1.380649e-23 * temperature
    mass = mass_u * 1.66053906892e-27
    translation = (2 * math.pi * mass * kt / 6.62607015e-34**2)**1.5 * kt / 1e5
    return math.exp(-thermal_g_rt) / translation


def curated_ln_k(temperature: float, record: dict) -> float:
    """Evaluate the saved methane Hermite free energy and NASA atomic background."""
    knots = record["knots_K"]
    if not knots[0] <= temperature <= knots[-1]:
        raise ValueError("Temperature outside the curated range")
    index = min(np.searchsorted(knots[1:], temperature, side="right"), len(knots)-2)
    u = (temperature-knots[index])/(knots[index+1]-knots[index])
    free_energy = sum(c * u**k for k, c in enumerate(record["coefficients"][index]))
    background = {"intervals": [{"bounds_K": [knots[0], knots[-1]],
                                 "coefficients": record["background_nasa9"]}]}
    return -free_energy/temperature - nasa_g(background, temperature)


def footprint() -> dict:
    """Re-evaluate the carbon share in an existing grid, without new equilibrium solves."""
    with np.load(GRID) as grid:
        molecule = grid["species"].tolist().index("C1H4")
        element = grid["elements"].tolist().index("C")
        shares = (np.exp(grid["ln_n"][:, molecule])
                  * grid["formula_matrix"][element, molecule] / grid["budget"][:, element])
        index = int(np.argmax(shares))
        return {"source": str(GRID.relative_to(ROOT)), "cases": len(shares),
                "case_index": index, "temperature_K": float(grid["temperature_K"][index]),
                "pressure_bar": float(grid["pressure_bar"][index]),
                "C_over_O": float(grid["C_over_O"][index]),
                "mole_fraction": float(np.exp(grid["ln_x"][index, molecule])),
                "carbon_fraction": float(shares[index]),
                "all_cases_converged": bool(np.all(grid["converged"])),
                "new_equilibrium_substitution": "not_run"}


def review() -> dict:
    """Separate source energy, thermal shape, spin conventions and file versions."""
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if file_hash(DATA / source["file"]) != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    for filename, expected in sources["existing_inputs"].items():
        if file_hash(ROOT / filename) != expected:
            raise ValueError("Input hash mismatch: " + filename)
    major = json.loads(MAJOR.read_text())
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    nasa = load_inputs()["nasa9_records"]
    production = json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    tables = {name: major["janaf"][name]["rows"] for name in STOICHIOMETRY}
    energy = {}
    # H0 is assigned in each source's 298 K elemental-reference gauge.
    # Only balanced reactions can be compared to ATcT formation values at 0 K.
    for family in ("janaf", "nasa", "atct"):
        values = {}
        for name in STOICHIOMETRY:
            if family == "janaf":
                values[name] = 1000*(tables[name]["298.15"][4]+tables[name]["0.0"][3])
            elif family == "nasa":
                record = major["nasa9_records"][name]
                values[name] = record["hf298_J_mol"]-record["h298_minus_h0_J_mol"]
            else:
                values[name] = 1000*major["atct"]["records"][name]["hf0_kJ_mol"]
        energy[family] = sum(STOICHIOMETRY[name]*value for name, value in values.items())
    source_rows = []
    for t in (300, 500, 1000, 1500, 2000, 3000, 4000, 6000):
        properties, components = {}, dict.fromkeys(("H298_anchor", "H_increment",
                                                    "minus_T_S298", "minus_T_S_increment"), 0.)
        for name, count in STOICHIOMETRY.items():
            j, ref = tables[name][str(float(t))], tables[name]["298.15"]
            nh, ns, ncp = nasa_properties(nasa[name], t, R)
            nh_ref, ns_ref, _ = nasa_properties(nasa[name], 298.15, R)
            jh = (ref[4]+j[3])*1000
            terms = {"H298_anchor": ref[4]*1000-nh_ref,
                     "H_increment": j[3]*1000-(nh-nh_ref),
                     "minus_T_S298": -t*(ref[1]-ns_ref),
                     "minus_T_S_increment": -t*((j[1]-ref[1])-(ns-ns_ref))}
            for key, value in terms.items():
                components[key] += count*value
            properties[name] = {"janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                                "janaf_S_J_mol_K": j[1], "nasa_S_J_mol_K": ns,
                                "janaf_Cp_J_mol_K": j[0], "nasa_Cp_J_mol_K": ncp,
                                "delta_G_J_mol": jh-nh-t*(j[1]-ns)}
        delta_g = sum(components.values())
        nasa_ln_k = -sum(count*nasa_g(nasa[name], t) for name, count in STOICHIOMETRY.items())
        raw_ratio = math.exp(-delta_g/(R*t))
        reference = next((row for row in baseline["samples"]
                          if row["species_id"] == "C1H4" and row["temperature_K"] == t), None)
        closure = None if reference is None else (
            math.log(raw_ratio)-(reference["ln_Kp"]["janaf"]-reference["ln_Kp"]["nasa"]))
        current = curated_ln_k(t, production["janaf_reactions"]["C1H4"]) if 1500 <= t <= 4000 else None
        source_rows.append({"temperature_K": t, "properties": properties,
                            "reaction_delta_G_components_J_mol": components,
                            "janaf_over_nasa": raw_ratio,
                            "H298_aligned_janaf_over_nasa": math.exp(-(delta_g-components["H298_anchor"])/(R*t)),
                            "atomic_delta_G_J_mol": delta_g-properties["CH4"]["delta_G_J_mol"],
                            "curated_over_raw_nasa": None if current is None else math.exp(current-nasa_ln_k),
                            "baseline_closure_ln_K": closure})
    partition_tables = {name: dict(np.loadtxt(DATA / filename)) for name, filename in
                        (("MM", "exomol_mm.pf"), ("TIPS2021", "hapi_tips2021_ch4.txt"),
                         ("TIPS2025", "hitran_q32_20261102.txt"))}
    wenger = {node["T_K"]: node for node in literature["wenger2008"]["nodes"]}
    partition_rows = []
    for t in (300, 500, 1000, 1500, 2000, 2500, 3000, 4000):
        if t == 2500:
            continuous = json.loads((ROOT / "examples/comparisons/data/curated_continuous_reference.json").read_text())
            cp, entropy, sensible = continuous["janaf"]["CH4"]["rows"][str(float(t))]
        else:
            node = tables["CH4"][str(float(t))]
            cp, entropy, sensible = node[0], node[1], node[3]
        jg = ((sensible-tables["CH4"]["0.0"][3])*1000/t-entropy)/R
        header = major["nasa9_records"]["CH4"]
        ng = nasa_g(nasa["CH4"], t) - (header["hf298_J_mol"]-header["h298_minus_h0_J_mol"])/(NASA_R*t)
        jq = internal_partition(t, jg, literature["identity"]["janaf_molar_mass_g_mol"])
        nq = internal_partition(t, ng, literature["identity"]["nasa_molar_mass_g_mol"])
        raw = {name: None if t not in values else float(values[t])
               for name, values in partition_tables.items()}
        normalized = {name: None if value is None else value/16 for name, value in raw.items()}
        partition_rows.append({"temperature_K": t, "janaf_effective_Q": jq, "nasa_effective_Q": nq,
                               "spectroscopic_Q_including_nuclear_spin": raw,
                               "Q_over_16_over_janaf": {k: None if q is None else q/jq for k, q in normalized.items()},
                               "Q_over_16_over_nasa": {k: None if q is None else q/nq for k, q in normalized.items()},
                               "TIPS2025_over_2021": None if raw["TIPS2021"] is None else raw["TIPS2025"]/raw["TIPS2021"],
                               "MM_within_published_completeness_estimate": t <= 2000,
                               "wenger2008_Q_including_nuclear_spin": wenger.get(t, {}).get("Q")})
    return {"reaction": "C(g) + 4 H(g) -> CH4(g)", "standard_pressure_Pa": 100000,
            "gas_constant_J_mol_K": R, "NASA_header_gas_constant_J_mol_K": NASA_R,
            "delta_convention": "JANAF minus NASA; molecule minus atoms",
            "balanced_H0_reaction_J_mol": energy,
            "source_samples": source_rows, "partition_samples": partition_rows,
            "partition_caveat": "Natural-mass thermochemical effective Q versus 12CH4; not an isotope-resolved equality or accuracy test.",
            "footprint": footprint(), "adoption": "none; retain current implementation provisionally"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
