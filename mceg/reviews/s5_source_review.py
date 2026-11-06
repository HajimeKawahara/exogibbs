"""Replay the S5 source decomposition and existing grid footprint offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs, nasa_g


DATA = ROOT / "tests/data/validation/s5_source_review/20261103"
ATOMIC_TABLE = ROOT / "tests/data/validation/s2o_source_investigation/20261102/S-006.txt"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"


def footprint() -> dict:
    """Read the archived 75-case grid; do not solve a substituted equilibrium."""
    with np.load(GRID) as grid:
        molecule = grid["species"].tolist().index("S5")
        element = grid["elements"].tolist().index("S")
        fractions = (grid["formula_matrix"][element, molecule]
                     * np.exp(grid["ln_n"][:, molecule]) / grid["budget"][:, element])
        index = int(np.argmax(fractions))
        return dict(source=str(GRID.relative_to(ROOT)), cases=len(fractions),
                    species_count=len(grid["species"]), case_index=index,
                    temperature_K=float(grid["temperature_K"][index]),
                    pressure_bar=float(grid["pressure_bar"][index]),
                    C_over_O=float(grid["C_over_O"][index]),
                    sulfur_fraction=float(fractions[index]),
                    mole_fraction=float(np.exp(grid["ln_x"][index, molecule])),
                    all_cases_converged=bool(np.all(grid["converged"])),
                    temperature_bounds_K=[float(grid["temperature_K"].min()),
                                          float(grid["temperature_K"].max())],
                    new_equilibrium_substitution="not_run")


def review() -> dict:
    """Separate native molecular and atomic energy/thermal differences."""
    sources = json.loads((DATA / "sources.json").read_text())
    for name, expected in sources["existing_inputs"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Input hash mismatch: " + name)
    for source in sources["archived_inputs"]:
        if file_hash(DATA / source["file"]) != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    inputs = load_inputs()
    nasa, r = inputs["nasa9_records"], inputs["gas_constant_J_mol_K"]
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    tables = {"S5": {float(t): row for t, row in json.loads(
        (DATA / "janaf_s5_nodes.json").read_text())["rows"].items()},
              "S": read_table(ATOMIC_TABLE)}
    printed = [float(value.replace("D", "E")) for value in re.findall(
        r"[-+]?\d+\.\d+D[-+]\d+", (DATA / "nasa-s5.txt").read_text())]
    if printed != [c for row in nasa["S5"]["intervals"] for c in row["coefficients"]]:
        raise ValueError("Printed NASA S5 coefficient mismatch")
    old_table = inputs["janaf_records"]["S5"]
    if tables["S5"][298.15][4] != old_table["hf298_kJ_mol"]:
        raise ValueError("JANAF S5 anchor mismatch")
    for old in old_table["samples"]:
        node = tables["S5"][old["temperature_K"]]
        if [node[3], node[1], node[5]] != [old[k] for k in (
                "enthalpy_increment_kJ_mol", "entropy_J_mol_K", "formation_gibbs_kJ_mol")]:
            raise ValueError("JANAF S5 frozen node mismatch")

    rows = []
    for t in sources["temperatures_K"]:
        properties = {}
        components = dict.fromkeys(("H298_anchor", "H_increment", "minus_T_S298",
                                    "minus_T_S_increment"), 0.0)
        raw_ln_k = 0.0
        for name, count in (("S5", 1), ("S", -5)):
            node, reference = tables[name][t], tables[name][298.15]
            nh, ns, ncp = nasa_properties(nasa[name], t, r)
            nh_ref, ns_ref, _ = nasa_properties(nasa[name], 298.15, r)
            jh = 1000*(reference[4]+node[3])
            terms = {"H298_anchor": 1000*reference[4]-nh_ref,
                     "H_increment": 1000*node[3]-(nh-nh_ref),
                     "minus_T_S298": -t*(reference[1]-ns_ref),
                     "minus_T_S_increment": -t*((node[1]-reference[1])-(ns-ns_ref))}
            for key, value in terms.items():
                components[key] += count*value
            raw_ln_k -= count*1000*node[5]/(r*t)
            properties[name] = dict(janaf_H_J_mol=jh, nasa_H_J_mol=nh,
                                    janaf_S_J_mol_K=node[1], nasa_S_J_mol_K=ns,
                                    janaf_Cp_J_mol_K=node[0], nasa_Cp_J_mol_K=ncp,
                                    delta_G_J_mol=jh-nh-t*(node[1]-ns))
        delta_g = sum(components.values())
        nasa_ln_k = 5*nasa_g(nasa["S"], t)-nasa_g(nasa["S5"], t)
        janaf_ln_k = nasa_ln_k-delta_g/(r*t)
        old = next((s for s in baseline["samples"]
                    if s["species_id"] == "S5" and s["temperature_K"] == t), None)
        curated = None if old is None else old["ln_Kp"]["curated"]
        rows.append(dict(temperature_K=t, properties=properties,
                         reaction_delta_G_components_J_mol=components,
                         reaction_delta_G_J_mol=delta_g,
                         molecular_delta_G_J_mol=properties["S5"]["delta_G_J_mol"],
                         atomic_delta_G_J_mol=delta_g-properties["S5"]["delta_G_J_mol"],
                         ln_Kp_janaf_hs=janaf_ln_k, ln_Kp_nasa=nasa_ln_k,
                         janaf_raw_minus_hs_ln_K=raw_ln_k-janaf_ln_k,
                         janaf_over_nasa=math.exp(-delta_g/(r*t)),
                         H298_aligned_janaf_over_nasa=math.exp(-(delta_g-components["H298_anchor"])/(r*t)),
                         curated_over_nasa=None if curated is None else math.exp(curated-nasa_ln_k),
                         curated_over_janaf=None if curated is None else math.exp(curated-janaf_ln_k),
                         baseline_closure_ln_K=None if old is None else
                         janaf_ln_k-old["ln_Kp"]["janaf"]))
    function_audit = [dict(temperature_K=t, reported_J_mol_K=node[2],
                           from_HS_J_mol_K=node[1]-1000*node[3]/t,
                           difference_J_mol_K=node[2]-(node[1]-1000*node[3]/t))
                      for t, node in tables["S5"].items()]
    h298, s298, cp298 = nasa_properties(nasa["S5"], 298.15, 8.31451)
    return dict(reaction="5 S(g) -> S5(g)", standard_pressure_Pa=100000,
                delta_convention="JANAF minus NASA; molecule minus five neutral atoms",
                gas_constant_J_mol_K=r, official_nasa_matching_coefficients=len(printed),
                fresh_janaf_frozen_matching_nodes=len(old_table["samples"]),
                nasa_header_check=dict(gas_constant_J_mol_K=8.31451,
                                       H298_J_mol=h298, S298_J_mol_K=s298,
                                       Cp298_J_mol_K=cp298),
                source_samples=rows, janaf_free_energy_function_audit=function_audit,
                footprint=footprint(),
                adoption="none; retain current fit provisionally",
                physical_accuracy="unknown")


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
