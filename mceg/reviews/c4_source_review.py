"""Replay the C4 source decomposition and existing grid footprint offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, load_inputs, nasa_g


DATA = ROOT / "tests/data/validation/c4_source_review/20261103"
ATOMIC_TABLE = ROOT / "tests/data/validation/c3_source_review/20261102/C-003.txt"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"


def footprint() -> dict:
    """Read the archived 75-case grid; do not solve a substituted equilibrium."""
    with np.load(GRID) as grid:
        molecule = grid["species"].tolist().index("C4")
        element = grid["elements"].tolist().index("C")
        fractions = (grid["formula_matrix"][element, molecule]
                     * np.exp(grid["ln_n"][:, molecule]) / grid["budget"][:, element])
        index = int(np.argmax(fractions))
        return dict(source=str(GRID.relative_to(ROOT)), cases=len(fractions),
                    species_count=len(grid["species"]), case_index=index,
                    temperature_K=float(grid["temperature_K"][index]),
                    pressure_bar=float(grid["pressure_bar"][index]),
                    C_over_O=float(grid["C_over_O"][index]),
                    carbon_fraction=float(fractions[index]),
                    mole_fraction=float(np.exp(grid["ln_x"][index, molecule])),
                    all_cases_converged=bool(np.all(grid["converged"])),
                    temperature_bounds_K=[float(grid["temperature_K"].min()),
                                          float(grid["temperature_K"].max())],
                    new_equilibrium_substitution="not_run")


def energy_zero_K(tables: dict) -> dict:
    """Balance explicit H(0) anchors; never extrapolate NASA polynomials."""
    # NASA header H298-H0 is a thermal increment, not Hf(0).
    atom = (ROOT / "tests/data/validation/c3_source_review/20261102/nasa_selected_records.txt").read_text().split("\n\n")[0].splitlines()
    molecule = (DATA / "nasa-c4.txt").read_text().splitlines()
    nasa_h0 = {name: float(lines[1].split()[-1])-float(lines[2].split()[-1])
               for name, lines in (("C", atom), ("C4", molecule))}
    janaf_h0 = {name: 1000*(table[298.15][4]+table[0.0][3])
                for name, table in tables.items()}
    native = {"janaf_D0_kJ_mol": (4*janaf_h0["C"]-janaf_h0["C4"])/1000,
              "nasa_D0_kJ_mol": (4*nasa_h0["C"]-nasa_h0["C4"])/1000}
    facts = json.loads((DATA / "sources.json").read_text())["w4_facts"]
    return dict(**native, w4_linear_D0_kJ_mol=facts["linear_TAE0_kcal_mol"]*4.184,
                w4_cyclic_D0_kJ_mol=facts["cyclic_TAE0_kcal_mol"]*4.184,
                w4_cyclic_minus_linear_energy_kJ_mol=(facts["linear_TAE0_kcal_mol"]
                    -facts["cyclic_TAE0_kcal_mol"])*4.184,
                interpretation="Native legacy anchors and published W4 totals; not a harmonized adopted candidate or a new quantum calculation.")


def lowest_excited_state() -> list:
    """Show one measured state's electronic weight, excluding all other modes."""
    facts = json.loads((DATA / "sources.json").read_text())["spectroscopy_facts"]
    kb_eV_K = 1.380649e-23/1.602176634e-19
    return [dict(temperature_K=t, added_electronic_weight_relative_to_triplet=
                 (2/3)*math.exp(-facts["lowest_linear_singlet_T0_eV"]/(kb_eV_K*t)))
            for t in (1500, 3000, 4000)]


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
    tables = {"C4": {float(t): row for t, row in json.loads(
        (DATA / "janaf_c4_nodes.json").read_text())["rows"].items()},
              "C": read_table(ATOMIC_TABLE)}
    printed = [float(value.replace("D", "E")) for value in re.findall(
        r"[-+]?\d+\.\d+D[-+]\d+", (DATA / "nasa-c4.txt").read_text())]
    if printed != [c for row in nasa["C4"]["intervals"] for c in row["coefficients"]]:
        raise ValueError("Printed NASA C4 coefficient mismatch")
    old_table = inputs["janaf_records"]["C4"]
    if tables["C4"][298.15][4] != old_table["hf298_kJ_mol"]:
        raise ValueError("JANAF C4 anchor mismatch")
    for old in old_table["samples"]:
        node = tables["C4"][old["temperature_K"]]
        if [node[3], node[1], node[5]] != [old[k] for k in (
                "enthalpy_increment_kJ_mol", "entropy_J_mol_K", "formation_gibbs_kJ_mol")]:
            raise ValueError("JANAF C4 frozen node mismatch")

    rows = []
    for t in sources["temperatures_K"]:
        properties = {}
        components = dict.fromkeys(("H298_anchor", "H_increment", "minus_T_S298",
                                    "minus_T_S_increment"), 0.0)
        raw_ln_k = 0.0
        for name, count in (("C4", 1), ("C", -4)):
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
        nasa_ln_k = 4*nasa_g(nasa["C"], t)-nasa_g(nasa["C4"], t)
        janaf_ln_k = nasa_ln_k-delta_g/(r*t)
        old = next((s for s in baseline["samples"]
                    if s["species_id"] == "C4" and s["temperature_K"] == t), None)
        curated = None if old is None else old["ln_Kp"]["curated"]
        rows.append(dict(temperature_K=t, properties=properties,
                         reaction_delta_G_components_J_mol=components,
                         reaction_delta_G_J_mol=delta_g,
                         molecular_delta_G_J_mol=properties["C4"]["delta_G_J_mol"],
                         atomic_delta_G_J_mol=delta_g-properties["C4"]["delta_G_J_mol"],
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
                      for t, node in tables["C4"].items() if t > 0]
    h298, s298, cp298 = nasa_properties(nasa["C4"], 298.15, 8.31451)
    return dict(reaction="4 C(g) -> C4(g)", standard_pressure_Pa=100000,
                delta_convention="JANAF minus NASA; molecule minus four neutral atoms",
                gas_constant_J_mol_K=r, official_nasa_matching_coefficients=len(printed),
                fresh_janaf_frozen_matching_nodes=len(old_table["samples"]),
                nasa_header_check=dict(gas_constant_J_mol_K=8.31451,
                                       H298_J_mol=h298, S298_J_mol_K=s298,
                                       Cp298_J_mol_K=cp298),
                source_samples=rows, janaf_free_energy_function_audit=function_audit,
                energy_zero_K=energy_zero_K(tables),
                lowest_excited_state=lowest_excited_state(), footprint=footprint(),
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
