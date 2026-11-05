"""Replay S4 source thermochemistry and archived abundance allocation offline."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, nasa_g

DATA = ROOT / "tests/data/validation/s4_source_review/20261105"
ATOMIC_TABLE = ROOT / "tests/data/validation/s2o_source_investigation/20261102/S-006.txt"
GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105/ions.npz"
NAME, COUNT = "S4", 4
R = 8.31446261815324
NASA_R = 8.31451


@lru_cache(maxsize=1)
def setup():
    """Use the named production version, independent of future default changes."""
    import jax
    from exogibbs.presets import curated_cos
    jax.config.update("jax_enable_x64", True)
    return curated_cos.chemsetup(ions=True)


def curated_ln_k(temperature: float) -> float:
    """Return ln K for formation from neutral atoms in the curated domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated range")
    model = setup()
    return -float(model.hvector_func(float(temperature))[model.species.index(NAME)])


def footprint() -> dict:
    """Read 75 archived endpoints; this is not a replacement calculation."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index(NAME)
        element = grid["elements"].tolist().index("S")
        shares = COUNT * np.exp(grid["ln_n"][:, column]) / grid["budget"][:, element]
        i = int(np.argmax(shares))
        return {"model": "curated-cos-v1", "cases": len(shares),
                "species_count": len(grid["species"]), "case_index": i,
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "sulfur_fraction": float(shares[i]),
                "mole_fraction": float(np.exp(grid["ln_x"][i, column])),
                "all_cases_converged": bool(np.all(grid["converged"])),
                "maximum_abs_target_h_error": max(abs(float(h)+curated_ln_k(t))
                    for t, h in zip(grid["temperature_K"], grid["h"][:, column])),
                "maximum_sulfur_conservation_residual": float(np.max(abs(
                    np.exp(grid["ln_n"]) @ grid["formula_matrix"][element]
                    / grid["budget"][:, element] - 1)))}


def review() -> dict:
    """Separate balanced energy, thermal and source-table consistency terms."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    baseline = json.loads((BASELINE / "inputs.json").read_text())
    nasa = baseline["nasa9_records"]
    tables = {NAME: {float(t): row for t, row in inputs["janaf_nodes"].items()},
              "S": read_table(ATOMIC_TABLE)}
    printed = [float(x.replace("D", "E")) for x in re.findall(
        r"[-+]?\d+\.\d+D[-+]\d+", (DATA / "nasa-record.txt").read_text())]
    if printed != [c for row in nasa[NAME]["intervals"] for c in row["coefficients"]]:
        raise ValueError("NASA printed coefficients differ from frozen record")
    old = baseline["janaf_records"][NAME]
    assert old["hf298_kJ_mol"] == tables[NAME][298.15][4]
    for sample in old["samples"]:
        node = tables[NAME][sample["temperature_K"]]
        assert [node[3], node[1], node[5]] == [sample[k] for k in (
            "enthalpy_increment_kJ_mol", "entropy_J_mol_K", "formation_gibbs_kJ_mol")]
    rows = []
    for t in inputs["temperatures_K"]:
        components = dict.fromkeys(("H298_anchor", "H_increment", "minus_T_S298",
                                    "minus_T_S_increment"), 0.0)
        properties, raw_ln_k = {}, 0.0
        for name, count in ((NAME, 1), ("S", -COUNT)):
            node, ref = tables[name][t], tables[name][298.15]
            nh, ns, ncp = nasa_properties(nasa[name], t, R)
            nh_ref, ns_ref, _ = nasa_properties(nasa[name], 298.15, R)
            jh = 1000 * (ref[4] + node[3])
            terms = {"H298_anchor": 1000*ref[4]-nh_ref,
                     "H_increment": 1000*node[3]-(nh-nh_ref),
                     "minus_T_S298": -t*(ref[1]-ns_ref),
                     "minus_T_S_increment": -t*((node[1]-ref[1])-(ns-ns_ref))}
            for key, value in terms.items():
                components[key] += count*value
            raw_ln_k -= count*1000*node[5]/(R*t)
            properties[name] = {"janaf_H_J_mol": jh, "nasa_H_J_mol": nh,
                "janaf_S_J_mol_K": node[1], "nasa_S_J_mol_K": ns,
                "janaf_Cp_J_mol_K": node[0], "nasa_Cp_J_mol_K": ncp,
                "delta_G_J_mol": jh-nh-t*(node[1]-ns)}
        delta_g = sum(components.values())
        nk = COUNT*nasa_g(nasa["S"], t)-nasa_g(nasa[NAME], t)
        jk = nk-delta_g/(R*t)
        ck = curated_ln_k(t) if t >= 1500 else None
        row = {"temperature_K": t, "properties": properties,
            "reaction_delta_G_components_J_mol": components,
            "reaction_delta_G_J_mol": delta_g,
            "atomic_delta_G_J_mol": delta_g-properties[NAME]["delta_G_J_mol"],
            "ln_Kp_janaf_hs": jk, "ln_Kp_nasa": nk, "ln_Kp_curated": ck,
            "janaf_raw_minus_hs_ln_K": raw_ln_k-jk,
            "janaf_over_nasa": math.exp(-delta_g/(R*t)),
            "H298_aligned_janaf_over_nasa": math.exp(-(delta_g-components["H298_anchor"])/(R*t)),
            "curated_over_nasa": None if ck is None else math.exp(ck-nk),
            "curated_over_janaf": None if ck is None else math.exp(ck-jk)}
        rows.append(row)
    audit = [{"temperature_K": t, "reported_J_mol_K": node[2],
              "from_HS_J_mol_K": node[1]-1000*node[3]/t,
              "difference_J_mol_K": node[2]-(node[1]-1000*node[3]/t)}
             for t, node in tables[NAME].items() if t > 0]
    h, s, cp = nasa_properties(nasa[NAME], 298.15, NASA_R)
    return {"molecule": NAME, "reaction": f"{COUNT} S(g) -> {NAME}(g)",
        "standard_pressure_Pa": 100000, "comparison_R_J_mol_K": R,
        "delta_convention": "JANAF minus NASA, molecule minus neutral atoms",
        "official_nasa_matching_coefficients": len(printed),
        "official_janaf_frozen_matching_nodes": len(old["samples"]),
        "nasa_header_check": {"R_J_mol_K": NASA_R, "H298_J_mol": h,
                              "S298_J_mol_K": s, "Cp298_J_mol_K": cp},
        "source_samples": rows, "janaf_free_energy_function_audit": audit,
        "modern_diagnostic": modern_diagnostic(inputs, tables),
        "footprint": footprint(), "curated_change": "none",
        "new_equilibrium_substitution": "not_run", "physical_accuracy": "unknown"}


def modern_diagnostic(inputs: dict, tables: dict) -> dict:
    """Illustrate an electronic Boltzmann weight, not a complete partition sum."""
    gap = inputs["matus2007"]["triplet_gap_kcal_mol"]*4184
    return {"triplet_gap_J_mol": gap,
            "approximation": "Equal rovibrational functions; electronic factor only, not adopted",
            "triplet_over_singlet_weights": {str(t): 3*math.exp(-gap/(R*t))
                for t in (1500, 3000, 4000)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
