"""Replay the four phosphorus-diatomic source packets without changing curated."""

import argparse
import bz2
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, BASELINE, file_hash
from ph3_source_review import current_ln_k, log_translation, R, NASA_R

GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105/ions.npz"
C2 = 6.62607015e-34 * 29979245800 / 1.380649e-23


def extract_states(raw: Path, molecule: str, output: Path) -> dict:
    """Extract all distributed levels; preserve electronic labels and their zero."""
    with bz2.open(raw, "rt") as stream:
        rows = [row for row in map(str.split, stream) if row]
    label_column = 8 if molecule == "PH" else 9
    labels = sorted({row[label_column] for row in rows})
    states = np.array([[float(row[i]) for i in (1, 2, 3)] +
                       [labels.index(row[label_column])] for row in rows])
    np.savez_compressed(output, states=states)
    return {"source_sha256": file_hash(raw), "columns": ["energy_cm-1", "gtot", "J", "electronic_index"],
            "electronic_labels": labels, "states": len(states),
            "maximum_energy_cm-1": float(states[:, 0].max()),
            "maximum_J": float(states[:, 2].max()),
            "gtot_over_2Jplus1": np.unique(states[:, 1]/(2*states[:, 2]+1)).tolist()}


def moments(states: np.ndarray, temperature: float) -> dict:
    """Evaluate ideal-gas calorics directly from first and second state moments."""
    energy = states[:, 0]*C2
    weight = states[:, 1]*np.exp(-energy/temperature)
    q = float(weight.sum())
    mean = float(weight @ energy/q)
    variance = float(weight @ ((energy-mean)**2)/q)
    return {"Q": q, "mean_energy_K": mean,
            "H_minus_H0_J_mol": R*(2.5*temperature+mean),
            "Cp_J_mol_K": R*(2.5+variance/temperature**2)}


def footprint(species_id: str, count: int) -> dict:
    """Read 75 saved endpoints; target h is checked against the named factory."""
    with np.load(GRID) as grid:
        col = grid["species"].tolist().index(species_id)
        elem = grid["elements"].tolist().index("P")
        shares = count*np.exp(grid["ln_n"][:, col])/grid["budget"][:, elem]
        i = int(np.argmax(shares))
        return {"model": "curated-cos-v1", "cases": len(shares), "species": len(grid["species"]),
                "maximum_phosphorus_fraction": float(shares[i]), "case_index": i,
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]), "C_over_O": float(grid["C_over_O"][i]),
                "mole_fraction": float(np.exp(grid["ln_x"][i, col])),
                "all_converged": bool(np.all(grid["converged"])),
                "maximum_abs_target_h_error": max(abs(float(h)+current_ln_k(t, species_id))
                    for t, h in zip(grid["temperature_K"], grid["h"][:, col]))}


def review(slug: str) -> dict:
    """Keep absolute energy, molecular/atomic thermal terms and accuracy separate."""
    data = ROOT / f"tests/data/validation/{slug}_source_review/20261106"
    inputs = json.loads((data / "inputs.json").read_text())
    baseline = json.loads((BASELINE / "inputs.json").read_text())
    name, composition = inputs["name"], inputs["composition"]
    records = inputs["nasa_records"]
    tables = {s: read_table(ROOT/p) for s, p in inputs["janaf_paths"].items()}
    for s, record in records.items():
        assert [r["coefficients"] for r in record["intervals"]] == [r["coefficients"] for r in baseline["nasa9_records"][s]["intervals"]]
    printed = [float(x.replace("D", "E")) for x in re.findall(r"[-+]?\d+\.\d+D[-+]\d+", (data / "nasa-selected.txt").read_text())]
    assert printed == [c for record in records.values() for row in record["intervals"] for c in row["coefficients"]]
    old = baseline["janaf_records"][inputs["species_id"]]
    for sample in old["samples"]:
        row = tables[name][sample["temperature_K"]]
        assert [row[3], row[1], row[5]] == [sample[k] for k in ("enthalpy_increment_kJ_mol", "entropy_J_mol_K", "formation_gibbs_kJ_mol")]
    terms = [(name, 1)]+[(s, -n) for s, n in composition.items()]
    h0n = {s: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for s, r in records.items()}
    h0j = {s: 1000*(tab[298.15][4]+tab[0][3]) for s, tab in tables.items()}
    d0n = -math.fsum(n*h0n[s] for s,n in terms)
    d0j = -math.fsum(n*h0j[s] for s,n in terms)
    rows = []
    for t in inputs["temperatures_K"]:
        props = {}
        for s in records:
            nh, ns, ncp = nasa_properties(records[s], t, NASA_R)
            node, ref = tables[s][t], tables[s][298.15]
            props[s] = {"nasa_H_J_mol": nh, "nasa_S_J_mol_K": ns, "nasa_Cp_J_mol_K": ncp,
                        "janaf_H_J_mol": 1000*(ref[4]+node[3]), "janaf_S_J_mol_K": node[1], "janaf_Cp_J_mol_K": node[0]}
        energy = d0n-d0j
        thermal = math.fsum(n*((props[s]["janaf_H_J_mol"]-h0j[s])-(props[s]["nasa_H_J_mol"]-h0n[s])) for s,n in terms)
        entropy = -t*math.fsum(n*(props[s]["janaf_S_J_mol_K"]-props[s]["nasa_S_J_mol_K"]) for s,n in terms)
        delta = math.fsum([energy,thermal,entropy])
        nk = -math.fsum(n*(props[s]["nasa_H_J_mol"]-t*props[s]["nasa_S_J_mol_K"]) for s,n in terms)/(R*t)
        jk = nk-delta/(R*t)
        ck = current_ln_k(t, inputs["species_id"]) if t>=1500 else None
        row = {"temperature_K":t,"properties":props,"delta_H0_J_mol":energy,"delta_H_thermal_J_mol":thermal,"minus_T_delta_S_J_mol":entropy,"reaction_delta_G_J_mol":delta,
               "ln_K_nasa":nk,"ln_K_janaf":jk,"ln_K_curated":ck,"janaf_over_nasa":math.exp(jk-nk),
               "H0_aligned_janaf_over_nasa":math.exp(-(thermal+entropy)/(R*t)),
               "curated_over_nasa":None if ck is None else math.exp(ck-nk),
               "curated_over_janaf":None if ck is None else math.exp(ck-jk)}
        if "evaluated_nodes" in inputs:
            ev=inputs["evaluated_nodes"][str(t)]
            # Source-specific molecular evaluation with the declared NASA neutral atoms.
            eg=1000*(inputs["evaluated_hf298_kJ_mol"]+ev[3])-t*ev[1]
            ng=props[name]["nasa_H_J_mol"]-t*props[name]["nasa_S_J_mol_K"]
            ek=nk-(eg-ng)/(R*t)
            row["ln_K_evaluated_with_nasa_atoms"]=ek
            row["curated_over_evaluated_with_nasa_atoms"]=None if ck is None else math.exp(ck-ek)
        rows.append(row)
    modern={}
    if (data / "states.npz").exists():
        inventory=json.loads((data / "state_inventory.json").read_text())
        with np.load(data / "states.npz") as packet: states=packet["states"]
        pf=dict(np.loadtxt(data / "partition.pf"))
        spin=inputs["nuclear_spin_factor"]
        modern["inventory"]=inventory
        modern["samples"]=[]
        for row in rows:
            t=row["temperature_K"]; result=moments(states,t); nh=row["properties"][name]["nasa_H_J_mol"]; ns=row["properties"][name]["nasa_S_J_mol_K"]
            q=result["Q"]/spin
            lntrans=log_translation(t,inputs["natural_mass_u"])
            entropy=R*(math.log(q)+lntrans+2.5+result["mean_energy_K"]/t)
            nasaq=math.exp(ns/R-lntrans-(nh-h0n[name])/(R*t))
            w=states[:,1]*np.exp(-C2*states[:,0]/t)
            result.update({"temperature_K":t,"Q_without_nuclear_spin":q,"distributed_pf":pf[t],"state_sum_over_distributed_pf":result["Q"]/pf[t],
                           "Q_over_nasa":q/nasaq,"delta_Hthermal_J_mol":result["H_minus_H0_J_mol"]-(nh-h0n[name]),"delta_S_J_mol_K":entropy-ns,
                           "delta_Cp_J_mol_K":result["Cp_J_mol_K"]-row["properties"][name]["nasa_Cp_J_mol_K"],
                           "isotope_translation_delta_S_J_mol_K":1.5*R*math.log(inputs["isotope_mass_u"]/inputs["natural_mass_u"]),
                           "top_20_percent_energy_Q_fraction":float(w[states[:,0]>.8*max(states[:,0])].sum()/w.sum()),
                           "top_20_percent_J_Q_fraction":float(w[states[:,2]>.8*max(states[:,2])].sum()/w.sum()),
                           "electronic_fractions":{label:float(w[states[:,3]==i].sum()/w.sum()) for i,label in enumerate(inventory["electronic_labels"])}})
            modern["samples"].append(result)
    if name=="P2":
        modern={"W5prelim1_TAE0_kcal_mol":116.38,"W5prelim2_TAE0_kcal_mol":116.23,"W4_17_TAE0_kcal_mol":116.22,"Gurvich_TAE0_kcal_mol":116.05,"Gurvich_uncertainty_kcal_mol":.09,
                "W5prelim2_minus_janaf_D0_J_mol":116.23*4184-d0j,
                "energy_only_K_multiplier_if_same_thermal":{str(t):math.exp((116.23*4184-d0j)/(R*t)) for t in (1500,3000,4000)}}
    return {"molecule":name,"reaction":inputs["reaction"],"standard_pressure_Pa":100000,"comparison_R_J_mol_K":R,"NASA_native_R_J_mol_K":NASA_R,
            "NASA_coefficients_matched":len(printed),"JANAF_frozen_nodes_matched":len(old["samples"]),"nasa_D0_J_mol":d0n,"janaf_D0_J_mol":d0j,
            "delta_convention":"JANAF minus NASA, molecule minus neutral atoms; uncorrected JANAF PH is diagnostic only",
            "samples":rows,"modern":modern,"footprint":footprint(inputs["species_id"],composition["P"]),"curated_change":"none","new_equilibrium_solves":0,"physical_accuracy":"unknown"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule",choices=("p2","po","ph","ps"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=json.dumps(review(args.molecule),indent=2,allow_nan=False)+"\n"
    if args.output: args.output.write_text(result)
    else: print(result,end="")
