"""Replay oxirane source thermodynamics and the final curated baseline offline."""

from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, BASELINE
from ph3_source_review import R, NASA_R, log_translation
from phosphorus_diatomic_source_review import moments, C2

DATA = ROOT / "tests/data/validation/oxirane_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"


@lru_cache(maxsize=1)
def setup():
    """Keep this review on its named, adopted baseline."""
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update("jax_enable_x64", True)
    return curated_tio2.chemsetup(ions=True)


@lru_cache(maxsize=None)
def current_properties(temperature: float, species: str) -> dict:
    """Differentiate the literal reaction free energy, without fitting an anchor."""
    import jax
    if not 1500 <= temperature <= 4000:
        raise ValueError("outside curated-tio2-v1 range")
    model = setup()
    column = model.species.index(species)
    h = lambda t: model.hvector_func(t)[column]
    t = float(temperature)
    value, first, second = float(h(t)), float(jax.grad(h)(t)), float(jax.grad(jax.grad(h))(t))
    return {"G_J_mol": R*t*value, "H_J_mol": -R*t*t*first,
            "S_J_mol_K": -R*(value+t*first), "Cp_J_mol_K": -R*(2*t*first+t*t*second),
            "ln_K": -value}


def review() -> dict:
    """Balance neutral atoms before aligning H0; preserve source gas constants."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    name, composition = inputs["name"], inputs["composition"]
    records = inputs["nasa_records"]
    baseline = json.loads((BASELINE / "inputs.json").read_text())
    tables = {s: read_table(DATA / p) for s, p in inputs["janaf_files"].items()}
    raw=(DATA/"nasa-selected.txt").read_text().splitlines()
    printed=[]; offset=0
    for species,record in records.items():
        count=int(raw[offset+1][:2])
        if raw[offset].split()[0] != record["source_name"]:
            raise ValueError("NASA identity mismatch")
        for i,interval in enumerate(record["intervals"]):
            if [float(v) for v in raw[offset+2+3*i][:22].split()] != interval["bounds_K"]:
                raise ValueError("NASA interval mismatch")
            a,z=raw[offset+3+3*i],raw[offset+4+3*i]
            values=[float(a[k:k+16].replace("D","E")) for k in range(0,80,16)]
            values += [float(z[k:k+16].replace("D","E")) for k in (0,16,48,64)]
            if values != interval["coefficients"]:
                raise ValueError("official NASA coefficient mismatch")
            printed.extend(values)
        if count != len(record["intervals"]):
            raise ValueError("NASA interval count mismatch")
        offset+=2+3*count
        previous=baseline["nasa9_records"].get(record["source_name"])
        if previous and [r["coefficients"] for r in record["intervals"]] != [r["coefficients"] for r in previous["intervals"]]:
            raise ValueError("NASA existing baseline mismatch")
    terms = [(name, 1)] + [(s, -n) for s, n in composition.items()]
    h0n = {s: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for s, r in records.items()}
    h0j = {s: 1000*(tab[298.15][4]+tab[0][3]) for s, tab in tables.items()}
    d0n, d0j = (-math.fsum(n*zero[s] for s,n in terms) for zero in (h0n,h0j))
    rows = []
    for t in inputs["temperatures_K"]:
        props = {}
        for s, record in records.items():
            nh, ns, ncp = nasa_properties(record,t,NASA_R)
            node, ref = tables[s][t], tables[s][298.15]
            jh, js = 1000*(ref[4]+node[3]), node[1]
            props[s] = {"nasa": {"H_J_mol":nh,"S_J_mol_K":ns,"Cp_J_mol_K":ncp,"G_J_mol":nh-t*ns},
                        "janaf": {"H_J_mol":jh,"S_J_mol_K":js,"Cp_J_mol_K":node[0],"G_J_mol":jh-t*js}}
        reaction = {source: {key: math.fsum(n*props[s][source][key] for s,n in terms)
                             for key in ("G_J_mol","H_J_mol","S_J_mol_K","Cp_J_mol_K")}
                    for source in ("nasa","janaf")}
        for source in reaction: reaction[source]["ln_K"] = -reaction[source]["G_J_mol"]/(R*t)
        reaction["curated"] = current_properties(t,inputs["species_id"])
        thermal = math.fsum(n*((props[s]["janaf"]["H_J_mol"]-h0j[s])-(props[s]["nasa"]["H_J_mol"]-h0n[s])) for s,n in terms)
        entropy = -t*(reaction["janaf"]["S_J_mol_K"]-reaction["nasa"]["S_J_mol_K"])
        rows.append({"temperature_K":t,"properties":props,"reaction":reaction,
                     "delta_H0_J_mol":d0n-d0j,"delta_Hthermal_J_mol":thermal,"minus_T_delta_S_J_mol":entropy,
                     "reaction_delta_G_J_mol":reaction["janaf"]["G_J_mol"]-reaction["nasa"]["G_J_mol"],
                     "janaf_over_nasa":math.exp(reaction["janaf"]["ln_K"]-reaction["nasa"]["ln_K"]),
                     "H0_aligned_janaf_over_nasa":math.exp(-(thermal+entropy)/(R*t)),
                     "curated_over_nasa":math.exp(reaction["curated"]["ln_K"]-reaction["nasa"]["ln_K"]),
                     "curated_over_janaf":math.exp(reaction["curated"]["ln_K"]-reaction["janaf"]["ln_K"])})
    with np.load(GRID) as grid:
        col=grid["species"].tolist().index(inputs["species_id"])
        shares={s:n*np.exp(grid["ln_n"][:,col])/grid["budget"][:,grid["elements"].tolist().index(s)] for s,n in composition.items()}
        element=max(shares,key=lambda s:max(shares[s])); case=int(np.argmax(shares[element])); mx=int(np.argmax(grid["ln_x"][:,col]))
        condition=lambda i:{"case_index":i,"temperature_K":float(grid["temperature_K"][i]),"pressure_bar":float(grid["pressure_bar"][i]),"C_over_O":float(grid["C_over_O"][i])}
        fp={"model":"curated-tio2-v1","cases":len(grid["temperature_K"]),"species":len(grid["species"]),"all_converged":bool(np.all(grid["converged"])),
            "maximum_element":element,"maximum_element_fraction":float(shares[element][case]),"maximum_share_condition":condition(case),
            "maximum_mole_fraction":float(np.exp(grid["ln_x"][mx,col])),"maximum_mole_fraction_condition":condition(mx),
            "maximum_abs_target_h_error":max(abs(float(h)-float(setup().hvector_func(float(t))[col])) for t,h in zip(grid["temperature_K"],grid["h"][:,col]))}
    modern=[]
    if (DATA/"states.npz").exists():
        with np.load(DATA/"states.npz") as packet: states=packet["states"]
        pf=dict(np.loadtxt(DATA/"partition.pf")); spin=inputs["nuclear_spin_factor"]
        for row in rows:
            t=row["temperature_K"]; v=moments(states,t); q=v["Q"]/spin
            entropy=R*(math.log(q)+log_translation(t,inputs["natural_mass_u"])+v["H_minus_H0_J_mol"]/(R*t))
            p=row["properties"][name]["nasa"]
            nq=math.exp(p["S_J_mol_K"]/R-log_translation(t,inputs["natural_mass_u"])-(p["H_J_mol"]-h0n[name])/(R*t))
            w=states[:,1]*np.exp(-C2*states[:,0]/t)
            v.update({"temperature_K":t,"Q_without_nuclear_spin":q,"state_sum_over_distributed_pf":v["Q"]/pf[t],"Q_over_nasa":q/nq,
                      "delta_Hthermal_J_mol":v["H_minus_H0_J_mol"]-(p["H_J_mol"]-h0n[name]),"delta_S_J_mol_K":entropy-p["S_J_mol_K"],
                      "delta_Cp_J_mol_K":v["Cp_J_mol_K"]-p["Cp_J_mol_K"],
                      "top_20_percent_energy_Q_fraction":float(w[states[:,0]>.8*states[:,0].max()].sum()/w.sum()),
                      "isotope_translation_delta_S_J_mol_K":1.5*R*math.log(inputs["isotope_mass_u"]/inputs["natural_mass_u"])})
            modern.append(v)
    energy_diagnostic={}
    if "atct" in inputs:
        evaluated=inputs["atct"]["species"]
        d0=1000*(math.fsum(n*evaluated[s]["hf0_kJ_mol"] for s,n in composition.items())-evaluated[name]["hf0_kJ_mol"])
        energy_diagnostic={"version":inputs["atct"]["version"],"D0_J_mol":d0,"minus_janaf_D0_J_mol":d0-d0j,"minus_nasa_D0_J_mol":d0-d0n,
                           "energy_only_K_multiplier_retaining_janaf_thermal":{str(t):math.exp((d0-d0j)/(R*t)) for t in inputs["temperatures_K"]},
                           "uncertainty":"Marginal species uncertainties are recorded; reaction covariance is not inferred by independent quadrature."}
    return {"molecule":name,"evaluated_energy":energy_diagnostic,"species_id":inputs["species_id"],"reaction":inputs["reaction"],"standard_pressure_Pa":100000,
            "R_J_mol_K":R,"NASA_native_R_J_mol_K":NASA_R,"nasa_D0_J_mol":d0n,"janaf_D0_J_mol":d0j,
            "NASA_coefficients_matched":len(printed),"samples":rows,"modern":modern,"footprint":fp,
            "curated_change":"none","new_equilibrium_solves":0,"physical_accuracy":"unknown"}


if __name__ == "__main__":
    print(json.dumps(review(),indent=2,allow_nan=False))
