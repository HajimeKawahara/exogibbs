"""Shared offline normalization for the CN, CH and CH3 source packets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, file_hash

R, NASA_R = 8.31446261815324, 8.314510
KB, PLANCK, AMU = 1.380649e-23, 6.62607015e-34, 1.66053906660e-27
C2 = PLANCK * 299792458 * 100 / KB
GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105/ions.npz"


def curated_properties(species: str, t: float) -> dict:
    """Differentiate the inherited literal fit only in the reviewed domain."""
    if not 1500 <= t <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    lines = (ROOT / "src/exogibbs/data/fastchem/logK/logK.dat").read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(species + " "))
    a, b, c, d, e = map(float, lines[index + 1].split())
    return {"ln_K": a/t+b*math.log(t)+c+d*t+e*t*t,
            "H_J_mol": R*(-a+b*t+d*t*t+2*e*t**3),
            "S_J_mol_K": R*(b*math.log(t)+b+c+2*d*t+3*e*t*t),
            "Cp_J_mol_K": R*(b+2*d*t+6*e*t*t)}


def source_comparison(config: dict, tables: dict, nasa: dict, t: float) -> dict:
    """Separate atom-balanced zero energy from molecular and atomic thermals."""
    name, comp = config["molecule"], config["composition"]
    nh0 = {n: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*(tb[298.15][4]+tb[0.0][3]) for n, tb in tables.items()}
    nd0 = math.fsum(c*nh0[n] for n, c in comp.items())-nh0[name]
    jd0 = math.fsum(c*jh0[n] for n, c in comp.items())-jh0[name]
    jp = {n: (1000*(tb[298.15][4]+tb[t][3]), tb[t][1], tb[t][0]) for n, tb in tables.items()}
    np_ = {n: nasa_properties(r, t, NASA_R) for n, r in nasa.items()}
    phi_j = {n: (h-jh0[n])/(R*t)-s/R for n, (h,s,cp) in jp.items()}
    phi_n = {n: (h-nh0[n])/(NASA_R*t)-s/NASA_R for n,(h,s,cp) in np_.items()}
    terms = {"energy": (jd0/R-nd0/NASA_R)/t,
             "molecular_thermal": phi_n[name]-phi_j[name],
             "atomic_thermal": math.fsum(c*(phi_j[n]-phi_n[n]) for n,c in comp.items())}
    ln_n = math.fsum(c*(np_[n][0]/t-np_[n][1])/NASA_R for n,c in comp.items())-(np_[name][0]/t-np_[name][1])/NASA_R
    reaction = {key: jp[name][i]-math.fsum(c*jp[n][i] for n,c in comp.items())
                for i,key in enumerate(("H_J_mol","S_J_mol_K","Cp_J_mol_K"))}
    ln_j = -(reaction["H_J_mol"]/t-reaction["S_J_mol_K"])/R
    return {"ln_K": ln_j, "nasa_ln_K": ln_n, "janaf_over_nasa": math.exp(ln_j-ln_n),
            "janaf_minus_nasa_ln_K_components": terms,
            "decomposition_closure_ln_K": ln_j-ln_n-math.fsum(terms.values()),
            "energy_aligned_janaf_over_nasa": math.exp(terms["molecular_thermal"]+terms["atomic_thermal"]),
            "janaf_D0_J_mol": jd0, "nasa_D0_J_mol": nd0,
            "molecular": {"janaf_HminusH0_J_mol": jp[name][0]-jh0[name], "janaf_S_J_mol_K": jp[name][1],
                          "janaf_Cp_J_mol_K": jp[name][2], "nasa_HminusH0_J_mol": np_[name][0]-nh0[name],
                          "nasa_S_J_mol_K": np_[name][1], "nasa_Cp_J_mol_K": np_[name][2]},
            "reaction": reaction}


def ln_translation(mass: float, t: float) -> float:
    """Ideal-gas translation at one bar, excluding internal degeneracies."""
    kt = KB*t
    return 1.5*math.log(2*math.pi*mass*AMU*kt/PLANCK**2)+math.log(kt/100000)


def partition_properties(q: float, mean: float, variance: float, t: float, mass: float) -> dict:
    """Compute H-H0, S and Cp from finite-state energy moments in cm-1."""
    return {"Q": q, "HminusH0_J_mol": R*(2.5*t+C2*mean),
            "S_J_mol_K": R*(math.log(q)+ln_translation(mass,t)+2.5+C2*mean/t),
            "Cp_J_mol_K": R*(2.5+(C2/t)**2*variance)}


def state_moments(energies: np.ndarray, weights: np.ndarray, t: float) -> tuple:
    """Return Q, mean and central variance with the archived absolute zero."""
    population = weights*np.exp(-C2*energies/t)
    q = float(np.sum(population))
    mean = float(np.dot(population, energies)/q)
    variance = float(np.dot(population, (energies-mean)**2)/q)
    return q, mean, variance


def partition_diagnostic(data: Path, config: dict, record: dict, t: float) -> dict:
    """Keep published PF, actual state data and derivative scope distinct."""
    part = config["partition"]
    h,s,cp = nasa_properties(record,t,NASA_R)
    h0 = record["hf298_J_mol"]-record["h298_minus_h0_J_mol"]
    q_nasa = math.exp(-(h-h0)/(NASA_R*t)+s/NASA_R-ln_translation(record["mass_u"],t))
    table = np.loadtxt(data / part["file"])
    match = table[table[:,0] == t]
    q = float(match[0,1])/part["nuclear_spin_divisor"] if len(match)==1 else None
    result = {"nasa_internal_Q": q_nasa, "published_spin_normalized_Q": q,
              "published_Q_over_nasa": None if q is None else q/q_nasa,
              "published_isotope_translation_K_proxy": None if q is None else q/q_nasa*(part["mass_u"]/record["mass_u"])**1.5,
              "published_range_K": [float(table[0,0]),float(table[-1,0])],
              "published_table_derivatives": "not inferred from rounded PF values"}
    if config["molecule"] == "CN":
        with np.load(data/part["states_file"]) as states:
            energy, weight = states["energy_cm1"], states["gtot"]/part["nuclear_spin_divisor"]
            moments = state_moments(energy,weight,t)
            props = partition_properties(*moments,t,record["mass_u"])
            # Translation at NASA mass isolates the internal isotope comparison.
            difference = {"HminusH0_J_mol": props["HminusH0_J_mol"]-(h-h0)*R/NASA_R,
                          "S_J_mol_K": props["S_J_mol_K"]-s*R/NASA_R,
                          "Cp_J_mol_K": props["Cp_J_mol_K"]-cp*R/NASA_R}
            result["states"] = {"properties_at_NASA_mass": props, "minus_NASA_thermal": difference,
                "Q_over_published": None if q is None else props["Q"]/q,
                "energy_above_40000cm1_Q_fraction": float(np.sum(weight[energy>40000]*np.exp(-C2*energy[energy>40000]/t))/moments[0]),
                "J_above_100_Q_fraction": float(np.sum(weight[states["J"]>100]*np.exp(-C2*energy[states["J"]>100]/t))/moments[0]),
                "excited_electronic_Q_fraction": float(np.sum(weight[states["electronic_state"]!='X(2SIGMA+)']*np.exp(-C2*energy[states["electronic_state"]!='X(2SIGMA+)']/t))/moments[0]),
                "coverage": "internal tail indicators, not an upper bound on omitted electronic states"}
    if config["molecule"] == "CH" and t >= 1000:
        a,b,c = config["masseron_log_Q_coefficients"]
        derivative = b+2*c*math.log(t)
        mq = math.exp(a+b*math.log(t)+c*math.log(t)**2)
        props = {"Q":mq,"HminusH0_J_mol":R*t*(2.5+derivative),
                 "S_J_mol_K":R*(math.log(mq)+ln_translation(record["mass_u"],t)+2.5+derivative),
                 "Cp_J_mol_K":R*(2.5+derivative+2*c)}
        result["masseron2014_analytic_fit"] = {"properties_at_NASA_mass":props,"Q_over_nasa":mq/q_nasa,
            "minus_NASA_thermal":{"HminusH0_J_mol":props["HminusH0_J_mol"]-(h-h0)*R/NASA_R,"S_J_mol_K":props["S_J_mol_K"]-s*R/NASA_R,"Cp_J_mol_K":props["Cp_J_mol_K"]-cp*R/NASA_R},
            "scope":"printed ln Q quadratic diagnostic only; derivatives are not independently validated source calorics"}
    return result


def state_inventory(data: Path, config: dict) -> dict | None:
    """Expose the actual downloaded fields without repairing suspect labels."""
    filename=config["partition"]["states_file"]
    if filename is None:
        return None
    with np.load(data/filename) as states:
        labels=states["electronic_state"]
        return {"rows":len(labels),"minimum_energy_cm1":float(states["energy_cm1"].min()),
                "maximum_J":float(states["J"].max()),
                "maximum_spin_weight_error":float(np.max(abs(states["gtot"]-config["partition"]["nuclear_spin_divisor"]*(2*states["J"]+1)))),
                "electronic_states":{n:{"rows":int(np.sum(labels==n)),"minimum_energy_cm1":float(states["energy_cm1"][labels==n].min()),"maximum_energy_cm1":float(states["energy_cm1"][labels==n].max())} for n in sorted(set(labels.tolist()))}}


def footprint(config: dict) -> dict:
    """Measure importance only on the frozen 75-case curated-cos-v1 grid."""
    with np.load(GRID) as grid:
        col=grid["species"].tolist().index(config["species_id"])
        el=grid["elements"].tolist().index(config["footprint_element"])
        n=np.exp(grid["ln_n"])
        share=grid["formula_matrix"][el,col]*n[:,col]/grid["budget"][:,el]
        i=int(np.argmax(share))
        return {"model":"curated-cos-v1","cases":len(n),"species_count":n.shape[1],"all_converged":bool(np.all(grid["converged"])),
                "element":config["footprint_element"],"maximum_element_share":float(share[i]),"case_index":i,
                "maximum_mole_fraction":float(np.exp(grid["ln_x"][:,col]).max()),"temperature_K":float(grid["temperature_K"][i]),
                "pressure_bar":float(grid["pressure_bar"][i]),"C_over_O":float(grid["C_over_O"][i]),
                "maximum_abs_h_error":max(abs(h+curated_properties(config["species_id"],float(t))["ln_K"]) for t,h in zip(grid["temperature_K"],grid["h"][:,col]))}


def review(slug: str) -> dict:
    """Replay a selected source packet without creating or adopting a model."""
    if slug not in ("cn","ch","ch3"):
        raise ValueError("Unknown reviewed carbon radical")
    data=ROOT/f"tests/data/validation/{slug}_source_review/20261106"
    config=json.loads((data/"inputs.json").read_text())
    name,comp=config["molecule"],config["composition"]
    nasa=json.loads((data/"nasa_records.json").read_text())
    tables={n:read_table(ROOT/p) for n,p in config["atomic_tables"].items()}
    tables[name]=read_table(data/config["molecular_janaf_table"])
    atct=config["atct"]
    modern=1000*(math.fsum(c*atct[n]["hf0_kJ_mol"] for n,c in comp.items())-atct[name]["hf0_kJ_mol"])
    rows=[]
    for t in config["temperatures_K"]:
        source=source_comparison(config,tables,nasa,t)
        current=curated_properties(config["species_id"],t) if t>=1500 else None
        diagnostic=source["nasa_ln_K"]+(modern/R-source["nasa_D0_J_mol"]/NASA_R)/t
        rows.append({"temperature_K":t,"source":source,"curated":current,
                     "curated_over_nasa":None if current is None else math.exp(current["ln_K"]-source["nasa_ln_K"]),
                     "curated_minus_janaf_reaction":None if current is None else {k:current[k]-v for k,v in source["reaction"].items()},
                     "same_thermal_ATcT_energy_only_K_ratio":math.exp((modern-source["janaf_D0_J_mol"])/(R*t)),
                     "NASA_thermal_ATcT_diagnostic_K_over_curated":None if current is None else math.exp(diagnostic-current["ln_K"]),
                     "partition":partition_diagnostic(data,config,nasa[name],t)})
    result={"molecule":name,"species_id":config["species_id"],"composition":comp,"standard_pressure_Pa":100000,
            "R_J_mol_K":R,"NASA_native_R_J_mol_K":NASA_R,"balanced_ATcT_D0_J_mol":modern,
            "inherited_janaf_D0_J_mol":rows[0]["source"]["janaf_D0_J_mol"],
            "ATcT_minus_janaf_D0_J_mol":modern-rows[0]["source"]["janaf_D0_J_mol"],
            "samples":rows,"state_inventory":state_inventory(data,config),"footprint":footprint(config),
            "production_change":"none","candidate_model":"not_created","equilibrium_substitution":"not_run","physical_accuracy":"unknown"}
    if slug=="ch":
        result["masseron2014_D0_J_mol"]=config["masseron_D0_cm-1"]*C2*R
        result["ATcT_quartet_minus_doublet_kJ_mol"]=atct["CH_quartet"]["hf0_kJ_mol"]-atct["CH"]["hf0_kJ_mol"]
    return result
