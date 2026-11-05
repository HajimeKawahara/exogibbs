"""Replay the CS2 source review offline; no production or equilibrium changes."""

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT
import select_major_molecules as major

DATA = ROOT / "tests/data/validation/cs2_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_oh_adoption/20261105/ions.npz"
TEMPERATURES = (300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK*29979245800/KB


@lru_cache(maxsize=1)
def setup():
    """Load the current packaged model once; float64 is required for replay."""
    import jax
    from exogibbs.presets import curated
    jax.config.update("jax_enable_x64", True)
    return curated.chemsetup(ions=True)


def curated_ln_k(temperature: float) -> float:
    """Return formation ln K only inside the declared production domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated range")
    inputs = json.loads((DATA / "inputs.json").read_text())
    model = setup()
    return -float(model.hvector_func(float(temperature))[model.species.index(inputs["species_id"])])


def log_translation(temperature: float, mass: float) -> float:
    """One-bar ideal-gas translation, with an explicit mass convention."""
    kt = KB*temperature
    return 1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*kt/PLANCK**2)+math.log(kt/100000)


def table_calorics(table: np.ndarray, temperature: float, step: float, fixed_decimals=None) -> dict:
    """Differentiate rounded ln Q with a five-point stencil; this is diagnostic.

    A second step size checks truncation. The propagated half-last-digit bound
    covers tabulation rounding only, not incomplete levels or physical error.
    """
    nodes = np.array([temperature+i*step for i in (-2,-1,0,1,2)])
    rows = table[np.isin(table[:, 0], nodes)]
    if len(rows) != 5:
        raise ValueError("Missing partition-table stencil node")
    lq = np.log(rows[:, 1])
    first = np.array([1,-8,0,8,-1])/(12*step)
    second = np.array([-1,16,-30,16,-1])/(12*step**2)
    d1, d2 = first @ lq, second @ lq
    # TIPS has seven significant digits; ExoMol PF uses fixed decimals.
    dq = (.5*10.0**(np.floor(np.log10(rows[:,1]))-6) if fixed_decimals is None
          else np.full(5, .5*10.0**(-fixed_decimals)))
    err = dq/rows[:,1]
    return {"Q": float(rows[2,1]), "H_increment_J_mol": float(major.R*(2.5*temperature+temperature**2*d1)),
            "Cp_J_mol_K": float(major.R*(2.5+2*temperature*d1+temperature**2*d2)),
            "Cp_rounding_bound_J_mol_K": float(major.R*((2*temperature*abs(first)+temperature**2*abs(second)) @ err))}


def footprint(species_id: str) -> dict:
    """Replay archived abundance allocation; never label this a new solve."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index(species_id)
        sulfur = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        shares = grid["formula_matrix"][sulfur,column]*n[:,column]/grid["budget"][:,sulfur]
        i = int(np.argmax(shares))
        return {"model":"curated-oh-v1", "cases":len(n), "species":n.shape[1],
                "all_converged":bool(np.all(grid["converged"])),
                "maximum_abs_h_error":max(abs(h+curated_ln_k(t)) for t,h in zip(grid["temperature_K"],grid["h"][:,column])),
                "maximum_sulfur_fraction":float(shares[i]),"case_index":i,
                "mole_fraction_at_maximum_sulfur_share":float(np.exp(grid["ln_x"][i,column])),
                "temperature_K":float(grid["temperature_K"][i]),"pressure_bar":float(grid["pressure_bar"][i]),"C_over_O":float(grid["C_over_O"][i]),
                "maximum_relative_sulfur_conservation_residual":float(np.max(abs(n @ grid["formula_matrix"][sulfur]/grid["budget"][:,sulfur]-1)))}


def review() -> dict:
    """Separate source energy, thermal functions and finite-state diagnostics."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    reference = json.loads(major.REFERENCE_PATH.read_text())
    name = inputs["name"]
    reference["species"][name] = {"composition":inputs["composition"]}
    reference["nasa9_records"].update(inputs["extra_nasa"])
    energy = {s:major.atomization_kelvin(name,s,reference) for s in ("nasa9","janaf","atct") if s=="nasa9" or name in reference.get(s,{}).get("records",reference.get(s,{}))}
    latest = inputs["atct_1.228"]
    new_d0 = None if latest is None else 1000*(sum(n*latest[e] for e,n in inputs["composition"].items())-latest[name])
    record = reference["nasa9_records"][name]
    tips = np.loadtxt(DATA/inputs["tips2025"]["file"]) if "tips2025" in inputs else None
    rows = []
    for t in TEMPERATURES:
        ng = major.thermal_g_rt(name,t,"nasa9",reference)
        h,s,cp = nasa_properties(record,t,major.NASA_R)
        qn = math.exp(-ng-log_translation(t,record["mass_u"]))
        nk = major.formation_log_kp(name,t,"nasa9",energy["nasa9"],reference)
        ck = curated_ln_k(t) if t>=1500 else None
        row = {"temperature_K":t,"nasa_ln_K":nk,"curated_ln_K":ck,
               "curated_over_nasa":None if ck is None else math.exp(ck-nk),
               "nasa_Q":qn,"nasa_common_R_Cp_J_mol_K":cp*major.R/major.NASA_R}
        if "janaf" in energy and str(float(t)) in reference["janaf"][name]["rows"]:
            jg=major.thermal_g_rt(name,t,"janaf",reference)
            node=major.janaf_row(name,t,reference)
            row["janaf"]={"same_D0_atoms_K_over_nasa":math.exp(ng-jg),"Cp_J_mol_K":node[0],
                          "delta_Cp_J_mol_K":node[0]-cp*major.R/major.NASA_R,
                          "native_ln_K":major.formation_log_kp(name,t,"janaf",energy["janaf"],reference,"janaf")}
        if name == "CS2" or str(float(t)) in reference["janaf"][name]["rows"]:
            adopted_thermal = "janaf" if name == "COS" else "nasa9"
            adopted_energy = energy["janaf" if name == "COS" else "atct"]
            row["reconstructed_adopted_ln_K"] = major.formation_log_kp(name,t,adopted_thermal,adopted_energy,reference)
        if new_d0 is not None:
            old_energy=energy["janaf" if name=="COS" else "atct"]
            row["new_ATcT_energy_only_K_ratio"]=math.exp((new_d0/major.R-old_energy)/t)
        if tips is not None:
            tc = table_calorics(tips,t,10)
            tc["Cp_step20_J_mol_K"] = table_calorics(tips,t,20)["Cp_J_mol_K"]
            tc["Q_over_nasa"] = tc["Q"]/qn
            tc["isotope_translation_K_proxy"] = tc["Q"]/qn*(inputs["tips2025"]["mass_u"]/record["mass_u"])**1.5
            tc["delta_Cp_J_mol_K"] = tc["Cp_J_mol_K"]-cp*major.R/major.NASA_R
            row["tips2025"] = tc
        rows.append(row)
    cross = 3380 if name=="COS" else 2100
    tq=dict(tips)
    join_ratio=tq[cross]*tq[cross-20]/tq[cross-10]**2
    return {"tips_cross_over_grid_K":cross,"tips_join_over_previous_log_slope":join_ratio,
            "molecule":name,"reaction_atoms":inputs["composition"],"standard_pressure_Pa":100000,
            "balanced_D0_J_mol":{s:e*(major.NASA_R if s=="nasa9" else major.R) for s,e in energy.items()},
            "atct_1.228_balanced_D0_J_mol":new_d0,"source_samples":rows,
            "footprint":footprint(inputs["species_id"]),"production_change":"none","equilibrium_substitution":"not_run"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=json.dumps(review(),indent=2,allow_nan=False)+"\n"
    if args.output: args.output.write_text(result)
    else: print(result,end="")
