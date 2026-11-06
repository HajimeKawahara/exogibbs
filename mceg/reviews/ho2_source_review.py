"""Replay HO2 thermochemical sources and the existing hot-gas footprint offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash


DATA = ROOT / "tests/data/validation/ho2_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_oh_adoption/20261105/ions.npz"
FASTCHEM = ROOT / "src/exogibbs/data/fastchem/logK/logK.dat"
NAME, SPECIES = "HO2", "H1O2"
COMPOSITION = {'H': 1, 'O': 2}
R, NASA_R = 8.31446261815324, 8.314510


def curated_properties(temperature: float) -> dict:
    """Return formation ln K and its analytic reaction H, S, Cp derivatives."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    lines = FASTCHEM.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + " "))
    a, b, c, d, e = map(float, lines[i+1].split())
    t = temperature
    value = a/t+b*math.log(t)+c+d*t+e*t*t
    return dict(ln_K=value, H_J_mol=R*(-a+b*t+d*t*t+2*e*t**3),
                S_J_mol_K=R*(b*math.log(t)+b+c+2*d*t+3*e*t*t),
                Cp_J_mol_K=R*(b+2*d*t+6*e*t*t))


def footprint(element: str) -> dict:
    """Reuse archived endpoints; no candidate substitution or new solve."""
    with np.load(GRID) as grid:
        col = grid["species"].tolist().index(SPECIES)
        el = grid["elements"].tolist().index(element)
        n = np.exp(grid["ln_n"])
        shares = grid["formula_matrix"][el, col]*n[:, col]/grid["budget"][:, el]
        i = int(np.argmax(shares))
        return dict(model="curated-oh-v1", cases=len(n), species_count=n.shape[1],
                    all_converged=bool(grid["converged"].all()), element=element,
                    case_index=i, maximum_element_share=float(shares[i]),
                    mole_fraction_at_maximum_share=float(np.exp(grid["ln_x"][i, col])),
                    maximum_mole_fraction=float(np.exp(grid["ln_x"][:, col]).max()),
                    temperature_K=float(grid["temperature_K"][i]),
                    pressure_bar=float(grid["pressure_bar"][i]),
                    C_over_O=float(grid["C_over_O"][i]),
                    maximum_abs_h_error=max(abs(h+curated_properties(t)["ln_K"])
                        for t, h in zip(grid["temperature_K"], grid["h"][:, col])),
                    maximum_relative_element_residual=float(np.max(np.abs(
                        n @ grid["formula_matrix"][el]/grid["budget"][:, el]-1))),
                    new_equilibrium_substitution="not_run")


def table_properties(table: dict, temperature: float) -> tuple:
    """Return H, S, Cp using fixed 298 K formation anchor and thermal increment."""
    node = table[temperature]
    return 1000*(table[298.15][4]+node[3]), node[1], node[0]


def source_comparison(tables: dict, nasa: dict, temperature: float) -> dict:
    """Balance energy and molecular/atomic thermal terms in native R conventions."""
    t = temperature
    nh0 = {n: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*(tb[298.15][4]+tb[0.0][3]) for n, tb in tables.items()}
    nd0 = sum(c*nh0[n] for n, c in COMPOSITION.items())-nh0[NAME]
    jd0 = sum(c*jh0[n] for n, c in COMPOSITION.items())-jh0[NAME]
    jp = {n: table_properties(tb, t) for n, tb in tables.items()}
    np_ = {n: nasa_properties(r, t, NASA_R) for n, r in nasa.items()}
    phi_j = {n: (h-jh0[n])/(R*t)-s/R for n, (h,s,cp) in jp.items()}
    phi_n = {n: (h-nh0[n])/(NASA_R*t)-s/NASA_R for n,(h,s,cp) in np_.items()}
    terms = dict(energy=(jd0/R-nd0/NASA_R)/t,
                 molecular_thermal=phi_n[NAME]-phi_j[NAME],
                 atomic_thermal=sum(c*(phi_j[n]-phi_n[n]) for n,c in COMPOSITION.items()))
    ln_n = sum(c*(np_[n][0]/t-np_[n][1])/NASA_R for n,c in COMPOSITION.items())-(np_[NAME][0]/t-np_[NAME][1])/NASA_R
    reaction = {key: jp[NAME][i]-sum(c*jp[n][i] for n,c in COMPOSITION.items())
                for i,key in enumerate(("H_J_mol","S_J_mol_K","Cp_J_mol_K"))}
    ln_j = -(reaction["H_J_mol"]/t-reaction["S_J_mol_K"])/R
    return dict(ln_K=ln_j, nasa_ln_K=ln_n, source_over_nasa=math.exp(ln_j-ln_n),
                source_minus_nasa_ln_K_components=terms,
                decomposition_closure_ln_K=ln_j-ln_n-sum(terms.values()),
                energy_aligned_source_over_nasa=math.exp(terms["molecular_thermal"]+terms["atomic_thermal"]),
                native_source_D0_kJ_mol=jd0/1000, native_nasa_D0_kJ_mol=nd0/1000,
                molecular=dict(source_H_J_mol=jp[NAME][0],source_S_J_mol_K=jp[NAME][1],
                    source_Cp_J_mol_K=jp[NAME][2],nasa_H_J_mol=np_[NAME][0],
                    nasa_S_J_mol_K=np_[NAME][1],nasa_Cp_J_mol_K=np_[NAME][2]),
                reaction=reaction)


def review() -> dict:
    """Verify pinned sources, reproduce exact nodes and keep applicability scoped."""
    sources = json.loads((DATA / "sources.json").read_text())
    for name, expected in sources["existing_inputs"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Input hash mismatch: "+name)
    for name, expected in sources["archived_inputs"].items():
        if file_hash(DATA / name) != expected:
            raise ValueError("Source hash mismatch: "+name)
    nasa = json.loads((DATA / "nasa_records.json").read_text())
    frozen = json.loads((BASELINE / "inputs.json").read_text())
    raw = (DATA / "nasa_selected_records.txt").read_text().split("\n\n")
    for name, record in nasa.items():
        old = frozen["nasa9_records"][name]
        coefficients = [x for row in record["intervals"] for x in row["coefficients"]]
        if coefficients != [x for row in old["intervals"] for x in row["coefficients"]]:
            raise ValueError("Frozen NASA coefficient mismatch: "+name)
        lines = next(block.splitlines() for block in raw if block.startswith(name+" "))
        if [float(lines[1].split()[-1]),float(lines[2].split()[-1])] != [record["hf298_J_mol"],record["h298_minus_h0_J_mol"]]:
            raise ValueError("NASA header mismatch: "+name)
        printed=[]
        for i in range(len(record["intervals"])):
            values=[float(x.replace("D","E")) for x in re.findall(r"[-+]?\d+\.\d+D[-+]\d+", "\n".join(lines[3+3*i:5+3*i]))]
            printed.extend(values[:7]+values[-2:])
        if printed != coefficients:
            raise ValueError("Printed NASA coefficient mismatch: "+name)
    tables = {n:read_table(ROOT/p) for n,p in sources["atomic_tables"].items()}
    tables[NAME] = read_table(DATA / sources["molecular_janaf_table"])
    selected = production_tables(tables)
    baseline = json.loads((BASELINE / "comparison.json").read_text())["samples"]
    rows=[]
    for t in sources["temperatures_K"]:
        legacy=source_comparison(tables,nasa,t)
        source=source_comparison(selected,nasa,t)
        old=next((r for r in baseline if r["species_id"]==SPECIES and r["temperature_K"]==t),None)
        current=curated_properties(t) if t>=1500 else None
        rows.append(dict(temperature_K=t,legacy_janaf=legacy,production_source=source,
            curated=current,curated_over_nasa=None if current is None else math.exp(current["ln_K"]-source["nasa_ln_K"]),
            curated_over_production_source=None if current is None else math.exp(current["ln_K"]-source["ln_K"]),
            curated_minus_source_reaction=None if current is None else {k:current[k]-v for k,v in source["reaction"].items()},
            frozen_janaf_closure_ln_K=None if old is None else legacy["ln_K"]-old["ln_Kp"]["janaf"],
            frozen_curated_closure_ln_K=None if old is None or current is None else current["ln_K"]-old["ln_Kp"]["curated"]))
    return dict(molecule=NAME,species_id=SPECIES,standard_pressure_Pa=100000,
                source_R_J_mol_K=R,nasa_R_J_mol_K=NASA_R,
                convention="Neutral atoms -> molecule; source minus NASA; explicit balanced H(0), no zero-K polynomial extrapolation",
                source_samples=rows,modern_diagnostic=modern_diagnostic(tables,nasa,rows),
                footprint=footprint(sources["footprint_element"]),
                adoption="none",decision=sources["decision"],physical_accuracy="unknown")


def production_tables(tables: dict) -> dict:
    return tables


def modern_diagnostic(tables: dict, nasa: dict, rows: list) -> dict:
    """Balance same-version ATcT atomization energy; keep thermal variants separate."""
    records = json.loads((DATA / "atct_nodes.json").read_text())["records"]
    d0 = sum(c*records[n]["hf0_kJ_mol"] for n,c in COMPOSITION.items())-records[NAME]["hf0_kJ_mol"]
    samples=[]
    for row in rows:
        t=row["temperature_K"]
        native=row["legacy_janaf"]
        nasa_target=native["nasa_ln_K"]+(1000*d0/R-1000*native["native_nasa_D0_kJ_mol"]/NASA_R)/t
        janaf_target=native["ln_K"]+1000*(d0-native["native_source_D0_kJ_mol"])/(R*t)-native["source_minus_nasa_ln_K_components"]["atomic_thermal"]
        samples.append(dict(temperature_K=t,nasa_thermal_ATcT_ln_K=nasa_target,
            janaf_thermal_ATcT_same_nasa_atoms_ln_K=janaf_target,
            aligned_thermal_ratio=math.exp(janaf_target-nasa_target),
            nasa_ATcT_over_curated=None if row["curated"] is None else math.exp(nasa_target-row["curated"]["ln_K"])))
    return dict(source="ATcT 1.172",D0_kJ_mol=d0,
        status="Pointwise energy/thermal diagnostic; continuous candidate and finite substitution not run",
        covariance="Published species uncertainties are not independent; no reaction uncertainty synthesized",
        samples=samples)



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=json.dumps(review(),indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result,end="")


if __name__ == "__main__":
    main()
