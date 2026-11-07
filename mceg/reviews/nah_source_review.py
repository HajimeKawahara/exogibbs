"""Replay the NaH source audit offline; production and equilibrium stay unchanged."""

import argparse
import bz2
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/nah_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_oh_adoption/20261105/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
R, NASA_R = 8.31446261815324, 8.314510
KB, PLANCK, AMU = 1.380649e-23, 6.62607015e-34, 1.66053906660e-27
C2 = PLANCK * 29979245800 / KB
TEMPERATURES = (300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def nasa(record: dict, t: float) -> dict:
    """Evaluate native NASA9 H, S and Cp without extrapolation."""
    interval = next((r for r in record["intervals"]
                     if r["bounds_K"][0] <= t <= r["bounds_K"][1]), None)
    if interval is None:
        raise ValueError("Temperature outside NASA range")
    a, b, c, d, e, f, g, h, s = interval["coefficients"]
    enthalpy = NASA_R*(-a/t+b*math.log(t)+c*t+d*t*t/2+e*t**3/3+f*t**4/4+g*t**5/5+h)
    entropy = NASA_R*(-a/(2*t*t)-b/t+c*math.log(t)+d*t+e*t*t/2+f*t**3/3+g*t**4/4+s)
    cp = NASA_R*(a/t**2+b/t+c+d*t+e*t*t+f*t**3+g*t**4)
    return {"H_J_mol": enthalpy, "S_J_mol_K": entropy, "Cp_J_mol_K": cp}


def h0(record: dict) -> float:
    return record["hf298_J_mol"]-record["h298_minus_h0_J_mol"]


def translation(t: float, mass: float) -> float:
    """Natural-mass translation at one bar; isotope changes are reported separately."""
    return 1.5*math.log(2*math.pi*mass*AMU*KB*t/PLANCK**2)+math.log(KB*t/100000)


def curated(t: float, inputs: dict) -> tuple:
    """Read the actual inherited fit and its analytic reaction heat capacity."""
    data = json.loads(CURATED.read_text())
    if not data["temperature_range_K"][0] <= t <= data["temperature_range_K"][1]:
        raise ValueError("Temperature outside curated range")
    sid = inputs["species_id"]
    for group in ("nasa9_reactions", "janaf_reactions", "partition_reactions", "logk_energy_shifts_K"):
        if sid in data[group]:
            raise ValueError("Reviewed inherited species now has an override")
    lines = FASTCHEM.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(sid+" "))
    a, b, c, d, e = map(float, lines[i+1].split())
    return a/t+b*math.log(t)+c+d*t+e*t*t, R*(b+2*d*t+6*e*t*t)


def footprint(inputs: dict) -> dict:
    """Audit the saved current allocation, never a finite substitution response."""
    with np.load(GRID) as grid:
        j = grid["species"].tolist().index(inputs["species_id"])
        e = grid["elements"].tolist().index(inputs["element"])
        n = np.exp(grid["ln_n"])
        share = n[:, j]/grid["budget"][:, e]
        i = int(np.argmax(share))
        return {
            "model": "curated-oh-v1", "cases": len(n), "species": n.shape[1],
            "all_converged": bool(np.all(grid["converged"])),
            "maximum_abs_h_error": float(max(abs(h+curated(t, inputs)[0]) for t, h in
                zip(grid["temperature_K"], grid["h"][:, j]))),
            "maximum_element_fraction": float(share[i]), "element": inputs["element"],
            "case_index": i, "temperature_K": float(grid["temperature_K"][i]),
            "pressure_bar": float(grid["pressure_bar"][i]), "C_over_O": float(grid["C_over_O"][i]),
            "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][i, j])),
            "maximum_relative_element_residual": float(np.max(np.abs(
                n @ grid["formula_matrix"][e]/grid["budget"][:, e]-1))),
        }


def moments(states: np.ndarray, t: float, spin: float) -> dict:
    """Remove only the common nuclear-spin product; preserve explicit electronic states."""
    energy = states[:, 0]*C2
    weight = states[:, 1]/spin*np.exp(-energy/t)
    q = weight.sum()
    mean = weight @ energy/q
    variance = weight @ (energy-mean)**2/q
    return {"Q": float(q), "H_increment_J_mol": float(R*(2.5*t+mean)),
            "Cp_J_mol_K": float(R*(2.5+variance/t**2))}


def extract(raw: Path, output: Path) -> dict:
    """Keep every distributed state; reject unreviewed upstream revisions."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    sources = json.loads((DATA / "sources.json").read_text())
    for ext in ("states.bz2", "pf", "def.json"):
        filename = inputs["molecule"]+"."+ext
        if digest(raw / filename) != sources["downloads"][filename]["sha256"]:
            raise ValueError("Unreviewed original download: "+filename)
    with bz2.open(raw / (inputs["molecule"]+".states.bz2"), "rt") as stream:
        rows = [line.split() for line in stream]
    label_column, v_column = inputs["state_columns"]
    labels = sorted(set(r[label_column] for r in rows))
    states = np.array([[float(r[i]) for i in (1, 2, 3, v_column)]
                       + [labels.index(r[label_column])] for r in rows])
    pf = np.loadtxt(raw / (inputs["molecule"]+".pf"))
    definition = json.loads((raw / (inputs["molecule"]+".def.json")).read_text())["dataset"]
    inventory = {
        "columns": ["energy_cm-1", "gtot", "J", "v", "electronic_label_index"],
        "labels": labels, "number_of_states": len(states),
        "energy_bounds_cm-1": [float(states[:, 0].min()), float(states[:, 0].max())],
        "maximum_J": float(states[:, 2].max()),
        "gtot_over_2Jplus1": np.unique(states[:, 1]/(2*states[:, 2]+1)).tolist(),
        "by_label": {label: {"states": int(np.sum(states[:, 4] == i)),
            "minimum_energy_cm-1": float(states[states[:, 4] == i, 0].min()),
            "maximum_v": float(states[states[:, 4] == i, 3].max())}
            for i, label in enumerate(labels)},
        "metadata_maximum_temperature_K": definition["max_temperature"],
        "metadata_version": definition["version"],
        "published_pf_first_last_K": pf[[0, -1], 0].tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", states=states, published_pf=pf)
    (output / "state_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    return inventory


def literature_diagnostics(inputs: dict, rows: list, d_janaf: float) -> dict:
    """Keep paper values distinct from current downloads and adopted reactions."""
    literature = json.loads((DATA / "literature.json").read_text())
    values = literature["paper_Q_including_nuclear_spin"]
    return {"BC2016_D0_J_mol": inputs["BC_D0_eV"]*96485.33212331002,
            "paper_quasibound_over_bound_only": {t: q["with_quasibound"]/q["bound_only"] for t, q in values.items()},
            "Walji_De_cm-1": literature["Walji2015"]["De_cm-1"],
            "Walji_D0_cm-1": None,
            "scope": "De is not D0. Current state distribution matches the paper's quasi-bound-inclusive column."}


def review() -> dict:
    """Separate reaction energy, molecular calorics and atomic-reference effects."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    molecule, atoms = inputs["molecule"], inputs["composition"]
    records, tables = inputs["nasa9"], inputs["janaf"]
    d_nasa = sum(n*h0(records[a]) for a, n in atoms.items())-h0(records[molecule])
    def jh0(name):
        return 1000*(tables[name]["298.15"][4]+tables[name]["0.0"][3])
    d_janaf = sum(n*jh0(a) for a, n in atoms.items())-jh0(molecule)
    states = pf = inventory = None
    if (DATA / "states.npz").exists():
        with np.load(DATA / "states.npz") as archive:
            states, pf = archive["states"], archive["published_pf"]
        inventory = json.loads((DATA / "state_inventory.json").read_text())
    rows = []
    for t in TEMPERATURES:
        props = {name: nasa(record, t) for name, record in records.items()}
        thermal = {name: (p["H_J_mol"]-h0(records[name]))/(NASA_R*t)-p["S_J_mol_K"]/NASA_R
                   for name, p in props.items()}
        nasa_ln_k = sum(n*thermal[a] for a, n in atoms.items())-thermal[molecule]+d_nasa/(NASA_R*t)
        q_nasa = math.exp(-thermal[molecule]-translation(t, records[molecule]["mass_u"]))
        row = {"temperature_K": t, "nasa_ln_K": nasa_ln_k, "Q_nasa": q_nasa,
               "nasa_common_R_Cp_J_mol_K": props[molecule]["Cp_J_mol_K"]*R/NASA_R}
        if t >= 1500:
            ln_k, cp = curated(t, inputs)
            row.update(curated_ln_K=ln_k, curated_over_nasa_K=math.exp(ln_k-nasa_ln_k),
                       curated_reaction_Cp_J_mol_K=cp,
                       nasa_reaction_Cp_J_mol_K=(props[molecule]["Cp_J_mol_K"]-
                           sum(n*props[a]["Cp_J_mol_K"] for a, n in atoms.items()))*R/NASA_R)
        if all(str(float(t)) in table for table in tables.values()):
            jt = {name: 1000*(table[str(float(t))][3]-table["0.0"][3])/(R*t)
                         -table[str(float(t))][1]/R for name, table in tables.items()}
            energy_shift = (d_janaf/R-d_nasa/NASA_R)/t
            molecular_shift = thermal[molecule]-jt[molecule]
            atomic_shift = sum(n*(jt[a]-thermal[a]) for a, n in atoms.items())
            row["janaf"] = {"ln_K": nasa_ln_k+energy_shift+molecular_shift+atomic_shift,
                            "energy_ln_K_shift": energy_shift,
                            "molecular_thermal_ln_K_shift": molecular_shift,
                            "atomic_thermal_ln_K_shift": atomic_shift,
                            "Cp_J_mol_K": tables[molecule][str(float(t))][0]}
        if states is not None:
            moment = moments(states, t, inputs["nuclear_spin_product"])
            h = moment["H_increment_J_mol"]
            entropy = R*(math.log(moment["Q"])+translation(t, records[molecule]["mass_u"]))+h/t
            pop = states[:, 1]*np.exp(-C2*states[:, 0]/t)
            nearest = int(np.argmin(abs(pf[:, 0]-t)))
            pt, pq = pf[nearest]
            published = None if t > pf[-1, 0] else {
                "temperature_K": float(pt), "Q_including_nuclear_spin": float(pq),
                "states_over_published_at_tabulated_T": moments(states, pt, inputs["nuclear_spin_product"])["Q"]*inputs["nuclear_spin_product"]/pq}
            moment.update(S_J_mol_K=entropy, Q_over_nasa=moment["Q"]/q_nasa,
                isotope_mass_K_proxy=moment["Q"]/q_nasa*(inputs["isotope_mass_u"]/records[molecule]["mass_u"])**1.5,
                delta_H_J_mol=h-(props[molecule]["H_J_mol"]-h0(records[molecule]))*R/NASA_R,
                delta_S_J_mol_K=entropy-props[molecule]["S_J_mol_K"]*R/NASA_R,
                delta_Cp_J_mol_K=moment["Cp_J_mol_K"]-row["nasa_common_R_Cp_J_mol_K"],
                top_J_10_percent_Q_fraction=float(pop[states[:, 2] >= .9*states[:, 2].max()].sum()/pop.sum()),
                electronic_Q_fractions={label: float(pop[states[:, 4] == i].sum()/pop.sum()) for i, label in enumerate(inventory["labels"])},
                published_pf=published)
            if str(t) in inputs.get("BC_Q", {}):
                moment["Q_over_BC2016"] = moment["Q"]/inputs["BC_Q"][str(t)]
            row["finite_states"] = moment
        if str(t) in inputs.get("BC_Q", {}):
            masses = [records[a]["mass_u"] for a in atoms]
            mu = masses[0]*masses[1]/sum(masses)*AMU
            atomic_q = math.prod(inputs["BC_atomic_Q"][a][str(t)] for a in atoms)
            bc_ln_k = (math.log(inputs["BC_Q"][str(t)]/atomic_q)
                       -1.5*math.log(2*math.pi*mu*KB*t/PLANCK**2)
                       +math.log(100000/(KB*t))+inputs["BC_D0_eV"]*1.602176634e-19/(KB*t))
            row["BC2016_raw_ln_K"] = bc_ln_k
            if t >= 1500:
                row["curated_over_BC2016_raw_K"] = math.exp(row["curated_ln_K"]-bc_ln_k)
        rows.append(row)
    return {"molecule": molecule, "reaction": inputs["reaction"], "standard_pressure_Pa": 100000,
            "balanced_D0_J_mol": {"nasa9": d_nasa, "janaf": d_janaf},
            "source_samples": rows, "footprint": footprint(inputs),
            "literature_diagnostics": literature_diagnostics(inputs, rows, d_janaf),
            "production_change": "none", "equilibrium_substitution": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract, args.packet) if args.extract else review()
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
