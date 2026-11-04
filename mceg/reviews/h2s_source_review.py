"""Audit H2S energy and thermal sources without changing the production model.

The default command replays the compact offline state packet. --extract accepts
original AYT2 downloads and a pinned HAPI Python source to rebuild that packet.
"""

import argparse
import ast
import bz2
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/h2s_source_review/20261104"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK*29979245800/KB
ISOTOPE_MASS = 33.98772106


def extract(raw: Path, hapi: Path, output: Path) -> dict:
    """Extract printed state fields and only the H2(32S) TIPS tables."""
    with bz2.open(raw / "AYT2.states.bz2", "rt") as stream:
        rows = [line.split() for line in stream]
    states = np.array([[float(row[i]) for i in (1, 2, 3, 13)] for row in rows])
    arrays = {"states": states}
    # Interpret only literal assignments, without importing/executing HAPI.
    env = {}
    targets = {f"TIPS_{y}_{suffix}" for y in (2017, 2021, 2025)
               for suffix in ("ISOT", "ISOT_HASH", "ISOQ_HASH")}
    def value(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Subscript):
            return value(node.value)[value(node.slice)]
        if isinstance(node, ast.Tuple):
            return tuple(value(item) for item in node.elts)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "float64" and len(node.args) == 1 and not node.keywords):
            return np.array(ast.literal_eval(node.args[0]))
        return ast.literal_eval(node)

    for node in ast.parse(hapi.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else (target.value.id
               if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) else "")
        if name not in targets | {"M", "I"}:
            continue
        if name in ("M", "I") and not isinstance(node.value, ast.Constant):
            continue
        if isinstance(target, ast.Name):
            env[name] = value(node.value)
        else:
            env[name][value(target.slice)] = value(node.value)
    for year in (2017, 2021, 2025):
        arrays[f"tips{year}"] = np.column_stack((env[f"TIPS_{year}_ISOT_HASH"][(31, 1)],
                                                env[f"TIPS_{year}_ISOQ_HASH"][(31, 1)]))
    arrays["published_pf"] = np.loadtxt(raw / "AYT2.pf")
    arrays["published_cp"] = np.loadtxt(raw / "AYT2.cp")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", **arrays)
    weight = states[:, 1]/(2*states[:, 2]+1)
    anomalies = [dict(state_id=int(rows[i][0]), energy_cm1=states[i, 0],
                      gtot=states[i, 1], J=states[i, 2], symmetry=rows[i][6],
                      source_type=rows[i][12], original_calculated_energy_cm1=states[i, 3])
                 for i in np.flatnonzero(~np.isin(weight, (1, 3)))]
    result = {
        "columns": ["energy_cm-1", "gtot", "J", "original_calculated_energy_cm-1"],
        "states": len(states), "maximum_energy_cm-1": float(states[:, 0].max()),
        "maximum_J": float(states[:, 2].max()),
        "ground_state": states[0].tolist(),
        "lowest_ortho_energy_cm-1": float(states[weight == 3, 0].min()),
        "weight_J_inconsistencies": anomalies,
        "states_with_changed_energy": int(np.sum(states[:, 0] != states[:, 3])),
        "source_type_counts": {key: sum(row[12] == key for row in rows)
                               for key in sorted({row[12] for row in rows})},
        "original_sha256": {name: file_hash(raw/name) for name in
                            ("AYT2.states.bz2", "AYT2.pf", "AYT2.cp")},
        "hapi_sha256": file_hash(hapi),
        "table_bounds_K": {name: a[[0, -1], 0].tolist() for name, a in arrays.items()
                           if name != "states"},
    }
    (output / "state_inventory.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def log_translation(t: float, mass: float) -> float:
    """Return the ideal-gas translation factor at one bar."""
    return (1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*KB*t/PLANCK**2)
            + math.log(KB*t/100000))


def moments(states: np.ndarray, t: float, energy_column: int = 0) -> dict:
    """Calculate analytic thermal moments of all distributed states."""
    e = states[:, energy_column]*C2
    p = states[:, 1]*np.exp(-e/t)
    q = p.sum()
    mean = p@e/q
    return {"Q_spin_included": float(q), "mean_energy_K": float(mean),
            "variance_energy_K2": float(p@(e-mean)**2/q),
            "ortho_over_para": float(p[states[:, 1] > 2*states[:, 2]+1].sum()
                                     /p[states[:, 1] == 2*states[:, 2]+1].sum()),
            "J_ge_35_fraction": float(p[states[:, 2] >= 35].sum()/q),
            "J_eq_40_fraction": float(p[states[:, 2] == 40].sum()/q),
            "energy_ge_20000_fraction": float(p[states[:, 0] >= 20000].sum()/q)}


def properties(m: dict, t: float, mass: float) -> dict:
    """Remove nuclear spin product four once; allowed states encode symmetry."""
    q = m["Q_spin_included"]/4
    h = 2.5*t+m["mean_energy_K"]
    s = math.log(q)+log_translation(t, mass)+h/t
    return {"Q_spin_free": q, "H_increment_J_mol": major.R*h,
            "S_J_mol_K": major.R*s, "Cp_J_mol_K": major.R*(2.5+m["variance_energy_K2"]/t**2),
            "G_minus_H0_J_mol": major.R*(h-t*s)}


def curated_ln_k(t: float) -> float:
    """Evaluate the immutable H2S column inherited by curated-h2o-v1."""
    data = json.loads(CURATED.read_text())
    lower, upper = data["temperature_range_K"]
    if not lower <= t <= upper:
        raise ValueError("Temperature outside curated review range")
    return -major.audit.nasa9_g_rt(t, {"coefficients": data["nasa9_reactions"]["H2S1"],
                                     "bounds_K": [lower, upper]})


def footprint() -> dict:
    """Report sulfur budgets in saved current-model endpoints, not a response."""
    with np.load(GRID) as grid:
        species = grid["species"].tolist()
        e = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        shares = n*grid["formula_matrix"][e]/grid["budget"][:, e, None]
        def point(j):
            i = int(np.argmax(shares[:, j]))
            return {"case_index": i, "maximum_sulfur_fraction": float(shares[i, j]),
                    "mole_fraction": float(np.exp(grid["ln_x"][i, j])),
                    "temperature_K": float(grid["temperature_K"][i]),
                    "pressure_bar": float(grid["pressure_bar"][i]),
                    "C_over_O": float(grid["C_over_O"][i])}
        selected = ("H2S1", "S1", "H1S1", "S2", "O1S1", "O2S1", "C1O1S1", "C1S1")
        return {"model": "curated-h2o-v1", "cases": len(n), "species": len(species),
                "all_converged": bool(np.all(grid["converged"])),
                "maximum_relative_sulfur_residual": float(np.max(abs(shares.sum(axis=1)-1))),
                "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                      zip(grid["temperature_K"], grid["h"][:, species.index("H2S1")])),
                "sulfur_reservoirs": {name: point(species.index(name)) for name in selected},
                "largest_sulfur_reservoirs": {species[j]: point(j) for j in
                          np.argsort(shares.max(axis=0))[-10:][::-1]}}


def review() -> dict:
    """Separate balanced energy, thermal completeness, source ancestry and scope."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    record = reference["nasa9_records"]["H2S"]
    energy = {s: major.atomization_kelvin("H2S", s, reference)
              for s in ("atct", "nasa9", "janaf")}
    atct = reference["atct"]["records"]
    first_d0 = 1000*(atct["H"]["hf0_kJ_mol"]+atct["SH"]["hf0_kJ_mol"]
                    - atct["H2S"]["hf0_kJ_mol"])
    with np.load(DATA / "states.npz") as a:
        tables = {name: a[name] for name in a.files}
    states = tables["states"]
    rows = []
    for t in TEMPERATURES:
        g = major.thermal_g_rt("H2S", t, "nasa9", reference)
        h, s, cp = nasa_properties(record, t, major.NASA_R)
        h = (h-major.assigned_h0("H2S", "nasa9", reference))/major.NASA_R*major.R
        nasa = {"Q_spin_free": math.exp(-g-log_translation(t, record["mass_u"])),
                "H_increment_J_mol": h, "S_J_mol_K": s/major.NASA_R*major.R,
                "Cp_J_mol_K": cp/major.NASA_R*major.R, "G_minus_H0_J_mol": major.R*t*g}
        m = moments(states, t)
        p = properties(m, t, record["mass_u"])
        legacy = moments(states, t, 3)
        e = states[:, 0]*C2
        population = states[:, 1]*np.exp(-e/t)
        # Do not silently fix the inconsistent J/weight row. The alternative
        # is a sensitivity, not an asserted correction of its assignment.
        proposed_weight = np.where(np.isin(states[:, 1]/(2*states[:, 2]+1), (1, 3)),
                                   states[:, 1], 3*(2*states[:, 2]+1))
        anomalous_delta = float(((proposed_weight-states[:, 1])*np.exp(-e/t)).sum()/population.sum())
        sources = {}
        for name, table in tables.items():
            if name == "states" or not table[0, 0] <= t <= table[-1, 0]:
                continue
            value = float(np.interp(t, table[:, 0], table[:, 1]))
            sources[name] = {"value": value, "state_over_table": (
                p["Cp_J_mol_K"] if name == "published_cp" else m["Q_spin_included"])/value}
        janaf = reference["janaf"]["H2S"]["rows"].get(str(float(t)))
        atom_q = {a: math.exp(-major.thermal_g_rt(a, t, "nasa9", reference)
                    - log_translation(t, reference["nasa9_records"][a]["mass_u"]))
                  for a in ("H", "S")}
        native = major.formation_log_kp("H2S", t, "nasa9", energy["nasa9"], reference)
        hybrid = major.formation_log_kp("H2S", t, "nasa9", energy["atct"], reference)
        ratio = p["Q_spin_free"]/nasa["Q_spin_free"]
        rows.append({"temperature_K": t, "nasa": nasa, "AYT2": {**m, **p},
                     "delta_AYT2_minus_nasa": {k: p[k]-nasa[k] for k in p if k != "Q_spin_free"},
                     "same_D0_atoms_natural_mass_K_ratio": ratio,
                     "same_D0_atoms_isotope_mass_K_proxy": ratio*(ISOTOPE_MASS/record["mass_u"])**1.5,
                     "nasa_ln_K": native, "reconstructed_hybrid_ln_K": hybrid,
                     "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
                     "curated_over_nasa": math.exp(hybrid-native),
                     "AYT2_same_anchor_ln_K": hybrid+math.log(ratio),
                     "original_calculated_Q": legacy["Q_spin_included"],
                     "current_over_original_calculated_Q": m["Q_spin_included"]/legacy["Q_spin_included"],
                     "displayed_J_weight_sensitivity_relative_Q": anomalous_delta,
                     "above_first_dissociation_fraction": float(population[e > first_d0/major.R].sum()/population.sum()),
                     "tables": sources, "nasa_atomic_Q": atom_q,
                     "S_three_ground_fine_levels_Q": 5+3*math.exp(-396.055*C2/t)+math.exp(-573.640*C2/t),
                     "S_five_lowest_levels_Q": sum(weight*math.exp(-level*C2/t) for level, weight in
                         ((0, 5), (396.055, 3), (573.640, 1), (9238.609, 5), (22179.954, 1))),
                     "janaf": None if janaf is None else {
                         "Cp_J_mol_K": janaf[0],
                         "same_D0_atoms_K_ratio": math.exp(g-major.thermal_g_rt("H2S", t, "janaf", reference)),
                         "native_ln_K": major.formation_log_kp("H2S", t, "janaf", energy["janaf"], reference, "janaf"),
                         "atomic_thermal_shift_ln_K": sum(n*(major.thermal_g_rt(a, t, "janaf", reference)
                              -major.thermal_g_rt(a, t, "nasa9", reference)) for a, n in (("H", 2), ("S", 1)))}})
    return {"reaction": "2 H(g) + S(g) -> H2S(g)", "standard_pressure_Pa": 100000,
            "atct_version": reference["atct"]["version"],
            "balanced_D0_J_mol": {s: e*(major.NASA_R if s == "nasa9" else major.R) for s, e in energy.items()},
            "balanced_D0_over_k_K": energy,
            "adopted_minus_nasa_D0_over_k_K": energy["atct"]-energy["nasa9"],
            "first_dissociation_H_plus_SH_J_mol": first_d0,
            "natural_mass_u": record["mass_u"], "representative_isotope_mass_u": ISOTOPE_MASS,
            "source_samples": rows, "footprint": footprint(),
            "production_change": "none", "equilibrium_substitution": "not_run",
            "interpretation": "Retain the current NASA thermal plus ATcT energy construction. AYT2 and TIPS agree numerically but share a truncated state distribution; agreement cannot validate completeness through 4000 K. Natural-mass comparisons use 1H2-32S levels as a proxy; isotope-mass values are not balanced isotope reactions. Atomic NASA functions and native NASA R are preserved; reported calorics use modern R. The one inconsistent J/gtot row is retained, with a separately labelled sensitivity."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--hapi", type=Path)
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract and not args.hapi:
        parser.error("--extract requires --hapi")
    result = extract(args.extract, args.hapi, args.packet) if args.extract else review()
    serialized = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
