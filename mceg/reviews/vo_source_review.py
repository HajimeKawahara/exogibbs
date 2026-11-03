"""Replay VO energy, normalized partition diagnostics and a saved gas footprint.

No network, new equilibrium solve or production thermochemistry is used.
Modern partition comparisons hold the adopted D0 and NASA atoms fixed; unresolved
energy-zero and isotopic conventions prevent treating them as adopted functions.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import read_table
from gas_source_comparison import BASELINE, ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/vo_source_review/20261103"
REFERENCE = ROOT / "examples/comparisons/data/major_molecule_reference.json"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
TABLES = {"VO": "O-026.txt", "V": "V-005.txt", "O": "O-001.txt"}


def curated_ln_k(temperature: float) -> float:
    """Evaluate the actual hybrid polynomial, not the inherited FastChem VO fit."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated review range")
    record = json.loads(CURATED.read_text())["janaf_reactions"]["O1V1"]
    knots = record["knots_K"]
    i = min(np.searchsorted(knots, temperature, side="right") - 1, len(knots)-2)
    u = (temperature-knots[i])/(knots[i+1]-knots[i])
    polynomial = sum(c*u**power for power, c in enumerate(record["coefficients"][i]))
    background = major.audit.nasa9_g_rt(
        temperature, {"bounds_K": [1500, 4000], "coefficients": record["background_nasa9"]})
    return -polynomial/temperature-background


def translation_log_q(temperature: float, mass_u: float) -> float:
    """Return ln(q_translation*kT/p0) per molecule at 1 bar."""
    kt = major.audit.BOLTZMANN_J_K*temperature
    return (1.5*math.log(2*math.pi*mass_u*major.ATOMIC_MASS_KG*kt/major.audit.PLANCK_J_S**2)
            + math.log(kt/100000))


def footprint() -> dict:
    """Reuse 75 solved endpoints with one V nucleus per VO molecule."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("O1V1")
        element = grid["elements"].tolist().index("V")
        shares = (grid["formula_matrix"][element, column]
                  * np.exp(grid["ln_n"][:, column])/grid["budget"][:, element])
        i = int(np.argmax(shares))
        h_error = max(abs(h+curated_ln_k(t))
                      for t, h in zip(grid["temperature_K"], grid["h"][:, column]))
        if h_error > 3e-11:
            raise ValueError("Archived VO potentials differ from the current hybrid")
        return {
            "source": str(GRID.relative_to(ROOT)), "source_sha256": file_hash(GRID),
            "model": "curated-c2h-v1", "cases": len(shares),
            "species_count": len(grid["species"]),
            "all_cases_converged": bool(np.all(grid["converged"])),
            "maximum_abs_archived_h_error": h_error,
            "maximum_vanadium_share": {
                "case_index": i, "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "mole_fraction": float(np.exp(grid["ln_x"][i, column])),
                "vanadium_fraction": float(shares[i]),
            },
            "interpretation": "Gas-only baseline allocation, not candidate response or optical depth.",
        }


def review() -> dict:
    """Separate native energy, molecular thermal and atomic-reference effects."""
    reference = json.loads(REFERENCE.read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    partitions = json.loads((DATA / "partitions.json").read_text())["rows"]
    tables = {name: read_table(DATA / filename) for name, filename in TABLES.items()}
    for name, table in tables.items():
        for key, row in reference["janaf"][name]["rows"].items():
            # The source stores an infinite 0-K Gibbs column as JSON null.
            for actual, old in zip(table[float(key)], row):
                if old is not None and actual != old:
                    raise ValueError("Fresh JANAF data differ from the adoption source")
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    energy = literature["merriles2020"]["D0_eV"]*major.EV_K
    d0 = {family: major.atomization_kelvin("VO", family, reference)
          for family in ("janaf", "nasa9")}
    rows = []
    for t in (500, 1000, 1500, 2000, 3000, 4000):
        nasa = major.formation_log_kp("VO", t, "nasa9", d0["nasa9"], reference)
        janaf = major.formation_log_kp("VO", t, "janaf", d0["janaf"], reference, "janaf")
        hybrid = major.formation_log_kp("VO", t, "janaf", energy, reference)
        same_d0_nasa = major.formation_log_kp("VO", t, "nasa9", energy, reference)
        atomic = {name: major.thermal_g_rt(name, t, "janaf", reference)
                  - major.thermal_g_rt(name, t, "nasa9", reference) for name in ("V", "O")}
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "O1V1" and r["temperature_K"] == t), None)
        current = curated_ln_k(t) if t >= 1500 else None
        closure = None
        if old is not None:
            closure = {"janaf": janaf-old["ln_Kp"]["janaf"],
                       "nasa": nasa-old["ln_Kp"]["nasa"]}
            if current is not None:
                closure["curated"] = current-old["ln_Kp"]["curated"]
        rows.append({
            "temperature_K": t, "janaf_ln_K": janaf, "nasa_ln_K": nasa,
            "hybrid_node_ln_K": hybrid, "curated_ln_K": current,
            "raw_janaf_over_nasa": math.exp(janaf-nasa),
            "curated_over_nasa": None if current is None else math.exp(current-nasa),
            "same_D0_janaf_over_nasa": math.exp(hybrid-same_d0_nasa),
            "curated_minus_nasa_ln_K_components": {
                "adopted_minus_nasa_D0": (energy-d0["nasa9"])/t,
                "molecular_thermal": hybrid-same_d0_nasa},
            "native_janaf_minus_nasa_atomic_thermal_g_RT": atomic,
            "raw_janaf_minus_hybrid_ln_K_components": {
                "native_minus_adopted_D0": (d0["janaf"]-energy)/t,
                "atomic_thermal": sum(atomic.values())},
            "quoted_D0_only_multiplicative_error": math.exp(.002*major.EV_K/t),
            "baseline_closure_ln_K": closure,
        })
    identity = literature["identity"]
    mass_factor = (identity["isotopologue_mass_u"]/identity["legacy_mass_u"])**1.5
    modern = []
    for row in partitions:
        t = row["temperature_K"]
        if str(float(t)) not in reference["janaf"]["VO"]["rows"]:
            continue
        old_q = math.exp(-major.thermal_g_rt("VO", t, "janaf", reference)
                         - translation_log_q(t, identity["legacy_mass_u"]))
        normalized = row["HyVO"]/identity["nuclear_spin_degeneracy"]
        point = {"temperature_K": t, "janaf_implied_Q": old_q,
                 "hyvo_spin_excluded_Q": normalized,
                 "hyvo_over_janaf_internal_Q": normalized/old_q,
                 "fixed_D0_atoms_zero_assumption_K_ratio": normalized/old_q*mass_factor,
                 "downloaded_vomyt_over_hyvo": row["VOMYT_download"]/row["HyVO"]}
        if "paper_extra_VOMYT" in row:
            point.update({"paper_outer_level_fraction": row["paper_extra_VOMYT"]/row["HyVO_paper"],
                          "paper_adjusted_hyvo_over_paper_vomyt":
                              (row["HyVO_paper"]+row["paper_extra_VOMYT"])/row["VOMYT_paper"],
                          "downloaded_over_paper_vomyt": row["VOMYT_download"]/row["VOMYT_paper"]})
        modern.append(point)
    balanced = {family: sum(major.assigned_h0(atom, family, reference) for atom in ("V", "O"))
                - major.assigned_h0("VO", family, reference) for family in d0}
    return {
        "reaction": "V(g) + O(g) -> VO(g)", "standard_pressure_Pa": 100000,
        "energy_source": "Merriles2020 neutral D0, not the derived cation value",
        "balanced_native_D0_J_mol": balanced,
        "D0_kelvin_used": {**d0, "adopted": energy},
        "source_samples": rows, "partition_samples": modern,
        "molecular_translation_mass_factor": mass_factor,
        "partition_diagnostic": "Same D0 and NASA atoms, 51V16O molecular mass; zero alignment assumed, not established. No isotope-resolved reaction or abundance prediction.",
        "footprint": footprint(), "equilibrium_substitution": "not_run", "production_change": "none",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
