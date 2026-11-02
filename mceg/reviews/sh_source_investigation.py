"""Replay the scoped SH source investigation from archived inputs, offline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/sh_source_investigation/20261102"
sys.path.insert(0, str(ROOT / "mceg/reviews"))
sys.path.insert(0, str(ROOT / "examples/comparisons"))

import gas_source_comparison as baseline
import select_major_molecules as adoption


def janaf_rows(filename: str) -> dict:
    """Read exact official table nodes; never interpolate or extrapolate."""
    rows = {}
    for line in (DATA / filename).read_text().splitlines()[2:]:
        cells = line.split()
        if len(cells) == 8:
            try:
                values = [float(c.replace("INFINITE", "inf")) for c in cells]
            except ValueError:
                continue
            rows[values[0]] = values[1:]
    return rows


def nasa_hs(record: dict, temperature: float) -> tuple[float, float]:
    """Return H/R and S/R using the same interval convention as the baseline."""
    interval = next(i for i in record["intervals"]
                    if i["bounds_K"][0] <= temperature <= i["bounds_K"][1])
    a, b, c, d, e, f, g, h, s = interval["coefficients"]
    t = temperature
    return (-a/t + b*math.log(t) + c*t + d*t*t/2 + e*t**3/3
            + f*t**4/4 + g*t**5/5 + h,
            -a/(2*t*t) - b/t + c*math.log(t) + d*t + e*t*t/2
            + f*t**3/3 + g*t**4/4 + s)


def investigate() -> dict:
    """Separate the formation-column error, source difference and energy update."""
    inputs = baseline.load_inputs()
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["sources"].values():
        if "archive" in source:
            path = ROOT / source["archive"]
            if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
    major = json.loads(adoption.REFERENCE_PATH.read_text())
    official = {name: janaf_rows(filename) for name, filename in
                {"SH": "H-046.txt", "H": "H-001.txt", "S": "S-006.txt"}.items()}
    nasa_energy = adoption.atomization_kelvin("SH", "nasa9", major)
    atct_energy = adoption.atomization_kelvin("SH", "atct", major)
    r = inputs["gas_constant_J_mol_K"]
    samples = []
    for t in inputs["temperatures_K"]:
        delta_h0, delta_thermal_h, delta_s, janaf, nasa, raw = 0., 0., 0., 0., 0., 0.
        for name, identifier, sign in [("SH", "H1S1", 1), ("H", "H1", -1), ("S", "S1", -1)]:
            row, row0 = official[name][t], official[name][298.15]
            frozen = inputs["janaf_records"][identifier]
            sample = next(s for s in frozen["samples"] if s["temperature_K"] == t)
            if (row[1], row[3], row[5], row0[4]) != (
                    sample["entropy_J_mol_K"], sample["enthalpy_increment_kJ_mol"],
                    sample["formation_gibbs_kJ_mol"], frozen["hf298_kJ_mol"]):
                raise ValueError(f"Official and frozen JANAF differ: {name} at {t}")
            nh, ns = nasa_hs(inputs["nasa9_records"][name], t)
            nh0, _ = nasa_hs(inputs["nasa9_records"][name], 298.15)
            delta_h0 += sign * (1000*row0[4]/r - nh0)
            delta_thermal_h += sign * (1000*row[3]/r - (nh-nh0))
            delta_s += sign * (row[1]/r - ns)
            janaf -= sign * baseline.janaf_g(frozen, t, r)
            nasa -= sign * baseline.nasa_g(inputs["nasa9_records"][name], t)
            raw -= sign * baseline.janaf_g(frozen, t, r, True)
        shift = (atct_energy - nasa_energy) / t
        samples.append({"temperature_K": t,
                        "ln_K_janaf_hs": janaf, "ln_K_nasa": nasa,
                        "janaf_over_nasa": math.exp(janaf-nasa),
                        "raw_minus_reconstructed_dex": (raw-janaf)/math.log(10),
                        "delta_reaction_H298_over_R_K": delta_h0,
                        "delta_reaction_thermal_H_over_R_K": delta_thermal_h,
                        "delta_reaction_S_over_R": delta_s,
                        "reconstructed_delta_ln_K": -(delta_h0+delta_thermal_h)/t+delta_s,
                        "atct_energy_shift_ln_K": shift,
                        "predicted_curated_over_nasa": math.exp(shift) if t >= 1500 else None})
    return {"schema_version": "1", "species_id": "H1S1", "reaction": "H(g) + S(g) -> SH(g)",
            "standard_pressure_Pa": 100000, "energy_nasa_D0_over_R_K": nasa_energy,
            "energy_atct_D0_over_R_K": atct_energy,
            "energy_atct_D0_kJ_mol": atct_energy*r/1000,
            "samples": samples,
            "limitations": ["NASA/JANAF delta terms use the baseline common R scale, not a new energy evaluation.",
                            "Predicted curated ratio uses the existing balanced ATcT energy shift; no second correction is applied.",
                            "No abundance solves or physical error bounds; historical transfer arithmetic remains unidentified."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = investigate()
    if args.check:
        expected = json.loads((DATA / "comparison.json").read_text())
        baseline.check_equal(report, expected)
        print("SH: 12 fresh JANAF nodes match baseline; 4 source comparisons replayed")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
