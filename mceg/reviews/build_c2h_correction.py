"""Build the opt-in C2H correction from the inspected, offline source records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from gas_source_comparison import ROOT, file_hash, load_inputs


REVIEW = ROOT / "tests/data/validation/c2h_source_review/20261102"
OUTPUT = ROOT / "src/exogibbs/data/curated/c2h_v1.json"
NASA_R = 8.314510
R = 8.31446261815324


def build_data() -> dict:
    """Retain NASA thermal functions and replace only balanced D0, in kelvin."""
    inputs = load_inputs()
    source = json.loads((REVIEW / "sources.json").read_text())
    headers = {}
    for block in (REVIEW / "nasa_selected_records.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].split()[0]
        coefficients = [float(v.replace("D", "E"))
                        for v in re.findall(r"[+-]?\d+\.\d+D[+-]\d+", block)]
        expected = [v for interval in inputs["nasa9_records"][name]["intervals"]
                    for v in interval["coefficients"]]
        if coefficients != expected:
            raise ValueError(f"Inspected NASA record differs: {name}")
        headers[name] = {"hf298_J_mol": float(lines[1].split()[-1]),
                         "h298_minus_h0_J_mol": float(lines[2].split()[-1])}
    counts = {"C2H": 1, "C": -2, "H": -1}
    raw = [sum(n * inputs["nasa9_records"][s]["intervals"][1]["coefficients"][i]
               for s, n in counts.items()) for i in range(9)]
    nasa_d0_K = -sum(n * (headers[s]["hf298_J_mol"] - headers[s]["h298_minus_h0_J_mol"])
                       for s, n in counts.items()) / NASA_R
    atct = source["atct_1_172"]
    atct_d0 = 2 * atct["C"]["hf0_kJ_mol"] + atct["H"]["hf0_kJ_mol"] - atct["CCH"]["hf0_kJ_mol"]
    corrected = raw.copy()
    corrected[7] += nasa_d0_K - atct_d0 * 1000 / R
    paths = [REVIEW / "sources.json", REVIEW / "nasa_selected_records.txt",
             ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json"]
    return {
        "id": "curated-c2h-v1", "species": "C2H1", "reaction": "2 C(g) + H(g) -> CCH(g)",
        "base_dataset": "curated-gas-v1",
        "base_data_sha256": file_hash(ROOT / "src/exogibbs/data/curated/gas_v1.json"),
        "temperature_range_K": [1500., 4000.], "standard_pressure_bar": 1.,
        "nasa9_reaction_coefficients": corrected,
        "raw_nasa9_reaction_coefficients": raw,
        "nasa_headers": headers, "nasa_gas_constant_J_mol_K": NASA_R,
        "gas_constant_J_mol_K": R, "nasa_atomization_over_R_K": nasa_d0_K,
        "atct_atomization_0K_kJ_mol": atct_d0,
        "energy_shift_K": atct_d0 * 1000 / R - nasa_d0_K,
        "atct_1_172": {k: v for k, v in atct.items() if k != "use"},
        "decision": {"name": "C2H", "status": "prefer_conditionally", "source_group": "molecular",
                     "thermal_source": "nasa9", "energy_source": "atct",
                     "reason": "Replace the 1967 JANAF estimate with identified NASA CCH thermal functions and balanced ATcT 1.172 D0.",
                     "condition": "1500-4000 K; high-temperature physical uncertainty remains unquantified."},
        "sources": {"nasa": "https://ntrs.nasa.gov/citations/20020085330",
                    "nasa_pages": "C p74, C2H p88, H p126; Appendix D",
                    "atct": "https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=103",
                    "review": "documents/mceg/c2h_source_review.rst"},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        "interpretation": "Opt-in, single-species correction of curated-gas-v1. Correlated energy anchors do not quantify thermal or whole-mixture physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False) + "\n")
    print(file_hash(args.output))
