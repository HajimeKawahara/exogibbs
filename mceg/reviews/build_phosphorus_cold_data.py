"""Extract explicitly selected NASA9 gas records from a pinned local source copy.

This creates an offline research fixture, not a packaged thermochemical preset.
Original polynomial coefficients and their source intervals are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VULCAN_REVISION = "aa447277372f964b9c9c4608dd1073fd1bcc5236"
SOURCE_HASHES = {
    "nasa9_2002_E.txt": "1ec5fe226af698ef6c900537d30c2a640b31f83483d3f6f51c47c813f8d009ee",
    "new_nasa9.TXT": "b4ac2ec3cbeb46d461722a360c0158d19f480766133aaa6e0c1847e9c76b978c",
}
BASE = "H He O P H2 OH H2O HO2 H2O2 O2 O3 PH PH2 PH3 PO PO2 P2 P4 P4O6 P4O10".split()
FULL_P = "HPO HPO3 H3PO3_phosphorous H3PO3_phosphonic P2O3 P2O4 P2O5 P3 P3O6 P4O7 P4O8 P4O9".split()
CNS = "C N S CO CO2 CH CH2 CH3 CH4 HCO H2CO CH2OH CH3OH C2H2 C2H4 C2H6 CN HCN HNC N2 NH NH2 NH3 NO NO2 N2O HNCO NCO H2S SH S2 S3 S4 S8 SO SO2 SO3 COS CS CS2 CP CHP PN PS".split()
ALIASES = {"H2CO": "HCHO,formaldehy", "C2H2": "C2H2,acetylene"}
BURCAT = {"HPO2": ("HPO2", 0), "H3PO4": ("H3PO4", 0), "HPO3": ("HPO3", 0),
          "H3PO3_phosphorous": ("H3PO3", 0), "H3PO3_phosphonic": ("H3PO3", 1),
          "CHP": ("CHP", 0)}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_records(path: Path, source_id: str, names: set) -> dict:
    """Read the fixed-width NASA9 layout, keeping duplicate names distinct."""
    lines = path.read_text().splitlines()
    records = {}
    for i, header in enumerate(lines[:-1]):
        metadata = lines[i + 1]
        if not header or not header[0].isalpha() or not metadata[:2].strip().isdigit():
            continue
        if header.split()[0] not in names:
            continue
        if len(metadata) < 80 or metadata[51:52] != "0":
            continue
        composition = {}
        for k in range(10, 50, 8):
            element = metadata[k:k + 2].strip()
            if element:
                amount = float(metadata[k + 2:k + 8])
                composition[element.capitalize()] = int(amount) if amount.is_integer() else amount
        intervals = []
        for j in range(int(metadata[:2])):
            interval, first, second = lines[i + 2 + 3*j:i + 5 + 3*j]
            coefficients = [float(first[k:k + 16]) for k in range(0, 80, 16)]
            coefficients += [float(second[k:k + 16]) for k in (0, 16, 48, 64)]
            intervals.append({"bounds_K": [float(interval[:11]), float(interval[11:22])],
                              "coefficients": coefficients})
        record = {"composition": composition, "phase": "gas", "intervals": intervals,
                  "source_id": source_id, "source_line": i + 1, "reference": header.strip(),
                  "hf298_J_mol": float(metadata[65:80]),
                  "raw_record_sha256": digest(("\n".join(lines[i:i + 2 + 3*len(intervals)]) + "\n").encode())}
        records.setdefault(header.split()[0], []).append(record)
    return records


def build_fixture(source_directory: Path) -> dict:
    """Select a common gas-only 200--1500 K research network."""
    filenames = {"cea2002_vulcan": "nasa9_2002_E.txt", "burcat2016_vulcan": "new_nasa9.TXT"}
    for filename, expected in SOURCE_HASHES.items():
        if digest((source_directory / filename).read_bytes()) != expected:
            raise ValueError(f"Source hash mismatch: {filename}")
    names = BASE + ["HPO2", "H3PO4"] + FULL_P + CNS
    cea_names = {ALIASES.get(name, name) for name in names if name not in BURCAT}
    parsed = {key: read_records(source_directory / filename, key,
                               cea_names if key == "cea2002_vulcan" else {x[0] for x in BURCAT.values()})
              for key, filename in filenames.items()}
    selected = {}
    for name in names:
        source, (original, occurrence) = (("burcat2016_vulcan", BURCAT[name]) if name in BURCAT
                                          else ("cea2002_vulcan", (ALIASES.get(name, name), 0)))
        record = parsed[source][original][occurrence]
        if any(not isinstance(n, int) for n in record["composition"].values()):
            raise ValueError(f"Noninteger composition for selected species {name}")
        # Preserve all original ranges, but require continuous coverage of the study interval.
        covering = [r for r in record["intervals"] if r["bounds_K"][1] > 200 and r["bounds_K"][0] < 1500]
        if covering[0]["bounds_K"][0] > 200 or covering[-1]["bounds_K"][1] < 1500:
            raise ValueError(f"Insufficient source range for {name}")
        if any(a["bounds_K"][1] != b["bounds_K"][0] for a, b in zip(covering, covering[1:])):
            raise ValueError(f"Gapped source ranges for {name}")
        selected[name] = record
    sources = {key: {"path": f"thermo/NASA9/{filename}",
                     "url": f"https://github.com/shami-EEG/VULCAN/blob/{VULCAN_REVISION}/thermo/NASA9/{filename}",
                     "sha256": digest((source_directory / filename).read_bytes())}
               for key, filename in filenames.items()}
    pressure_probe = read_records(source_directory / "new_nasa9.TXT", "burcat2016_vulcan", {"O"})["O"][0]
    return {"id": "phosphorus-cold-research-20261030", "representation": "NASA9 absolute standard functions",
            "standard_pressure_bar": 1.0, "study_temperature_range_K": [200., 1500.],
            "source_gas_constant_J_mol_K": 8.314510,
            "standard_state_evidence": {
                "cea2002": {
                    "url": "https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf",
                    "location": "Printed pages 2 and 7: ideal gases at 1 bar; R = 8.314510 J/(mol K)."},
                "burcat_gas_constant": {
                    "url": "https://respecth.elte.hu/burcat/CAPOLD.FOR",
                    "location": "BLOCK DATA specifies RR = 8.314510; all selected source Hf298 values agree with this R within 0.004 J/mol."},
                "burcat_pressure": {
                    "status": "Inferred from a numerical entropy check; no explicit pressure declaration found in the pinned source header or introductory report.",
                    "method": "O(1D) at 298.15 K, mass 15.9994 u, electronic degeneracy 5: S/R = ln[(2*pi*m*k*T)^(3/2)*k*T/(h^3*p)] + 5/2 + ln(5). Modern exact SI k, h and NA are used only for this check.",
                    "source_entropy_J_mol_K": 156.81600001917914,
                    "sackur_tetrode_1bar_J_mol_K": 156.81622324208422,
                    "sackur_tetrode_1atm_J_mol_K": 156.7067794589816,
                    "conclusion": "The source matches 1 bar within 0.000224 J/(mol K); 1 atm differs by 0.10922 J/(mol K). This supports the shared 1 bar convention but does not independently audit each molecular entropy.",
                    "probe_record_outside_network": pressure_probe}},
            "source_revision": VULCAN_REVISION, "sources": sources,
            "groups": {"base": BASE, "additions": ["HPO2", "H3PO4"], "full_p": FULL_P, "cns": CNS},
            "records": selected,
            "source_policy": "CEA2002 background, atoms and available P gases; Burcat2016 only for explicitly missing gas identities. No fitted coefficient or energy anchor is altered.",
            "identity_notes": {"HPO2": "HOPO", "H3PO4": "O=P(OH)3",
                               "H3PO3_phosphorous": "P(OH)3", "H3PO3_phosphonic": "O=PH(OH)2",
                               "C2H2": "acetylene", "H2CO": "formaldehyde"},
            "limitations": ["Gas-only selected network; omitted gases and condensates are not proven negligible.",
                            "P4S3 gas is absent from both pinned sources and is not included.",
                            "Public Ohno input snapshots are unavailable; this is an explicit experimental model, not an exact reconstruction.",
                            "Source database identity and numerical consistency do not establish physical accuracy."],
            "redistribution": {"scope": "Offline noncommercial research/validation fixture, outside runtime package data.",
                               "burcat_source_header": "Noncommercial use with citation; commercial inclusion requires written permission.",
                               "citation": "Goos, Burcat and Ruscic, New NASA Thermodynamic Polynomials Database with Active Thermochemical Tables updates, ANL 05/20 TAE 960, 27 September 2016.",
                               "cea_reference": "McBride, Zehe and Gordon, NASA/TP-2002-211556; coefficients retained from the pinned VULCAN copy."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=ROOT.parent / "VULCAN/thermo/NASA9")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/data/validation/phosphorus_cold/thermo.json")
    args = parser.parse_args()
    fixture = build_fixture(args.source_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Saved {len(fixture['records'])} gas records to {args.output}")


if __name__ == "__main__":
    main()
