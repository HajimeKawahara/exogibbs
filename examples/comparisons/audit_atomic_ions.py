"""Audit the 49 default-network FastChem4 atomic ions using pinned local data.

Run with Python's standard library only. MCEG records the model and evidence
separately; neither this audit nor ExoGibbs imports it. See the accompanying
atomic_ion_audit.rst for reaction conventions and limits of the reference.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "examples/comparisons/data/atomic_ion_reference.json"
FASTCHEM_PATH = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
NASA_PATH = ROOT / "src/exogibbs/data/nasa9/fastchem4_hot.json"
BOLTZMANN_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
ELECTRON_MASS_KG = 9.1093837139e-31
LIGHT_SPEED_M_S = 299792458.0
EV_J = 1.602176634e-19
GAS_CONSTANT_J_MOL_K = 8.31446261815324
STANDARD_PRESSURE_PA = 1.0e5
LN10 = math.log(10.0)


def read_fastchem_ions(text: str, elements: set) -> dict:
    """Read singly charged atomic fits, checking their electron stoichiometry."""
    lines = text.splitlines()
    records = {}
    for i, line in enumerate(lines):
        match = re.match(r"^([A-Z][a-z]?)1([+-])\s", line)
        if match is None or match[1] not in elements:
            continue
        element, sign = match.groups()
        species = element + "1" + sign
        composition = line.split(":", 1)[1].split("#", 1)[0].split()
        expected = [element, "1", "e-", "-1" if sign == "+" else "1"]
        coefficients = [float(value) for value in lines[i + 1].split()]
        if composition != expected or len(coefficients) != 5 or species in records:
            raise ValueError(f"Unexpected atomic-ion record: {species}")
        records[species] = {
            "element": element,
            "charge": 1 if sign == "+" else -1,
            "coefficients": coefficients,
            "source_line_1based": i + 1,
            "source": line.split("#", 1)[1].strip(),
        }
    return records


def fastchem_log_kp(temperature: float, coefficients: list) -> float:
    """Evaluate the tabulated natural-log, dimensionless formation Kp."""
    a1, a2, a3, a4, a5 = coefficients
    return (a1 / temperature + a2 * math.log(temperature) + a3
            + a4 * temperature + a5 * temperature**2)


def nasa9_g_rt(temperature: float, record: dict) -> float:
    """Evaluate NASA9 g/(RT) inside the selected interval, without extrapolation."""
    lo, hi = record["bounds_K"]
    if not math.isfinite(temperature) or not lo <= temperature <= hi:
        raise ValueError("Temperature is outside the selected NASA9 interval")
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = record["coefficients"]
    t = temperature
    h_rt = (-a1 / t**2 + a2 * math.log(t) / t + a3 + a4 * t / 2
            + a5 * t**2 / 3 + a6 * t**3 / 4 + a7 * t**4 / 5 + b1 / t)
    s_r = (-a1 / (2 * t**2) - a2 / t + a3 * math.log(t) + a4 * t
           + a5 * t**2 / 2 + a6 * t**3 / 3 + a7 * t**4 / 4 + b2)
    return h_rt - s_r


def reference_log_kp(temperature: float, element: str, charge: int,
                     reference: dict) -> float:
    """Return ionization (+1) or attachment (-1) ln Kp at an exact table node.

    The electron has spin degeneracy two. Heavy-particle translational mass
    ratios are approximated by one, consistently with the prior Ni/Co audit.
    """
    if charge not in (-1, 1):
        raise ValueError("Only singly charged atomic ions are audited")
    index = reference["temperature_K"].index(temperature)
    atom = reference["atoms"][element]
    neutral_q = atom["partition_neutral"][index]
    kt = BOLTZMANN_J_K * temperature
    electron = (math.log(2.0)
                + 1.5 * math.log(2 * math.pi * ELECTRON_MASS_KG * kt / PLANCK_J_S**2)
                + math.log(kt / STANDARD_PRESSURE_PA))
    if charge == 1:
        ion_q = atom["partition_cation"][index]
        energy = -atom["ionization_energy_eV"]
    else:
        anion = reference["anions"][element]
        if anion["status"] != "bound_state_reference":
            raise ValueError(f"No stable bound-anion reference for {element}-")
        ion_q = sum(level[2] * math.exp(
            -PLANCK_J_S * LIGHT_SPEED_M_S * 100 * level[1] / kt
        ) for level in anion["levels"])
        energy = anion["electron_affinity_eV"]
    return math.log(ion_q / neutral_q) + charge * electron + energy * EV_J / kt


def compute_audit() -> tuple:
    """Return sampled comparisons and per-species summaries, entirely offline."""
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    nasa = json.loads(NASA_PATH.read_text(encoding="utf-8"))["records"]
    nasa.update(reference["nasa9_supplement"])
    fits = read_fastchem_ions(FASTCHEM_PATH.read_text(), set(reference["atoms"]))
    expected = ({el + "1+" for el in reference["atoms"]}
                | {el + "1-" for el in reference["anions"]})
    if set(fits) != expected:
        raise ValueError("FastChem inventory changed; review reference coverage")
    rows, summaries = [], []
    for species, fit in sorted(fits.items()):
        element, charge = fit["element"], fit["charge"]
        status = ("saha_reference" if charge == 1
                  else reference["anions"][element]["status"])
        selected = []
        for t in reference["temperature_K"]:
            ref = (None if status == "no_stable_bound_anion"
                   else reference_log_kp(t, element, charge, reference) / LN10)
            fc = fastchem_log_kp(t, fit["coefficients"]) / LN10
            nasa_value = None
            janaf_value = None
            if species == "Cu1+":
                table = reference["janaf_cu_crosscheck"]
                i = table["temperature_K"].index(t)
                delta_g = (table["cation_delta_f_G_kJ_mol"][i]
                           - table["neutral_delta_f_G_kJ_mol"][i])
                janaf_value = -1000 * delta_g / (GAS_CONSTANT_J_MOL_K * t * LN10)
            if species in nasa:
                # A -> A+ + e- or A + e- -> A-; NASA uses absolute electron G.
                nasa_value = (nasa9_g_rt(t, nasa[element])
                              - nasa9_g_rt(t, nasa[species])
                              - charge * nasa9_g_rt(t, nasa["e-"])) / LN10
            selected.append({
                "species": species, "charge": charge, "temperature_K": t,
                "reference_status": status,
                "log10_Kp_reference": ref, "log10_Kp_fastchem": fc,
                "log10_Kp_nasa9": nasa_value,
                "log10_Kp_janaf": janaf_value,
                "dex_fastchem_over_reference": None if ref is None else fc - ref,
                "dex_nasa9_over_reference": (None if ref is None or nasa_value is None
                                             else nasa_value - ref),
            })
        summary = {
            "species": species, "charge": charge, "reference_status": status,
            "fastchem_source_line_1based": fit["source_line_1based"],
            "fastchem_source": fit["source"],
        }
        for model in ("fastchem", "nasa9"):
            field = "dex_" + model + "_over_reference"
            available = [row for row in selected if row[field] is not None]
            worst = max(available, key=lambda row: abs(row[field])) if available else None
            deviation = abs(worst[field]) if worst else None
            if status == "no_stable_bound_anion":
                assessment = "no_stable_bound_anion"
            elif worst is None:
                assessment = "unavailable"
            elif deviation > math.log10(2):
                assessment = "deviation_over_factor_2"
            elif deviation > math.log10(1.1):
                assessment = "deviation_over_factor_1.1"
            else:
                assessment = "within_factor_1.1_at_samples"
            summary[model + "_assessment"] = assessment
            summary[model + "_max_abs_dex"] = deviation
            summary[model + "_worst_temperature_K"] = worst["temperature_K"] if worst else None
            summary[model + "_ratio_at_worst"] = 10**worst[field] if worst else None
        rows.extend(selected)
        summaries.append(summary)
    return rows, summaries


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Write comparisons, summaries and a receipt with selected source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/atomic_ion_audit")
    args = parser.parse_args()
    rows, summaries = compute_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("comparison.csv", rows), ("summary.csv", summaries)):
        _write_csv(args.output_dir / name, records)
    paths = [Path(__file__).resolve(), REFERENCE_PATH, FASTCHEM_PATH, NASA_PATH]
    receipt = {
        "command": "python3 examples/comparisons/audit_atomic_ions.py",
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_identity": "Selected file hashes include uncommitted audit additions.",
        "python": platform.python_version(),
        "temperature_K": sorted({row["temperature_K"] for row in rows}),
        "standard_pressure_Pa": STANDARD_PRESSURE_PA,
        "species_count": len(summaries),
        "numerical_reference_count": sum(
            row["reference_status"] != "no_stable_bound_anion" for row in summaries),
        "selected_file_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "output_sha256": {
            name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            for name in ("comparison.csv", "summary.csv")
        },
        "interpretation": "Factor 1.1 and factor 2 are review thresholds, not uncertainty bounds or physical validation. Maxima apply only to sampled temperatures.",
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Audited {len(summaries)} ions at {len(receipt['temperature_K'])} temperatures; "
          f"{receipt['numerical_reference_count']} numerical references. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
