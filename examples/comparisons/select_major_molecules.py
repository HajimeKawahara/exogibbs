"""Reproduce the major-molecule source-adoption study with no runtime changes.

Only Python's standard library and vendored numerical excerpts are needed.
The explicit source choices are scientific judgements, not a residual ranking.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import audit_atomic_ions as audit


REFERENCE_PATH = audit.ROOT / "examples/comparisons/data/major_molecule_reference.json"
CHOICES_PATH = audit.ROOT / "examples/comparisons/data/major_molecule_adoption.json"
NASA_R = 8.314510
R = audit.GAS_CONSTANT_J_MOL_K
EV_K = audit.EV_J / audit.BOLTZMANN_J_K
ATOMIC_MASS_KG = 1.66053906892e-27


def nasa_interval(name: str, temperature: float, data: dict) -> dict:
    """Select a source interval without extrapolation."""
    for interval in data["nasa9_records"][name]["intervals"]:
        lo, hi = interval["bounds_K"]
        if math.isfinite(temperature) and lo <= temperature <= hi:
            return interval
    raise ValueError("Temperature outside the NASA intervals for " + name)


def janaf_row(name: str, temperature: float, data: dict) -> list:
    """Return an exact table node; interpolation is deliberately unspecified."""
    try:
        return data["janaf"][name]["rows"][str(float(temperature))]
    except KeyError as error:
        raise ValueError("No JANAF node for this species and temperature") from error


def assigned_h0(name: str, family: str, data: dict) -> float:
    """Return assigned H(0) in the source's 298 K elemental reference gauge."""
    if family == "nasa9":
        record = data["nasa9_records"][name]
        return record["hf298_J_mol"] - record["h298_minus_h0_J_mol"]
    if family == "janaf":
        return 1000 * (janaf_row(name, 298.15, data)[4]
                       + janaf_row(name, 0, data)[3])
    raise ValueError("Unknown thermal source")


def atomization_kelvin(name: str, family: str, data: dict) -> float:
    """Obtain D0/kB from a balanced reaction, never from b1 alone."""
    composition = data["species"][name]["composition"]
    if family == "atct":
        records = data["atct"]["records"]
        energy = (sum(n * records[el]["hf0_kJ_mol"] for el, n in composition.items())
                  - records[name]["hf0_kJ_mol"])
        return 1000 * energy / R
    energy = (sum(n * assigned_h0(el, family, data) for el, n in composition.items())
              - assigned_h0(name, family, data))
    return energy / (NASA_R if family == "nasa9" else R)


def thermal_g_rt(name: str, temperature: float, family: str, data: dict) -> float:
    """Evaluate (G-H(0))/(RT), retaining the source entropy and heat capacity."""
    if family == "nasa9":
        return (audit.nasa9_g_rt(temperature, nasa_interval(name, temperature, data))
                - assigned_h0(name, family, data) / (NASA_R * temperature))
    if family == "janaf":
        row = janaf_row(name, temperature, data)
        sensible_h = 1000 * (row[3] - janaf_row(name, 0, data)[3])
        return sensible_h / (R * temperature) - row[1] / R
    raise ValueError("Unknown thermal source")


def formation_log_kp(name: str, temperature: float, family: str, energy_kelvin: float,
                     data: dict, atomic_family: str = "nasa9") -> float:
    """Form a molecule from neutral atoms at p0=1 bar with an explicit D0.

    Adopted NASA and JANAF candidates share NASA neutral-atom thermal anchors.
    Raw JANAF comparisons instead pass atomic_family='janaf'.
    """
    atoms = sum(n * thermal_g_rt(el, temperature, atomic_family, data)
                for el, n in data["species"][name]["composition"].items())
    return atoms - thermal_g_rt(name, temperature, family, data) + energy_kelvin / temperature


def partition_log_kp(name: str, temperature: float, data: dict) -> float:
    """Independently reconstruct BC Eq. 7 from Q, masses and D0 (SI units)."""
    index = data["temperature_K"].index(temperature)
    record = data["barklem_collet"][name]
    left, right = record["dissociation"]["components"]
    masses = [data["nasa9_records"][el]["mass_u"] for el in (left, right)]
    reduced_mass = masses[0] * masses[1] / sum(masses) * ATOMIC_MASS_KG
    kt = audit.BOLTZMANN_J_K * temperature
    qa, qb = [data["atomic_partitions"][el]["values"][index] for el in (left, right)]
    qm = record["partition"]["values"][index]
    return (math.log(qm / (qa * qb))
            - 1.5 * math.log(2 * math.pi * reduced_mass * kt / audit.PLANCK_J_S**2)
            + math.log(audit.STANDARD_PRESSURE_PA / kt)
            + record["dissociation"]["dissociation_eV"] * EV_K / temperature)


def target_energy(name: str, choice: dict, data: dict) -> float:
    """Return the explicitly selected ground-state atomization energy / kB."""
    source = choice["energy_source"]
    if source in ("atct", "janaf", "nasa9"):
        return atomization_kelvin(name, source, data)
    if source == "barklem_collet":
        return data["barklem_collet"][name]["dissociation"]["dissociation_eV"] * EV_K
    if source == "spectroscopy":
        return data["spectroscopy"][name]["dissociation_eV"] * EV_K
    if source == "burrows":
        return data["tih_burrows"]["dissociation_eV"] * EV_K
    raise ValueError("Unknown energy source")


def _peak(rows, first, second):
    available = [r for r in rows if r[first] is not None and r[second] is not None]
    if not available:
        return None, None
    row = max(available, key=lambda r: abs(r[first] - r[second]))
    return 10 ** abs(row[first] - row[second]), row["temperature_K"]


def compute_adoption() -> tuple:
    """Return 350 comparison rows and 35 explicit adoption decisions."""
    data = json.loads(REFERENCE_PATH.read_text())
    choices = json.loads(CHOICES_PATH.read_text())["choices"]
    if set(choices) != set(data["species"]) or data["nasa9_gas_constant_J_mol_K"] != NASA_R:
        raise ValueError("Review the source-selection scope or NASA convention")
    source = data["sources"]["fastchem"]
    if hashlib.sha256((audit.ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("FastChem data changed; review this pinned study")
    rows, decisions = [], []
    for name, record in data["species"].items():
        choice = choices[name]
        energy = target_energy(name, choice, data)
        energies = {family: atomization_kelvin(name, family, data)
                    for family, key in (("nasa9", "nasa9_records"), ("janaf", "janaf"))
                    if name in data[key]}
        samples = []
        for index, t in enumerate(data["temperature_K"]):
            fc = audit.fastchem_log_kp(t, record["coefficients"])
            values = {"fastchem": fc, "nasa9": None, "janaf": None, "bc": None,
                      "nasa9_updated": None, "janaf_updated": None, "fc_energy_only": None}
            for family, source_energy in energies.items():
                values[family] = formation_log_kp(name, t, family, source_energy, data, family)
                values[family + "_updated"] = formation_log_kp(name, t, family, energy, data)
            if name in data["barklem_collet"]:
                bc = data["barklem_collet"][name]
                values["bc"] = (5 - bc["dissociation_log10_pK_Pa"]["values"][index]) * audit.LN10
            if "JANAF" in record["source"] and "janaf" in energies:
                values["fc_energy_only"] = fc + (energy - energies["janaf"]) / t
            selected = fc
            if choice["thermal_source"] in ("nasa9", "janaf"):
                selected = values[choice["thermal_source"] + "_updated"]
            elif choice["energy_source"] == "spectroscopy":
                old = data["barklem_collet"][name]["dissociation"]["dissociation_eV"] * EV_K
                selected += (energy - old) / t
            lo, hi = choice["review_bounds_K"]
            values["selected"] = selected if lo <= t <= hi else None
            row = {"species": name, "fastchem_species": record["fastchem_species"],
                   "temperature_K": t, **{"log10_Kp_" + k: None if v is None else v / audit.LN10
                                          for k, v in values.items()}}
            row["Cp_nasa9_J_mol_K"] = None
            if "nasa9" in energies:
                c = nasa_interval(name, t, data)["coefficients"]
                row["Cp_nasa9_J_mol_K"] = R * (c[0] / t**2 + c[1] / t
                    + sum(c[i] * t**(i-2) for i in range(2, 7)))
            row["Cp_janaf_J_mol_K"] = janaf_row(name, t, data)[0] if "janaf" in energies else None
            samples.append(row)
        decision = {"species": name, "fastchem_species": record["fastchem_species"],
                    "fastchem_source": record["source"], **choice,
                    "adopted_D0_eV": energy / EV_K,
                    "nasa9_source_D0_eV": energies.get("nasa9", math.nan) / EV_K,
                    "janaf_source_D0_eV": energies.get("janaf", math.nan) / EV_K}
        for key in ("nasa9_source_D0_eV", "janaf_source_D0_eV"):
            if not math.isfinite(decision[key]):
                decision[key] = None
        for first, second in (("fastchem", "nasa9"), ("fastchem", "janaf"), ("fastchem", "bc"),
                              ("nasa9_updated", "janaf_updated"), ("selected", "fastchem")):
            label = first + "_vs_" + second
            factor, t = _peak(samples, "log10_Kp_" + first, "log10_Kp_" + second)
            decision[label + "_max_factor"] = factor
            decision[label + "_worst_K"] = t
        decisions.append(decision)
        rows.extend(samples)
    return rows, decisions


def main() -> None:
    """Write the adoption table, numerical comparisons and their file hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=audit.ROOT / "results/major_molecule_adoption")
    args = parser.parse_args()
    rows, decisions = compute_adoption()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit._write_csv(args.output_dir / "comparison.csv", rows)
    audit._write_csv(args.output_dir / "decisions.csv", decisions)
    columns = ["species", "fastchem_source", "selected_source", "temperature_scope",
               "status", "difference_cause", "reason", "follow_up"]
    audit._write_csv(args.output_dir / "adoption_table.csv",
                     [{k: row[k] for k in columns} for row in decisions])
    paths = [Path(__file__).resolve(), Path(audit.__file__).resolve(), REFERENCE_PATH,
             CHOICES_PATH, audit.FASTCHEM_PATH]
    data = json.loads(REFERENCE_PATH.read_text())
    partition_errors = [abs(partition_log_kp(name, t, data) / audit.LN10
                            - (5 - rec["dissociation_log10_pK_Pa"]["values"][i]))
                        for name, rec in data["barklem_collet"].items()
                        for i, t in enumerate(data["temperature_K"])]
    receipt = {
        "command": "python3 examples/comparisons/select_major_molecules.py",
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=audit.ROOT, text=True).strip(),
        "python": platform.python_version(), "species_count": len(decisions),
        "comparison_rows": len(rows), "temperature_K": data["temperature_K"],
        "standard_pressure_Pa": audit.STANDARD_PRESSURE_PA,
        "selection_counts": dict(Counter(r["selected_source"] for r in decisions)),
        "status_counts": dict(Counter(r["status"] for r in decisions)),
        "bc_partition_reconstruction_max_abs_dex": max(partition_errors),
        "selected_file_sha256": {str(p.relative_to(audit.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths},
        "output_sha256": {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                          for name in ("comparison.csv", "decisions.csv", "adoption_table.csv")},
        "interpretation": "Source-selection study, not a production preset or physical validation. Maxima are sampled symmetric Kp ratios, not abundance errors or uncertainties. Common NASA neutral thermal anchors are used for adopted NASA/JANAF candidates. Retained FastChem curves are formation-K models; their source agreement is not independent validation. ATcT 1.172 is a reviewed snapshot, not the latest release.",
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Recorded {len(decisions)} source decisions and {len(rows)} comparisons in {args.output_dir}")


if __name__ == "__main__":
    main()
