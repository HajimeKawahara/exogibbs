"""Record species-specific source choices following the atomic-ion audit.

This offline study evaluates candidate models; it does not install a preset.
Run beside audit_atomic_ions.py with Python's standard library only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import audit_atomic_ions as audit


DATA_PATH = audit.ROOT / "examples/comparisons/data/atomic_ion_adoption.json"
NASA_GAS_CONSTANT = 8.314510  # Source convention, NASA/TP-2002-211556 Appendix A.
PARTITION_CHOICES = {
    "Co1+": "Updating the threshold leaves a cation partition discrepancy; use NIST IE and CDS neutral/cation partitions.",
    "C1-": "NASA has an incompatible low-temperature partition weight; use evaluated bound levels and EA. FastChem is a close comparator.",
    "Ca1-": "No NASA record; FastChem differs by factor 2.44. Use evaluated bound levels and EA.",
    "Ge1-": "Updating EA does not repair the NASA partition discrepancy; use evaluated bound levels and EA.",
}


def enthalpy_shift_kelvin(element: str, charge: int, reference: dict,
                         data: dict) -> float:
    """Return the ion b1 shift from independent ground-state thresholds.

    The source header supplies assigned H(0), not the high-temperature b1.
    Converting its reaction energy with NASA's original R preserves the
    coefficient convention before inserting the modern spectroscopic E/kB.
    """
    if charge not in (-1, 1):
        raise ValueError("Only singly charged ions have an adopted threshold")
    species = element + ("1+" if charge == 1 else "1-")
    if charge == 1:
        threshold = reference["atoms"][element]["ionization_energy_eV"]
    else:
        anion = reference["anions"][element]
        if anion["status"] != "bound_state_reference":
            raise ValueError("No bound-state threshold for " + species)
        threshold = anion["electron_affinity_eV"]
    records = data["nasa9_enthalpy_records"]
    h0 = {key: records[key]["hf298_J_mol"] - records[key]["h298_minus_h0_J_mol"]
          for key in (element, species, "e-")}
    reaction_h0 = h0[species] + charge * h0["e-"] - h0[element]
    return (charge * threshold * audit.EV_J / audit.BOLTZMANN_J_K
            - reaction_h0 / NASA_GAS_CONSTANT)


def updated_nasa9_record(record: dict, shift_kelvin: float) -> dict:
    """Copy one interval, changing only its enthalpy integration constant."""
    updated = {**record, "coefficients": list(record["coefficients"])}
    updated["coefficients"][7] += shift_kelvin
    return updated


def compute_adoption() -> tuple:
    """Return all 49 explicit choices and the seven-node numerical comparison."""
    reference = json.loads(audit.REFERENCE_PATH.read_text())
    data = json.loads(DATA_PATH.read_text())
    if data["nasa9_gas_constant_J_mol_K"] != NASA_GAS_CONSTANT:
        raise ValueError("Review the NASA coefficient convention")
    rows, summaries = audit.compute_audit()
    decisions = []
    for summary in summaries:
        species, charge = summary["species"], summary["charge"]
        element = species[:-2]
        unbound = summary["reference_status"] == "no_stable_bound_anion"
        samples = [row for row in rows if row["species"] == species]
        shift = None
        if not unbound and species in data["nasa9_enthalpy_records"]:
            shift = enthalpy_shift_kelvin(element, charge, reference, data)
        if unbound:
            family, condition = "exclude_unbound", "no_stable_bound_state"
            reason = "The adopted evaluation gives no stable bound anion; a formal fit is insufficient for inclusion."
        elif species in PARTITION_CHOICES:
            family, condition = "spectroscopic_partition", "table_nodes_only"
            reason = PARTITION_CHOICES[species]
        else:
            family, condition = "nasa9_threshold_updated", "candidate_1000_6000_K"
            reason = "Retain smooth NASA thermal functions and replace the ground-state threshold using spectroscopy."
            if element in ("Na", "K"):
                condition = "high_temperature_density_review"
                reason += " Upper-temperature differences originate mainly in the neutral partition; review level dissolution for the application."
        for row in samples:
            value = (None if shift is None else
                     row["log10_Kp_nasa9"] - shift / (row["temperature_K"] * audit.LN10))
            ref = row["log10_Kp_reference"]
            row["log10_Kp_threshold_updated"] = value
            row["dex_threshold_updated_over_reference"] = None if value is None else value - ref
            row["selected_family"] = family
            row["log10_Kp_selected"] = (ref if family == "spectroscopic_partition" else
                                        value if family == "nasa9_threshold_updated" else None)
            row["comparison_kind"] = ("reference_by_construction" if family == "spectroscopic_partition"
                                       else "no_model" if unbound else "source_comparison")
        available = [row for row in samples if row["log10_Kp_threshold_updated"] is not None]
        worst = max(available, key=lambda r: abs(r["dex_threshold_updated_over_reference"])) if available else None
        threshold = (None if unbound else reference["atoms"][element]["ionization_energy_eV"]
                     if charge == 1 else reference["anions"][element]["electron_affinity_eV"])
        decisions.append({
            "species": species, "selected_family": family, "condition": condition,
            "adopted_threshold_eV": threshold, "ion_b1_shift_K": shift,
            "fastchem_max_factor": (None if unbound else 10**summary["fastchem_max_abs_dex"]),
            "nasa9_max_factor": (None if summary["nasa9_max_abs_dex"] is None
                                 else 10**summary["nasa9_max_abs_dex"]),
            "threshold_updated_nasa9_max_factor": (None if worst is None else
                                                   10**abs(worst["dex_threshold_updated_over_reference"])),
            "threshold_updated_worst_temperature_K": None if worst is None else worst["temperature_K"],
            "reason": reason,
        })
    return rows, decisions


def main() -> None:
    """Write decisions, comparisons and a receipt without changing runtime data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=audit.ROOT / "results/atomic_ion_adoption")
    args = parser.parse_args()
    rows, decisions = compute_adoption()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("comparison.csv", rows), ("decisions.csv", decisions)):
        audit._write_csv(args.output_dir / name, values)
    paths = [Path(__file__).resolve(), Path(audit.__file__).resolve(), DATA_PATH,
             audit.REFERENCE_PATH, audit.FASTCHEM_PATH, audit.NASA_PATH]
    counts = dict(Counter(row["selected_family"] for row in decisions))
    receipt = {
        "command": "python examples/comparisons/select_atomic_ions.py",
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip(),
        "implementation_identity": "Selected file hashes include uncommitted study additions.",
        "python": platform.python_version(),
        "species_count": len(decisions), "selection_counts": counts,
        "temperature_K": sorted({row["temperature_K"] for row in rows}),
        "standard_pressure_Pa": audit.STANDARD_PRESSURE_PA,
        "selected_file_sha256": {
            str(path.relative_to(audit.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "output_sha256": {
            name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            for name in ("comparison.csv", "decisions.csv")
        },
        "interpretation": "Source-adoption decisions, not a production preset. Numerical agreement is not physical validation; the four partition selections equal the reference by construction. Maxima apply only to seven sampled temperatures.",
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Recorded {len(decisions)} choices: {counts}. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
