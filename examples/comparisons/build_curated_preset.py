"""Rebuild the versioned curated gas data from reviewed, offline excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import audit_atomic_ions as audit
import select_atomic_ions as atomic
import select_major_molecules as molecular


REFERENCE_PATH = Path(__file__).with_name("data") / "curated_continuous_reference.json"
OUTPUT_PATH = audit.ROOT / "src/exogibbs/data/curated/gas_v1.json"
KNOTS = np.array([1500., 2000., 3000., 4000.])
ABUNDANCE_PATH = "FastChem4/element_abundances/asplund_2021.dat"


def partition_fit(temperatures: np.ndarray, values: np.ndarray, ground: float) -> dict:
    """Fit positive Boltzmann weights, preserving nonnegative internal Cv.

    Excitation energies form a fixed numerical basis; the resulting terms
    are not a spectroscopic level list. Relative least squares respects the
    tabulated precision across the large dynamic range of partition values.
    """
    energies = np.geomspace(1., 150000., 300)
    matrix = np.exp(-energies[None, :] / temperatures[:, None]) / values[:, None]
    weights, _ = nnls(matrix, (values - ground) / values, maxiter=30000)
    keep = weights > 0
    return {"excitation_K": [0., *energies[keep].tolist()],
            "weights": [ground, *weights[keep].tolist()],
            "representation": "positive_Boltzmann_fit_to_BC2016_table8"}


def janaf_polynomials(name: str, data: dict) -> list:
    """Quintic Hermite coefficients for (G-H0)/R in u=(T-T_left)/width.

    Endpoint value, slope and curvature are fixed by H, -S and -Cp/T.
    Thus a single C2 free energy defines all three thermal functions.
    """
    record = data["janaf"][name]
    values = []
    for temperature in KNOTS:
        cp, entropy, sensible = record["rows"][str(temperature)]
        enthalpy = 1000 * (sensible - record["h0_minus_h298_kJ_mol"])
        values.append(np.array([enthalpy - temperature * entropy, -entropy,
                                -cp / temperature]) / molecular.R)
    coefficients = []
    for left, right, width in zip(values[:-1], values[1:], np.diff(KNOTS)):
        c0, c1, c2 = left * [1, width, width**2 / 2]
        r0 = right[0] - c0 - c1 - c2
        r1 = width * right[1] - c1 - 2 * c2
        r2 = width**2 * right[2] - 2 * c2
        coefficients.append([c0, c1, c2, 10*r0 - 4*r1 + r2/2,
                             -15*r0 + 7*r1 - r2, 6*r0 - 3*r1 + r2/2])
    return np.asarray(coefficients).tolist()


def build_data() -> dict:
    """Assemble source choices and continuous representations without MCEG."""
    reference = json.loads(audit.REFERENCE_PATH.read_text())
    atom_data = json.loads(atomic.DATA_PATH.read_text())
    molecules = json.loads(molecular.REFERENCE_PATH.read_text())
    choices = json.loads(molecular.CHOICES_PATH.read_text())["choices"]
    continuous = json.loads(REFERENCE_PATH.read_text())
    nasa = json.loads(audit.NASA_PATH.read_text())["records"]
    nasa.update(reference["nasa9_supplement"])
    _, atomic_decisions = atomic.compute_adoption()
    nasa_reactions, partition_reactions, decisions, excluded = {}, {}, {}, {}
    for decision in atomic_decisions:
        name = decision["species"]
        family = decision["selected_family"]
        if family == "exclude_unbound":
            excluded[name] = decision["reason"]
            continue
        element, charge = name[:-2], (1 if name[-1] == "+" else -1)
        decisions[name] = {
            "family": family, "condition": decision["condition"],
            "status": ("prefer_conditionally" if element in ("Na", "K") else "prefer"),
            "reason": decision["reason"],
            "threshold_eV": decision["adopted_threshold_eV"],
            "source_group": "atomic",
        }
        # The continuous representation replaces the study's node-only condition.
        if family == "spectroscopic_partition":
            decisions[name]["condition"] = "continuous_partition_representation_1500_4000_K"
            partition_reactions[name] = {
                "neutral": element + "_I",
                "ion": element + ("_II" if charge == 1 else "_anion"),
                "charge": charge,
                "threshold_K": decision["adopted_threshold_eV"] * molecular.EV_K,
            }
        else:
            coefficients = (np.array(nasa[name]["coefficients"])
                            + charge * np.array(nasa["e-"]["coefficients"])
                            - np.array(nasa[element]["coefficients"]))
            coefficients[7] += decision["ion_b1_shift_K"]
            nasa_reactions[name] = coefficients.tolist()

    janaf_reactions, energy_shifts = {}, {}
    for name, choice in choices.items():
        record = molecules["species"][name]
        key = record["fastchem_species"]
        decisions[key] = {**choice, "name": name, "source_group": "molecular"}
        energy = molecular.target_energy(name, choice, molecules)
        family = choice["thermal_source"]
        if family == "nasa9":
            coefficients = np.array(molecular.nasa_interval(name, 2500., molecules)["coefficients"])
            for element, count in record["composition"].items():
                coefficients -= count * np.array(
                    molecular.nasa_interval(element, 2500., molecules)["coefficients"])
            coefficients[7] += molecular.atomization_kelvin(name, "nasa9", molecules) - energy
            nasa_reactions[key] = coefficients.tolist()
        elif family == "janaf":
            background = np.zeros(9)
            for element, count in record["composition"].items():
                background -= count * np.array(
                    molecular.nasa_interval(element, 2500., molecules)["coefficients"])
                background[7] += count * molecular.assigned_h0(element, "nasa9", molecules) / molecular.NASA_R
            background[7] -= energy
            janaf_reactions[key] = {
                "knots_K": KNOTS.tolist(), "coefficients": janaf_polynomials(name, continuous),
                "background_nasa9": background.tolist(),
                "table_id": continuous["janaf"][name]["table_id"],
            }
        elif choice["energy_source"] == "spectroscopy":
            old = molecules["barklem_collet"][name]["dissociation"]["dissociation_eV"]
            energy_shifts[key] = energy - old * molecular.EV_K

    partitions = {}
    temperatures = np.array(continuous["partition_temperature_K"])
    for name, record in continuous["partitions"].items():
        partitions[name] = partition_fit(temperatures, np.array(record["values"]), record["ground_weight"])
    for element in ("C", "Ca", "Ge"):
        levels = reference["anions"][element]["levels"]
        partitions[element + "_anion"] = {
            "excitation_K": [level[1] * 100 * audit.PLANCK_J_S * audit.LIGHT_SPEED_M_S
                             / audit.BOLTZMANN_J_K for level in levels],
            "weights": [level[2] for level in levels],
            "representation": "evaluated_bound_spectroscopic_levels",
        }

    paths = [audit.REFERENCE_PATH, atomic.DATA_PATH, audit.NASA_PATH,
             molecular.REFERENCE_PATH, molecular.CHOICES_PATH, REFERENCE_PATH]
    abundance = audit.ROOT / "src/exogibbs/data" / ABUNDANCE_PATH
    return {
        "id": "curated-gas-v1", "temperature_range_K": [1500., 4000.],
        "standard_pressure_bar": 1.,
        "baseline": {"path": "FastChem4/logK/logK.dat",
                     "sha256": hashlib.sha256(audit.FASTCHEM_PATH.read_bytes()).hexdigest()},
        "abundances": {"path": ABUNDANCE_PATH, "sha256": hashlib.sha256(abundance.read_bytes()).hexdigest(),
                       "normalization": "sum of nuclear abundances equals one; zero net charge"},
        "source_sha256": {str(p.relative_to(audit.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "sources": {"atomic": reference["sources"], "atomic_thresholds": atom_data["sources"],
                    "molecular": molecules["sources"], "partitions": continuous["partition_source"],
                    "janaf": {n: {k: r[k] for k in ("table_id", "url", "evaluation_note")}
                              for n, r in continuous["janaf"].items()},
                    "atct": {k: v for k, v in molecules["atct"].items() if k != "records"},
                    "spectroscopy": molecules["spectroscopy"]},
        "excluded_species": excluded, "decisions": decisions,
        "nasa9_reactions": nasa_reactions, "janaf_reactions": janaf_reactions,
        "partition_functions": partitions, "partition_reactions": partition_reactions,
        "logk_energy_shifts_K": energy_shifts,
        "interpretation": "A versioned source-adoption model, not a whole-mixture physical validation. "
                          "Unreviewed species retain their pinned FastChem fits. Atomic translation uses "
                          "unit heavy-particle mass ratios and electron spin two, as in the approved audit. "
                          "Partition fit terms are numerical basis functions, not measured new levels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False) + "\n")
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
