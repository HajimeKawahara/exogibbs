"""Replay NaOH sources and a saved footprint; no production replacement."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, BASELINE, janaf_g, nasa_g

DATA = ROOT / "tests/data/validation/naoh_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105"
R = 8.31446261815324
NASA_R = 8.314510
KB = 1.380649e-23
PLANCK = 6.62607015e-34
AMU = 1.66053906660e-27


@lru_cache(maxsize=1)
def setup():
    """Pin the named reviewed model instead of following the moving default."""
    import jax
    from exogibbs.presets import curated_cos
    jax.config.update("jax_enable_x64", True)
    return curated_cos.chemsetup(ions=True)


def current_ln_k(temperature: float, species_id: str) -> float:
    """Return the production formation constant only in its declared domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("outside curated-cos-v1 range")
    model = setup()
    return -float(model.hvector_func(float(temperature))[model.species.index(species_id)])


def log_translation(temperature: float, mass_u: float) -> float:
    """Ideal-gas one-bar translation with explicit isotope or mean mass."""
    kt = KB * temperature
    return 1.5 * math.log(2 * math.pi * mass_u * AMU * kt / PLANCK**2) + math.log(kt / 100000)


def partition_calorics(table: np.ndarray, temperature: float, step: float) -> dict:
    """Use five exact nodes; report rounding and a separate step-size check.

    Endpoint stencils are one-sided. These derivatives diagnose the downloaded
    rounded state sum; they are not a production interpolation or accuracy bound.
    """
    offsets = np.arange(-2, 3) if temperature + 2 * step <= table[-1, 0] else np.arange(-4, 1)
    nodes = temperature + step * offsets
    rows = table[np.isin(table[:, 0], nodes)]
    if len(rows) != 5:
        raise ValueError("missing exact partition stencil")
    matrix = np.array([offsets**i for i in range(5)], dtype=float)
    first = np.linalg.solve(matrix, [0, 1, 0, 0, 0]) / step
    second = np.linalg.solve(matrix, [0, 0, 2, 0, 0]) / step**2
    # Remove the constant before differencing to reduce cancellation.
    anchor = rows[np.flatnonzero(offsets == 0)[0], 1]
    ln_q = np.log(rows[:, 1] / anchor)
    d1, d2 = first @ ln_q, second @ ln_q
    # The downloaded PF files print four digits after the decimal point.
    rounding = 0.00005 / rows[:, 1]
    return {"H_minus_H0_J_mol": float(R * (2.5 * temperature + temperature**2 * d1)),
            "Cp_J_mol_K": float(R * (2.5 + 2 * temperature * d1 + temperature**2 * d2)),
            "Cp_rounding_bound_J_mol_K": float(R * ((2 * temperature * abs(first) + temperature**2 * abs(second)) @ rounding)),
            "stencil": "central" if offsets[0] == -2 else "backward"}


def footprint(species_id: str, element: str) -> dict:
    """Read both archived networks; this is not a newly solved response."""
    result = {}
    for mode in ("ions", "neutral"):
        path = GRID / (mode + ".npz")
        with np.load(path) as grid:
            i = grid["species"].tolist().index(species_id)
            e = grid["elements"].tolist().index(element)
            amount = np.exp(grid["ln_n"][:, i])
            share = grid["formula_matrix"][e, i] * amount / grid["budget"][:, e]
            k = int(np.argmax(share))
            result[mode] = {"model": "curated-cos-v1", "source_sha256": sha256(path.read_bytes()).hexdigest(),
                           "cases": len(share), "species_count": len(grid["species"]),
                           "all_converged": bool(np.all(grid["converged"])), "maximum_element_fraction": float(share[k]),
                           "case_index": k, "temperature_K": float(grid["temperature_K"][k]),
                           "pressure_bar": float(grid["pressure_bar"][k]), "C_over_O": float(grid["C_over_O"][k]),
                           "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][k, i])),
                           "maximum_abs_target_h_error": max(abs(float(h) + current_ln_k(t, species_id)) for t, h in zip(grid["temperature_K"], grid["h"][:, i]))}
    return result


def review() -> dict:
    """Separate energy, thermal and elemental reference effects offline."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    baseline = json.loads((BASELINE / "inputs.json").read_text())
    name, species = inputs["name"], inputs["species_id"]
    records, composition = inputs["nasa_records"], inputs["composition"]
    record = records[name]
    table = read_table(DATA / "janaf.txt")
    jh298 = 1000 * table[298.15][4]
    nh298 = record["hf298_J_mol"] * R / NASA_R
    samples = []
    for t in inputs["source_temperatures_K"]:
        nh, ns, ncp = nasa_properties(record, t, R)
        node = table[t]
        molecular = {"H298_anchor_J_mol": jh298 - nh298,
                     "enthalpy_increment_J_mol": 1000 * node[3] - (nh - nh298),
                     "entropy_J_mol": -t * (node[1] - ns)}
        atoms = -sum(n * (janaf_g(baseline["janaf_records"][baseline["atomic_references"][e]["janaf_id"]], t, R) - nasa_g(records[e], t)) * R * t for e, n in composition.items())
        nk = sum(n * nasa_g(records[e], t) for e, n in composition.items()) - nasa_g(record, t)
        ck = current_ln_k(t, species) if t >= 1500 else None
        difference = sum(molecular.values()) + atoms
        samples.append({"temperature_K": t, "molecular_JANAF_minus_NASA_components": molecular,
                        "atomic_reference_contribution_J_mol": atoms,
                        "reaction_JANAF_minus_NASA_G_J_mol": difference,
                        "janaf_over_nasa": math.exp(-difference / (R * t)),
                        "nasa_ln_K": nk, "curated_ln_K": ck,
                        "curated_over_nasa": None if ck is None else math.exp(ck - nk),
                        "JANAF_minus_NASA_Cp_J_mol_K": node[0] - ncp})
    pf = np.loadtxt(DATA / (inputs["line_list"] + ".pf"))
    h0 = record["hf298_J_mol"] - record["h298_minus_h0_J_mol"]
    partition_rows = []
    for t in inputs["partition_temperatures_K"]:
        ng = nasa_g(record, t) - h0 / (NASA_R * t)
        qn = math.exp(-ng - log_translation(t, record["mass_u"]))
        row = {"temperature_K": t, "NASA_internal_Q": qn}
        where = np.flatnonzero(pf[:, 0] == t)
        if len(where):
            q = float(pf[where[0], 1]) / inputs["nuclear_spin_divisor"]
            step = 30 if name == "PH3" else 10
            cal = partition_calorics(pf, t, step)
            cal["Cp_alternate_step_J_mol_K"] = partition_calorics(pf, t, 2 * step)["Cp_J_mol_K"]
            cal["NASA_Cp_common_R_J_mol_K"] = nasa_properties(record, t, R)[2]
            cal["normalized_Q"] = q
            cal["Q_over_NASA"] = q / qn
            cal["same_energy_atoms_isotope_mass_K_proxy"] = q / qn * (inputs["isotopologue_mass_u"] / record["mass_u"])**1.5
            row["downloaded_partition"] = cal
        partition_rows.append(row)
    printed = []
    for t, cp, entropy, increment in inputs["published_table"]:
        nh, ns, ncp = nasa_properties(record, t, NASA_R)
        ph = 1000 * increment + inputs["published_hf298_J_mol"]
        if inputs["published_increment_reference_K"] == 0:
            ph -= inputs["published_h298_minus_h0_J_mol"]
        printed.append({"temperature_K": t, "NASA_minus_published_H_J_mol": nh - ph,
                        "NASA_minus_published_S_J_mol_K": ns - entropy,
                        "NASA_minus_published_Cp_J_mol_K": ncp - cp})
    return {"molecule": name, "species_id": species, "reaction_atoms": composition,
            "standard_pressure_Pa": 100000, "production_model": "curated-cos-v1",
            "source_samples": samples, "partition_samples": partition_rows,
            "published_table_replay": printed, "ground_state_spin_audit": {"raw_ground_weight": int((DATA / "first10.states").read_text().splitlines()[0].split()[2]), "normalization": inputs["nuclear_spin_divisor"]},
            "nuclear_spin_divisor": inputs["nuclear_spin_divisor"],
            "downloaded_definition_weights": inputs["definition_nuclear_spin_weights"],
            "NASA_native_balanced_D0_J_mol": sum(n * (records[e]["hf298_J_mol"] - records[e]["h298_minus_h0_J_mol"]) for e, n in composition.items()) - h0,
            "footprint": footprint(species, inputs["inventory_element"]),
            "production_change": "none", "equilibrium_substitution": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
