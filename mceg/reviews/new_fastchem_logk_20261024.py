"""Attribute molecular abundance differences in the archived #297 columns.

This replays mass action at five declared samples. It performs no spectrum or
retrieval calculation and introduces no MCEG runtime dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import numpy as np
from scipy.special import logsumexp

from examples.comparisons.compare_new_fastchem_logk import comparison_setups


RESULTS = ROOT / "tests/data/validation/new_fastchem_logk"
SAMPLES = (("isothermal", "C1H1N1_hcn", 70.79457843841388),
           ("inverted", "C1H1N1_hcn", .01),
           ("inverted", "S2", 10.**-2.45),
           ("inverted", "H2S1", 10.**-2.7),
           ("inverted", "S1Si1", 100.))


def attribution() -> dict:
    """Separate formation constants from the constituent-atom abundance product."""
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX float64 before replaying the archived columns")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "logK.dat"
        with zipfile.ZipFile(RESULTS / "logK.zip") as archive:
            path.write_bytes(archive.read("logK.dat"))
        setups = comparison_setups(path)
    rows = []
    for profile, molecule, pressure in SAMPLES:
        with np.load(RESULTS / f"{profile}.npz") as data:
            species = list(data["species"])
            i = int(np.argmin(np.abs(np.log(data["pressure_bar"] / pressure))))
            if not np.isclose(data["pressure_bar"][i], pressure, rtol=1e-12, atol=0):
                raise ValueError("Requested attribution sample is absent from the archive")
            j = species.index(molecule)
            temperature = float(data["temperature_K"][i])
            logx = {m: data[f"ln_n_{m}"][i] - logsumexp(data[f"ln_n_{m}"][i])
                    for m in ("new", "curated")}
            delta = logx["new"] - logx["curated"]
            delta_logk = float(setups["curated"].hvector_func(temperature)[j]
                               - setups["new"].hvector_func(temperature)[j])
            atoms, atom_log_ratio = {}, 0.
            for element, number in zip(data["elements"], data["formula_matrix"][:, j]):
                if number == 0:
                    continue
                atom = "e1-" if element == "e-" else element + "1"
                atoms[atom] = {"stoichiometry": int(number),
                               "new_over_curated": float(np.exp(delta[species.index(atom)]))}
                atom_log_ratio += number * delta[species.index(atom)]
            residual = float(delta[j] - delta_logk - atom_log_ratio)
            if not np.isfinite(residual) or abs(residual) > 1e-8:
                raise RuntimeError("Saved abundances failed the mass-action factorization")
            rows.append({"profile": profile, "species": molecule, "temperature_K": temperature,
                         "pressure_bar": float(data["pressure_bar"][i]),
                         "K_new_over_curated": float(np.exp(delta_logk)),
                         "atom_product_ratio": float(np.exp(atom_log_ratio)),
                         "abundance_new_over_curated": float(np.exp(delta[j])),
                         "new_x": float(np.exp(logx["new"][j])),
                         "curated_x": float(np.exp(logx["curated"][j])),
                         "ln_factorization_residual": residual, "atomic_ratios": atoms})
    with (ROOT / "tests/data/validation/major_molecule_adoption/decisions.csv").open() as stream:
        hcn = next(row for row in csv.DictReader(stream) if row["species"] == "HCN")
    delta_d0 = float(hcn["adopted_D0_eV"]) - float(hcn["janaf_source_D0_eV"])
    energy_only = float(np.exp(-delta_d0 / (8.617333262145e-5 * 2500.)))
    return {
        "samples": rows,
        "HCN_energy_at_2500_K": {
            "adopted_D0_eV": float(hcn["adopted_D0_eV"]),
            "janaf_D0_eV": float(hcn["janaf_source_D0_eV"]), "delta_D0_eV": delta_d0,
            "energy_only_K_new_over_curated": energy_only,
            "remaining_thermal_and_fit_factor": rows[1]["K_new_over_curated"] / energy_only,
        },
        "interpretation": "Fixed-T,P source sensitivity, not physical error bars. "
                          "The atomic-product term is a coupled equilibrium response, "
                          "not an isolated intervention on atomic-ion coefficients. "
                          "No spectrum, metallicity retrieval, uncertainty distribution "
                          "or whole-mixture physical validation was calculated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/mceg/new_fastchem_logk_20261024/attribution.json")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = attribution()
    comparison_receipt = json.loads((RESULTS / "receipt.json").read_text())
    paths = [ROOT / p for p in comparison_receipt["selected_file_sha256"]]
    paths.extend([Path(__file__), RESULTS / "receipt.json", RESULTS / "logK.zip",
                  RESULTS / "isothermal.npz", RESULTS / "inverted.npz",
                  ROOT / "tests/data/validation/major_molecule_adoption/decisions.csv"])
    report["selected_file_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Five molecular samples passed; saved {args.output}")


if __name__ == "__main__":
    main()
