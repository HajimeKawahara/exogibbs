"""Compare M -> M+ + e- using local FastChem, JANAF, NASA9, and Saha data.

All Kp values are dimensionless at 1 bar. Saha uses full tabulated atomic
partition functions; the electron spin factor 2 is included separately.
Run with --output-dir results/ni_co_ionization to save CSV and two figures.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

from jax.experimental import enable_x64
import numpy as np

from exogibbs.io.load_data import get_data_filepath, load_JANAF_rawtxt
from exogibbs.presets import fastchem, fastchem4
from exogibbs.thermo.standard import _nasa9
from exogibbs.utils.constants import R_gas_constant_si


DATA_DIR = Path(__file__).resolve().parent / "data"
REFERENCE_PATH = DATA_DIR / "ni_co_ionization_reference.json"
FASTCHEM_PATH = "FastChem4/logK/logK.dat"
BOLTZMANN_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
ELECTRON_MASS_KG = 9.1093837139e-31
ATOMIC_MASS_KG = 1.66053906892e-27
EV_J = 1.602176634e-19
STANDARD_PRESSURE_PA = 1.0e5


def saha_log_kp(
    temperature: np.ndarray, partition_ratio: np.ndarray, ionization_energy_eV: float
) -> np.ndarray:
    """Return ln Kp; Kn in SI is n(M+) n(e-) / n(M), in m^-3."""
    thermal_energy = BOLTZMANN_J_K * np.asarray(temperature)
    log_kn = (
        np.log(2.0 * np.asarray(partition_ratio))
        + 1.5 * np.log(2.0 * np.pi * ELECTRON_MASS_KG * thermal_energy / PLANCK_J_S**2)
        - ionization_energy_eV * EV_J / thermal_energy
    )
    return log_kn + np.log(thermal_energy / STANDARD_PRESSURE_PA)


def log_kp_to_log_kn_cm3(log_kp: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Convert ln Kp to ln[n(M+) n(e-) / n(M)] with densities in cm^-3."""
    return np.asarray(log_kp) + np.log(
        STANDARD_PRESSURE_PA / (BOLTZMANN_J_K * np.asarray(temperature))
    ) - np.log(1.0e6)


def _janaf_formation_g(table_name, temperature):
    # Select actual table nodes; no interpolation or extrapolation is needed.
    table = load_JANAF_rawtxt(DATA_DIR / table_name).set_index("T(K)")
    return table.loc[temperature, "delta-f G"].to_numpy(dtype=float) * 1000.0


def janaf_partition(table_name: str, temperature: np.ndarray, mass_u: float) -> np.ndarray:
    """Recover electronic Q from JANAF entropy and H(T)-H(0), including J weights."""
    table = load_JANAF_rawtxt(DATA_DIR / table_name).set_index("T(K)")
    entropy = table.loc[temperature, "S"].to_numpy(dtype=float)
    delta_h = (table.loc[temperature, "H-H(Tr)"].to_numpy(dtype=float)
               - float(table.loc[0.0, "H-H(Tr)"])) * 1000.0
    thermal_energy = BOLTZMANN_J_K * np.asarray(temperature)
    log_translation = (
        1.5 * np.log(2.0 * np.pi * mass_u * ATOMIC_MASS_KG * thermal_energy / PLANCK_J_S**2)
        + np.log(thermal_energy / STANDARD_PRESSURE_PA)
    )
    return np.exp(entropy / R_gas_constant_si
                  - delta_h / (R_gas_constant_si * temperature) - log_translation)


def compute_comparison() -> list[dict]:
    """Return one row per element and tabulated temperature, entirely offline."""
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    temperature = np.asarray(reference["temperature_K"], dtype=float)
    nasa = json.loads(get_data_filepath("nasa9/fastchem4_hot.json").read_text())
    coefficients, _ = fastchem._parse_fastchem_coeffs(
        get_data_filepath(FASTCHEM_PATH).read_text()
    )
    electron_g = _janaf_formation_g("D-020.txt", temperature)
    rows = []
    with enable_x64():
        setup = fastchem4.chemsetup(path=FASTCHEM_PATH, silent=True)
        hvector = np.asarray(setup.hvector_func(temperature))
        for element in ("Ni", "Co"):
            ion = element + "1+"
            atom = reference["elements"][element]
            ratio = np.asarray(atom["partition_ion"]) / atom["partition_neutral"]
            saha = saha_log_kp(temperature, ratio, atom["ionization_energy_eV"])
            fitted = np.asarray(fastchem.logk(temperature, coefficients[ion]))
            preset = (
                hvector[:, setup.species.index(element + "1")]
                - hvector[:, setup.species.index(ion)]
                - hvector[:, setup.species.index("e1-")]
            )
            # JANAF formation energies share an elemental reference; its
            # electron formation G is zero, unlike absolute NASA electron G.
            janaf = -(
                _janaf_formation_g(element + "-006.txt", temperature)
                + electron_g
                - _janaf_formation_g(element + "-005.txt", temperature)
            ) / (R_gas_constant_si * temperature)
            nasa_ion = (nasa["records"][ion] if element == "Ni"
                        else reference["nasa9_Co_ion"])
            q_neutral = janaf_partition(
                element + "-005.txt", temperature, nasa["records"][element]["mass_u"]
            )
            q_ion = janaf_partition(element + "-006.txt", temperature, nasa_ion["mass_u"])
            nasa_coefficients = np.asarray([
                nasa["records"][element]["coefficients"],
                nasa_ion["coefficients"],
                nasa["records"]["e-"]["coefficients"],
            ])
            g = np.asarray(_nasa9(temperature, nasa_coefficients)[0])
            nasa_log_kp = g[:, 0] - g[:, 1] - g[:, 2]
            for i, value in enumerate(temperature):
                rows.append({
                    "element": element,
                    "temperature_K": float(value),
                    "log10_Kp_fastchem": float(fitted[i] / np.log(10.0)),
                    "log10_Kp_exogibbs": float(preset[i] / np.log(10.0)),
                    "log10_Kp_janaf": float(janaf[i] / np.log(10.0)),
                    "log10_Kp_saha": float(saha[i] / np.log(10.0)),
                    "log10_Kp_nasa9": float(nasa_log_kp[i] / np.log(10.0)),
                    "fastchem_over_saha": float(np.exp(fitted[i] - saha[i])),
                    "fastchem_over_janaf": float(np.exp(fitted[i] - janaf[i])),
                    "janaf_over_saha": float(np.exp(janaf[i] - saha[i])),
                    "Q_janaf_neutral": float(q_neutral[i]),
                    "Q_reference_neutral": float(atom["partition_neutral"][i]),
                    "Q_janaf_ion": float(q_ion[i]),
                    "Q_reference_ion": float(atom["partition_ion"][i]),
                })
    return rows


def _write_csv(rows, stream):
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _plot(rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, element in zip(axes, ("Ni", "Co")):
        selected = [row for row in rows if row["element"] == element]
        temperature = [row["temperature_K"] for row in selected]
        for source, label in (("fastchem", "FastChem v3/v4"), ("janaf", "JANAF"),
                              ("nasa9", "NASA9")):
            residual = [row["log10_Kp_" + source] - row["log10_Kp_saha"]
                        for row in selected]
            ax.plot(temperature, residual, "o-", label=label)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set(title=element + " -> " + element + "+ + e-", xlabel="Temperature [K]",
               ylabel="log10(Kp / Kp,Saha)")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_partitions(rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, element in zip(axes, ("Ni", "Co")):
        selected = [row for row in rows if row["element"] == element]
        temperature = [row["temperature_K"] for row in selected]
        for state, label in (("neutral", element), ("ion", element + "+")):
            ratio = [row["Q_janaf_" + state] / row["Q_reference_" + state]
                     for row in selected]
            ax.semilogy(temperature, ratio, "o-", label=label)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set(title=element + " electronic partition functions",
               xlabel="Temperature [K]", ylabel="Q(JANAF) / Q(Barklem & Collet 2016)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Save comparison.csv and comparison/partition figures")
    args = parser.parse_args()
    rows = compute_comparison()
    _write_csv(rows, sys.stdout)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            _write_csv(rows, stream)
        _plot(rows, args.output_dir / "comparison.png")
        _plot_partitions(rows, args.output_dir / "partition_functions.png")


if __name__ == "__main__":
    main()
