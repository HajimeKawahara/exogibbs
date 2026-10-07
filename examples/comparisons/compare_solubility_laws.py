"""Plot the six implemented volatile-solubility laws in their native bases.

Run from the repository root:
    python examples/comparisons/compare_solubility_laws.py

Each curve is an independent ideal pure-component pressure sweep, with
partial pressure = fugacity = total melt pressure. This is an illustrative
law comparison, not a coupled gas-magma equilibrium calculation. Solid lines
mark the metadata's calibration pressure interval; dashed lines extrapolate.
The temperature is 1700 K, within all six metadata temperature intervals.
Only the N law explicitly depends on temperature, redox, and composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from exogibbs.solubility import (
    MELTYQ_SOLUBILITY_METADATA,
    ch4_ardia2013,
    co2_lichtenberg2021,
    co_yoshioka2019,
    h2_hirschmann2012,
    h2o_lichtenberg2021,
    n2_dasgupta2022,
)


def main() -> None:
    """Save a PNG/PDF comparison and print values at a shared calibration P."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "solubility",
        help="Directory for the PNG and PDF figures.",
    )
    args = parser.parse_args()

    # Include calibration boundaries exactly so line styles meet without gaps.
    boundaries_bar = [
        bound * 1.0e4
        for metadata in MELTYQ_SOLUBILITY_METADATA.values()
        for bound in metadata.calibration_total_pressure_gpa
        if 1.0e-4 <= bound <= 3.0
    ]
    pressure_bar = np.unique(
        np.r_[np.geomspace(1.0, 3.0e4, 400), boundaries_bar, 7500.0]
    )
    pressure_gpa = pressure_bar / 1.0e4
    pressure_pa = pressure_bar * 1.0e5
    curves = (
        ("h2_hirschmann2012", r"H$_2$ | Hirschmann (2012)",
         h2_hirschmann2012(pressure_bar, pressure_gpa)),
        ("ch4_ardia2013", r"CH$_4$ | Ardia (2013)",
         ch4_ardia2013(pressure_gpa, pressure_gpa)),
        ("h2o_lichtenberg2021", r"H$_2$O | Lichtenberg (2021)",
         h2o_lichtenberg2021(pressure_pa)),
        ("co2_lichtenberg2021", r"CO$_2$ | Lichtenberg (2021)",
         co2_lichtenberg2021(pressure_pa)),
        ("co_yoshioka2019", "C from CO | Yoshioka (2019)",
         co_yoshioka2019(pressure_bar)),
        ("n2_dasgupta2022", r"Total N from N$_2$ | Dasgupta (2022)",
         n2_dasgupta2022(pressure_gpa, 1700.0, pressure_gpa, 0.0)),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    axes_by_basis = {"mole_fraction": axes[0], "mass_fraction": axes[1]}
    print("At P = 7500 bar, T = 1700 K, delta IW = 0:")
    for index, (name, label, values) in enumerate(curves):
        metadata = MELTYQ_SOLUBILITY_METADATA[name]
        axis = axes_by_basis[metadata.output_basis]
        values = np.asarray(values)
        low, high = metadata.calibration_total_pressure_gpa
        calibrated = (pressure_bar >= low * 1.0e4) & (pressure_bar <= high * 1.0e4)
        color = f"C{index}"
        axis.loglog(pressure_bar, values, "--", color=color, linewidth=1.5)
        axis.loglog(
            pressure_bar, np.where(calibrated, values, np.nan),
            color=color, linewidth=2.5,
        )
        axis.plot([], [], color=color, linewidth=2.5, label=label)
        value = values[pressure_bar == 7500.0].item()
        print(f"  {name:24s} {value:.6e} ({metadata.output_basis})")

    for axis, title, ylabel in zip(
        axes,
        ("Mole-fraction laws", "Mass-fraction laws"),
        ("Dissolved mole fraction", "Dissolved mass fraction"),
    ):
        axis.set(title=title, xlabel="Total melt pressure (bar)", ylabel=ylabel)
        axis.set_xlim(1.0, 3.0e4)
        axis.set_ylim(1.0e-8, 1.0)
        axis.grid(which="major", alpha=0.25)
        axis.legend(loc="upper left", fontsize=9, framealpha=0.95)
    figure.suptitle("ExoGibbs volatile-solubility laws", fontsize=16)
    figure.legend(
        handles=[
            Line2D([], [], color="0.3", linewidth=2.5,
                   label="Within calibration pressure interval"),
            Line2D([], [], color="0.3", linestyle="--", linewidth=1.5,
                   label="Pressure extrapolation"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, frameon=False,
    )
    figure.text(
        0.5, 0.025,
        r"Independent ideal pure gases: $p_i=f_i=P_{\rm melt}$ | $T=1700$ K | "
        r"N: $\Delta$IW$=0$, $(x_{\rm SiO_2},x_{\rm Al_2O_3},x_{\rm TiO_2})"
        r"=(0.56,0.11,0.01)$",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = args.output_dir / f"solubility_laws.{extension}"
        figure.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
