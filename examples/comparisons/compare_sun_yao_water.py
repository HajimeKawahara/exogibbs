"""Plot conditional Sun--Yao isotherms; no experimental validation implied."""

import argparse
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.solubility import h2o_sun_yao2026, h2o_sun_yao2026_terms, h2o_lichtenberg2021


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    pw = np.geomspace(.01, 440., 160)
    c, a = h2o_sun_yao2026_terms(2173.15, 440., x_na2o=.002, x_al2o3=.025)
    parts = 100*np.asarray(c)*pw[:, None]**np.asarray(a)
    total = 100*np.asarray(h2o_sun_yao2026(pw, 2173.15, 440., x_na2o=.002, x_al2o3=.025))
    legacy = 100*np.asarray(h2o_lichtenberg2021(pw*1e5))
    np.savetxt(args.output/"water_curves.csv", np.column_stack((pw, parts, total, legacy)), delimiter=",",
               header="water_partial_bar,molecular_h2o_wt_percent,hydroxyl_h2o_equivalent_wt_percent,total_h2o_wt_percent,lichtenberg_wt_percent", comments="")
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.loglog(pw, total, label="Sun & Yao (MAGEMin transcription)", lw=2)
    ax.loglog(pw, parts[:, 0], "--", label="Molecular H$_2$O")
    ax.loglog(pw, parts[:, 1], "--", label="OH as H$_2$O equivalent")
    ax.loglog(pw, legacy, ":", label="Lichtenberg (T extrapolation)", color="black")
    ax.set(xlabel="Water partial pressure [bar]", ylabel="Dissolved H$_2$O equivalent [wt%]",
           title="Conditional isotherm, 2173.15 K; total pressure 440 bar\nDry oxide mole fractions: Na$_2$O=0.002, Al$_2$O$_3$=0.025")
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.savefig(args.output/"water_curves.png", dpi=180)


if __name__ == "__main__":
    main()
