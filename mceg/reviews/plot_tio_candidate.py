"""Plot the archived TiO finite substitution without running equilibrium."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/tio_equilibrium_impact/20261103/equilibria.npz"


def plot(output: Path) -> None:
    """Show all 75 abundance ratios; color describes response, not accuracy."""
    with np.load(DATA) as saved:
        index = saved["species"].tolist().index("O1Ti1")
        logx = saved["ln_n"]-logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        ratio = np.exp(logx[1, :, index]-logx[0, :, index])
        t, p, co = (saved[key] for key in ("temperature_K", "pressure_bar", "C_over_O"))
        temperatures, pressures = np.unique(t), np.unique(p)
        figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
        for ax, carbon_ratio in zip(axes, np.unique(co)):
            table = np.array([[ratio[np.flatnonzero((t == temp) & (p == pressure)
                                                   & (co == carbon_ratio))[0]]
                               for pressure in pressures] for temp in temperatures])
            im = ax.imshow(table, vmin=.6, vmax=1., cmap="cividis", aspect="auto")
            ax.set_xticks(range(5), ["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1", "100"])
            ax.set_yticks(range(5), [f"{value:g}" for value in temperatures])
            ax.set_xlabel("Pressure (bar)")
            ax.set_title(f"C/O = {carbon_ratio:.3f}")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{table[i, j]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if table[i, j] < .79 else "black")
        axes[0].set_ylabel("Temperature (K)")
        figure.colorbar(im, ax=axes, shrink=.85, label="TiO mole-fraction ratio")
        figure.suptitle("TOTO/Jain candidate / current curated\nFixed gas-only network; condensation and transport excluded",
                        fontsize=11)
        figure.savefig(output, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"documents/_static/tio_candidate_abundance.png")
    plot(parser.parse_args().output)
