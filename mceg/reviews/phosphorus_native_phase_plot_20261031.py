"""Plot the frozen native GGChem / ExoGibbs gas-and-condensate comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/phosphorus_cold_validation"


def main() -> None:
    receipt = json.loads((BASE / "condensate/comparison_receipt.json").read_text())
    reference = json.loads((BASE / "native7_phase_receipt.json").read_text())
    assert receipt["states_passed"] == len(receipt["states"]) == 31
    temperature = np.asarray(receipt["temperature_K"])
    selected = np.flatnonzero(temperature <= 1500)[::-1]
    t = temperature[selected]
    logx = np.array([state["gas_logx"] for state in receipt["states"]])[selected]
    native_logx = np.array([
        state["native_gas_logx_reconstructed"] for state in receipt["states"]
    ])[selected]
    gas_amounts = np.exp(np.array([
        state["gas_log_amounts"] for state in receipt["states"]
    ]))[selected]
    phase_amounts = np.array([
        state["condensate_amounts_per_H"] for state in receipt["states"]
    ])[selected]
    native_phase = np.array([
        state["native_condensate_amounts_per_H"] for state in receipt["states"]
    ])[selected]
    phosphorus = receipt["elements"].index("P")
    gas_stoich = np.asarray(receipt["gas_formula_matrix"])[phosphorus]
    phase_stoich = np.asarray(receipt["condensate_formula_matrix"])[phosphorus]
    budget = receipt["budget"][phosphorus]
    fig, (ax, inventory) = plt.subplots(1, 2, figsize=(12.4, 5.2), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    species = ("PH3", "H3PO4", "HPO2", "P4O10", "P2", "PO", "P4O6", "P4S3")
    for color, name in zip(colors, species):
        column = receipt["gas_species"].index(name)
        ax.plot(t, np.exp(logx[:, column]), color=color, label=name, linewidth=1.8)
        ax.plot(t, np.exp(native_logx[:, column]), color=color, marker="o",
                markersize=4.5, markerfacecolor="none", linestyle="none")
    ax.set(xscale="log", yscale="log", xlim=(200, 1500), ylim=(1e-30, 1e-5),
           xlabel="Temperature [K]", ylabel="Gas mole fraction",
           title="Gas composition with condensation")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.text(0.03, 0.04, "Lines: ExoGibbs\nCircles: native GGChem", transform=ax.transAxes,
            fontsize=10, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    inventory.plot(t, gas_amounts @ gas_stoich / budget, color="tab:blue", label="P in gas (ExoGibbs)")
    inventory.plot(t, np.asarray(reference["gas_P_fraction"])[selected], color="tab:blue",
                   marker="o", markerfacecolor="none", linestyle="none", markersize=5)
    inventory.plot(t, phase_amounts @ phase_stoich / budget, color="tab:orange", label="P in condensates (ExoGibbs)")
    inventory.plot(t, native_phase @ phase_stoich / budget, color="tab:orange",
                   marker="o", markerfacecolor="none", linestyle="none", markersize=5)
    inventory.plot(t, np.asarray(reference["matched_gas_only"]["gas_P_fraction"])[selected],
                   color="0.4", linestyle="--", label="P in gas (gas-only reference)")
    inventory.set(xscale="log", yscale="log", xlim=(200, 1500), ylim=(1e-24, 2),
                  xlabel="Temperature [K]", ylabel="Fraction of elemental P inventory",
                  title="Phosphorus depletion into H3PO4(s/l)")
    inventory.legend(loc="lower right", fontsize=9)
    for panel in (ax, inventory):
        panel.set_xticks([200, 300, 400, 600, 1000, 1500], ["200", "300", "400", "600", "1000", "1500"])
        panel.grid(alpha=0.2, which="both")
    fig.suptitle("1 mbar; solar H–He–C–N–O–P–S; 1156 gases + 22 pure condensates\n"
                 "Matched native thermochemical functions; 23 of 31 validated temperatures shown", fontsize=12)
    target = BASE / "native7_condensate_comparison.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
