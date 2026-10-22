"""Plagioclase exsolution in a teaching regular-solution model
===========================================================

Two copies of an albite (Ab, NaAlSi3O8) / anorthite (An, CaAl2Si2O8)
solid solution can coexist at different compositions. The illustrative
interaction parameter is not fitted to natural labradorite. See
``documents/plagioclase_exsolution.rst`` for the model and its limitations.

Run ``python examples/plagioclase_exsolution.py`` to compare a hot single
phase with a cold homogeneous state and a cold exsolved pair, and save a
free-energy plot under ``results/``. No external provider or data are needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api import multiphase


GAS_CONSTANT_J_MOL_K = 8.31446261815324
INTERACTION_J_MOL = 16000.0  # Illustrative; not a measured plagioclase parameter.
HOT_TEMPERATURE_K = 1200.0
COLD_TEMPERATURE_K = 800.0
PRESSURE_PA = 1.0e5
BULK_AN_FRACTION = 0.6
ELEMENTS = ("Na", "Ca", "Al", "Si", "O")
# Columns are Ab and An; each column counts atoms per formula unit.
ATOM_MATRIX = np.array([[1, 0], [0, 1], [1, 2], [3, 2], [8, 8]], dtype=float)


def regular_solution_gibbs_j(
    temperature_k: float, pressure_pa: float,
    amounts_mol: jax.Array, parameters: dict[str, float],
) -> jax.Array:
    """Return extensive G (J) relative to the shared endmember references."""
    total = jnp.sum(amounts_mol)
    fractions = amounts_mol / jnp.where(total > 0, total, 1.0)
    log_fractions = jnp.log(jnp.where(amounts_mol > 0, fractions, 1.0))
    ideal = GAS_CONSTANT_J_MOL_K * temperature_k * (amounts_mol @ log_fractions)
    excess = parameters["interaction_j_mol"] * total * fractions[0] * fractions[1]
    return ideal + excess


def run_example() -> dict[str, multiphase.Result]:
    """Solve three declared supports for one mole of bulk An60 plagioclase."""
    jax.config.update("jax_enable_x64", True)
    problem = multiphase.Problem(
        elements=ELEMENTS,
        phases=tuple(
            multiphase.Phase(
                name=name, components=("Ab", "An"), atom_matrix=ATOM_MATRIX,
                gibbs_j=regular_solution_gibbs_j,
                standard_state="shared Ab/An reference; endmember terms subtracted",
            )
            for name in ("plagioclase_alpha", "plagioclase_beta")
        ),
    )
    bulk = np.array([1.0 - BULK_AN_FRACTION, BULK_AN_FRACTION])
    budget_mol = ATOM_MATRIX @ bulk
    single_init = np.concatenate((bulk, np.zeros(2)))
    # Seed different compositions (An10 and An90), conserving the bulk by
    # the lever rule. These guesses are not the equilibrium compositions.
    initial_x = np.array([0.1, 0.9])
    beta = (BULK_AN_FRACTION - initial_x[0]) / (initial_x[1] - initial_x[0])
    split_init = (
        np.array([1.0 - beta, beta])[:, None]
        * np.column_stack((1.0 - initial_x, initial_x))
    ).ravel()
    results = {}
    for name, temperature, components, initial in (
        ("hot_single", HOT_TEMPERATURE_K, (0, 1), single_init),
        ("cold_single", COLD_TEMPERATURE_K, (0, 1), single_init),
        ("cold_split", COLD_TEMPERATURE_K, (0, 1, 2, 3), split_init),
    ):
        result = multiphase.solve_fixed_support(
            problem, temperature_k=temperature, pressure_pa=PRESSURE_PA,
            element_amounts_mol=jnp.asarray(budget_mol), init=jnp.asarray(initial),
            parameters={"interaction_j_mol": INTERACTION_J_MOL},
            support=multiphase.Support(components=components),
        )
        if not all(bool(value) for value in (
            result.converged, result.local_kkt_valid,
            result.minimum_valid, result.support_valid,
        )):
            raise RuntimeError(f"{name} failed its fixed-support numerical checks.")
        np.testing.assert_allclose(
            problem.atom_matrix @ np.asarray(result.amounts_mol), budget_mol,
            rtol=1.0e-9, atol=1.0e-10,
        )
        results[name] = result
    if not float(results["cold_split"].gibbs_j) < float(results["cold_single"].gibbs_j):
        raise RuntimeError("The cold split state did not lower Gibbs energy.")
    return results


def plot_results(results: dict[str, multiphase.Result], output: Path) -> None:
    """Plot molar mixing energies and the computed coexistence tie line."""
    import matplotlib.pyplot as plt

    grid = jnp.linspace(0.0, 1.0, 501)
    parameters = {"interaction_j_mol": INTERACTION_J_MOL}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, temperature, state in zip(
        axes, (HOT_TEMPERATURE_K, COLD_TEMPERATURE_K), ("hot_single", "cold_single"),
    ):
        energies = jax.vmap(lambda x: regular_solution_gibbs_j(
            temperature, PRESSURE_PA, jnp.array([1.0 - x, x]), parameters,
        ))(grid)
        axis.plot(grid, energies / 1000.0, label="Homogeneous solid solution")
        axis.scatter(BULK_AN_FRACTION, float(results[state].gibbs_j) / 1000.0,
                     color="C1", zorder=3, label="Homogeneous An60 bulk")
        axis.set(title=f"{temperature:.0f} K", xlabel="Anorthite mole fraction, x(An)",
                 ylabel="Mixing Gibbs energy (kJ/mol)", xlim=(0, 1))
        axis.grid(alpha=0.2)
    split = results["cold_split"]
    amounts = np.asarray(split.amounts_mol).reshape(2, 2)
    x = amounts[:, 1] / amounts.sum(axis=1)
    coexistence_g = [float(regular_solution_gibbs_j(
        COLD_TEMPERATURE_K, PRESSURE_PA, jnp.array([1.0 - value, value]), parameters,
    )) / 1000.0 for value in x]
    axes[1].plot(x, coexistence_g, "o--", color="C2", label="Two-phase common tangent")
    axes[1].scatter(BULK_AN_FRACTION, float(split.gibbs_j) / 1000.0,
                    marker="*", s=110, color="C2", zorder=3, label="Exsolved An60 bulk")
    for axis in axes:
        axis.legend(fontsize=8, loc="upper center")
    figure.suptitle("Illustrative Ab-An regular solution (not a fitted labradorite model)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/plagioclase_exsolution.png"))
    args = parser.parse_args()
    results = run_example()
    for name, result in results.items():
        amounts = np.asarray(result.amounts_mol).reshape(2, 2)
        totals = amounts.sum(axis=1)
        present = totals > 0.0
        print(f"{name}: x(An)={amounts[present, 1] / totals[present]}, "
              f"phase amounts (mol)={totals[present]}, G={float(result.gibbs_j):.6f} J")
    print("Fixed-support stability status: unchecked; see the tutorial's analytic checks.")
    plot_results(results, args.output)
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
