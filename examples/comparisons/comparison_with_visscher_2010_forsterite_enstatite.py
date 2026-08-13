"""Demonstrate forsterite-enstatite-quartz phase competition.

Two one-bar local-equilibrium scans share the same reduced protosolar gas
catalog and elemental inventory.  Run A allows forsterite, enstatite, and
silica to condense.  Run B removes only enstatite, exposing silica as the
alternative sink for the silicon left after forsterite formation.

The reactive chemistry is the H/O/Mg/Si system used in the condensation
reactions of Visscher et al. (2010).  Inert helium retains its contribution to
total pressure, while C and CO provide the leading protosolar oxygen sink.
Every temperature is solved independently from the same Lodders (2003)
elemental budget.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup


config.update("jax_enable_x64", True)

PRESSURE_BAR = 1.0
DEFAULT_TEMPERATURES_K = np.unique(
    np.concatenate(
        (
            np.arange(1300.0, 2001.0, 10.0),
            np.arange(1540.0, 1621.0, 2.0),
            np.arange(1680.0, 1741.0, 2.0),
        )
    )
)
DEFAULT_FIGURE = (
    REPOSITORY_ROOT
    / "results"
    / "visscher_2010_forsterite_enstatite"
    / "forsterite_enstatite_competition.png"
)

ELEMENTS = ("H", "He", "C", "O", "Mg", "Si")
GAS_SPECIES = ("H2", "He1", "C1O1", "H2O1", "Mg1", "O1Si1")
FORSTERITE = "Mg2SiO4(s,l)"
ENSTATITE = "MgSiO3(s,l)"
QUARTZ = "SiO2(s,l)"
ALL_CONDENSATES = (FORSTERITE, ENSTATITE, QUARTZ)
PHASE_LABELS = {
    FORSTERITE: "Forsterite",
    ENSTATITE: "Enstatite",
    QUARTZ: "SiO2 (quartz diagnostic)",
}
PHASE_COLORS = {
    FORSTERITE: "tab:blue",
    ENSTATITE: "tab:green",
    QUARTZ: "tab:purple",
}


@dataclass(frozen=True)
class CompetitionRun:
    """One explicitly selected condensate catalog."""

    key: str
    label: str
    condensate_species: tuple[str, ...]


RUN_A = CompetitionRun(
    key="with_enstatite",
    label="Run A: enstatite allowed",
    condensate_species=ALL_CONDENSATES,
)
RUN_B = CompetitionRun(
    key="without_enstatite",
    label="Run B: enstatite excluded",
    condensate_species=(FORSTERITE, QUARTZ),
)
COMPETITION_RUNS = (RUN_A, RUN_B)


@dataclass(frozen=True)
class DemoSolution:
    """One reduced-system local-equilibrium temperature scan."""

    temperatures_k: np.ndarray
    gas_ln_n: np.ndarray
    gas_x: np.ndarray
    gas_ntot: np.ndarray
    condensate_amounts: np.ndarray
    converged: np.ndarray
    status: tuple[str, ...]


def forsterite_condensation_temperature(
    pressure_bar: float | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Visscher et al. (2010) forsterite fit in kelvin."""

    pressure = np.asarray(pressure_bar, dtype=np.float64)
    denominator = (
        5.89
        - 0.37 * np.log10(pressure)
        - 0.73 * float(metallicity_dex)
    )
    return 1.0e4 / denominator


def enstatite_condensation_temperature(
    pressure_bar: float | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Visscher et al. (2010) enstatite fit in kelvin."""

    pressure = np.asarray(pressure_bar, dtype=np.float64)
    denominator = (
        6.26
        - 0.35 * np.log10(pressure)
        - 0.70 * float(metallicity_dex)
    )
    return 1.0e4 / denominator


def _subset_hvector(
    function: Callable[[jnp.ndarray], jnp.ndarray],
    indices: tuple[int, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    index_array = jnp.asarray(indices, dtype=jnp.int32)

    @jax.jit
    def hvector(temperature: jnp.ndarray) -> jnp.ndarray:
        return function(temperature)[..., index_array]

    return hvector


def build_reduced_setup(run: CompetitionRun) -> CondensateChemicalSetup:
    """Build the shared gas catalog and one explicit condensate catalog."""

    full = condensate_chemical_setup(
        species_default_elements=False,
        element_file="FastChem4/element_abundances/lodders_2003.dat",
        silent=True,
    )
    element_indices = tuple(full.elements.index(name) for name in ELEMENTS)
    gas_indices = tuple(full.gas_species.index(name) for name in GAS_SPECIES)
    condensate_indices = tuple(
        full.condensate_species.index(name)
        for name in run.condensate_species
    )

    gas_matrix = np.asarray(full.formula_matrix)[
        np.ix_(element_indices, gas_indices)
    ]
    condensate_matrix = np.asarray(full.formula_matrix_cond)[
        np.ix_(element_indices, condensate_indices)
    ]
    reference = np.asarray(full.gas_setup.element_vector_reference)[
        np.asarray(element_indices)
    ]
    reference = reference / np.sum(reference)

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(gas_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.gas_setup.hvector_func,
            gas_indices,
        ),
        elements=ELEMENTS,
        species=GAS_SPECIES,
        element_vector_reference=jnp.asarray(reference, dtype=jnp.float64),
        metadata={
            "source": "FastChem4-format reduced silicate competition demo",
            "element_abundances": "Lodders (2003)",
        },
    )

    validity = full.condensate_setup.temperature_validity_upper
    if validity is None:
        raise ValueError("FastChem4 condensates have no validity bounds.")
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(condensate_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.condensate_setup.hvector_func,
            condensate_indices,
        ),
        elements=ELEMENTS,
        species=run.condensate_species,
        temperature_validity_upper=tuple(
            float(validity[index]) for index in condensate_indices
        ),
        metadata={
            "source": "FastChem4-format silicate condensate records",
            "condensate_catalog_mode": "reduced_explicit",
        },
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def solar_element_budget(setup: CondensateChemicalSetup) -> jnp.ndarray:
    """Return the normalized Lodders (2003) reduced elemental inventory."""

    reference = setup.gas_setup.element_vector_reference
    if reference is None:
        raise ValueError("The reduced setup has no reference element budget.")
    budget = jnp.asarray(reference, dtype=jnp.float64)
    return budget / jnp.sum(budget)


def solve_exogibbs(
    setup: CondensateChemicalSetup,
    temperatures_k: Sequence[float] | np.ndarray,
) -> DemoSolution:
    """Solve independent local equilibria with the production profile API."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    pressures = np.full_like(temperatures, PRESSURE_BAR)
    profile = solve_condensate_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(pressures, dtype=jnp.float64),
        b=solar_element_budget(setup),
        options=CondensateEquilibriumOptions(),
    )
    jax.block_until_ready(profile.batched_arrays)
    solution = DemoSolution(
        temperatures_k=temperatures,
        gas_ln_n=np.stack(
            [np.asarray(layer.gas_ln_n) for layer in profile.layers]
        ),
        gas_x=np.stack(
            [np.asarray(layer.gas_x) for layer in profile.layers]
        ),
        gas_ntot=np.asarray(
            [float(np.asarray(layer.gas_ntot)) for layer in profile.layers]
        ),
        condensate_amounts=np.stack(
            [
                np.asarray(layer.condensate_amounts)
                for layer in profile.layers
            ]
        ),
        converged=np.asarray(
            [layer.converged for layer in profile.layers],
            dtype=bool,
        ),
        status=tuple(layer.status for layer in profile.layers),
    )
    if not np.all(solution.converged):
        failed = np.flatnonzero(~solution.converged).tolist()
        raise RuntimeError(f"ExoGibbs did not converge at profile rows {failed}.")
    return solution


def condensation_bracket(
    temperatures_k: Sequence[float] | np.ndarray,
    amounts: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Bracket a phase appearance using a scale-relative amount floor."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    values = np.asarray(amounts, dtype=np.float64)
    if temperatures.ndim != 1 or values.shape != temperatures.shape:
        raise ValueError("temperatures and amounts must be matching 1D arrays.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("amounts must be finite and non-negative.")
    threshold = float(np.max(values)) * 1.0e-12
    positive = values > threshold
    if not np.any(positive) or np.all(positive):
        raise ValueError("The grid must cross the condensation boundary.")
    cold_edge = float(np.max(temperatures[positive]))
    warmer = temperatures[(temperatures > cold_edge) & ~positive]
    if warmer.size == 0:
        raise ValueError("No warmer uncondensed point brackets the boundary.")
    return cold_edge, float(np.min(warmer))


def phase_element_fractions(
    solution: DemoSolution,
    setup: CondensateChemicalSetup,
    element: str,
) -> dict[str, np.ndarray]:
    """Return the inventory fraction of one element held in each phase."""

    element_index = setup.elements.index(element)
    budget = float(solar_element_budget(setup)[element_index])
    stoichiometry = np.asarray(setup.formula_matrix_cond)[element_index]
    return {
        name: solution.condensate_amounts[:, index]
        * stoichiometry[index]
        / budget
        for index, name in enumerate(setup.condensate_species)
    }


def log_saturation_ratios(
    solution: DemoSolution,
    diagnostic_setup: CondensateChemicalSetup,
) -> np.ndarray:
    """Reconstruct ln(S) for the diagnostic setup from each gas solution."""

    if solution.gas_ln_n.shape[1] != len(diagnostic_setup.gas_species):
        raise ValueError("Solution and diagnostic gas catalogs differ in size.")
    temperatures = solution.temperatures_k
    gas_standard_source = np.stack(
        [
            np.asarray(
                diagnostic_setup.gas_setup.hvector_func(temperature),
                dtype=np.float64,
            )
            for temperature in temperatures
        ]
    )
    gas_stationarity_source = (
        gas_standard_source
        + np.log(PRESSURE_BAR)
        - np.log(solution.gas_ntot)[:, None]
    )
    gas_matrix = np.asarray(
        diagnostic_setup.formula_matrix,
        dtype=np.float64,
    )
    element_potential = np.stack(
        [
            np.linalg.lstsq(
                gas_matrix.T,
                gas_ln_n + source,
                rcond=None,
            )[0]
            for gas_ln_n, source in zip(
                solution.gas_ln_n,
                gas_stationarity_source,
            )
        ]
    )
    condensate_standard_source = np.stack(
        [
            np.asarray(
                diagnostic_setup.condensate_setup.hvector_func(temperature),
                dtype=np.float64,
            )
            for temperature in temperatures
        ]
    )
    condensate_matrix = np.asarray(
        diagnostic_setup.formula_matrix_cond,
        dtype=np.float64,
    )
    return (
        element_potential @ condensate_matrix
        - condensate_standard_source
    )


def make_figure(
    *,
    setups: dict[str, CondensateChemicalSetup],
    solutions: dict[str, DemoSolution],
) -> plt.Figure:
    """Create the phase-allocation, gas-depletion, and activity figure."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.4),
        sharex=True,
        constrained_layout=True,
    )
    ax_run_a, ax_run_b, ax_gas, ax_saturation = axes.flat
    run_a_setup = setups[RUN_A.key]
    temperatures = solutions[RUN_A.key].temperatures_k
    if not np.array_equal(
        temperatures,
        solutions[RUN_B.key].temperatures_k,
    ):
        raise ValueError("Competition runs use different temperature grids.")

    for run, axis in ((RUN_A, ax_run_a), (RUN_B, ax_run_b)):
        setup = setups[run.key]
        fractions = phase_element_fractions(
            solutions[run.key],
            setup,
            "Si",
        )
        for phase in run.condensate_species:
            axis.plot(
                temperatures,
                fractions[phase],
                color=PHASE_COLORS[phase],
                linewidth=2.0,
                label=PHASE_LABELS[phase],
            )
        axis.set_ylim(-0.025, 1.025)
        axis.set_ylabel("Fraction of Si inventory in phase")
        axis.set_title(run.label)
        axis.legend(fontsize=8, loc="best")

    gas_styles = (
        (RUN_A, "-"),
        (RUN_B, "--"),
    )
    for run, linestyle in gas_styles:
        setup = setups[run.key]
        solution = solutions[run.key]
        for species, label, color in (
            ("Mg1", "Mg", "tab:blue"),
            ("O1Si1", "SiO", "tab:orange"),
        ):
            ax_gas.plot(
                temperatures,
                solution.gas_x[:, setup.gas_species.index(species)],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{label}, {run.key.replace('_', ' ')}",
            )
    ax_gas.set_yscale("log")
    ax_gas.set_ylabel("Gas mole fraction")
    ax_gas.set_title("Mg and SiO depletion")
    ax_gas.legend(fontsize=8, loc="best")

    saturation_a = log_saturation_ratios(
        solutions[RUN_A.key],
        run_a_setup,
    )
    saturation_b = log_saturation_ratios(
        solutions[RUN_B.key],
        run_a_setup,
    )
    enstatite_index = run_a_setup.condensate_species.index(ENSTATITE)
    quartz_index = run_a_setup.condensate_species.index(QUARTZ)
    ax_saturation.plot(
        temperatures,
        saturation_a[:, enstatite_index],
        color=PHASE_COLORS[ENSTATITE],
        linewidth=2.0,
        label="Enstatite, Run A",
    )
    ax_saturation.plot(
        temperatures,
        saturation_b[:, enstatite_index],
        color=PHASE_COLORS[ENSTATITE],
        linestyle="--",
        linewidth=2.0,
        label="Enstatite, excluded from Run B",
    )
    ax_saturation.plot(
        temperatures,
        saturation_a[:, quartz_index],
        color=PHASE_COLORS[QUARTZ],
        linewidth=2.0,
        label="SiO2, Run A",
    )
    ax_saturation.plot(
        temperatures,
        saturation_b[:, quartz_index],
        color=PHASE_COLORS[QUARTZ],
        linestyle="--",
        linewidth=2.0,
        label="SiO2, Run B",
    )
    ax_saturation.axhline(0.0, color="black", linewidth=1.0)
    ax_saturation.set_ylabel("ln saturation ratio")
    ax_saturation.set_title("Candidate-phase activity diagnostics")
    ax_saturation.legend(fontsize=7, loc="best")

    forsterite_reference = float(
        forsterite_condensation_temperature(PRESSURE_BAR)
    )
    enstatite_reference = float(
        enstatite_condensation_temperature(PRESSURE_BAR)
    )
    for axis in axes.flat:
        axis.axvline(
            forsterite_reference,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
        )
        axis.axvline(
            enstatite_reference,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
        )
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("Temperature (K)")

    figure.suptitle(
        "Forsterite-enstatite-quartz active-set competition\n"
        "1 bar, reduced Lodders (2003) protosolar system"
    )
    return figure


def print_summary(
    *,
    setups: dict[str, CondensateChemicalSetup],
    solutions: dict[str, DemoSolution],
) -> None:
    """Print transition brackets and the cold-side silicon allocation."""

    print("One-bar forsterite-enstatite-quartz competition")
    for run, phases in (
        (RUN_A, (FORSTERITE, ENSTATITE)),
        (RUN_B, (FORSTERITE, QUARTZ)),
    ):
        setup = setups[run.key]
        solution = solutions[run.key]
        print(f"{run.label}: {np.count_nonzero(solution.converged)}/"
              f"{solution.converged.size} converged")
        for phase in phases:
            index = setup.condensate_species.index(phase)
            cold, warm = condensation_bracket(
                solution.temperatures_k,
                solution.condensate_amounts[:, index],
            )
            print(
                f"  {PHASE_LABELS[phase]} boundary in "
                f"({cold:.1f}, {warm:.1f}] K"
            )
        cold_fractions = phase_element_fractions(solution, setup, "Si")
        allocation = ", ".join(
            f"{PHASE_LABELS[phase]}={cold_fractions[phase][0]:.4f}"
            for phase in run.condensate_species
        )
        print(f"  Si allocation at {solution.temperatures_k[0]:.0f} K: "
              f"{allocation}")
    diagnostic_temperature = 1550.0
    temperature_index = list(
        solutions[RUN_A.key].temperatures_k
    ).index(diagnostic_temperature)
    diagnostic_setup = setups[RUN_A.key]
    saturation_a = np.exp(
        log_saturation_ratios(
            solutions[RUN_A.key],
            diagnostic_setup,
        )[temperature_index]
    )
    saturation_b = np.exp(
        log_saturation_ratios(
            solutions[RUN_B.key],
            diagnostic_setup,
        )[temperature_index]
    )
    print(
        f"At {diagnostic_temperature:.0f} K: Run A SiO2 S="
        f"{saturation_a[diagnostic_setup.condensate_species.index(QUARTZ)]:.3f}, "
        "Run B excluded-enstatite S="
        f"{saturation_b[diagnostic_setup.condensate_species.index(ENSTATITE)]:.3f}"
    )
    print(
        "Literature fits: forsterite="
        f"{float(forsterite_condensation_temperature(PRESSURE_BAR)):.2f} K, "
        "enstatite="
        f"{float(enstatite_condensation_temperature(PRESSURE_BAR)):.2f} K"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate the Visscher et al. (2010) forsterite-enstatite-"
            "quartz active-set competition with ExoGibbs."
        )
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setups = {
        run.key: build_reduced_setup(run) for run in COMPETITION_RUNS
    }
    solutions = {
        run.key: solve_exogibbs(
            setups[run.key],
            DEFAULT_TEMPERATURES_K,
        )
        for run in COMPETITION_RUNS
    }
    print_summary(setups=setups, solutions=solutions)

    figure = make_figure(setups=setups, solutions=solutions)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=200, bbox_inches="tight")
    print(f"figure: {args.figure}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
