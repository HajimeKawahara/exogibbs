"""Contrast local equilibrium and sequential rainout in an Fe--FeS system.

Both calculations use the same one-bar temperature profile, reduced H/Fe/S
species catalog, and Lodders (2003) elemental inventory.  Local equilibrium
solves every layer from the original inventory, so condensed iron remains
available to form FeS at low temperature.  Sequential rainout instead starts
at the hot lower boundary and passes only the post-condensation elemental
inventory upward.  Iron is therefore removed before FeS becomes stable and
H2S remains in the gas.

The profile API uses top-to-bottom input order.  Temperatures are consequently
stored cold-to-hot here; ``rainout=True`` processes them in reverse.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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
IRON_MELTING_TEMPERATURE_K = 1809.0
PLOT_FLOOR = 1.0e-30
DEFAULT_TEMPERATURES_K = np.asarray(
    [
        500.0,
        600.0,
        650.0,
        700.0,
        900.0,
        1100.0,
        1300.0,
        1500.0,
        1650.0,
        1750.0,
        1800.0,
        1825.0,
        1850.0,
        1875.0,
        1950.0,
        2100.0,
        2200.0,
    ],
    dtype=np.float64,
)
DEFAULT_FIGURE = (
    REPOSITORY_ROOT
    / "results"
    / "fe_fes_rainout"
    / "fe_fes_rainout_demo.png"
)

ELEMENTS = ("H", "Fe", "S")
GAS_SPECIES = ("H1", "Fe1", "S1", "H2", "H2S1")
IRON = "Fe(s,l)"
IRON_SULFIDE = "FeS(s,l)"
CONDENSATE_SPECIES = (IRON, IRON_SULFIDE)


@dataclass(frozen=True)
class DemoSolution:
    """One local-equilibrium or rainout profile."""

    temperatures_k: np.ndarray
    gas_n: np.ndarray
    gas_x: np.ndarray
    condensate_amounts: np.ndarray
    element_inventory_target: np.ndarray
    element_inventory_out: np.ndarray
    converged: np.ndarray
    status: tuple[str, ...]
    method: str
    rainout: bool


def iron_condensation_temperature(
    pressure_bar: float | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Visscher et al. (2010) iron fit in kelvin."""

    pressure = np.asarray(pressure_bar, dtype=np.float64)
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("pressure_bar must be finite and positive.")
    denominator = (
        5.44
        - 0.48 * np.log10(pressure)
        - 0.48 * float(metallicity_dex)
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


def build_reduced_setup() -> CondensateChemicalSetup:
    """Build the minimal packaged H/Fe/S gas and condensate system."""

    full = condensate_chemical_setup(
        species_default_elements=False,
        element_file="FastChem4/element_abundances/lodders_2003.dat",
        silent=True,
    )
    element_indices = tuple(full.elements.index(name) for name in ELEMENTS)
    gas_indices = tuple(full.gas_species.index(name) for name in GAS_SPECIES)
    condensate_indices = tuple(
        full.condensate_species.index(name) for name in CONDENSATE_SPECIES
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
            "source": "FastChem4-format reduced Fe-FeS rainout demo",
            "element_abundances": "Lodders (2003)",
        },
    )

    validity = full.condensate_setup.temperature_validity_upper
    if validity is None:
        raise ValueError("FastChem4 condensates have no validity bounds.")
    reduced_condensates = ChemicalSetup(
        formula_matrix=jnp.asarray(condensate_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.condensate_setup.hvector_func,
            condensate_indices,
        ),
        elements=ELEMENTS,
        species=CONDENSATE_SPECIES,
        temperature_validity_upper=tuple(
            float(validity[index]) for index in condensate_indices
        ),
        metadata={
            "source": "FastChem4/JANAF reduced Fe-FeS condensates",
            "condensate_catalog_mode": "reduced_explicit",
        },
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=reduced_condensates,
    )


def solar_element_budget(setup: CondensateChemicalSetup) -> jnp.ndarray:
    """Return the normalized Lodders (2003) H/Fe/S inventory."""

    reference = setup.gas_setup.element_vector_reference
    if reference is None:
        raise ValueError("The reduced setup has no reference element budget.")
    budget = jnp.asarray(reference, dtype=jnp.float64)
    return budget / jnp.sum(budget)


def solve_exogibbs(
    setup: CondensateChemicalSetup,
    temperatures_k: Sequence[float] | np.ndarray,
    *,
    rainout: bool,
) -> DemoSolution:
    """Solve one cold-to-hot input profile with the production API."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    if (
        temperatures.ndim != 1
        or temperatures.size == 0
        or np.any(~np.isfinite(temperatures))
        or np.any(np.diff(temperatures) <= 0.0)
    ):
        raise ValueError(
            "temperatures_k must be a finite, strictly increasing 1D array."
        )
    pressures = np.full_like(temperatures, PRESSURE_BAR)
    budget = solar_element_budget(setup)
    options = CondensateEquilibriumOptions(
        rainout=rainout,
        profile_method="scan_hot_from_bottom" if rainout else None,
    )
    profile = solve_condensate_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(pressures, dtype=jnp.float64),
        b=budget,
        options=options,
    )
    jax.block_until_ready(profile.batched_arrays)

    fixed_inventory = np.repeat(
        np.asarray(budget, dtype=np.float64)[None, :],
        temperatures.size,
        axis=0,
    )
    target = (
        fixed_inventory
        if profile.element_inventory_target is None
        else np.asarray(profile.element_inventory_target, dtype=np.float64)
    )
    inventory_out = (
        fixed_inventory
        if profile.rainout_element_inventory_out is None
        else np.asarray(
            profile.rainout_element_inventory_out,
            dtype=np.float64,
        )
    )
    solution = DemoSolution(
        temperatures_k=temperatures,
        gas_n=np.stack([np.asarray(layer.gas_n) for layer in profile.layers]),
        gas_x=np.stack([np.asarray(layer.gas_x) for layer in profile.layers]),
        condensate_amounts=np.stack(
            [np.asarray(layer.condensate_amounts) for layer in profile.layers]
        ),
        element_inventory_target=target,
        element_inventory_out=inventory_out,
        converged=np.asarray(
            [layer.converged for layer in profile.layers],
            dtype=bool,
        ),
        status=tuple(layer.status for layer in profile.layers),
        method=str(profile.method),
        rainout=bool(profile.rainout),
    )
    if not np.all(solution.converged):
        failed = np.flatnonzero(~solution.converged).tolist()
        raise RuntimeError(f"ExoGibbs did not converge at profile rows {failed}.")
    return solution


def condensed_element_fractions(
    solution: DemoSolution,
    setup: CondensateChemicalSetup,
    element: str,
) -> dict[str, np.ndarray]:
    """Return per-layer condensate amounts relative to the initial element."""

    element_index = setup.elements.index(element)
    initial = float(solar_element_budget(setup)[element_index])
    stoichiometry = np.asarray(setup.formula_matrix_cond)[element_index]
    return {
        name: solution.condensate_amounts[:, index]
        * stoichiometry[index]
        / initial
        for index, name in enumerate(setup.condensate_species)
    }


def condensation_bracket(
    temperatures_k: Sequence[float] | np.ndarray,
    amounts: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Bracket the warm edge of a condensate using a relative amount floor."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    values = np.asarray(amounts, dtype=np.float64)
    if temperatures.ndim != 1 or values.shape != temperatures.shape:
        raise ValueError("temperatures and amounts must be matching 1D arrays.")
    threshold = float(np.max(values)) * 1.0e-12
    positive = values > threshold
    if not np.any(positive) or np.all(positive):
        raise ValueError("The grid must cross the condensation boundary.")
    cold_edge = float(np.max(temperatures[positive]))
    warmer = temperatures[(temperatures > cold_edge) & ~positive]
    if warmer.size == 0:
        raise ValueError("No warmer uncondensed point brackets the boundary.")
    return cold_edge, float(np.min(warmer))


def make_figure(
    *,
    setup: CondensateChemicalSetup,
    local: DemoSolution,
    rainout: DemoSolution,
) -> plt.Figure:
    """Plot condensates, H2S, and the inventory propagated by rainout."""

    if not np.array_equal(local.temperatures_k, rainout.temperatures_k):
        raise ValueError("Local and rainout solutions use different grids.")
    temperatures = local.temperatures_k
    local_fe = condensed_element_fractions(local, setup, "Fe")
    rainout_fe = condensed_element_fractions(rainout, setup, "Fe")
    h2s_index = setup.gas_species.index("H2S1")
    iron_index = setup.elements.index("Fe")
    sulfur_index = setup.elements.index("S")
    initial = np.asarray(solar_element_budget(setup), dtype=np.float64)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.8),
        sharex=True,
        constrained_layout=True,
    )
    ax_condensate, ax_h2s, ax_inventory = axes
    phase_colors = {IRON: "tab:gray", IRON_SULFIDE: "tab:orange"}
    phase_labels = {IRON: "Fe(s,l)", IRON_SULFIDE: "FeS(s,l)"}
    for phase in CONDENSATE_SPECIES:
        ax_condensate.plot(
            temperatures,
            local_fe[phase],
            color=phase_colors[phase],
            linewidth=2.1,
            label=f"{phase_labels[phase]}, local",
        )
        ax_condensate.plot(
            temperatures,
            rainout_fe[phase],
            color=phase_colors[phase],
            linestyle="--",
            marker="o",
            markersize=3.2,
            linewidth=1.7,
            label=f"{phase_labels[phase]}, rainout layer",
        )
    ax_condensate.set_ylim(-0.025, 1.025)
    ax_condensate.set_ylabel("Condensate Fe / initial Fe (solver gauge)")
    ax_condensate.set_title("Condensate formed in each layer")
    ax_condensate.legend(fontsize=7.5, loc="best")

    ax_h2s.plot(
        temperatures,
        local.gas_x[:, h2s_index],
        color="tab:blue",
        linewidth=2.1,
        label="Local equilibrium",
    )
    ax_h2s.plot(
        temperatures,
        rainout.gas_x[:, h2s_index],
        color="tab:red",
        linestyle="--",
        marker="o",
        markersize=3.2,
        linewidth=1.7,
        label="Sequential rainout",
    )
    ax_h2s.set_yscale("log")
    ax_h2s.set_ylabel("H2S mole fraction in reduced gas")
    ax_h2s.set_title("Sulfur left in the gas")
    ax_h2s.legend(fontsize=8, loc="best")

    ax_inventory.plot(
        temperatures,
        np.clip(
            rainout.element_inventory_target[:, iron_index]
            / initial[iron_index],
            PLOT_FLOOR,
            None,
        ),
        color="tab:gray",
        linewidth=2.1,
        label="Fe target",
    )
    ax_inventory.plot(
        temperatures,
        np.clip(
            rainout.element_inventory_target[:, sulfur_index]
            / initial[sulfur_index],
            PLOT_FLOOR,
            None,
        ),
        color="tab:purple",
        linewidth=2.1,
        label="S target",
    )
    ax_inventory.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="Local fixed target",
    )
    ax_inventory.set_yscale("log")
    ax_inventory.set_ylim(PLOT_FLOOR, 2.0)
    ax_inventory.set_ylabel("Target / initial element inventory")
    ax_inventory.set_title("Inventory entering each layer")
    ax_inventory.legend(fontsize=8, loc="best")

    iron_reference = float(iron_condensation_temperature(PRESSURE_BAR))
    for axis in axes:
        axis.axvline(
            iron_reference,
            color="0.35",
            linestyle="--",
            linewidth=1.1,
        )
        axis.axvline(
            IRON_MELTING_TEMPERATURE_K,
            color="0.55",
            linestyle=":",
            linewidth=1.1,
        )
        axis.set_xlim(float(np.max(temperatures)), float(np.min(temperatures)))
        axis.set_xlabel("Temperature (K; hot bottom to cold top)")
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Fe-FeS local equilibrium versus sequential rainout\n"
        "1 bar, reduced Lodders (2003) H/Fe/S system"
    )
    return figure


def print_summary(
    *,
    setup: CondensateChemicalSetup,
    local: DemoSolution,
    rainout: DemoSolution,
) -> None:
    """Print the transition brackets and cold-layer rainout contrast."""

    iron_index = setup.condensate_species.index(IRON)
    fes_index = setup.condensate_species.index(IRON_SULFIDE)
    h2s_index = setup.gas_species.index("H2S1")
    element_iron_index = setup.elements.index("Fe")
    initial_iron = float(solar_element_budget(setup)[element_iron_index])
    cold_index = int(np.argmin(local.temperatures_k))
    cold_rainout_iron_fraction = (
        rainout.element_inventory_target[cold_index, element_iron_index]
        / initial_iron
    )
    iron_bracket = condensation_bracket(
        local.temperatures_k,
        local.condensate_amounts[:, iron_index],
    )
    fes_bracket = condensation_bracket(
        local.temperatures_k,
        local.condensate_amounts[:, fes_index],
    )

    print("One-bar Fe-FeS local-equilibrium/rainout demonstration")
    print(
        f"local: {np.count_nonzero(local.converged)}/"
        f"{local.converged.size} converged ({local.method})"
    )
    print(
        f"rainout: {np.count_nonzero(rainout.converged)}/"
        f"{rainout.converged.size} converged ({rainout.method})"
    )
    print(
        f"local Fe(s,l) warm edge in ({iron_bracket[0]:.0f}, "
        f"{iron_bracket[1]:.0f}] K"
    )
    print(
        f"local FeS(s,l) warm edge in ({fes_bracket[0]:.0f}, "
        f"{fes_bracket[1]:.0f}] K"
    )
    print(
        "Visscher et al. Fe fit: "
        f"{float(iron_condensation_temperature(PRESSURE_BAR)):.2f} K; "
        f"JANAF Fe phase boundary: {IRON_MELTING_TEMPERATURE_K:.0f} K"
    )
    print(
        f"At {local.temperatures_k[cold_index]:.0f} K: "
        f"local FeS={local.condensate_amounts[cold_index, fes_index]:.3e}, "
        f"rainout FeS={rainout.condensate_amounts[cold_index, fes_index]:.3e}"
    )
    print(
        f"At {local.temperatures_k[cold_index]:.0f} K: "
        f"local H2S={local.gas_x[cold_index, h2s_index]:.3e}, "
        f"rainout H2S={rainout.gas_x[cold_index, h2s_index]:.3e}"
    )
    print(
        "Cold rainout Fe target / initial Fe="
        f"{cold_rainout_iron_fraction:.3e}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local equilibrium and sequential rainout in a reduced "
            "one-bar Fe-FeS system."
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
    setup = build_reduced_setup()
    local = solve_exogibbs(
        setup,
        DEFAULT_TEMPERATURES_K,
        rainout=False,
    )
    rainout = solve_exogibbs(
        setup,
        DEFAULT_TEMPERATURES_K,
        rainout=True,
    )
    print_summary(setup=setup, local=local, rainout=rainout)

    figure = make_figure(setup=setup, local=local, rainout=rainout)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=200, bbox_inches="tight")
    print(f"figure: {args.figure}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
