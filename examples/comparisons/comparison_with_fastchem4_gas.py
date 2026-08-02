"""Visual comparison of gas-only ExoGibbs and FastChem 4 calculations.

This example is the current FastChem 4 replacement for the historical
``comparison_with_fastchem.py`` and ``comparison_with_fastchem_extended.py``
scripts.  It compares two independent gas-only calculations that read the
same thermochemical and elemental-abundance files.

FastChem output is never used to initialize or configure ExoGibbs.  The
formal provenance and machine-readable comparison remain in
``benchmarks/fastchem4``; this script is intentionally a readable plot demo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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

from benchmarks.fastchem4.comparison import (
    align_species_values,
    element_budget_metrics,
    gas_major_species_metrics,
    occurrence_keys,
)
from benchmarks.fastchem4.fastchem_executable import (
    run_fastchem_executable,
)
from exogibbs.api.gas import (
    EquilibriumOptions,
    solve_profile as solve_gas_profile,
)
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.utils.fastchem_parity import (
    build_aligned_abundance_vector,
)


config.update("jax_enable_x64", True)

DATA_ROOT = REPOSITORY_ROOT / "src" / "exogibbs" / "data" / "FastChem4"
ELEMENT_FILE = DATA_ROOT / "element_abundances" / "asplund_2021.dat"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK_wo_ions.dat"

TEMPERATURE_K = 3000.0
PRESSURES_BAR = np.logspace(-8.0, 2.0, 41)
MAJOR_GAS_THRESHOLD = 1.0e-8
PLOT_FLOOR = 1.0e-30
PLOT_SPECIES = (
    "H2",
    "H1",
    "H2O1",
    "C1O1",
    "C1O2",
    "C1H4",
    "H3N1",
    "O1Ti1",
)
DISPLAY_NAMES = {
    "H1": "H",
    "H2": r"H$_2$",
    "H2O1": r"H$_2$O",
    "C1O1": "CO",
    "C1O2": r"CO$_2$",
    "C1H4": r"CH$_4$",
    "H3N1": r"NH$_3$",
    "O1Ti1": "TiO",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare gas-only ExoGibbs with an independent FastChem 4 "
            "standalone calculation."
        )
    )
    parser.add_argument(
        "--fastchem-executable",
        required=True,
        type=Path,
        help="Path to the FastChem 4 standalone executable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "fastchem4_examples"
            / "comparison_with_fastchem4_gas.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(
        np.isfinite(array) & (array > PLOT_FLOOR),
        array,
        np.nan,
    )


def _validate_comparison_contract(*, setup, fastchem) -> None:
    exogibbs_catalog = sorted(occurrence_keys(setup.species))
    fastchem_catalog = sorted(occurrence_keys(fastchem.gas_names))
    if exogibbs_catalog != fastchem_catalog:
        raise RuntimeError(
            "FastChem and ExoGibbs gas catalogs do not match. Check the "
            "FastChem version and shared input files."
        )

    failed = np.flatnonzero(~fastchem.converged)
    unconserved = np.flatnonzero(~fastchem.elements_conserved)
    gas_density = np.sum(fastchem.gas_number_densities, axis=1)
    invalid_gas_density = np.flatnonzero(
        ~np.isfinite(gas_density) | (gas_density <= 0.0)
    )
    invalid_element_density = np.flatnonzero(
        ~np.isfinite(fastchem.total_element_density)
        | (fastchem.total_element_density <= 0.0)
    )
    if (
        failed.size
        or unconserved.size
        or invalid_gas_density.size
        or invalid_element_density.size
    ):
        raise RuntimeError(
            "FastChem gas-only output failed validation: "
            f"not converged={failed.tolist()}, "
            f"elements not conserved={unconserved.tolist()}, "
            f"invalid gas density={invalid_gas_density.tolist()}, "
            "invalid total element density="
            f"{invalid_element_density.tolist()}."
        )


def _print_summary(
    *,
    setup,
    budget: np.ndarray,
    exogibbs_n: np.ndarray,
    exogibbs_x: np.ndarray,
    exogibbs_converged: np.ndarray,
    fastchem,
    fastchem_amounts: np.ndarray,
    fastchem_x: np.ndarray,
) -> None:
    empty_formula = np.zeros((len(setup.elements), 0), dtype=np.float64)
    empty_amounts = np.zeros((0,), dtype=np.float64)
    layer_rows = []
    for layer_index, pressure in enumerate(PRESSURES_BAR):
        gas_metrics = gas_major_species_metrics(
            names=setup.species,
            left_values=exogibbs_x[layer_index],
            right_values=fastchem_x[layer_index],
            threshold=MAJOR_GAS_THRESHOLD,
            excluded_names=("e-",),
        )
        exogibbs_budget = element_budget_metrics(
            gas_formula_matrix=setup.formula_matrix,
            condensate_formula_matrix=empty_formula,
            gas_amounts=exogibbs_n[layer_index],
            condensate_amounts=empty_amounts,
            target=budget,
            element_names=setup.elements,
        )
        fastchem_budget = element_budget_metrics(
            gas_formula_matrix=setup.formula_matrix,
            condensate_formula_matrix=empty_formula,
            gas_amounts=fastchem_amounts[layer_index],
            condensate_amounts=empty_amounts,
            target=budget,
            element_names=setup.elements,
        )
        layer_rows.append(
            {
                "pressure": pressure,
                "gas": gas_metrics,
                "exogibbs_budget": exogibbs_budget[
                    "max_absolute_relative_residual"
                ],
                "fastchem_budget": fastchem_budget[
                    "max_absolute_relative_residual"
                ],
                "total_gas_relative_difference": (
                    np.sum(exogibbs_n[layer_index])
                    / np.sum(fastchem_amounts[layer_index])
                    - 1.0
                ),
            }
        )

    worst = max(
        layer_rows,
        key=lambda row: row["gas"]["max_absolute_log10_ratio"],
    )
    worst_species = worst["gas"]["top_rows"][0]["name"]
    print("Gas-only comparison")
    print(f"  temperature: {TEMPERATURE_K:g} K")
    print(
        "  pressure range: "
        f"{PRESSURES_BAR.min():.0e}--{PRESSURES_BAR.max():.0e} bar "
        f"({PRESSURES_BAR.size} layers)"
    )
    print(
        "  converged layers: "
        f"ExoGibbs {np.count_nonzero(exogibbs_converged)}/"
        f"{exogibbs_converged.size}; "
        f"FastChem {np.count_nonzero(fastchem.converged)}/"
        f"{fastchem.converged.size}"
    )
    print(
        "  FastChem element-conservation layers: "
        f"{np.count_nonzero(fastchem.elements_conserved)}/"
        f"{fastchem.elements_conserved.size}"
    )
    print(
        "  minimum major-gas set Jaccard: "
        f"{min(row['gas']['major_set_jaccard'] for row in layer_rows):.3f}"
    )
    print(
        "  worst major-gas difference: "
        f"{worst['gas']['max_absolute_log10_ratio']:.3g} dex "
        f"for {worst_species} at {worst['pressure']:.3g} bar"
    )
    print(
        "  maximum absolute relative total-gas difference: "
        f"{max(abs(row['total_gas_relative_difference']) for row in layer_rows):.3g}"
    )
    print(
        "  maximum relative elemental-budget residual (ExoGibbs/FastChem): "
        f"{max(row['exogibbs_budget'] for row in layer_rows):.3g}/"
        f"{max(row['fastchem_budget'] for row in layer_rows):.3g}"
    )


def _plot_comparison(
    *,
    setup,
    exogibbs_x: np.ndarray,
    fastchem_x: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    fig, (ax_abundance, ax_difference) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.0),
        sharey=True,
        gridspec_kw={"width_ratios": (1.6, 1.0)},
    )
    colors = plt.get_cmap("tab10")

    plotted = 0
    for species_index, species in enumerate(PLOT_SPECIES):
        if species not in setup.species:
            continue
        slot = setup.species.index(species)
        exogibbs_values = _positive(exogibbs_x[:, slot])
        fastchem_values = _positive(fastchem_x[:, slot])
        if not np.any(np.isfinite(exogibbs_values)) and not np.any(
            np.isfinite(fastchem_values)
        ):
            continue

        color = colors(species_index % 10)
        label = DISPLAY_NAMES.get(species, species)
        ax_abundance.plot(
            fastchem_values,
            PRESSURES_BAR,
            color=color,
            linewidth=1.8,
        )
        ax_abundance.plot(
            exogibbs_values,
            PRESSURES_BAR,
            "--",
            color=color,
            linewidth=1.8,
            label=label,
        )
        absolute_dex = np.abs(
            np.log10(
                np.clip(exogibbs_x[:, slot], PLOT_FLOOR, None)
                / np.clip(fastchem_x[:, slot], PLOT_FLOOR, None)
            )
        )
        ax_difference.plot(
            absolute_dex,
            PRESSURES_BAR,
            color=color,
            linewidth=1.6,
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("None of the requested gas species could be plotted.")

    ax_abundance.set_xscale("log")
    ax_abundance.set_yscale("log")
    ax_abundance.invert_yaxis()
    ax_abundance.set_xlabel("Gas mixing ratio")
    ax_abundance.set_ylabel("Pressure [bar]")
    ax_abundance.set_title("FastChem solid; ExoGibbs dashed")
    ax_abundance.legend(fontsize=8, ncol=2)
    ax_abundance.grid(alpha=0.25)

    ax_difference.set_xscale("log")
    ax_difference.set_xlabel(r"Absolute difference $|\Delta\log_{10} x|$ [dex]")
    ax_difference.set_title("Species-by-species difference")
    ax_difference.grid(alpha=0.25)

    fig.suptitle(
        f"Gas-only equilibrium at {TEMPERATURE_K:g} K",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    temperatures = np.full(PRESSURES_BAR.shape, TEMPERATURE_K)

    # 1. Run FastChem as an independent gas-only process.
    fastchem = run_fastchem_executable(
        executable=args.fastchem_executable,
        temperatures=temperatures,
        pressures=PRESSURES_BAR,
        element_abundance_file=ELEMENT_FILE,
        gas_logk_file=GAS_LOGK_FILE,
        chemistry_mode="gas",
    )

    # 2. Build and run ExoGibbs only from the shared input files.
    setup = chemsetup(
        path="FastChem4/logK/logK_wo_ions.dat",
        species_default_elements=False,
        element_file="FastChem4/element_abundances/asplund_2021.dat",
        silent=True,
    )
    aligned_abundance = build_aligned_abundance_vector(
        setup.elements,
        source="fastchem_file",
        normalize=True,
        element_file=ELEMENT_FILE,
    )
    budget = jnp.asarray(aligned_abundance.vector, dtype=jnp.float64)
    exogibbs, diagnostics = solve_gas_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(PRESSURES_BAR, dtype=jnp.float64),
        b=budget,
        options=EquilibriumOptions(),
        return_diagnostics=True,
    )
    jax.block_until_ready((exogibbs.x, diagnostics["converged"]))

    # 3. Align names only after both independent calculations are complete.
    exogibbs_n = np.asarray(exogibbs.n, dtype=np.float64)
    exogibbs_x = np.asarray(exogibbs.x, dtype=np.float64)
    exogibbs_converged = np.asarray(
        diagnostics["converged"],
        dtype=bool,
    )
    if not np.all(exogibbs_converged):
        failed = np.flatnonzero(~exogibbs_converged).tolist()
        raise RuntimeError(
            f"ExoGibbs gas-only solve did not converge at layers {failed}."
        )
    _validate_comparison_contract(setup=setup, fastchem=fastchem)

    fastchem_density = align_species_values(
        setup.species,
        fastchem.gas_names,
        fastchem.gas_number_densities,
    )
    fastchem_x = fastchem_density / np.sum(
        fastchem_density,
        axis=1,
        keepdims=True,
    )
    fastchem_amounts = (
        fastchem_density / fastchem.total_element_density[:, None]
    )

    # 4. Print a compact numerical summary and make the visual comparison.
    _print_summary(
        setup=setup,
        budget=np.asarray(budget, dtype=np.float64),
        exogibbs_n=exogibbs_n,
        exogibbs_x=exogibbs_x,
        exogibbs_converged=exogibbs_converged,
        fastchem=fastchem,
        fastchem_amounts=fastchem_amounts,
        fastchem_x=fastchem_x,
    )
    _plot_comparison(
        setup=setup,
        exogibbs_x=exogibbs_x,
        fastchem_x=fastchem_x,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
