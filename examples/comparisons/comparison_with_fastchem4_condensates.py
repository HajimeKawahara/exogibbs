"""Visual comparison of production ExoGibbs and FastChem 4 condensation.

The four points are the v0.4 validation-demo conditions.  Both programs read
the same gas thermochemistry, condensate thermochemistry, element abundances,
temperatures, and pressures.  FastChem runs as an independent process and its
output is never supplied to an ExoGibbs constructor, initializer, support
selector, retry, or route decision.

The formal provenance and machine-readable comparison remain in
``benchmarks/fastchem4``; this script exposes the essential solve, alignment,
summary, and plotting steps for a human reader.
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
    condensate_comparison_metrics,
    gas_major_species_metrics,
    occurrence_keys,
)
from benchmarks.fastchem4.fastchem_executable import (
    run_fastchem_executable,
)
from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)
from exogibbs.presets.fastchem4_cond import (
    condensate_chemical_setup,
)
from exogibbs.utils.fastchem_parity import (
    build_aligned_abundance_vector,
)


config.update("jax_enable_x64", True)

DATA_ROOT = REPOSITORY_ROOT / "src" / "exogibbs" / "data" / "FastChem4"
ELEMENT_FILE = DATA_ROOT / "element_abundances" / "asplund_2021.dat"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK_wo_ions.dat"
CONDENSATE_LOGK_FILE = DATA_ROOT / "logK" / "logK_condensates.dat"

TEMPERATURES_K = np.asarray([1800.0, 1600.0, 1400.0, 1200.0])
PRESSURES_BAR = np.full(TEMPERATURES_K.shape, 0.1)
MAJOR_GAS_THRESHOLD = 1.0e-8
ACTIVE_CONDENSATE_FLOOR = 1.0e-8
PLOT_FLOOR = 1.0e-20
GAS_SPECIES = (
    "H2",
    "H2O1",
    "C1O1",
    "C1H4",
    "H3N1",
    "O1Ti1",
)
CONDENSATE_SPECIES = (
    "Al2O3(s,l)",
    "MgAl2O4(s,l)",
    "Fe(s,l)",
    "MgSiO3(s,l)",
    "Mg2SiO4(s,l)",
    "CaMgSi2O6(s)",
    "SiO(s)",
    "Ti3O5(s,l)",
)
DISPLAY_NAMES = {
    "H2": r"H$_2$",
    "H2O1": r"H$_2$O",
    "C1O1": "CO",
    "C1H4": r"CH$_4$",
    "H3N1": r"NH$_3$",
    "O1Ti1": "TiO",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ExoGibbs production equilibrium condensation with an "
            "independent FastChem 4 standalone calculation."
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
            / "comparison_with_fastchem4_condensates.png"
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


def _validate_comparison_contract(*, setup, fastchem, exogibbs) -> None:
    gas_catalog_matches = sorted(
        occurrence_keys(setup.gas_species)
    ) == sorted(occurrence_keys(fastchem.gas_names))
    condensate_catalog_matches = occurrence_keys(
        setup.condensate_species
    ) == occurrence_keys(fastchem.condensate_names)
    if not gas_catalog_matches or not condensate_catalog_matches:
        raise RuntimeError(
            "FastChem and ExoGibbs species catalogs do not match. Check the "
            "FastChem version and shared input files."
        )

    exogibbs_failed = [
        index
        for index, layer in enumerate(exogibbs.layers)
        if not layer.converged
    ]
    fastchem_failed = np.flatnonzero(~fastchem.converged)
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
        exogibbs_failed
        or fastchem_failed.size
        or unconserved.size
        or invalid_gas_density.size
        or invalid_element_density.size
    ):
        raise RuntimeError(
            "Comparison output failed validation: "
            f"ExoGibbs not converged={exogibbs_failed}, "
            f"FastChem not converged={fastchem_failed.tolist()}, "
            f"FastChem elements not conserved={unconserved.tolist()}, "
            f"invalid gas density={invalid_gas_density.tolist()}, "
            "invalid total element density="
            f"{invalid_element_density.tolist()}."
        )


def _print_summary(
    *,
    setup,
    exogibbs_result,
    exogibbs_x: np.ndarray,
    exogibbs_condensates: np.ndarray,
    fastchem,
    fastchem_x: np.ndarray,
    fastchem_condensates: np.ndarray,
) -> None:
    print("Production gas-plus-condensate comparison")
    print(
        " T [K] | status Exo/FC/FC-elements | major gases Exo/FC | "
        "Jaccard | mean/max [dex] | active condensates Exo/FC | Jaccard"
    )
    for layer_index, temperature in enumerate(TEMPERATURES_K):
        gas_metrics = gas_major_species_metrics(
            names=setup.gas_species,
            left_values=exogibbs_x[layer_index],
            right_values=fastchem_x[layer_index],
            threshold=MAJOR_GAS_THRESHOLD,
            excluded_names=("e-",),
        )
        condensate_metrics = condensate_comparison_metrics(
            names=setup.condensate_species,
            left_values=exogibbs_condensates[layer_index],
            right_values=fastchem_condensates[layer_index],
            active_floor=ACTIVE_CONDENSATE_FLOOR,
            ratio_floor=ACTIVE_CONDENSATE_FLOOR,
        )
        exogibbs_status = exogibbs_result.layers[layer_index].status
        fastchem_status = str(fastchem.status[layer_index])
        fastchem_conserved = str(
            fastchem.element_conservation_status[layer_index]
        )
        print(
            f" {temperature:5.0f} | "
            f"{exogibbs_status}/{fastchem_status}/{fastchem_conserved} | "
            f"{gas_metrics['left_major_count']:2d}/"
            f"{gas_metrics['right_major_count']:2d} | "
            f"{gas_metrics['major_set_jaccard']:.3f} | "
            f"{gas_metrics['mean_absolute_log10_ratio']:.3g}/"
            f"{gas_metrics['max_absolute_log10_ratio']:.3g} | "
            f"{condensate_metrics['left_active_count']:2d}/"
            f"{condensate_metrics['right_active_count']:2d} | "
            f"{condensate_metrics['active_set_jaccard']:.3f}"
        )


def _plot_comparison(
    *,
    setup,
    exogibbs_x: np.ndarray,
    exogibbs_condensates: np.ndarray,
    fastchem_x: np.ndarray,
    fastchem_condensates: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    fig, (ax_gas, ax_condensate) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.0),
    )
    gas_colors = plt.get_cmap("tab10")
    cond_colors = plt.get_cmap("tab20")

    for species_index, species in enumerate(GAS_SPECIES):
        if species not in setup.gas_species:
            continue
        slot = setup.gas_species.index(species)
        color = gas_colors(species_index % 10)
        label = DISPLAY_NAMES.get(species, species)
        ax_gas.plot(
            TEMPERATURES_K,
            _positive(fastchem_x[:, slot]),
            color=color,
            linewidth=1.8,
        )
        ax_gas.plot(
            TEMPERATURES_K,
            _positive(exogibbs_x[:, slot]),
            "--o",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    for species_index, species in enumerate(CONDENSATE_SPECIES):
        if species not in setup.condensate_species:
            continue
        slot = setup.condensate_species.index(species)
        color = cond_colors(species_index % 20)
        ax_condensate.plot(
            TEMPERATURES_K,
            _positive(fastchem_condensates[:, slot]),
            color=color,
            linewidth=1.8,
        )
        ax_condensate.plot(
            TEMPERATURES_K,
            _positive(exogibbs_condensates[:, slot]),
            "--o",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=species,
        )

    for axis in (ax_gas, ax_condensate):
        axis.set_yscale("log")
        axis.set_xlabel("Temperature [K]")
        axis.grid(alpha=0.25)
        axis.invert_xaxis()

    ax_gas.set_ylabel("Gas mixing ratio")
    ax_gas.set_title("Gas: FastChem solid; ExoGibbs dashed")
    ax_gas.legend(fontsize=8, ncol=2)

    ax_condensate.set_ylabel(
        "Condensate amount / total element density"
    )
    ax_condensate.set_title(
        "Condensates: FastChem solid; ExoGibbs dashed"
    )
    ax_condensate.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Equilibrium condensation at 0.1 bar",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    # 1. Run FastChem as an independent equilibrium-condensation process.
    fastchem = run_fastchem_executable(
        executable=args.fastchem_executable,
        temperatures=TEMPERATURES_K,
        pressures=PRESSURES_BAR,
        element_abundance_file=ELEMENT_FILE,
        gas_logk_file=GAS_LOGK_FILE,
        condensate_logk_file=CONDENSATE_LOGK_FILE,
        chemistry_mode="equilibrium_condensation",
    )

    # 2. Build and run ExoGibbs only from the shared input files.
    setup = condensate_chemical_setup(
        gas_path="FastChem4/logK/logK_wo_ions.dat",
        condensate_path="FastChem4/logK/logK_condensates.dat",
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
    exogibbs = solve_condensate_profile(
        setup,
        T=jnp.asarray(TEMPERATURES_K, dtype=jnp.float64),
        P=jnp.asarray(PRESSURES_BAR, dtype=jnp.float64),
        b=budget,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )
    jax.block_until_ready(exogibbs.batched_arrays)

    # 3. Align names only after both independent calculations are complete.
    _validate_comparison_contract(
        setup=setup,
        fastchem=fastchem,
        exogibbs=exogibbs,
    )
    fastchem_gas_density = align_species_values(
        setup.gas_species,
        fastchem.gas_names,
        fastchem.gas_number_densities,
    )
    fastchem_condensate_density = align_species_values(
        setup.condensate_species,
        fastchem.condensate_names,
        fastchem.condensate_number_densities,
    )
    fastchem_x = fastchem_gas_density / np.sum(
        fastchem_gas_density,
        axis=1,
        keepdims=True,
    )
    fastchem_condensates = (
        fastchem_condensate_density
        / fastchem.total_element_density[:, None]
    )
    exogibbs_x = np.stack(
        [
            np.asarray(layer.gas_x, dtype=np.float64)
            for layer in exogibbs.layers
        ]
    )
    exogibbs_condensates = np.stack(
        [
            np.asarray(layer.condensate_amounts, dtype=np.float64)
            for layer in exogibbs.layers
        ]
    )

    # 4. Print a compact numerical summary and make the visual comparison.
    _print_summary(
        setup=setup,
        exogibbs_result=exogibbs,
        exogibbs_x=exogibbs_x,
        exogibbs_condensates=exogibbs_condensates,
        fastchem=fastchem,
        fastchem_x=fastchem_x,
        fastchem_condensates=fastchem_condensates,
    )
    _plot_comparison(
        setup=setup,
        exogibbs_x=exogibbs_x,
        exogibbs_condensates=exogibbs_condensates,
        fastchem_x=fastchem_x,
        fastchem_condensates=fastchem_condensates,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
