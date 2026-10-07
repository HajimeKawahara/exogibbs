"""Visual comparison of production ExoGibbs and FastChem 4 condensation.

The default four points are the v0.4 validation-demo conditions.  The optional
``l-dwarf`` profile adds a pressure-profile visualization with gas and
condensate rows and FastChem and ExoGibbs columns.  Its gas panels overlay
gas-only equilibrium and the gas phase in local equilibrium with condensates.
It is an illustrative local-equilibrium trajectory, not a self-consistent
atmosphere or cloud model.

Both programs read the same gas thermochemistry, condensate thermochemistry,
element abundances, temperatures, and pressures.  FastChem runs as an
independent process and its output is never supplied to an ExoGibbs
constructor, initializer, support selector, retry, or route decision.

The formal provenance and machine-readable comparison remain in
``benchmarks/fastchem4``; this script exposes the essential solve, alignment,
summary, and plotting steps for a human reader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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
from exogibbs.api.gas import (
    EquilibriumOptions as GasEquilibriumOptions,
    solve_profile as solve_gas_profile,
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

VALIDATION_TEMPERATURES_K = np.asarray([1800.0, 1600.0, 1400.0, 1200.0])
VALIDATION_PRESSURES_BAR = np.full(VALIDATION_TEMPERATURES_K.shape, 0.1)
# Smooth illustrative warm-substellar trajectory. It is deliberately analytic,
# so it cannot be mistaken for a specific radiative-convective atmosphere.
L_DWARF_PRESSURES_BAR = np.logspace(-4.0, 2.0, 13)
L_DWARF_LOG_PRESSURE_COORDINATE = np.log10(L_DWARF_PRESSURES_BAR) + 4.0
L_DWARF_TEMPERATURES_K = (
    1100.0
    + 160.0 * L_DWARF_LOG_PRESSURE_COORDINATE
    + 15.0 * L_DWARF_LOG_PRESSURE_COORDINATE**2
)
PROFILE_CHOICES = ("validation", "l-dwarf")
MAJOR_GAS_THRESHOLD = 1.0e-8
ACTIVE_CONDENSATE_FLOOR = 1.0e-8
GAS_AGREEMENT_TOLERANCE_DEX = 1.0e-3
CONDENSATE_AGREEMENT_TOLERANCE_DEX = 1.0e-3
GAS_PLOT_FLOOR = 1.0e-20
CONDENSATE_PLOT_FLOOR = 1.0e-20
GAS_SPECIES = (
    "H2",
    "H2O1",
    "C1O1",
    "C1H4",
    "O1Ti1",
    "O1Si1",
    "Mg1",
    "Fe1",
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
    "O1Ti1": "TiO",
    "O1Si1": "SiO",
    "Mg1": "Mg",
    "Fe1": "Fe",
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
        "--profile",
        choices=PROFILE_CHOICES,
        default="validation",
        help=(
            "Conditions and plot layout: the v0.4 validation points or an "
            "illustrative L-dwarf-like pressure profile."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path; the filename defaults to the selected profile.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _profile_conditions(profile: str) -> tuple[np.ndarray, np.ndarray]:
    if profile == "validation":
        return (
            VALIDATION_TEMPERATURES_K.copy(),
            VALIDATION_PRESSURES_BAR.copy(),
        )
    if profile == "l-dwarf":
        return L_DWARF_TEMPERATURES_K.copy(), L_DWARF_PRESSURES_BAR.copy()
    raise ValueError(f"Unknown comparison profile: {profile!r}")


def _default_output_path(profile: str) -> Path:
    filename = (
        "comparison_with_fastchem4_condensates.png"
        if profile == "validation"
        else "comparison_with_fastchem4_ldwarf_profile.png"
    )
    return REPOSITORY_ROOT / "results" / "fastchem4_examples" / filename


def _positive(values: np.ndarray, *, floor: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(
        np.isfinite(array) & (array > floor),
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


def _validate_gas_only_comparison_contract(
    *,
    setup,
    fastchem,
    exogibbs_x: np.ndarray,
    exogibbs_converged: np.ndarray,
) -> None:
    gas_catalog_matches = sorted(
        occurrence_keys(setup.gas_species)
    ) == sorted(occurrence_keys(fastchem.gas_names))
    if not gas_catalog_matches:
        raise RuntimeError(
            "FastChem and ExoGibbs gas-only catalogs do not match. Check the "
            "FastChem version and shared input files."
        )

    expected_shape = (exogibbs_converged.size, len(setup.gas_species))
    if exogibbs_x.shape != expected_shape:
        raise RuntimeError(
            "ExoGibbs gas-only mixing ratios have shape "
            f"{exogibbs_x.shape}; expected {expected_shape}."
        )

    exogibbs_failed = np.flatnonzero(~exogibbs_converged)
    invalid_exogibbs_x = np.flatnonzero(
        np.any(~np.isfinite(exogibbs_x) | (exogibbs_x < 0.0), axis=1)
        | (np.sum(exogibbs_x, axis=1) <= 0.0)
    )
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
        exogibbs_failed.size
        or invalid_exogibbs_x.size
        or fastchem_failed.size
        or unconserved.size
        or invalid_gas_density.size
        or invalid_element_density.size
    ):
        raise RuntimeError(
            "Gas-only comparison output failed validation: "
            f"ExoGibbs not converged={exogibbs_failed.tolist()}, "
            f"invalid ExoGibbs mixing ratios={invalid_exogibbs_x.tolist()}, "
            f"FastChem not converged={fastchem_failed.tolist()}, "
            f"FastChem elements not conserved={unconserved.tolist()}, "
            f"invalid gas density={invalid_gas_density.tolist()}, "
            "invalid total element density="
            f"{invalid_element_density.tolist()}."
        )


def _validate_gas_release_metrics(
    gas_metrics: dict,
    *,
    comparison_label: str,
) -> None:
    """Apply the shared major-gas release gate to one comparison row."""

    values = np.asarray(
        (
            gas_metrics["major_set_jaccard"],
            gas_metrics["mean_absolute_log10_ratio"],
            gas_metrics["max_absolute_log10_ratio"],
        ),
        dtype=np.float64,
    )
    finite = gas_metrics["finite"]
    if type(finite) is not bool or not finite:
        raise RuntimeError(
            f"The {comparison_label} comparison contains non-finite amounts."
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"The {comparison_label} comparison produced non-finite metrics."
        )
    if gas_metrics["major_set_jaccard"] < 1.0:
        raise RuntimeError(
            f"{comparison_label}: FastChem and ExoGibbs disagree on the "
            "major-gas set: "
            f"Jaccard={gas_metrics['major_set_jaccard']:.6g}."
        )
    if (
        gas_metrics["max_absolute_log10_ratio"]
        > GAS_AGREEMENT_TOLERANCE_DEX
    ):
        raise RuntimeError(
            f"{comparison_label}: FastChem and ExoGibbs gas abundances differ by "
            f"{gas_metrics['max_absolute_log10_ratio']:.6g} dex; limit "
            f"{GAS_AGREEMENT_TOLERANCE_DEX:.1e} dex."
        )


def _validate_release_metrics(
    *,
    gas_metrics: dict,
    condensate_metrics: dict,
) -> None:
    """Require the reported cross-code metrics to pass the release gate."""

    _validate_gas_release_metrics(
        gas_metrics,
        comparison_label="gas-plus-condensate",
    )
    values = np.asarray(
        (
            condensate_metrics["active_set_jaccard"],
            condensate_metrics["max_absolute_log10_ratio"],
        ),
        dtype=np.float64,
    )
    finite = condensate_metrics["finite"]
    if type(finite) is not bool or not finite:
        raise RuntimeError(
            "The gas-plus-condensate comparison contains non-finite amounts."
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "The gas-plus-condensate comparison produced non-finite metrics."
        )
    if condensate_metrics["active_set_jaccard"] < 1.0:
        raise RuntimeError(
            "FastChem and ExoGibbs disagree on the active condensate set: "
            f"Jaccard={condensate_metrics['active_set_jaccard']:.6g}."
        )
    if (
        condensate_metrics["max_absolute_log10_ratio"]
        > CONDENSATE_AGREEMENT_TOLERANCE_DEX
    ):
        raise RuntimeError(
            "FastChem and ExoGibbs condensate amounts differ by "
            f"{condensate_metrics['max_absolute_log10_ratio']:.6g} dex; "
            f"limit {CONDENSATE_AGREEMENT_TOLERANCE_DEX:.1e} dex."
        )


def _validate_gas_profile_release_metrics(
    *,
    names: Sequence[str],
    left_values: np.ndarray,
    right_values: np.ndarray,
    comparison_label: str,
) -> None:
    """Apply the shared gas release gate across a complete profile."""

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    expected_columns = len(names)
    if (
        left.ndim != 2
        or left.shape[0] == 0
        or left.shape[1] != expected_columns
        or right.shape != left.shape
    ):
        raise RuntimeError(
            f"The {comparison_label} comparison has invalid profile shapes."
        )
    for left_row, right_row in zip(left, right):
        metrics = gas_major_species_metrics(
            names=names,
            left_values=left_row,
            right_values=right_row,
            threshold=MAJOR_GAS_THRESHOLD,
            excluded_names=("e-",),
        )
        _validate_gas_release_metrics(
            metrics,
            comparison_label=comparison_label,
        )


def _print_summary(
    *,
    profile: str,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    setup,
    exogibbs_result,
    exogibbs_x: np.ndarray,
    exogibbs_condensates: np.ndarray,
    fastchem,
    fastchem_x: np.ndarray,
    fastchem_condensates: np.ndarray,
) -> None:
    print("Production gas-plus-condensate comparison")
    print(f"  profile: {profile}")
    if profile == "l-dwarf":
        print(
            "  interpretation: illustrative local equilibrium without "
            "rainout or cloud transport"
        )
    print(
        " T [K] | P [bar] | status Exo/FC/FC-elements | major gases Exo/FC | "
        "Jaccard | mean/max [dex] | active condensates Exo/FC | Jaccard"
    )
    for layer_index, (temperature, pressure) in enumerate(
        zip(temperatures, pressures)
    ):
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
        _validate_release_metrics(
            gas_metrics=gas_metrics,
            condensate_metrics=condensate_metrics,
        )
        exogibbs_status = exogibbs_result.layers[layer_index].status
        fastchem_status = str(fastchem.status[layer_index])
        fastchem_conserved = str(
            fastchem.element_conservation_status[layer_index]
        )
        print(
            f" {temperature:5.0f} | {pressure:7.1e} | "
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
    print(
        "  release gate: passed "
        f"(gas <= {GAS_AGREEMENT_TOLERANCE_DEX:.1e} dex; condensate <= "
        f"{CONDENSATE_AGREEMENT_TOLERANCE_DEX:.1e} dex; "
        "exact major and active sets"
        f"{' including gas-only profile' if profile == 'l-dwarf' else ''})"
    )


def _plot_validation_comparison(
    *,
    temperatures: np.ndarray,
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
            temperatures,
            _positive(fastchem_x[:, slot], floor=GAS_PLOT_FLOOR),
            color=color,
            linewidth=1.8,
        )
        ax_gas.plot(
            temperatures,
            _positive(exogibbs_x[:, slot], floor=GAS_PLOT_FLOOR),
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
            temperatures,
            _positive(
                fastchem_condensates[:, slot],
                floor=CONDENSATE_PLOT_FLOOR,
            ),
            color=color,
            linewidth=1.8,
        )
        ax_condensate.plot(
            temperatures,
            _positive(
                exogibbs_condensates[:, slot],
                floor=CONDENSATE_PLOT_FLOOR,
            ),
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


def _make_l_dwarf_profile_figure(
    *,
    pressures: np.ndarray,
    temperatures: np.ndarray,
    setup,
    exogibbs_gas_only_x: np.ndarray,
    exogibbs_x: np.ndarray,
    exogibbs_condensates: np.ndarray,
    fastchem_gas_only_x: np.ndarray,
    fastchem_x: np.ndarray,
    fastchem_condensates: np.ndarray,
) -> Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        sharex="row",
        sharey=True,
    )
    fastchem_gas_axis, exogibbs_gas_axis = axes[0]
    fastchem_cond_axis, exogibbs_cond_axis = axes[1]
    gas_colors = plt.get_cmap("tab10")
    condensate_colors = plt.get_cmap("tab20")

    plotted_gases = 0
    for species_index, species in enumerate(GAS_SPECIES):
        if species not in setup.gas_species:
            continue
        slot = setup.gas_species.index(species)
        color = gas_colors(species_index % 10)
        label = DISPLAY_NAMES.get(species, species)
        fastchem_gas_axis.plot(
            _positive(fastchem_x[:, slot], floor=GAS_PLOT_FLOOR),
            pressures,
            "-o",
            color=color,
            linewidth=1.8,
            markersize=3.0,
            label=label,
        )
        fastchem_gas_axis.plot(
            _positive(fastchem_gas_only_x[:, slot], floor=GAS_PLOT_FLOOR),
            pressures,
            "--",
            color=color,
            linewidth=1.3,
            alpha=0.9,
        )
        exogibbs_gas_axis.plot(
            _positive(exogibbs_x[:, slot], floor=GAS_PLOT_FLOOR),
            pressures,
            "-o",
            color=color,
            linewidth=1.8,
            markersize=3.0,
        )
        exogibbs_gas_axis.plot(
            _positive(exogibbs_gas_only_x[:, slot], floor=GAS_PLOT_FLOOR),
            pressures,
            "--",
            color=color,
            linewidth=1.3,
            alpha=0.9,
        )
        plotted_gases += 1

    plotted_condensates = 0
    for species_index, species in enumerate(CONDENSATE_SPECIES):
        if species not in setup.condensate_species:
            continue
        slot = setup.condensate_species.index(species)
        color = condensate_colors(species_index % 20)
        fastchem_cond_axis.plot(
            _positive(
                fastchem_condensates[:, slot],
                floor=CONDENSATE_PLOT_FLOOR,
            ),
            pressures,
            "-o",
            color=color,
            linewidth=1.6,
            markersize=3.0,
            label=species,
        )
        exogibbs_cond_axis.plot(
            _positive(
                exogibbs_condensates[:, slot],
                floor=CONDENSATE_PLOT_FLOOR,
            ),
            pressures,
            "-o",
            color=color,
            linewidth=1.6,
            markersize=3.0,
        )
        plotted_condensates += 1

    if plotted_gases == 0 or plotted_condensates == 0:
        raise RuntimeError(
            "None of the requested gas or condensate species could be plotted."
        )

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    fastchem_gas_axis.set_ylim(
        float(np.max(pressures)) * 1.15,
        float(np.min(pressures)) / 1.15,
    )

    fastchem_gas_axis.set_title("FastChem 4 — gas phase")
    exogibbs_gas_axis.set_title("ExoGibbs — gas phase")
    fastchem_cond_axis.set_title("FastChem 4 — condensates")
    exogibbs_cond_axis.set_title("ExoGibbs — condensates")

    for axis in axes[0]:
        axis.set_xlabel("Gas mixing ratio")
    for axis in axes[1]:
        axis.set_xlabel("Condensate amount / total element density")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pressure [bar]")

    fastchem_gas_axis.legend(fontsize=7, ncol=2, loc="best")
    gas_state_handles = (
        Line2D(
            (0,),
            (0,),
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Gas-only",
        ),
        Line2D(
            (0,),
            (0,),
            color="black",
            linestyle="-",
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            label="With condensates",
        ),
    )
    exogibbs_gas_axis.legend(
        handles=gas_state_handles,
        fontsize=7,
        loc="best",
        title="Gas state",
        title_fontsize=7,
    )
    fastchem_cond_axis.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(
        "Equilibrium condensation along an illustrative L-dwarf-like profile\n"
        f"{np.min(temperatures):.0f}--{np.max(temperatures):.0f} K",
        fontsize=12,
    )
    profile_note = (
        r"$10^{-4} \leq P \leq 10^{2}$ bar; "
        r"$T=1100+160q+15q^2$ K, "
        r"$q=\log_{10}(P/\mathrm{bar})+4$; local equilibrium, no rainout"
    )
    fig.text(0.5, 0.012, profile_note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    return fig


def _plot_l_dwarf_profile_comparison(
    *,
    pressures: np.ndarray,
    temperatures: np.ndarray,
    setup,
    exogibbs_gas_only_x: np.ndarray,
    exogibbs_x: np.ndarray,
    exogibbs_condensates: np.ndarray,
    fastchem_gas_only_x: np.ndarray,
    fastchem_x: np.ndarray,
    fastchem_condensates: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    fig = _make_l_dwarf_profile_figure(
        pressures=pressures,
        temperatures=temperatures,
        setup=setup,
        exogibbs_gas_only_x=exogibbs_gas_only_x,
        exogibbs_x=exogibbs_x,
        exogibbs_condensates=exogibbs_condensates,
        fastchem_gas_only_x=fastchem_gas_only_x,
        fastchem_x=fastchem_x,
        fastchem_condensates=fastchem_condensates,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    executable_path = args.fastchem_executable.resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError(
            f"FastChem executable is not an executable file: {executable_path}."
        )
    temperatures, pressures = _profile_conditions(args.profile)
    output_path = args.output or _default_output_path(args.profile)
    output_path.unlink(missing_ok=True)

    # 1. Run FastChem as an independent equilibrium-condensation process.
    fastchem = run_fastchem_executable(
        executable=executable_path,
        temperatures=temperatures,
        pressures=pressures,
        element_abundance_file=ELEMENT_FILE,
        gas_logk_file=GAS_LOGK_FILE,
        condensate_logk_file=CONDENSATE_LOGK_FILE,
        chemistry_mode="equilibrium_condensation",
    )
    fastchem_gas_only = None
    if args.profile == "l-dwarf":
        fastchem_gas_only = run_fastchem_executable(
            executable=executable_path,
            temperatures=temperatures,
            pressures=pressures,
            element_abundance_file=ELEMENT_FILE,
            gas_logk_file=GAS_LOGK_FILE,
            chemistry_mode="gas",
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
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(pressures, dtype=jnp.float64),
        b=budget,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )
    jax.block_until_ready(exogibbs.batched_arrays)

    exogibbs_gas_only_x = None
    exogibbs_gas_only_converged = None
    if args.profile == "l-dwarf":
        exogibbs_gas_only, gas_only_diagnostics = solve_gas_profile(
            setup.gas_setup,
            T=jnp.asarray(temperatures, dtype=jnp.float64),
            P=jnp.asarray(pressures, dtype=jnp.float64),
            b=budget,
            options=GasEquilibriumOptions(),
            return_diagnostics=True,
        )
        jax.block_until_ready(
            (exogibbs_gas_only.x, gas_only_diagnostics["converged"])
        )
        exogibbs_gas_only_x = np.asarray(
            exogibbs_gas_only.x,
            dtype=np.float64,
        )
        exogibbs_gas_only_converged = np.asarray(
            gas_only_diagnostics["converged"],
            dtype=bool,
        )

    # 3. Align names only after both independent calculations are complete.
    _validate_comparison_contract(
        setup=setup,
        fastchem=fastchem,
        exogibbs=exogibbs,
    )
    if fastchem_gas_only is not None:
        assert exogibbs_gas_only_x is not None
        assert exogibbs_gas_only_converged is not None
        _validate_gas_only_comparison_contract(
            setup=setup,
            fastchem=fastchem_gas_only,
            exogibbs_x=exogibbs_gas_only_x,
            exogibbs_converged=exogibbs_gas_only_converged,
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
    fastchem_gas_only_x = None
    if fastchem_gas_only is not None:
        fastchem_gas_only_density = align_species_values(
            setup.gas_species,
            fastchem_gas_only.gas_names,
            fastchem_gas_only.gas_number_densities,
        )
        fastchem_gas_only_x = fastchem_gas_only_density / np.sum(
            fastchem_gas_only_density,
            axis=1,
            keepdims=True,
        )
    exogibbs_x = np.stack(
        [
            np.asarray(layer.gas_x, dtype=np.float64)
            for layer in exogibbs.layers
        ]
    )
    if exogibbs_gas_only_x is not None and fastchem_gas_only_x is not None:
        _validate_gas_profile_release_metrics(
            names=setup.gas_species,
            left_values=exogibbs_gas_only_x,
            right_values=fastchem_gas_only_x,
            comparison_label="gas-only",
        )
    exogibbs_condensates = np.stack(
        [
            np.asarray(layer.condensate_amounts, dtype=np.float64)
            for layer in exogibbs.layers
        ]
    )

    # 4. Print a compact numerical summary and make the visual comparison.
    _print_summary(
        profile=args.profile,
        temperatures=temperatures,
        pressures=pressures,
        setup=setup,
        exogibbs_result=exogibbs,
        exogibbs_x=exogibbs_x,
        exogibbs_condensates=exogibbs_condensates,
        fastchem=fastchem,
        fastchem_x=fastchem_x,
        fastchem_condensates=fastchem_condensates,
    )
    if args.profile == "validation":
        _plot_validation_comparison(
            temperatures=temperatures,
            setup=setup,
            exogibbs_x=exogibbs_x,
            exogibbs_condensates=exogibbs_condensates,
            fastchem_x=fastchem_x,
            fastchem_condensates=fastchem_condensates,
            output_path=output_path,
            show=args.show,
        )
    else:
        assert exogibbs_gas_only_x is not None
        assert fastchem_gas_only_x is not None
        _plot_l_dwarf_profile_comparison(
            pressures=pressures,
            temperatures=temperatures,
            setup=setup,
            exogibbs_gas_only_x=exogibbs_gas_only_x,
            exogibbs_x=exogibbs_x,
            exogibbs_condensates=exogibbs_condensates,
            fastchem_gas_only_x=fastchem_gas_only_x,
            fastchem_x=fastchem_x,
            fastchem_condensates=fastchem_condensates,
            output_path=output_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
