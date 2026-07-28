"""Trace the historical FastChem grid-initializer comparison.

This example expects an operator-supplied FastChem v4.0.3 standalone
executable and runs it with ExoGibbs' packaged legacy FastChem v3.1.3
thermochemical data.  It is therefore a traceability example for the saved
``fastchem`` equilibrium grid, not the FastChem 4 dataset used by the v0.4
production-validation demo.  This lightweight example does not verify the
executable version or hash; use the formal benchmark runner for that preflight.

ExoGibbs is solved twice, once with the packaged grid initializer and once
with its default uniform initializer.  FastChem is then run independently.
No FastChem result is passed to either ExoGibbs initializer or solver.
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
from exogibbs.api import (
    compute_physical_log10_z_over_z_sun,
    get_default_equilibrium_grid_path,
    load_equilibrium_grid_netcdf,
)
from exogibbs.api.gas import (
    EquilibriumOptions,
    GridEquilibriumInitializer,
    solve_profile as solve_gas_profile,
)
from exogibbs.presets.fastchem import chemsetup
from exogibbs.utils.fastchem_parity import (
    build_aligned_abundance_vector,
)


config.update("jax_enable_x64", True)

DATA_ROOT = REPOSITORY_ROOT / "src" / "exogibbs" / "data" / "fastchem"
ELEMENT_FILE = DATA_ROOT / "element_abundances" / "asplund_2020.dat"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK.dat"

TEMPERATURE_K = 2870.0
PRESSURES_BAR = np.logspace(-8.0, 2.0, 100)
MAJOR_GAS_THRESHOLD = 1.0e-8
PLOT_FLOOR = 1.0e-30
ELEMENT_BUDGET_TOLERANCE = 2.0e-4
INITIALIZER_AGREEMENT_TOLERANCE_DEX = 1.0e-3
FASTCHEM_AGREEMENT_TOLERANCE_DEX = 1.0e-3
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
            "Compare a legacy-data FastChem standalone calculation with "
            "grid- and default-initialized ExoGibbs gas solves."
        )
    )
    parser.add_argument(
        "--fastchem-executable",
        required=True,
        type=Path,
        help=(
            "Path to an operator-supplied FastChem standalone executable "
            "(v4.0.3 expected; version/hash not checked here)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "fastchem4_examples"
            / "comparison_with_fastchem_initializer.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _validate_fastchem_contract(*, setup, fastchem) -> None:
    exogibbs_catalog = sorted(occurrence_keys(setup.species))
    fastchem_catalog = sorted(occurrence_keys(fastchem.gas_names))
    if exogibbs_catalog != fastchem_catalog:
        exogibbs_only = sorted(set(exogibbs_catalog) - set(fastchem_catalog))
        fastchem_only = sorted(set(fastchem_catalog) - set(exogibbs_catalog))
        raise RuntimeError(
            "FastChem and ExoGibbs gas catalogs do not match: "
            f"ExoGibbs-only={exogibbs_only[:10]}, "
            f"FastChem-only={fastchem_only[:10]}."
        )

    failed = np.flatnonzero(~fastchem.converged)
    unconserved = np.flatnonzero(~fastchem.elements_conserved)
    invalid_gas = np.flatnonzero(
        ~np.all(np.isfinite(fastchem.gas_number_densities), axis=1)
        | np.any(fastchem.gas_number_densities < 0.0, axis=1)
        | (np.sum(fastchem.gas_number_densities, axis=1) <= 0.0)
    )
    invalid_element_density = np.flatnonzero(
        ~np.isfinite(fastchem.total_element_density)
        | (fastchem.total_element_density <= 0.0)
    )
    if (
        failed.size
        or unconserved.size
        or invalid_gas.size
        or invalid_element_density.size
    ):
        raise RuntimeError(
            "FastChem output failed validation: "
            f"not converged={failed.tolist()}, "
            f"elements not conserved={unconserved.tolist()}, "
            f"invalid gas={invalid_gas.tolist()}, "
            "invalid total element density="
            f"{invalid_element_density.tolist()}."
        )


def _validate_exogibbs_result(
    *,
    label: str,
    result,
    diagnostics,
) -> None:
    converged = np.asarray(diagnostics["converged"], dtype=bool)
    failed = np.flatnonzero(~converged)
    amounts = np.asarray(result.n, dtype=np.float64)
    invalid = np.flatnonzero(
        ~np.all(np.isfinite(amounts), axis=1)
        | np.any(amounts < 0.0, axis=1)
        | (np.sum(amounts, axis=1) <= 0.0)
    )
    if failed.size or invalid.size:
        raise RuntimeError(
            f"ExoGibbs {label} solve failed validation: "
            f"not converged={failed.tolist()}, "
            f"invalid amounts={invalid.tolist()}."
        )


def _gas_profile_metrics(
    *,
    names,
    left: np.ndarray,
    right: np.ndarray,
) -> list[dict]:
    return [
        gas_major_species_metrics(
            names=names,
            left_values=left[layer_index],
            right_values=right[layer_index],
            threshold=MAJOR_GAS_THRESHOLD,
            excluded_names=("e-",),
        )
        for layer_index in range(PRESSURES_BAR.size)
    ]


def _budget_residuals(
    *,
    setup,
    budget: np.ndarray,
    amounts: np.ndarray,
) -> np.ndarray:
    empty_formula = np.zeros((len(setup.elements), 0), dtype=np.float64)
    empty_amounts = np.zeros((0,), dtype=np.float64)
    return np.asarray(
        [
            element_budget_metrics(
                gas_formula_matrix=setup.formula_matrix,
                condensate_formula_matrix=empty_formula,
                gas_amounts=amounts[layer_index],
                condensate_amounts=empty_amounts,
                target=budget,
                element_names=setup.elements,
            )["max_absolute_relative_residual"]
            for layer_index in range(PRESSURES_BAR.size)
        ],
        dtype=np.float64,
    )


def _worst_metric(
    metrics: list[dict],
) -> tuple[int, dict, str]:
    layer_index = max(
        range(len(metrics)),
        key=lambda index: metrics[index]["max_absolute_log10_ratio"],
    )
    metric = metrics[layer_index]
    species = (
        metric["top_rows"][0]["name"] if metric["top_rows"] else "(none)"
    )
    return layer_index, metric, species


def _validate_comparison_metrics(
    *,
    grid_uniform_metrics: list[dict],
    grid_fastchem_metrics: list[dict],
    budget_residuals: dict[str, np.ndarray],
) -> None:
    comparison_contracts = (
        (
            "Grid and default initializers",
            grid_uniform_metrics,
            INITIALIZER_AGREEMENT_TOLERANCE_DEX,
        ),
        (
            "Grid-initialized ExoGibbs and FastChem",
            grid_fastchem_metrics,
            FASTCHEM_AGREEMENT_TOLERANCE_DEX,
        ),
    )
    for label, metrics, tolerance in comparison_contracts:
        minimum_jaccard = min(
            metric["major_set_jaccard"] for metric in metrics
        )
        maximum_dex = max(
            metric["max_absolute_log10_ratio"] for metric in metrics
        )
        if minimum_jaccard < 1.0 or maximum_dex > tolerance:
            raise RuntimeError(
                f"{label} did not agree on the major-gas solution: "
                f"minimum Jaccard={minimum_jaccard:.6g}, "
                f"maximum difference={maximum_dex:.6g} dex "
                f"(limit {tolerance:.1e} dex)."
            )

    failed_budget = {
        label: float(np.max(values))
        for label, values in budget_residuals.items()
        if not np.all(np.isfinite(values))
        or float(np.max(values)) > ELEMENT_BUDGET_TOLERANCE
    }
    if failed_budget:
        raise RuntimeError(
            "Elemental-budget closure exceeded "
            f"{ELEMENT_BUDGET_TOLERANCE:.1e}: {failed_budget}."
        )


def _print_grid_provenance(*, grid_path: Path, metadata) -> None:
    print("Packaged grid provenance")
    print(f"  file: {grid_path}")
    print(f"  preset/source: {metadata.preset_name}/{metadata.source}")
    print(
        "  thermochemical source recorded by grid: "
        f"{metadata.preset_setup_metadata.get('source')}"
    )
    print(
        "  historical FastChem verification enabled/passed: "
        f"{bool(metadata.verify_exogibbs_against_fastchem)}/"
        f"{bool(metadata.verification_passed)}"
    )
    print(
        "  historical verification points/species comparisons: "
        f"{metadata.verification_points_checked}/"
        f"{metadata.verification_species_compared}"
    )
    print(
        "  historical tolerance/worst deviation: "
        f"{metadata.verification_tolerance_percent:g}%/"
        f"{metadata.verification_max_abs_percent_deviation:.6g}%"
    )


def _print_summary(
    *,
    setup,
    metallicity: float,
    grid_result,
    uniform_result,
    grid_diagnostics,
    uniform_diagnostics,
    fastchem,
    fastchem_amounts: np.ndarray,
    fastchem_x: np.ndarray,
    grid_uniform_metrics: list[dict],
    grid_fastchem_metrics: list[dict],
    budget_residuals: dict[str, np.ndarray],
) -> None:
    grid_iterations = np.asarray(grid_diagnostics["n_iter"], dtype=np.int64)
    uniform_iterations = np.asarray(
        uniform_diagnostics["n_iter"],
        dtype=np.int64,
    )
    grid_uniform_layer, grid_uniform_worst, grid_uniform_species = (
        _worst_metric(grid_uniform_metrics)
    )
    fastchem_layer, fastchem_worst, fastchem_species = _worst_metric(
        grid_fastchem_metrics
    )

    grid_total = np.asarray(grid_result.ntot, dtype=np.float64)
    uniform_total = np.asarray(uniform_result.ntot, dtype=np.float64)
    fastchem_total = np.sum(fastchem_amounts, axis=1)

    print("Legacy-grid initializer comparison")
    print(f"  temperature: {TEMPERATURE_K:g} K")
    print(
        "  pressure range: "
        f"{PRESSURES_BAR.min():.0e}--{PRESSURES_BAR.max():.0e} bar "
        f"({PRESSURES_BAR.size} layers)"
    )
    print(f"  physical log10(Z/Zsun): {metallicity:.6f}")
    print(
        "  catalogs: "
        f"{len(setup.species)} gas species in both implementations"
    )
    print(
        "  converged layers (grid/default/FastChem): "
        f"{np.count_nonzero(grid_diagnostics['converged'])}/"
        f"{np.count_nonzero(uniform_diagnostics['converged'])}/"
        f"{np.count_nonzero(fastchem.converged)} "
        f"of {PRESSURES_BAR.size}"
    )
    print(
        "  ExoGibbs iteration median/max (grid vs default): "
        f"{np.median(grid_iterations):g}/{np.max(grid_iterations)} vs "
        f"{np.median(uniform_iterations):g}/{np.max(uniform_iterations)}"
    )
    print(
        "  grid-vs-default final solution: "
        f"minimum major-set Jaccard "
        f"{min(row['major_set_jaccard'] for row in grid_uniform_metrics):.3f}; "
        f"worst difference "
        f"{grid_uniform_worst['max_absolute_log10_ratio']:.3g} dex "
        f"for {grid_uniform_species} at "
        f"{PRESSURES_BAR[grid_uniform_layer]:.3g} bar"
    )
    print(
        "  grid-initialized ExoGibbs vs FastChem: "
        f"minimum major-set Jaccard "
        f"{min(row['major_set_jaccard'] for row in grid_fastchem_metrics):.3f}; "
        f"worst difference "
        f"{fastchem_worst['max_absolute_log10_ratio']:.3g} dex "
        f"for {fastchem_species} at "
        f"{PRESSURES_BAR[fastchem_layer]:.3g} bar"
    )
    print(
        "  maximum absolute relative total-gas difference "
        "(grid/default): "
        f"{np.max(np.abs(grid_total / uniform_total - 1.0)):.3g}"
    )
    print(
        "  maximum absolute relative total-gas difference "
        "(ExoGibbs/FastChem): "
        f"{np.max(np.abs(grid_total / fastchem_total - 1.0)):.3g}"
    )
    print(
        "  maximum elemental-budget residual "
        "(grid/default/FastChem): "
        f"{np.max(budget_residuals['grid']):.3g}/"
        f"{np.max(budget_residuals['default']):.3g}/"
        f"{np.max(budget_residuals['fastchem']):.3g}"
    )
    print(
        "  FastChem element-conservation layers: "
        f"{np.count_nonzero(fastchem.elements_conserved)}/"
        f"{PRESSURES_BAR.size}"
    )


def _positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(
        np.isfinite(array) & (array > PLOT_FLOOR),
        array,
        np.nan,
    )


def _plot_comparison(
    *,
    setup,
    grid_x: np.ndarray,
    fastchem_x: np.ndarray,
    grid_diagnostics,
    uniform_diagnostics,
    output_path: Path,
    show: bool,
) -> None:
    fig, (ax_abundance, ax_difference, ax_iterations) = plt.subplots(
        1,
        3,
        figsize=(13.0, 5.0),
        sharey=True,
        gridspec_kw={"width_ratios": (1.6, 1.0, 1.0)},
    )
    colors = plt.get_cmap("tab10")

    plotted = 0
    for species_index, species in enumerate(PLOT_SPECIES):
        if species not in setup.species:
            continue
        slot = setup.species.index(species)
        exogibbs_values = _positive(grid_x[:, slot])
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
                np.clip(grid_x[:, slot], PLOT_FLOOR, None)
                / np.clip(fastchem_x[:, slot], PLOT_FLOOR, None)
            )
        )
        ax_difference.plot(
            np.clip(absolute_dex, 1.0e-12, None),
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
    ax_difference.set_title("FastChem vs grid initializer")
    ax_difference.grid(alpha=0.25)

    ax_iterations.plot(
        np.asarray(grid_diagnostics["n_iter"]),
        PRESSURES_BAR,
        label="Grid initializer",
        linewidth=1.8,
    )
    ax_iterations.plot(
        np.asarray(uniform_diagnostics["n_iter"]),
        PRESSURES_BAR,
        "--",
        label="Default initializer",
        linewidth=1.8,
    )
    ax_iterations.set_xlabel("ExoGibbs iterations")
    ax_iterations.set_title("Initializer cost")
    ax_iterations.legend(fontsize=8)
    ax_iterations.grid(alpha=0.25)

    fig.suptitle(
        f"Legacy FastChem-data gas equilibrium at {TEMPERATURE_K:g} K",
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

    # Both ExoGibbs runs are fully specified before FastChem is invoked.
    setup = chemsetup(
        path="fastchem/logK/logK.dat",
        species_default_elements=True,
        silent=True,
    )
    aligned_abundance = build_aligned_abundance_vector(
        setup.elements,
        source="fastchem_file",
        normalize=True,
        element_file=ELEMENT_FILE,
    )
    budget = jnp.asarray(aligned_abundance.vector, dtype=jnp.float64)
    grid_path = get_default_equilibrium_grid_path("fastchem")
    grid = load_equilibrium_grid_netcdf(str(grid_path))
    grid_initializer = GridEquilibriumInitializer(
        grid=grid,
        preset_name="fastchem",
    )
    options = EquilibriumOptions(
        epsilon_crit=1.0e-12,
        method="vmap_cold",
    )

    grid_result, grid_diagnostics = solve_gas_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(PRESSURES_BAR, dtype=jnp.float64),
        b=budget,
        initializer=grid_initializer,
        options=options,
        return_diagnostics=True,
    )
    uniform_result, uniform_diagnostics = solve_gas_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(PRESSURES_BAR, dtype=jnp.float64),
        b=budget,
        options=options,
        return_diagnostics=True,
    )
    jax.block_until_ready(
        (
            grid_result.x,
            grid_diagnostics["converged"],
            uniform_result.x,
            uniform_diagnostics["converged"],
        )
    )
    _validate_exogibbs_result(
        label="grid-initialized",
        result=grid_result,
        diagnostics=grid_diagnostics,
    )
    _validate_exogibbs_result(
        label="default-initialized",
        result=uniform_result,
        diagnostics=uniform_diagnostics,
    )

    # FastChem is an independent comparator; its values never feed ExoGibbs.
    fastchem = run_fastchem_executable(
        executable=args.fastchem_executable,
        temperatures=temperatures,
        pressures=PRESSURES_BAR,
        element_abundance_file=ELEMENT_FILE,
        gas_logk_file=GAS_LOGK_FILE,
        chemistry_mode="gas",
    )
    _validate_fastchem_contract(setup=setup, fastchem=fastchem)

    grid_x = np.asarray(grid_result.x, dtype=np.float64)
    uniform_x = np.asarray(uniform_result.x, dtype=np.float64)
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

    grid_uniform_metrics = _gas_profile_metrics(
        names=setup.species,
        left=grid_x,
        right=uniform_x,
    )
    grid_fastchem_metrics = _gas_profile_metrics(
        names=setup.species,
        left=grid_x,
        right=fastchem_x,
    )
    budget_values = np.asarray(budget, dtype=np.float64)
    budget_residuals = {
        "grid": _budget_residuals(
            setup=setup,
            budget=budget_values,
            amounts=np.asarray(grid_result.n, dtype=np.float64),
        ),
        "default": _budget_residuals(
            setup=setup,
            budget=budget_values,
            amounts=np.asarray(uniform_result.n, dtype=np.float64),
        ),
        "fastchem": _budget_residuals(
            setup=setup,
            budget=budget_values,
            amounts=fastchem_amounts,
        ),
    }
    _validate_comparison_metrics(
        grid_uniform_metrics=grid_uniform_metrics,
        grid_fastchem_metrics=grid_fastchem_metrics,
        budget_residuals=budget_residuals,
    )

    metallicity = float(
        compute_physical_log10_z_over_z_sun(setup, budget_values)
    )
    print(
        "Dataset boundary: operator-supplied FastChem standalone "
        "(v4.0.3 expected, not verified here) with packaged FastChem v3.1.3 "
        "data; not the v0.4 FastChem 4 dataset validation."
    )
    print(f"  shared element-abundance file: {ELEMENT_FILE}")
    print(f"  shared gas logK file: {GAS_LOGK_FILE}")
    print(
        "  FastChem-to-ExoGibbs initialization flow: none "
        "(name alignment occurs after all solves)"
    )
    _print_grid_provenance(grid_path=grid_path, metadata=grid.metadata)
    _print_summary(
        setup=setup,
        metallicity=metallicity,
        grid_result=grid_result,
        uniform_result=uniform_result,
        grid_diagnostics=grid_diagnostics,
        uniform_diagnostics=uniform_diagnostics,
        fastchem=fastchem,
        fastchem_amounts=fastchem_amounts,
        fastchem_x=fastchem_x,
        grid_uniform_metrics=grid_uniform_metrics,
        grid_fastchem_metrics=grid_fastchem_metrics,
        budget_residuals=budget_residuals,
    )
    _plot_comparison(
        setup=setup,
        grid_x=grid_x,
        fastchem_x=fastchem_x,
        grid_diagnostics=grid_diagnostics,
        uniform_diagnostics=uniform_diagnostics,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
