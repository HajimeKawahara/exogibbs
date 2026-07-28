"""Reproduce the historical comparison with the YK B4 gas calculation.

This is a regression of equilibrium amounts against a frozen 160-species
result at 500 K and 10 bar.  The original YK B4 program is not bundled, so
this example is not an independently rerunnable external-code comparison.
See ``examples/data/README.md`` for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

# Select deterministic CPU/x64 behavior before importing JAX or ExoGibbs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.presets.ykb4 import chemsetup


config.update("jax_enable_x64", True)

REFERENCE_PATH = REPOSITORY_ROOT / "examples" / "data" / "p10.txt"
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "ykb4"
    / "comparison_with_ykcode.png"
)

TEMPERATURE_K = 500.0
PRESSURE_BAR = 10.0
REFERENCE_PRESSURE_BAR = 1.0
MAJOR_SPECIES_FLOOR = 1.0e-14
MAX_RELATIVE_ERROR_LIMIT = 0.051

# Exact element order and budget used by the historical passing comparison.
EXPECTED_ELEMENTS = (
    "C",
    "H",
    "He",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Ti",
    "V",
    "e-",
)
LEGACY_ELEMENT_BUDGET = (
    4.8774824e-04,
    1.6749767e00,
    1.6143440e-01,
    2.5438149e-07,
    1.3435642e-04,
    3.9624806e-06,
    9.7356725e-04,
    5.7690579e-07,
    3.0653933e-05,
    1.6687756e-07,
    1.9870969e-08,
    0.0000000e00,
)

EXPECTED_SPECIES_COUNT = 160
EXPECTED_SPECIES_ORDER_SHA256 = (
    "3f020b61342d0034c7d01b110fca2e62f63ea135e3cf7045c99a551c739f492a"
)
EXPECTED_REFERENCE_SHA256 = (
    "062a0d21768f85871b7980ae3883d34f2466b9e11618255060e19d32c4a8612b"
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ExoGibbs with the frozen historical YK B4 "
            "160-species result at 500 K and 10 bar."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="PNG output path (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args(argv)


def _species_order_digest(species: Tuple[str, ...]) -> str:
    """Return the stable fingerprint used for the ordered species catalog."""

    payload = "\0".join(species).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_reference() -> np.ndarray:
    """Load and validate the exact historical 160-value snapshot."""

    reference_bytes = REFERENCE_PATH.read_bytes()
    reference_digest = hashlib.sha256(reference_bytes).hexdigest()
    if reference_digest != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(
            "The YK B4 reference file does not match the documented snapshot: "
            f"{reference_digest}."
        )

    reference = np.loadtxt(REFERENCE_PATH, delimiter=",")
    if reference.shape != (EXPECTED_SPECIES_COUNT,):
        raise RuntimeError(
            "The YK B4 reference must contain exactly "
            f"{EXPECTED_SPECIES_COUNT} ordered values; got {reference.shape}."
        )
    if not np.all(np.isfinite(reference)) or np.any(reference < 0.0):
        raise RuntimeError("The YK B4 reference contains invalid abundances.")
    return reference


def _validate_catalog(
    elements: Tuple[str, ...],
    species: Tuple[str, ...],
) -> None:
    """Fail closed if either catalog no longer matches the frozen snapshot."""

    if elements != EXPECTED_ELEMENTS:
        raise RuntimeError(
            "The YKB4 element order changed; the legacy element budget cannot "
            "be mapped safely."
        )
    if len(species) != EXPECTED_SPECIES_COUNT:
        raise RuntimeError(
            "The YKB4 species count changed: "
            f"expected {EXPECTED_SPECIES_COUNT}, got {len(species)}."
        )

    species_digest = _species_order_digest(species)
    if species_digest != EXPECTED_SPECIES_ORDER_SHA256:
        raise RuntimeError(
            "The ordered YKB4 species catalog changed; comparison with p10.txt "
            "would be misaligned."
        )


def _plot_comparison(
    species: Tuple[str, ...],
    exogibbs_amounts: np.ndarray,
    reference_amounts: np.ndarray,
    major_mask: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    """Plot major abundances and their signed relative differences."""

    major_indices = np.flatnonzero(major_mask)
    order = major_indices[
        np.argsort(reference_amounts[major_indices], kind="stable")[::-1]
    ]
    labels = [species[index] for index in order]
    relative_percent = (
        exogibbs_amounts[order] / reference_amounts[order] - 1.0
    ) * 100.0
    positions = np.arange(order.size)

    figure, (axis_abundance, axis_difference) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0), "hspace": 0.08},
    )
    axis_abundance.semilogy(
        positions,
        reference_amounts[order],
        "o",
        markersize=6,
        label="YK B4 frozen snapshot",
    )
    axis_abundance.semilogy(
        positions,
        exogibbs_amounts[order],
        "x",
        markersize=7,
        markeredgewidth=1.6,
        label="ExoGibbs production gas solver",
    )
    axis_abundance.set_ylabel("Equilibrium amount")
    axis_abundance.set_title(
        "Historical YK B4 regression at 500 K and 10 bar"
    )
    axis_abundance.grid(True, which="both", alpha=0.25)
    axis_abundance.legend()

    axis_difference.bar(positions, relative_percent, color="tab:blue")
    tolerance_percent = 100.0 * MAX_RELATIVE_ERROR_LIMIT
    axis_difference.axhline(
        tolerance_percent,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    axis_difference.axhline(
        -tolerance_percent,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"legacy limit: {tolerance_percent:.1f}%",
    )
    axis_difference.axhline(0.0, color="black", linewidth=0.8)
    axis_difference.set_ylabel("Relative\ndifference (%)")
    axis_difference.set_xticks(positions)
    axis_difference.set_xticklabels(labels, rotation=55, ha="right")
    axis_difference.set_xlabel("Species (ordered by YK B4 abundance)")
    axis_difference.grid(True, axis="y", alpha=0.25)
    axis_difference.legend(loc="lower left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the frozen-snapshot regression and write its comparison plot."""

    args = _parse_args(argv)
    reference_amounts = _load_reference()

    setup = chemsetup()
    if setup.elements is None or setup.species is None:
        raise RuntimeError("The YKB4 preset must provide ordered catalogs.")
    elements = tuple(setup.elements)
    species = tuple(setup.species)
    _validate_catalog(elements, species)

    result, diagnostics = solve(
        setup,
        T=TEMPERATURE_K,
        P=PRESSURE_BAR,
        b=jnp.asarray(LEGACY_ELEMENT_BUDGET, dtype=jnp.float64),
        Pref=REFERENCE_PRESSURE_BAR,
        options=EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000),
        return_diagnostics=True,
    )

    converged = bool(np.asarray(diagnostics["converged"]))
    hit_max_iter = bool(np.asarray(diagnostics["hit_max_iter"]))
    if not converged or hit_max_iter:
        raise RuntimeError(
            "ExoGibbs did not converge for the historical YK B4 point: "
            f"residual={float(diagnostics['final_residual']):.3e}, "
            f"iterations={int(diagnostics['n_iter'])}."
        )

    exogibbs_amounts = np.asarray(result.n, dtype=np.float64)
    if exogibbs_amounts.shape != reference_amounts.shape:
        raise RuntimeError(
            "ExoGibbs and the reference do not have the same species length: "
            f"{exogibbs_amounts.shape} versus {reference_amounts.shape}."
        )
    if not np.all(np.isfinite(exogibbs_amounts)):
        raise RuntimeError("ExoGibbs returned non-finite equilibrium amounts.")

    major_mask = reference_amounts > MAJOR_SPECIES_FLOOR
    relative_error = (
        exogibbs_amounts[major_mask] / reference_amounts[major_mask] - 1.0
    )
    max_relative_error = float(np.max(np.abs(relative_error)))
    worst_major_index = int(
        np.flatnonzero(major_mask)[np.argmax(np.abs(relative_error))]
    )

    print("Historical YK B4 regression snapshot")
    print(f"  condition: {TEMPERATURE_K:.0f} K, {PRESSURE_BAR:.0f} bar")
    print(
        "  convergence: "
        f"{int(diagnostics['n_iter'])} iterations, "
        f"residual={float(diagnostics['final_residual']):.3e}"
    )
    print(
        "  catalog: "
        f"{len(species)} species, exact ordered fingerprint matched"
    )
    print(
        "  major species: "
        f"{int(np.count_nonzero(major_mask))} with reference amount > "
        f"{MAJOR_SPECIES_FLOOR:.0e}"
    )
    print(
        "  worst major-species difference: "
        f"{species[worst_major_index]} = "
        f"{100.0 * max_relative_error:.3f}% "
        f"(limit {100.0 * MAX_RELATIVE_ERROR_LIMIT:.1f}%)"
    )

    if max_relative_error >= MAX_RELATIVE_ERROR_LIMIT:
        raise AssertionError(
            "Historical YK B4 regression exceeded its 5.1% limit: "
            f"{max_relative_error:.6f}."
        )

    _plot_comparison(
        species,
        exogibbs_amounts,
        reference_amounts,
        major_mask,
        args.output,
        args.show,
    )
    print(f"  plot: {args.output}")
    print(
        "  scope: frozen regression snapshot; the original YK B4 solver "
        "is not rerun"
    )


if __name__ == "__main__":
    main()
