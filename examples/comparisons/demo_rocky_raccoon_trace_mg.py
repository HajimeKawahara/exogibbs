"""Solve a Rocky Raccoon-like layer with a positive trace Mg inventory.

This is a provider-side regression example, not a reproduction of the full
atmospheric structure in Misener et al. (2026).  The explicit gas network does
not add neutral atoms or a free electron as reference species.  Charged gas
species are balanced by the zero-inventory ``e-`` stoichiometric constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate import (
    CondensateChemicalSetup,
    CondensateEquilibriumOptions,
    CondensateEquilibriumProfileResult,
    build_condensate_chemical_setup,
    solve_profile as solve_condensate_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


config.update("jax_enable_x64", True)

TEMPERATURE_K = 1433.7645951014717
PRESSURE_BAR = 8796.093022208004
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "results"
    / "rocky_raccoon_trace_mg"
    / "trace_mg_audit.json"
)
ELEMENTS = ("H", "Mg", "Si", "O", "C", "e-")
ELEMENT_INVENTORY = np.asarray(
    [
        0.9996763591270366,
        2.415508476464756e-12,
        1.824264250909887e-5,
        3.6485287433706685e-5,
        2.689129406051728e-4,
        0.0,
    ],
    dtype=np.float64,
)

# Misener et al. (2026), Appendix A, Table 1, mapped to the packaged
# FastChem4 labels.  Neutral atoms and the free electron are intentionally
# absent; ions are coupled through the final charge-constraint row.
GAS_SPECIES = (
    "C1H1",
    "C1H1O1",
    "C1H2",
    "C1H2O1",
    "C1H3",
    "C1H4",
    "C1H4O2",
    "C1O1",
    "C1O2",
    "C1Si1",
    "C1Si2",
    "C2",
    "C2H1",
    "C2H2",
    "C2H2O2",
    "C2H2O4",
    "C2H4",
    "C2H4O1",
    "C2H4O3",
    "C2H6O2",
    "C2Si1",
    "C2Si2",
    "C2O1",
    "C3",
    "C3H1",
    "C3O2",
    "C4",
    "C4H6O4",
    "C5",
    "C1+",
    "C1-",
    "C1H1+",
    "C1H1-",
    "C1H1O1+",
    "C1O2-",
    "C2-",
    "H1Mg1",
    "H1Mg1O1",
    "H1O1",
    "H1O2",
    "H1Si1",
    "H2",
    "H2Mg1O2",
    "H2O1",
    "H2O2",
    "H2Si1",
    "H3Si1",
    "H4Si1",
    "Mg1O1",
    "Mg2",
    "O1Si1",
    "O2",
    "O2Si1",
    "O3",
    "H1+",
    "H1-",
    "H1Mg1O1+",
    "H1O1+",
    "H1O1-",
    "H1Si1+",
    "H2+",
    "H2-",
    "H3O1+",
    "Mg1+",
    "O1+",
    "O1-",
    "O2+",
    "O2-",
    "Si1+",
    "Si1-",
)

# Appendix A, Table 2, excluding the paper's SiO(s) sensitivity species.
CONDENSATE_SPECIES = (
    "C(s)",
    "SiO2(s,l)",
    "Si(s,l)",
    "SiC(s)",
    "MgH2(s)",
    "Mg(OH)2(s)",
    "Mg(s,l)",
    "MgO(s,l)",
    "MgSiO3(s,l)",
    "Mg2SiO4(s,l)",
    "Mg2Si(s,l)",
    "MgCO3(s)",
    "MgC2(s)",
    "Mg2C3(s)",
)

REFERENCE_GAS_SPECIES = frozenset(
    {"H1", "Mg1", "Si1", "O1", "C1", "e1-"}
)


def _ordered_indices(
    requested: Sequence[str],
    available: Sequence[str],
    *,
    label: str,
) -> tuple[int, ...]:
    """Return exact catalog indices and reject missing or duplicate names."""

    names = tuple(requested)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names.")
    lookup = {name: index for index, name in enumerate(available)}
    missing = tuple(name for name in names if name not in lookup)
    if missing:
        raise ValueError(f"Unknown {label}: {missing!r}.")
    return tuple(lookup[name] for name in names)


def _subset_hvector(
    function: Callable[[jnp.ndarray], jnp.ndarray],
    indices: tuple[int, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    index_array = jnp.asarray(indices, dtype=jnp.int32)

    @jax.jit
    def hvector(temperature: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(function(temperature), index_array, axis=-1)

    return hvector


def build_reduced_setup(
    *, condensate_species: Sequence[str] = CONDENSATE_SPECIES,
) -> CondensateChemicalSetup:
    """Build the explicit H/Mg/Si/O/C/charge Rocky-like network."""

    condensate_species = tuple(condensate_species)
    full = condensate_chemical_setup(
        gas_path="FastChem4/logK/logK.dat",
        silent=True,
    )
    element_indices = _ordered_indices(
        ELEMENTS,
        full.elements,
        label="elements",
    )
    gas_indices = _ordered_indices(
        GAS_SPECIES,
        full.gas_species,
        label="gas species",
    )
    condensate_indices = _ordered_indices(
        condensate_species,
        full.condensate_species,
        label="condensate species",
    )

    full_gas_matrix = np.asarray(full.formula_matrix)
    selected_element_indices = frozenset(element_indices)
    excluded_rows = tuple(
        index
        for index in range(full_gas_matrix.shape[0])
        if index not in selected_element_indices
    )
    if excluded_rows and np.any(
        full_gas_matrix[np.ix_(excluded_rows, gas_indices)] != 0.0
    ):
        raise ValueError("Selected gas species use an excluded element.")

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            full_gas_matrix[np.ix_(element_indices, gas_indices)],
            dtype=jnp.float64,
        ),
        hvector_func=_subset_hvector(
            full.gas_setup.hvector_func,
            gas_indices,
        ),
        elements=ELEMENTS,
        species=GAS_SPECIES,
        element_vector_reference=jnp.asarray(
            ELEMENT_INVENTORY,
            dtype=jnp.float64,
        ),
        metadata={
            **dict(full.gas_setup.metadata or {}),
            "network": "rocky_raccoon_trace_mg",
        },
    )

    full_condensate_matrix = np.asarray(full.formula_matrix_cond)
    if excluded_rows and np.any(
        full_condensate_matrix[np.ix_(excluded_rows, condensate_indices)]
        != 0.0
    ):
        raise ValueError("Selected condensates use an excluded element.")
    condensate_metadata = dict(full.condensate_setup.metadata or {})
    original_validity = full.condensate_setup.temperature_validity_upper
    if original_validity is not None:
        condensate_metadata["original_temperature_validity_upper"] = tuple(
            float(original_validity[index])
            for index in condensate_indices
        )
    condensate_metadata.pop("temperature_validity_upper", None)
    condensate_metadata.update(
        {
            "network": "rocky_raccoon_trace_mg",
            "validity_mode": "paper_extrapolated",
        }
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            full_condensate_matrix[
                np.ix_(element_indices, condensate_indices)
            ],
            dtype=jnp.float64,
        ),
        hvector_func=_subset_hvector(
            full.condensate_setup.hvector_func,
            condensate_indices,
        ),
        elements=ELEMENTS,
        species=condensate_species,
        metadata=condensate_metadata,
        temperature_validity_upper=None,
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def solve_trace_mg_profile(
    setup: CondensateChemicalSetup,
) -> CondensateEquilibriumProfileResult:
    """Solve the captured one-layer rainout state through the public API."""

    profile = solve_condensate_profile(
        setup,
        T=jnp.asarray([TEMPERATURE_K], dtype=jnp.float64),
        P=jnp.asarray([PRESSURE_BAR], dtype=jnp.float64),
        b=jnp.asarray(ELEMENT_INVENTORY, dtype=jnp.float64),
        options=CondensateEquilibriumOptions(
            rainout=True,
            profile_method="scan_hot_from_bottom",
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )
    jax.block_until_ready(profile.batched_arrays)
    if len(profile.layers) != 1 or not profile.layers[0].converged:
        raise RuntimeError("The positive-trace Mg layer did not converge.")
    return profile


def audit_trace_mg_profile(
    setup: CondensateChemicalSetup,
    profile: CondensateEquilibriumProfileResult,
) -> dict[str, object]:
    """Return the model-neutral physical acceptance checks for the layer."""

    if len(profile.layers) != 1:
        raise ValueError("The trace-Mg profile must contain exactly one layer.")
    layer = profile.layers[0]
    gas_amounts = np.asarray(layer.gas_n, dtype=np.float64)
    condensate_amounts = np.asarray(
        layer.condensate_amounts,
        dtype=np.float64,
    )
    reconstructed = (
        np.asarray(setup.formula_matrix) @ gas_amounts
        + np.asarray(setup.formula_matrix_cond) @ condensate_amounts
    )
    nonzero = ELEMENT_INVENTORY != 0.0
    relative_residual = np.abs(
        (reconstructed[nonzero] - ELEMENT_INVENTORY[nonzero])
        / ELEMENT_INVENTORY[nonzero]
    )
    magnesium_index = setup.elements.index("Mg")
    charge_index = setup.elements.index("e-")
    gas_magnesium = float(
        np.asarray(setup.formula_matrix)[magnesium_index] @ gas_amounts
    )
    condensed_magnesium = float(
        np.asarray(setup.formula_matrix_cond)[magnesium_index]
        @ condensate_amounts
    )
    charge_residual = float(reconstructed[charge_index])
    maximum_relative_budget_residual = float(np.max(relative_residual))
    gas_fraction_sum_error = abs(
        float(np.sum(np.asarray(layer.gas_x, dtype=np.float64))) - 1.0
    )
    reference_species_present = tuple(
        sorted(REFERENCE_GAS_SPECIES.intersection(setup.gas_species))
    )
    accepted = bool(
        layer.converged
        and np.all(np.isfinite(gas_amounts))
        and np.all(np.isfinite(condensate_amounts))
        and np.all(gas_amounts >= 0.0)
        and np.all(condensate_amounts >= 0.0)
        and gas_magnesium > 0.0
        and condensed_magnesium > 0.0
        and maximum_relative_budget_residual <= 1.0e-8
        and abs(charge_residual) <= 1.0e-14
        and gas_fraction_sum_error <= 1.0e-10
        and not reference_species_present
    )
    return {
        "accepted": accepted,
        "temperature_k": TEMPERATURE_K,
        "pressure_bar": PRESSURE_BAR,
        "status": layer.status,
        "acceptance_tier": layer.acceptance_tier,
        "support": tuple(layer.condensate_support_names),
        "gas_magnesium": gas_magnesium,
        "condensed_magnesium": condensed_magnesium,
        "target_magnesium": float(ELEMENT_INVENTORY[magnesium_index]),
        "charge_residual": charge_residual,
        "maximum_relative_budget_residual": (
            maximum_relative_budget_residual
        ),
        "gas_fraction_sum_error": gas_fraction_sum_error,
        "reference_gas_species_present": reference_species_present,
    }


def write_audit(output_path: Path, audit: dict[str, object]) -> None:
    """Write one physical audit as deterministic JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one Rocky Raccoon-like positive trace-Mg layer."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.unlink(missing_ok=True)
    setup = build_reduced_setup()
    profile = solve_trace_mg_profile(setup)
    audit = audit_trace_mg_profile(setup, profile)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["accepted"]:
        raise RuntimeError("The positive-trace Mg physical audit failed.")
    write_audit(args.output, audit)
    print(f"audit: {args.output}")


if __name__ == "__main__":
    main()
