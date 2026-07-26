"""Native curated condensate profile definitions for demos and regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)


@dataclass(frozen=True)
class CuratedProfileDefinition:
    """Small native pressure/temperature profile for one curated family."""

    family: str
    temperatures: tuple[float, ...]
    pressures: tuple[float, ...]
    support_species: tuple[str, ...] = ()
    carbon_to_oxygen_ratio: float | None = None
    empty_condensate_support: bool = False
    seed_fraction: float = 1.0e-3
    max_seed_amount: float = 1.0e-3


def _logspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.logspace(start, stop, count))


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(start, stop, count))


FRESH_CURATED_PROFILES: Mapping[str, CuratedProfileDefinition] = {
    "solar_highT_no_condensate_gas_regression": CuratedProfileDefinition(
        family="solar_highT_no_condensate_gas_regression",
        temperatures=tuple(2200.0 for _ in range(18)),
        pressures=_logspace(-6.0, 2.0, 18),
        empty_condensate_support=True,
    ),
    "solar_silicate_first_condensation": CuratedProfileDefinition(
        family="solar_silicate_first_condensation",
        temperatures=_linspace(1600.0, 1300.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
    ),
    "solar_water_condensation": CuratedProfileDefinition(
        family="solar_water_condensation",
        temperatures=_linspace(360.0, 240.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("H2O(s,l)",),
    ),
    "solar_metal_sulfide_or_Fe_Ni_S_region": CuratedProfileDefinition(
        family="solar_metal_sulfide_or_Fe_Ni_S_region",
        temperatures=_linspace(850.0, 600.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("Fe(s,l)", "FeS(s,l)", "Ni(s,l)", "NiS(s,l)"),
    ),
    "carbon_rich_graphite_window": CuratedProfileDefinition(
        family="carbon_rich_graphite_window",
        temperatures=_linspace(1500.0, 1100.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("C(s)",),
        carbon_to_oxygen_ratio=2.0,
    ),
    "carbon_rich_CaS_MgS_AlN_window": CuratedProfileDefinition(
        family="carbon_rich_CaS_MgS_AlN_window",
        temperatures=_linspace(850.0, 600.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("CaS(s)", "MgS(s)", "AlN(s)"),
        carbon_to_oxygen_ratio=2.0,
    ),
    "SiO_s_condensate_window": CuratedProfileDefinition(
        family="SiO_s_condensate_window",
        temperatures=_linspace(1050.0, 750.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("SiO(s)",),
    ),
    "lowT_strong_condensation_budget_stress": CuratedProfileDefinition(
        family="lowT_strong_condensation_budget_stress",
        temperatures=_linspace(600.0, 350.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=(
            "H2O(s,l)",
            "MgSiO3(s,l)",
            "Mg2SiO4(s,l)",
            "Fe(s,l)",
            "FeS(s,l)",
        ),
    ),
    "near_phase_boundary_support_sensitivity": CuratedProfileDefinition(
        family="near_phase_boundary_support_sensitivity",
        temperatures=_linspace(1550.0, 1450.0, 9),
        pressures=_logspace(-1.0, 1.0, 9),
        support_species=(
            "MgSiO3(s,l)",
            "Mg2SiO4(s,l)",
            "Fe(s,l)",
            "CaTiO3(s)",
            "TiO2(s,l)",
        ),
    ),
    "complex_heavy_element_or_boron_titanium_zirconium_case": CuratedProfileDefinition(
        family="complex_heavy_element_or_boron_titanium_zirconium_case",
        temperatures=_linspace(1250.0, 950.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("TiO2(s,l)", "TiC(s,l)", "TiN(s,l)", "CaTiO3(s)"),
    ),
}


def fresh_profile_definition(family: str) -> CuratedProfileDefinition:
    """Return the native fresh-profile definition for a curated family."""

    try:
        return FRESH_CURATED_PROFILES[family]
    except KeyError as exc:
        raise ValueError(f"No fresh curated profile definition for family={family!r}.") from exc


def element_budget_for_profile(setup: Any, definition: CuratedProfileDefinition) -> jnp.ndarray:
    """Build the native element budget used by one fresh curated profile."""

    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    if definition.carbon_to_oxygen_ratio is not None:
        element_index = {name: index for index, name in enumerate(setup.elements)}
        budget = budget.at[element_index["C"]].set(
            float(definition.carbon_to_oxygen_ratio) * budget[element_index["O"]]
        )
    return budget


def pressure_label(pressure: float) -> str:
    """Return a filesystem-friendly pressure label."""

    return f"{pressure:g}".replace(".", "p").replace("-", "m")


def case_id_for_profile(
    definition: CuratedProfileDefinition,
    temperature: float,
    pressure: float,
) -> str:
    """Return the standard curated case id for one profile layer."""

    return (
        f"{definition.family}__T{int(round(float(temperature)))}"
        f"_P{pressure_label(float(pressure))}"
    )


def support_payload_for_profile(
    setup: Any,
    definition: CuratedProfileDefinition,
    budget: jnp.ndarray,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Build the explicit support payload used by curated v2 profiles."""

    if definition.empty_condensate_support:
        return (), ()
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}
    missing = [name for name in definition.support_species if name not in species_index]
    if missing:
        raise ValueError(
            f"Fresh curated profile {definition.family!r} references unknown condensates: "
            f"{missing}"
        )
    support_indices = tuple(species_index[name] for name in definition.support_species)
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=budget,
        condensate_species_order=setup.condensate_species,
        support_indices=support_indices,
        seed_fraction=definition.seed_fraction,
        max_seed_amount=definition.max_seed_amount,
        min_seed_amount=1.0e-300,
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_fresh_curated_profile_budget",
        },
    )
    return support_indices, tuple(float(value) for value in seed.recommended_amounts)


__all__ = (
    "CuratedProfileDefinition",
    "FRESH_CURATED_PROFILES",
    "case_id_for_profile",
    "element_budget_for_profile",
    "fresh_profile_definition",
    "pressure_label",
    "support_payload_for_profile",
)
