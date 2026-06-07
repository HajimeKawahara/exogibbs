"""Fresh API regressions for the curated condensate HEAD route rows."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from jax import config
import jax.numpy as jnp

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumOptions,
    condensate_equilibrium,
)
from exogibbs.condensates.head_route_standard_gate import CONVERGED, CONVERGED_WITH_CAVEAT
from exogibbs.condensates.initialization_policy import recommend_budget_preserving_seed_amounts
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)

ACCEPTED_STATUSES = {CONVERGED, CONVERGED_WITH_CAVEAT}


def _solar_budget(setup):
    return jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)


def _carbon_rich_budget(setup):
    budget = _solar_budget(setup)
    element_index = {name: index for index, name in enumerate(setup.elements)}
    return budget.at[element_index["C"]].set(2.0 * budget[element_index["O"]])


FRESH_CURATED_CASES = (
    (
        "solar_silicate_first_condensation__T1400_P0p1",
        1400.0,
        0.1,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
        _solar_budget,
    ),
    (
        "solar_silicate_first_condensation__T1500_P1",
        1500.0,
        1.0,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
        _solar_budget,
    ),
    (
        "solar_water_condensation__T300_P1",
        300.0,
        1.0,
        ("H2O(s,l)",),
        _solar_budget,
    ),
    (
        "solar_water_condensation__T300_P0p1",
        300.0,
        0.1,
        ("H2O(s,l)",),
        _solar_budget,
    ),
    (
        "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P1",
        700.0,
        1.0,
        ("Fe(s,l)", "FeS(s,l)", "Ni(s,l)", "NiS(s,l)"),
        _solar_budget,
    ),
    (
        "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P0p1",
        700.0,
        0.1,
        ("Fe(s,l)", "FeS(s,l)", "Ni(s,l)", "NiS(s,l)"),
        _solar_budget,
    ),
    (
        "carbon_rich_graphite_window__T1300_P1_corrected",
        1300.0,
        1.0,
        ("C(s)",),
        _carbon_rich_budget,
    ),
    (
        "carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected",
        700.0,
        1.0,
        ("CaS(s)", "MgS(s)", "AlN(s)"),
        _carbon_rich_budget,
    ),
    (
        "SiO_s_condensate_window__T900_P0p1_corrected",
        900.0,
        0.1,
        ("SiO(s)",),
        _solar_budget,
    ),
    (
        "lowT_strong_condensation_budget_stress__T500_P1",
        500.0,
        1.0,
        ("H2O(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "FeS(s,l)"),
        _solar_budget,
    ),
    (
        "lowT_strong_condensation_budget_stress__T500_P0p1",
        500.0,
        0.1,
        ("H2O(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "FeS(s,l)"),
        _solar_budget,
    ),
    (
        "near_phase_boundary_support_sensitivity__T1490_P1",
        1490.0,
        1.0,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "CaTiO3(s)", "TiO2(s,l)"),
        _solar_budget,
    ),
    (
        "near_phase_boundary_support_sensitivity__T1510_P1",
        1510.0,
        1.0,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "CaTiO3(s)", "TiO2(s,l)"),
        _solar_budget,
    ),
    (
        "complex_heavy_element_or_boron_titanium_zirconium_case__T1100_P1_corrected",
        1100.0,
        1.0,
        ("TiO2(s,l)", "TiC(s,l)", "TiN(s,l)", "CaTiO3(s)"),
        _solar_budget,
    ),
)


def test_all_14_curated_rows_succeed_through_fresh_api() -> None:
    setup = condensate_chemical_setup(silent=True)
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}

    for case_id, temperature, pressure, support_names, budget_builder in FRESH_CURATED_CASES:
        element_budget = budget_builder(setup)
        support_indices = tuple(species_index[name] for name in support_names)
        result = condensate_equilibrium(
            setup,
            temperature,
            pressure,
            element_budget,
            support_indices=support_indices,
            support_amounts_init=tuple(1.0e-12 for _ in support_indices),
            options=CondensateEquilibriumOptions(
                case_id=case_id,
                return_diagnostics=True,
                max_inner_iterations=80,
                max_outer_iterations=20,
                max_positive_support_count=max(1, len(support_indices)),
                allow_empty_positive_support=False,
            ),
        )

        assert result.status in ACCEPTED_STATUSES, case_id
        assert result.converged is True
        assert bool(jnp.all(jnp.isfinite(result.gas_ln_n)))
        assert bool(jnp.all(jnp.isfinite(result.condensate_amounts)))
        assert result.diagnostics is not None
        assert result.diagnostics["solver_success"] is True
        assert result.diagnostics["acceptance_tier"] != "runtime_solver_failed"
        assert result.diagnostics["support_selection"]["fastchem4_trace_values_used"] is False
        assert (
            result.diagnostics["support_selection"][
                "fastchem4_public_values_used_as_constructor_inputs"
            ]
            is False
        )
        assert (
            result.diagnostics["head_route_lifecycle"][
                "fastchem4_trace_public_runtime_constructor_inputs_used"
            ]
            is False
        )


def test_water_mid_layer_lifecycle_failure_uses_head_route_v1_1_fallback() -> None:
    setup = condensate_chemical_setup(silent=True)
    element_budget = _solar_budget(setup)
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}
    support_indices = (species_index["H2O(s,l)"],)
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=element_budget,
        condensate_species_order=setup.condensate_species,
        support_indices=support_indices,
        seed_fraction=1.0e-3,
        max_seed_amount=1.0e-3,
        min_seed_amount=1.0e-300,
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_fresh_curated_profile_budget",
        },
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        0.1,
        element_budget,
        support_indices=support_indices,
        support_amounts_init=tuple(float(value) for value in seed.recommended_amounts),
        options=CondensateEquilibriumOptions(
            case_id="solar_water_condensation__T300_P0p1",
            return_diagnostics=True,
            max_inner_iterations=40,
            allow_empty_positive_support=False,
        ),
    )

    assert result.status == CONVERGED_WITH_CAVEAT
    assert result.converged is True
    assert result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
    assert result.diagnostics is not None
    assert result.diagnostics["solver_success"] is True
    assert result.diagnostics["restricted_solver_success"] is True
    assert result.diagnostics["native_seed_fallback"]["accepted"] is True
    assert (
        result.diagnostics["head_route_lifecycle"]["route_result"]["selected_route"]
        == "support_boundary_construction_required_before_selector"
    )
