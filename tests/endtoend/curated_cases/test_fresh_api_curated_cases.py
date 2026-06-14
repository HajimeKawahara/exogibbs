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
from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    case_id_for_profile,
    element_budget_for_profile,
)
from exogibbs.condensates.head_route_standard_gate import (
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
    NOT_CONVERGED,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)

ACCEPTED_STATUSES = {CONVERGED, CONVERGED_WITH_CAVEAT}
FULL_BUDGET_GATE_REJECTED_TIER = "full_condensate_element_budget_residual_failed"

EXPLICIT_SUPPORT_V1_4_EXPECTED_STATUSES = {
    "solar_silicate_first_condensation__T1400_P0p1": CONVERGED_WITH_CAVEAT,
    "solar_silicate_first_condensation__T1500_P1": CONVERGED_WITH_CAVEAT,
    "solar_water_condensation__T300_P1": CONVERGED_WITH_CAVEAT,
    "solar_water_condensation__T300_P0p1": CONVERGED_WITH_CAVEAT,
    "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P1": CONVERGED_WITH_CAVEAT,
    "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P0p1": NOT_CONVERGED,
    "carbon_rich_graphite_window__T1300_P1_corrected": NOT_CONVERGED,
    "carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected": NOT_CONVERGED,
    "SiO_s_condensate_window__T900_P0p1_corrected": NOT_CONVERGED,
    "lowT_strong_condensation_budget_stress__T500_P1": NOT_CONVERGED,
    "lowT_strong_condensation_budget_stress__T500_P0p1": CONVERGED_WITH_CAVEAT,
    "near_phase_boundary_support_sensitivity__T1490_P1": CONVERGED_WITH_CAVEAT,
    "near_phase_boundary_support_sensitivity__T1510_P1": CONVERGED_WITH_CAVEAT,
    "complex_heavy_element_or_boron_titanium_zirconium_case__T1100_P1_corrected": (
        CONVERGED_WITH_CAVEAT
    ),
}

SUPPORT_FREE_MIDLAYER_EXPECTED_ROUTES = {
    "solar_highT_no_condensate_gas_regression": "head_v1_empty_positive_support_gas_only",
    "solar_silicate_first_condensation": "m4310_full_promoted_policy_route",
    "solar_water_condensation": "m4310_full_promoted_policy_route",
    "solar_metal_sulfide_or_Fe_Ni_S_region": "m4310_full_promoted_policy_route",
    "carbon_rich_graphite_window": "m4310_full_promoted_policy_route",
    "carbon_rich_CaS_MgS_AlN_window": "m4310_full_promoted_policy_route",
    "SiO_s_condensate_window": "m4310_full_promoted_policy_route",
    "lowT_strong_condensation_budget_stress": "m4310_full_promoted_policy_route",
    "near_phase_boundary_support_sensitivity": "m4310_full_promoted_policy_route",
    "complex_heavy_element_or_boron_titanium_zirconium_case": (
        "m4310_full_promoted_policy_route"
    ),
}

SUPPORT_FREE_MIDLAYER_EXPECTED_STATUSES = {
    "solar_highT_no_condensate_gas_regression": CONVERGED,
    "solar_silicate_first_condensation": CONVERGED,
    "solar_water_condensation": CONVERGED,
    "solar_metal_sulfide_or_Fe_Ni_S_region": CONVERGED,
    "carbon_rich_graphite_window": CONVERGED,
    "carbon_rich_CaS_MgS_AlN_window": CONVERGED,
    "SiO_s_condensate_window": CONVERGED,
    "lowT_strong_condensation_budget_stress": CONVERGED,
    "near_phase_boundary_support_sensitivity": CONVERGED,
    "complex_heavy_element_or_boron_titanium_zirconium_case": CONVERGED,
}

SUPPORT_FREE_STAGED_RETRY_REGRESSION_ROWS = (
    ("solar_water_condensation", 7),
)


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

        expected_status = EXPLICIT_SUPPORT_V1_4_EXPECTED_STATUSES[case_id]
        assert result.status == expected_status, case_id
        assert result.converged is (expected_status in ACCEPTED_STATUSES)
        assert bool(jnp.all(jnp.isfinite(result.gas_ln_n)))
        assert bool(jnp.all(jnp.isfinite(result.condensate_amounts)))
        assert result.diagnostics is not None
        gate = result.diagnostics["full_condensate_budget_residual_gate"]
        assert gate["accepted"] is (expected_status in ACCEPTED_STATUSES)
        if expected_status == NOT_CONVERGED:
            assert result.acceptance_tier == FULL_BUDGET_GATE_REJECTED_TIER
            pre_gate_status = result.diagnostics[
                "pre_full_condensate_budget_gate_status"
            ]
            assert pre_gate_status in ACCEPTED_STATUSES
        else:
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


def test_water_mid_layer_default_api_preserves_v1_3_route_with_v1_4_gate() -> None:
    setup = condensate_chemical_setup(silent=True)
    element_budget = _solar_budget(setup)

    result = condensate_equilibrium(
        setup,
        300.0,
        0.1,
        element_budget,
        options=CondensateEquilibriumOptions(
            case_id="solar_water_condensation__T300_P0p1",
            return_diagnostics=True,
            max_inner_iterations=40,
        ),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.acceptance_tier == "tier_1_tight_residual_production_adjacent_candidate"
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert len(result.condensate_support_names) > 1
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    assert result.diagnostics["solver_success"] is True
    assert result.diagnostics["restricted_solver_success"] is True
    assert (
        result.diagnostics["support_selection"]["solver_inputs"]["seed_initialization_policy"]
        == "max_density"
    )


def test_support_free_curated_midlayers_use_retry_defaults() -> None:
    setup = condensate_chemical_setup(silent=True)
    route_counts = {
        "head_v1_empty_positive_support_gas_only": 0,
        "m4310_full_promoted_policy_route": 0,
        "native_budget_seed_fallback_budget_tradeoff": 0,
    }
    status_counts = {CONVERGED: 0, CONVERGED_WITH_CAVEAT: 0, NOT_CONVERGED: 0}

    for family, definition in FRESH_CURATED_PROFILES.items():
        layer_index = len(definition.temperatures) // 2
        temperature = definition.temperatures[layer_index]
        pressure = definition.pressures[layer_index]
        result = condensate_equilibrium(
            setup,
            float(temperature),
            float(pressure),
            element_budget_for_profile(setup, definition),
            options=CondensateEquilibriumOptions(
                case_id=(
                    f"{case_id_for_profile(definition, temperature, pressure)}"
                    "__support_free"
                ),
                return_diagnostics=True,
                max_inner_iterations=definition.max_inner_iterations,
            ),
        )

        expected_status = SUPPORT_FREE_MIDLAYER_EXPECTED_STATUSES[family]
        assert result.status == expected_status, family
        assert result.converged is (expected_status in ACCEPTED_STATUSES)
        assert result.selected_route == SUPPORT_FREE_MIDLAYER_EXPECTED_ROUTES[family]
        assert result.diagnostics is not None
        gate = result.diagnostics["full_condensate_budget_residual_gate"]
        assert gate["accepted"] is (expected_status in ACCEPTED_STATUSES)
        if expected_status == NOT_CONVERGED:
            assert result.acceptance_tier == FULL_BUDGET_GATE_REJECTED_TIER
        support_selection = result.diagnostics["support_selection"]
        assert support_selection["selection_mode"] == "activity_driven_support_outer_loop"
        assert (
            support_selection["solver_inputs"]["seed_initialization_policy"]
            == "max_density"
        )
        assert support_selection["fastchem4_trace_values_used"] is False
        retry_report = result.diagnostics.get("head_route_center_gate_retry")
        if retry_report is not None:
            assert retry_report["triggered"] is True
            assert retry_report["accepted"] is True
            assert retry_report["initial_stopped_reason"] == "current_barrier_not_centered"
            assert retry_report["center_tolerance_multiplier"] == 1.0e11
        residual_retry_report = result.diagnostics.get(
            "head_route_residual_worsening_retry"
        )
        if residual_retry_report is not None:
            assert residual_retry_report["triggered"] is True
            assert residual_retry_report["initial_stopped_reason"] == "no_p_armijo_trial"
            assert residual_retry_report["residual_worsening_tolerance"] == 2.0e-2
        soft_restoration_retry = result.diagnostics.get(
            "head_route_soft_restoration_retry"
        )
        if soft_restoration_retry is not None:
            assert soft_restoration_retry["triggered"] is True
            assert soft_restoration_retry["center_tolerance_multiplier"] == 1.0e11
            ipopt_h_type_retry = result.diagnostics.get("head_route_ipopt_h_type_retry")
            if (
                result.selected_route != "native_budget_seed_fallback_budget_tradeoff"
                and ipopt_h_type_retry is None
            ):
                assert soft_restoration_retry["accepted"] is True
        ipopt_h_type_retry = result.diagnostics.get("head_route_ipopt_h_type_retry")
        if ipopt_h_type_retry is not None:
            assert ipopt_h_type_retry["triggered"] is True
            assert ipopt_h_type_retry["trial_acceptance_policy"] == "ipopt_persistent_h_type"
            assert ipopt_h_type_retry["center_tolerance_multiplier"] == 1.0e11
            if result.selected_route != "native_budget_seed_fallback_budget_tradeoff":
                assert ipopt_h_type_retry["accepted"] is True
        support_cap_retry = result.diagnostics.get("support_cap_retry")
        if support_cap_retry is not None:
            assert support_cap_retry["triggered"] is True
            assert support_cap_retry["route_promoted"] is True
            assert support_cap_retry["accepted"] is result.converged
        route_counts[result.selected_route] += 1
        status_counts[result.status] += 1

    assert route_counts == {
        "head_v1_empty_positive_support_gas_only": 1,
        "m4310_full_promoted_policy_route": 9,
        "native_budget_seed_fallback_budget_tradeoff": 0,
    }
    assert status_counts == {CONVERGED: 10, CONVERGED_WITH_CAVEAT: 0, NOT_CONVERGED: 0}


def test_support_free_staged_retry_regression_layers_promote_to_primary() -> None:
    setup = condensate_chemical_setup(silent=True)

    for family, layer_index in SUPPORT_FREE_STAGED_RETRY_REGRESSION_ROWS:
        definition = FRESH_CURATED_PROFILES[family]
        temperature = float(definition.temperatures[layer_index])
        pressure = float(definition.pressures[layer_index])
        result = condensate_equilibrium(
            setup,
            temperature,
            pressure,
            element_budget_for_profile(setup, definition),
            options=CondensateEquilibriumOptions(
                case_id=(
                    f"{case_id_for_profile(definition, temperature, pressure)}"
                    "__support_free_exception_regression"
                ),
                return_diagnostics=True,
                max_inner_iterations=definition.max_inner_iterations,
            ),
        )

        assert result.status == CONVERGED, family
        assert result.converged is True
        assert result.acceptance_tier == "tier_1_tight_residual_production_adjacent_candidate"
        assert result.selected_route == "m4310_full_promoted_policy_route"
        assert len(result.condensate_support_names) > 0
        assert result.diagnostics is not None
        gate = result.diagnostics["full_condensate_budget_residual_gate"]
        assert gate["accepted"] is True
        retry_report = result.diagnostics["support_growth_staging_retry"]
        assert retry_report["triggered"] is True
        assert retry_report["accepted"] is True
        assert retry_report["route_promoted"] is True
        assert retry_report["max_support_add_per_round"] == 64
        assert (
            retry_report["initial_selected_route"]
            == "native_budget_seed_fallback_budget_tradeoff"
        )
        support_selection = result.diagnostics["support_selection"]
        assert support_selection["selection_mode"] == "activity_driven_support_outer_loop"
        assert support_selection["fastchem4_trace_values_used"] is False


def test_water_explicit_support_primary_center_tolerance_opt_in_reaches_tier1() -> None:
    setup = condensate_chemical_setup(silent=True)
    element_budget = _solar_budget(setup)
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}
    support_indices = (species_index["H2O(s,l)"],)

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        element_budget,
        support_indices=support_indices,
        support_amounts_init=(1.0e-12,),
        options=CondensateEquilibriumOptions(
            case_id="solar_water_condensation__T300_P1",
            return_diagnostics=True,
            max_inner_iterations=80,
            max_outer_iterations=20,
            max_positive_support_count=1,
            allow_empty_positive_support=False,
            head_route_primary_center_tolerance_multiplier=4.0e8,
            head_route_primary_acceptance_guard="tight_weighted_components",
        ),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert result.acceptance_tier == "tier_1_tight_residual_production_adjacent_candidate"
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    lifecycle = result.diagnostics["head_route_lifecycle"]
    continuation = lifecycle["primary_execution_report"]["continuation_report"]
    assert continuation["converged_at_final_barrier"] is True
    assert continuation["center_tolerance_multiplier"] == 4.0e8
    guard = lifecycle["route_result"]["diagnostics"]["primary_acceptance_guard"]
    assert guard["accepted"] is True


def test_primary_acceptance_guard_blocks_tier1_when_components_are_loose() -> None:
    setup = condensate_chemical_setup(silent=True)
    element_budget = _solar_budget(setup)
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}
    support_indices = (species_index["H2O(s,l)"],)

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        element_budget,
        support_indices=support_indices,
        support_amounts_init=(1.0e-12,),
        options=CondensateEquilibriumOptions(
            case_id="solar_water_condensation__T300_P1",
            return_diagnostics=True,
            max_inner_iterations=80,
            max_outer_iterations=20,
            max_positive_support_count=1,
            allow_empty_positive_support=False,
            head_route_primary_center_tolerance_multiplier=4.0e8,
            head_route_primary_acceptance_guard="tight_weighted_components",
            head_route_primary_guard_max_condensate_stationarity=1.0,
        ),
    )

    assert result.status == CONVERGED_WITH_CAVEAT
    assert result.converged is True
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert result.acceptance_tier == "tier_3_raw_gas_caveat_diagnostic_only"
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    guard = result.diagnostics["head_route_lifecycle"]["route_result"]["diagnostics"][
        "primary_acceptance_guard"
    ]
    assert guard["accepted"] is False
    assert guard["component_checks"]["condensate"] is False
