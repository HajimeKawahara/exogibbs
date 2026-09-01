"""Regression tests for the public Rocky Raccoon-like trace-Mg example."""

from __future__ import annotations

import ast
from pathlib import Path
import runpy

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    solve as solve_condensate_equilibrium,
    solve_profile as solve_condensate_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "demo_rocky_raccoon_trace_mg.py"
)
UPPER_WARM_FIXTURE_PATH = (
    Path(__file__).with_name("rocky_raccoon_upper_warm.npz")
)
LOWER_WARM_FIXTURE_PATH = (
    Path(__file__).with_name("rocky_raccoon_lower_warm.npz")
)
TRACE_MG_WARM_FIXTURE_PATH = (
    Path(__file__).with_name("rocky_raccoon_480_warm.npz")
)
BOUNDARY_CASES = {
    "backend_parity_step_378": (
        1188.1415292259892,
        4478.5100542051532,
        (
            0.9997289110989905,
            2.15359688402623e-16,
            7.206079734611716e-7,
            1.4412159471491326e-6,
            2.6892707708562876e-4,
            0.0,
        ),
    ),
    "backend_parity_step_380": (
        1181.3459388985098,
        4389.3877041264705,
        (
            0.9997290935901879,
            1.6631578573605774e-16,
            6.597612108647926e-7,
            1.3195224219399422e-6,
            2.68927126175761e-4,
            0.0,
        ),
    ),
    "backend_parity_step_380_gpu_trajectory": (
        1181.3459388985098,
        4389.3877041264705,
        (
            0.9997290935901882,
            1.6631596020972237e-16,
            6.5976121087567e-7,
            1.3195224219290858e-6,
            2.6892712617576094e-4,
            0.0,
        ),
    ),
    "equivalent_support_envelope_step_397": (
        1125.13099533797,
        3700.0044883165101,
        (
            0.9997301567050019,
            1.735221393093766e-17,
            3.0529428023151673e-7,
            6.105885605244276e-7,
            2.6892741215364603e-4,
            0.0,
        ),
    ),
    "terminal_support_envelope_after_step_397": (
        1127.1281529338487,
        3590.1106550090217,
        (
            0.9997302762741915,
            1.1504960536426218e-17,
            2.6542716229454013e-7,
            5.308543246120143e-7,
            2.689274443177579e-4,
            0.0,
        ),
    ),
    "support_release_trace_partition": (
        1029.4199562443821,
        2709.5257545305749,
        (
            0.9997308692704813,
            2.0670115092041315e-19,
            6.770856018415504e-8,
            1.354171203802319e-7,
            2.6892760383375976e-4,
            0.0,
        ),
    ),
    "trace_capacity": (
        1173.1942732095774,
        4132.5213914599017,
        (
            0.9997295583246257,
            7.589000706073114e-17,
            5.0480806048163e-7,
            1.0096161210834609e-6,
            2.689272511893248e-4,
            0.0,
        ),
    ),
    "rank_deficient_support": (
        1269.1589798706555,
        5643.1822694059156,
        (
            0.9997253184018701,
            3.774168837574554e-15,
            1.9184958230696084e-6,
            3.83699164992498e-6,
            2.689261106501034e-4,
            0.0,
        ),
    ),
    "amount_gauge": (
        1334.4049016146876,
        6495.780683442079,
        (
            0.9997209426096871,
            1.964558116000134e-14,
            3.3774855697403563e-6,
            6.7549711591333905e-6,
            2.689249335620061e-4,
            0.0,
        ),
    ),
    "optimizer_limit": (
        1561.8193557386803,
        11290.04441816559,
        (
            0.9996545882751542,
            6.928615382247818e-12,
            2.5501542246828624e-5,
            5.100309142229654e-5,
            2.68907084246017e-4,
            0.0,
        ),
    ),
    "mixed_charge_budget_step_702": (
        475.01010900904657,
        172.55859783339542,
        (
            0.9997310723415463,
            6.300502379398082e-47,
            3.5801015757132734e-17,
            8.331738184445885e-17,
            0.000268927658459876,
            0.0,
        ),
    ),
    "failed_basic_support_release_step_774": (
        386.57556568831939,
        83.689430815806617,
        (
            0.9997310723415466,
            2.139116395339677e-56,
            3.504937031478576e-27,
            1.1715327176146462e-17,
            0.00026892765845987606,
            0.0,
        ),
    ),
}
TRACE_DEPLETION_BOUNDARY = (
    203.06986826073876,
    8.7214641233652035,
    np.asarray(
        (
            0.9997310723415466,
            1.3681948091591687e-93,
            4.137051394836369e-84,
            1.1715327169136589e-17,
            0.00026892765845987606,
            0.0,
        ),
        dtype=np.float64,
    ),
)


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        EXAMPLE_PATH,
        run_name="rocky_raccoon_trace_mg_example_test_module",
    )


@pytest.fixture(scope="module")
def setup(example):
    return example["build_reduced_setup"]()


@pytest.fixture(scope="module")
def solved(example, setup):
    profile = example["solve_trace_mg_profile"](setup)
    audit = example["audit_trace_mg_profile"](setup, profile)
    return profile, audit


@pytest.fixture(scope="module")
def boundary_profiles(setup):
    profiles = {}
    for name, (temperature, pressure, inventory) in BOUNDARY_CASES.items():
        profiles[name] = solve_condensate_profile(
            setup,
            T=jnp.asarray([temperature], dtype=jnp.float64),
            P=jnp.asarray([pressure], dtype=jnp.float64),
            b=jnp.asarray(inventory, dtype=jnp.float64),
            options=CondensateEquilibriumOptions(
                rainout=True,
                profile_method="scan_hot_from_bottom",
                return_diagnostics=True,
            ),
            return_diagnostics=True,
        )
    return profiles


@pytest.fixture(scope="module")
def upper_warm_state():
    with np.load(UPPER_WARM_FIXTURE_PATH) as fixture:
        return {name: np.asarray(fixture[name]).copy() for name in fixture.files}


@pytest.fixture(scope="module")
def lower_warm_state():
    with np.load(LOWER_WARM_FIXTURE_PATH) as fixture:
        return {name: np.asarray(fixture[name]).copy() for name in fixture.files}


@pytest.fixture(scope="module")
def trace_mg_warm_state():
    with np.load(TRACE_MG_WARM_FIXTURE_PATH) as fixture:
        return {name: np.asarray(fixture[name]).copy() for name in fixture.files}


def test_example_is_main_guarded_and_compiles() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))

    compile(source, str(EXAMPLE_PATH), "exec")
    guarded_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
            for statement in node.body
        )
    ]
    assert len(guarded_calls) == 1


def test_setup_has_exact_network_without_reference_gases(example, setup) -> None:
    assert setup.elements == example["ELEMENTS"]
    assert setup.gas_species == example["GAS_SPECIES"]
    assert setup.condensate_species == example["CONDENSATE_SPECIES"]
    assert setup.formula_matrix.shape == (6, 70)
    assert setup.formula_matrix_cond.shape == (6, 14)
    assert np.linalg.matrix_rank(np.asarray(setup.formula_matrix)) == 6
    assert example["REFERENCE_GAS_SPECIES"].isdisjoint(setup.gas_species)

    charge = np.asarray(setup.formula_matrix)[-1]
    assert np.any(charge > 0.0)
    assert np.any(charge < 0.0)
    inventory = example["ELEMENT_INVENTORY"]
    assert 0.0 < inventory[setup.elements.index("Mg")] < 1.0e-11
    assert inventory[setup.elements.index("e-")] == 0.0


def test_public_profile_resolves_positive_trace_magnesium(
    example,
    setup,
    solved,
) -> None:
    profile, audit = solved
    layer = profile.layers[0]

    assert profile.rainout
    assert profile.method == "scan_hot_from_bottom"
    assert layer.converged
    assert audit["accepted"]
    assert audit["reference_gas_species_present"] == ()
    assert audit["gas_magnesium"] > 0.0
    assert audit["condensed_magnesium"] > 0.0
    assert audit["maximum_relative_budget_residual"] <= 1.0e-8
    assert abs(audit["charge_residual"]) <= 1.0e-14
    assert audit["gas_magnesium"] + audit[
        "condensed_magnesium"
    ] == pytest.approx(audit["target_magnesium"], rel=1.0e-10)


@pytest.mark.parametrize("case_name", tuple(BOUNDARY_CASES))
def test_public_profile_certifies_exact_rocky_boundary(
    case_name,
    boundary_profiles,
) -> None:
    profile = boundary_profiles[case_name]
    layer = profile.layers[0]

    assert profile.rainout
    assert profile.method == "scan_hot_from_bottom"
    assert layer.converged
    assert layer.status == "converged"
    assert np.all(np.isfinite(np.asarray(layer.gas_n)))
    assert np.all(np.asarray(layer.gas_n) >= 0.0)
    assert np.all(np.isfinite(np.asarray(layer.condensate_amounts)))
    assert np.all(np.asarray(layer.condensate_amounts) >= 0.0)
    assert np.sum(np.asarray(layer.gas_x)) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        np.asarray(profile.rainout_abundance_scale),
        [1.0],
    )

    rainout = layer.diagnostics["rainout"]
    floorless = rainout["floorless_budget_certification"]
    assert floorless["accepted"]
    assert floorless["maximum_positive_relative_residual"] <= (
        floorless["relative_tolerance"]
    )
    lifecycle = layer.diagnostics["fixed_support_v2"]
    caller_audit = lifecycle["caller_gauge_zero_barrier_kkt"]
    assert caller_audit["accepted"]
    assert caller_audit["gas_stationarity_max_abs"] <= 1.0e-8
    assert caller_audit["active_condensate_driving_max_abs"] <= 1.0e-8
    assert caller_audit["inactive_condensate_violation_max_abs"] <= 1.0e-8
    assert caller_audit["budget_scaled_max_abs"] <= 1.0e-8
    assert caller_audit["total_density_scaled_abs"] <= 1.0e-8


def test_public_amount_gauge_boundary_uses_exact_polish_rescue(
    boundary_profiles,
) -> None:
    profile = boundary_profiles["amount_gauge"]
    lifecycle = profile.layers[0].diagnostics["fixed_support_v2"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert profile.diagnostics["layers"][0]["attempts"][-1][
        "lifecycle_outcome"
    ] == (
        lifecycle["outcome"]
    )
    assert lifecycle["zero_barrier_initializer"]["source"] == (
        "fixed_support_terminal_state"
    )
    fallback = lifecycle["pre_pdipm_zero_barrier_fallback"]
    assert not fallback["eligible"]
    assert not fallback["attempted"]
    assert fallback["skip_reason"] == "terminal_state_initializer_preferred"
    assert lifecycle["zero_barrier_active_support_polish"]["accepted"]


def test_public_trace_capacity_boundary_uses_pre_pdipm_initializer(
    boundary_profiles,
) -> None:
    profile = boundary_profiles["trace_capacity"]
    lifecycle = profile.layers[0].diagnostics["fixed_support_v2"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert lifecycle["zero_barrier_initializer"]["source"] == (
        "pre_pdipm_finite_support_state"
    )
    reduction = lifecycle["finite_barrier_initial_support_reduction"]
    assert reduction["applied"]
    assert reduction["initial_support_nullity"] == 2
    assert reduction["output_support_nullity"] == 0
    assert set(reduction["output_support_indices"]).issubset(
        reduction["initial_support_indices"]
    )
    assert reduction["output_support_rank"] == len(
        reduction["output_support_indices"]
    )
    assert reduction["output_scaled_inventory_residual_max_abs"] <= (
        reduction["scaled_inventory_residual_tolerance"]
    )
    fallback = lifecycle["pre_pdipm_zero_barrier_fallback"]
    assert fallback["eligible"]
    assert fallback["attempted"]
    assert fallback["internal_accepted"]
    assert fallback["caller_gauge_accepted"]
    assert fallback["accepted"]
    assert fallback["trace_capacity"]["trace_capacity_detected"]
    assert (
        fallback["trace_capacity"]["minimum_capacity_to_barrier_ratio"]
        < 1.0
    )
    assert lifecycle["zero_barrier_active_support_polish"]["accepted"]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_pre_pdipm_initializer_expands_the_initial_support_envelope(
    boundary_profiles,
) -> None:
    lifecycle = boundary_profiles[
        "equivalent_support_envelope_step_397"
    ].layers[0].diagnostics["fixed_support_v2"]
    fallback = lifecycle["pre_pdipm_zero_barrier_fallback"]
    initializer = lifecycle["zero_barrier_initializer"]
    envelope = initializer["initial_support_envelope"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert initializer["source"] == "pre_pdipm_finite_support_state"
    assert fallback["eligible"]
    assert fallback["attempted"]
    assert fallback["accepted"]
    assert envelope["expanded"]
    assert envelope["initializer_state_preserved"]
    assert envelope["added_support_amounts_zero"]
    assert envelope["source_support_indices"] == (
        lifecycle["finite_barrier_initial_support_reduction"][
            "output_support_indices"
        ]
    )
    assert len(envelope["envelope_support_indices"]) > len(
        envelope["source_support_indices"]
    )
    assert envelope["added_support_indices"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    assert polish["accepted"]
    assert polish["basic_support_reduction"][
        "initial_support_indices"
    ] == envelope["envelope_support_indices"]
    assert polish["basic_support_reduction"]["initial_support_nullity"] > 0


def test_terminal_initializer_expands_the_initial_support_envelope(
    boundary_profiles,
) -> None:
    lifecycle = boundary_profiles[
        "terminal_support_envelope_after_step_397"
    ].layers[0].diagnostics["fixed_support_v2"]
    initializer = lifecycle["zero_barrier_initializer"]
    envelope = initializer["initial_support_envelope"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert initializer["source"] == "fixed_support_terminal_state"
    assert envelope["expanded"]
    assert envelope["initializer_state_preserved"]
    assert envelope["added_support_amounts_zero"]
    assert envelope["added_support_indices"]
    fallback = lifecycle["pre_pdipm_zero_barrier_fallback"]
    assert not fallback["eligible"]
    assert fallback["skip_reason"] == "terminal_state_initializer_preferred"
    polish = lifecycle["zero_barrier_active_support_polish"]
    assert polish["accepted"]
    assert polish["basic_support_reduction"][
        "initial_support_indices"
    ] == envelope["envelope_support_indices"]
    assert polish["basic_support_reduction"]["initial_support_nullity"] > 0
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_support_release_relaxes_only_the_initializer_partition(
    boundary_profiles,
) -> None:
    lifecycle = boundary_profiles[
        "support_release_trace_partition"
    ].layers[0].diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    reduction = polish["basic_support_reduction"]
    release = polish["support_release_portfolio"]
    generation = release["candidate_generation"]
    closure = polish["exact_active_set_closure"]
    alternatives = polish["alternative_basic_support_portfolio"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert reduction["applied"]
    assert reduction["initial_support_nullity"] > 0
    assert reduction["output_support_nullity"] == 0
    assert release["enabled"]
    assert release["eligible"]
    assert release["attempted"]
    assert not release["accepted"]
    assert release["local_kkt_selected"]
    assert release["role"] == "initializer_only"
    assert not release["condensate_inventory_preserved"]
    assert release["final_physical_audit_authoritative"]
    assert alternatives["support_release_function_evaluation_reserve"] > 0
    assert alternatives["portfolio_function_evaluation_limit"] > 0

    source = tuple(generation["source_support_indices"])
    selected = tuple(release["selected_support_indices"])
    assert source == tuple(reduction["output_support_indices"])
    assert generation["source_support_rank"] == len(source)
    assert set(selected) < set(source)

    rounds = closure["rounds"]
    assert closure["accepted"]
    assert closure["termination_reason"] == "accepted"
    assert rounds[0]["support_release_selected"]
    assert tuple(rounds[0]["support_release_indices"]) == selected
    assert rounds[0]["local_kkt_passed"]
    assert rounds[0]["action"] == "add_inactive_phase"
    assert rounds[0]["added_support_index"] is not None
    assert rounds[-1]["accepted"]
    assert closure["cumulative_function_evaluations"] == sum(
        round_report["function_evaluations"] for round_report in rounds
    )
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_warm_boundary_uses_signed_zero_regularized_portfolio(
    setup,
    upper_warm_state,
) -> None:
    target = upper_warm_state["target_inventory"]
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray(
            upper_warm_state["warm_gas_ln_n"], dtype=jnp.float64
        ),
        gas_ntot=jnp.asarray(
            upper_warm_state["warm_gas_ntot"], dtype=jnp.float64
        ),
    )
    layer = solve_condensate_equilibrium(
        setup,
        T=float(upper_warm_state["target_temperature"]),
        P=float(upper_warm_state["target_pressure"]),
        b=jnp.asarray(target, dtype=jnp.float64),
        init=initial,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
    )

    assert target[setup.elements.index("e-")] == 0.0
    charge = np.asarray(setup.formula_matrix)[-1]
    assert np.any(charge < 0.0)
    assert np.any(charge > 0.0)
    assert layer.converged
    assert layer.status == "converged"

    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    regularization = polish["initializer_regularization"]
    closure = polish["exact_active_set_closure"]
    caller_audit = lifecycle["caller_gauge_zero_barrier_kkt"]

    assert regularization["applied"]
    assert regularization["element_potential_recomputed"]
    assert regularization["reduced_primary_full_rank_fit"]
    assert regularization["reduced_primary_structure_eligible"]
    assert regularization["reduced_primary_structure_reason"] == "eligible"
    assert closure["accepted"]
    assert closure["termination_reason"] == "accepted"
    assert tuple(closure["final_support_indices"]) == (1, 8)
    assert 8 in closure["added_support_indices"]
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert caller_audit["accepted"]

    portfolio = polish["normalized_gas_reduced_initializer_portfolio"]
    if portfolio["raw_retry_reason"] is not None:
        assert portfolio["raw_retry_attempted"]
        assert portfolio["raw_retry_reason"] == (
            "self_reopening_support_drop"
        )
        assert portfolio["deferred_initializer"] == "capacity_regularized"
        assert portfolio["selected_initializer"] == "unregularized"
        assert portfolio["attempts"][0][
            "self_reopening_dropped_support_indices"
        ]


def test_public_warm_boundary_ignores_signed_rows_for_gas_capacity(
    setup,
    lower_warm_state,
) -> None:
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray(
            lower_warm_state["warm_gas_ln_n"], dtype=jnp.float64
        ),
        gas_ntot=jnp.asarray(
            lower_warm_state["warm_gas_ntot"], dtype=jnp.float64
        ),
    )
    layer = solve_condensate_equilibrium(
        setup,
        T=float(lower_warm_state["target_temperature"]),
        P=float(lower_warm_state["target_pressure"]),
        b=jnp.asarray(
            lower_warm_state["target_inventory"], dtype=jnp.float64
        ),
        init=initial,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
    )

    assert layer.converged
    assert layer.status == "converged"
    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    closure = polish["exact_active_set_closure"]
    rounds = closure["rounds"]

    assert closure["accepted"]
    assert closure["termination_reason"] == "accepted"
    assert tuple(closure["final_support_indices"]) == (1, 8)
    regularization = polish["initializer_regularization"]
    assert regularization["monotone_constraint_row_mask"] == (
        True,
        True,
        True,
        True,
        True,
        False,
    )
    assert not rounds[0][
        "normalized_dimensionless_unit_restart_eligible"
    ]
    assert rounds[0]["normalized_dimensionless_unit_restart_reason"] == (
        "local_kkt_already_satisfied"
    )
    assert not rounds[0][
        "normalized_dimensionless_unit_restart_attempted"
    ]
    assert rounds[0]["final_physical_audit_authoritative"]
    assert rounds[0]["selected_normalized_initializer"] == (
        "capacity_regularized"
    )
    assert rounds[0]["selected_normalized_variable_scaling"] == (
        "initializer_relative"
    )
    assert rounds[0][
        "regularized_normalized_initializer_attempt_count"
    ] == 1
    assert rounds[0]["action"] == "add_inactive_phase"
    assert rounds[0]["added_support_index"] == 8
    pivot = rounds[0]["rank_one_simplex_pivot"]
    assert pivot["applied"]
    assert pivot["leaving_support_index"] == 9
    assert tuple(pivot["candidate_support_indices"]) == (1, 8)
    assert rounds[-1]["accepted"]
    assert closure["cumulative_function_evaluations"] == sum(
        round_report["function_evaluations"] for round_report in rounds
    )
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_trace_mg_boundary_restarts_stalled_normalized_solve(
    setup,
    trace_mg_warm_state,
) -> None:
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray(
            trace_mg_warm_state["warm_gas_ln_n"], dtype=jnp.float64
        ),
        gas_ntot=jnp.asarray(
            trace_mg_warm_state["warm_gas_ntot"], dtype=jnp.float64
        ),
    )
    layer = solve_condensate_equilibrium(
        setup,
        T=float(trace_mg_warm_state["target_temperature"]),
        P=float(trace_mg_warm_state["target_pressure"]),
        b=jnp.asarray(
            trace_mg_warm_state["target_inventory"], dtype=jnp.float64
        ),
        init=initial,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
    )

    assert layer.converged
    assert layer.status == "converged"
    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    closure = polish["exact_active_set_closure"]
    rounds = closure["rounds"]

    assert closure["accepted"]
    assert closure["termination_reason"] == "accepted"
    assert tuple(closure["final_support_indices"]) == (1, 8)
    assert rounds[0]["normalized_dimensionless_unit_restart_eligible"]
    assert rounds[0]["normalized_dimensionless_unit_restart_attempted"]
    assert rounds[0]["selected_normalized_initializer"] == (
        "capacity_regularized"
    )
    assert rounds[0]["selected_normalized_variable_scaling"] == (
        "dimensionless_unit"
    )
    assert rounds[0][
        "regularized_normalized_initializer_attempt_count"
    ] == 2
    assert rounds[0]["action"] == "add_inactive_phase"
    assert rounds[0]["added_support_index"] == 8
    assert rounds[-1]["accepted"]
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_rank_deficient_boundary_selects_an_exact_basic_support(
    boundary_profiles,
    setup,
) -> None:
    lifecycle = boundary_profiles[
        "rank_deficient_support"
    ].layers[0].diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]

    assert lifecycle["outcome"] == "zero_barrier_active_support_rescued"
    assert polish["accepted"]
    reduction = polish["basic_support_reduction"]
    assert reduction["initial_support_nullity"] == 2
    portfolio = polish["alternative_basic_support_portfolio"]
    selected_support = tuple(polish["final_support_indices"])
    assert np.linalg.matrix_rank(
        np.asarray(setup.formula_matrix_cond)[:, selected_support]
    ) == len(selected_support)
    closure = polish["exact_active_set_closure"]
    if reduction["applied"]:
        assert not portfolio["attempted"]
    else:
        assert portfolio["attempted"]
        assert portfolio["local_kkt_selected"] or portfolio["accepted"]
        assert closure["rounds"][0]["selected_numerical_formulation"] == (
            "alternative_basic_support_normalized_gas_reduced_linear_amounts"
        )
    assert closure["termination_reason"] == "accepted"


def test_public_mixed_charge_budget_boundary_uses_log_fallback(
    boundary_profiles,
) -> None:
    layer = boundary_profiles["mixed_charge_budget_step_702"].layers[0]
    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    alternative = polish["alternative_basic_support_portfolio"]

    assert layer.converged
    assert tuple(layer.condensate_support_indices) == (1, 8)
    assert polish["selected_numerical_formulation"] == (
        "alternative_basic_support_reduced_log_domain"
    )
    assert alternative["accepted"]
    assert alternative["selected_support_indices"] == (1, 8)
    assert alternative["selected_formulation"] == "reduced_log_domain"
    selected = next(
        attempt
        for attempt in alternative["solve_attempts"]
        if attempt["accepted"]
    )
    assert selected["solve"]["log_budget_rows"] == (0, 1, 2, 3, 4)
    assert selected["solve"]["linear_budget_rows"] == (5,)
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_failed_basic_support_boundary_releases_a_proper_face(
    boundary_profiles,
) -> None:
    layer = boundary_profiles[
        "failed_basic_support_release_step_774"
    ].layers[0]
    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    alternative = polish["alternative_basic_support_portfolio"]
    release = polish["support_release_portfolio"]
    closure = polish["exact_active_set_closure"]

    assert layer.converged
    assert tuple(layer.condensate_support_indices) == (1, 8)
    assert not alternative["accepted"]
    assert alternative["support_release_source_indices"] == (1, 4)
    assert alternative["support_release_source_indices"] == tuple(
        alternative["candidate_generation"]["feasible_support_indices"][0]
    )
    assert release["trigger"] == "failed_basic_support_alternatives_rejected"
    assert release["source"] == "first_alternative_basic_support_candidate"
    assert release["source_support_indices"] == (1, 4)
    assert release["prefer_log_domain"]
    assert release["role"] == "initializer_only"
    assert release["final_physical_audit_authoritative"]
    assert not release["accepted"]
    assert release["local_kkt_selected"]
    assert release["selected_support_indices"] == (1,)
    assert release["selected_formulation"] == "reduced_log_domain"
    release_limit = release["support_release_function_evaluation_limit"]
    assert alternative["downstream_function_evaluation_reserve"] == (
        2 * release_limit
    )
    assert (
        alternative["portfolio_function_evaluation_limit"]
        + alternative["downstream_function_evaluation_reserve"]
        == closure["function_evaluation_limit"]
    )
    assert release["outer_closure_function_evaluation_reserve"] >= (
        release_limit
    )
    assert closure["accepted"]
    assert closure["termination_reason"] == "accepted"
    assert closure["added_support_indices"] == (8,)
    assert closure["final_support_indices"] == (1, 8)
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert closure["rounds"][0]["selected_numerical_formulation"] == (
        "support_release_reduced_log_domain"
    )
    assert closure["rounds"][0]["action"] == "add_inactive_phase"
    assert closure["rounds"][0]["added_support_index"] == 8
    assert closure["rounds"][-1]["accepted"]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_trace_depletion_uses_optimizer_directed_support_release(
    boundary_profiles,
    setup,
) -> None:
    anchor = boundary_profiles[
        "failed_basic_support_release_step_774"
    ].layers[0]
    temperature, pressure, inventory = TRACE_DEPLETION_BOUNDARY
    gas_amounts = np.asarray(anchor.gas_n, dtype=np.float64).copy()
    formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    physical = np.asarray(
        [
            str(element).strip().lower() not in {"e-", "electron"}
            for element in setup.elements
        ],
        dtype=bool,
    )
    active = physical & (inventory > 0.0)
    gas_amounts *= float(np.sum(inventory[active])) / float(
        np.sum((formula @ gas_amounts)[active])
    )
    gas_total = float(np.sum(gas_amounts))
    gas_amounts = np.maximum(gas_amounts, gas_total * 1.0e-300)
    layer = solve_condensate_equilibrium(
        setup,
        T=temperature,
        P=pressure,
        b=jnp.asarray(inventory, dtype=jnp.float64),
        init=CondensateEquilibriumInit(
            gas_ln_n=jnp.log(jnp.asarray(gas_amounts)),
            gas_ntot=jnp.asarray(np.sum(gas_amounts)),
        ),
        options=CondensateEquilibriumOptions(return_diagnostics=True),
    )

    assert layer.converged
    assert tuple(layer.condensate_support_indices) == (1, 8)
    lifecycle = layer.diagnostics["fixed_support_v2"]
    polish = lifecycle["zero_barrier_active_support_polish"]
    alternative = polish["alternative_basic_support_portfolio"]
    directed = alternative["optimizer_directed_support_release"]
    release = polish["support_release_portfolio"]
    closure = polish["exact_active_set_closure"]

    assert directed["selected"]
    assert directed["selected_support_indices"] == (0, 1, 8)
    assert directed["nonpositive_support_indices"] == (0,)
    assert alternative["selected_support_release_source"] == (
        "optimizer_terminated_nonpositive_alternative_basis"
    )
    assert alternative["selected_support_release_source_indices"] == (
        0,
        1,
        8,
    )
    assert release["source"] == (
        "optimizer_terminated_nonpositive_alternative_basis"
    )
    assert release["source_support_indices"] == (0, 1, 8)
    assert release["prefer_log_domain"]
    assert release["accepted"]
    assert release["selected_support_indices"] == (1, 8)
    release_limit = release["support_release_function_evaluation_limit"]
    assert alternative["downstream_function_evaluation_reserve"] == (
        2 * release_limit
    )
    assert release["outer_closure_function_evaluation_reserve"] >= (
        release_limit
    )
    assert closure["accepted"]
    assert closure["cumulative_function_evaluations"] <= closure[
        "function_evaluation_limit"
    ]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_public_optimizer_limit_boundary_uses_physical_certificate(
    boundary_profiles,
) -> None:
    lifecycle = boundary_profiles["optimizer_limit"].layers[0].diagnostics[
        "fixed_support_v2"
    ]
    polish = lifecycle["zero_barrier_active_support_polish"]

    assert polish["accepted"]
    assert polish["polish_schema"] == (
        "exogibbs_zero_barrier_active_support_polish_v2"
    )
    assert polish["exact_active_set_closure"]["schema"] == (
        "exogibbs_zero_barrier_exact_active_set_closure_v2"
    )
    assert polish["optimizer_termination_eligible"]
    assert polish["physical_root_certified"]
    if polish["optimizer_success"]:
        assert polish["acceptance_source"] == "optimizer_success"
    else:
        assert polish["optimizer_status"] == 0
        assert polish["acceptance_source"] == (
            "physical_kkt_after_optimizer_limit"
        )
    assert polish["gas_stationarity_max_abs"] <= (
        polish["stationarity_tolerance"]
    )
    assert polish["active_condensate_driving_max_abs"] <= (
        polish["stationarity_tolerance"]
    )
    assert polish["inactive_condensate_violation_max_abs"] <= (
        polish["support_closure_tolerance"]
    )
    assert polish["budget_scaled_max_abs"] <= polish["budget_tolerance"]
    assert polish["total_density_scaled_abs"] <= (
        polish["total_density_tolerance"]
    )
