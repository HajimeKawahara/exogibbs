"""Regression tests for the public Rocky Raccoon-like trace-Mg example."""

from __future__ import annotations

import ast
from pathlib import Path
import runpy

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "demo_rocky_raccoon_trace_mg.py"
)
BOUNDARY_CASES = {
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
}


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
    assert reduction["output_support_indices"] == (1, 8)
    assert reduction["output_support_nullity"] == 0
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
