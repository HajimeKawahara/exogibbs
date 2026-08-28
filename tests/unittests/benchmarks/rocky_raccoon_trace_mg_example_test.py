"""Regression tests for the public Rocky Raccoon-like trace-Mg example."""

from __future__ import annotations

import ast
from pathlib import Path
import runpy

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "demo_rocky_raccoon_trace_mg.py"
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
