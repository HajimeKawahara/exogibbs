"""Tests for condensate result acceptance."""

import math
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.equilibrium.condensate.acceptance import (
    accept_condensate_result_state,
)
from exogibbs.equilibrium.condensate.setup import (
    build_condensate_chemical_setup,
)


def _setup():
    gas = ChemicalSetup(
        formula_matrix=jnp.eye(2),
        hvector_func=lambda temperature: jnp.zeros(2),
        elements=("H", "O"),
        species=("H", "O"),
    )
    condensate = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0], [1.0]]),
        hvector_func=lambda temperature: jnp.zeros(1),
        elements=("H", "O"),
        species=("H2O_s",),
    )
    return build_condensate_chemical_setup(
        gas_setup=gas,
        condensate_setup=condensate,
    )


def test_acceptance_rejects_nan_gas_amount_without_budget_gate() -> None:
    state = accept_condensate_result_state(
        setup=_setup(),
        gas_ln_n=jnp.asarray([jnp.nan, 0.0]),
        condensate_amounts=jnp.zeros(1),
        solver_success=True,
        diagnostics=None,
        element_inventory_target=None,
        enable_full_condensate_budget_residual_gate=False,
        full_condensate_budget_relative_tolerance=1.0e-3,
    )

    assert state.status == "not_converged"
    assert state.acceptance_tier == "physical_amount_state_failed"


def test_acceptance_rejects_negative_condensate_when_budget_closes() -> None:
    state = accept_condensate_result_state(
        setup=_setup(),
        gas_ln_n=jnp.asarray([math.log(3.0), math.log(2.0)]),
        condensate_amounts=jnp.asarray([-1.0]),
        solver_success=True,
        diagnostics=None,
        element_inventory_target=jnp.asarray([1.0, 1.0]),
        enable_full_condensate_budget_residual_gate=True,
        full_condensate_budget_relative_tolerance=1.0e-3,
    )

    assert state.diagnostics[
        "full_condensate_budget_residual_gate"
    ]["accepted"]
    assert state.status == "not_converged"
    assert state.acceptance_tier == "physical_amount_state_failed"


def test_acceptance_checks_negative_condensate_before_float32_cast() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root / "src"), environment.get("PYTHONPATH", ""))
    )
    script = """
import numpy as np
import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    build_condensate_chemical_setup,
    build_condensate_equilibrium_result_from_solver_payload,
)

gas = ChemicalSetup(
    formula_matrix=jnp.ones((1, 2)),
    hvector_func=lambda temperature: jnp.zeros(2),
    elements=("H",),
    species=("H", "H2"),
)
condensate = ChemicalSetup(
    formula_matrix=jnp.ones((1, 1)),
    hvector_func=lambda temperature: jnp.zeros(1),
    elements=("H",),
    species=("H_s",),
)
setup = build_condensate_chemical_setup(
    gas_setup=gas,
    condensate_setup=condensate,
)
state = build_condensate_equilibrium_result_from_solver_payload(
    setup=setup,
    gas_ln_n=jnp.zeros(2),
    support_indices=(0,),
    support_amounts=np.asarray([-1.0e-100], dtype=np.float64),
    selected_route="head_v2_fixed_support_lifecycle",
    solver_success=True,
    diagnostics=None,
    element_inventory_target=None,
    enable_full_condensate_budget_residual_gate=False,
    full_condensate_budget_relative_tolerance=1.0e-3,
)
assert state.status == "not_converged"
assert state.acceptance_tier == "physical_amount_state_failed"

state = build_condensate_equilibrium_result_from_solver_payload(
    setup=setup,
    gas_ln_n=np.asarray([-1.0e100, 0.0], dtype=np.float64),
    support_indices=(),
    support_amounts=(),
    selected_route="head_v2_fixed_support_lifecycle",
    solver_success=True,
    diagnostics=None,
    element_inventory_target=None,
    enable_full_condensate_budget_residual_gate=False,
    full_condensate_budget_relative_tolerance=1.0e-3,
)
assert state.status == "not_converged"
assert state.acceptance_tier == "physical_amount_state_failed"

state = build_condensate_equilibrium_result_from_solver_payload(
    setup=setup,
    gas_ln_n=jnp.zeros(2),
    support_indices=(0,),
    support_amounts=np.asarray([1.0e100], dtype=np.float64),
    selected_route="head_v2_fixed_support_lifecycle",
    solver_success=True,
    diagnostics={
        "fixed_support_v2": {
            "zero_barrier_active_support_polish": {"accepted": True}
        }
    },
    element_inventory_target=None,
    enable_full_condensate_budget_residual_gate=False,
    full_condensate_budget_relative_tolerance=1.0e-3,
)
assert state.status == "not_converged"
assert state.acceptance_tier == "physical_amount_state_failed"
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
