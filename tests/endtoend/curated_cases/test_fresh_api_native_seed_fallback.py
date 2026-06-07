"""Fresh API regressions for the native condensate seed fallback."""

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
from exogibbs.condensates.head_route_standard_gate import CONVERGED_WITH_CAVEAT
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)


FRESH_CASES = (
    (
        "solar_silicate_first_condensation__T1400_P0p1",
        1400.0,
        0.1,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
    ),
    (
        "solar_silicate_first_condensation__T1500_P1",
        1500.0,
        1.0,
        ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
    ),
    (
        "solar_water_condensation__T300_P1",
        300.0,
        1.0,
        ("H2O(s,l)",),
    ),
    (
        "solar_water_condensation__T300_P1",
        300.0,
        1.0,
        ("H2O(s,l)",),
    ),
)


def test_four_curated_rows_succeed_through_fresh_api_native_seed_fallback() -> None:
    setup = condensate_chemical_setup(silent=True)
    element_budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}

    for case_id, temperature, pressure, support_names in FRESH_CASES:
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

        assert result.status == CONVERGED_WITH_CAVEAT
        assert result.converged is True
        assert result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
        assert result.condensate_support_names == support_names
        assert bool(jnp.all(jnp.isfinite(result.gas_ln_n)))
        assert bool(jnp.all(jnp.isfinite(result.condensate_amounts)))
        assert result.diagnostics is not None
        assert result.diagnostics["solver_success"] is True
        assert result.diagnostics["restricted_solver_success"] is False
        assert result.diagnostics["native_seed_fallback"]["accepted"] is True
        assert (
            result.diagnostics["native_seed_fallback"][
                "fastchem4_trace_public_runtime_constructor_inputs_used"
            ]
            is False
        )
