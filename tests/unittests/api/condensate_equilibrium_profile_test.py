"""Production v2 condensate profile tests."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import exogibbs.api.condensate_equilibrium as condmod
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    CondensateChemicalSetup,
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
)
from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate.fixed_support.types import (
    KKTComponentNorms,
    TerminalStatus,
)


def _fake_setup() -> CondensateChemicalSetup:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (2,),
            dtype=jnp.float64,
        ),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (2,),
            dtype=jnp.float64,
        ),
        elements=("H", "O"),
        species=("H[s]", "O[s]"),
        metadata={},
    )
    return CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=gas_setup.formula_matrix,
        formula_matrix_cond=condensate_setup.formula_matrix,
        gas_species=gas_setup.species,
        condensate_species=condensate_setup.species,
        elements=gas_setup.elements,
    )


def test_head_v2_profile_expands_support_outside_solver_until_closed(
    monkeypatch,
):
    setup = _fake_setup()
    calls = []

    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        lambda **kwargs: (
            (0,),
            (0.2,),
            {"policy": "test_initial_support"},
        ),
    )

    def fake_run_fixed_support_profile(**kwargs):
        calls.append(kwargs)
        support = tuple(kwargs["buckets"][0].support_indices)
        closed = len(support) == 2
        condensate_amounts = (
            jnp.asarray([[0.2, 0.2]], dtype=jnp.float64)
            if closed
            else jnp.asarray([[0.2, 0.0]], dtype=jnp.float64)
        )
        zeros = jnp.zeros((1,), dtype=jnp.float64)
        return {
            "backend": "cpu",
            "compilation_seconds": 1.0,
            "execution_seconds": 0.1,
            "diagnostic_seconds": 0.01,
            "gas_log_amounts": jnp.log(
                jnp.asarray([[0.8, 0.8]], dtype=jnp.float64)
            ),
            "condensate_amounts": condensate_amounts,
            "total_gas_log_amount": jnp.log(
                jnp.asarray([1.6], dtype=jnp.float64)
            ),
            "element_potential": jnp.zeros((1, 2), dtype=jnp.float64),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.CONVERGED)],
                dtype=jnp.int32,
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([True]),
            "support_closed": jnp.asarray([closed]),
            "support_expansion_mask": jnp.asarray(
                [[False, not closed]],
                dtype=bool,
            ),
            "inactive_condensate_driving": jnp.asarray(
                [[0.0, 0.0 if closed else -1.0]],
                dtype=jnp.float64,
            ),
        }

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.batch."
            "run_fixed_support_profile"
        ),
        fake_run_fixed_support_profile,
    )
    monkeypatch.setattr(
        _lifecycle,
        "evaluate_profile_support_closure",
        lambda result, **kwargs: result,
    )

    def fake_zero_barrier_polish(**kwargs):
        amounts = np.asarray(
            kwargs["condensate_amounts_init"], dtype=np.float64
        ).copy()
        initial_support = tuple(kwargs["support_indices"])
        final_support = initial_support[:-1]
        amounts[initial_support[-1]] = 0.0
        return SimpleNamespace(
            accepted=True,
            gas_log_amounts=np.asarray(
                kwargs["gas_log_amounts_init"], dtype=np.float64
            ),
            condensate_amounts=amounts,
            total_gas_log_amount=float(
                kwargs["total_gas_log_amount_init"]
            ),
            element_potential=np.asarray(
                kwargs["element_potential_init"], dtype=np.float64
            ),
            support_indices=final_support,
            report={
                "accepted": True,
                "polish_schema": "unit_test",
                "initial_support_indices": initial_support,
                "final_support_indices": final_support,
                "dropped_support_indices": initial_support[-1:],
            },
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.8, 0.8], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(1.6, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.2,),
        element_potential=jnp.zeros((2,), dtype=jnp.float64),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    assert len(calls) == 2
    assert tuple(calls[0]["buckets"][0].support_indices) == (0,)
    assert tuple(calls[1]["buckets"][0].support_indices) == (0, 1)
    assert result.layers[0].converged
    assert result.layers[0].selected_route == (
        "head_v2_fixed_support_lifecycle"
    )
    assert result.layers[0].head_route_version == "v2.0"
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == "closed"
    assert lifecycle["independent_kkt_passed"]
    assert lifecycle["rounds"][0]["added_support_indices"] == (1,)
    assert lifecycle["support_indices_after_polish"] == (0,)
    assert result.layers[0].condensate_support_names == ("H[s]",)
    assert result.layers[0].condensate_amounts.tolist() == pytest.approx(
        [0.2, 0.0]
    )
    assert result.diagnostics["route"] == "head_v2"


@pytest.mark.parametrize(
    (
        "raw_support_closed",
        "exact_accepted",
        "expected_calls",
        "expected_outcome",
    ),
    (
        (True, True, 1, "zero_barrier_active_support_rescued"),
        (True, False, 1, "zero_barrier_active_support_polish_failed"),
        (False, True, 0, "fixed_support_failed"),
    ),
)
def test_head_v2_failed_closed_state_only_initializes_exact_polish(
    monkeypatch,
    raw_support_closed,
    exact_accepted,
    expected_calls,
    expected_outcome,
):
    setup = _fake_setup()
    exact_calls = []

    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        lambda **kwargs: (
            (0,),
            (0.2,),
            {"policy": "test_initial_support"},
        ),
    )

    def fake_run_fixed_support_profile(**kwargs):
        zeros = jnp.zeros((1,), dtype=jnp.float64)
        return {
            "backend": "cpu",
            "compilation_seconds": 0.0,
            "execution_seconds": 0.0,
            "diagnostic_seconds": 0.0,
            "gas_log_amounts": jnp.log(
                jnp.asarray([[0.8, 1.0]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.2, 0.0]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([1.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.zeros((1, 2), dtype=jnp.float64),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.NORMAL_DUAL_STEP_FAILED)],
                dtype=jnp.int32,
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros,
                jnp.asarray([1.676e-8], dtype=jnp.float64),
                zeros,
                zeros,
                zeros,
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([False]),
            "support_closed": jnp.asarray([raw_support_closed]),
            "support_expansion_mask": jnp.asarray(
                [[False, not raw_support_closed]], dtype=bool
            ),
            "inactive_condensate_driving": jnp.asarray(
                [[0.0, 1.0 if raw_support_closed else -1.0]],
                dtype=jnp.float64,
            ),
        }

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.batch."
            "run_fixed_support_profile"
        ),
        fake_run_fixed_support_profile,
    )
    monkeypatch.setattr(
        _lifecycle,
        "evaluate_profile_support_closure",
        lambda result, **kwargs: result,
    )

    def fake_zero_barrier_polish(**kwargs):
        exact_calls.append(kwargs)
        return SimpleNamespace(
            accepted=exact_accepted,
            gas_log_amounts=np.asarray(
                kwargs["gas_log_amounts_init"], dtype=np.float64
            ),
            condensate_amounts=np.asarray(
                kwargs["condensate_amounts_init"], dtype=np.float64
            ),
            total_gas_log_amount=float(
                kwargs["total_gas_log_amount_init"]
            ),
            element_potential=np.asarray(
                kwargs["element_potential_init"], dtype=np.float64
            ),
            support_indices=tuple(kwargs["support_indices"]),
            report={
                "accepted": exact_accepted,
                "polish_schema": "unit_test",
            },
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.8, 1.0], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(1.8, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.2,),
        element_potential=jnp.zeros((2,), dtype=jnp.float64),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    assert len(exact_calls) == expected_calls
    assert result.layers[0].converged is (
        raw_support_closed and exact_accepted
    )
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == expected_outcome
    assert not lifecycle["fixed_support_converged"]
    assert lifecycle["terminal_status_name"] == "NORMAL_DUAL_STEP_FAILED"
    assert not lifecycle["independent_kkt_passed"]
    initializer = lifecycle["zero_barrier_initializer"]
    assert initializer["eligible"] is raw_support_closed
    assert initializer["attempted"] is raw_support_closed
    assert initializer["role"] == "initializer_only"
    assert initializer["raw_noncondensate_kkt_passed"]
    assert initializer["rescue_attempted"] is raw_support_closed


def test_head_v2_rejects_hot_scan_method():
    setup = _fake_setup()

    with pytest.raises(ValueError, match="head_v2 currently supports"):
        condmod.condensate_equilibrium_profile(
            setup,
            T=np.asarray([1000.0]),
            P=np.asarray([1.0]),
            b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            method="scan_hot_from_top",
        )


def test_explicit_vmap_method_overrides_options_hot_scan(monkeypatch):
    setup = _fake_setup()
    expected = object()
    calls = []

    def fake_run_head_v2_profile(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        condmod,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        method="vmap_cold",
        options=CondensateEquilibriumOptions(
            profile_method="scan_hot_from_bottom",
        ),
    )

    assert result is expected
    assert len(calls) == 1


def test_head_v2_rejects_empty_profile():
    setup = _fake_setup()

    with pytest.raises(ValueError, match="at least one profile layer"):
        condmod.condensate_equilibrium_profile(
            setup,
            T=np.asarray([]),
            P=np.asarray([]),
            b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        )


def test_head_v2_empty_initial_support_uses_gas_only_outcome(
    monkeypatch,
):
    setup = _fake_setup()
    gas_ln_n = jnp.log(jnp.asarray([0.5, 0.5], dtype=jnp.float64))

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.solve.equilibrium",
        lambda *args, **kwargs: SimpleNamespace(
            ln_n=gas_ln_n,
            ntot=jnp.asarray(1.0, dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        (
            "exogibbs.condensates.support_selection_policy."
            "select_activity_driven_support_candidates"
        ),
        lambda **kwargs: SimpleNamespace(
            positive_support_indices=(),
            as_dict=lambda: {},
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
    )

    layer = result.layers[0]
    assert layer.converged
    assert layer.selected_route == "head_v2_gas_only_no_candidate"
    assert layer.condensate_support_indices.size == 0
    assert layer.diagnostics["fixed_support_v2"]["outcome"] == (
        "gas_only_no_candidate"
    )


def test_head_v2_independent_kkt_gate_rejects_failed_component():
    kkt = {
        "gas_stationarity": 1.0e-9,
        "condensate_stationarity": 1.0e-9,
        "budget_scaled": 1.0e-9,
        "complementarity": 1.0e-9,
        "total_density_scaled": 1.0e-9,
    }
    tolerances = {
        "stationarity_tolerance": 1.0e-8,
        "budget_tolerance": 1.0e-8,
        "complementarity_tolerance": 1.0e-8,
        "total_density_tolerance": 1.0e-8,
    }

    assert condmod._head_v2_kkt_passed(kkt, **tolerances)
    assert not condmod._head_v2_kkt_passed(
        {**kkt, "budget_scaled": 1.0e-7},
        **tolerances,
    )
