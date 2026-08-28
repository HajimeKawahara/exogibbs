"""Contracts for the dependent condensate-rainout profile scheduler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math

import jax.numpy as jnp
import numpy as np
import pytest

import exogibbs.api.condensate_equilibrium as condmod
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate.policy import (
    fixed_support_v2_production_policy,
)
from exogibbs.equilibrium.condensate.profile import (
    _accept_trace_capacity_candidate,
    _conservation_rainout_inventory,
    _gas_warm_start_for_next_layer,
    _rainout_gauge_scales,
    _scale_initial_guess,
    _trace_capacity_acceptance_report,
)
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.types import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
)


def _fake_setup() -> CondensateChemicalSetup:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (2,), dtype=jnp.float64
        ),
        elements=("H", "O", "e-"),
        species=("H", "O"),
        metadata={},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0], [0.0], [0.0]], dtype=jnp.float64
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (1,), dtype=jnp.float64
        ),
        elements=("H", "O", "e-"),
        species=("H[s]",),
        metadata={},
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def _fake_setup_with_absent_oxygen() -> CondensateChemicalSetup:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (3,), dtype=jnp.float64
        ),
        elements=("H", "C", "O", "e-"),
        species=("H", "C", "HO"),
        metadata={},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0], [0.0], [1.0], [0.0]], dtype=jnp.float64
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (1,), dtype=jnp.float64
        ),
        elements=("H", "C", "O", "e-"),
        species=("HO[s]",),
        metadata={},
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def _fake_setup_with_reduced_and_compatible_condensates(
) -> CondensateChemicalSetup:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (3,), dtype=jnp.float64
        ),
        elements=("H", "C", "O", "e-"),
        species=("H", "C", "HO"),
        metadata={},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.zeros(
            (2,), dtype=jnp.float64
        ),
        elements=("H", "C", "O", "e-"),
        species=("HO[s]", "H[s]"),
        metadata={},
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def _layer_result(
    gas_n: Sequence[float],
    *,
    condensate_amount: float = 0.0,
    converged: bool = True,
) -> CondensateEquilibriumResult:
    amounts = jnp.asarray(gas_n, dtype=jnp.float64)
    total = jnp.sum(amounts)
    return CondensateEquilibriumResult(
        gas_ln_n=jnp.log(amounts),
        gas_n=amounts,
        gas_x=amounts / total,
        gas_ntot=total,
        condensate_amounts=jnp.asarray(
            [condensate_amount], dtype=jnp.float64
        ),
        condensate_support_indices=jnp.asarray([], dtype=jnp.int32),
        condensate_support_names=(),
        acceptance_tier="strict" if converged else "rejected",
        selected_route="head_v2_fixed_support_lifecycle",
        status="converged" if converged else "not_converged",
        converged=converged,
        diagnostics={},
    )


def _one_layer_profile(
    gas_n: Sequence[float],
    *,
    condensate_amount: float = 0.0,
    converged: bool = True,
) -> CondensateEquilibriumProfileResult:
    return CondensateEquilibriumProfileResult(
        layers=(
            _layer_result(
                gas_n,
                condensate_amount=condensate_amount,
                converged=converged,
            ),
        ),
        method="vmap_cold",
    )


def _trace_test_candidate(
    *,
    gas_n: Sequence[float],
    condensate_amount: float,
    budget_target: Sequence[float],
    budget_reconstructed: Sequence[float],
    budget_gate_enabled: bool = True,
) -> CondensateEquilibriumResult:
    diagnostics = {
        "fixed_support_v2": {
            "outcome": "fixed_support_failed",
            "terminal_status_name": "NORMAL_MAX_ITER",
            "support_closed": True,
            "final_state_values_finite": True,
            "independent_kkt": {
                "gas_stationarity": 0.0,
                "condensate_stationarity": 2.0e-6,
                "budget_scaled": 1.0e-15,
                "complementarity": 1.0e-15,
                "total_density_scaled": 1.0e-15,
            },
        },
        "full_condensate_budget_residual_gate": {
            "accepted": True,
            "enabled": budget_gate_enabled,
            "relative_tolerance": 1.0e-3,
            "element_budget_target": tuple(budget_target),
            "element_budget_reconstructed": tuple(budget_reconstructed),
        },
        "acceptance_tier": "fixed_support_v2_solver_failed",
    }
    return replace(
        _layer_result(
            gas_n,
            condensate_amount=condensate_amount,
            converged=False,
        ),
        condensate_support_indices=jnp.asarray([0], dtype=jnp.int32),
        condensate_support_names=("H[s]",),
        diagnostics=diagnostics,
    )


def test_rainout_amount_scaling_shifts_barrier_epsilon() -> None:
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray([0.0]),
        gas_ntot=jnp.asarray(1.0),
        condensate_amounts=jnp.asarray([0.25]),
        support_amounts=(0.25,),
        barrier_epsilon=jnp.asarray(-11.0),
    )

    scaled = _scale_initial_guess(initial, 1.0e8)

    assert float(scaled.barrier_epsilon) == pytest.approx(
        -11.0 + math.log(1.0e8)
    )


def test_rainout_warm_start_floor_scales_with_inventory() -> None:
    base = _gas_warm_start_for_next_layer(
        np.asarray([1.0, 0.0]),
        inventory_sum=1.0,
        conservation_inventory_sum=1.0,
    )
    scaled = _gas_warm_start_for_next_layer(
        np.asarray([1.0e-12, 0.0]),
        inventory_sum=1.0e-12,
        conservation_inventory_sum=1.0e-12,
    )

    np.testing.assert_allclose(
        np.asarray(scaled.gas_ln_n) - math.log(1.0e-12),
        np.asarray(base.gas_ln_n),
        rtol=1.0e-12,
    )


def test_rainout_scans_bottom_to_top_and_returns_original_order(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    temperatures = np.asarray([100.0, 200.0, 300.0])
    pressures = np.asarray([1.0, 10.0, 100.0])
    initial_inventory = jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64)
    condensate_by_temperature = {
        300.0: 0.2,
        200.0: 0.1,
        100.0: 0.0,
    }
    lnphi_func = lambda temperature, pressure_bar, mole_fractions: jnp.zeros(2)
    calls = []

    def fake_run_head_v2_profile(**kwargs):
        temperature = float(kwargs["temperatures"][0])
        scaled_budget = np.asarray(kwargs["b"], dtype=np.float64)
        abundance_scale = float(np.sum(scaled_budget))
        caller_budget = scaled_budget / abundance_scale
        condensate_amount = (
            condensate_by_temperature[temperature] * abundance_scale
        )
        gas_n = scaled_budget[:2].copy()
        gas_n[0] -= condensate_amount
        if temperature == 100.0:
            # A finite accepted gas residual must not alter a gas-only
            # rainout inventory.
            gas_n[0] *= 1.0005
        calls.append(
            {
                "temperature": temperature,
                "pressure": float(kwargs["pressures"][0]),
                "b": caller_budget,
                "scale": abundance_scale,
                "init": kwargs["explicit_inits"][0],
                "lnphi_func": kwargs["lnphi_func"],
            }
        )
        return _one_layer_profile(
            gas_n,
            condensate_amount=condensate_amount,
        )

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=temperatures,
        P=pressures,
        b=initial_inventory,
        options=CondensateEquilibriumOptions(
            rainout=True,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
        lnphi_func=lnphi_func,
    )

    assert [call["temperature"] for call in calls] == [300.0, 200.0, 100.0]
    assert [call["pressure"] for call in calls] == [100.0, 10.0, 1.0]
    assert all(call["lnphi_func"] is lnphi_func for call in calls)
    np.testing.assert_allclose(calls[0]["b"], [0.6, 0.4, 0.0])
    np.testing.assert_allclose(calls[1]["b"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(calls[2]["b"], [4.0 / 9.0, 5.0 / 9.0, 0.0])
    np.testing.assert_allclose(
        [call["scale"] for call in calls],
        [1.0, 1.0, 1.0],
    )

    assert result.method == "scan_hot_from_bottom"
    assert result.rainout
    assert result.diagnostics["processing_indices"] == (2, 1, 0)
    np.testing.assert_allclose(
        np.asarray([layer.gas_n for layer in result.layers]),
        [[(4.0 / 9.0) * 1.0005, 5.0 / 9.0], [0.4, 0.5], [0.4, 0.4]],
    )
    np.testing.assert_allclose(
        np.asarray(result.element_inventory_target),
        [[4.0 / 9.0, 5.0 / 9.0, 0.0], [0.5, 0.5, 0.0], [0.6, 0.4, 0.0]],
    )
    np.testing.assert_allclose(
        np.asarray(result.gas_element_inventory),
        [
            [(4.0 / 9.0) * 1.0005, 5.0 / 9.0, 0.0],
            [0.4, 0.5, 0.0],
            [0.4, 0.4, 0.0],
        ],
    )
    np.testing.assert_allclose(
        np.asarray(result.rainout_element_inventory_out),
        [
            [4.0 / 9.0, 5.0 / 9.0, 0.0],
            [4.0 / 9.0, 5.0 / 9.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(result.rainout_element_inventory_out), axis=1),
        1.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.batched_arrays["element_inventory_target"]),
        np.asarray(result.element_inventory_target),
    )
    np.testing.assert_allclose(
        np.asarray(result.batched_arrays["condensate_amounts"])[:, 0],
        [0.0, 0.1, 0.2],
    )
    np.testing.assert_array_equal(
        np.asarray(result.rainout_abundance_scale),
        [1.0, 1.0, 1.0],
    )
    # Propagation uses budget minus condensates, not the slightly imperfect
    # gas reconstruction at the top gas-only layer.
    np.testing.assert_array_equal(
        np.asarray(result.rainout_element_inventory_out)[0],
        np.asarray(result.element_inventory_target)[0],
    )


def test_rainout_initializer_receives_current_budget_previous_gas_and_index(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    temperatures = np.asarray([100.0, 200.0, 300.0])
    pressures = np.asarray([1.0, 10.0, 100.0])
    bottom_user_init = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.55, 0.45], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(1.0, dtype=jnp.float64),
    )
    requests = []
    solver_inits = []

    class RecordingInitializer:
        def __call__(self, request):
            requests.append(request)
            if request.user_init is not None:
                return request.user_init
            if request.previous_solution is not None:
                return request.previous_solution
            return CondensateEquilibriumInit()

    def fake_run_head_v2_profile(**kwargs):
        solver_inits.append(kwargs["explicit_inits"][0])
        temperature = float(kwargs["temperatures"][0])
        scaled_budget = np.asarray(kwargs["b"], dtype=np.float64)
        scale = float(np.sum(scaled_budget))
        condensate_amount = {
            300.0: 0.2,
            200.0: 0.1,
            100.0: 0.0,
        }[temperature] * scale
        gas_n = scaled_budget[:2].copy()
        gas_n[0] -= condensate_amount
        return _one_layer_profile(
            gas_n, condensate_amount=condensate_amount
        )

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    condmod.condensate_equilibrium_profile(
        setup,
        T=temperatures,
        P=pressures,
        b=jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64),
        init=(None, None, bottom_user_init),
        initializer=RecordingInitializer(),
        options=CondensateEquilibriumOptions(rainout=True),
    )

    assert [request.layer_index for request in requests] == [2, 1, 0]
    assert [float(request.T) for request in requests] == [300.0, 200.0, 100.0]
    assert requests[0].user_init is bottom_user_init
    assert requests[0].previous_solution is None
    assert solver_inits[0] is bottom_user_init
    np.testing.assert_allclose(
        np.exp(np.asarray(solver_inits[0].gas_ln_n)),
        np.asarray([0.55, 0.45]),
    )
    assert all(request.previous_solution is not None for request in requests[1:])
    assert requests[1].user_init is None
    np.testing.assert_allclose(np.asarray(requests[0].b), [0.6, 0.4, 0.0])
    np.testing.assert_allclose(np.asarray(requests[1].b), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(
        np.asarray(requests[2].b), [4.0 / 9.0, 5.0 / 9.0, 0.0]
    )
    # The carried state contains gas only and is normalized to the next budget.
    assert requests[1].previous_solution.condensate_amounts is None
    assert requests[1].previous_solution.support_amounts is None
    np.testing.assert_allclose(
        np.exp(np.asarray(requests[1].previous_solution.gas_ln_n)),
        [0.5, 0.5],
    )


def test_rainout_depletion_snap_prevents_trace_element_resurrection(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    initial_h = 1.0e-20

    def fake_run_head_v2_profile(**kwargs):
        temperature = float(kwargs["temperatures"][0])
        scaled_budget = np.asarray(kwargs["b"], dtype=np.float64)
        scale = float(np.sum(scaled_budget))
        if temperature == 300.0:
            # Remove all resolvable H.  The raw gas solution deliberately
            # retains a much smaller positive log-space amount.
            return _one_layer_profile(
                (1.0e-30 * scale, scaled_budget[1]),
                condensate_amount=scaled_budget[0],
            )
        # Reproduce the former L569 failure mode: the full-network gas solve
        # creates an absolute-floor H amount from an exact-zero input row.
        return _one_layer_profile((1.0e-25 * scale, scaled_budget[1]))

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([100.0, 200.0, 300.0]),
        P=np.asarray([1.0, 10.0, 100.0]),
        b=jnp.asarray([initial_h, 1.0, 0.0], dtype=jnp.float64),
        options=CondensateEquilibriumOptions(
            rainout=True,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    targets = np.asarray(result.element_inventory_target)
    outputs = np.asarray(result.rainout_element_inventory_out)
    raw_gas = np.asarray(result.gas_element_inventory)
    assert targets[2, 0] == initial_h
    assert outputs[2, 0] == 0.0
    np.testing.assert_array_equal(targets[:2, 0], [0.0, 0.0])
    np.testing.assert_array_equal(outputs[:2, 0], [0.0, 0.0])
    assert np.all(raw_gas[:2, 0] > 0.0)
    propagation = result.layers[1].diagnostics["rainout"]["propagation"]
    assert propagation["ignored_gas_species_indices"] == (0,)
    certification = result.layers[1].diagnostics["rainout"][
        "floorless_budget_certification"
    ]
    assert certification["accepted"]
    assert certification["zero_budget_handling"] == "reduced_propagation"
    assert (
        certification[
            "raw_solver_zero_budget_maximum_absolute_reconstructed"
        ]
        > 0.0
    )
    bottom_propagation = result.layers[2].diagnostics["rainout"][
        "propagation"
    ]
    assert bottom_propagation["depletion_snap_element_indices"] == (0,)
    assert (
        bottom_propagation["depletion_snap_amount"][0]
        <= bottom_propagation["depletion_snap_error_bound"][0]
    )


def test_rainout_initially_zero_element_cannot_change_other_abundances(
    monkeypatch,
) -> None:
    setup = _fake_setup_with_absent_oxygen()
    artifact_fraction = 0.1
    warm_gas_by_temperature = {}

    def fake_run_head_v2_profile(**kwargs):
        temperature = float(kwargs["temperatures"][0])
        initial_gas = kwargs["explicit_inits"][0].gas_ln_n
        warm_gas_by_temperature[temperature] = (
            None
            if initial_gas is None
            else np.exp(np.asarray(initial_gas, dtype=np.float64))
        )
        scaled_budget = np.asarray(kwargs["b"], dtype=np.float64)
        scale = float(np.sum(scaled_budget))
        gas_n = jnp.asarray(
            [
                scaled_budget[0],
                scaled_budget[1],
                artifact_fraction * scale,
            ],
            dtype=jnp.float64,
        )
        gas_ntot = jnp.sum(gas_n)
        layer = CondensateEquilibriumResult(
            gas_ln_n=jnp.log(gas_n),
            gas_n=gas_n,
            gas_x=gas_n / gas_ntot,
            gas_ntot=gas_ntot,
            condensate_amounts=jnp.asarray(
                [artifact_fraction * scale], dtype=jnp.float64
            ),
            condensate_support_indices=jnp.asarray([0], dtype=jnp.int32),
            condensate_support_names=("HO[s]",),
            acceptance_tier="strict",
            selected_route="head_v2_fixed_support_lifecycle",
            status="converged",
            converged=True,
            diagnostics={},
        )
        return CondensateEquilibriumProfileResult(
            layers=(layer,),
            method="vmap_cold",
        )

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([100.0, 200.0]),
        P=np.asarray([1.0, 10.0]),
        b=jnp.asarray([0.5, 0.5, 0.0, 0.0], dtype=jnp.float64),
        options=CondensateEquilibriumOptions(
            rainout=True,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(
        np.asarray(result.element_inventory_target),
        [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        np.asarray(result.rainout_element_inventory_out),
        [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
    )
    assert np.all(np.asarray(result.gas_element_inventory)[:, 2] > 0.0)
    assert warm_gas_by_temperature[200.0] is None
    top_warm_gas = warm_gas_by_temperature[100.0]
    assert top_warm_gas is not None
    np.testing.assert_allclose(top_warm_gas[:2], [0.5, 0.5])
    assert 0.0 < top_warm_gas[2] < 1.0e-280

    raw_condensate_inventory = np.asarray(
        result.batched_arrays["raw_condensate_element_inventory"]
    )
    propagation_condensate_inventory = np.asarray(
        result.batched_arrays[
            "rainout_propagation_condensate_element_inventory"
        ]
    )
    legacy_condensate_inventory = np.asarray(
        result.batched_arrays["condensate_element_inventory"]
    )
    np.testing.assert_allclose(
        raw_condensate_inventory,
        [[artifact_fraction, 0.0, artifact_fraction, 0.0]] * 2,
    )
    np.testing.assert_array_equal(propagation_condensate_inventory, 0.0)
    np.testing.assert_array_equal(
        legacy_condensate_inventory,
        propagation_condensate_inventory,
    )
    assert result.diagnostics["batched_array_aliases"] == {
        "condensate_element_inventory": (
            "rainout_propagation_condensate_element_inventory"
        )
    }
    for layer in result.layers:
        rainout = layer.diagnostics["rainout"]
        certification = rainout["floorless_budget_certification"]
        propagation = rainout["propagation"]
        assert certification["accepted"]
        assert certification["zero_element_indices"] == (2,)
        assert certification["zero_budget_handling"] == "reduced_propagation"
        assert propagation["ignored_gas_species_indices"] == (2,)
        assert propagation["ignored_condensate_species_indices"] == (0,)
        assert propagation["no_condensate_removal"]
        assert propagation["source"] == (
            "b_current_minus_"
            "rainout_propagation_condensate_element_inventory"
        )
        assert propagation["raw_condensate_element_inventory"] == pytest.approx(
            (artifact_fraction, 0.0, artifact_fraction, 0.0)
        )
        assert propagation[
            "rainout_propagation_condensate_element_inventory"
        ] == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_rainout_depletion_snap_uses_reduced_crosscheck_error() -> None:
    setup = _fake_setup_with_reduced_and_compatible_condensates()
    gas_n = jnp.asarray([0.1, 0.5, 0.1], dtype=jnp.float64)
    condensate_amounts = jnp.asarray([0.1, 0.4], dtype=jnp.float64)
    result = CondensateEquilibriumResult(
        gas_ln_n=jnp.log(gas_n),
        gas_n=gas_n,
        gas_x=gas_n / jnp.sum(gas_n),
        gas_ntot=jnp.sum(gas_n),
        condensate_amounts=condensate_amounts,
        condensate_support_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        condensate_support_names=("HO[s]", "H[s]"),
        acceptance_tier="strict",
        selected_route="head_v2_fixed_support_lifecycle",
        status="converged",
        converged=True,
        diagnostics={},
    )

    propagation = _conservation_rainout_inventory(
        setup=setup,
        result=result,
        conserved_mask=np.asarray([True, True, True, False]),
        normalization_mask=np.asarray([True, True, False, False]),
        inventory_target=np.asarray([0.5, 0.5, 0.0, 0.0]),
        inventory_sum=1.0,
        roundoff_multiplier=64.0,
    )

    # Raw HO/HO[s] artifacts create a 0.2 H residual, but both are excluded
    # from the exact-zero-compatible state.  That raw residual must not erase
    # the physical 0.1 H remainder left after H[s] condensation.
    assert propagation["crosscheck_residual"][0] == pytest.approx(0.2)
    assert propagation["propagation_crosscheck_residual"][0] == pytest.approx(
        0.0
    )
    assert propagation["snap_error_bound"][0] < 1.0e-12
    assert not propagation["snap_mask"][0]
    np.testing.assert_allclose(
        propagation["next_inventory"],
        [1.0 / 6.0, 5.0 / 6.0, 0.0, 0.0],
    )


def test_rainout_retries_cold_after_converged_warm_fails_floorless_budget(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    calls = []

    def fake_run_head_v2_profile(**kwargs):
        scaled_budget = np.asarray(kwargs["b"], dtype=np.float64)
        scale = float(np.sum(scaled_budget))
        calls.append(scale)
        gas_n = scaled_budget[:2].copy()
        if len(calls) == 1:
            gas_n[0] *= 1.1
        return _one_layer_profile(gas_n)

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([200.0]),
        P=np.asarray([10.0]),
        b=jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64),
        init=(
            CondensateEquilibriumInit(
                gas_ln_n=jnp.log(
                    jnp.asarray([0.6, 0.4], dtype=jnp.float64)
                ),
                gas_ntot=jnp.asarray(1.0, dtype=jnp.float64),
            ),
        ),
        options=CondensateEquilibriumOptions(rainout=True),
    )

    attempts = result.layers[0].diagnostics["rainout"]["attempts"]
    assert attempts[0]["rainout_floorless_budget_accepted"] is False
    assert attempts[0]["rainout_floorless_relative_tolerance"] == 1.0e-3
    assert attempts[0]["rainout_floorless_element_budget_residual"][0] > 0.0
    assert attempts[1]["rainout_floorless_budget_accepted"] is True
    np.testing.assert_allclose(calls, [calls[0], calls[0]])


def test_rainout_method_conflicts_and_one_layer_api_are_rejected() -> None:
    setup = _fake_setup()
    inventory = jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64)

    with pytest.raises(ValueError, match="requires profile method"):
        condmod.condensate_equilibrium_profile(
            setup,
            T=np.asarray([100.0, 200.0]),
            P=np.asarray([1.0, 10.0]),
            b=inventory,
            method="vmap_cold",
            options=CondensateEquilibriumOptions(rainout=True),
        )

    with pytest.raises(ValueError, match="dependent profile operation"):
        condmod.condensate_equilibrium(
            setup,
            T=200.0,
            P=10.0,
            b=inventory,
            options=CondensateEquilibriumOptions(rainout=True),
        )


def test_explicit_profile_method_overrides_options_method(monkeypatch) -> None:
    setup = _fake_setup()
    inventory = jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64)
    expected = CondensateEquilibriumProfileResult(
        layers=(_layer_result((0.6, 0.4)),),
        method="scan_hot_from_bottom",
        rainout=True,
    )
    calls = []

    def fake_run_rainout_profile(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(condmod, "run_rainout_profile", fake_run_rainout_profile)

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([200.0]),
        P=np.asarray([10.0]),
        b=inventory,
        method="scan_hot_from_bottom",
        options=CondensateEquilibriumOptions(
            rainout=True,
            profile_method="vmap_cold",
        ),
    )

    assert result is expected
    assert len(calls) == 1


def test_rainout_stops_before_upper_layers_after_nonconvergence(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    temperatures_seen = []

    def fake_run_head_v2_profile(**kwargs):
        temperature = float(kwargs["temperatures"][0])
        temperatures_seen.append(temperature)
        if temperature == 200.0:
            return _one_layer_profile((0.3, 0.2), converged=False)
        return _one_layer_profile(np.asarray(kwargs["b"])[:2])

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    with pytest.raises(RuntimeError, match="original profile index 1"):
        condmod.condensate_equilibrium_profile(
            setup,
            T=np.asarray([100.0, 200.0, 300.0]),
            P=np.asarray([1.0, 10.0, 100.0]),
            b=jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64),
            options=CondensateEquilibriumOptions(rainout=True),
        )

    assert temperatures_seen == [300.0, 200.0, 200.0]


def test_rainout_retries_cold_without_changing_the_caller_gauge(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    scales_seen = []
    initial_gas_seen = []

    def fake_run_head_v2_profile(**kwargs):
        scale = float(np.asarray(kwargs["b"])[0] / 0.6)
        scales_seen.append(scale)
        gas_ln_n = kwargs["explicit_inits"][0].gas_ln_n
        initial_gas_seen.append(
            None if gas_ln_n is None else np.exp(np.asarray(gas_ln_n))
        )
        if gas_ln_n is not None:
            raise ValueError("element_potential must contain only finite values.")
        return _one_layer_profile(np.asarray(kwargs["b"])[:2])

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([200.0]),
        P=np.asarray([10.0]),
        b=jnp.asarray([0.6, 1.0e-8, 0.0], dtype=jnp.float64),
        init=(
            CondensateEquilibriumInit(
                gas_ln_n=jnp.log(
                    jnp.asarray([0.6, 1.0e-8], dtype=jnp.float64)
                ),
                gas_ntot=jnp.asarray(0.60000001, dtype=jnp.float64),
                condensate_amounts=jnp.asarray([0.2], dtype=jnp.float64),
            ),
        ),
        options=CondensateEquilibriumOptions(rainout=True),
    )

    np.testing.assert_allclose(
        scales_seen,
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        initial_gas_seen[0],
        np.asarray([0.6, 1.0e-8]),
    )
    assert initial_gas_seen[1] is None
    np.testing.assert_allclose(np.asarray(result.layers[0].gas_n), [0.6, 1.0e-8])
    np.testing.assert_allclose(
        np.asarray(result.layers[0].condensate_amounts), [0.0]
    )
    np.testing.assert_allclose(
        np.asarray(result.rainout_abundance_scale), [1.0]
    )
    rainout_diagnostics = result.layers[0].diagnostics["rainout"]
    assert rainout_diagnostics["schema"] == (
        "exogibbs_condensate_rainout_layer_v2"
    )
    assert rainout_diagnostics["solver_diagnostics_gauge"] == (
        "canonical_internal_amount_gauge"
    )
    assert rainout_diagnostics["budget_audit_gauge"] == (
        "caller_abundance_gauge_times_abundance_scale"
    )
    assert rainout_diagnostics["public_result_gauge"] == (
        "caller_abundance_gauge"
    )


def test_rainout_preserves_caller_gauge_below_maximum_total() -> None:
    scales = _rainout_gauge_scales(
        np.asarray([0.965, 1.0e-17, 0.035]),
        np.asarray([True, True, True]),
        maximum_total=1.0e9,
    )

    assert scales == (1.0,)


def test_rainout_gauge_downscales_an_input_above_the_maximum_total() -> None:
    inventory = np.asarray([6.0e9, 4.0e9, 0.0])
    scales = _rainout_gauge_scales(
        inventory,
        np.asarray([True, True, False]),
        maximum_total=1.0e9,
    )

    expected = np.nextafter(0.1, 0.0)
    assert scales == (expected,)
    assert scales[0] * np.sum(inventory) <= 1.0e9


def test_rainout_applies_and_reverses_only_the_overflow_downscale(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    inventory = np.asarray([6.0e9, 4.0e9, 0.0])
    scale = np.nextafter(3.0e8 / np.sum(inventory), 0.0)
    calls = []

    def fake_run_head_v2_profile(**kwargs):
        initial_guess = kwargs["explicit_inits"][0]
        calls.append(kwargs)
        if initial_guess.gas_ln_n is not None:
            raise ValueError("element_potential must contain only finite values.")
        scaled_inventory = np.asarray(kwargs["b"], dtype=np.float64)
        condensate_amount = 1.0e9 * scale
        gas_n = scaled_inventory[:2].copy()
        gas_n[0] -= condensate_amount
        return _one_layer_profile(
            gas_n,
            condensate_amount=condensate_amount,
        )

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([200.0]),
        P=np.asarray([10.0]),
        b=jnp.asarray(inventory, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts_init=(1.0e9,),
        init=(
            CondensateEquilibriumInit(
                gas_ln_n=jnp.log(
                    jnp.asarray([5.0e9, 4.0e9], dtype=jnp.float64)
                ),
                gas_ntot=jnp.asarray(9.0e9, dtype=jnp.float64),
                condensate_amounts=jnp.asarray([1.0e9], dtype=jnp.float64),
                support_indices=(0,),
                support_amounts=(1.0e9,),
            ),
        ),
        options=CondensateEquilibriumOptions(rainout=True),
    )

    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(np.asarray(call["b"]), inventory * scale)
        np.testing.assert_allclose(
            call["support_amounts_init"],
            np.asarray([1.0e9]) * scale,
        )
    warm_init = calls[0]["explicit_inits"][0]
    np.testing.assert_allclose(
        np.exp(np.asarray(warm_init.gas_ln_n)),
        np.asarray([5.0e9, 4.0e9]) * scale,
    )
    np.testing.assert_allclose(
        np.asarray(warm_init.condensate_amounts),
        np.asarray([1.0e9]) * scale,
    )
    assert calls[1]["explicit_inits"][0].gas_ln_n is None
    np.testing.assert_allclose(np.asarray(result.layers[0].gas_n), [5.0e9, 4.0e9])
    np.testing.assert_allclose(
        np.asarray(result.layers[0].condensate_amounts), [1.0e9]
    )
    np.testing.assert_array_equal(
        np.asarray(result.element_inventory_target),
        inventory[None, :],
    )
    np.testing.assert_allclose(
        np.asarray(result.rainout_abundance_scale), [scale]
    )


def test_rainout_retries_cold_after_a_failed_warm_transition(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    calls = []

    def fake_run_head_v2_profile(**kwargs):
        temperature = float(kwargs["temperatures"][0])
        initial_guess = kwargs["explicit_inits"][0]
        has_gas_warm_start = initial_guess.gas_ln_n is not None
        calls.append((temperature, has_gas_warm_start))
        if temperature == 100.0 and has_gas_warm_start:
            return _one_layer_profile((0.4, 0.6), converged=False)
        return _one_layer_profile(np.asarray(kwargs["b"])[:2])

    monkeypatch.setattr(
        _lifecycle,
        "_run_head_v2_profile",
        fake_run_head_v2_profile,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([100.0, 200.0]),
        P=np.asarray([1.0, 10.0]),
        b=jnp.asarray([0.6, 0.4, 0.0], dtype=jnp.float64),
        options=CondensateEquilibriumOptions(rainout=True),
    )

    assert calls == [(200.0, False), (100.0, True), (100.0, False)]
    assert all(layer.converged for layer in result.layers)
    attempts = result.layers[0].diagnostics["rainout"]["attempts"]
    assert [attempt["initialization"] for attempt in attempts] == [
        "resolved",
        "cold_fallback",
    ]


def test_rainout_trace_capacity_tier_requires_all_other_kkt_gates() -> None:
    setup = _fake_setup()
    diagnostics = {
        "fixed_support_v2": {
            "outcome": "fixed_support_failed",
            "terminal_status_name": "NORMAL_MAX_ITER",
            "support_closed": True,
            "final_state_values_finite": True,
            "independent_kkt": {
                "gas_stationarity": 0.0,
                "condensate_stationarity": 2.0e-6,
                "budget_scaled": 1.0e-15,
                "complementarity": 1.0e-15,
                "total_density_scaled": 1.0e-15,
            },
        },
        "full_condensate_budget_residual_gate": {
            "accepted": True,
            "enabled": True,
            "relative_tolerance": 1.0e-3,
            "element_budget_target": (1.0e-20, 1.0, 0.0),
            "element_budget_reconstructed": (1.0e-20, 1.0, 0.0),
        },
    }
    candidate = replace(
        _layer_result(
            (1.0e-300, 1.0),
            condensate_amount=1.0e-20,
            converged=False,
        ),
        condensate_support_indices=jnp.asarray([0], dtype=jnp.int32),
        condensate_support_names=("H[s]",),
        diagnostics=diagnostics,
    )
    policy = fixed_support_v2_production_policy()
    inventory = np.asarray([1.0e-20, 1.0, 0.0])

    # NORMAL_MAX_ITER is not a physical depletion decision in production.
    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=candidate,
            abundance_scale=1.0,
            policy=policy,
        )
        is None
    )
    policy = replace(policy, rainout_allow_trace_capacity_acceptance=True)
    report = _trace_capacity_acceptance_report(
        setup=setup,
        inventory=inventory,
        inventory_sum=float(np.sum(inventory)),
        candidate=candidate,
        abundance_scale=1.0,
        policy=policy,
    )

    assert report is not None
    assert report["maximum_relative_capacity"] == pytest.approx(1.0e-20)
    accepted = _accept_trace_capacity_candidate(candidate, report)
    assert accepted.converged
    assert accepted.status == "converged"
    assert accepted.acceptance_tier == "rainout_trace_capacity_accepted"
    assert accepted.diagnostics["acceptance_tier"] == (
        "rainout_trace_capacity_accepted"
    )
    assert accepted.diagnostics[
        "pre_rainout_trace_capacity_acceptance_tier"
    ] == "rejected"

    failed_kkt = dict(diagnostics["fixed_support_v2"]["independent_kkt"])
    failed_kkt["budget_scaled"] = 1.0e-4
    rejected = replace(
        candidate,
        diagnostics={
            **diagnostics,
            "fixed_support_v2": {
                **diagnostics["fixed_support_v2"],
                "independent_kkt": failed_kkt,
            },
        },
    )
    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=rejected,
            abundance_scale=1.0,
            policy=policy,
        )
        is None
    )


def test_rainout_trace_capacity_rejects_a_disabled_budget_gate() -> None:
    setup = _fake_setup()
    inventory = np.asarray([1.0e-20, 1.0, 0.0])
    candidate = _trace_test_candidate(
        gas_n=(1.0e-300, 1.0),
        condensate_amount=1.0e-20,
        budget_target=inventory,
        budget_reconstructed=inventory,
        budget_gate_enabled=False,
    )

    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=candidate,
            abundance_scale=1.0,
            policy=replace(
                fixed_support_v2_production_policy(),
                rainout_allow_trace_capacity_acceptance=True,
            ),
        )
        is None
    )


def test_rainout_trace_capacity_rejects_a_zero_capacity_support() -> None:
    setup = _fake_setup()
    inventory = np.asarray([0.0, 1.0, 0.0])
    candidate = _trace_test_candidate(
        gas_n=(1.0e-300, 1.0),
        condensate_amount=0.0,
        budget_target=inventory,
        budget_reconstructed=inventory,
    )

    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=candidate,
            abundance_scale=1.0,
            policy=replace(
                fixed_support_v2_production_policy(),
                rainout_allow_trace_capacity_acceptance=True,
            ),
        )
        is None
    )


def test_rainout_trace_capacity_rejects_floor_hidden_trace_error() -> None:
    setup = _fake_setup()
    abundance_scale = 3.0e8
    inventory = np.asarray([1.0e-20, 1.0, 0.0])
    scaled_target = inventory * abundance_scale
    reconstructed = scaled_target.copy()
    reconstructed[0] = 1.0e-10
    candidate = _trace_test_candidate(
        gas_n=(9.7e-11, abundance_scale),
        condensate_amount=3.0e-12,
        budget_target=scaled_target,
        budget_reconstructed=reconstructed,
    )

    # The ordinary 1e-6 floor hides a 32x physical relative error here.
    assert abs(reconstructed[0] - scaled_target[0]) / 1.0e-6 < 1.0e-3
    assert reconstructed[0] / scaled_target[0] > 30.0
    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=candidate,
            abundance_scale=abundance_scale,
            policy=replace(
                fixed_support_v2_production_policy(),
                rainout_allow_trace_capacity_acceptance=True,
            ),
        )
        is None
    )


def test_rainout_trace_capacity_rejects_amount_above_capacity() -> None:
    setup = _fake_setup()
    inventory = np.asarray([1.0e-20, 1.0, 0.0])
    candidate = _trace_test_candidate(
        gas_n=(1.0e-300, 1.0),
        condensate_amount=2.0e-20,
        budget_target=inventory,
        budget_reconstructed=inventory,
    )

    assert (
        _trace_capacity_acceptance_report(
            setup=setup,
            inventory=inventory,
            inventory_sum=float(np.sum(inventory)),
            candidate=candidate,
            abundance_scale=1.0,
            policy=replace(
                fixed_support_v2_production_policy(),
                rainout_allow_trace_capacity_acceptance=True,
            ),
        )
        is None
    )
