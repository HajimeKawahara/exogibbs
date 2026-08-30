"""Production v2 condensate profile tests."""

import math
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
from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    _physical_zero_barrier_audit,
)
from exogibbs.equilibrium.condensate.policy import (
    fixed_support_v2_production_policy,
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


def _amount_gauge_fake_setup() -> CondensateChemicalSetup:
    base = _fake_setup()
    condensate_setup = ChemicalSetup(
        formula_matrix=base.formula_matrix_cond,
        hvector_func=lambda temperature: jnp.asarray(
            [math.log(0.5), 0.0], dtype=jnp.float64
        ),
        elements=base.elements,
        species=base.condensate_species,
        metadata={},
    )
    return CondensateChemicalSetup(
        gas_setup=base.gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=base.formula_matrix,
        formula_matrix_cond=base.formula_matrix_cond,
        gas_species=base.gas_species,
        condensate_species=base.condensate_species,
        elements=base.elements,
    )


def _rank_deficient_initializer_fake_setup() -> CondensateChemicalSetup:
    gas_setup = _fake_setup().gas_setup
    gas_amounts = np.asarray([0.4, 0.3], dtype=np.float64)
    gas_total = float(np.sum(gas_amounts))
    element_potential = np.log(gas_amounts) - math.log(gas_total)
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float64,
        ),
        hvector_func=lambda temperature: jnp.asarray(
            [0.0, element_potential[0], element_potential[1]],
            dtype=jnp.float64,
        ),
        elements=gas_setup.elements,
        species=("H-high[s]", "H-low[s]", "O[s]"),
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


def _prepared_real_support(bucket, row: int = 0) -> tuple[int, ...]:
    indices = np.asarray(bucket.support_indices)
    mask = getattr(bucket, "condensate_slot_mask", None)
    if indices.ndim == 1:
        return tuple(int(index) for index in indices)
    if mask is None:
        return tuple(int(index) for index in indices[row])
    return tuple(
        int(index)
        for index in indices[row][np.asarray(mask, dtype=bool)[row]]
    )


def test_head_v2_prepared_buckets_apply_layer_fugacity_correction(
    monkeypatch,
) -> None:
    setup = _fake_setup()
    captured = {}
    provider_calls = []

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        provider_calls.append(
            (float(temperature), float(pressure_bar), mole_fractions)
        )
        return jnp.asarray(
            [float(temperature) / 1000.0, math.log(float(pressure_bar))],
            dtype=jnp.float64,
        )

    def fake_prepare_fixed_support_v2_buckets(**kwargs):
        captured.update(kwargs)
        return ("prepared",)

    monkeypatch.setattr(
        "exogibbs.equilibrium.condensate.fixed_support.batch."
        "prepare_fixed_support_v2_buckets",
        fake_prepare_fixed_support_v2_buckets,
    )
    states = tuple(
        _lifecycle._HeadV2LayerState(
            support_indices=(0,),
            gas_ln_n=jnp.zeros((2,), dtype=jnp.float64),
            condensate_log_amounts=jnp.zeros((1,), dtype=jnp.float64),
            total_gas_log_amount=jnp.asarray(0.0, dtype=jnp.float64),
            element_potential=jnp.zeros((2,), dtype=jnp.float64),
        )
        for _ in range(2)
    )

    buckets = _lifecycle._head_v2_prepared_buckets(
        setup=setup,
        temperatures=(800.0, 1200.0),
        pressures=(1.0, 10.0),
        b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        Pref=1.0,
        states=states,
        lnphi_func=lnphi_func,
    )

    assert buckets == ("prepared",)
    assert provider_calls == [(800.0, 1.0, None), (1200.0, 10.0, None)]
    np.testing.assert_allclose(
        np.asarray(captured["hvector_by_layer"]),
        [[0.8, 0.0], [1.2, math.log(10.0)]],
    )


def test_head_v2_element_potential_uses_fugacity_corrected_gamma() -> None:
    setup = _fake_setup()
    provider_calls = []

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        provider_calls.append(
            (float(temperature), float(pressure_bar), mole_fractions)
        )
        return jnp.asarray([0.25, -0.5], dtype=jnp.float64)

    gas_amounts = jnp.asarray([0.4, 0.6], dtype=jnp.float64)
    element_potential = _lifecycle._head_v2_best_residual_element_potential(
        setup=setup,
        T=900.0,
        P=2.0,
        Pref=1.0,
        b=gas_amounts,
        support_indices=(),
        support_amounts=(),
        gas_ln_n=jnp.log(gas_amounts),
        total_gas_log_amount=jnp.asarray(0.0, dtype=jnp.float64),
        epsilon=-10.0,
        lnphi_func=lnphi_func,
    )

    assert provider_calls == [(900.0, 2.0, None)]
    np.testing.assert_allclose(
        np.asarray(element_potential),
        np.log(np.asarray(gas_amounts))
        + np.asarray([0.25, -0.5])
        + math.log(2.0),
    )


def test_amount_gauge_scale_and_initializer_normalization() -> None:
    setup = SimpleNamespace(elements=("H", "O", "e-"))
    scale = _lifecycle._inventory_amount_gauge_scale(
        setup,
        jnp.asarray([2.0e-12, 3.0e-12, 1.0e9]),
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([2.0e-12, 3.0e-12])),
        gas_ntot=jnp.asarray(5.0e-12),
        condensate_amounts=jnp.asarray([1.0e-12]),
        support_indices=(0,),
        support_amounts=(1.0e-12,),
        element_potential=jnp.asarray([1.0, 2.0, 3.0]),
        rho=jnp.asarray([4.0]),
        barrier_epsilon=jnp.asarray(-11.0 + math.log(5.0e-12)),
    )

    normalized = _lifecycle._normalize_condensate_init_amount_gauge(
        initial,
        scale,
    )

    assert scale == pytest.approx(5.0e-12)
    np.testing.assert_allclose(jnp.exp(normalized.gas_ln_n), [0.4, 0.6])
    assert float(normalized.gas_ntot) == pytest.approx(1.0)
    np.testing.assert_allclose(normalized.condensate_amounts, [0.2])
    assert normalized.support_amounts == pytest.approx((0.2,))
    np.testing.assert_array_equal(
        normalized.element_potential, initial.element_potential
    )
    np.testing.assert_array_equal(normalized.rho, initial.rho)
    assert float(normalized.barrier_epsilon) == pytest.approx(-11.0)


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ([1.0, 0.0], "one value per element"),
        ([math.nan, 0.0, 0.0], "only finite values"),
        ([-1.0, 0.0, 1.0], "must be non-negative"),
        ([0.0, 0.0, 1.0], "positive non-charge amount"),
    ),
)
def test_amount_gauge_scale_rejects_invalid_targets(target, message):
    setup = SimpleNamespace(elements=("H", "O", "e-"))

    with pytest.raises(ValueError, match=message):
        _lifecycle._inventory_amount_gauge_scale(setup, target)


def test_caller_budget_audit_keeps_trace_row_scale_finite() -> None:
    amount_scale = 1.0e-12
    target = np.asarray([amount_scale, 1.0e-312], dtype=np.float64)
    gas_log_amounts = np.log(target)
    total_gas_log_amount = float(np.log(np.sum(target)))
    budget_scale = _lifecycle._canonical_budget_scale_for_caller_audit(
        target,
        amount_scale=amount_scale,
        relative_floor=1.0e-6,
    )

    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=np.eye(2),
        condensate_formula_matrix_full=np.zeros((2, 0)),
        target_inventory=target,
        gas_standard_source=np.zeros(2),
        condensate_standard_source_full=np.zeros(0),
        gas_log_amounts=gas_log_amounts,
        condensate_amounts=np.zeros(0),
        total_gas_log_amount=total_gas_log_amount,
        element_potential=gas_log_amounts - total_gas_log_amount,
        support_indices=(),
        condensate_valid_mask=np.zeros(0, dtype=bool),
        budget_scale=budget_scale,
        optimizer_success=True,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        budget_residual_amount_scale=amount_scale,
    )

    assert np.all(np.isfinite(budget_scale))
    assert audit["accepted"]
    assert audit["budget_scaled_max_abs"] < 1.0e-8


@pytest.mark.parametrize("amount_scale", (1.0e-305, 1.0e-12, 1.0, 1.0e8))
def test_head_v2_uses_one_canonical_amount_gauge_and_rescales_results(
    monkeypatch,
    amount_scale,
):
    setup = _amount_gauge_fake_setup()
    captured = {}

    def fake_support_payload(**kwargs):
        captured["support_target"] = np.asarray(kwargs["b"])
        captured["gas_init"] = kwargs["gas_equilibrium_init"]
        return (0,), (0.2,), {"policy": "test_canonical_support"}

    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        fake_support_payload,
    )

    def fake_run_fixed_support_profile(**kwargs):
        bucket = kwargs["buckets"][0]
        captured["bucket_target"] = np.asarray(
            bucket.element_inventory_target
        )
        captured["bucket_gas"] = np.exp(np.asarray(bucket.ln_nk_init))
        slot_amounts = np.exp(np.asarray(bucket.ln_mk_init))
        slot_mask = np.asarray(bucket.condensate_slot_mask, dtype=bool)
        captured["bucket_condensate"] = tuple(
            tuple(row[row_mask])
            for row, row_mask in zip(slot_amounts, slot_mask)
        )
        zeros = jnp.zeros((1,), dtype=jnp.float64)
        return {
            "backend": "cpu",
            "compilation_seconds": 0.0,
            "execution_seconds": 0.0,
            "diagnostic_seconds": 0.0,
            "gas_log_amounts": jnp.log(
                jnp.asarray([[0.4, 0.4]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.2, 0.0]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.full(
                (1, 2), math.log(0.5), dtype=jnp.float64
            ),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.CONVERGED)], dtype=jnp.int32
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros, zeros, zeros, zeros, zeros
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([True]),
            "support_closed": jnp.asarray([True]),
            "support_expansion_mask": jnp.asarray(
                [[False, False]], dtype=bool
            ),
            "inactive_condensate_driving": jnp.zeros(
                (1, 2), dtype=jnp.float64
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
        captured["zero_barrier_target"] = np.asarray(
            kwargs["target_inventory"]
        )
        return SimpleNamespace(
            accepted=True,
            gas_log_amounts=np.asarray(kwargs["gas_log_amounts_init"]),
            condensate_amounts=np.asarray(
                kwargs["condensate_amounts_init"]
            ),
            total_gas_log_amount=float(
                kwargs["total_gas_log_amount_init"]
            ),
            element_potential=np.asarray(kwargs["element_potential_init"]),
            support_indices=tuple(kwargs["support_indices"]),
            report={"accepted": True, "polish_schema": "unit_test"},
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(
            amount_scale * jnp.asarray([0.4, 0.4], dtype=jnp.float64)
        ),
        gas_ntot=jnp.asarray(0.8 * amount_scale, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.2 * amount_scale,),
        element_potential=jnp.full(
            (2,), math.log(0.5), dtype=jnp.float64
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=amount_scale * jnp.asarray([0.6, 0.4], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )

    np.testing.assert_allclose(captured["support_target"], [0.6, 0.4])
    np.testing.assert_allclose(captured["bucket_target"], [[0.6, 0.4]])
    np.testing.assert_allclose(captured["bucket_gas"], [[0.4, 0.4]])
    np.testing.assert_allclose(captured["bucket_condensate"], [[0.2]])
    np.testing.assert_allclose(captured["zero_barrier_target"], [0.6, 0.4])
    np.testing.assert_allclose(
        jnp.exp(captured["gas_init"].ln_nk), [0.4, 0.4]
    )
    layer = result.layers[0]
    np.testing.assert_allclose(
        layer.gas_n / amount_scale,
        [0.4, 0.4],
    )
    np.testing.assert_allclose(
        layer.condensate_amounts / amount_scale,
        [0.2, 0.0],
    )
    assert float(layer.gas_ntot / amount_scale) == pytest.approx(0.8)
    np.testing.assert_allclose(layer.gas_x, [0.5, 0.5])
    np.testing.assert_allclose(
        result.batched_arrays["gas_x"][0],
        layer.gas_x,
    )
    assert layer.condensate_support_names == ("H[s]",)
    gauge = layer.diagnostics["fixed_support_v2"]["amount_gauge"]
    assert gauge["caller_inventory_amount_scale"] == pytest.approx(
        amount_scale
    )
    assert gauge["normalized_epsilon_schedule"] == (
        -11.0,
        -13.0,
        -15.0,
        -17.0,
    )
    assert gauge["caller_equivalent_epsilon_schedule"][0] == pytest.approx(
        -11.0 + math.log(amount_scale)
    )
    caller_kkt = layer.diagnostics["fixed_support_v2"][
        "caller_gauge_zero_barrier_kkt"
    ]
    assert caller_kkt["accepted"]
    assert caller_kkt["budget_scaled_max_abs"] < 1.0e-12
    budget = layer.diagnostics["full_condensate_budget_residual_gate"]
    assert budget["accepted"]
    assert budget["amount_gauge_scale"] == pytest.approx(amount_scale)
    assert budget["absolute_floor"] == pytest.approx(
        1.0e-6 * amount_scale
    )


def test_head_v2_reduces_rank_deficient_finite_barrier_initializer(
    monkeypatch,
) -> None:
    setup = _rank_deficient_initializer_fake_setup()
    captured = {}

    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        lambda **kwargs: (
            (2, 0, 1),
            (0.1, 0.1, 0.1),
            {"policy": "test_rank_deficient_activity_support"},
        ),
    )

    def fake_run_fixed_support_profile(**kwargs):
        bucket = kwargs["buckets"][0]
        captured["support"] = _prepared_real_support(bucket)
        slot_amounts = np.exp(np.asarray(bucket.ln_mk_init))
        slot_mask = np.asarray(bucket.condensate_slot_mask, dtype=bool)
        captured["amounts"] = tuple(slot_amounts[0, slot_mask[0]])
        zeros = jnp.zeros((1,), dtype=jnp.float64)
        return {
            "backend": "cpu",
            "compilation_seconds": 0.0,
            "execution_seconds": 0.0,
            "diagnostic_seconds": 0.0,
            "gas_log_amounts": jnp.log(
                jnp.asarray([[0.4, 0.3]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.0, 0.2, 0.1]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.7], dtype=jnp.float64)
            ),
            "element_potential": jnp.asarray(
                [
                    [
                        math.log(0.4) - math.log(0.7),
                        math.log(0.3) - math.log(0.7),
                    ]
                ],
                dtype=jnp.float64,
            ),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.CONVERGED)], dtype=jnp.int32
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros, zeros, zeros, zeros, zeros
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([True]),
            "support_closed": jnp.asarray([True]),
            "support_expansion_mask": jnp.zeros((1, 3), dtype=bool),
            "inactive_condensate_driving": jnp.zeros(
                (1, 3), dtype=jnp.float64
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
        return SimpleNamespace(
            accepted=True,
            gas_log_amounts=np.asarray(kwargs["gas_log_amounts_init"]),
            condensate_amounts=np.asarray(kwargs["condensate_amounts_init"]),
            total_gas_log_amount=float(kwargs["total_gas_log_amount_init"]),
            element_potential=np.asarray(kwargs["element_potential_init"]),
            support_indices=tuple(kwargs["support_indices"]),
            report={"accepted": True, "polish_schema": "unit_test"},
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    element_potential = jnp.asarray(
        [
            math.log(0.4) - math.log(0.7),
            math.log(0.3) - math.log(0.7),
        ],
        dtype=jnp.float64,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.4, 0.3], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(0.7, dtype=jnp.float64),
        support_indices=(2, 0, 1),
        support_amounts=(0.1, 0.1, 0.1),
        element_potential=element_potential,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.6, 0.4], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )

    assert captured["support"] == (1, 2)
    assert captured["amounts"] == pytest.approx((0.2, 0.1))
    assert result.layers[0].converged
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    reduction = lifecycle["finite_barrier_initial_support_reduction"]
    assert reduction["role"] == "finite_barrier_pdipm_initializer"
    assert reduction["attempted"]
    assert reduction["applied"]
    assert reduction["input_support_rank"] == 2
    assert reduction["output_support_rank"] == 2
    assert reduction["output_dropped_support_indices"] == (0,)
    assert reduction["output_scaled_inventory_residual_max_abs"] <= (
        reduction["scaled_inventory_residual_tolerance"]
    )
    assert reduction["fallback_reason"] is None
    assert lifecycle["initial_support_indices"] == (1, 2)
    profile_record = result.diagnostics["layers"][0]
    assert profile_record["initial_support_indices"] == (1, 2)
    assert profile_record[
        "finite_barrier_initial_support_reduction"
    ] == reduction


@pytest.mark.parametrize("early_internal_accepted", (False, True))
def test_head_v2_profile_expands_support_outside_solver_until_closed(
    monkeypatch,
    early_internal_accepted,
):
    setup = _amount_gauge_fake_setup()
    calls = []
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
        calls.append(kwargs)
        support = _prepared_real_support(kwargs["buckets"][0])
        closed = len(support) == 2
        condensate_amounts = (
            jnp.asarray([[0.2, 0.1]], dtype=jnp.float64)
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
                jnp.asarray([[0.4, 0.4]], dtype=jnp.float64)
            ),
            "condensate_amounts": condensate_amounts,
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.full(
                (1, 2), math.log(0.5), dtype=jnp.float64
            ),
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
        exact_calls.append(kwargs)
        amounts = np.asarray(
            kwargs["condensate_amounts_init"], dtype=np.float64
        ).copy()
        initial_support = tuple(kwargs["support_indices"])
        closed_support = len(initial_support) == 2
        accepted = closed_support or early_internal_accepted
        final_support = (
            initial_support[:-1] if closed_support else initial_support
        )
        if closed_support:
            amounts[initial_support[-1]] = 0.0
            gas_log_amounts = np.asarray(
                kwargs["gas_log_amounts_init"], dtype=np.float64
            )
        elif not accepted:
            # A rejected exact candidate must not seed the next finite round.
            amounts[:] = 0.9
            gas_log_amounts = np.log(np.asarray([0.1, 0.1]))
        else:
            gas_log_amounts = np.asarray(
                kwargs["gas_log_amounts_init"], dtype=np.float64
            )
        return SimpleNamespace(
            accepted=accepted,
            gas_log_amounts=gas_log_amounts,
            condensate_amounts=amounts,
            total_gas_log_amount=float(
                kwargs["total_gas_log_amount_init"]
            ),
            element_potential=np.asarray(
                kwargs["element_potential_init"], dtype=np.float64
            ),
            support_indices=final_support,
            report={
                "accepted": accepted,
                "polish_schema": "unit_test",
                "initial_support_indices": initial_support,
                "final_support_indices": final_support,
                "dropped_support_indices": (
                    initial_support[-1:] if closed_support else ()
                ),
            },
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    caller_audit_calls = []

    def fake_caller_audit(**kwargs):
        caller_audit_calls.append(kwargs)
        if early_internal_accepted and len(caller_audit_calls) == 1:
            return {
                "accepted": False,
                "finite": True,
                "positive_active_amounts": True,
                "gas_stationarity_max_abs": 0.0,
                "active_condensate_driving_max_abs": 0.0,
                "inactive_condensate_violation_max_abs": 0.0,
                "budget_scaled_max_abs": 1.0,
                "total_density_scaled_abs": 0.0,
            }
        return _physical_zero_barrier_audit(**kwargs)

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "_physical_zero_barrier_audit"
        ),
        fake_caller_audit,
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
        b=jnp.asarray([1.2, 0.8], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    assert len(calls) == 2
    assert [
        tuple(call["support_indices"]) for call in exact_calls
    ] == [(0,), (0, 1)]
    assert _prepared_real_support(calls[0]["buckets"][0]) == (0,)
    assert _prepared_real_support(calls[1]["buckets"][0]) == (0, 1)
    assert calls[0]["buckets"][0].support_indices.shape == (1, 2)
    assert calls[1]["buckets"][0].support_indices.shape == (1, 2)
    np.testing.assert_allclose(
        np.exp(np.asarray(calls[1]["buckets"][0].ln_nk_init)),
        [[0.4, 0.4]],
    )
    assert result.layers[0].converged
    assert result.layers[0].selected_route == (
        "head_v2_fixed_support_lifecycle"
    )
    assert result.layers[0].head_route_version == "v2.0"
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == "closed"
    assert lifecycle["independent_kkt_passed"]
    assert lifecycle["rounds"][0]["added_support_indices"] == (1,)
    assert lifecycle["rounds"][0]["early_zero_barrier_eligible"]
    assert lifecycle["rounds"][0][
        "early_zero_barrier_internal_accepted"
    ] is early_internal_accepted
    assert not lifecycle["rounds"][0]["early_zero_barrier_accepted"]
    assert lifecycle["support_indices_after_polish"] == (0,)
    assert result.layers[0].condensate_support_names == ("H[s]",)
    assert result.layers[0].condensate_amounts.tolist() == pytest.approx(
        [0.4, 0.0]
    )
    assert result.diagnostics["route"] == "head_v2"


def test_head_v2_closes_open_support_before_finite_support_expansion(
    monkeypatch,
):
    setup = _amount_gauge_fake_setup()
    fixed_support_calls = []
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
        fixed_support_calls.append(kwargs)
        zeros = jnp.zeros((1,), dtype=jnp.float64)
        return {
            "backend": "cpu",
            "compilation_seconds": 0.0,
            "execution_seconds": 0.0,
            "diagnostic_seconds": 0.0,
            "gas_log_amounts": jnp.log(
                jnp.asarray([[0.4, 0.4]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.2, 0.0]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.full(
                (1, 2), math.log(0.5), dtype=jnp.float64
            ),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.CONVERGED)], dtype=jnp.int32
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros, zeros, zeros, zeros, zeros
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([True]),
            "support_closed": jnp.asarray([False]),
            "support_expansion_mask": jnp.asarray(
                [[False, True]], dtype=bool
            ),
            "inactive_condensate_driving": jnp.asarray(
                [[0.0, -1.0]], dtype=jnp.float64
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
            accepted=True,
            gas_log_amounts=np.asarray(kwargs["gas_log_amounts_init"]),
            condensate_amounts=np.asarray(
                kwargs["condensate_amounts_init"]
            ),
            total_gas_log_amount=float(
                kwargs["total_gas_log_amount_init"]
            ),
            element_potential=np.asarray(kwargs["element_potential_init"]),
            support_indices=tuple(kwargs["support_indices"]),
            report={"accepted": True, "polish_schema": "unit_test"},
        )

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        fake_zero_barrier_polish,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.4, 0.4], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(0.8, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.2,),
        element_potential=jnp.full(
            (2,), math.log(0.5), dtype=jnp.float64
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.6, 0.4], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    assert len(fixed_support_calls) == 1
    assert len(exact_calls) == 1
    assert tuple(exact_calls[0]["support_indices"]) == (0,)
    assert result.layers[0].converged
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == "zero_barrier_open_support_rescued"
    assert lifecycle["zero_barrier_initializer"]["source"] == (
        "open_converged_finite_support_state"
    )
    assert lifecycle["rounds"][0]["early_zero_barrier_accepted"]
    assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


def test_head_v2_discards_exact_candidate_rejected_in_caller_gauge(
    monkeypatch,
):
    setup = _amount_gauge_fake_setup()
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
                jnp.asarray([[0.4, 0.4]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.2, 0.0]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.full(
                (1, 2), math.log(0.5), dtype=jnp.float64
            ),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.CONVERGED)], dtype=jnp.int32
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros, zeros, zeros, zeros, zeros
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([True]),
            "support_closed": jnp.asarray([True]),
            "support_expansion_mask": jnp.asarray(
                [[False, False]], dtype=bool
            ),
            "inactive_condensate_driving": jnp.zeros(
                (1, 2), dtype=jnp.float64
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
    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "polish_zero_barrier_active_support"
        ),
        lambda **kwargs: SimpleNamespace(
            accepted=True,
            gas_log_amounts=np.log(np.asarray([0.1, 0.1])),
            condensate_amounts=np.asarray([0.0, 0.9]),
            total_gas_log_amount=math.log(0.2),
            element_potential=np.asarray([-1.0, -1.0]),
            support_indices=(1,),
            report={"accepted": True, "polish_schema": "unit_test"},
        ),
    )

    def reject_caller_audit(**kwargs):
        return {
            "accepted": False,
            "finite": True,
            "positive_active_amounts": True,
            "gas_stationarity_max_abs": 0.0,
            "active_condensate_driving_max_abs": 0.0,
            "inactive_condensate_violation_max_abs": 0.0,
            "budget_scaled_max_abs": 1.0,
            "total_density_scaled_abs": 0.0,
        }

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "_physical_zero_barrier_audit"
        ),
        reject_caller_audit,
    )
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray([0.4, 0.4], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(0.8, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.2,),
        element_potential=jnp.full(
            (2,), math.log(0.5), dtype=jnp.float64
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.6, 0.4], dtype=jnp.float64),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    layer = result.layers[0]
    assert not layer.converged
    assert layer.condensate_support_names == ("H[s]",)
    np.testing.assert_allclose(layer.gas_n, [0.4, 0.4])
    np.testing.assert_allclose(layer.condensate_amounts, [0.2, 0.0])
    lifecycle = layer.diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == (
        "caller_gauge_zero_barrier_kkt_failed"
    )
    assert not lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]


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
    setup = _amount_gauge_fake_setup()
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
                jnp.asarray([[0.4, 0.4]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.2, 0.0]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.8], dtype=jnp.float64)
            ),
            "element_potential": jnp.full(
                (1, 2), math.log(0.5), dtype=jnp.float64
            ),
            "terminal_status": jnp.asarray(
                [int(TerminalStatus.NORMAL_DUAL_STEP_FAILED)],
                dtype=jnp.int32,
            ),
            "final_kkt_norms": KKTComponentNorms(
                jnp.asarray([1.430511474609375e-6], dtype=jnp.float64),
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
        gas_ln_n=jnp.log(jnp.asarray([0.8, 0.8], dtype=jnp.float64)),
        gas_ntot=jnp.asarray(1.6, dtype=jnp.float64),
        support_indices=(0,),
        support_amounts=(0.4,),
        element_potential=jnp.full(
            (2,), math.log(0.5), dtype=jnp.float64
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([1.2, 0.8], dtype=jnp.float64),
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
    assert result.diagnostics["layers"][0]["outcome"] == expected_outcome
    assert not lifecycle["fixed_support_converged"]
    assert lifecycle["terminal_status_name"] == "NORMAL_DUAL_STEP_FAILED"
    assert not lifecycle["independent_kkt_passed"]
    initializer = lifecycle["zero_barrier_initializer"]
    assert initializer["eligible"] is raw_support_closed
    assert initializer["attempted"] is raw_support_closed
    assert initializer["role"] == "initializer_only"
    assert initializer["raw_noncondensate_kkt_passed"]
    assert initializer["rescue_attempted"] is raw_support_closed
    assert lifecycle[
        "zero_barrier_initializer_gas_stationarity_tolerance"
    ] == pytest.approx(1.0e-5)


@pytest.mark.parametrize(
    (
        "trace_inventory",
        "terminal_status",
        "exact_accepted",
        "caller_accepted",
        "expected_exact_calls",
        "expected_outcome",
        "expected_skip_reason",
    ),
    (
        (
            1.0e-12,
            TerminalStatus.RESTORATION_MAX_ITER,
            True,
            True,
            1,
            "zero_barrier_active_support_rescued",
            None,
        ),
        (
            1.0e-12,
            TerminalStatus.RESTORATION_LOCALLY_INFEASIBLE,
            True,
            True,
            1,
            "zero_barrier_active_support_rescued",
            None,
        ),
        (
            1.0e-12,
            TerminalStatus.RESTORATION_MAX_ITER,
            False,
            False,
            1,
            "zero_barrier_active_support_polish_failed",
            None,
        ),
        (
            1.0e-12,
            TerminalStatus.RESTORATION_MAX_ITER,
            True,
            False,
            1,
            "caller_gauge_zero_barrier_kkt_failed",
            None,
        ),
        (
            1.0,
            TerminalStatus.RESTORATION_MAX_ITER,
            True,
            True,
            0,
            "fixed_support_failed",
            "capacity_not_below_initial_barrier",
        ),
        (
            1.0e-12,
            TerminalStatus.NORMAL_LINEAR_SOLVE_FAILED,
            True,
            True,
            0,
            "fixed_support_failed",
            "terminal_status_not_eligible",
        ),
    ),
)
def test_head_v2_trace_capacity_fallback_uses_pre_pdipm_state(
    monkeypatch,
    trace_inventory,
    terminal_status,
    exact_accepted,
    caller_accepted,
    expected_exact_calls,
    expected_outcome,
    expected_skip_reason,
):
    setup = _amount_gauge_fake_setup()
    exact_calls = []
    caller_audit_calls = []
    initial_support_amount = 0.8 * trace_inventory

    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        lambda **kwargs: (
            (1,),
            (initial_support_amount,),
            {"policy": "test_trace_support"},
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
                jnp.asarray([[0.2, 0.3]], dtype=jnp.float64)
            ),
            "condensate_amounts": jnp.asarray(
                [[0.0, 0.25]], dtype=jnp.float64
            ),
            "total_gas_log_amount": jnp.log(
                jnp.asarray([0.5], dtype=jnp.float64)
            ),
            "element_potential": jnp.asarray(
                [[-9.0, -8.0]], dtype=jnp.float64
            ),
            "terminal_status": jnp.asarray(
                [int(terminal_status)],
                dtype=jnp.int32,
            ),
            "final_kkt_norms": KKTComponentNorms(
                zeros,
                jnp.asarray([1.0e8], dtype=jnp.float64),
                jnp.asarray([1.0e-3], dtype=jnp.float64),
                zeros,
                zeros,
            ),
            "final_state_values_finite": jnp.asarray([True]),
            "fixed_support_converged": jnp.asarray([False]),
            "support_closed": jnp.asarray([True]),
            "support_expansion_mask": jnp.asarray(
                [[False, False]], dtype=bool
            ),
            "inactive_condensate_driving": jnp.zeros(
                (1, 2), dtype=jnp.float64
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

    def fake_caller_audit(**kwargs):
        caller_audit_calls.append(kwargs)
        return {
            "accepted": caller_accepted,
            "finite": True,
            "positive_active_amounts": True,
            "gas_stationarity_max_abs": 0.0,
            "active_condensate_driving_max_abs": 0.0,
            "inactive_condensate_violation_max_abs": 0.0,
            "budget_scaled_max_abs": 0.0 if caller_accepted else 1.0,
            "total_density_scaled_abs": 0.0,
        }

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
            "_physical_zero_barrier_audit"
        ),
        fake_caller_audit,
    )
    initial_gas_amounts = np.asarray(
        [0.7, 0.2 * trace_inventory], dtype=np.float64
    )
    initial_gas_total = float(np.sum(initial_gas_amounts))
    initial_potential = np.asarray([-0.25, -0.75], dtype=np.float64)
    initial = CondensateEquilibriumInit(
        gas_ln_n=jnp.log(jnp.asarray(initial_gas_amounts)),
        gas_ntot=jnp.asarray(initial_gas_total, dtype=jnp.float64),
        support_indices=(1,),
        support_amounts=(initial_support_amount,),
        element_potential=jnp.asarray(initial_potential),
    )
    caller_inventory = np.asarray(
        [1.0, trace_inventory], dtype=np.float64
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray(caller_inventory),
        init=(initial,),
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )

    assert len(exact_calls) == expected_exact_calls
    assert len(caller_audit_calls) == (
        expected_exact_calls * int(exact_accepted)
    )
    amount_scale = float(np.sum(caller_inventory))
    expected_q = np.log(initial_gas_amounts) - math.log(amount_scale)
    expected_qtot = math.log(initial_gas_total) - math.log(amount_scale)
    expected_full_amounts = np.asarray(
        [0.0, initial_support_amount / amount_scale], dtype=np.float64
    )
    if expected_exact_calls:
        exact_call = exact_calls[0]
        assert exact_call["support_indices"] == (1,)
        np.testing.assert_allclose(
            exact_call["gas_log_amounts_init"], expected_q
        )
        np.testing.assert_allclose(
            exact_call["condensate_amounts_init"], expected_full_amounts
        )
        assert exact_call["total_gas_log_amount_init"] == pytest.approx(
            expected_qtot
        )
        np.testing.assert_allclose(
            exact_call["element_potential_init"], initial_potential
        )
    assert result.layers[0].converged is (
        bool(expected_exact_calls) and exact_accepted and caller_accepted
    )
    if not result.layers[0].converged:
        np.testing.assert_allclose(
            result.layers[0].gas_n,
            amount_scale * np.asarray([0.2, 0.3]),
        )
        np.testing.assert_allclose(
            result.layers[0].condensate_amounts,
            amount_scale * np.asarray([0.0, 0.25]),
        )
    lifecycle = result.layers[0].diagnostics["fixed_support_v2"]
    assert lifecycle["outcome"] == expected_outcome
    assert lifecycle["terminal_status_name"] == terminal_status.name
    assert not lifecycle["zero_barrier_initializer_kkt_passed"]
    initializer = lifecycle["zero_barrier_initializer"]
    expected_source = (
        "pre_pdipm_finite_support_state"
        if expected_exact_calls
        else "fixed_support_terminal_state"
    )
    assert initializer["source"] == expected_source
    assert initializer["selected_before_lifecycle_terminal_round"] is bool(
        expected_exact_calls
    )
    fallback = lifecycle["pre_pdipm_zero_barrier_fallback"]
    assert fallback["eligible"] is bool(expected_exact_calls)
    assert fallback["attempted"] is bool(expected_exact_calls)
    assert fallback["skip_reason"] == expected_skip_reason
    assert fallback["internal_accepted"] is (
        bool(expected_exact_calls) and exact_accepted
    )
    assert fallback["caller_gauge_accepted"] is (
        bool(expected_exact_calls) and exact_accepted and caller_accepted
    )
    assert fallback["accepted"] is (
        bool(expected_exact_calls) and exact_accepted and caller_accepted
    )
    assert fallback["source_support_indices"] == (1,)
    if expected_skip_reason == "terminal_status_not_eligible":
        assert fallback["trace_capacity"] is None
    else:
        assert fallback["trace_capacity"]["trace_capacity_detected"] is (
            trace_inventory < 1.0
        )


def test_disabled_pre_pdipm_fallback_does_not_materialize_device_state(
    monkeypatch,
) -> None:
    setup = _amount_gauge_fake_setup()
    state = _lifecycle._HeadV2LayerState(
        support_indices=(0,),
        gas_ln_n=jnp.zeros((2,), dtype=jnp.float64),
        condensate_log_amounts=jnp.zeros((1,), dtype=jnp.float64),
        total_gas_log_amount=jnp.asarray(0.0, dtype=jnp.float64),
        element_potential=jnp.zeros((2,), dtype=jnp.float64),
    )

    def fail_device_get(value):
        pytest.fail("disabled fallback materialized device state")

    monkeypatch.setattr(_lifecycle.jax, "device_get", fail_device_get)

    payload, report = _lifecycle._head_v2_pre_pdipm_zero_barrier_candidate(
        setup=setup,
        state=state,
        target_inventory=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
        log_barrier=-11.0,
        valid_condensates=None,
        enabled=False,
        disabled_reason="terminal_status_not_eligible",
    )

    assert payload is None
    assert not report["eligible"]
    assert report["skip_reason"] == "terminal_status_not_eligible"
    assert report["trace_capacity"] is None


def test_head_v2_zero_barrier_initializer_uses_bounded_gas_kkt_gate():
    policy = fixed_support_v2_production_policy()
    final_tolerances = policy.solver_config.normal
    initializer_gas_tolerance = (
        policy.zero_barrier_initializer_gas_stationarity_tolerance
    )
    kkt = {
        "gas_stationarity": 1.430511474609375e-6,
        "condensate_stationarity": 1.0,
        "budget_scaled": 1.0e-9,
        "complementarity": 1.0e-9,
        "total_density_scaled": 1.0e-9,
    }
    arguments = {
        "gas_stationarity_tolerance": initializer_gas_tolerance,
        "budget_tolerance": final_tolerances.budget_tolerance,
        "complementarity_tolerance": (
            final_tolerances.complementarity_tolerance
        ),
        "total_density_tolerance": (
            final_tolerances.total_density_tolerance
        ),
    }

    assert final_tolerances.stationarity_tolerance == pytest.approx(1.0e-8)
    assert initializer_gas_tolerance == pytest.approx(1.0e-5)
    assert _lifecycle._head_v2_zero_barrier_initializer_kkt_passed(
        kkt, **arguments
    )
    for name in (
        "budget_scaled",
        "complementarity",
        "total_density_scaled",
    ):
        assert not _lifecycle._head_v2_zero_barrier_initializer_kkt_passed(
            {**kkt, name: 1.0e-7}, **arguments
        )
    assert not _lifecycle._head_v2_zero_barrier_initializer_kkt_passed(
        {**kkt, "gas_stationarity": 1.0e-4}, **arguments
    )
    assert not _lifecycle._head_v2_zero_barrier_initializer_kkt_passed(
        {**kkt, "gas_stationarity": math.inf}, **arguments
    )


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
    lnphi_func = lambda temperature, pressure_bar, mole_fractions: jnp.zeros(2)
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
        lnphi_func=lnphi_func,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["lnphi_func"] is lnphi_func


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
    warmup_calls = []

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.solve.equilibrium",
        lambda *args, **kwargs: SimpleNamespace(
            ln_n=gas_ln_n,
            ntot=jnp.asarray(1.0, dtype=jnp.float64),
        ),
    )

    def fake_run_fixed_support_profile(**kwargs):
        warmup_calls.append(kwargs)
        return {
            "compilation_seconds": 0.25,
            "execution_seconds": 0.5,
            "diagnostic_seconds": 0.0,
            "diagnostic_compilation_seconds": 0.0,
            "diagnostic_execution_seconds": 0.0,
            "backend": "cpu",
        }

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.batch."
            "run_fixed_support_profile"
        ),
        fake_run_fixed_support_profile,
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
    assert len(warmup_calls) == 1
    warmup_bucket = warmup_calls[0]["buckets"][0]
    assert warmup_bucket.support_indices.shape == (1, 2)
    assert not np.any(np.asarray(warmup_bucket.condensate_slot_mask))
    assert layer.diagnostics["fixed_support_v2"]["outcome"] == (
        "gas_only_no_candidate"
    )
    assert layer.diagnostics["fixed_support_v2"]["fixed_shape_warmup"]


def test_head_v2_empty_catalog_skips_unnecessary_fixed_shape_warmup(
    monkeypatch,
):
    base = _fake_setup()
    empty_condensates = ChemicalSetup(
        formula_matrix=jnp.zeros((2, 0), dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.zeros((0,), dtype=jnp.float64),
        elements=base.elements,
        species=(),
        metadata={},
    )
    setup = CondensateChemicalSetup(
        gas_setup=base.gas_setup,
        condensate_setup=empty_condensates,
        formula_matrix=base.formula_matrix,
        formula_matrix_cond=empty_condensates.formula_matrix,
        gas_species=base.gas_species,
        condensate_species=(),
        elements=base.elements,
    )
    gas_ln_n = jnp.log(jnp.asarray([0.5, 0.5], dtype=jnp.float64))
    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.solve.equilibrium",
        lambda *args, **kwargs: SimpleNamespace(
            ln_n=gas_ln_n,
            ntot=jnp.asarray(1.0, dtype=jnp.float64),
        ),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
    )

    assert result.layers[0].converged
    assert not result.layers[0].diagnostics["fixed_support_v2"][
        "fixed_shape_warmup"
    ]


def test_head_v2_grid_like_gas_init_seeds_exact_gas_solves(
    monkeypatch,
):
    setup = _fake_setup()
    seed_ln_n = jnp.log(jnp.asarray([0.25, 0.75], dtype=jnp.float64))
    exact_ln_n = jnp.log(jnp.asarray([0.5, 0.5], dtype=jnp.float64))
    gas_inits = []
    activity_gas_states = []

    def fake_gas_equilibrium(*args, **kwargs):
        gas_inits.append(kwargs.get("init"))
        return SimpleNamespace(
            ln_n=exact_ln_n,
            ntot=jnp.asarray(1.0, dtype=jnp.float64),
        )

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.solve.equilibrium",
        fake_gas_equilibrium,
    )

    def fake_element_potential(*, gas_ln_n, **kwargs):
        activity_gas_states.append(gas_ln_n)
        return jnp.zeros((2,), dtype=jnp.float64)

    monkeypatch.setattr(
        _lifecycle,
        "_least_squares_element_potential",
        fake_element_potential,
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
    initial = CondensateEquilibriumInit(
        gas_ln_n=seed_ln_n,
        gas_ntot=jnp.asarray(1.0, dtype=jnp.float64),
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        T=np.asarray([1000.0]),
        P=np.asarray([1.0]),
        b=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
        init=(initial,),
    )

    assert result.layers[0].converged
    assert len(gas_inits) == 2
    for gas_init in gas_inits:
        assert gas_init is not None
        np.testing.assert_allclose(gas_init.ln_nk, seed_ln_n)
        assert float(gas_init.ln_ntot) == pytest.approx(0.0)
    assert len(activity_gas_states) == 1
    np.testing.assert_allclose(activity_gas_states[0], exact_ln_n)


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


def test_head_v2_real_solver_is_amount_gauge_covariant(monkeypatch):
    setup = _amount_gauge_fake_setup()
    monkeypatch.setattr(
        _lifecycle,
        "_native_activity_expanded_profile_support_payload",
        lambda **kwargs: (
            (0,),
            (0.2,),
            {"policy": "test_deterministic_support"},
        ),
    )

    scales = (1.0e-12, 1.0, 1.0e8)
    layers = []
    for scale in scales:
        initial = CondensateEquilibriumInit(
            gas_ln_n=jnp.log(
                scale * jnp.asarray([0.4, 0.4], dtype=jnp.float64)
            ),
            gas_ntot=jnp.asarray(0.8 * scale, dtype=jnp.float64),
            support_indices=(0,),
            support_amounts=(0.2 * scale,),
            element_potential=jnp.full(
                (2,), math.log(0.5), dtype=jnp.float64
            ),
        )
        profile = condmod.condensate_equilibrium_profile(
            setup,
            T=np.asarray([1000.0]),
            P=np.asarray([1.0]),
            b=scale * jnp.asarray([0.6, 0.4], dtype=jnp.float64),
            init=(initial,),
            options=CondensateEquilibriumOptions(return_diagnostics=True),
            return_diagnostics=True,
        )
        layer = profile.layers[0]
        lifecycle = layer.diagnostics["fixed_support_v2"]
        assert layer.converged
        polish = lifecycle["zero_barrier_active_support_polish"]
        closure = polish["exact_active_set_closure"]
        assert polish["accepted"]
        assert closure["accepted"]
        assert closure["termination_reason"] == "accepted"
        assert closure["round_count"] >= 1
        assert lifecycle["caller_gauge_zero_barrier_kkt"]["accepted"]
        layers.append(layer)

    reference = layers[1]
    reference_polish = reference.diagnostics["fixed_support_v2"][
        "zero_barrier_active_support_polish"
    ]
    for scale, layer in zip(scales, layers):
        np.testing.assert_allclose(
            np.asarray(layer.gas_n) / scale,
            np.asarray(reference.gas_n),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(layer.condensate_amounts) / scale,
            np.asarray(reference.condensate_amounts),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(layer.gas_x),
            np.asarray(reference.gas_x),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        polish = layer.diagnostics["fixed_support_v2"][
            "zero_barrier_active_support_polish"
        ]
        np.testing.assert_allclose(
            polish["element_potential"],
            reference_polish["element_potential"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
