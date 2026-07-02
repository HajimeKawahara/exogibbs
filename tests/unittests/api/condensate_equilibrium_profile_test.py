import jax.numpy as jnp
import pytest

import exogibbs.api.condensate_equilibrium as condmod
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    CondensateChemicalSetup,
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    CondensateEquilibriumResult,
)


def _fake_setup() -> CondensateChemicalSetup:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=lambda T: jnp.zeros((2,), dtype=jnp.float64),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    cond_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=lambda T: jnp.zeros((2,), dtype=jnp.float64),
        elements=("H", "O"),
        species=("H[s]", "O[s]"),
        metadata={},
    )
    return CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=cond_setup,
        formula_matrix=gas_setup.formula_matrix,
        formula_matrix_cond=cond_setup.formula_matrix,
        gas_species=gas_setup.species,
        condensate_species=cond_setup.species,
        elements=gas_setup.elements,
    )


def _fake_result(layer_value: float, support_index: int = 1) -> CondensateEquilibriumResult:
    amounts = jnp.zeros((2,), dtype=jnp.float64).at[support_index].set(layer_value)
    return CondensateEquilibriumResult(
        gas_ln_n=jnp.asarray([layer_value, layer_value + 1.0], dtype=jnp.float64),
        gas_n=jnp.exp(jnp.asarray([layer_value, layer_value + 1.0], dtype=jnp.float64)),
        gas_x=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        gas_ntot=jnp.asarray(1.0, dtype=jnp.float64),
        condensate_amounts=amounts,
        condensate_support_indices=jnp.asarray([support_index], dtype=jnp.int32),
        condensate_support_names=("O[s]",),
        acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
        selected_route="m4310_full_promoted_policy_route",
        status="converged",
        converged=True,
        diagnostics={},
    )


def test_condensate_profile_scan_hot_from_top_uses_previous_layer_init(monkeypatch):
    setup = _fake_setup()
    calls = []

    def fake_condensate_equilibrium(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(float(len(calls)))

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0, 1200.0]),
        jnp.asarray([1.0, 1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        method="scan_hot_from_top",
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )

    assert result.method == "scan_hot_from_top"
    assert len(result.layers) == 3
    assert calls[0].get("support_indices") is None
    assert calls[1]["support_indices"] == (1,)
    assert calls[1]["support_amounts_init"] == (1.0,)
    assert isinstance(calls[1]["init"], CondensateEquilibriumInit)
    assert result.diagnostics["warm_start_attempt_count"] == 2


def test_condensate_profile_scan_hot_from_bottom_preserves_output_order(monkeypatch):
    setup = _fake_setup()
    temperatures_seen = []

    def fake_condensate_equilibrium(setup_arg, T, *args, **kwargs):
        temperatures_seen.append(float(T))
        return _fake_result(float(T) / 1000.0)

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0, 1200.0]),
        jnp.asarray([1.0, 1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        method="scan_hot_from_bottom",
    )

    assert temperatures_seen == [1200.0, 1100.0, 1000.0]
    assert [float(layer.condensate_amounts[1]) for layer in result.layers] == [
        1.0,
        1.1,
        1.2,
    ]


def test_condensate_profile_initializer_defaults_to_independent_layers(monkeypatch):
    setup = _fake_setup()
    previous_seen = []
    calls = []

    class ExplicitInitializer:
        def __call__(self, request):
            previous_seen.append(request.previous_solution)
            return CondensateEquilibriumInit(
                gas_ln_n=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
                gas_ntot=jnp.asarray(2.0, dtype=jnp.float64),
                support_indices=(0,),
                support_amounts=(0.5,),
            )

    def fake_condensate_equilibrium(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(float(len(calls)), support_index=0)

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        initializer=ExplicitInitializer(),
    )

    assert result.method == "vmap_cold"
    assert previous_seen == [None, None]
    assert calls[0]["support_indices"] == (0,)
    assert calls[1]["support_indices"] == (0,)


def test_condensate_profile_method_can_be_selected_from_options(monkeypatch):
    setup = _fake_setup()
    calls = []

    def fake_condensate_equilibrium(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(float(len(calls)))

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(profile_method="scan_hot_from_top"),
    )

    assert result.method == "scan_hot_from_top"
    assert calls[0].get("support_indices") is None
    assert calls[1]["support_indices"] == (1,)


def test_condensate_profile_auto_defaults_to_scan_without_fixed_support(monkeypatch):
    setup = _fake_setup()
    calls = []

    def fake_condensate_equilibrium(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(float(len(calls)))

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
    )

    assert result.method == "scan_hot_from_top"
    assert calls[0].get("support_indices") is None
    assert calls[1]["support_indices"] == (1,)


def test_condensate_profile_can_warm_start_gas_with_explicit_support(monkeypatch):
    setup = _fake_setup()
    calls = []

    def fake_condensate_equilibrium(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(float(len(calls)), support_index=1)

    monkeypatch.setattr(
        condmod,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
        raising=True,
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        support_indices=(0,),
        support_amounts_init=(0.25,),
        options=CondensateEquilibriumOptions(
            profile_method="scan_hot_from_top",
            profile_warm_start_support_policy="explicit_payload",
        ),
        return_diagnostics=True,
    )

    assert calls[0]["support_indices"] == (0,)
    assert calls[0]["init"].gas_ln_n is None
    assert calls[1]["support_indices"] == (0,)
    assert calls[1]["support_amounts_init"] == (0.25,)
    assert isinstance(calls[1]["init"], CondensateEquilibriumInit)
    assert result.diagnostics["layers"][1]["initialization_mode"] == (
        "initializer_gas_with_explicit_support_payload"
    )


def test_condensate_profile_experimental_fixed_support_batch_path():
    def profile_hvector(T):
        T = jnp.asarray(T)
        if T.ndim > 0:
            return jnp.zeros((T.shape[0], 2), dtype=jnp.float64)
        return jnp.zeros((2,), dtype=jnp.float64)

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    cond_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H[s]", "O[s]"),
        metadata={},
    )
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=cond_setup,
        formula_matrix=gas_setup.formula_matrix,
        formula_matrix_cond=cond_setup.formula_matrix,
        gas_species=gas_setup.species,
        condensate_species=cond_setup.species,
        elements=gas_setup.elements,
    )
    init = tuple(
        CondensateEquilibriumInit(
            gas_ln_n=jnp.log(jnp.asarray([0.8, 0.8], dtype=jnp.float64)),
            gas_ntot=jnp.asarray(1.6, dtype=jnp.float64),
            support_indices=(0, 1),
            support_amounts=(0.2, 0.2),
        )
        for _ in range(2)
    )

    result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=init,
        options=CondensateEquilibriumOptions(
            profile_method="vmap_cold",
            profile_warm_start_support_policy="explicit_payload",
            enable_experimental_profile_fixed_support_batch=True,
            enable_full_condensate_budget_residual_gate=False,
            max_inner_iterations=8,
        ),
        return_diagnostics=True,
    )

    assert result.method == "vmap_cold"
    assert len(result.layers) == 2
    assert result.diagnostics["fresh_fallback_count"] == 0
    assert all(layer.converged for layer in result.layers)
    assert tuple(result.layers[0].condensate_support_indices.tolist()) == (0, 1)
    assert result.layers[0].selected_route == "m4310_full_promoted_policy_route"

    fast_result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=init,
        options=CondensateEquilibriumOptions(
            profile_method="vmap_cold",
            profile_warm_start_support_policy="explicit_payload",
            enable_experimental_profile_fixed_support_batch=True,
            enable_full_condensate_budget_residual_gate=False,
            max_inner_iterations=8,
        ),
        return_diagnostics=False,
    )

    assert fast_result.batched_arrays is None
    assert tuple(fast_result.layers[0].condensate_support_indices.tolist()) == (0, 1)

    rescue_profile_result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=init,
        options=CondensateEquilibriumOptions(
            profile_method="vmap_cold",
            profile_warm_start_support_policy="explicit_payload",
            enable_experimental_profile_fixed_support_batch=True,
            enable_experimental_profile_fixed_support_fallback_rescue=True,
            enable_full_condensate_budget_residual_gate=False,
            max_inner_iterations=8,
        ),
        return_diagnostics=True,
    )

    assert rescue_profile_result.method == "vmap_cold"
    assert all(layer.converged for layer in rescue_profile_result.layers)
    rescue_profile_diagnostics = rescue_profile_result.diagnostics[
        "experimental_profile_fixed_support_batch"
    ]
    assert (
        rescue_profile_diagnostics["route"]
        == "experimental_profile_fixed_support_batch_fallback_rescue"
    )
    assert rescue_profile_diagnostics["fallback_rescue"]["mode"] == "none"
    assert (
        rescue_profile_result.layers[0].selected_route
        == "experimental_profile_fixed_support_batch_fallback_rescue"
    )

    auto_rescue_profile_result = condmod.condensate_equilibrium_profile(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=init,
        options=CondensateEquilibriumOptions(
            enable_full_condensate_budget_residual_gate=False,
            max_inner_iterations=8,
        ),
        return_diagnostics=True,
    )

    assert auto_rescue_profile_result.method == "vmap_cold"
    auto_profile_diagnostics = auto_rescue_profile_result.diagnostics[
        "experimental_profile_fixed_support_batch"
    ]
    assert (
        auto_profile_diagnostics["route"]
        == "experimental_profile_fixed_support_batch_fallback_rescue"
    )

    plan = condmod.prepare_experimental_profile_fixed_support_batch_plan(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=init,
        options=CondensateEquilibriumOptions(max_inner_iterations=8),
    )
    planned_arrays = condmod.run_experimental_profile_fixed_support_batch_plan(plan)
    planned_arrays_with_shared_b = (
        condmod.run_experimental_profile_fixed_support_batch_plan(
            plan,
            element_inventory_target=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        )
    )
    planned_arrays_with_layer_b = (
        condmod.run_experimental_profile_fixed_support_batch_plan(
            plan,
            element_inventory_target=jnp.asarray(
                [[1.0, 1.0], [1.0, 1.0]],
                dtype=jnp.float64,
            ),
        )
    )

    assert isinstance(
        plan,
        condmod.ExperimentalCondensateProfileFixedSupportBatchPlan,
    )
    assert planned_arrays["gas_ln_n"].shape == (2, 2)
    assert planned_arrays["condensate_amounts"].shape == (2, 2)
    assert planned_arrays["fallback_required"].shape == (2,)
    assert jnp.array_equal(
        planned_arrays["fallback_required"],
        ~planned_arrays["converged"],
    )
    assert jnp.allclose(
        planned_arrays_with_shared_b["gas_ln_n"],
        planned_arrays["gas_ln_n"],
    )
    assert jnp.allclose(
        planned_arrays_with_layer_b["condensate_amounts"],
        planned_arrays["condensate_amounts"],
    )
    with pytest.raises(ValueError, match="element_inventory_target"):
        condmod.run_experimental_profile_fixed_support_batch_plan(
            plan,
            element_inventory_target=jnp.asarray([1.0], dtype=jnp.float64),
        )

    many_arrays = condmod.run_experimental_profile_fixed_support_batch_plan_many(
        plan,
        jnp.asarray(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float64,
        ),
    )
    layer_many_arrays = condmod.run_experimental_profile_fixed_support_batch_plan_many(
        plan,
        jnp.asarray(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=jnp.float64,
        ),
    )

    assert many_arrays["gas_ln_n"].shape == (2, 2, 2)
    assert many_arrays["condensate_amounts"].shape == (2, 2, 2)
    assert many_arrays["fallback_required"].shape == (2, 2)
    assert jnp.array_equal(
        many_arrays["fallback_required"],
        ~many_arrays["converged"],
    )
    assert set(many_arrays["residual_components"]) == {
        "budget",
        "complementarity",
        "condensate_stationarity",
        "gas",
        "total_density",
    }
    for component in many_arrays["residual_components"].values():
        assert component.shape == (2, 2)
        assert jnp.all(jnp.isfinite(component))
    reconstructed_residual = jnp.sqrt(
        sum(
            component * component
            for component in many_arrays["residual_components"].values()
        )
    )
    assert jnp.allclose(
        reconstructed_residual,
        many_arrays["final_residual"],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert set(many_arrays["step_diagnostics"]) == {
        "accepted_iteration_count",
        "fallback_accepted_iteration_count",
        "initial_residual",
        "lambda_selection_index",
        "normal_accepted_iteration_count",
        "stationarity_restoration_accepted_iteration_count",
    }
    for diagnostic in many_arrays["step_diagnostics"].values():
        assert diagnostic.shape == (2, 2)
    assert jnp.all(
        many_arrays["step_diagnostics"]["accepted_iteration_count"]
        == (
            many_arrays["step_diagnostics"]["normal_accepted_iteration_count"]
            + many_arrays["step_diagnostics"]["fallback_accepted_iteration_count"]
        )
    )
    assert jnp.allclose(many_arrays["gas_ln_n"][0], planned_arrays["gas_ln_n"])
    assert jnp.allclose(many_arrays["gas_ln_n"][1], planned_arrays["gas_ln_n"])
    assert jnp.allclose(
        layer_many_arrays["condensate_amounts"][0],
        planned_arrays["condensate_amounts"],
    )
    direct_rescue_base_arrays = condmod.run_experimental_profile_fixed_support_batch_plan(
        plan,
        element_inventory_target=jnp.asarray(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=jnp.float64,
        ),
        rho_initialization="complementarity",
        lambda_initialization="best_residual",
        residual_tolerance_multiplier=1.0e9,
    )
    direct_many_rescue_base_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_many(
            plan,
            jnp.asarray(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                dtype=jnp.float64,
            ),
            rho_initialization="complementarity",
            lambda_initialization="best_residual",
            residual_tolerance_multiplier=1.0e9,
        )
    )
    rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue(
            plan,
            element_inventory_target=jnp.asarray(
                [[1.0, 1.0], [1.0, 1.0]],
                dtype=jnp.float64,
            ),
        )
    )
    many_rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_many_with_fallback_rescue(
            plan,
            jnp.asarray(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                dtype=jnp.float64,
            ),
        )
    )
    empty_rescue = condmod.prepare_experimental_profile_fixed_support_prune_rescue_plan(
        plan,
        (),
    )
    prepared_rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_with_prepared_fallback_rescue(
            plan,
            empty_rescue,
            element_inventory_target=jnp.asarray(
                [[1.0, 1.0], [1.0, 1.0]],
                dtype=jnp.float64,
            ),
        )
    )
    prepared_many_rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_many_with_prepared_fallback_rescue(
            plan,
            empty_rescue,
            jnp.asarray(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                dtype=jnp.float64,
            ),
        )
    )
    rescue_cache = (
        condmod.ExperimentalCondensateProfileFixedSupportPruneRescueCache()
    )
    cached_rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_with_cached_fallback_rescue(
            plan,
            rescue_cache,
            element_inventory_target=jnp.asarray(
                [[1.0, 1.0], [1.0, 1.0]],
                dtype=jnp.float64,
            ),
        )
    )
    cached_many_rescue_arrays = (
        condmod.run_experimental_profile_fixed_support_batch_plan_many_with_cached_fallback_rescue(
            plan,
            rescue_cache,
            jnp.asarray(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                dtype=jnp.float64,
            ),
        )
    )
    assert rescue_arrays["fallback_required"].shape == (2,)
    assert rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert rescue_arrays["fallback_rescue"]["expanded_layer_count"] == 0
    assert rescue_arrays["fallback_rescue"]["replaced_count"] == 0
    assert jnp.allclose(
        rescue_arrays["gas_ln_n"],
        direct_rescue_base_arrays["gas_ln_n"],
    )
    assert many_rescue_arrays["fallback_required"].shape == (2, 2)
    assert many_rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert many_rescue_arrays["fallback_rescue"]["expanded_layer_count"] == 0
    assert jnp.allclose(
        many_rescue_arrays["gas_ln_n"],
        direct_many_rescue_base_arrays["gas_ln_n"],
    )
    assert prepared_rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert prepared_many_rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert jnp.allclose(
        prepared_rescue_arrays["gas_ln_n"],
        direct_rescue_base_arrays["gas_ln_n"],
    )
    assert jnp.allclose(
        prepared_many_rescue_arrays["gas_ln_n"],
        direct_many_rescue_base_arrays["gas_ln_n"],
    )
    assert rescue_cache.prepare_count == 1
    assert rescue_cache.hit_count == 1
    assert cached_rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert cached_many_rescue_arrays["fallback_rescue"]["mode"] == "none"
    assert jnp.allclose(
        cached_rescue_arrays["gas_ln_n"],
        direct_rescue_base_arrays["gas_ln_n"],
    )
    assert jnp.allclose(
        cached_many_rescue_arrays["gas_ln_n"],
        direct_many_rescue_base_arrays["gas_ln_n"],
    )

    prune_init = tuple(
        CondensateEquilibriumInit(
            gas_ln_n=jnp.log(jnp.asarray([0.8, 0.8], dtype=jnp.float64)),
            gas_ntot=jnp.asarray(1.6, dtype=jnp.float64),
            support_indices=(0, 1),
            support_amounts=(0.2, 1.0e-4),
        )
        for _ in range(2)
    )
    prune_plan = condmod.prepare_experimental_profile_fixed_support_batch_plan(
        setup,
        jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        init=prune_init,
        options=CondensateEquilibriumOptions(max_inner_iterations=8),
    )
    prune_rescue = condmod.prepare_experimental_profile_fixed_support_prune_rescue_plan(
        prune_plan,
        (0,),
        prune_relative_floors=(1.0e-5, 1.0e-3),
    )
    rescue_plan = prune_rescue.rescue_plan
    rescue_metadata = prune_rescue.metadata
    assert rescue_plan is not None
    assert rescue_plan.n_layers == 2
    assert rescue_metadata["expanded_to_original_layer"] == (0, 0)
    assert rescue_metadata["candidate_labels"] == (
        "current",
        "prune_amount_ge_0.001_max",
    )
    assert rescue_metadata["candidate_support_counts"] == (2, 1)
    with pytest.raises(ValueError, match="element_inventory_targets"):
        condmod.run_experimental_profile_fixed_support_batch_plan_many(
            plan,
            jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        )


def test_condensate_profile_fixed_support_batch_builds_default_depleted_gas_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def profile_hvector(T):
        T = jnp.asarray(T)
        if T.ndim > 0:
            return jnp.zeros((T.shape[0], 2), dtype=jnp.float64)
        return jnp.zeros((2,), dtype=jnp.float64)

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    cond_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H[s]", "O[s]"),
        metadata={},
    )
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=cond_setup,
        formula_matrix=gas_setup.formula_matrix,
        formula_matrix_cond=cond_setup.formula_matrix,
        gas_species=gas_setup.species,
        condensate_species=cond_setup.species,
        elements=gas_setup.elements,
    )
    captured = {}

    def fake_prepare_buckets(**kwargs):
        captured["init_states"] = kwargs["init_states"]
        raise RuntimeError("stop after init construction")

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "_prepare_pdipm_rgie_v11_activity_correction_profile_buckets",
        fake_prepare_buckets,
    )

    with pytest.raises(RuntimeError, match="stop after init construction"):
        condmod.condensate_equilibrium_profile(
            setup,
            jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
            jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            support_indices=(0,),
            support_amounts_init=(0.25,),
            options=CondensateEquilibriumOptions(
                profile_method="vmap_cold",
                profile_warm_start_support_policy="explicit_payload",
                enable_experimental_profile_fixed_support_batch=True,
                fixed_support_gas_init_policy="depleted_budget",
                enable_profile_native_activity_support_expansion=False,
            ),
        )

    init_states = captured["init_states"]
    assert len(init_states) == 2
    assert all(
        init.ln_nk_source_trace["fixed_support_gas_init_policy"] == "depleted_budget"
        for init in init_states
    )
    assert tuple(float(value) for value in jnp.exp(init_states[0].ln_nk)) == pytest.approx(
        (0.75, 1.0)
    )


def test_condensate_profile_batch_can_expand_support_from_native_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def profile_hvector(T):
        T = jnp.asarray(T)
        if T.ndim > 0:
            return jnp.zeros((T.shape[0], 2), dtype=jnp.float64)
        return jnp.zeros((2,), dtype=jnp.float64)

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    cond_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        hvector_func=profile_hvector,
        elements=("H", "O"),
        species=("H[s]", "O[s]"),
        metadata={},
    )
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=cond_setup,
        formula_matrix=gas_setup.formula_matrix,
        formula_matrix_cond=cond_setup.formula_matrix,
        gas_species=gas_setup.species,
        condensate_species=cond_setup.species,
        elements=gas_setup.elements,
    )

    class FakeGasResult:
        ln_n = jnp.asarray([0.0, 2.0], dtype=jnp.float64)
        ntot = jnp.asarray(1.0, dtype=jnp.float64)

    def fake_equilibrium(*_args, **_kwargs):
        return FakeGasResult()

    captured = {}

    def fake_prepare_buckets(**kwargs):
        captured["support_indices_by_layer"] = kwargs["support_indices_by_layer"]
        captured["init_states"] = kwargs["init_states"]
        raise RuntimeError("stop after support expansion")

    import exogibbs.api.equilibrium as api_equilibrium
    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(api_equilibrium, "equilibrium", fake_equilibrium)
    monkeypatch.setattr(
        minimize_cond,
        "_prepare_pdipm_rgie_v11_activity_correction_profile_buckets",
        fake_prepare_buckets,
    )

    with pytest.raises(RuntimeError, match="stop after support expansion"):
        condmod.condensate_equilibrium_profile(
            setup,
            jnp.asarray([1000.0], dtype=jnp.float64),
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            support_indices=(0,),
            support_amounts_init=(0.25,),
            options=CondensateEquilibriumOptions(
                profile_method="vmap_cold",
                profile_warm_start_support_policy="explicit_payload",
                enable_experimental_profile_fixed_support_batch=True,
                enable_profile_native_activity_support_expansion=True,
                profile_native_activity_support_topk=2,
                profile_native_activity_max_support_count=2,
                seed_fraction=0.25,
                max_seed_amount=1.0,
            ),
        )

    assert captured["support_indices_by_layer"] == ((0, 1),)
    init = captured["init_states"][0]
    assert tuple(float(value) for value in jnp.exp(init.ln_mk)) == pytest.approx(
        (0.25, 0.25)
    )
    assert (
        init.ln_nk_source_trace["fixed_support_gas_init_policy"]
        == "depleted_budget"
    )
