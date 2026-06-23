import jax.numpy as jnp

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
