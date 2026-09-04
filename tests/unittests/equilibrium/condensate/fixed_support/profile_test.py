from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from exogibbs.api.condensate_equilibrium import (
    ExperimentalCondensateProfileFixedSupportBatchPlan,
    run_experimental_profile_fixed_support_v2_batch_plan,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    ContinuationConfig,
    FixedSupportV2Config,
    SOCConfig,
    SolverLimitConfig,
    TerminalStatus,
)
from exogibbs.equilibrium.condensate.fixed_support.batch import (
    FixedSupportV2BatchShape,
    PreparedFixedSupportV2LayerState,
    _compiled_solver_factory,
    _prepared_original_state_batch,
    _prepared_problem_batch,
    prepare_fixed_support_v2_buckets,
    run_fixed_support_profile,
)
from exogibbs.equilibrium.condensate.support import (
    evaluate_profile_support_closure,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _PreparedBucket:
    support_indices: tuple[int, ...]
    layer_indices: tuple[int, ...]
    formula_matrix_cond_active: jax.Array
    ln_nk_init: jax.Array
    ln_mk_init: jax.Array
    ln_ntot_init: jax.Array
    element_potential_init: jax.Array | None
    rho_init: jax.Array | None
    barrier_epsilon_init: jax.Array | None
    element_inventory_target: jax.Array
    hvector: jax.Array
    hvector_cond_active: jax.Array
    ln_normalized_pressure: jax.Array


def _bucket():
    qtot = jnp.log(jnp.asarray([0.8, 0.8]))
    return _PreparedBucket(
        support_indices=(0,),
        layer_indices=(0, 1),
        formula_matrix_cond_active=jnp.asarray([[1.0]]),
        ln_nk_init=jnp.log(jnp.asarray([[0.8], [0.8]])),
        ln_mk_init=jnp.log(jnp.asarray([[0.2], [0.2]])),
        ln_ntot_init=qtot,
        element_potential_init=jnp.zeros((2, 1)),
        rho_init=None,
        barrier_epsilon_init=None,
        element_inventory_target=jnp.ones((2, 1)),
        hvector=jnp.zeros((2, 1)),
        hvector_cond_active=jnp.full((2, 1), 0.5),
        ln_normalized_pressure=jnp.zeros((2,)),
    )


def _fixed_shape_prepare_kwargs():
    return {
        "init_states": (
            PreparedFixedSupportV2LayerState(
                ln_nk=jnp.log(jnp.asarray([0.8])),
                ln_mk=jnp.log(jnp.asarray([0.2])),
                ln_ntot=jnp.log(jnp.asarray(0.8)),
                element_potential=jnp.zeros((1,)),
            ),
            PreparedFixedSupportV2LayerState(
                ln_nk=jnp.log(jnp.asarray([0.8])),
                ln_mk=jnp.log(jnp.asarray([0.2])),
                ln_ntot=jnp.log(jnp.asarray(0.8)),
                element_potential=jnp.zeros((1,)),
            ),
        ),
        "support_indices_by_layer": ((0,), (1,)),
        "formula_matrix_cond": jnp.asarray([[1.0, 1.0]]),
        "element_inventory_target_by_layer": jnp.ones((2, 1)),
        "hvector_by_layer": jnp.zeros((2, 1)),
        "hvector_cond_by_layer": jnp.asarray([[0.5, 2.0], [2.0, 0.5]]),
        "ln_normalized_pressure_by_layer": jnp.zeros((2,)),
    }


def _fixed_shape_bucket():
    return prepare_fixed_support_v2_buckets(
        **_fixed_shape_prepare_kwargs(),
        fixed_shape=FixedSupportV2BatchShape(
            support_capacity=2,
            batch_capacity=3,
        ),
    )[0]


def _config():
    return FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(math.log(0.1),),
        ),
        soc=SOCConfig(enabled=False),
        limits=SolverLimitConfig(
            max_normal_iterations=4,
            max_line_search_trials=4,
            max_restoration_calls=1,
            max_restoration_iterations=4,
            max_restoration_line_search_trials=4,
        ),
    )


def _run_profile(
    inactive_source,
    *,
    condensate_valid_mask=None,
    include_terminal_diagnostics=True,
    support_closure_tolerance=1.0e-8,
):
    fixed_support_result = run_fixed_support_profile(
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        layer_count=2,
        condensate_count=2,
        config=_config(),
        include_terminal_diagnostics=include_terminal_diagnostics,
    )
    return evaluate_profile_support_closure(
        fixed_support_result,
        formula_matrix=jnp.asarray([[1.0]]),
        formula_matrix_cond_full=jnp.asarray([[1.0, 1.0]]),
        condensate_standard_source_full=jnp.asarray(
            [[0.5, inactive_source], [0.5, inactive_source]]
        ),
        condensate_valid_mask=condensate_valid_mask,
        support_closure_tolerance=support_closure_tolerance,
    )


def test_prepared_profile_runs_one_masked_bucket_and_reports_closed_support():
    result = _run_profile(2.0)

    assert result["schema"] == "exogibbs_fixed_support_v2_prepared_profile_v1"
    assert result["experimental"] is True
    assert result["production_preset_promoted"] is False
    assert result["backend"] == jax.default_backend()
    assert result["compilation_seconds"] >= 0.0
    assert result["execution_seconds"] >= 0.0
    assert result["diagnostic_seconds"] >= 0.0
    assert result["diagnostic_compilation_seconds"] >= 0.0
    assert result["diagnostic_execution_seconds"] >= 0.0
    assert result["diagnostic_seconds"] == pytest.approx(
        result["diagnostic_compilation_seconds"]
        + result["diagnostic_execution_seconds"]
    )
    assert result["terminal_status"] == pytest.approx(
        jnp.full((2,), int(TerminalStatus.CONVERGED))
    )
    assert jnp.all(result["fixed_support_converged"])
    assert jnp.all(result["final_state_values_finite"])
    assert jnp.all(result["support_closed"])
    assert jnp.all(result["converged_with_support_closure"])
    assert result["inventory_residual"] == pytest.approx(jnp.zeros((2, 1)))
    assert result["inventory_residual_scaled"] == pytest.approx(
        jnp.zeros((2, 1))
    )
    assert result["final_kkt_norms"].budget_scaled == pytest.approx(
        jnp.zeros((2,))
    )
    assert result["condensate_amounts"] == pytest.approx(
        jnp.asarray([[0.2, 0.0], [0.2, 0.0]])
    )
    assert len(result["bucket_reports"]) == 1
    report = result["bucket_reports"][0]
    assert report["diagnostic_compilation_seconds"] >= 0.0
    assert report["diagnostic_execution_seconds"] >= 0.0
    restoration = report["terminal_restoration_diagnostics"]
    assert restoration["available"].shape == (2,)
    assert not jnp.any(restoration["available"])
    assert restoration["next_iteration_replay"]["alphas"].shape == (2, 4)
    normal = report["terminal_normal_diagnostics"]
    assert normal["available"].shape == (2,)
    assert not jnp.any(normal["available"])
    assert normal["alphas"].shape == (2, 4)
    assert normal["rejection_reasons"].shape == (2, 4)
    assert report[
        "stage_restoration_accepted_iteration_counts"
    ].shape == (2, 1)
    assert report["stage_last_return_diagnostics"].alpha_dual.shape == (2, 1)
    assert report["last_return_diagnostics"].alpha_dual.shape == (2,)


def test_fixed_support_batch_does_not_own_full_catalog_closure():
    result = run_fixed_support_profile(
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        layer_count=2,
        condensate_count=2,
        config=_config(),
    )

    assert "inactive_condensate_driving" not in result
    assert "support_expansion_mask" not in result
    assert "support_closed" not in result
    assert result["support_mask"].shape == (2, 2)


def test_fixed_shape_preparation_builds_heterogeneous_anchored_slots():
    bucket = _fixed_shape_bucket()
    problems = _prepared_problem_batch(
        bucket,
        jnp.asarray([[1.0]]),
        budget_relative_floor=1.0e-6,
    )
    states = _prepared_original_state_batch(bucket, problems, _config())
    epsilon = math.log(0.1)

    assert bucket.valid_batch_size == 2
    assert bucket.layer_indices == (0, 1)
    assert bucket.support_indices.tolist() == [[0, 0], [1, 0], [1, 0]]
    assert bucket.condensate_slot_mask.tolist() == [
        [True, False],
        [True, False],
        [True, False],
    ]
    assert bucket.formula_matrix_cond_active.shape == (3, 1, 2)
    assert bucket.hvector_cond_active == pytest.approx(
        jnp.asarray([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
    )
    assert problems.condensate_formula_matrix[:, :, 1] == pytest.approx(
        jnp.zeros((3, 1))
    )
    assert states.r[:, 1] == pytest.approx(jnp.full((3,), epsilon))
    assert states.rho[:, 1] == pytest.approx(jnp.zeros((3,)))


def test_fixed_shape_problem_leaves_do_not_depend_on_real_support_count():
    shape = FixedSupportV2BatchShape(support_capacity=2, batch_capacity=3)
    first = _fixed_shape_bucket()
    changed_kwargs = _fixed_shape_prepare_kwargs()
    changed_kwargs["support_indices_by_layer"] = ((0, 1), (1,))
    changed_kwargs["init_states"] = (
        replace(
            changed_kwargs["init_states"][0],
            ln_mk=jnp.log(jnp.asarray([0.1, 0.1])),
        ),
        changed_kwargs["init_states"][1],
    )
    changed = prepare_fixed_support_v2_buckets(
        **changed_kwargs,
        fixed_shape=shape,
    )[0]

    def problem_signature(bucket):
        problem = _prepared_problem_batch(
            bucket,
            jnp.asarray([[1.0]]),
            budget_relative_floor=1.0e-6,
        )
        return tuple(
            (tuple(leaf.shape), leaf.dtype)
            for leaf in jax.tree_util.tree_leaves(problem)
        )

    assert problem_signature(first) == problem_signature(changed)


def test_fixed_shape_profile_scatter_excludes_dummy_slots_and_batch_rows():
    result = run_fixed_support_profile(
        buckets=(_fixed_shape_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        layer_count=2,
        condensate_count=2,
        config=_config(),
        include_terminal_diagnostics=False,
    )

    assert jnp.all(result["fixed_support_converged"])
    assert result["condensate_amounts"] == pytest.approx(
        jnp.asarray([[0.2, 0.0], [0.0, 0.2]])
    )
    assert result["support_mask"].tolist() == [
        [True, False],
        [False, True],
    ]
    report = result["bucket_reports"][0]
    assert report["support_indices_by_layer"] == ((0,), (1,))
    assert report["support_capacity"] == 2
    assert report["batch_capacity"] == 3
    assert report["valid_batch_size"] == 2
    assert report["terminal_status"].shape == (2,)
    assert report["final_kkt_norms"].budget_scaled.shape == (2,)


def test_fixed_shape_all_dummy_bucket_warms_gas_only_executable():
    bucket = prepare_fixed_support_v2_buckets(
        init_states=(
            PreparedFixedSupportV2LayerState(
                ln_nk=jnp.log(jnp.asarray([1.0])),
                ln_mk=jnp.asarray([], dtype=jnp.float64),
                ln_ntot=jnp.asarray(0.0),
                element_potential=jnp.zeros((1,)),
            ),
        ),
        support_indices_by_layer=((),),
        formula_matrix_cond=jnp.asarray([[1.0]]),
        element_inventory_target_by_layer=jnp.ones((1, 1)),
        hvector_by_layer=jnp.zeros((1, 1)),
        hvector_cond_by_layer=jnp.zeros((1, 1)),
        ln_normalized_pressure_by_layer=jnp.zeros((1,)),
        fixed_shape=FixedSupportV2BatchShape(
            support_capacity=2,
            batch_capacity=1,
        ),
    )[0]

    result = run_fixed_support_profile(
        buckets=(bucket,),
        formula_matrix=jnp.asarray([[1.0]]),
        layer_count=1,
        condensate_count=1,
        config=_config(),
        include_terminal_diagnostics=False,
    )

    assert jnp.all(result["fixed_support_converged"])
    assert result["condensate_amounts"] == pytest.approx(jnp.zeros((1, 1)))
    assert not jnp.any(result["support_mask"])
    assert result["bucket_reports"][0]["support_indices_by_layer"] == ((),)


def test_all_dummy_and_real_support_share_one_compiled_shape():
    config = _config()
    shape = FixedSupportV2BatchShape(support_capacity=2, batch_capacity=1)

    def bucket_for(support, ln_mk):
        return prepare_fixed_support_v2_buckets(
            init_states=(
                PreparedFixedSupportV2LayerState(
                    ln_nk=jnp.log(jnp.asarray([1.0])),
                    ln_mk=jnp.asarray(ln_mk, dtype=jnp.float64),
                    ln_ntot=jnp.asarray(0.0),
                    element_potential=jnp.zeros((1,)),
                ),
            ),
            support_indices_by_layer=(support,),
            formula_matrix_cond=jnp.asarray([[1.0]]),
            element_inventory_target_by_layer=jnp.ones((1, 1)),
            hvector_by_layer=jnp.zeros((1, 1)),
            hvector_cond_by_layer=jnp.asarray([[0.5]]),
            ln_normalized_pressure_by_layer=jnp.zeros((1,)),
            fixed_shape=shape,
        )[0]

    dummy_bucket = bucket_for((), ())
    real_bucket = bucket_for((0,), (math.log(0.2),))
    compiled = _compiled_solver_factory(config, False)
    compiled.clear_cache()

    def run(bucket):
        problems = _prepared_problem_batch(
            bucket,
            jnp.asarray([[1.0]]),
            budget_relative_floor=1.0e-6,
        )
        states = _prepared_original_state_batch(bucket, problems, config)
        result = compiled(problems, states)
        jax.block_until_ready(result)

    run(dummy_bucket)
    cache_size_after_dummy = compiled._cache_size()
    run(real_bucket)

    assert cache_size_after_dummy == 1
    assert compiled._cache_size() == cache_size_after_dummy


@pytest.mark.parametrize(
    ("fixed_shape", "message"),
    [
        (
            FixedSupportV2BatchShape(support_capacity=1, batch_capacity=2),
            "support exceeds fixed support capacity",
        ),
        (
            FixedSupportV2BatchShape(support_capacity=2, batch_capacity=1),
            "layer count exceeds fixed batch capacity",
        ),
    ],
)
def test_fixed_shape_capacity_overflow_fails_without_resizing(
    fixed_shape,
    message,
):
    kwargs = _fixed_shape_prepare_kwargs()
    if fixed_shape.support_capacity == 1:
        kwargs["support_indices_by_layer"] = ((0, 1), (1,))
        kwargs["init_states"] = (
            replace(
                kwargs["init_states"][0],
                ln_mk=jnp.log(jnp.asarray([0.1, 0.1])),
            ),
            kwargs["init_states"][1],
        )

    with pytest.raises(ValueError, match=message):
        prepare_fixed_support_v2_buckets(
            **kwargs,
            fixed_shape=fixed_shape,
        )


def test_fixed_shape_rejects_negative_padding_sentinel():
    bucket = _fixed_shape_bucket()
    invalid = replace(
        bucket,
        support_indices=bucket.support_indices.at[0, 1].set(-1),
    )

    with pytest.raises(ValueError, match="must not use -1 sentinels"):
        run_fixed_support_profile(
            buckets=(invalid,),
            formula_matrix=jnp.asarray([[1.0]]),
            layer_count=2,
            condensate_count=2,
            config=_config(),
            include_terminal_diagnostics=False,
        )


def test_historical_profile_adapter_retains_closure_report():
    from exogibbs.optimize.fixed_support_v2_profile import (
        run_prepared_profile_v2,
    )

    result = run_prepared_profile_v2(
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        formula_matrix_cond_full=jnp.asarray([[1.0, 1.0]]),
        condensate_standard_source_full=jnp.asarray(
            [[0.5, 2.0], [0.5, 2.0]]
        ),
        layer_count=2,
        condensate_count=2,
        config=_config(),
    )

    assert jnp.all(result["support_closed"])


def test_prepared_solver_factory_reuses_the_jit_identity_for_one_config():
    config = _config()

    assert _compiled_solver_factory(config) is _compiled_solver_factory(config)


def test_support_expansion_is_reported_separately_from_solver_convergence():
    result = _run_profile(-1.0)

    assert jnp.all(result["fixed_support_converged"])
    assert not jnp.any(result["support_closed"])
    assert not jnp.any(result["converged_with_support_closure"])
    assert jnp.all(result["support_expansion_mask"][:, 1])
    assert not jnp.any(result["support_expansion_mask"][:, 0])


def test_support_closure_rejects_nonfinite_inactive_driving():
    result = _run_profile(float("nan"))

    assert jnp.all(result["fixed_support_converged"])
    assert jnp.all(jnp.isnan(result["inactive_condensate_driving"][:, 1]))
    assert not jnp.any(result["support_closed"])
    assert not jnp.any(result["converged_with_support_closure"])
    assert jnp.all(result["support_expansion_mask"][:, 1])
    assert not jnp.any(result["support_expansion_mask"][:, 0])


@pytest.mark.parametrize("tolerance", (-1.0, float("nan"), float("inf")))
def test_support_closure_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _run_profile(-1.0, support_closure_tolerance=tolerance)


def test_prepared_profile_can_skip_terminal_diagnostic_compilation():
    result = _run_profile(
        2.0,
        include_terminal_diagnostics=False,
    )

    assert result["diagnostic_seconds"] == 0.0
    assert result["diagnostic_compilation_seconds"] == 0.0
    assert result["diagnostic_execution_seconds"] == 0.0
    assert result["bucket_reports"][0]["diagnostic_compilation_seconds"] == 0.0
    assert result["bucket_reports"][0]["diagnostic_execution_seconds"] == 0.0
    assert result["bucket_reports"][0]["terminal_restoration_diagnostics"] is None
    assert result["bucket_reports"][0]["terminal_normal_diagnostics"] is None
    assert result["bucket_reports"][0]["stage_statuses"] is None
    assert result["bucket_reports"][0]["last_return_diagnostics"] is None


@pytest.mark.parametrize("inactive_source", (-1.0, float("nan")))
def test_support_closure_ignores_temperature_invalid_condensates(
    inactive_source,
):
    result = _run_profile(
        inactive_source,
        condensate_valid_mask=jnp.asarray(
            [[True, False], [True, False]]
        ),
    )

    assert jnp.all(result["fixed_support_converged"])
    assert jnp.all(result["support_closed"])
    assert not jnp.any(result["support_expansion_mask"][:, 1])


def test_prepared_exact_state_policy_preserves_rho_and_barrier_epsilon():
    bucket = _bucket()
    supplied_rho = jnp.asarray([[1.25], [-2.5]])
    supplied_epsilon = jnp.asarray([math.log(0.1), math.log(0.1)])
    bucket = _PreparedBucket(
        **{
            **bucket.__dict__,
            "rho_init": supplied_rho,
            "barrier_epsilon_init": supplied_epsilon,
        }
    )
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(math.log(0.1),),
            initial_state_policy="provided",
        )
    )
    problems = _prepared_problem_batch(
        bucket,
        jnp.asarray([[1.0]]),
        budget_relative_floor=1.0e-6,
    )

    states = _prepared_original_state_batch(bucket, problems, config)

    assert states.rho == pytest.approx(supplied_rho)
    assert states.epsilon == pytest.approx(supplied_epsilon)


def test_prepared_center_policy_centers_bound_multipliers():
    bucket = _bucket()
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(math.log(0.1),),
            initial_state_policy="center",
        )
    )
    problems = _prepared_problem_batch(
        bucket,
        jnp.asarray([[1.0]]),
        budget_relative_floor=1.0e-6,
    )

    states = _prepared_original_state_batch(bucket, problems, config)

    epsilon = jnp.full((2,), math.log(0.1))
    assert states.rho == pytest.approx(
        epsilon[:, None] - bucket.ln_mk_init
    )
    assert states.epsilon == pytest.approx(epsilon)


def test_prepared_exact_state_policy_requires_complete_state():
    bucket = _bucket()
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(math.log(0.1),),
            initial_state_policy="provided",
        )
    )
    problems = _prepared_problem_batch(
        bucket,
        jnp.asarray([[1.0]]),
        budget_relative_floor=1.0e-6,
    )

    with pytest.raises(ValueError, match="requires prepared rho"):
        _prepared_original_state_batch(bucket, problems, config)


def test_profile_buckets_must_cover_every_layer_exactly_once():
    bucket = _bucket()
    incomplete = _PreparedBucket(
        **{
            **bucket.__dict__,
            "layer_indices": (0,),
            "ln_nk_init": bucket.ln_nk_init[:1],
            "ln_mk_init": bucket.ln_mk_init[:1],
            "ln_ntot_init": bucket.ln_ntot_init[:1],
            "element_potential_init": bucket.element_potential_init[:1],
            "element_inventory_target": bucket.element_inventory_target[:1],
            "hvector": bucket.hvector[:1],
            "hvector_cond_active": bucket.hvector_cond_active[:1],
            "ln_normalized_pressure": bucket.ln_normalized_pressure[:1],
        }
    )

    with pytest.raises(ValueError, match="cover every profile layer"):
        run_fixed_support_profile(
            buckets=(incomplete,),
            formula_matrix=jnp.asarray([[1.0]]),
            layer_count=2,
            condensate_count=2,
            config=_config(),
        )


def test_public_v2_route_reuses_the_existing_prepared_plan_without_promotion():
    condensate_setup = SimpleNamespace(
        hvector_func=lambda temperatures: jnp.broadcast_to(
            jnp.asarray([0.5, 2.0]),
            (temperatures.shape[0], 2),
        )
    )
    setup = SimpleNamespace(
        condensate_setup=condensate_setup,
        formula_matrix_cond=jnp.asarray([[1.0, 1.0]]),
    )
    plan = ExperimentalCondensateProfileFixedSupportBatchPlan(
        setup=setup,
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        max_iter=4,
        n_layers=2,
        condensate_count=2,
        bucket_layer_index_arrays=(jnp.asarray([0, 1]),),
        temperatures=jnp.asarray([1000.0, 1100.0]),
    )

    result = run_experimental_profile_fixed_support_v2_batch_plan(
        plan,
        config=_config(),
        element_inventory_target=jnp.asarray([1.0]),
    )

    assert jnp.all(result["converged_with_support_closure"])
    assert result["production_preset_promoted"] is False


def test_public_v2_defaults_enable_restoration_and_pass_temperature_validity(
    monkeypatch,
):
    captured = {}

    def fake_run_fixed_support_profile(**kwargs):
        captured["fixed_support"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.fixed_support.batch."
            "run_fixed_support_profile"
        ),
        fake_run_fixed_support_profile,
    )
    monkeypatch.setattr(
        (
            "exogibbs.equilibrium.condensate.lifecycle."
            "evaluate_profile_support_closure"
        ),
        lambda result, **kwargs: captured.setdefault("closure", kwargs)
        and result,
    )
    condensate_setup = SimpleNamespace(
        hvector_func=lambda temperatures: jnp.broadcast_to(
            jnp.asarray([0.5, 2.0]),
            (temperatures.shape[0], 2),
        ),
        metadata={"temperature_validity_upper": jnp.asarray([2000.0, 500.0])},
    )
    setup = SimpleNamespace(
        condensate_setup=condensate_setup,
        formula_matrix_cond=jnp.asarray([[1.0, 1.0]]),
    )
    plan = ExperimentalCondensateProfileFixedSupportBatchPlan(
        setup=setup,
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        max_iter=7,
        n_layers=2,
        condensate_count=2,
        bucket_layer_index_arrays=(jnp.asarray([0, 1]),),
        temperatures=jnp.asarray([1000.0, 1100.0]),
    )

    result = run_experimental_profile_fixed_support_v2_batch_plan(plan)

    assert result == {"ok": True}
    assert (
        captured["fixed_support"]["config"].limits.max_normal_iterations == 7
    )
    assert (
        captured["fixed_support"]["config"].limits.max_restoration_calls == 2
    )
    assert jnp.array_equal(
        captured["closure"]["condensate_valid_mask"],
        jnp.asarray([[True, False], [True, False]]),
    )
