from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from exogibbs.api.condensate_equilibrium import (
    ExperimentalCondensateProfileFixedSupportBatchPlan,
    run_experimental_profile_fixed_support_v2_batch_plan,
)
from exogibbs.optimize.fixed_support_v2.types import (
    ContinuationConfig,
    FixedSupportV2Config,
    SOCConfig,
    SolverLimitConfig,
    TerminalStatus,
)
from exogibbs.optimize.fixed_support_v2_profile import (
    _compiled_solver_factory,
    _prepared_original_state_batch,
    _prepared_problem_batch,
    run_prepared_profile_v2,
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
    gas_stationarity_source_init: jax.Array | None
    element_inventory_target: jax.Array
    hvector: jax.Array
    hvector_cond_active: jax.Array
    ln_normalized_pressure: jax.Array


def _bucket(*, legacy_source=False):
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
        gas_stationarity_source_init=(
            -qtot[:, None] if legacy_source else None
        ),
        element_inventory_target=jnp.ones((2, 1)),
        hvector=jnp.zeros((2, 1)),
        hvector_cond_active=jnp.full((2, 1), 0.5),
        ln_normalized_pressure=jnp.zeros((2,)),
    )


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


def _run_profile(inactive_source):
    return run_prepared_profile_v2(
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        formula_matrix_cond_full=jnp.asarray([[1.0, 1.0]]),
        condensate_standard_source_full=jnp.asarray(
            [[0.5, inactive_source], [0.5, inactive_source]]
        ),
        layer_count=2,
        condensate_count=2,
        config=_config(),
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


def test_support_closure_ignores_temperature_invalid_condensates():
    result = run_prepared_profile_v2(
        buckets=(_bucket(),),
        formula_matrix=jnp.asarray([[1.0]]),
        formula_matrix_cond_full=jnp.asarray([[1.0, 1.0]]),
        condensate_standard_source_full=jnp.asarray(
            [[0.5, -1.0], [0.5, -1.0]]
        ),
        condensate_valid_mask=jnp.asarray(
            [[True, False], [True, False]]
        ),
        layer_count=2,
        condensate_count=2,
        config=_config(),
    )

    assert jnp.all(result["fixed_support_converged"])
    assert jnp.all(result["support_closed"])
    assert not jnp.any(result["support_expansion_mask"][:, 1])


def test_legacy_gas_source_is_converted_to_iterate_independent_gamma():
    problems = _prepared_problem_batch(
        _bucket(legacy_source=True),
        jnp.asarray([[1.0]]),
        budget_relative_floor=1.0e-6,
    )

    assert problems.gamma == pytest.approx(jnp.zeros((2, 1)))


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
        run_prepared_profile_v2(
            buckets=(incomplete,),
            formula_matrix=jnp.asarray([[1.0]]),
            formula_matrix_cond_full=jnp.asarray([[1.0, 1.0]]),
            condensate_standard_source_full=jnp.asarray(
                [[0.5, 2.0], [0.5, 2.0]]
            ),
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

    def fake_run_prepared_profile_v2(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "exogibbs.optimize.fixed_support_v2_profile.run_prepared_profile_v2",
        fake_run_prepared_profile_v2,
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
    assert captured["config"].limits.max_normal_iterations == 7
    assert captured["config"].limits.max_restoration_calls == 2
    assert jnp.array_equal(
        captured["condensate_valid_mask"],
        jnp.asarray([[True, False], [True, False]]),
    )
