from exogibbs.condensates.fixed_support_v2_policy import (
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
    fixed_support_v2_production_policy,
)


def test_validated_v2_production_policy_matches_archived_gpu_configuration():
    policy = fixed_support_v2_production_policy()
    config = policy.solver_config

    assert policy.name == FIXED_SUPPORT_V2_VALIDATED_PRESET
    assert config.continuation.epsilon_schedule == (-11.0, -13.0, -15.0, -17.0)
    assert config.continuation.initial_state_policy == "center"
    assert config.limits.max_normal_iterations == 1000
    assert config.limits.max_line_search_trials == 20
    assert config.limits.max_restoration_calls == 2
    assert config.limits.max_restoration_iterations == 100
    assert config.limits.max_restoration_line_search_trials == 20
    assert policy.initial_support_topk == 8
    assert policy.initial_support_limit == 16
    assert policy.support_add_per_round == 8
    assert policy.support_limit == 128
    assert policy.lifecycle_max_rounds == 15
    assert policy.runtime_budget_name == "a100_40gb_2026_07"
    assert policy.max_cold_compilation_seconds == 900.0
    assert policy.max_cold_wall_seconds == 960.0
    assert policy.max_warm_execution_seconds == 20.0
    assert policy.max_warm_wall_seconds == 25.0
