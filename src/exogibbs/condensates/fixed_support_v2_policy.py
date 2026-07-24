"""Production policy for the fixed-support v2 solver.

The policy in this module is intentionally immutable.  Production-facing
callers select the named preset instead of assembling numerical solver
configuration piecemeal.
"""

from __future__ import annotations

from dataclasses import dataclass

from exogibbs.optimize.fixed_support_v2.types import (
    ContinuationConfig,
    FixedSupportV2Config,
    SolverLimitConfig,
)


FIXED_SUPPORT_V2_VALIDATED_PRESET = "validated_2026_07"


@dataclass(frozen=True)
class FixedSupportV2ProductionPolicy:
    """Solver and outer-lifecycle policy validated for production migration."""

    name: str
    solver_config: FixedSupportV2Config
    budget_relative_floor: float = 1.0e-6
    support_closure_tolerance: float = 1.0e-8
    initial_support_topk: int = 8
    initial_support_limit: int = 16
    support_add_per_round: int = 8
    support_limit: int = 128
    lifecycle_max_rounds: int = 15
    seed_fraction: float = 0.8
    max_seed_amount: float = 1.0
    min_seed_amount: float = 1.0e-300
    runtime_budget_name: str = "a100_40gb_2026_07"
    max_cold_compilation_seconds: float = 900.0
    max_cold_wall_seconds: float = 960.0
    max_warm_execution_seconds: float = 20.0
    max_warm_wall_seconds: float = 25.0


def fixed_support_v2_production_policy(
    name: str = FIXED_SUPPORT_V2_VALIDATED_PRESET,
) -> FixedSupportV2ProductionPolicy:
    """Return the immutable fixed-support v2 production policy."""

    if name != FIXED_SUPPORT_V2_VALIDATED_PRESET:
        raise ValueError(
            f"Unknown fixed-support v2 preset {name!r}. "
            f"Expected {FIXED_SUPPORT_V2_VALIDATED_PRESET!r}."
        )
    return FixedSupportV2ProductionPolicy(
        name=name,
        solver_config=FixedSupportV2Config(
            continuation=ContinuationConfig(
                epsilon_schedule=(-11.0, -13.0, -15.0, -17.0),
                initial_state_policy="center",
            ),
            limits=SolverLimitConfig(
                max_normal_iterations=1000,
                max_line_search_trials=20,
                max_restoration_calls=2,
                max_restoration_iterations=100,
                max_restoration_line_search_trials=20,
            ),
        ),
    )


__all__ = [
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "FixedSupportV2ProductionPolicy",
    "fixed_support_v2_production_policy",
]
