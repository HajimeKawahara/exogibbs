"""Production policy and option validation for condensate equilibrium."""

from dataclasses import dataclass
import math
from typing import Optional

from exogibbs.equilibrium.condensate.fixed_support.types import (
    ContinuationConfig,
    FixedSupportV2Config,
    SolverLimitConfig,
)
from exogibbs.equilibrium.condensate.types import (
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
    HEAD_ROUTE_V2,
    CondensateEquilibriumOptions,
    CondensateProfileMethod,
)


@dataclass(frozen=True)
class FixedSupportV2ProductionPolicy:
    """Immutable numerical and lifecycle policy validated for production."""

    name: str
    solver_config: FixedSupportV2Config
    budget_relative_floor: float = 1.0e-6
    support_closure_tolerance: float = 1.0e-8
    # This bounded precondition only authorizes an exact zero-barrier solve.
    # Final physical acceptance continues to use the solver's 1e-8 KKT gates.
    zero_barrier_initializer_gas_stationarity_tolerance: float = 1.0e-5
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
    rainout_gauge_maximum_total: float = 3.0e8
    rainout_trace_capacity_relative_tolerance: float = 1.0e-18
    rainout_trace_condensate_stationarity_tolerance: float = 1.0e-5
    rainout_allow_trace_capacity_acceptance: bool = False
    rainout_depletion_roundoff_multiplier: float = 64.0


def fixed_support_v2_production_policy(
    name: str = FIXED_SUPPORT_V2_VALIDATED_PRESET,
) -> FixedSupportV2ProductionPolicy:
    """Return the named fixed-support production policy."""

    if name != FIXED_SUPPORT_V2_VALIDATED_PRESET:
        raise ValueError(
            f"Unknown fixed-support v2 preset {name!r}. "
            f"Expected {FIXED_SUPPORT_V2_VALIDATED_PRESET!r}."
        )
    return FixedSupportV2ProductionPolicy(
        name=name,
        solver_config=FixedSupportV2Config(
            continuation=ContinuationConfig(
                # These are log barriers after the lifecycle normalizes the
                # positive non-charge element inventory to unit total.
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


def validate_condensate_options(
    options: CondensateEquilibriumOptions,
    *,
    profile_method: Optional[CondensateProfileMethod] = None,
) -> None:
    """Validate the compact production-v2 option contract."""

    if options.route != HEAD_ROUTE_V2:
        raise ValueError(
            f"Unsupported condensate route {options.route!r}; "
            f"expected {HEAD_ROUTE_V2!r}."
        )
    if options.fixed_support_v2_preset != FIXED_SUPPORT_V2_VALIDATED_PRESET:
        raise ValueError(
            "fixed_support_v2_preset must be "
            f"{FIXED_SUPPORT_V2_VALIDATED_PRESET!r}."
        )
    effective_profile_method = (
        profile_method if profile_method is not None else options.profile_method
    )
    if not isinstance(options.rainout, bool):
        raise TypeError("rainout must be a bool.")
    if options.rainout and effective_profile_method not in {
        None,
        "auto",
        "scan_hot_from_bottom",
    }:
        raise ValueError(
            "rainout=True requires profile method 'auto' or "
            "'scan_hot_from_bottom'."
        )
    if (
        not options.rainout
        and effective_profile_method not in {None, "auto", "vmap_cold"}
    ):
        raise ValueError(
            "head_v2 currently supports profile method 'auto' or "
            "'vmap_cold' when rainout=False. "
            "'scan_hot_from_bottom' is reserved for rainout=True."
        )
    if not isinstance(options.return_diagnostics, bool):
        raise TypeError("return_diagnostics must be a bool.")
    if not isinstance(
        options.enable_full_condensate_budget_residual_gate,
        bool,
    ):
        raise TypeError(
            "enable_full_condensate_budget_residual_gate must be a bool."
        )
    if not math.isfinite(
        float(options.full_condensate_budget_relative_tolerance)
    ) or float(options.full_condensate_budget_relative_tolerance) <= 0.0:
        raise ValueError(
            "full_condensate_budget_relative_tolerance must be finite and "
            "positive."
        )
    if not math.isfinite(
        float(options.full_condensate_budget_relative_floor)
    ) or float(options.full_condensate_budget_relative_floor) <= 0.0:
        raise ValueError(
            "full_condensate_budget_relative_floor must be finite and "
            "positive."
        )


__all__ = (
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "FixedSupportV2ProductionPolicy",
    "fixed_support_v2_production_policy",
    "validate_condensate_options",
)
