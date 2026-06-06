"""Explicit diagnostic adapter from positive-support plans to solver inputs.

This module does not call production solvers and does not wire any default
behavior. It only materializes explicit restricted-support input arrays from a
validated positive-support initialization plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from exogibbs.condensates.positive_support_plan import (
    PositiveSupportInitializationPlan,
)


@dataclass(frozen=True)
class PositiveSupportRestrictedSolverInputs:
    """Default-off diagnostic restricted-support solver input record."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    adapter_schema: str
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    support_names: tuple[str, ...]
    empty_positive_support: bool
    empty_support_policy: str
    source_plan_schema: str
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_restricted_support_solver_inputs_from_plan(
    plan: PositiveSupportInitializationPlan,
    *,
    allow_empty_support: bool = False,
) -> PositiveSupportRestrictedSolverInputs:
    """Convert a positive-support plan into explicit restricted-solver inputs."""

    if not plan.diagnostic_only or not plan.default_off:
        raise ValueError("positive-support plan must be diagnostic-only and default-off.")
    if plan.production_behavior_change:
        raise ValueError("positive-support plan must not imply production behavior changes.")
    if plan.fastchem4_trace_values_used:
        raise ValueError("FastChem4 trace values are not valid solver constructor inputs.")
    if plan.fastchem4_public_values_used_as_constructor_inputs:
        raise ValueError("FastChem4 public values are not valid solver constructor inputs.")
    if len(plan.positive_support_indices) != len(plan.recommended_amounts):
        raise ValueError("positive support indices and recommended amounts must have matching length.")
    if len(plan.positive_support_indices) != len(plan.positive_support_names):
        raise ValueError("positive support indices and names must have matching length.")
    if any(amount <= 0.0 for amount in plan.recommended_amounts):
        raise ValueError("recommended solver support amounts must be positive.")

    if not plan.positive_support_indices:
        if not allow_empty_support:
            raise ValueError("empty positive support requires explicit allow_empty_support=True.")
        return PositiveSupportRestrictedSolverInputs(
            diagnostic_only=True,
            default_off=True,
            production_behavior_change=False,
            adapter_schema="exogibbs_condensate_positive_support_solver_inputs_v1",
            support_indices=(),
            support_amounts_init=(),
            support_names=(),
            empty_positive_support=True,
            empty_support_policy="skip_restricted_support_solver_or_use_gas_only_boundary",
            source_plan_schema=plan.plan_schema,
            fastchem4_trace_values_used=False,
            fastchem4_public_values_used_as_constructor_inputs=False,
        )

    return PositiveSupportRestrictedSolverInputs(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        adapter_schema="exogibbs_condensate_positive_support_solver_inputs_v1",
        support_indices=tuple(int(index) for index in plan.positive_support_indices),
        support_amounts_init=tuple(float(amount) for amount in plan.recommended_amounts),
        support_names=tuple(str(name) for name in plan.positive_support_names),
        empty_positive_support=False,
        empty_support_policy="not_empty",
        source_plan_schema=plan.plan_schema,
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )
