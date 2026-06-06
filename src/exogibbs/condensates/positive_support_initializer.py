"""Explicit experimental positive-support initializer diagnostics.

This module combines positive-support planning and restricted-solver input
materialization behind one explicit import. It does not call solvers and does
not connect to production defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from exogibbs.condensates.positive_support_plan import (
    PositiveSupportInitializationPlan,
    build_positive_support_initialization_plan,
)
from exogibbs.condensates.positive_support_solver_adapter import (
    PositiveSupportRestrictedSolverInputs,
    build_restricted_support_solver_inputs_from_plan,
)


@dataclass(frozen=True)
class PositiveSupportInitializerReport:
    """Explicit experimental initializer report for condensate diagnostics."""

    diagnostic_only: bool
    experimental: bool
    default_off: bool
    production_behavior_change: bool
    interface_schema: str
    plan: PositiveSupportInitializationPlan
    solver_inputs: PositiveSupportRestrictedSolverInputs
    production_wiring_allowed_now: bool
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = self.plan.as_dict()
        payload["solver_inputs"] = self.solver_inputs.as_dict()
        return payload


def build_positive_support_initializer_report(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    condensate_species_order: Sequence[str],
    hvector_cond: Sequence[float],
    max_positive_support_count: int = 1,
    seed_fraction: float = 1.0e-3,
    max_seed_amount: float = 1.0e-3,
    min_seed_amount: float = 1.0e-300,
    allow_empty_positive_support: bool = True,
    field_provenance: Mapping[str, str] | None = None,
) -> PositiveSupportInitializerReport:
    """Build an explicit diagnostic initializer report from native arrays."""

    plan = build_positive_support_initialization_plan(
        formula_matrix_cond=formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=condensate_species_order,
        hvector_cond=hvector_cond,
        max_positive_support_count=max_positive_support_count,
        seed_fraction=seed_fraction,
        max_seed_amount=max_seed_amount,
        min_seed_amount=min_seed_amount,
        allow_empty_positive_support=allow_empty_positive_support,
        field_provenance=field_provenance,
    )
    solver_inputs = build_restricted_support_solver_inputs_from_plan(
        plan,
        allow_empty_support=allow_empty_positive_support,
    )
    return PositiveSupportInitializerReport(
        diagnostic_only=True,
        experimental=True,
        default_off=True,
        production_behavior_change=False,
        interface_schema="exogibbs_condensate_positive_support_initializer_v1",
        plan=plan,
        solver_inputs=solver_inputs,
        production_wiring_allowed_now=False,
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )
