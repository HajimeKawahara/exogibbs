"""Compatibility facade for the fixed-support profile adapter."""

from typing import Any, Sequence

from exogibbs.equilibrium.condensate import support as _support
from exogibbs.equilibrium.condensate.fixed_support import batch as _batch
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FixedSupportV2Config,
)


for _name, _value in vars(_batch).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def run_prepared_profile_v2(
    *,
    buckets: Sequence[Any],
    formula_matrix,
    formula_matrix_cond_full,
    condensate_standard_source_full,
    condensate_valid_mask=None,
    layer_count: int,
    condensate_count: int,
    config: FixedSupportV2Config = FixedSupportV2Config(),
    budget_relative_floor: float = 1.0e-6,
    support_closure_tolerance: float = 1.0e-8,
    include_terminal_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run the historical fixed-support-plus-closure adapter."""

    fixed_support_result = _batch.run_fixed_support_profile(
        buckets=buckets,
        formula_matrix=formula_matrix,
        layer_count=layer_count,
        condensate_count=condensate_count,
        config=config,
        budget_relative_floor=budget_relative_floor,
        include_terminal_diagnostics=include_terminal_diagnostics,
    )
    return _support.evaluate_profile_support_closure(
        fixed_support_result,
        formula_matrix=formula_matrix,
        formula_matrix_cond_full=formula_matrix_cond_full,
        condensate_standard_source_full=condensate_standard_source_full,
        condensate_valid_mask=condensate_valid_mask,
        budget_relative_floor=budget_relative_floor,
        support_closure_tolerance=support_closure_tolerance,
    )


__all__ = tuple(
    name
    for name in (*vars(_batch), "run_prepared_profile_v2")
    if not name.startswith("_")
)
