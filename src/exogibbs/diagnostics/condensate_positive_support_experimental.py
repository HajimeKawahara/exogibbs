"""Public experimental positive-support initialization boundary.

This module exposes an explicit opt-in diagnostics boundary for injecting
positive condensate support seeds into the restricted support condensate solver
callsite. It is a release-candidate experimental surface, not a production
initialization path.

The API contract is intentionally narrow:

* callers must pass ``enable_experimental_positive_support=True``;
* ``seed_fraction`` and ``max_seed_amount`` are capped at ``1.0e-3``;
* callers must provide ExoGibbs-native arrays and thermochemistry functions;
* FastChem4 public, runtime, trace, branch replay, and reference-fit values are
  rejected by the provenance firewall;
* KKT residuals are reported only as solver-stage diagnostics;
* no production solver defaults, presets, or return signatures are changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from exogibbs.api.chemistry import ThermoState
from exogibbs.diagnostics.condensate_positive_support_callsite_experiment import (
    PositiveSupportCallsiteExperimentResult,
    run_explicit_positive_support_callsite_experiment,
)


EXPERIMENTAL_POSITIVE_SUPPORT_API_SCHEMA = (
    "exogibbs_condensate_positive_support_experimental_api_v1"
)
MAX_SAFE_SEED_FRACTION = 1.0e-3
MAX_SAFE_SEED_AMOUNT = 1.0e-3


@dataclass(frozen=True)
class PositiveSupportExperimentalConfig:
    """Configuration for the explicit opt-in positive-support experiment.

    Parameters
    ----------
    enable_experimental_positive_support
        Required opt-in flag. The public boundary rejects calls unless this is
        ``True``.
    max_positive_support_count
        Maximum number of condensates to seed in the positive support set. The
        current release-candidate policy uses ``1`` as the top1 experimental
        candidate.
    seed_fraction, max_seed_amount
        Safe seed envelope controls. Both must stay within ``1.0e-3``.
    allow_empty_positive_support
        If ``True``, cases with no positive support are reported as an empty
        boundary and the restricted solver call is skipped.

    Notes
    -----
    This configuration is diagnostic-only and default-off. It is not connected
    to production presets or normal equilibrium entry points.
    """

    enable_experimental_positive_support: bool = False
    max_positive_support_count: int = 1
    seed_fraction: float = MAX_SAFE_SEED_FRACTION
    max_seed_amount: float = MAX_SAFE_SEED_AMOUNT
    allow_empty_positive_support: bool = True
    gas_epsilon_crit: float = 1.0e-12
    gas_max_iter: int = 30
    epsilon: float = -5.0
    max_iter: int = 100
    api_schema: str = EXPERIMENTAL_POSITIVE_SUPPORT_API_SCHEMA
    diagnostic_only: bool = True
    experimental: bool = True
    default_off: bool = True
    production_behavior_change: bool = False
    production_wiring_allowed_now: bool = False

    def validate(self) -> None:
        """Validate the opt-in flag and safe experimental seed envelope."""

        if not self.enable_experimental_positive_support:
            raise ValueError("enable_experimental_positive_support=True is required.")
        if self.max_positive_support_count < 0:
            raise ValueError("max_positive_support_count must be nonnegative.")
        if self.seed_fraction < 0.0:
            raise ValueError("seed_fraction must be nonnegative.")
        if self.max_seed_amount < 0.0:
            raise ValueError("max_seed_amount must be nonnegative.")
        if self.seed_fraction > MAX_SAFE_SEED_FRACTION:
            raise ValueError("seed_fraction exceeds the safe experimental envelope.")
        if self.max_seed_amount > MAX_SAFE_SEED_AMOUNT:
            raise ValueError("max_seed_amount exceeds the safe experimental envelope.")
        if self.gas_max_iter < 0:
            raise ValueError("gas_max_iter must be nonnegative.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be nonnegative.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_positive_support_experimental_callsite(
    *,
    config: PositiveSupportExperimentalConfig,
    state: ThermoState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    hvector_func: Callable[[Any], Any],
    hvector_cond_func: Callable[[Any], Any],
    condensate_species_order: Sequence[str],
    element_order: Sequence[str],
    baseline_result: Optional[Mapping[str, Any]] = None,
    field_provenance: Mapping[str, str] | None = None,
) -> PositiveSupportCallsiteExperimentResult:
    """Run the public experimental positive-support callsite boundary.

    This function validates the explicit opt-in config, builds positive support
    solver inputs from native arrays, and forwards those inputs to the
    restricted support condensate solver callsite. The returned report includes
    support shape checks, pre-solver budget use, solver success/failure fields,
    post-solver budget residuals, and KKT diagnostics.

    The function never imports or calls FastChem4 and does not use FastChem4
    public, runtime, or trace values as constructor inputs.
    """

    config.validate()
    return run_explicit_positive_support_callsite_experiment(
        state=state,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        condensate_species_order=condensate_species_order,
        element_order=element_order,
        max_positive_support_count=config.max_positive_support_count,
        seed_fraction=config.seed_fraction,
        max_seed_amount=config.max_seed_amount,
        allow_empty_positive_support=config.allow_empty_positive_support,
        gas_epsilon_crit=config.gas_epsilon_crit,
        gas_max_iter=config.gas_max_iter,
        epsilon=config.epsilon,
        max_iter=config.max_iter,
        baseline_result=baseline_result,
        field_provenance=field_provenance,
    )


__all__ = (
    "EXPERIMENTAL_POSITIVE_SUPPORT_API_SCHEMA",
    "MAX_SAFE_SEED_AMOUNT",
    "MAX_SAFE_SEED_FRACTION",
    "PositiveSupportExperimentalConfig",
    "run_positive_support_experimental_callsite",
)
