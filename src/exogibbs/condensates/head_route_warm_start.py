"""Warm-start candidate generation for the condensate HEAD route."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp

from exogibbs.api.chemistry import ThermoState
from exogibbs.condensates.depleted_gas_refresh import (
    DepletedGasRefreshReport,
    build_depleted_gas_refresh_init,
)
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit


@dataclass(frozen=True)
class CondensateHeadRouteWarmStartCandidate:
    """One restricted-solver initialization candidate for HEAD route calls."""

    candidate_name: str
    candidate_kind: str
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    initial_log_state_override: CondensateEquilibriumInit | None
    refresh_report: DepletedGasRefreshReport | None
    finite_solver_inputs: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        init = self.initial_log_state_override
        payload["initial_log_state_override"] = None if init is None else {
            "ln_nk_shape": None if init.ln_nk is None else list(jnp.asarray(init.ln_nk).shape),
            "ln_mk_shape": None if init.ln_mk is None else list(jnp.asarray(init.ln_mk).shape),
            "ln_ntot_shape": None if init.ln_ntot is None else list(jnp.asarray(init.ln_ntot).shape),
            "ln_nk_source_trace": init.ln_nk_source_trace,
        }
        payload["refresh_report"] = (
            None if self.refresh_report is None else self.refresh_report.as_dict()
        )
        return payload


@dataclass(frozen=True)
class CondensateHeadRouteWarmStartReport:
    """Candidate-generation report for HEAD route restricted solver attempts."""

    report_schema: str
    explicit_opt_in: bool
    candidate_count: int
    candidates: tuple[CondensateHeadRouteWarmStartCandidate, ...]
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = tuple(candidate.as_dict() for candidate in self.candidates)
        return payload


def _finite_candidate_inputs(
    *,
    support_amounts_init: Sequence[float],
    initial_log_state_override: CondensateEquilibriumInit | None,
) -> bool:
    amounts = jnp.asarray(support_amounts_init, dtype=jnp.float64)
    finite = bool(
        amounts.ndim == 1
        and jnp.all(jnp.isfinite(amounts))
        and jnp.all(amounts > 0.0)
    )
    if initial_log_state_override is None:
        return finite
    for value in (
        initial_log_state_override.ln_nk,
        initial_log_state_override.ln_mk,
        initial_log_state_override.ln_ntot,
    ):
        if value is None:
            return False
        array = jnp.asarray(value, dtype=jnp.float64)
        finite = finite and bool(jnp.all(jnp.isfinite(array)))
    return finite


def build_condensate_head_route_warm_start_report(
    *,
    explicit_opt_in: bool,
    state: ThermoState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    hvector_func: Callable[[Any], Any],
    support_indices: Sequence[int],
    support_amounts_init: Sequence[float],
    enable_depleted_gas_refresh: bool = True,
    gas_refresh_policy: str = "native_gas_solver",
    field_provenance: Mapping[str, str] | None = None,
) -> CondensateHeadRouteWarmStartReport:
    """Build baseline and refresh warm-start candidates from native inputs."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for HEAD route warm-start candidates.")
    indices = tuple(int(index) for index in support_indices)
    amounts = tuple(float(value) for value in jnp.asarray(support_amounts_init, dtype=jnp.float64).tolist())
    if len(indices) != len(amounts):
        raise ValueError("support_indices and support_amounts_init must have matching length.")
    if len(indices) != len(set(indices)):
        raise ValueError("support_indices must not contain duplicate indices.")
    baseline = CondensateHeadRouteWarmStartCandidate(
        candidate_name="baseline_positive_support_seed",
        candidate_kind="baseline",
        support_indices=indices,
        support_amounts_init=amounts,
        initial_log_state_override=None,
        refresh_report=None,
        finite_solver_inputs=_finite_candidate_inputs(
            support_amounts_init=amounts,
            initial_log_state_override=None,
        ),
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )
    candidates: list[CondensateHeadRouteWarmStartCandidate] = [baseline]
    if enable_depleted_gas_refresh and indices:
        ln_mk = jnp.log(jnp.maximum(jnp.asarray(amounts, dtype=jnp.float64), 1.0e-300))
        refresh_init, refresh_report = build_depleted_gas_refresh_init(
            explicit_opt_in=True,
            state=state,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            support_indices=indices,
            ln_mk=ln_mk,
            gas_refresh_policy=gas_refresh_policy,
            field_provenance=field_provenance,
        )
        candidates.append(
            CondensateHeadRouteWarmStartCandidate(
                candidate_name=f"depleted_gas_refresh_{gas_refresh_policy}",
                candidate_kind="depleted_gas_refresh",
                support_indices=indices,
                support_amounts_init=amounts,
                initial_log_state_override=refresh_init,
                refresh_report=refresh_report,
                finite_solver_inputs=_finite_candidate_inputs(
                    support_amounts_init=amounts,
                    initial_log_state_override=refresh_init,
                ),
                production_behavior_change=False,
                production_return_signature_change=False,
                preset_default_wiring_change=False,
                fastchem4_trace_public_runtime_constructor_inputs_used=False,
            )
        )
    return CondensateHeadRouteWarmStartReport(
        report_schema="exogibbs_condensate_head_route_warm_start_report_v1",
        explicit_opt_in=True,
        candidate_count=len(candidates),
        candidates=tuple(candidates),
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "CondensateHeadRouteWarmStartCandidate",
    "CondensateHeadRouteWarmStartReport",
    "build_condensate_head_route_warm_start_report",
)
