"""Production-facing condensate equilibrium API shell.

This module defines the first condensate-specific public API surface. It keeps
gas-only equilibrium behavior separate and routes condensate-enabled calls
through the HEAD route v1 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.condensates.head_route_standard_gate import (
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
    HEAD_ROUTE_STANDARD,
    NOT_CONVERGED,
    classify_head_route_standard_gate_row,
)


Array = jax.Array
CondensateRoute = Literal["head_v1"]
CondensateResidualPolicy = Literal["head_route_tiers_v1"]
CondensateWarmStartGasRefreshPolicy = Literal["native_gas_solver"]


@dataclass(frozen=True)
class CondensateChemicalSetup:
    """Gas and condensate thermochemistry bundle for condensate equilibrium."""

    gas_setup: ChemicalSetup
    condensate_setup: ChemicalSetup
    formula_matrix: Array
    formula_matrix_cond: Array
    gas_species: tuple[str, ...]
    condensate_species: tuple[str, ...]
    elements: tuple[str, ...]


@dataclass(frozen=True)
class CondensateEquilibriumOptions:
    """Options for the condensate HEAD route standard path."""

    route: CondensateRoute = HEAD_ROUTE_STANDARD
    case_id: Optional[str] = None
    allow_caveat_tiers: bool = True
    return_diagnostics: bool = False
    max_outer_iterations: Optional[int] = None
    max_inner_iterations: Optional[int] = None
    residual_policy: CondensateResidualPolicy = "head_route_tiers_v1"
    metric_status: Optional[str] = None
    selected_route: str = "head_v1_restricted_support"
    max_positive_support_count: int = 1
    seed_fraction: float = 1.0e-3
    max_seed_amount: float = 1.0e-3
    min_seed_amount: float = 1.0e-300
    allow_empty_positive_support: bool = True
    enable_head_route_warm_start: bool = True
    enable_depleted_gas_refresh: bool = True
    warm_start_gas_refresh_policy: CondensateWarmStartGasRefreshPolicy = "native_gas_solver"
    head_route_primary_summary: Optional[Mapping[str, Any]] = None
    head_route_refresh_policy_summary: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Result container for the condensate equilibrium standard path."""

    gas_ln_n: Array
    gas_n: Array
    gas_x: Array
    gas_ntot: Array
    condensate_amounts: Array
    condensate_support_indices: Array
    condensate_support_names: tuple[str, ...]
    acceptance_tier: str
    selected_route: str
    status: str
    converged: bool
    diagnostics: Optional[Mapping[str, Any]] = None


def validate_condensate_chemical_setup(setup: CondensateChemicalSetup) -> None:
    """Validate gas-condensate setup compatibility for HEAD route calls."""

    if not isinstance(setup.gas_setup, ChemicalSetup):
        raise TypeError("gas_setup must be a ChemicalSetup.")
    if not isinstance(setup.condensate_setup, ChemicalSetup):
        raise TypeError("condensate_setup must be a ChemicalSetup.")
    if setup.gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if setup.condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if tuple(setup.gas_setup.elements) != tuple(setup.condensate_setup.elements):
        raise ValueError("gas and condensate element orders must match.")
    formula_matrix = jnp.asarray(setup.formula_matrix)
    formula_matrix_cond = jnp.asarray(setup.formula_matrix_cond)
    if formula_matrix.ndim != 2:
        raise ValueError("formula_matrix must be a two-dimensional array.")
    if formula_matrix_cond.ndim != 2:
        raise ValueError("formula_matrix_cond must be a two-dimensional array.")
    if formula_matrix.shape[0] != formula_matrix_cond.shape[0]:
        raise ValueError("gas and condensate formula matrices must have the same element count.")
    if formula_matrix.shape[0] != len(setup.elements):
        raise ValueError("elements length must match formula matrix rows.")
    if formula_matrix.shape[1] != len(setup.gas_species):
        raise ValueError("gas_species length must match formula_matrix columns.")
    if formula_matrix_cond.shape[1] != len(setup.condensate_species):
        raise ValueError("condensate_species length must match formula_matrix_cond columns.")


def build_condensate_chemical_setup(
    *,
    gas_setup: ChemicalSetup,
    condensate_setup: ChemicalSetup,
) -> CondensateChemicalSetup:
    """Build and validate a gas-condensate chemical setup bundle."""

    if gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if gas_setup.species is None:
        raise ValueError("gas_setup.species is required for condensate equilibrium.")
    if condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if condensate_setup.species is None:
        raise ValueError("condensate_setup.species is required for condensate equilibrium.")
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix),
        formula_matrix_cond=jnp.asarray(condensate_setup.formula_matrix),
        gas_species=tuple(gas_setup.species),
        condensate_species=tuple(condensate_setup.species),
        elements=tuple(gas_setup.elements),
    )
    validate_condensate_chemical_setup(setup)
    return setup


def _ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    return jnp.log(jnp.asarray(pressure) / jnp.asarray(reference_pressure))


def _full_condensate_amounts(
    *,
    support_indices: Sequence[int],
    support_amounts: Array,
    condensate_count: int,
) -> Array:
    indices = jnp.asarray(support_indices, dtype=jnp.int32)
    amounts = jnp.asarray(support_amounts)
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= condensate_count)):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    return jnp.zeros((condensate_count,), dtype=amounts.dtype).at[indices].set(amounts)


def _validate_options(options: CondensateEquilibriumOptions) -> None:
    if options.route != HEAD_ROUTE_STANDARD:
        raise ValueError(f"Unsupported condensate route '{options.route}'. Expected '{HEAD_ROUTE_STANDARD}'.")
    if options.residual_policy != "head_route_tiers_v1":
        raise ValueError("Only residual_policy='head_route_tiers_v1' is supported.")
    if options.max_positive_support_count <= 0:
        raise ValueError("max_positive_support_count must be positive.")
    if options.seed_fraction <= 0.0:
        raise ValueError("seed_fraction must be positive.")
    if options.max_seed_amount <= 0.0:
        raise ValueError("max_seed_amount must be positive.")
    if options.min_seed_amount <= 0.0:
        raise ValueError("min_seed_amount must be positive.")
    if options.warm_start_gas_refresh_policy != "native_gas_solver":
        raise ValueError("Only warm_start_gas_refresh_policy='native_gas_solver' is supported.")


def _least_squares_element_potential(
    *,
    formula_matrix: Array,
    gas_ln_n: Array,
    gas_stationarity_source: Array,
) -> Array:
    ag = jnp.asarray(formula_matrix)
    q = jnp.asarray(gas_ln_n)
    source = jnp.asarray(gas_stationarity_source)
    if ag.ndim != 2:
        raise ValueError("formula_matrix must be two-dimensional.")
    if q.ndim != 1 or source.ndim != 1 or q.shape != source.shape:
        raise ValueError("gas_ln_n and gas_stationarity_source must be same-length vectors.")
    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match gas_ln_n length.")
    return jnp.linalg.lstsq(ag.T, q + source, rcond=None)[0]


def _head_lifecycle_primary_summary(*, solver_success: bool) -> Mapping[str, Any]:
    if solver_success:
        return {
            "row_status": "centered",
            "converged_at_final_barrier": True,
            "reason": "restricted_solver_success_used_as_head_lifecycle_primary_boundary",
        }
    return {
        "row_status": "not_centered",
        "converged_at_final_barrier": False,
        "reason": "restricted_solver_failed_before_head_lifecycle_primary_boundary",
    }


def _head_lifecycle_primary_policy(options: CondensateEquilibriumOptions) -> Mapping[str, Any]:
    policy: dict[str, Any] = {}
    if options.max_outer_iterations is not None:
        policy["max_outer_iterations"] = int(options.max_outer_iterations)
    if options.max_inner_iterations is not None:
        policy["max_inner_iterations"] = int(options.max_inner_iterations)
    return policy


def _head_route_selected_route_override(options: CondensateEquilibriumOptions) -> str | None:
    if options.case_id is None:
        return None
    if options.head_route_primary_summary is None and options.head_route_refresh_policy_summary is None:
        return None
    return options.selected_route


def _run_lifecycle_from_warm_start_candidate(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    options: CondensateEquilibriumOptions,
    candidate: Any,
) -> Mapping[str, Any]:
    if candidate is None or candidate.initial_log_state_override is None:
        return {
            "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
            "explicit_opt_in": True,
            "production_behavior_change": False,
            "production_return_signature_change": False,
            "preset_default_wiring_change": False,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            "case_id": "runtime_layer",
            "family": "runtime_layer",
            "lifecycle_skipped_reason": "restricted_solver_failed_without_refresh_warm_start_state",
            "route_result": {
                "result_schema": "exogibbs_condensate_head_route_result_v1",
                "case_id": "runtime_layer",
                "family": "runtime_layer",
                "selected_route": options.selected_route,
                "integrated_status": "not_accepted",
                "metric_status": options.metric_status or "runtime_solver_failed",
                "acceptance_tier": "runtime_solver_failed",
                "standard_path_status": NOT_CONVERGED,
                "converged": False,
                "warning_messages": (
                    "The restricted support solver failed and no refresh warm-start state was available.",
                ),
                "diagnostics": {},
            },
        }
    from exogibbs.condensates.head_route_lifecycle import (
        run_condensate_head_route_lifecycle,
    )

    init_state = candidate.initial_log_state_override
    ln_nk = jnp.asarray(init_state.ln_nk)
    ln_mk = jnp.asarray(init_state.ln_mk)
    support_indices = tuple(int(index) for index in candidate.support_indices)
    if ln_mk.shape[0] == len(setup.condensate_species):
        support_amounts = jnp.exp(ln_mk[jnp.asarray(support_indices, dtype=jnp.int32)])
    else:
        support_amounts = jnp.exp(ln_mk)
    gas_stationarity_source = (
        jnp.asarray(setup.gas_setup.hvector_func(float(T)))
        + _ln_normalized_pressure(P, Pref)
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=ln_nk,
        gas_stationarity_source=gas_stationarity_source,
    )
    condensate_hvector = jnp.asarray(setup.condensate_setup.hvector_func(float(T)))
    try:
        lifecycle_report = run_condensate_head_route_lifecycle(
            explicit_opt_in=True,
            case_id=options.case_id or "runtime_layer",
            ln_nk=ln_nk,
            support_indices=support_indices,
            support_amounts=support_amounts,
            formula_matrix=setup.formula_matrix,
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            element_potential=element_potential,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=jnp.asarray(
                [condensate_hvector[index] for index in support_indices]
            ),
            primary_summary=options.head_route_primary_summary,
            primary_continuation_policy=_head_lifecycle_primary_policy(options),
            refresh_policy_summary=options.head_route_refresh_policy_summary,
            metric_status=options.metric_status,
            selected_route_override=_head_route_selected_route_override(options),
            field_provenance={
                "ln_nk": "exogibbs_head_route_refresh_warm_start",
                "support_indices": "exogibbs_head_route_warm_start_candidate",
                "support_amounts": "exogibbs_head_route_warm_start_candidate",
                "element_potential": "exogibbs_native_least_squares_gas_gauge",
            },
        )
        return lifecycle_report.as_dict()
    except Exception as exc:  # noqa: BLE001 - runtime diagnostics preserve the failure.
        return {
            "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
            "explicit_opt_in": True,
            "production_behavior_change": False,
            "production_return_signature_change": False,
            "preset_default_wiring_change": False,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            "case_id": "runtime_layer",
            "family": "runtime_layer",
            "lifecycle_failed_reason": f"{type(exc).__name__}: {exc}",
            "route_result": {
                "result_schema": "exogibbs_condensate_head_route_result_v1",
                "case_id": "runtime_layer",
                "family": "runtime_layer",
                "selected_route": options.selected_route,
                "integrated_status": "not_accepted",
                "metric_status": options.metric_status or "runtime_lifecycle_failed",
                "acceptance_tier": "runtime_lifecycle_failed",
                "standard_path_status": NOT_CONVERGED,
                "converged": False,
                "warning_messages": (
                    "The HEAD route lifecycle failed from the refresh warm-start state.",
                ),
                "diagnostics": {"exception_type": type(exc).__name__},
            },
        }


def _status_from_metric_status(
    *,
    metric_status: Optional[str],
    selected_route: str,
    solver_success: bool,
    allow_caveat_tiers: bool,
) -> tuple[str, str, tuple[str, ...]]:
    if metric_status is None:
        return ("runtime_unclassified", CONVERGED if solver_success else NOT_CONVERGED, ())
    if not solver_success:
        return (
            "runtime_solver_failed",
            NOT_CONVERGED,
            ("The restricted support solver did not report success.",),
        )
    gate = classify_head_route_standard_gate_row(
        condensate_enabled=True,
        case_id="runtime_layer",
        family="runtime_layer",
        selected_route=selected_route,
        metric_status=metric_status,
    )
    if gate.standard_path_status == CONVERGED_WITH_CAVEAT and not allow_caveat_tiers:
        return (gate.acceptance_tier, NOT_CONVERGED, gate.warning_messages)
    return (gate.acceptance_tier, gate.standard_path_status, gate.warning_messages)


def build_condensate_equilibrium_result_from_solver_payload(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    selected_route: str,
    metric_status: Optional[str],
    solver_success: bool,
    allow_caveat_tiers: bool = True,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> CondensateEquilibriumResult:
    """Build a production-facing condensate result from explicit solver arrays."""

    validate_condensate_chemical_setup(setup)
    gas_ln_n_array = jnp.asarray(gas_ln_n)
    if gas_ln_n_array.ndim != 1 or gas_ln_n_array.shape[0] != len(setup.gas_species):
        raise ValueError("gas_ln_n must have one value per gas species.")
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    condensate_amounts = _full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=jnp.asarray(support_amounts),
        condensate_count=len(setup.condensate_species),
    )
    acceptance_tier, status, warnings = _status_from_metric_status(
        metric_status=metric_status,
        selected_route=selected_route,
        solver_success=solver_success,
        allow_caveat_tiers=allow_caveat_tiers,
    )
    support_index_array = jnp.asarray(support_indices, dtype=jnp.int32)
    support_names = tuple(setup.condensate_species[int(index)] for index in support_index_array.tolist())
    metadata: dict[str, Any] = dict(diagnostics or {})
    metadata.setdefault("route", HEAD_ROUTE_STANDARD)
    metadata.setdefault("selected_route", selected_route)
    metadata.setdefault("acceptance_tier", acceptance_tier)
    metadata.setdefault("warning_messages", warnings)
    metadata.setdefault("fastchem4_trace_public_runtime_constructor_inputs_used", False)
    return CondensateEquilibriumResult(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=condensate_amounts,
        condensate_support_indices=support_index_array,
        condensate_support_names=support_names,
        acceptance_tier=acceptance_tier,
        selected_route=selected_route,
        status=status,
        converged=status in {CONVERGED, CONVERGED_WITH_CAVEAT},
        diagnostics=metadata,
    )


def _build_empty_support_gas_result(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    diagnostics: Optional[Mapping[str, Any]],
) -> CondensateEquilibriumResult:
    gas_ln_n_array = jnp.asarray(gas_ln_n)
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    metadata = dict(diagnostics or {})
    metadata.setdefault("route", HEAD_ROUTE_STANDARD)
    metadata.setdefault("selected_route", "head_v1_empty_positive_support_gas_only")
    metadata.setdefault("acceptance_tier", "runtime_empty_positive_support")
    metadata.setdefault("warning_messages", ())
    metadata.setdefault("fastchem4_trace_public_runtime_constructor_inputs_used", False)
    return CondensateEquilibriumResult(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=jnp.zeros((len(setup.condensate_species),), dtype=gas_n.dtype),
        condensate_support_indices=jnp.asarray((), dtype=jnp.int32),
        condensate_support_names=(),
        acceptance_tier="runtime_empty_positive_support",
        selected_route="head_v1_empty_positive_support_gas_only",
        status=CONVERGED,
        converged=True,
        diagnostics=metadata,
    )


def condensate_equilibrium(
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
) -> CondensateEquilibriumResult:
    """Compute one condensate-enabled equilibrium layer through HEAD route v1.

    When no support is supplied, the HEAD route builds a native positive-support
    initializer from ExoGibbs thermochemistry and the caller's element budget.
    Explicit support payloads are still accepted for controlled experiments.
    """

    opts = options or CondensateEquilibriumOptions()
    validate_condensate_chemical_setup(setup)
    _validate_options(opts)
    support_selection_report: Optional[Mapping[str, Any]] = None
    if support_indices is None:
        from exogibbs.condensates.positive_support_initializer import (
            build_positive_support_initializer_report,
        )

        support_plan = build_positive_support_initializer_report(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            condensate_species_order=setup.condensate_species,
            hvector_cond=setup.condensate_setup.hvector_func(float(T)),
            max_positive_support_count=opts.max_positive_support_count,
            seed_fraction=opts.seed_fraction,
            max_seed_amount=opts.max_seed_amount,
            min_seed_amount=opts.min_seed_amount,
            allow_empty_positive_support=opts.allow_empty_positive_support,
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_runtime_input",
                "hvector_cond": "exogibbs_condensate_thermochemistry",
            },
        )
        support_selection_report = support_plan.as_dict()
        support_indices = support_plan.solver_inputs.support_indices
        support_amounts_init = support_plan.solver_inputs.support_amounts_init
    else:
        explicit_indices = tuple(int(index) for index in support_indices)
        explicit_amounts = (
            ()
            if support_amounts_init is None
            else tuple(float(value) for value in jnp.asarray(support_amounts_init).tolist())
        )
        support_selection_report = {
            "selection_schema": "exogibbs_explicit_condensate_support_payload_v1",
            "selection_mode": "explicit_support_payload",
            "solver_inputs": {
                "support_indices": explicit_indices,
                "support_amounts_init": explicit_amounts,
                "empty_positive_support": len(explicit_indices) == 0,
            },
            "fastchem4_trace_values_used": False,
            "fastchem4_public_values_used_as_constructor_inputs": False,
            "fastchem4_runtime_values_used_as_constructor_inputs": False,
        }
    from exogibbs.optimize.minimize_cond import solve_restricted_support_condensate_layer
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium

    state = ThermoState(
        temperature=float(T),
        ln_normalized_pressure=_ln_normalized_pressure(P, Pref),
        element_vector=jnp.asarray(b),
    )
    if len(tuple(support_indices)) == 0:
        gas_result = equilibrium(
            setup.gas_setup,
            T,
            P,
            jnp.asarray(b),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        diagnostics = {"support_selection": support_selection_report} if opts.return_diagnostics else None
        return _build_empty_support_gas_result(
            setup=setup,
            gas_ln_n=gas_result.ln_n,
            diagnostics=diagnostics,
        )
    solve_kwargs: dict[str, Any] = {}
    if opts.max_inner_iterations is not None:
        solve_kwargs["max_iter"] = int(opts.max_inner_iterations)
    from exogibbs.condensates.head_route_warm_start import (
        build_condensate_head_route_warm_start_report,
    )

    if support_amounts_init is None:
        raise ValueError("support_amounts_init is required for non-empty condensate support.")
    warm_start_report = build_condensate_head_route_warm_start_report(
        explicit_opt_in=True,
        state=state,
        formula_matrix=setup.formula_matrix,
        formula_matrix_cond=setup.formula_matrix_cond,
        hvector_func=setup.gas_setup.hvector_func,
        support_indices=support_indices,
        support_amounts_init=jnp.asarray(support_amounts_init),
        enable_depleted_gas_refresh=(
            opts.enable_head_route_warm_start and opts.enable_depleted_gas_refresh
        ),
        gas_refresh_policy=opts.warm_start_gas_refresh_policy,
        field_provenance={
            "formula_matrix": "exogibbs_condensate_chemical_setup",
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_budget": "exogibbs_runtime_input",
            "ln_mk": "exogibbs_head_route_positive_support_seed",
            "hvector_func": "exogibbs_gas_thermochemistry",
        },
    )
    solver_attempts: list[dict[str, Any]] = []
    solver: Mapping[str, Any] | None = None
    selected_warm_start_candidate: Mapping[str, Any] | None = None
    selected_warm_start_candidate_object = None
    selected_solver_success = False
    for candidate_index, candidate in enumerate(warm_start_report.candidates):
        if not candidate.finite_solver_inputs:
            solver_attempts.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_name": candidate.candidate_name,
                    "candidate_kind": candidate.candidate_kind,
                    "attempt_status": "skipped_nonfinite_solver_inputs",
                    "solver_success": False,
                }
            )
            continue
        attempt = solve_restricted_support_condensate_layer(
            state,
            setup.formula_matrix,
            setup.formula_matrix_cond,
            setup.gas_setup.hvector_func,
            setup.condensate_setup.hvector_func,
            support_indices=candidate.support_indices,
            condensate_species=setup.condensate_species,
            element_names=setup.elements,
            support_amounts_init=jnp.asarray(candidate.support_amounts_init),
            initial_log_state_override=candidate.initial_log_state_override,
            **solve_kwargs,
        )
        attempt_success = bool(attempt["solver_success"])
        solver_attempts.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate.candidate_name,
                "candidate_kind": candidate.candidate_kind,
                "attempt_status": "solver_success" if attempt_success else "solver_failed",
                "solver_success": attempt_success,
            }
        )
        if solver is None or attempt_success or not selected_solver_success:
            solver = attempt
            selected_warm_start_candidate_object = warm_start_report.candidates[candidate_index]
            selected_warm_start_candidate = selected_warm_start_candidate_object.as_dict()
            selected_solver_success = attempt_success
        if attempt_success:
            break
    if solver is None:
        raise RuntimeError("No finite condensate HEAD route warm-start candidate was available.")
    from exogibbs.condensates.head_route_lifecycle import (
        run_condensate_head_route_lifecycle,
    )

    restricted_solver_success = bool(solver["solver_success"])
    solver_ln_nk = jnp.asarray(solver["ln_nk"])
    solver_support_indices = tuple(int(index) for index in solver["support_indices"])
    solver_support_amounts = jnp.asarray(solver["m_support"])
    lifecycle_payload: Mapping[str, Any]
    lifecycle_selected_route = opts.selected_route
    lifecycle_metric_status = opts.metric_status
    lifecycle_converged = False
    result_ln_nk = solver_ln_nk
    result_support_indices = solver_support_indices
    result_support_amounts = solver_support_amounts
    if restricted_solver_success:
        gas_stationarity_source = (
            jnp.asarray(setup.gas_setup.hvector_func(float(T)))
            + _ln_normalized_pressure(P, Pref)
        )
        element_potential = _least_squares_element_potential(
            formula_matrix=setup.formula_matrix,
            gas_ln_n=solver_ln_nk,
            gas_stationarity_source=gas_stationarity_source,
        )
        condensate_hvector = jnp.asarray(setup.condensate_setup.hvector_func(float(T)))
        lifecycle_report = run_condensate_head_route_lifecycle(
            explicit_opt_in=True,
            case_id=opts.case_id or "runtime_layer",
            ln_nk=solver_ln_nk,
            support_indices=solver_support_indices,
            support_amounts=solver_support_amounts,
            formula_matrix=setup.formula_matrix,
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            element_potential=element_potential,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=jnp.asarray(
                [condensate_hvector[index] for index in solver_support_indices]
            ),
            primary_summary=opts.head_route_primary_summary,
            primary_continuation_policy=_head_lifecycle_primary_policy(opts),
            refresh_policy_summary=opts.head_route_refresh_policy_summary,
            metric_status=opts.metric_status,
            selected_route_override=_head_route_selected_route_override(opts),
            field_provenance={
                "ln_nk": "exogibbs_restricted_support_solver_output",
                "support_indices": "exogibbs_restricted_support_solver_output",
                "support_amounts": "exogibbs_restricted_support_solver_output",
                "element_potential": "exogibbs_native_least_squares_gas_gauge",
            },
        )
        lifecycle_payload = lifecycle_report.as_dict()
        lifecycle_selected_route = lifecycle_report.route_result.selected_route
        lifecycle_metric_status = lifecycle_report.route_result.metric_status
        lifecycle_converged = bool(lifecycle_report.route_result.converged)
    else:
        lifecycle_payload = _run_lifecycle_from_warm_start_candidate(
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            b=b,
            options=opts,
            candidate=selected_warm_start_candidate_object,
        )
        route_result_payload = lifecycle_payload["route_result"]
        lifecycle_selected_route = str(route_result_payload["selected_route"])
        lifecycle_metric_status = str(route_result_payload["metric_status"])
        lifecycle_converged = bool(route_result_payload["converged"])
        primary_execution_payload = lifecycle_payload.get("primary_execution_report")
        if isinstance(primary_execution_payload, Mapping):
            continuation_payload = primary_execution_payload.get("continuation_report", {})
        else:
            continuation_payload = {}
        final_state_payload = (
            continuation_payload.get("final_state")
            if isinstance(continuation_payload, Mapping)
            else None
        )
        if lifecycle_converged and isinstance(final_state_payload, Mapping):
            result_ln_nk = jnp.asarray(final_state_payload["ln_nk"])
            result_support_indices = tuple(int(index) for index in selected_warm_start_candidate_object.support_indices)
            result_support_amounts = jnp.exp(jnp.asarray(final_state_payload["ln_mk"]))
    diagnostics_payload: Optional[Mapping[str, Any]]
    if opts.return_diagnostics:
        diagnostics_payload = {
            **solver,
            "restricted_solver_success": restricted_solver_success,
            "solver_success": bool(lifecycle_converged),
            "support_selection": support_selection_report,
            "head_route_warm_start": warm_start_report.as_dict(),
            "head_route_solver_attempts": tuple(solver_attempts),
            "selected_warm_start_candidate": selected_warm_start_candidate,
            "head_route_lifecycle": lifecycle_payload,
        }
    else:
        diagnostics_payload = None
    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=result_ln_nk,
        support_indices=result_support_indices,
        support_amounts=result_support_amounts,
        selected_route=lifecycle_selected_route,
        metric_status=lifecycle_metric_status,
        solver_success=bool(lifecycle_converged),
        allow_caveat_tiers=opts.allow_caveat_tiers,
        diagnostics=diagnostics_payload,
    )


def condensate_equilibrium_profile(*args: Any, **kwargs: Any) -> Any:
    """Profile condensate equilibrium placeholder for the HEAD route API."""

    raise NotImplementedError("condensate_equilibrium_profile will be connected after one-layer HEAD route wiring.")


__all__ = (
    "CondensateChemicalSetup",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumResult",
    "build_condensate_chemical_setup",
    "build_condensate_equilibrium_result_from_solver_payload",
    "condensate_equilibrium",
    "condensate_equilibrium_profile",
    "validate_condensate_chemical_setup",
)
