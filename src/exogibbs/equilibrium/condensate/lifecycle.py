"""Host-side support lifecycle for production condensate equilibrium.

The fixed-support solver owns one support solve. Support discovery and
expansion remain an API-level lifecycle outside that solver.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.condensate.initialization import (
    resolve_condensate_initial_guess,
)
from exogibbs.equilibrium.condensate.acceptance import (
    accept_condensate_result_state,
    independent_kkt_passed as _head_v2_kkt_passed,
    least_squares_element_potential as _least_squares_element_potential,
)
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    condensate_temperature_validity_upper,
    validate_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.results import (
    build_condensate_equilibrium_result,
    full_condensate_amounts,
    merge_external_condensate_amounts,
)
from exogibbs.equilibrium.condensate.support import (
    evaluate_profile_support_closure,
)
from exogibbs.equilibrium.condensate.support import (
    positive_support_amounts_for_warm_start as _positive_support_amounts_for_warm_start,
)
from exogibbs.equilibrium.condensate.support import (
    support_payload_from_condensate_init as _support_payload_from_condensate_init,
)
from exogibbs.equilibrium.condensate.types import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
    DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR,
    HEAD_ROUTE_V2,
    Array,
    CondensateEquilibriumInit,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumInitRequest,
    CondensateEquilibriumOptions,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
    CondensateRoute,
    CondensateProfileFixedSupportSeedPolicy,
    CondensateProfileNativeActivitySource,
    CondensateProfileNativeActivitySupportPolicy,
    ExperimentalCondensateProfileFixedSupportBatchPlan,
    HeadV2LayerState,
)
from exogibbs.equilibrium.gas.types import EquilibriumInit, ThermoState


_ExperimentalProfileFixedSupportBatchPlan = (
    ExperimentalCondensateProfileFixedSupportBatchPlan
)
_HeadV2LayerState = HeadV2LayerState


def build_condensate_equilibrium_result_from_solver_payload(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    external_condensate_amounts: Sequence[float] | Array | None = None,
    selected_route: str,
    solver_success: bool,
    route: CondensateRoute = HEAD_ROUTE_V2,
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION,
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME,
    diagnostics: Optional[Mapping[str, Any]] = None,
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    ),
) -> CondensateEquilibriumResult:
    """Accept a solver payload, then construct its public result."""

    condensate_amounts = full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=jnp.asarray(support_amounts, dtype=jnp.float64),
        condensate_count=len(setup.condensate_species),
    )
    condensate_amounts = merge_external_condensate_amounts(
        condensate_amounts=condensate_amounts,
        external_condensate_amounts=external_condensate_amounts,
    )
    accepted_state = accept_condensate_result_state(
        setup=setup,
        gas_ln_n=jnp.asarray(gas_ln_n, dtype=jnp.float64),
        condensate_amounts=condensate_amounts,
        solver_success=solver_success,
        diagnostics=diagnostics,
        element_inventory_target=element_inventory_target,
        enable_full_condensate_budget_residual_gate=(
            enable_full_condensate_budget_residual_gate
        ),
        full_condensate_budget_relative_tolerance=(
            full_condensate_budget_relative_tolerance
        ),
        full_condensate_budget_relative_floor=(
            full_condensate_budget_relative_floor
        ),
    )
    return build_condensate_equilibrium_result(
        setup=setup,
        accepted_state=accepted_state,
        support_indices=support_indices,
        selected_route=selected_route,
        route=route,
        head_route_version=head_route_version,
        head_route_name=head_route_name,
    )


def _build_empty_support_gas_result(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    diagnostics: Optional[Mapping[str, Any]],
    route: CondensateRoute = HEAD_ROUTE_V2,
    selected_route: str = "head_v2_gas_only_no_candidate",
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION,
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME,
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    ),
) -> CondensateEquilibriumResult:
    """Accept and construct a gas-only lifecycle result."""

    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=(),
        support_amounts=(),
        selected_route=selected_route,
        solver_success=True,
        route=route,
        head_route_version=head_route_version,
        head_route_name=head_route_name,
        diagnostics=diagnostics,
        element_inventory_target=element_inventory_target,
        enable_full_condensate_budget_residual_gate=(
            enable_full_condensate_budget_residual_gate
        ),
        full_condensate_budget_relative_tolerance=(
            full_condensate_budget_relative_tolerance
        ),
        full_condensate_budget_relative_floor=(
            full_condensate_budget_relative_floor
        ),
    )


def _ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    return jnp.log(jnp.asarray(pressure) / jnp.asarray(reference_pressure))


def _inventory_amount_gauge_scale(
    setup: CondensateChemicalSetup,
    element_inventory_target: Array,
) -> float:
    """Return the fixed extensive scale for one production lifecycle."""

    target = np.asarray(element_inventory_target, dtype=np.float64)
    if target.shape != (len(setup.elements),):
        raise ValueError(
            "element_inventory_target must have one value per element."
        )
    if not np.all(np.isfinite(target)):
        raise ValueError(
            "element_inventory_target must contain only finite values."
        )
    physical_rows = np.asarray(
        [
            str(element) not in {"e-", "electron"}
            for element in setup.elements
        ],
        dtype=bool,
    )
    if np.any(target[physical_rows] < 0.0):
        raise ValueError(
            "Non-charge element inventory targets must be non-negative."
        )
    positive = physical_rows & (target > 0.0)
    if not np.any(positive):
        raise ValueError(
            "element_inventory_target must contain a positive non-charge "
            "amount."
        )
    scale = float(np.sum(target[positive]))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Unable to construct a finite amount gauge scale.")
    return scale


def _normalize_condensate_init_amount_gauge(
    init: CondensateEquilibriumInit | None,
    amount_scale: float,
) -> CondensateEquilibriumInit | None:
    """Convert one caller-gauge initializer to the canonical amount gauge."""

    if not math.isfinite(amount_scale) or amount_scale <= 0.0:
        raise ValueError("amount_scale must be finite and positive.")
    if init is None or amount_scale == 1.0:
        return init
    log_scale = math.log(amount_scale)
    return replace(
        init,
        gas_ln_n=(
            None
            if init.gas_ln_n is None
            else jnp.asarray(init.gas_ln_n, dtype=jnp.float64) - log_scale
        ),
        gas_ntot=(
            None
            if init.gas_ntot is None
            else jnp.asarray(init.gas_ntot, dtype=jnp.float64) / amount_scale
        ),
        condensate_amounts=(
            None
            if init.condensate_amounts is None
            else jnp.asarray(init.condensate_amounts, dtype=jnp.float64)
            / amount_scale
        ),
        support_amounts=(
            None
            if init.support_amounts is None
            else tuple(float(value) / amount_scale for value in init.support_amounts)
        ),
        barrier_epsilon=(
            None
            if init.barrier_epsilon is None
            else jnp.asarray(init.barrier_epsilon, dtype=jnp.float64)
            - log_scale
        ),
    )


def _canonical_budget_scale_for_caller_audit(
    target_inventory: Array,
    *,
    amount_scale: float,
    relative_floor: float,
) -> np.ndarray:
    """Return canonical row scales for a caller-gauge budget audit."""

    target = np.asarray(target_inventory, dtype=np.float64) / amount_scale
    representation_floor = 1.0e-300
    nonzero = target != 0.0
    inventory_max = float(np.max(np.abs(target[nonzero]), initial=0.0))
    zero_target_floor = max(
        relative_floor,
        np.finfo(np.float64).eps * inventory_max,
        representation_floor,
    )
    denominator = np.where(
        nonzero,
        np.maximum(np.abs(target), representation_floor),
        zero_target_floor,
    )
    return 1.0 / denominator


def _gas_init_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    gas_species_count: int,
) -> EquilibriumInit | None:
    """Return a valid gas-solver seed from a condensate initializer."""

    if init is None or init.gas_ln_n is None or init.gas_ntot is None:
        return None
    gas_ln_n = jnp.asarray(init.gas_ln_n, dtype=jnp.float64)
    gas_ntot = jnp.asarray(init.gas_ntot, dtype=jnp.float64)
    if (
        gas_ln_n.shape != (gas_species_count,)
        or gas_ntot.ndim != 0
        or not bool(jnp.all(jnp.isfinite(gas_ln_n)))
        or not bool(jnp.isfinite(gas_ntot))
        or float(gas_ntot) <= 0.0
    ):
        return None
    return EquilibriumInit(
        ln_nk=gas_ln_n,
        ln_ntot=jnp.log(gas_ntot),
    )


def _solver_log_state_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    support_amounts_init: Sequence[float],
    source: str,
) -> Any | None:
    """Build a v2-owned prepared state from a profile initializer."""

    if init is None or init.gas_ln_n is None:
        return None
    gas_ln_n = jnp.asarray(init.gas_ln_n, dtype=jnp.float64)
    if gas_ln_n.ndim != 1 or gas_ln_n.shape[0] != len(setup.gas_species):
        return None
    if not bool(jnp.all(jnp.isfinite(gas_ln_n))):
        return None
    if init.gas_ntot is None:
        gas_ntot = jnp.sum(jnp.exp(gas_ln_n))
    else:
        gas_ntot = jnp.asarray(init.gas_ntot, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(gas_ntot))) or float(gas_ntot) <= 0.0:
        return None
    support_amounts = jnp.asarray(support_amounts_init, dtype=jnp.float64)
    if (
        support_amounts.ndim != 1
        or not bool(jnp.all(jnp.isfinite(support_amounts)))
        or not bool(jnp.all(support_amounts > 0.0))
    ):
        return None
    from exogibbs.equilibrium.condensate.fixed_support.batch import (
        PreparedFixedSupportV2LayerState,
    )

    element_potential = None
    if init.element_potential is not None:
        element_potential = jnp.asarray(init.element_potential, dtype=jnp.float64)
        if (
            element_potential.ndim != 1
            or element_potential.shape[0] != len(setup.elements)
            or not bool(jnp.all(jnp.isfinite(element_potential)))
        ):
            element_potential = None
    rho = None
    if init.rho is not None:
        rho = jnp.asarray(init.rho, dtype=jnp.float64)
        if (
            rho.ndim != 1
            or rho.shape[0] not in {len(support_amounts_init), len(setup.condensate_species)}
            or not bool(jnp.all(jnp.isfinite(rho)))
        ):
            rho = None
    barrier_epsilon = None
    if init.barrier_epsilon is not None:
        barrier_epsilon = jnp.asarray(init.barrier_epsilon, dtype=jnp.float64)
        if barrier_epsilon.ndim != 0 or not bool(jnp.isfinite(barrier_epsilon)):
            barrier_epsilon = None
    del source
    return PreparedFixedSupportV2LayerState(
        ln_nk=gas_ln_n,
        ln_mk=jnp.log(jnp.maximum(support_amounts, 1.0e-300)),
        ln_ntot=jnp.log(jnp.asarray(gas_ntot, dtype=jnp.float64)),
        element_potential=element_potential,
        rho=rho,
        barrier_epsilon=barrier_epsilon,
    )


def _native_activity_expanded_profile_support_payload(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    support_indices: Sequence[int],
    support_policy: CondensateProfileNativeActivitySupportPolicy,
    support_topk: int,
    max_support_count: int,
    activity_threshold: float,
    activity_source_requested: CondensateProfileNativeActivitySource,
    seed_policy: CondensateProfileFixedSupportSeedPolicy,
    seed_fraction: float,
    max_seed_amount: float,
    min_seed_amount: float,
    activity_gas_ln_n: Sequence[float] | Array | None = None,
    activity_gas_ntot: Sequence[float] | Array | float | None = None,
    activity_gas_stationarity_source: Sequence[float] | Array | None = None,
    gas_equilibrium_init: EquilibriumInit | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], Mapping[str, Any]]:
    from exogibbs.equilibrium.gas.solve import equilibrium
    from exogibbs.equilibrium.gas.types import EquilibriumOptions
    from exogibbs.condensates.support_selection_policy import (
        select_activity_driven_support_candidates,
    )

    base_support = tuple(dict.fromkeys(int(index) for index in support_indices))
    activity_source = str(activity_source_requested)
    gas_ln_n_for_activity = None
    gas_stationarity_source = None
    if activity_source == "initializer_gas" and activity_gas_ln_n is not None:
        candidate_ln_n = jnp.asarray(activity_gas_ln_n, dtype=jnp.float64)
        if (
            candidate_ln_n.ndim == 1
            and candidate_ln_n.shape[0] == len(setup.gas_species)
            and bool(jnp.all(jnp.isfinite(candidate_ln_n)))
        ):
            gas_ln_n_for_activity = candidate_ln_n
            if activity_gas_stationarity_source is not None:
                candidate_source = jnp.asarray(
                    activity_gas_stationarity_source,
                    dtype=jnp.float64,
                )
                if (
                    candidate_source.ndim == 1
                    and candidate_source.shape[0] == len(setup.gas_species)
                    and bool(jnp.all(jnp.isfinite(candidate_source)))
                ):
                    gas_stationarity_source = candidate_source
            if gas_stationarity_source is None:
                if activity_gas_ntot is None:
                    gas_ntot = jnp.sum(jnp.exp(candidate_ln_n))
                else:
                    gas_ntot = jnp.asarray(activity_gas_ntot, dtype=jnp.float64)
                gas_stationarity_source = setup.gas_setup.hvector_func(float(T)) + (
                    _ln_normalized_pressure(P, Pref)
                ) - jnp.log(jnp.clip(gas_ntot, 1.0e-300))
    if gas_ln_n_for_activity is None or gas_stationarity_source is None:
        activity_source = "gas_only_full_budget"
        gas_result = equilibrium(
            setup.gas_setup,
            float(T),
            float(P),
            jnp.asarray(b, dtype=jnp.float64),
            Pref=Pref,
            init=gas_equilibrium_init,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        gas_ln_n_for_activity = jnp.asarray(gas_result.ln_n, dtype=jnp.float64)
        gas_stationarity_source = setup.gas_setup.hvector_func(float(T)) + (
            _ln_normalized_pressure(P, Pref)
        ) - jnp.log(jnp.asarray(gas_result.ntot, dtype=jnp.float64))
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=gas_ln_n_for_activity,
        gas_stationarity_source=gas_stationarity_source,
    )
    element_potential_provenance = (
        "exogibbs_profile_initializer_gas_state"
        if activity_source == "initializer_gas"
        else "exogibbs_native_gas_only_equilibrium"
    )
    activity_gauge_note = (
        "FastChem-style log_activity = A_cond.T @ element_potential "
        "- hvector_cond, with element_potential recovered from the selected "
        "ExoGibbs-native gas state."
    )
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=b,
        condensate_species_order=setup.condensate_species,
        hvector_cond=setup.condensate_setup.hvector_func(float(T)),
        element_potential=element_potential,
        max_positive_support_count=int(support_topk),
        activity_threshold=float(activity_threshold),
        existing_support_indices=base_support,
        temperature=float(T),
        condensate_temperature_validity_upper=(
            condensate_temperature_validity_upper(setup)
        ),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_profile_budget",
            "hvector_cond": "exogibbs_condensate_thermochemistry",
            "element_potential": element_potential_provenance,
            "condensate_temperature_validity_upper": (
                "exogibbs_condensate_temperature_validity_metadata"
            ),
        },
    )
    if support_policy == "fastchem_activity_all":
        report = select_activity_driven_support_candidates(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=b,
            condensate_species_order=setup.condensate_species,
            hvector_cond=setup.condensate_setup.hvector_func(float(T)),
            element_potential=element_potential,
            max_positive_support_count=None,
            activity_threshold=float(activity_threshold),
            existing_support_indices=base_support,
            temperature=float(T),
            condensate_temperature_validity_upper=(
                condensate_temperature_validity_upper(setup)
            ),
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_profile_budget",
                "hvector_cond": "exogibbs_condensate_thermochemistry",
                "element_potential": element_potential_provenance,
                "condensate_temperature_validity_upper": (
                    "exogibbs_condensate_temperature_validity_metadata"
                ),
                "activity_gauge": activity_gauge_note,
            },
        )
    additions = tuple(
        int(index)
        for index in report.positive_support_indices
        if int(index) not in set(base_support)
    )
    support_limit = max(
        int(max_support_count),
        len(base_support),
    )
    expanded_support = tuple(dict.fromkeys((*base_support, *additions)))[:support_limit]
    seeded_support = expanded_support
    if seeded_support:
        from exogibbs.condensates.initialization_policy import (
            recommend_budget_preserving_seed_amounts,
        )

        seed = recommend_budget_preserving_seed_amounts(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            condensate_species_order=setup.condensate_species,
            support_indices=seeded_support,
            seed_fraction=1.0 if seed_policy == "max_density" else seed_fraction,
            max_seed_amount=(
                1.0e300 if seed_policy == "max_density" else max_seed_amount
            ),
            min_seed_amount=min_seed_amount,
            preserve_budget_fraction=(seed_policy == "budget_preserving_fraction"),
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_profile_budget",
                "recommended_amounts": (
                    "derived_from_native_budget_capacity_with_shared_budget_fraction"
                    if seed_policy == "budget_preserving_fraction"
                    else "derived_from_native_budget_capacity_without_shared_budget_rescale"
                ),
            },
        )
        seeded_amounts = tuple(
            float(value) for value in seed.recommended_amounts
        )
    else:
        seeded_amounts = ()
    trace = {
        "policy": (
            "fastchem_style_activity_all_support_expansion"
            if support_policy == "fastchem_activity_all"
            else "native_gas_activity_curated_support_expansion"
        ),
        "profile_native_activity_support_policy": support_policy,
        "profile_native_activity_source": activity_source,
        "profile_native_activity_source_requested": activity_source_requested,
        "activity_threshold_semantics": (
            "FastChem selectActiveCondensates-compatible: log_activity >= "
            "threshold, where ExoGibbs computes log_activity as "
            "A_cond.T @ element_potential - hvector_cond."
        ),
        "fastchem4_public_values_used_as_constructor_inputs": False,
        "base_support_count": len(base_support),
        "max_support_count": support_limit,
        "profile_fixed_support_seed_policy": seed_policy,
        "seed_policy_semantics": (
            "budget_preserving_fraction keeps the existing batch seed behavior; "
            "max_density uses ExoGibbs-native condensate capacity without shared "
            "support rescaling, analogous to FastChem maxDensity initialization."
        ),
        "uncapped_positive_activity_count": len(report.positive_support_indices),
        "expanded_support_count": len(seeded_support),
        "added_support_count": max(0, len(seeded_support) - len(base_support)),
        "added_support_indices": tuple(
            index for index in seeded_support if index not in set(base_support)
        ),
        "added_support_names": tuple(
            setup.condensate_species[int(index)]
            for index in seeded_support
            if int(index) not in set(base_support)
        ),
        "selection_report": report.as_dict(),
    }
    return seeded_support, seeded_amounts, trace


def _head_v2_best_residual_element_potential(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    gas_ln_n: Array,
    total_gas_log_amount: Array,
    epsilon: float,
) -> Array:
    """Return the validated global best-residual multiplier initializer."""

    ag = jnp.asarray(setup.formula_matrix, dtype=jnp.float64)
    support = jnp.asarray(
        tuple(int(index) for index in support_indices), dtype=jnp.int32
    )
    ac = jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64)[:, support]
    q = jnp.asarray(gas_ln_n, dtype=jnp.float64)
    r = jnp.log(jnp.asarray(support_amounts, dtype=jnp.float64))
    qtot = jnp.asarray(total_gas_log_amount, dtype=jnp.float64)
    gamma = jnp.asarray(
        setup.gas_setup.hvector_func(float(T)), dtype=jnp.float64
    ) + _ln_normalized_pressure(P, Pref)
    hcond = jnp.asarray(
        setup.condensate_setup.hvector_func(float(T)), dtype=jnp.float64
    )[support]
    eta = jnp.ones_like(r)
    gas_matrix = ag.T
    gas_rhs = q + gamma - qtot
    joint_matrix = jnp.concatenate([ag.T, ac.T], axis=0)
    joint_rhs = jnp.concatenate([gas_rhs, hcond - eta])

    def damped_lstsq(matrix: Array, rhs: Array) -> Array:
        column_scale = jnp.maximum(
            jnp.linalg.norm(matrix, axis=0),
            jnp.asarray(1.0e-300, dtype=matrix.dtype),
        )
        scaled = matrix / column_scale[None, :]
        normal = scaled.T @ scaled
        normal_rhs = scaled.T @ rhs
        damping = jnp.maximum(
            jnp.asarray(1.0e-12, dtype=matrix.dtype)
            * jnp.mean(jnp.diag(normal)),
            jnp.asarray(1.0e-30, dtype=matrix.dtype),
        )
        solution = jnp.linalg.solve(
            normal + damping * jnp.eye(normal.shape[0], dtype=matrix.dtype),
            normal_rhs,
        )
        return jnp.nan_to_num(
            solution / column_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    candidates = jnp.stack(
        [
            jnp.zeros((ag.shape[0],), dtype=jnp.float64),
            jnp.linalg.lstsq(gas_matrix, gas_rhs, rcond=None)[0],
            jnp.linalg.lstsq(joint_matrix, joint_rhs, rcond=None)[0],
            damped_lstsq(gas_matrix, gas_rhs),
            damped_lstsq(joint_matrix, joint_rhs),
        ],
        axis=0,
    )

    def residual_norm(element_potential: Array) -> Array:
        values = jnp.concatenate(
            [
                q + gamma - qtot - ag.T @ element_potential,
                hcond - ac.T @ element_potential - eta,
                ag @ jnp.exp(q)
                + ac @ jnp.exp(r)
                - jnp.asarray(b, dtype=jnp.float64),
                r - jnp.asarray(epsilon, dtype=jnp.float64),
                jnp.asarray(
                    [jnp.sum(jnp.exp(q)) - jnp.exp(qtot)],
                    dtype=jnp.float64,
                ),
            ]
        )
        scale = jnp.max(
            jnp.abs(values),
            initial=jnp.asarray(0.0, dtype=values.dtype),
        )
        return jnp.where(
            scale == 0.0,
            0.0,
            scale * jnp.linalg.norm(values / scale),
        )

    residuals = jax.vmap(residual_norm)(candidates)
    return candidates[jnp.argmin(residuals)]


def _head_v2_initial_state(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    initial_guess: CondensateEquilibriumInit | None,
    first_epsilon: float,
) -> _HeadV2LayerState:
    """Build one v2 lifecycle state from gas equilibrium and support seeds."""

    from exogibbs.equilibrium.gas.solve import equilibrium
    from exogibbs.equilibrium.gas.types import EquilibriumOptions

    support = tuple(int(index) for index in support_indices)
    amounts = jnp.asarray(support_amounts, dtype=jnp.float64)
    candidate_q = None if initial_guess is None else initial_guess.gas_ln_n
    if candidate_q is None:
        gas_result = equilibrium(
            setup.gas_setup,
            float(T),
            float(P),
            jnp.asarray(b, dtype=jnp.float64),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        q = jnp.asarray(gas_result.ln_n, dtype=jnp.float64)
        qtot = jnp.log(jnp.asarray(gas_result.ntot, dtype=jnp.float64))
    else:
        q = jnp.asarray(candidate_q, dtype=jnp.float64)
        if q.ndim != 1 or q.shape[0] != len(setup.gas_species):
            raise ValueError(
                "head_v2 initial gas_ln_n must have one value per gas species."
            )
        if initial_guess is not None and initial_guess.gas_ntot is not None:
            gas_ntot = jnp.asarray(initial_guess.gas_ntot, dtype=jnp.float64)
        else:
            gas_ntot = jnp.sum(jnp.exp(q))
        qtot = jnp.log(jnp.clip(gas_ntot, 1.0e-300))
    r = jnp.log(jnp.maximum(amounts, 1.0e-300))
    supplied_potential = (
        None if initial_guess is None else initial_guess.element_potential
    )
    if supplied_potential is None:
        element_potential = _head_v2_best_residual_element_potential(
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            b=b,
            support_indices=support,
            support_amounts=amounts,
            gas_ln_n=q,
            total_gas_log_amount=qtot,
            epsilon=first_epsilon,
        )
    else:
        element_potential = jnp.asarray(
            supplied_potential, dtype=jnp.float64
        )
        if (
            element_potential.ndim != 1
            or element_potential.shape[0] != len(setup.elements)
        ):
            raise ValueError(
                "head_v2 element_potential must have one value per element."
            )
    return _HeadV2LayerState(
        support_indices=support,
        gas_ln_n=q,
        condensate_log_amounts=r,
        total_gas_log_amount=qtot,
        element_potential=element_potential,
    )


def _head_v2_prepared_buckets(
    *,
    setup: CondensateChemicalSetup,
    temperatures: Sequence[float],
    pressures: Sequence[float],
    b: Array,
    Pref: float,
    states: Sequence[_HeadV2LayerState],
) -> tuple[Any, ...]:
    """Group pending v2 lifecycle states by exact fixed support."""

    from exogibbs.equilibrium.condensate.fixed_support.batch import (
        PreparedFixedSupportV2LayerState,
        prepare_fixed_support_v2_buckets,
    )

    init_states = tuple(
        PreparedFixedSupportV2LayerState(
            ln_nk=state.gas_ln_n,
            ln_mk=state.condensate_log_amounts,
            ln_ntot=state.total_gas_log_amount,
            element_potential=state.element_potential,
        )
        for state in states
    )
    target = jnp.asarray(b, dtype=jnp.float64)
    return prepare_fixed_support_v2_buckets(
        init_states=init_states,
        support_indices_by_layer=tuple(
            state.support_indices for state in states
        ),
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target_by_layer=jnp.stack(
            [target for _state in states]
        ),
        hvector_by_layer=jnp.stack(
            [
                jnp.asarray(
                    setup.gas_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        ),
        hvector_cond_by_layer=jnp.stack(
            [
                jnp.asarray(
                    setup.condensate_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        ),
        ln_normalized_pressure_by_layer=jnp.stack(
            [
                _ln_normalized_pressure(float(pressure), Pref)
                for pressure in pressures
            ]
        ),
    )


def _head_v2_kkt_row(kkt_norms: Any, index: int) -> Mapping[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value))[index])
        for name, value in kkt_norms._asdict().items()
    }


def _head_v2_zero_barrier_initializer_kkt_passed(
    kkt: Mapping[str, float],
    *,
    stationarity_tolerance: float,
    budget_tolerance: float,
    complementarity_tolerance: float,
    total_density_tolerance: float,
) -> bool:
    """Return whether a finite-barrier state may initialize exact polish.

    Active-condensate stationarity is deliberately omitted: its finite-barrier
    residual contains the ``mu / m`` bias that the zero-barrier solve removes.
    The remaining components are only an initializer-quality gate and never a
    physical acceptance decision.
    """

    required = (
        ("gas_stationarity", stationarity_tolerance),
        ("budget_scaled", budget_tolerance),
        ("complementarity", complementarity_tolerance),
        ("total_density_scaled", total_density_tolerance),
    )
    return bool(
        all(
            math.isfinite(float(kkt[name]))
            and float(kkt[name]) <= float(tolerance)
            for name, tolerance in required
        )
    )


def _resolve_condensate_initial_guess(
    initializer: Optional[CondensateEquilibriumInitializer],
    request: CondensateEquilibriumInitRequest,
) -> CondensateEquilibriumInit:
    return resolve_condensate_initial_guess(initializer, request)


def _run_head_v2_profile(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: CondensateEquilibriumInitializer | None,
    support_indices: Sequence[int] | None,
    support_amounts_init: Sequence[float] | None,
    options: CondensateEquilibriumOptions,
    return_diagnostics: bool,
) -> CondensateEquilibriumProfileResult:
    """Run the production v2 route and its external support lifecycle."""

    from exogibbs.equilibrium.gas.solve import equilibrium
    from exogibbs.equilibrium.gas.types import EquilibriumOptions
    from exogibbs.condensates.fixed_support_payload import (
        seed_fixed_support_payload,
    )
    from exogibbs.equilibrium.condensate.policy import (
        fixed_support_v2_production_policy,
    )
    from exogibbs.equilibrium.condensate.fixed_support.batch import (
        run_fixed_support_profile,
    )
    from exogibbs.equilibrium.condensate.fixed_support.types import (
        TerminalStatus,
    )
    from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
        _physical_zero_barrier_audit,
        polish_zero_barrier_active_support,
    )

    policy = fixed_support_v2_production_policy(
        options.fixed_support_v2_preset
    )
    caller_inventory = jnp.asarray(b, dtype=jnp.float64)
    amount_scale = _inventory_amount_gauge_scale(setup, caller_inventory)
    log_amount_scale = math.log(amount_scale)
    b = caller_inventory / amount_scale
    caller_explicit_inits = tuple(explicit_inits)
    if support_amounts_init is not None:
        support_amounts_init = tuple(
            float(value) / amount_scale for value in support_amounts_init
        )
    epsilon_schedule = policy.solver_config.continuation.epsilon_schedule
    amount_gauge = {
        "schema": "exogibbs_condensate_amount_gauge_v1",
        "scale_basis": "sum_positive_non_charge_element_inventory",
        "caller_inventory_amount_scale": amount_scale,
        "log_caller_inventory_amount_scale": log_amount_scale,
        "internal_positive_inventory_sum": 1.0,
        "normalized_epsilon_schedule": epsilon_schedule,
        "caller_equivalent_epsilon_schedule": tuple(
            epsilon + log_amount_scale for epsilon in epsilon_schedule
        ),
        "internal_gauge": "normalized_element_inventory",
        "public_result_gauge": "caller_element_inventory",
    }
    n_layers = int(temperatures.shape[0])
    records: list[dict[str, Any]] = [
        {"layer_index": index, "rounds": []} for index in range(n_layers)
    ]
    pending: dict[int, _HeadV2LayerState] = {}
    gas_only_layers: set[int] = set()
    initial_guesses: list[CondensateEquilibriumInit] = []
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=caller_inventory,
                Pref=Pref,
                layer_index=layer_index,
                user_init=caller_explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        initial_guess = _normalize_condensate_init_amount_gauge(
            initial_guess,
            amount_scale,
        )
        initial_guesses.append(initial_guess)
        gas_equilibrium_init = _gas_init_from_condensate_init(
            initial_guess,
            gas_species_count=len(setup.gas_species),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                raise ValueError(
                    "head_v2 requires support_amounts_init with explicit "
                    "support_indices."
                )
            base_support = tuple(int(index) for index in support_indices)
            base_amounts = _positive_support_amounts_for_warm_start(
                support_amounts_init,
                min_seed_amount=policy.min_seed_amount,
            )
        else:
            payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=policy.min_seed_amount,
            )
            base_support = () if payload is None else payload[0]
            base_amounts = () if payload is None else payload[1]
        if len(base_support) != len(base_amounts):
            raise ValueError(
                "head_v2 support indices and initial amounts must have the "
                "same length."
            )
        if len(set(base_support)) != len(base_support):
            raise ValueError("head_v2 support indices must be unique.")
        if any(
            index < 0 or index >= len(setup.condensate_species)
            for index in base_support
        ):
            raise ValueError(
                "head_v2 support contains an out-of-range condensate index."
            )
        (
            initial_support,
            initial_amounts,
            support_trace,
        ) = _native_activity_expanded_profile_support_payload(
            setup=setup,
            T=float(temperatures[layer_index]),
            P=float(pressures[layer_index]),
            b=b,
            Pref=Pref,
            support_indices=base_support,
            support_policy="topk_capacity",
            support_topk=policy.initial_support_topk,
            max_support_count=policy.initial_support_limit,
            activity_threshold=0.0,
            activity_source_requested="gas_only_full_budget",
            seed_policy="budget_preserving_fraction",
            seed_fraction=policy.seed_fraction,
            max_seed_amount=policy.max_seed_amount,
            min_seed_amount=policy.min_seed_amount,
            activity_gas_ln_n=None,
            activity_gas_ntot=None,
            activity_gas_stationarity_source=None,
            gas_equilibrium_init=gas_equilibrium_init,
        )
        base_amount_by_index = {
            int(index): float(amount)
            for index, amount in zip(base_support, base_amounts)
        }
        initial_amounts = tuple(
            base_amount_by_index.get(int(index), float(amount))
            for index, amount in zip(initial_support, initial_amounts)
        )
        records[layer_index]["initial_support_indices"] = initial_support
        records[layer_index]["initial_support_count"] = len(initial_support)
        records[layer_index]["initial_support_selection"] = support_trace
        if not initial_support:
            records[layer_index]["outcome"] = "gas_only_no_candidate"
            gas_only_layers.add(layer_index)
            continue
        pending[layer_index] = _head_v2_initial_state(
            setup=setup,
            T=float(temperatures[layer_index]),
            P=float(pressures[layer_index]),
            b=b,
            Pref=Pref,
            support_indices=initial_support,
            support_amounts=initial_amounts,
            initial_guess=initial_guess,
            first_epsilon=policy.solver_config.continuation.epsilon_schedule[0],
        )

    def polish_layer_state(
        *,
        layer_index: int,
        gas_log_amounts: Array,
        condensate_amounts: Array,
        total_gas_log_amount: float,
        element_potential: Array,
        support: Sequence[int],
        valid_condensates: Sequence[bool],
    ) -> Any:
        temperature = float(temperatures[layer_index])
        return polish_zero_barrier_active_support(
            gas_formula_matrix=setup.formula_matrix,
            condensate_formula_matrix_full=setup.formula_matrix_cond,
            target_inventory=b,
            gas_standard_source=(
                setup.gas_setup.hvector_func(temperature)
                + _ln_normalized_pressure(
                    float(pressures[layer_index]), Pref
                )
            ),
            condensate_standard_source_full=(
                setup.condensate_setup.hvector_func(temperature)
            ),
            gas_log_amounts_init=gas_log_amounts,
            condensate_amounts_init=condensate_amounts,
            total_gas_log_amount_init=total_gas_log_amount,
            element_potential_init=element_potential,
            support_indices=support,
            condensate_valid_mask=valid_condensates,
            stationarity_tolerance=(
                policy.solver_config.normal.stationarity_tolerance
            ),
            budget_tolerance=policy.solver_config.normal.budget_tolerance,
            total_density_tolerance=(
                policy.solver_config.normal.total_density_tolerance
            ),
            support_closure_tolerance=policy.support_closure_tolerance,
            budget_relative_floor=policy.budget_relative_floor,
        )

    def audit_exact_in_caller_gauge(
        *,
        layer_index: int,
        exact: Any,
        valid_condensates: Sequence[bool],
    ) -> dict[str, Any]:
        temperature = float(temperatures[layer_index])
        return _physical_zero_barrier_audit(
            gas_formula_matrix=np.asarray(
                setup.formula_matrix, dtype=np.float64
            ),
            condensate_formula_matrix_full=np.asarray(
                setup.formula_matrix_cond, dtype=np.float64
            ),
            target_inventory=np.asarray(
                caller_inventory, dtype=np.float64
            ),
            gas_standard_source=np.asarray(
                setup.gas_setup.hvector_func(temperature)
                + _ln_normalized_pressure(
                    float(pressures[layer_index]), Pref
                ),
                dtype=np.float64,
            ),
            condensate_standard_source_full=np.asarray(
                setup.condensate_setup.hvector_func(temperature),
                dtype=np.float64,
            ),
            gas_log_amounts=(
                np.asarray(exact.gas_log_amounts, dtype=np.float64)
                + log_amount_scale
            ),
            condensate_amounts=(
                np.asarray(exact.condensate_amounts, dtype=np.float64)
                * amount_scale
            ),
            total_gas_log_amount=(
                float(exact.total_gas_log_amount) + log_amount_scale
            ),
            element_potential=np.asarray(
                exact.element_potential, dtype=np.float64
            ),
            support_indices=exact.support_indices,
            condensate_valid_mask=np.asarray(
                valid_condensates, dtype=bool
            ),
            budget_scale=_canonical_budget_scale_for_caller_audit(
                caller_inventory,
                amount_scale=amount_scale,
                relative_floor=policy.budget_relative_floor,
            ),
            budget_residual_amount_scale=amount_scale,
            optimizer_success=True,
            stationarity_tolerance=(
                policy.solver_config.normal.stationarity_tolerance
            ),
            budget_tolerance=policy.solver_config.normal.budget_tolerance,
            total_density_tolerance=(
                policy.solver_config.normal.total_density_tolerance
            ),
            support_closure_tolerance=policy.support_closure_tolerance,
        )

    def caller_audit_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: audit[key]
            for key in (
                "accepted",
                "finite",
                "positive_active_amounts",
                "gas_stationarity_max_abs",
                "active_condensate_driving_max_abs",
                "inactive_condensate_violation_max_abs",
                "budget_scaled_max_abs",
                "total_density_scaled_abs",
            )
        }

    last_outputs: dict[int, dict[str, Any]] = {}
    early_zero_barrier_results: dict[int, Any] = {}
    early_zero_barrier_caller_audits: dict[int, dict[str, Any]] = {}
    early_zero_barrier_provenance: dict[int, dict[str, Any]] = {}
    early_zero_barrier_attempted: set[int] = set()
    compilation_seconds = 0.0
    execution_seconds = 0.0
    diagnostic_seconds = 0.0
    backend = jax.default_backend()
    for round_index in range(policy.lifecycle_max_rounds):
        if not pending:
            break
        source_indices = tuple(sorted(pending))
        round_states = tuple(pending[index] for index in source_indices)
        round_temperatures = tuple(
            float(temperatures[index]) for index in source_indices
        )
        round_pressures = tuple(
            float(pressures[index]) for index in source_indices
        )
        buckets = _head_v2_prepared_buckets(
            setup=setup,
            temperatures=round_temperatures,
            pressures=round_pressures,
            b=b,
            Pref=Pref,
            states=round_states,
        )
        hcond_full = jnp.stack(
            [
                jnp.asarray(
                    setup.condensate_setup.hvector_func(temperature),
                    dtype=jnp.float64,
                )
                for temperature in round_temperatures
            ]
        )
        validity_upper = condensate_temperature_validity_upper(setup)
        if validity_upper is None:
            valid_mask = jnp.ones(
                (len(source_indices), len(setup.condensate_species)),
                dtype=bool,
            )
        else:
            upper = jnp.asarray(validity_upper, dtype=jnp.float64)
            if upper.shape != (len(setup.condensate_species),):
                raise ValueError(
                    "temperature_validity_upper must have one value per "
                    "condensate."
                )
            valid_mask = (
                jnp.asarray(round_temperatures, dtype=jnp.float64)[:, None]
                <= upper[None, :]
            )
        raw = run_fixed_support_profile(
            buckets=buckets,
            formula_matrix=setup.formula_matrix,
            layer_count=len(source_indices),
            condensate_count=len(setup.condensate_species),
            config=policy.solver_config,
            budget_relative_floor=policy.budget_relative_floor,
            include_terminal_diagnostics=return_diagnostics,
        )
        raw = evaluate_profile_support_closure(
            raw,
            formula_matrix=setup.formula_matrix,
            formula_matrix_cond_full=setup.formula_matrix_cond,
            condensate_standard_source_full=hcond_full,
            condensate_valid_mask=valid_mask,
            budget_relative_floor=policy.budget_relative_floor,
            support_closure_tolerance=policy.support_closure_tolerance,
        )
        compilation_seconds += float(raw["compilation_seconds"])
        execution_seconds += float(raw["execution_seconds"])
        diagnostic_seconds += float(raw["diagnostic_seconds"])
        backend = str(raw["backend"])
        converged = np.asarray(
            jax.device_get(raw["fixed_support_converged"]), dtype=bool
        )
        closed = np.asarray(jax.device_get(raw["support_closed"]), dtype=bool)
        expansion = np.asarray(
            jax.device_get(raw["support_expansion_mask"]), dtype=bool
        )
        driving = np.asarray(
            jax.device_get(raw["inactive_condensate_driving"]),
            dtype=np.float64,
        )
        terminal = np.asarray(
            jax.device_get(raw["terminal_status"]), dtype=np.int64
        )
        next_pending: dict[int, _HeadV2LayerState] = {}
        for local_index, source_index in enumerate(source_indices):
            current = pending[source_index]
            candidate_indices = np.flatnonzero(expansion[local_index])
            ordered = tuple(
                int(index)
                for index in candidate_indices[
                    np.argsort(
                        driving[local_index, candidate_indices],
                        kind="stable",
                    )
                ]
            )
            additions = ordered[: policy.support_add_per_round]
            remaining_capacity = max(
                0, policy.support_limit - len(current.support_indices)
            )
            additions = additions[:remaining_capacity]
            expanded_support = tuple(
                dict.fromkeys((*current.support_indices, *additions))
            )
            terminal_code = int(terminal[local_index])
            independent_kkt = _head_v2_kkt_row(
                raw["final_kkt_norms"], local_index
            )
            tolerances = policy.solver_config.normal
            independent_kkt_passed = _head_v2_kkt_passed(
                independent_kkt,
                stationarity_tolerance=tolerances.stationarity_tolerance,
                budget_tolerance=tolerances.budget_tolerance,
                complementarity_tolerance=(
                    tolerances.complementarity_tolerance
                ),
                total_density_tolerance=(
                    tolerances.total_density_tolerance
                ),
            )
            zero_barrier_initializer_kkt_passed = (
                _head_v2_zero_barrier_initializer_kkt_passed(
                    independent_kkt,
                    stationarity_tolerance=(
                        tolerances.stationarity_tolerance
                    ),
                    budget_tolerance=tolerances.budget_tolerance,
                    complementarity_tolerance=(
                        tolerances.complementarity_tolerance
                    ),
                    total_density_tolerance=(
                        tolerances.total_density_tolerance
                    ),
                )
            )
            final_state_values_finite = bool(
                np.asarray(
                    jax.device_get(
                        raw["final_state_values_finite"][local_index]
                    )
                )
            )
            round_record = {
                "round_index": round_index,
                "support_indices": current.support_indices,
                "support_count": len(current.support_indices),
                "fixed_support_converged": bool(converged[local_index]),
                "support_closed": bool(closed[local_index]),
                "terminal_status": terminal_code,
                "terminal_status_name": TerminalStatus(terminal_code).name,
                "positive_inactive_count": int(candidate_indices.size),
                "added_support_indices": additions,
                "independent_kkt": independent_kkt,
                "independent_kkt_passed": independent_kkt_passed,
                "zero_barrier_initializer_kkt_passed": (
                    zero_barrier_initializer_kkt_passed
                ),
                "final_state_values_finite": final_state_values_finite,
            }
            records[source_index]["rounds"].append(round_record)
            last_outputs[source_index] = {
                "raw": raw,
                "local_index": local_index,
                "round_index": round_index,
                "support_indices": current.support_indices,
                "fixed_support_converged": bool(converged[local_index]),
                "support_closed": bool(closed[local_index]),
                "terminal_status": terminal_code,
                "independent_kkt": independent_kkt,
                "independent_kkt_passed": independent_kkt_passed,
                "zero_barrier_initializer_kkt_passed": (
                    zero_barrier_initializer_kkt_passed
                ),
                "final_state_values_finite": final_state_values_finite,
            }
            early_exact_eligible = bool(
                current.support_indices
                and converged[local_index]
                and independent_kkt_passed
                and final_state_values_finite
                and not closed[local_index]
                and source_index not in early_zero_barrier_attempted
            )
            round_record["early_zero_barrier_eligible"] = (
                early_exact_eligible
            )
            if early_exact_eligible:
                # A converged central-path state is a reliable initializer,
                # but expanding its open support can create a rank-deficient
                # finite-barrier problem. Try the bounded, catalog-wide exact
                # closure once before changing that support. Its independent
                # physical audit remains the only acceptance authority.
                early_zero_barrier_attempted.add(source_index)
                full_amounts = np.asarray(
                    jax.device_get(
                        raw["condensate_amounts"][local_index]
                    ),
                    dtype=np.float64,
                )
                early_exact = polish_layer_state(
                    layer_index=source_index,
                    gas_log_amounts=np.asarray(
                        jax.device_get(
                            raw["gas_log_amounts"][local_index]
                        ),
                        dtype=np.float64,
                    ),
                    condensate_amounts=full_amounts,
                    total_gas_log_amount=float(
                        np.asarray(
                            jax.device_get(
                                raw["total_gas_log_amount"][local_index]
                            )
                        )
                    ),
                    element_potential=np.asarray(
                        jax.device_get(
                            raw["element_potential"][local_index]
                        ),
                        dtype=np.float64,
                    ),
                    support=current.support_indices,
                    valid_condensates=np.asarray(
                        jax.device_get(valid_mask[local_index]), dtype=bool
                    ),
                )
                provenance = {
                    "schema": (
                        "exogibbs_zero_barrier_initializer_provenance_v1"
                    ),
                    "eligible": True,
                    "attempted": True,
                    "role": "initializer_only",
                    "source": "open_converged_finite_support_state",
                    "source_round_index": round_index,
                    "lifecycle_terminal_round_index": round_index,
                    "selected_before_lifecycle_terminal_round": False,
                    "rescue_attempted": True,
                    "raw_fixed_support_converged": True,
                    "raw_support_closed": False,
                    "raw_independent_kkt_passed": True,
                    "raw_noncondensate_kkt_passed": (
                        zero_barrier_initializer_kkt_passed
                    ),
                    "raw_final_state_values_finite": True,
                    "raw_terminal_status": terminal_code,
                    "raw_terminal_status_name": TerminalStatus(
                        terminal_code
                    ).name,
                }
                round_record["early_zero_barrier_initializer"] = provenance
                round_record["early_zero_barrier_active_support_polish"] = (
                    early_exact.report
                )
                round_record["early_zero_barrier_internal_accepted"] = bool(
                    early_exact.accepted
                )
                if early_exact.accepted:
                    early_caller_audit = audit_exact_in_caller_gauge(
                        layer_index=source_index,
                        exact=early_exact,
                        valid_condensates=np.asarray(
                            jax.device_get(valid_mask[local_index]),
                            dtype=bool,
                        ),
                    )
                    round_record[
                        "early_caller_gauge_zero_barrier_kkt"
                    ] = caller_audit_summary(early_caller_audit)
                    if early_caller_audit["accepted"]:
                        early_zero_barrier_results[source_index] = early_exact
                        early_zero_barrier_caller_audits[source_index] = (
                            early_caller_audit
                        )
                        early_zero_barrier_provenance[source_index] = (
                            provenance
                        )
                        round_record["proposed_added_support_indices"] = (
                            additions
                        )
                        round_record["added_support_indices"] = ()
                        round_record[
                            "finite_support_expansion_skipped"
                        ] = True
                        round_record["early_zero_barrier_accepted"] = True
                        records[source_index]["outcome"] = (
                            "zero_barrier_open_support_rescued"
                        )
                        continue
                round_record["early_zero_barrier_accepted"] = False
            if not converged[local_index]:
                records[source_index]["outcome"] = "fixed_support_failed"
                continue
            if not independent_kkt_passed:
                records[source_index]["outcome"] = "independent_kkt_failed"
                continue
            if not final_state_values_finite:
                records[source_index]["outcome"] = "nonfinite_final_state"
                continue
            if closed[local_index]:
                records[source_index]["outcome"] = "closed"
                continue
            if not additions:
                records[source_index]["outcome"] = (
                    "open_at_support_or_round_limit"
                )
                continue
            seeded_support, seeded_amounts = seed_fixed_support_payload(
                setup=setup,
                element_inventory_target=b,
                support_indices=expanded_support,
                seed_fraction=policy.seed_fraction,
                max_seed_amount=policy.max_seed_amount,
                min_seed_amount=policy.min_seed_amount,
            )
            full_amounts = np.asarray(
                jax.device_get(raw["condensate_amounts"][local_index]),
                dtype=np.float64,
            )
            warm_amounts = tuple(
                float(full_amounts[index])
                if math.isfinite(float(full_amounts[index]))
                and float(full_amounts[index]) > 0.0
                else float(seed)
                for index, seed in zip(seeded_support, seeded_amounts)
            )
            next_pending[source_index] = _HeadV2LayerState(
                support_indices=seeded_support,
                gas_ln_n=jnp.asarray(
                    raw["gas_log_amounts"][local_index],
                    dtype=jnp.float64,
                ),
                condensate_log_amounts=jnp.log(
                    jnp.asarray(warm_amounts, dtype=jnp.float64)
                ),
                total_gas_log_amount=jnp.asarray(
                    raw["total_gas_log_amount"][local_index],
                    dtype=jnp.float64,
                ),
                element_potential=jnp.asarray(
                    raw["element_potential"][local_index],
                    dtype=jnp.float64,
                ),
            )
        pending = next_pending
    for source_index in pending:
        records[source_index]["outcome"] = "open_at_round_limit"

    layer_results: list[CondensateEquilibriumResult] = []
    for layer_index in range(n_layers):
        lifecycle_summary = {
            "schema": "exogibbs_head_v2_fixed_support_lifecycle_v1",
            "preset": policy.name,
            "outcome": records[layer_index].get(
                "outcome", "internal_missing_outcome"
            ),
            "rounds": tuple(records[layer_index]["rounds"]),
            "compilation_seconds_total": compilation_seconds,
            "execution_seconds_total": execution_seconds,
            "diagnostic_seconds_total": diagnostic_seconds,
            "backend": backend,
            "production_preset_promoted": True,
            "amount_gauge": amount_gauge,
        }
        if layer_index in gas_only_layers:
            gas_equilibrium_init = _gas_init_from_condensate_init(
                initial_guesses[layer_index],
                gas_species_count=len(setup.gas_species),
            )
            gas_result = equilibrium(
                setup.gas_setup,
                float(temperatures[layer_index]),
                float(pressures[layer_index]),
                jnp.asarray(b, dtype=jnp.float64),
                Pref=Pref,
                init=gas_equilibrium_init,
                options=EquilibriumOptions(),
                return_diagnostics=False,
            )
            result = _build_empty_support_gas_result(
                setup=setup,
                gas_ln_n=(
                    jnp.asarray(gas_result.ln_n, dtype=jnp.float64)
                    + log_amount_scale
                ),
                diagnostics={"fixed_support_v2": lifecycle_summary},
                route=HEAD_ROUTE_V2,
                selected_route="head_v2_gas_only_no_candidate",
                head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
                head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
                element_inventory_target=caller_inventory,
                enable_full_condensate_budget_residual_gate=(
                    options.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
                full_condensate_budget_relative_floor=(
                    options.full_condensate_budget_relative_floor
                ),
            )
        else:
            terminal_output = last_outputs[layer_index]
            raw = terminal_output["raw"]
            local_index = int(terminal_output["local_index"])
            support = terminal_output["support_indices"]
            full_amounts = jnp.asarray(
                raw["condensate_amounts"][local_index], dtype=jnp.float64
            )
            gas_log_amounts = jnp.asarray(
                raw["gas_log_amounts"][local_index], dtype=jnp.float64
            )
            total_gas_log_amount = float(
                np.asarray(
                    jax.device_get(
                        raw["total_gas_log_amount"][local_index]
                    )
                )
            )
            element_potential = np.asarray(
                jax.device_get(raw["element_potential"][local_index]),
                dtype=np.float64,
            )
            lifecycle_summary.update(
                {
                    "terminal_status": terminal_output["terminal_status"],
                    "terminal_status_name": TerminalStatus(
                        terminal_output["terminal_status"]
                    ).name,
                    "fixed_support_converged": terminal_output[
                        "fixed_support_converged"
                    ],
                    "support_closed": terminal_output["support_closed"],
                    "independent_kkt": terminal_output["independent_kkt"],
                    "independent_kkt_passed": terminal_output[
                        "independent_kkt_passed"
                    ],
                    "zero_barrier_initializer_kkt_passed": terminal_output[
                        "zero_barrier_initializer_kkt_passed"
                    ],
                    "final_state_values_finite": terminal_output[
                        "final_state_values_finite"
                    ],
                }
            )
            fixed_support_accepted = bool(
                terminal_output["fixed_support_converged"]
                and terminal_output["support_closed"]
                and terminal_output["independent_kkt_passed"]
                and terminal_output["final_state_values_finite"]
            )
            early_exact = early_zero_barrier_results.get(layer_index)
            exact_initializer_eligible = bool(
                early_exact is not None
                or (
                    support
                    and terminal_output["support_closed"]
                    and terminal_output[
                        "zero_barrier_initializer_kkt_passed"
                    ]
                    and terminal_output["final_state_values_finite"]
                )
            )
            rescue_attempted = bool(
                exact_initializer_eligible and not fixed_support_accepted
            )
            if early_exact is not None:
                initializer_report = dict(
                    early_zero_barrier_provenance[layer_index]
                )
            else:
                initializer_report = {
                    "schema": (
                        "exogibbs_zero_barrier_initializer_provenance_v1"
                    ),
                    "eligible": exact_initializer_eligible,
                    "attempted": exact_initializer_eligible,
                    "role": "initializer_only",
                    "source": "fixed_support_terminal_state",
                    "source_round_index": terminal_output["round_index"],
                    "lifecycle_terminal_round_index": terminal_output[
                        "round_index"
                    ],
                    "selected_before_lifecycle_terminal_round": False,
                    "rescue_attempted": rescue_attempted,
                    "raw_fixed_support_converged": terminal_output[
                        "fixed_support_converged"
                    ],
                    "raw_support_closed": terminal_output[
                        "support_closed"
                    ],
                    "raw_independent_kkt_passed": terminal_output[
                        "independent_kkt_passed"
                    ],
                    "raw_noncondensate_kkt_passed": terminal_output[
                        "zero_barrier_initializer_kkt_passed"
                    ],
                    "raw_final_state_values_finite": terminal_output[
                        "final_state_values_finite"
                    ],
                    "raw_terminal_status": terminal_output[
                        "terminal_status"
                    ],
                    "raw_terminal_status_name": TerminalStatus(
                        terminal_output["terminal_status"]
                    ).name,
                }
            lifecycle_summary["zero_barrier_initializer"] = (
                initializer_report
            )
            accepted = False
            if exact_initializer_eligible:
                temperature = float(temperatures[layer_index])
                upper = condensate_temperature_validity_upper(setup)
                valid_mask = (
                    np.ones(len(setup.condensate_species), dtype=bool)
                    if upper is None
                    else temperature <= np.asarray(upper, dtype=np.float64)
                )
                exact = early_exact
                if exact is None:
                    exact = polish_layer_state(
                        layer_index=layer_index,
                        gas_log_amounts=gas_log_amounts,
                        condensate_amounts=full_amounts,
                        total_gas_log_amount=total_gas_log_amount,
                        element_potential=element_potential,
                        support=support,
                        valid_condensates=valid_mask,
                    )
                lifecycle_summary[
                    "zero_barrier_active_support_polish"
                ] = exact.report
                accepted = bool(exact.accepted)
                if accepted:
                    caller_audit = (
                        early_zero_barrier_caller_audits[layer_index]
                        if early_exact is not None
                        else audit_exact_in_caller_gauge(
                            layer_index=layer_index,
                            exact=exact,
                            valid_condensates=valid_mask,
                        )
                    )
                    lifecycle_summary["caller_gauge_zero_barrier_kkt"] = (
                        caller_audit_summary(caller_audit)
                    )
                    accepted = bool(caller_audit["accepted"])
                    if not accepted:
                        lifecycle_summary["outcome"] = (
                            "caller_gauge_zero_barrier_kkt_failed"
                        )
                        records[layer_index]["outcome"] = (
                            "caller_gauge_zero_barrier_kkt_failed"
                        )
                    else:
                        support = exact.support_indices
                        gas_log_amounts = jnp.asarray(
                            exact.gas_log_amounts, dtype=jnp.float64
                        )
                        full_amounts = jnp.asarray(
                            exact.condensate_amounts, dtype=jnp.float64
                        )
                        total_gas_log_amount = float(
                            exact.total_gas_log_amount
                        )
                        element_potential = np.asarray(
                            exact.element_potential,
                            dtype=np.float64,
                        )
                        lifecycle_summary[
                            "support_indices_after_polish"
                        ] = support
                        if early_exact is not None:
                            lifecycle_summary["outcome"] = (
                                "zero_barrier_open_support_rescued"
                            )
                        elif rescue_attempted:
                            lifecycle_summary["outcome"] = (
                                "zero_barrier_active_support_rescued"
                            )
                else:
                    lifecycle_summary["outcome"] = (
                        "zero_barrier_active_support_polish_failed"
                    )
            caller_gas_log_amounts = gas_log_amounts + log_amount_scale
            caller_full_amounts = full_amounts * amount_scale
            support_amounts = caller_full_amounts[
                jnp.asarray(support, dtype=jnp.int32)
            ]
            result = build_condensate_equilibrium_result_from_solver_payload(
                setup=setup,
                gas_ln_n=caller_gas_log_amounts,
                support_indices=support,
                support_amounts=support_amounts,
                selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
                solver_success=accepted,
                route=HEAD_ROUTE_V2,
                head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
                head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
                diagnostics={"fixed_support_v2": lifecycle_summary},
                element_inventory_target=caller_inventory,
                enable_full_condensate_budget_residual_gate=(
                    options.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
                full_condensate_budget_relative_floor=(
                    options.full_condensate_budget_relative_floor
                ),
            )
        layer_results.append(result)

    gas_ln_n = jnp.stack([result.gas_ln_n for result in layer_results])
    gas_n = jnp.stack([result.gas_n for result in layer_results])
    gas_x = jnp.stack([result.gas_x for result in layer_results])
    gas_ntot = jnp.stack([result.gas_ntot for result in layer_results])
    condensate_amounts = jnp.stack(
        [result.condensate_amounts for result in layer_results]
    )
    profile_diagnostics = None
    if return_diagnostics:
        profile_diagnostics = {
            "profile_schema": "exogibbs_condensate_equilibrium_profile_v2",
            "route": HEAD_ROUTE_V2,
            "preset": policy.name,
            "method": "vmap_cold",
            "layer_count": n_layers,
            "compilation_seconds": compilation_seconds,
            "execution_seconds": execution_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "backend": backend,
            "layers": tuple(records),
            "amount_gauge": amount_gauge,
        }
    return CondensateEquilibriumProfileResult(
        layers=tuple(layer_results),
        method="vmap_cold",
        diagnostics=profile_diagnostics,
        batched_arrays={
            "gas_ln_n": gas_ln_n,
            "gas_n": gas_n,
            "gas_x": gas_x,
            "gas_ntot": gas_ntot,
            "condensate_amounts": condensate_amounts,
        },
    )


def _prepare_experimental_profile_fixed_support_batch_plan(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: Optional[CondensateEquilibriumInitializer],
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    max_iter: int,
    min_seed_amount: float,
) -> _ExperimentalProfileFixedSupportBatchPlan | None:
    n_layers = int(temperatures.shape[0])
    solver_inits = []
    support_by_layer = []
    states = []
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                return None
            support_payload = (
                tuple(int(index) for index in support_indices),
                _positive_support_amounts_for_warm_start(
                    support_amounts_init,
                    min_seed_amount=min_seed_amount,
                ),
            )
        else:
            support_payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=min_seed_amount,
            )
        if support_payload is None:
            return None
        layer_support_indices, layer_support_amounts = support_payload
        if len(layer_support_indices) == 0:
            return None
        solver_init = _solver_log_state_from_condensate_init(
            initial_guess,
            setup=setup,
            support_amounts_init=layer_support_amounts,
            source="exogibbs_experimental_profile_fixed_support_batch_plan",
        )
        if solver_init is None:
            return None
        solver_inits.append(solver_init)
        support_by_layer.append(layer_support_indices)
        states.append(
            ThermoState(
                temperature=float(temperatures[layer_index]),
                ln_normalized_pressure=_ln_normalized_pressure(
                    float(pressures[layer_index]),
                    Pref,
                ),
                element_vector=jnp.asarray(b, dtype=jnp.float64),
            )
        )
    from exogibbs.equilibrium.condensate.fixed_support.batch import (
        prepare_fixed_support_v2_buckets,
    )

    temperature_array = jnp.asarray(temperatures, dtype=jnp.float64)
    hvector_by_layer = jnp.asarray(
        setup.gas_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_by_layer.ndim != 2 or hvector_by_layer.shape[0] != n_layers:
        hvector_by_layer = jnp.stack(
            [
                jnp.asarray(
                    setup.gas_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        )
    hvector_cond_by_layer = jnp.asarray(
        setup.condensate_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_cond_by_layer.ndim != 2 or hvector_cond_by_layer.shape[0] != n_layers:
        hvector_cond_by_layer = jnp.stack(
            [
                jnp.asarray(
                    setup.condensate_setup.hvector_func(float(temperature)),
                    dtype=jnp.float64,
                )
                for temperature in temperatures
            ]
        )
    buckets = prepare_fixed_support_v2_buckets(
        init_states=tuple(solver_inits),
        support_indices_by_layer=tuple(support_by_layer),
        formula_matrix_cond=jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64),
        element_inventory_target_by_layer=jnp.stack(
            [jnp.asarray(state.element_vector, dtype=jnp.float64) for state in states]
        ),
        hvector_by_layer=hvector_by_layer,
        hvector_cond_by_layer=hvector_cond_by_layer,
        ln_normalized_pressure_by_layer=jnp.stack(
            [
                jnp.asarray(
                    state.ln_normalized_pressure,
                    dtype=jnp.float64,
                )
                for state in states
            ]
        ),
    )
    return _ExperimentalProfileFixedSupportBatchPlan(
        setup=setup,
        buckets=buckets,
        formula_matrix=jnp.asarray(setup.formula_matrix, dtype=jnp.float64),
        max_iter=max_iter,
        n_layers=n_layers,
        condensate_count=len(setup.condensate_species),
        bucket_layer_index_arrays=tuple(
            jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
            for bucket in buckets
        ),
        temperatures=temperature_array,
    )


def prepare_experimental_profile_fixed_support_batch_plan(
    setup: CondensateChemicalSetup,
    T: Sequence[float] | Array,
    P: Sequence[float] | Array,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[Sequence[CondensateEquilibriumInit | None]] = None,
    initializer: Optional[CondensateEquilibriumInitializer] = None,
    max_iter: int = 100,
    min_seed_amount: float = 1.0e-300,
) -> ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Prepare a reusable experimental fixed-support batch profile plan.

    The returned plan can be run repeatedly with
    :func:`run_experimental_profile_fixed_support_v2_batch_plan` to avoid
    rebuilding support buckets and thermochemical vectors for every call.
    """

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    if isinstance(max_iter, bool) or int(max_iter) <= 0:
        raise ValueError("max_iter must be positive.")
    if not math.isfinite(float(min_seed_amount)) or float(
        min_seed_amount
    ) <= 0.0:
        raise ValueError("min_seed_amount must be finite and positive.")
    n_layers = int(temperatures.shape[0])
    if init is None:
        explicit_inits: tuple[CondensateEquilibriumInit | None, ...] = (
            None,
        ) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    plan = _prepare_experimental_profile_fixed_support_batch_plan(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        b=b,
        Pref=Pref,
        explicit_inits=explicit_inits,
        initializer=initializer,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        max_iter=int(max_iter),
        min_seed_amount=float(min_seed_amount),
    )
    if plan is None:
        raise ValueError(
            "experimental fixed-support batch plan requires non-empty explicit "
            "support and solver log-state initialization for every layer."
        )
    return plan


def run_experimental_profile_fixed_support_v2_batch_plan(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    *,
    element_inventory_target: Optional[Array] = None,
    config: Optional[Any] = None,
    budget_relative_floor: float = 1.0e-6,
    support_closure_tolerance: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Run an existing prepared profile plan through fixed-support v2.

    This route is explicitly opt-in. It reports fixed-support convergence and
    full inactive-condensate support closure separately, and does not alter the
    production preset.
    """

    from exogibbs.equilibrium.condensate.fixed_support.batch import (
        run_fixed_support_profile,
    )
    from exogibbs.equilibrium.condensate.fixed_support.types import (
        FixedSupportV2Config,
    )

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if plan.temperatures is None:
        raise ValueError(
            "The prepared plan has no temperature array required by v2. "
            "Rebuild it with prepare_experimental_profile_fixed_support_batch_plan."
        )
    active_config = FixedSupportV2Config() if config is None else config
    if not isinstance(active_config, FixedSupportV2Config):
        raise TypeError("config must be a FixedSupportV2Config.")
    if config is None:
        # The integrated public v2 route includes the M2 restoration solver.
        # Component tests may still opt out explicitly with zero calls.
        active_config = replace(
            active_config,
            limits=replace(active_config.limits, max_restoration_calls=2),
        )
    active_config = replace(
        active_config,
        limits=replace(
            active_config.limits,
            max_normal_iterations=plan.max_iter,
        ),
    )

    buckets = plan.buckets
    if element_inventory_target is not None:
        target = jnp.asarray(element_inventory_target, dtype=jnp.float64)
        n_elements = int(plan.formula_matrix.shape[0])
        if target.ndim == 1:
            if target.shape[0] != n_elements:
                raise ValueError(
                    "element_inventory_target length must match elements."
                )
            target = jnp.broadcast_to(target, (plan.n_layers, n_elements))
        elif target.ndim == 2:
            if target.shape != (plan.n_layers, n_elements):
                raise ValueError(
                    "element_inventory_target must have shape "
                    f"({plan.n_layers}, {n_elements})."
                )
        else:
            raise ValueError(
                "element_inventory_target must be one- or two-dimensional."
            )
        buckets = tuple(
            replace(
                bucket,
                element_inventory_target=target[
                    jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
                ],
            )
            for bucket in plan.buckets
        )

    temperatures = jnp.asarray(plan.temperatures, dtype=jnp.float64)
    hcond_full = jnp.asarray(
        plan.setup.condensate_setup.hvector_func(temperatures),
        dtype=jnp.float64,
    )
    expected_shape = (plan.n_layers, plan.condensate_count)
    if hcond_full.shape != expected_shape:
        hcond_full = jax.vmap(plan.setup.condensate_setup.hvector_func)(
            temperatures
        )
    validity_upper = condensate_temperature_validity_upper(plan.setup)
    if validity_upper is None:
        condensate_valid_mask = jnp.ones(expected_shape, dtype=bool)
    else:
        upper = jnp.asarray(validity_upper, dtype=jnp.float64)
        if upper.shape != (plan.condensate_count,):
            raise ValueError(
                "temperature_validity_upper must have one value per condensate."
            )
        condensate_valid_mask = temperatures[:, None] <= upper[None, :]

    fixed_support_result = run_fixed_support_profile(
        buckets=buckets,
        formula_matrix=plan.formula_matrix,
        layer_count=plan.n_layers,
        condensate_count=plan.condensate_count,
        config=active_config,
        budget_relative_floor=budget_relative_floor,
    )
    return evaluate_profile_support_closure(
        fixed_support_result,
        formula_matrix=plan.formula_matrix,
        formula_matrix_cond_full=plan.setup.formula_matrix_cond,
        condensate_standard_source_full=hcond_full,
        condensate_valid_mask=condensate_valid_mask,
        budget_relative_floor=budget_relative_floor,
        support_closure_tolerance=support_closure_tolerance,
    )


__all__ = (
    "ExperimentalCondensateProfileFixedSupportBatchPlan",
    "prepare_experimental_profile_fixed_support_batch_plan",
    "run_experimental_profile_fixed_support_v2_batch_plan",
)
