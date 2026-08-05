"""One-layer and profile application service for condensate equilibrium."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate import acceptance as _acceptance
from exogibbs.equilibrium.condensate.initialization import (
    DefaultCondensateEquilibriumInitializer,
)
from exogibbs.equilibrium.condensate.policy import (
    validate_condensate_options as _validate_options,
)
from exogibbs.equilibrium.condensate.profile import run_rainout_profile
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
    validate_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.types import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
    FIXED_SUPPORT_V2_VALIDATED_PRESET,
    HEAD_ROUTE_V2,
    Array,
    CondensateEquilibriumInit,
    CondensateEquilibriumInitializer,
    CondensateEquilibriumInitRequest,
    CondensateEquilibriumOptions,
    CondensateEquilibriumProfileResult,
    CondensateEquilibriumResult,
    CondensateFixedSupportV2Preset,
    CondensateProfileMethod,
    ExperimentalCondensateProfileFixedSupportBatchPlan,
)


# Supported compatibility helpers remain reachable from the historical API.
_head_v2_kkt_passed = _acceptance.independent_kkt_passed
_apply_full_condensate_budget_residual_gate = (
    _acceptance.apply_full_condensate_budget_residual_gate
)
_full_condensate_element_budget_residual_report = (
    _acceptance.full_condensate_element_budget_residual_report
)
_polish_gas_log_amounts_for_full_condensate_budget_gate = (
    _acceptance.polish_gas_log_amounts_for_full_condensate_budget_gate
)
build_condensate_equilibrium_result_from_solver_payload = (
    _lifecycle.build_condensate_equilibrium_result_from_solver_payload
)
_build_empty_support_gas_result = (
    _lifecycle._build_empty_support_gas_result
)
_run_head_v2_profile = _lifecycle._run_head_v2_profile
prepare_experimental_profile_fixed_support_batch_plan = (
    _lifecycle.prepare_experimental_profile_fixed_support_batch_plan
)
run_experimental_profile_fixed_support_v2_batch_plan = (
    _lifecycle.run_experimental_profile_fixed_support_v2_batch_plan
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
    init: Optional[CondensateEquilibriumInit] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
) -> CondensateEquilibriumResult:
    """Compute one layer through the production fixed-support v2 route."""

    opts = options or CondensateEquilibriumOptions()
    validate_condensate_chemical_setup(setup)
    _validate_options(opts)
    if opts.rainout:
        raise ValueError(
            "rainout=True is a dependent profile operation; use "
            "condensate_equilibrium_profile instead of the one-layer solver."
        )
    profile = _run_head_v2_profile(
        setup=setup,
        temperatures=np.asarray([T], dtype=np.float64),
        pressures=np.asarray([P], dtype=np.float64),
        b=b,
        Pref=Pref,
        explicit_inits=(init,),
        initializer=None,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=opts,
        return_diagnostics=opts.return_diagnostics,
    )
    return profile.layers[0]


def condensate_equilibrium_profile(
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
    options: Optional[CondensateEquilibriumOptions] = None,
    method: Optional[CondensateProfileMethod] = None,
    return_diagnostics: bool = False,
) -> CondensateEquilibriumProfileResult:
    """Compute a 1D profile through the production v2 lifecycle."""

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    if temperatures.shape[0] == 0:
        raise ValueError("T and P must contain at least one profile layer.")
    opts = options or CondensateEquilibriumOptions()
    _validate_options(opts, profile_method=method)
    n_layers = int(temperatures.shape[0])
    if init is None:
        explicit_inits: tuple[CondensateEquilibriumInit | None, ...] = (
            None,
        ) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    requested_method = method if method is not None else opts.profile_method
    if opts.rainout:
        if requested_method not in {
            None,
            "auto",
            "scan_hot_from_bottom",
        }:
            raise ValueError(
                "rainout=True requires profile method 'auto' or "
                "'scan_hot_from_bottom'."
            )
        return run_rainout_profile(
            setup=setup,
            temperatures=temperatures,
            pressures=pressures,
            b=b,
            Pref=Pref,
            explicit_inits=explicit_inits,
            initializer=initializer,
            support_indices=support_indices,
            support_amounts_init=support_amounts_init,
            options=opts,
            return_diagnostics=(
                return_diagnostics or opts.return_diagnostics
            ),
        )
    if requested_method not in {None, "auto", "vmap_cold"}:
        raise ValueError(
            "head_v2 currently supports profile method 'auto' or "
            "'vmap_cold' when rainout=False. "
            "'scan_hot_from_bottom' is reserved for rainout=True."
        )
    return _run_head_v2_profile(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        b=b,
        Pref=Pref,
        explicit_inits=explicit_inits,
        initializer=initializer,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        options=opts,
        return_diagnostics=return_diagnostics or opts.return_diagnostics,
    )


solve = condensate_equilibrium
solve_profile = condensate_equilibrium_profile


__all__ = (
    "CONDENSATE_HEAD_V2_ROUTE_NAME",
    "CONDENSATE_HEAD_V2_ROUTE_VERSION",
    "FIXED_SUPPORT_V2_VALIDATED_PRESET",
    "HEAD_ROUTE_V2",
    "CondensateChemicalSetup",
    "CondensateEquilibriumInit",
    "CondensateEquilibriumInitRequest",
    "CondensateEquilibriumInitializer",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumProfileResult",
    "CondensateEquilibriumResult",
    "CondensateFixedSupportV2Preset",
    "CondensateProfileMethod",
    "DefaultCondensateEquilibriumInitializer",
    "ExperimentalCondensateProfileFixedSupportBatchPlan",
    "build_condensate_chemical_setup",
    "build_condensate_equilibrium_result_from_solver_payload",
    "condensate_equilibrium",
    "condensate_equilibrium_profile",
    "prepare_experimental_profile_fixed_support_batch_plan",
    "run_experimental_profile_fixed_support_v2_batch_plan",
    "solve",
    "solve_profile",
    "validate_condensate_chemical_setup",
)
