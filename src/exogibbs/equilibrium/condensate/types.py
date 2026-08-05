"""Public and lifecycle state types for condensate equilibrium."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable

import jax

from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup


Array = jax.Array
DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR = 1.0e-6
FIXED_SUPPORT_V2_VALIDATED_PRESET = "validated_2026_07"
CondensateRoute = Literal["head_v2"]
CondensateFixedSupportV2Preset = Literal["validated_2026_07"]
CondensateProfileMethod = Literal[
    "auto",
    "vmap_cold",
    "scan_hot_from_bottom",
]
CondensateProfileNativeActivitySupportPolicy = Literal[
    "topk_capacity",
    "fastchem_activity_all",
]
CondensateProfileFixedSupportSeedPolicy = Literal[
    "budget_preserving_fraction",
    "max_density",
]
CondensateProfileNativeActivitySource = Literal[
    "gas_only_full_budget",
    "initializer_gas",
]
CONDENSATE_HEAD_V2_ROUTE_VERSION = "v2.0"
CONDENSATE_HEAD_V2_ROUTE_NAME = "head_v2_fixed_support_lifecycle"
HEAD_ROUTE_V2 = "head_v2"
CONVERGED = "converged"
NOT_CONVERGED = "not_converged"


@dataclass(frozen=True)
class CondensateEquilibriumOptions:
    """Options for the production fixed-support v2 route."""

    route: CondensateRoute = HEAD_ROUTE_V2
    fixed_support_v2_preset: CondensateFixedSupportV2Preset = (
        FIXED_SUPPORT_V2_VALIDATED_PRESET
    )
    profile_method: Optional[CondensateProfileMethod] = None
    return_diagnostics: bool = False
    enable_full_condensate_budget_residual_gate: bool = True
    full_condensate_budget_relative_tolerance: float = 1.0e-3
    full_condensate_budget_relative_floor: float = (
        DEFAULT_FULL_CONDENSATE_BUDGET_RELATIVE_FLOOR
    )
    rainout: bool = False


@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Result container for the production fixed-support v2 route."""

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
    head_route_version: str = CONDENSATE_HEAD_V2_ROUTE_VERSION
    head_route_name: str = CONDENSATE_HEAD_V2_ROUTE_NAME


@dataclass(frozen=True)
class AcceptedCondensateState:
    """Numerical state after all configured post-solve acceptance steps."""

    gas_ln_n: Array
    gas_n: Array
    gas_x: Array
    gas_ntot: Array
    condensate_amounts: Array
    status: str
    acceptance_tier: str
    warning_messages: tuple[str, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CondensateEquilibriumInit:
    """Optional condensate profile initial guess for one layer."""

    gas_ln_n: Optional[Array] = None
    gas_ntot: Optional[Array] = None
    condensate_amounts: Optional[Array] = None
    support_indices: Optional[Sequence[int]] = None
    support_amounts: Optional[Sequence[float]] = None
    element_potential: Optional[Array] = None
    rho: Optional[Array] = None
    barrier_epsilon: Optional[Array] = None


@dataclass(frozen=True)
class CondensateEquilibriumInitRequest:
    """Inputs available to a condensate profile initializer for one layer."""

    setup: CondensateChemicalSetup
    T: float
    P: float
    b: Array
    Pref: float = 1.0
    layer_index: Optional[int] = None
    user_init: Optional[CondensateEquilibriumInit] = None
    previous_solution: Optional[CondensateEquilibriumInit] = None


@runtime_checkable
class CondensateEquilibriumInitializer(Protocol):
    """Produce an optional condensate initial guess for one profile layer."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        ...


@dataclass(frozen=True)
class CondensateEquilibriumProfileResult:
    """Result container for a Python-level condensate profile solve."""

    layers: tuple[CondensateEquilibriumResult, ...]
    method: CondensateProfileMethod
    diagnostics: Optional[Mapping[str, Any]] = None
    batched_arrays: Optional[Mapping[str, Array]] = None
    rainout: bool = False
    element_inventory_target: Optional[Array] = None
    gas_element_inventory: Optional[Array] = None
    rainout_element_inventory_out: Optional[Array] = None
    rainout_abundance_scale: Optional[Array] = None


@dataclass(frozen=True)
class ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Reusable experimental fixed-support profile plan."""

    setup: CondensateChemicalSetup
    buckets: Sequence[Any]
    formula_matrix: Array
    max_iter: int
    n_layers: int
    condensate_count: int
    bucket_layer_index_arrays: tuple[Array, ...] = ()
    temperatures: Optional[Array] = None


@dataclass(frozen=True)
class HeadV2LayerState:
    """One host-owned fixed-support state between outer lifecycle rounds."""

    support_indices: tuple[int, ...]
    gas_ln_n: Array
    condensate_log_amounts: Array
    total_gas_log_amount: Array
    element_potential: Array


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
    "ExperimentalCondensateProfileFixedSupportBatchPlan",
)
