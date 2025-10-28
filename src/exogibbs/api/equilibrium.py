"""
High-level equilibrium interface over the Gibbs minimizer.

This module provides a user-friendly API that stays loosely coupled to the
optimizer and data sources. Users only need a ChemicalSetup (A matrix and
an h(T) function). No JANAF or I/O details leak into this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Tuple, Union
import jax.numpy as jnp
from jax import tree_util
import jax
from exogibbs.api.chemistry import ChemicalSetup, ChemicalSetupCond, ThermoState
from exogibbs.optimize.minimize import minimize_gibbs

Array = jax.Array


@dataclass(frozen=True)
class EquilibriumOptions:
    """Solver options for equilibrium.

    Attributes:
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations.
        enable_condensed: Enable condensed species handling when present.
        nonneg_param: Strategy for keeping condensed amounts non-negative.
        barrier_init: Initial barrier parameter for interior-style updates.
        kkt_tol: Complementarity tolerance for condensed KKT checks.
        max_active_set_iter: Max iterations for active-set switching.

    Note:
        these default values are chosen based on the comparison with FastChem 
        in the range of 300-3000K and 1e-8 - 1e2 bar. See #17 and comparison_with_fastchem.py
    """

    epsilon_crit: float = 1.0e-15
    max_iter: int = 1000
    enable_condensed: bool = True
    nonneg_param: Literal["softplus", "projection", "logfloor"] = "softplus"
    barrier_init: float = 1.0e-3
    kkt_tol: float = 1.0e-12
    max_active_set_iter: int = 10


@dataclass(frozen=True)
class EquilibriumInit:
    """Optional initial guess for the solver.

    Provide both fields to override the default uniform initialization.
    """

    ln_nk: Optional[Array] = None
    ln_ntot: Optional[Array] = None


@dataclass(frozen=True)
class EquilibriumInitCond:
    """Optional initial guesses for condensed equilibrium runs.

    Only the gas initialization is used by the placeholder implementation.
    """

    gas: Optional[EquilibriumInit] = None
    condensed: Optional[Array] = None


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EquilibriumResult:
    """Result container for equilibrium composition.

    Fields are JAX arrays to support downstream transforms.
    """

    ln_n: Array  # (K,)
    n: Array  # (K,)
    x: Array  # (K,)
    ntot: Array  # scalar array to remain JAX-friendly
    iterations: Optional[int] = None
    metadata: Optional[Mapping[str, Union[bool, float, int]]] = None

    # Make this dataclass a JAX pytree (so vmap/jit can pass it around)
    def tree_flatten(self):
        # Avoid coercing to jnp.asarray here to keep compatibility with
        # transformation-time abstract values (e.g., vmap/jit tracing).
        children = (self.ln_n, self.n, self.x, self.ntot)
        aux = (self.iterations, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        iterations, metadata = aux_data
        ln_n, n, x, ntot = children
        return cls(
            ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=iterations, metadata=metadata
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EquilibriumResultCond:
    """Placeholder container for condensate-aware results."""

    gas: EquilibriumResult
    condensed_ln_n: Array
    condensed_n: Array
    condensed_species: Tuple[str, ...]
    status: str
    metadata: Optional[Mapping[str, Union[bool, float, int, str]]] = None

    def tree_flatten(self):
        children = (self.gas, self.condensed_ln_n, self.condensed_n)
        aux = (self.condensed_species, self.status, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        condensed_species, status, metadata = aux_data
        gas, condensed_ln_n, condensed_n = children
        return cls(
            gas=gas,
            condensed_ln_n=condensed_ln_n,
            condensed_n=condensed_n,
            condensed_species=condensed_species,
            status=status,
            metadata=metadata,
        )


def _default_init(b_vec: Array, K: int) -> Tuple[Array, float]:
    """Numerically robust uniform initialization: n_k = 1 for all species."""
    ln_nk0 = jnp.zeros((K,), dtype=jnp.result_type(b_vec.dtype, jnp.float32))
    ln_ntot0 = jnp.log(jnp.asarray(K, dtype=jnp.result_type(b_vec.dtype, jnp.float32)))
    return ln_nk0, ln_ntot0


def _prepare_init(
    init: Optional[EquilibriumInit], b_vec: Array, K: int
) -> Tuple[Array, float]:
    if init is not None and init.ln_nk is not None and init.ln_ntot is not None:
        return jnp.asarray(init.ln_nk), jnp.asarray(init.ln_ntot)
    return _default_init(b_vec, K)


def _ln_normalized_pressure(P: float, Pref: float) -> float:
    return jnp.log(P / Pref)


def _extract_gas_init(
    init: Optional[Union[EquilibriumInit, EquilibriumInitCond]]
) -> Optional[EquilibriumInit]:
    if init is None:
        return None
    if isinstance(init, EquilibriumInit):
        return init
    if isinstance(init, EquilibriumInitCond):
        return init.gas
    raise TypeError("Unsupported init type for equilibrium.")


def _equilibrium_gas(
    setup: ChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    init: Optional[EquilibriumInit] = None,
    options: EquilibriumOptions,
) -> EquilibriumResult:
    A = setup.formula_matrix
    K = int(A.shape[1])

    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b has length {b.shape[0]} but A expects {A.shape[0]} elements.")

    lnP = _ln_normalized_pressure(P, Pref)
    ln_nk0, ln_ntot0 = _prepare_init(init, b, K)

    hfunc = setup.hvector_func
    state = ThermoState(T, lnP, b)
    ln_n = minimize_gibbs(
        state,
        ln_nk0,
        ln_ntot0,
        A,
        hfunc,
        epsilon_crit=options.epsilon_crit,
        max_iter=options.max_iter,
    )

    n = jnp.exp(ln_n)
    ntot = jnp.asarray(jnp.sum(n))
    x = n / jnp.clip(ntot, 1e-300)
    return EquilibriumResult(
        ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=None, metadata=None
    )


def _condensed_placeholder_result(
    gas_result: EquilibriumResult,
    setup: ChemicalSetupCond,
) -> EquilibriumResultCond:
    cond_indices = tuple(setup.phase_registry.cond_indices)
    cond_count = len(cond_indices)
    if cond_count == 0:
        condensed_ln_n = jnp.zeros((0,), dtype=gas_result.ln_n.dtype)
        condensed_n = jnp.zeros((0,), dtype=gas_result.n.dtype)
        condensed_names: Tuple[str, ...] = tuple()
    else:
        condensed_ln_n = jnp.full((cond_count,), -jnp.inf, dtype=gas_result.ln_n.dtype)
        condensed_n = jnp.zeros((cond_count,), dtype=gas_result.n.dtype)
        condensed_names = tuple(setup.species[idx].name for idx in cond_indices)
    metadata = {
        "condensed_stub": True,
        "condensed_species_count": cond_count,
    }
    return EquilibriumResultCond(
        gas=gas_result,
        condensed_ln_n=condensed_ln_n,
        condensed_n=condensed_n,
        condensed_species=condensed_names,
        status="condensed_not_implemented",
        metadata=metadata,
    )


def _equilibrium_cond_placeholder(
    setup: ChemicalSetupCond,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float,
    init: Optional[EquilibriumInit],
    options: EquilibriumOptions,
) -> EquilibriumResultCond:
    if not options.enable_condensed and setup.phase_registry.cond_indices:
        raise ValueError("enable_condensed=False but condensed species are registered.")
    gas_result = _equilibrium_gas(
        setup.gas_setup,
        T,
        P,
        b,
        Pref=Pref,
        init=init,
        options=options,
    )
    return _condensed_placeholder_result(gas_result, setup)


def equilibrium(
    setup: Union[ChemicalSetup, ChemicalSetupCond],
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    init: Optional[Union[EquilibriumInit, EquilibriumInitCond]] = None,
    options: Optional[EquilibriumOptions] = None,
) -> Union[EquilibriumResult, EquilibriumResultCond]:
    opts = options or EquilibriumOptions()
    gas_init = _extract_gas_init(init)
    if isinstance(setup, ChemicalSetupCond):
        return _equilibrium_cond_placeholder(
            setup,
            T,
            P,
            b,
            Pref=Pref,
            init=gas_init,
            options=opts,
        )
    if isinstance(setup, ChemicalSetup):
        return _equilibrium_gas(
            setup,
            T,
            P,
            b,
            Pref=Pref,
            init=gas_init,
            options=opts,
        )
    raise TypeError("Unsupported setup type for equilibrium.")


def equilibrium_profile(
    setup: ChemicalSetup,
    T: Array,
    P: Array,
    b: Array,
    *,
    Pref: float = 1.0,
    options: Optional[EquilibriumOptions] = None,
) -> EquilibriumResult:
    if isinstance(setup, ChemicalSetupCond):
        raise NotImplementedError("Condensed equilibrium profiles are not implemented yet.")

    T = jnp.asarray(T)
    P = jnp.asarray(P)
    if T.ndim != 1 or P.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if T.shape[0] != P.shape[0]:
        raise ValueError("T and P must have the same length.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array shared across layers.")

    layer_fn = jax.vmap(
        lambda Ti, Pi: equilibrium(
            setup,
            Ti,
            Pi,
            b,
            Pref=Pref,
            init=None,
            options=options,
        ),
        in_axes=(0, 0),
    )
    batched = layer_fn(T, P)
    return EquilibriumResult(
        ln_n=batched.ln_n,
        n=batched.n,
        x=batched.x,
        ntot=batched.ntot,
        iterations=None,
        metadata=batched.metadata,
    )


__all__ = [
    "equilibrium",
    "equilibrium_profile",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumInitCond",
    "EquilibriumResult",
    "EquilibriumResultCond",
]
