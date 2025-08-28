"""
High-level equilibrium interface over the Gibbs minimizer.

This module provides a user-friendly API that stays loosely coupled to the
optimizer and data sources. Users only need a ChemicalSetup (A matrix and
an h(T) function). No JANAF or I/O details leak into this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union
import jax.numpy as jnp
from jax import tree_util
import jax
from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.optimize.minimize import minimize_gibbs

Array = jax.Array


@dataclass(frozen=True)
class EquilibriumOptions:
    """Solver options for equilibrium.

    Attributes:
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations.
    """

    epsilon_crit: float = 1.0e-11
    max_iter: int = 1000


@dataclass(frozen=True)
class EquilibriumInit:
    """Optional initial guess for the solver.

    Provide both fields to override the default uniform initialization.
    """

    ln_nk: Optional[Array] = None
    ln_ntot: Optional[Array] = None


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


def equilibrium(
    setup: ChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    init: Optional[EquilibriumInit] = None,
    options: Optional[EquilibriumOptions] = None,
) -> EquilibriumResult:
    """Compute equilibrium composition at (T, P, b) via Gibbs minimization.

    Args:
        setup: ChemicalSetup with formula matrix and hvector_func(T).
        T: Temperature (K).
        P: Pressure (bar).
        b: Elemental abundances; array of shape (E,).
        Pref: Reference pressure (bar) for normalization.
        init: Optional initial guess for ln n and ln n_tot.
        options: Solver options.

    Returns:
        EquilibriumResult with ln n, n, mole fractions x, and n_tot.
    """
    opts = options or EquilibriumOptions()
    A = setup.formula_matrix
    K = int(A.shape[1])

    # Validate b dimension and size
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
        epsilon_crit=opts.epsilon_crit,
        max_iter=opts.max_iter,
    )

    n = jnp.exp(ln_n)
    ntot = jnp.asarray(jnp.sum(n))
    x = n / jnp.clip(ntot, 1e-300)
    return EquilibriumResult(
        ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=None, metadata=None
    )

def equilibrium_profile(
    setup: ChemicalSetup,
    T: Array,
    P: Array,
    b: Array,
    *,
    Pref: float = 1.0,
    options: Optional[EquilibriumOptions] = None,
) -> EquilibriumResult:
    """Vectorized equilibrium along a 1D T/P profile (layers).

    This computes equilibrium independently for each (T[i], P[i]) pair while
    keeping the elemental abundances ``b`` fixed across layers.

    Args:
        setup: ChemicalSetup with formula matrix and hvector_func(T).
        T: Temperatures, shape (N,).
        P: Pressures, shape (N,).
        b: Elemental abundances, shape (E,), shared across layers.
        Pref: Reference pressure (bar).
        options: Solver options.

    Returns:
        Batched EquilibriumResult with fields stacked over the leading dimension N:
        - ln_n: (N, K)
        - n: (N, K)
        - x: (N, K)
        - ntot: (N,)
    """
    T = jnp.asarray(T)
    P = jnp.asarray(P)
    if T.ndim != 1 or P.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if T.shape[0] != P.shape[0]:
        raise ValueError("T and P must have the same length.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array shared across layers.")

    # Vectorize over T and P; keep setup and b static. Pass Pref/options as kwargs.
    layer_fn = jax.vmap(
        lambda Ti, Pi: equilibrium(
            setup,
            Ti,
            Pi,
            b,
            Pref=Pref,
            options=options,
        ),
        in_axes=(0, 0),
    )
    return layer_fn(T, P)


__all__ = [
    "equilibrium",
    "equilibrium_profile",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumResult",
]
