"""
High-level equilibrium interface over the Gibbs minimizer.

This module provides a user-friendly API that stays loosely coupled to the
optimizer and data sources. Users only need a ChemicalSetup (A matrix and
an h(T) function). No JANAF or I/O details leak into this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import tree_util

from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.optimize.minimize import minimize_gibbs, minimize_gibbs_core

Array = jnp.ndarray


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
    ln_ntot: Optional[float] = None


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
        children = (self.ln_n, self.n, self.x, jnp.asarray(self.ntot))
        aux = (self.iterations, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        iterations, metadata = aux_data
        ln_n, n, x, ntot = children
        return cls(ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=iterations, metadata=metadata)


def _make_b_vector(b: Union[Array, Mapping[str, float]], elems: Optional[Tuple[str, ...]]) -> Array:
    """Coerce an element-abundance input to a vector aligned to elems."""
    if isinstance(b, Mapping):
        if elems is None:
            raise ValueError("ChemicalSetup.elems is None; pass 'b' as an array instead of a dict.")
        return jnp.asarray([b.get(e, 0.0) for e in elems])
    arr = jnp.asarray(b)
    return arr


def _default_init(b_vec: Array, K: int) -> Tuple[Array, float]:
    """Numerically robust uniform initialization: n_k = 1 for all species."""
    ln_nk0 = jnp.zeros((K,), dtype=jnp.result_type(b_vec.dtype, jnp.float32))
    ln_ntot0 = jnp.log(jnp.asarray(K, dtype=jnp.result_type(b_vec.dtype, jnp.float32)))
    return ln_nk0, ln_ntot0


def _smart_init(formula_matrix: Array, b_vec: Array) -> Tuple[Array, float]:
    """Construct a physics-informed positive initial guess.

    Two-stage regularized initialization:
    1) Species-ridge LS: (A^T A + λI) n = A^T b + λ n_base, with n_base=1
       to keep all species away from zero and stabilize A diag(n) A^T.
    2) Positivity clip and fallback to element-space LS if needed.
    """
    b = jnp.asarray(b_vec)
    A = jnp.asarray(formula_matrix, dtype=b.dtype)
    E, K = A.shape
    AT = A.T

    # Stage 1: species-ridge LS around ones
    lam = jnp.asarray(1e-2, dtype=A.dtype)
    I_K = jnp.eye(K, dtype=A.dtype)
    n_base = jnp.ones((K,), dtype=A.dtype)
    lhs = AT @ A + lam * I_K
    rhs = AT @ b + lam * n_base
    n0 = jnp.linalg.solve(lhs, rhs)

    # Ensure strictly positive, with a modest floor
    n0 = jnp.clip(n0, 1e-12)

    # If sum is pathological, refine with element-space LS (regularized)
    # Regularize AA^T to ensure stability
    AA = A @ A.T
    regI = jnp.eye(E, dtype=AA.dtype) * jnp.asarray(1e-12, dtype=AA.dtype)
    y = jnp.linalg.solve(AA + regI, b)
    n_ls = AT @ y
    n_ls = jnp.clip(n_ls, 1e-12)

    # Blend to avoid extremes: average of species- and element-regularized guesses
    n0 = 0.5 * (n0 + n_ls)

    ntot0 = jnp.sum(n0)
    ln_nk0 = jnp.log(n0)
    ln_ntot0 = jnp.log(jnp.clip(ntot0, 1e-300))
    # Slight perturbation to avoid resn==0 making the block system singular at k=0
    ln_ntot0 = ln_ntot0 + jnp.asarray(1e-3, dtype=ln_ntot0.dtype)
    return ln_nk0, ln_ntot0


def _prepare_init(
    init: Optional[EquilibriumInit], b_vec: Array, K: int, formula_matrix: Array
) -> Tuple[Array, float]:
    if init is not None and init.ln_nk is not None and init.ln_ntot is not None:
        return jnp.asarray(init.ln_nk), jnp.asarray(init.ln_ntot)
    # Try physics-informed init, fall back to robust uniform
    try:
        return _smart_init(formula_matrix, b_vec)
    except Exception:
        return _default_init(b_vec, K)


def _ln_normalized_pressure(P: float, Pref: float) -> float:
    return jnp.log(P / Pref)


def equilibrium(
    setup: ChemicalSetup,
    T: float,
    P: float,
    b: Union[Array, Mapping[str, float]],
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
        b: Elemental abundances; array of shape (E,) or dict {elem: value}.
        Pref: Reference pressure (bar) for normalization.
        init: Optional initial guess for ln n and ln n_tot.
        options: Solver options.

    Returns:
        EquilibriumResult with ln n, n, mole fractions x, and n_tot.
    """
    opts = options or EquilibriumOptions()
    A = setup.formula_matrix
    K = int(A.shape[1])
    b_vec = _make_b_vector(b, setup.elems)

    # Validate b dimension if available
    if b_vec.ndim != 1:
        raise ValueError("b must be a 1D array or dict mapping element names to values")

    lnP = _ln_normalized_pressure(P, Pref)
    ln_nk0, ln_ntot0 = _prepare_init(init, b_vec, K, A)

    # Wrap h(T) to sanitize any non-finite values from interpolation
    def hfunc_clean(t):
        hv = setup.hvector_func(t)
        return jnp.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)

    state = ThermoState(T, lnP, b_vec)
    ln_n = minimize_gibbs(
        state,
        ln_nk0,
        ln_ntot0,
        A,
        hfunc_clean,
        epsilon_crit=opts.epsilon_crit,
        max_iter=opts.max_iter,
    )

    n = jnp.exp(ln_n)
    ntot = jnp.sum(n)
    x = n / jnp.clip(ntot, 1e-300)
    return EquilibriumResult(ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=None, metadata={"converged": True})


def equilibrium_diagnostics(
    setup: ChemicalSetup,
    T: float,
    P: float,
    b: Union[Array, Mapping[str, float]],
    *,
    Pref: float = 1.0,
    init: Optional[EquilibriumInit] = None,
    options: Optional[EquilibriumOptions] = None,
) -> EquilibriumResult:
    """Like equilibrium(), but returns iteration count and ln n_tot from core."""
    opts = options or EquilibriumOptions()
    A = setup.formula_matrix
    K = int(A.shape[1])
    b_vec = _make_b_vector(b, setup.elems)

    if b_vec.ndim != 1:
        raise ValueError("b must be a 1D array or dict mapping element names to values")

    lnP = _ln_normalized_pressure(P, Pref)
    ln_nk0, ln_ntot0 = _prepare_init(init, b_vec, K, A)

    # Wrap h(T) to sanitize any non-finite values from interpolation
    def hfunc_clean(t):
        hv = setup.hvector_func(t)
        return jnp.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)

    state = ThermoState(T, lnP, b_vec)
    ln_n, ln_ntot, iters = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        A,
        hfunc_clean,
        epsilon_crit=opts.epsilon_crit,
        max_iter=opts.max_iter,
    )

    n = jnp.exp(ln_n)
    x = n / jnp.clip(jnp.sum(n), 1e-300)
    return EquilibriumResult(
        ln_n=ln_n,
        n=n,
        x=x,
        ntot=jnp.exp(ln_ntot),
        iterations=int(jax.device_get(iters)),
        metadata={"converged": True},
    )


# Optional: a batched interface. Note that returning a dataclass requires pytree registration
# which we implemented above on EquilibriumResult. Users can also vmap over inner fields.
equilibrium_map = jax.vmap(
    equilibrium,
    in_axes=(None, 0, 0, 0, None, None, None),
)
