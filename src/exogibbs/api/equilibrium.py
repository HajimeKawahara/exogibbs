"""
High-level equilibrium interface over the Gibbs minimizer.

This module provides a user-friendly API that stays loosely coupled to the
optimizer and data sources. Users only need a ChemicalSetup (A matrix and
an h(T) function). No JANAF or I/O details leak into this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable
import jax.numpy as jnp
from jax import tree_util
import jax
from jax import lax
from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.optimize.minimize import minimize_gibbs, minimize_gibbs_with_diagnostics

Array = jax.Array

if TYPE_CHECKING:
    from exogibbs.api.equilibrium_grid import EquilibriumGrid


_PROFILE_SCAN_BODY_CACHE: Dict[Tuple[int, int, float, int, int, bool], Callable] = {}


@dataclass(frozen=True)
class EquilibriumOptions:
    """Solver options for equilibrium.

    Attributes:
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations.
        method: Method for solving equilibrium along a profile. Options:
            - "vmap_cold": Independent solves for each layer with cold starts (no state carryover).
            - "scan_hot_from_top": Sequential solves from top to bottom, carrying converged state as hot start for next layer.
            - "scan_hot_from_bottom": Sequential solves from bottom to top, carrying converged state as hot start for next layer.
    Note:
        these default values are chosen based on the comparison with FastChem 
        in the range of 300-3000K and 1e-8 - 1e2 bar. See #17 and comparison_with_fastchem.py
    """

    epsilon_crit: float = 1.0e-10
    max_iter: int = 1000
    method: Literal["vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom"] = "scan_hot_from_top"


@dataclass(frozen=True)
class EquilibriumInit:
    """Optional initial guess for the solver.

    Provide both fields to override the default uniform initialization.
    """

    ln_nk: Optional[Array] = None
    ln_ntot: Optional[Array] = None


@dataclass(frozen=True)
class EquilibriumInitRequest:
    """Inputs available to an initializer for one layer's Newton guess.

    This request always describes a single layer/state. In profile solves,
    ``T`` and ``P`` are the per-layer values for the current layer, while
    ``b`` is currently shared across all layers in the profile.
    """

    setup: ChemicalSetup
    T: float
    P: float
    b: Array
    K: int
    user_init: Optional[EquilibriumInit] = None
    previous_solution: Optional[EquilibriumInit] = None


@runtime_checkable
class EquilibriumInitializer(Protocol):
    """Produce an initial guess for one layer's Newton solve."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        ...


@dataclass(frozen=True)
class DefaultEquilibriumInitializer:
    """Current one-layer initialization behavior: explicit init, then hot start, else uniform."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        if request.user_init is not None and request.user_init.ln_nk is not None and request.user_init.ln_ntot is not None:
            return EquilibriumInit(
                ln_nk=jnp.asarray(request.user_init.ln_nk),
                ln_ntot=jnp.asarray(request.user_init.ln_ntot),
            )
        if (
            request.previous_solution is not None
            and request.previous_solution.ln_nk is not None
            and request.previous_solution.ln_ntot is not None
        ):
            return EquilibriumInit(
                ln_nk=jnp.asarray(request.previous_solution.ln_nk),
                ln_ntot=jnp.asarray(request.previous_solution.ln_ntot),
            )
        ln_nk0, ln_ntot0 = _default_init(request.b, request.K)
        return EquilibriumInit(ln_nk=ln_nk0, ln_ntot=ln_ntot0)


@dataclass(frozen=True)
class GridEquilibriumInitializer:
    """Minimal shell for a future grid-based one-layer initializer.

    The stored grid is validated against the runtime setup/preset when the
    initializer is called. Actual grid lookup/interpolation is not implemented yet.
    """

    grid: "EquilibriumGrid"
    preset_name: str
    expected_composition_axis_name: str = "log10(Z/Zsun)"

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        from exogibbs.api.equilibrium_grid import validate_equilibrium_grid_compatibility

        validate_equilibrium_grid_compatibility(
            self.grid,
            request.setup,
            self.preset_name,
            expected_composition_axis_name=self.expected_composition_axis_name,
        )
        raise NotImplementedError(
            "GridEquilibriumInitializer grid lookup/interpolation is not implemented yet."
        )


@dataclass(frozen=True)
class LearnedEquilibriumInitializer:
    """Placeholder for a future learned one-layer initializer."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        raise NotImplementedError("LearnedEquilibriumInitializer is not implemented yet.")


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


_DEFAULT_INITIALIZER = DefaultEquilibriumInitializer()


def _resolve_initial_guess(
    initializer: Optional[EquilibriumInitializer], request: EquilibriumInitRequest
) -> EquilibriumInit:
    active_initializer = initializer or _DEFAULT_INITIALIZER
    return active_initializer(request)


def _ln_normalized_pressure(P: float, Pref: float) -> float:
    return jnp.log(P / Pref)


def _get_profile_scan_body(
    setup: ChemicalSetup,
    b: Array,
    Pref: float,
    opts: "EquilibriumOptions",
    initializer: Optional[EquilibriumInitializer],
    return_diagnostics: bool,
) -> Callable:
    # Reuse one body callable per stable input bundle so repeated profile calls
    # do not hand lax.scan a fresh Python function identity each time.
    key = (id(setup), id(b), float(Pref), id(opts), id(initializer or _DEFAULT_INITIALIZER), return_diagnostics)
    scan_body = _PROFILE_SCAN_BODY_CACHE.get(key)
    if scan_body is not None:
        return scan_body

    K = int(setup.formula_matrix.shape[1])

    if return_diagnostics:
        def scan_body(carry, tp_pair):
            ln_nk_prev, ln_ntot_prev = carry
            Ti, Pi = tp_pair
            solver_init = _resolve_initial_guess(
                initializer,
                EquilibriumInitRequest(
                    setup=setup,
                    T=Ti,
                    P=Pi,
                    b=b,
                    K=K,
                    previous_solution=EquilibriumInit(ln_nk=ln_nk_prev, ln_ntot=ln_ntot_prev),
                ),
            )
            result, diag = equilibrium(
                setup,
                Ti,
                Pi,
                b,
                Pref=Pref,
                init=solver_init,
                options=opts,
                return_diagnostics=True,
            )
            ln_ntot_next = jnp.log(jnp.clip(result.ntot, 1e-300))
            carry_next = (result.ln_n, ln_ntot_next)
            return carry_next, (result, diag)
    else:
        def scan_body(carry, tp_pair):
            ln_nk_prev, ln_ntot_prev = carry
            Ti, Pi = tp_pair
            solver_init = _resolve_initial_guess(
                initializer,
                EquilibriumInitRequest(
                    setup=setup,
                    T=Ti,
                    P=Pi,
                    b=b,
                    K=K,
                    previous_solution=EquilibriumInit(ln_nk=ln_nk_prev, ln_ntot=ln_ntot_prev),
                ),
            )
            result = equilibrium(
                setup,
                Ti,
                Pi,
                b,
                Pref=Pref,
                init=solver_init,
                options=opts,
                return_diagnostics=False,
            )
            ln_ntot_next = jnp.log(jnp.clip(result.ntot, 1e-300))
            carry_next = (result.ln_n, ln_ntot_next)
            return carry_next, result

    _PROFILE_SCAN_BODY_CACHE[key] = scan_body
    return scan_body


def equilibrium(
    setup: ChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    init: Optional[EquilibriumInit] = None,
    initializer: Optional[EquilibriumInitializer] = None,
    options: Optional[EquilibriumOptions] = None,
    return_diagnostics: bool = False,
) -> Union[EquilibriumResult, Tuple[EquilibriumResult, Mapping[str, Array]]]:
    """Compute equilibrium composition at (T, P, b) via Gibbs minimization.

    Args:
        setup: ChemicalSetup with formula matrix and hvector_func(T).
        T: Temperature (K).
        P: Pressure (bar).
        b: Elemental abundances; array of shape (E,).
        Pref: Reference pressure (bar) for normalization.
        init: Optional explicit initial guess for ln n and ln n_tot.
        initializer: Optional strategy object that produces an initial guess.
        options: Solver options.

    Returns:
        EquilibriumResult with ln n, n, mole fractions x, and n_tot.
        If ``return_diagnostics=True``, returns ``(result, diagnostics)`` where
        diagnostics is a pytree-compatible mapping with solver metrics.
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
    solver_init = _resolve_initial_guess(
        initializer,
        EquilibriumInitRequest(setup=setup, T=T, P=P, b=b, K=K, user_init=init),
    )
    ln_nk0, ln_ntot0 = _prepare_init(solver_init, b, K)

    hfunc = setup.hvector_func
    state = ThermoState(T, lnP, b)
    diagnostics = None
    if return_diagnostics:
        ln_n, diagnostics = minimize_gibbs_with_diagnostics(
            state,
            ln_nk0,
            ln_ntot0,
            A,
            hfunc,
            epsilon_crit=opts.epsilon_crit,
            max_iter=opts.max_iter,
        )
    else:
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
    result = EquilibriumResult(
        ln_n=ln_n, n=n, x=x, ntot=ntot, iterations=None, metadata=None
    )
    if return_diagnostics:
        return result, diagnostics
    return result

def equilibrium_profile(
    setup: ChemicalSetup,
    T: Array,
    P: Array,
    b: Array,
    *,
    Pref: float = 1.0,
    initializer: Optional[EquilibriumInitializer] = None,
    options: Optional[EquilibriumOptions] = None,
    return_diagnostics: bool = False,
) -> Union[EquilibriumResult, Tuple[EquilibriumResult, Mapping[str, Array]]]:
    """Vectorized equilibrium along a 1D T/P profile (layers).

    This computes equilibrium over layers indexed by ``i``. ``T[i]`` and
    ``P[i]`` vary by layer, while the elemental abundances ``b`` are currently
    shared across all layers.

    Args:
        setup: ChemicalSetup with formula matrix and hvector_func(T).
        T: Temperatures, shape (N,).
        P: Pressures, shape (N,).
        b: Elemental abundances, shape (E,), shared across layers.
        Pref: Reference pressure (bar).
        initializer: Optional strategy object that produces each layer's initial guess.
        options: Solver options.
        return_diagnostics: If True, returns per-layer solver diagnostics.

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
    opts = options or EquilibriumOptions()
    method = opts.method
    valid_methods = ("vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom")
    if method not in valid_methods:
        raise ValueError(f"Unknown solve method '{method}'. Expected one of {valid_methods}.")

    # Baseline path: vmap over layers for T and P, with b shared across layers.
    # Each layer solve uses a cold start unless an initializer overrides it.
    if method == "vmap_cold":
        if return_diagnostics:
            layer_fn = jax.vmap(
                lambda Ti, Pi: equilibrium(
                    setup,
                    Ti,
                    Pi,
                    b,
                    Pref=Pref,
                    initializer=initializer,
                    options=opts,
                    return_diagnostics=True,
                ),
                in_axes=(0, 0),
            )
            return layer_fn(T, P)

        layer_fn = jax.vmap(
            lambda Ti, Pi: equilibrium(
                setup,
                Ti,
                Pi,
                b,
                Pref=Pref,
                initializer=initializer,
                options=opts,
                return_diagnostics=False,
            ),
            in_axes=(0, 0),
        )
        return layer_fn(T, P)

    # Hot-start scan path: solve layers sequentially and carry converged state.
    if method == "scan_hot_from_bottom":
        T_in = jnp.flip(T, axis=0)
        P_in = jnp.flip(P, axis=0)
    else:
        T_in = T
        P_in = P

    K = int(setup.formula_matrix.shape[1])
    first_init = _resolve_initial_guess(
        initializer,
        EquilibriumInitRequest(setup=setup, T=T_in[0], P=P_in[0], b=b, K=K),
    )
    ln_nk0, ln_ntot0 = _prepare_init(first_init, b, K)

    if return_diagnostics:
        scan_body = _get_profile_scan_body(setup, b, Pref, opts, initializer, True)
        _, (result_seq, diag_seq) = lax.scan(scan_body, (ln_nk0, ln_ntot0), (T_in, P_in))
    else:
        scan_body = _get_profile_scan_body(setup, b, Pref, opts, initializer, False)
        _, result_seq = lax.scan(scan_body, (ln_nk0, ln_ntot0), (T_in, P_in))
        diag_seq = None

    if method == "scan_hot_from_bottom":
        result_seq = EquilibriumResult(
            ln_n=jnp.flip(result_seq.ln_n, axis=0),
            n=jnp.flip(result_seq.n, axis=0),
            x=jnp.flip(result_seq.x, axis=0),
            ntot=jnp.flip(result_seq.ntot, axis=0),
            iterations=result_seq.iterations,
            metadata=result_seq.metadata,
        )
        if diag_seq is not None:
            diag_seq = {k: jnp.flip(v, axis=0) for k, v in diag_seq.items()}

    if return_diagnostics:
        return result_seq, diag_seq
    return result_seq


__all__ = [
    "equilibrium",
    "equilibrium_profile",
    "EquilibriumInitializer",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumInitRequest",
    "EquilibriumResult",
    "DefaultEquilibriumInitializer",
    "GridEquilibriumInitializer",
    "LearnedEquilibriumInitializer",
]
