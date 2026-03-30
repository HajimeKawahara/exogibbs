"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_rgie_cond import (
    minimize_gibbs_cond_with_diagnostics as _minimize_gibbs_cond_with_diagnostics_raw,
)

Array = jax.Array
CondensateProfileMethod = Literal["vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumInit:
    """Explicit condensate solver initialization state.

    This is intentionally small and can be reused as a future hot-start carrier.
    """

    ln_nk: Optional[Array] = None
    ln_mk: Optional[Array] = None
    ln_ntot: Optional[Array] = None

    def tree_flatten(self):
        children = (self.ln_nk, self.ln_mk, self.ln_ntot)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        ln_nk, ln_mk, ln_ntot = children
        return cls(ln_nk=ln_nk, ln_mk=ln_mk, ln_ntot=ln_ntot)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumDiagnostics:
    """Lightweight convergence diagnostics for one condensate solve."""

    n_iter: Array
    converged: Array
    hit_max_iter: Array
    final_residual: Array
    residual_crit: Array
    max_iter: Array
    epsilon: Array
    final_step_size: Array
    invalid_numbers_detected: Array
    debug_nan: Array

    def tree_flatten(self):
        children = (
            self.n_iter,
            self.converged,
            self.hit_max_iter,
            self.final_residual,
            self.residual_crit,
            self.max_iter,
            self.epsilon,
            self.final_step_size,
            self.invalid_numbers_detected,
            self.debug_nan,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    @classmethod
    def from_mapping(cls, diagnostics):
        return cls(
            n_iter=diagnostics["n_iter"],
            converged=diagnostics["converged"],
            hit_max_iter=diagnostics["hit_max_iter"],
            final_residual=diagnostics["final_residual"],
            residual_crit=diagnostics["residual_crit"],
            max_iter=diagnostics["max_iter"],
            epsilon=diagnostics["epsilon"],
            final_step_size=diagnostics["final_step_size"],
            invalid_numbers_detected=diagnostics["invalid_numbers_detected"],
            debug_nan=diagnostics["debug_nan"],
        )

    def asdict(self):
        return {
            "n_iter": self.n_iter,
            "converged": self.converged,
            "hit_max_iter": self.hit_max_iter,
            "final_residual": self.final_residual,
            "residual_crit": self.residual_crit,
            "max_iter": self.max_iter,
            "epsilon": self.epsilon,
            "final_step_size": self.final_step_size,
            "invalid_numbers_detected": self.invalid_numbers_detected,
            "debug_nan": self.debug_nan,
        }


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Structured condensate solve result with final state and diagnostics."""

    ln_nk: Array
    ln_mk: Array
    ln_ntot: Array
    diagnostics: CondensateEquilibriumDiagnostics

    def tree_flatten(self):
        children = (self.ln_nk, self.ln_mk, self.ln_ntot, self.diagnostics)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        ln_nk, ln_mk, ln_ntot, diagnostics = children
        return cls(ln_nk=ln_nk, ln_mk=ln_mk, ln_ntot=ln_ntot, diagnostics=diagnostics)

    def to_init(self) -> CondensateEquilibriumInit:
        return CondensateEquilibriumInit(
            ln_nk=self.ln_nk,
            ln_mk=self.ln_mk,
            ln_ntot=self.ln_ntot,
        )


def _prepare_condensate_init(init: CondensateEquilibriumInit) -> CondensateEquilibriumInit:
    if init.ln_nk is None or init.ln_mk is None or init.ln_ntot is None:
        raise ValueError(
            "CondensateEquilibriumInit requires ln_nk, ln_mk, and ln_ntot for the current solver path."
        )
    return CondensateEquilibriumInit(
        ln_nk=jnp.asarray(init.ln_nk),
        ln_mk=jnp.asarray(init.ln_mk),
        ln_ntot=jnp.asarray(init.ln_ntot),
    )


def _validate_profile_inputs(
    temperatures: Array,
    ln_normalized_pressures: Array,
    element_vector: Array,
) -> tuple[Array, Array, Array]:
    temperatures = jnp.asarray(temperatures)
    ln_normalized_pressures = jnp.asarray(ln_normalized_pressures)
    element_vector = jnp.asarray(element_vector)

    if temperatures.ndim != 1 or ln_normalized_pressures.ndim != 1:
        raise ValueError("temperatures and ln_normalized_pressures must be 1D arrays.")
    if temperatures.shape[0] != ln_normalized_pressures.shape[0]:
        raise ValueError("temperatures and ln_normalized_pressures must have the same length.")
    if element_vector.ndim != 1:
        raise ValueError("element_vector must be a 1D array shared across profile layers.")
    return temperatures, ln_normalized_pressures, element_vector


def _profile_init_is_batched(init: CondensateEquilibriumInit, n_layers: int) -> bool:
    prepared = _prepare_condensate_init(init)
    ln_nk = prepared.ln_nk
    ln_mk = prepared.ln_mk
    ln_ntot = prepared.ln_ntot

    if ln_nk.ndim == 1 and ln_mk.ndim == 1 and ln_ntot.ndim == 0:
        return False
    if ln_nk.ndim == 2 and ln_mk.ndim == 2 and ln_ntot.ndim == 1:
        if (
            ln_nk.shape[0] != n_layers
            or ln_mk.shape[0] != n_layers
            or ln_ntot.shape[0] != n_layers
        ):
            raise ValueError("Batched condensate profile init must have leading dimension equal to the number of layers.")
        return True
    raise ValueError(
        "CondensateEquilibriumInit for profile solves must be either unbatched "
        "(ln_nk[K], ln_mk[M], ln_ntot[]) or batched "
        "(ln_nk[N,K], ln_mk[N,M], ln_ntot[N])."
    )


def _profile_init_at(
    init: CondensateEquilibriumInit,
    n_layers: int,
    layer_index: int,
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    if not _profile_init_is_batched(prepared, n_layers):
        return prepared
    return CondensateEquilibriumInit(
        ln_nk=prepared.ln_nk[layer_index],
        ln_mk=prepared.ln_mk[layer_index],
        ln_ntot=prepared.ln_ntot[layer_index],
    )


def _broadcast_profile_init(
    init: CondensateEquilibriumInit,
    n_layers: int,
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    if _profile_init_is_batched(prepared, n_layers):
        return prepared
    return CondensateEquilibriumInit(
        ln_nk=jnp.broadcast_to(prepared.ln_nk, (n_layers,) + prepared.ln_nk.shape),
        ln_mk=jnp.broadcast_to(prepared.ln_mk, (n_layers,) + prepared.ln_mk.shape),
        ln_ntot=jnp.broadcast_to(prepared.ln_ntot, (n_layers,)),
    )


def _flip_condensate_profile_result(
    result: CondensateEquilibriumResult,
) -> CondensateEquilibriumResult:
    return tree_util.tree_map(lambda x: jnp.flip(x, axis=0), result)


def minimize_gibbs_cond(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
) -> CondensateEquilibriumResult:
    """Run the active condensate solver using a structured init/result interface."""

    init_prepared = _prepare_condensate_init(init)
    ln_nk, ln_mk, ln_ntot, diagnostics_raw = _minimize_gibbs_cond_with_diagnostics_raw(
        state,
        ln_nk_init=init_prepared.ln_nk,
        ln_mk_init=init_prepared.ln_mk,
        ln_ntot_init=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
    )
    return CondensateEquilibriumResult(
        ln_nk=ln_nk,
        ln_mk=ln_mk,
        ln_ntot=ln_ntot,
        diagnostics=CondensateEquilibriumDiagnostics.from_mapping(diagnostics_raw),
    )


def minimize_gibbs_cond_with_diagnostics(*args, **kwargs) -> CondensateEquilibriumResult:
    """Alias of :func:`minimize_gibbs_cond` kept for explicit diagnostics-oriented callers."""

    return minimize_gibbs_cond(*args, **kwargs)


def minimize_gibbs_cond_profile(
    temperatures: Array,
    ln_normalized_pressures: Array,
    element_vector: Array,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon_start: float = 0.0,
    epsilon_crit: float = -40.0,
    n_step: int = 100,
    max_iter: int = 100,
    method: CondensateProfileMethod = "scan_hot_from_bottom",
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
) -> CondensateEquilibriumResult:
    """Run the condensate solver over a 1D profile with cold- or hot-start execution.

    The per-layer epsilon continuation schedule is intentionally unchanged from the
    current example path: each layer steps from ``epsilon_start`` to
    ``epsilon_crit`` and then performs one final solve at ``epsilon_crit`` so the
    returned diagnostics correspond to the final layer solve.

    ``method="scan_hot_from_top"`` and ``method="scan_hot_from_bottom"`` carry
    structured :class:`CondensateEquilibriumInit` state layer-to-layer using
    :meth:`CondensateEquilibriumResult.to_init`. ``method="vmap_cold"`` keeps the
    existing independent-layer behavior.
    """

    if n_step < 1:
        raise ValueError("n_step must be at least 1.")
    valid_methods = ("vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom")
    if method not in valid_methods:
        raise ValueError(f"Unknown condensate profile solve method '{method}'. Expected one of {valid_methods}.")

    temperatures, ln_normalized_pressures, element_vector = _validate_profile_inputs(
        temperatures,
        ln_normalized_pressures,
        element_vector,
    )
    n_layers = int(temperatures.shape[0])
    epsilons = jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:]

    def solve_layer(
        temperature: Array,
        ln_normalized_pressure: Array,
        layer_init: CondensateEquilibriumInit,
    ) -> CondensateEquilibriumResult:
        thermo_state = ThermoState(
            temperature=temperature,
            ln_normalized_pressure=ln_normalized_pressure,
            element_vector=element_vector,
        )

        def body_fn(i, init_state):
            epsilon = epsilons[i]
            residual_crit = jnp.exp(epsilon)
            result = minimize_gibbs_cond(
                thermo_state,
                init=init_state,
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                residual_crit=residual_crit,
                max_iter=max_iter,
                element_indices=element_indices,
                debug_nan=debug_nan,
            )
            return result.to_init()

        final_init = lax.fori_loop(
            0,
            n_step,
            body_fn,
            _prepare_condensate_init(layer_init),
        )
        final_epsilon = epsilons[-1]
        return minimize_gibbs_cond(
            thermo_state,
            init=final_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=final_epsilon,
            residual_crit=jnp.exp(final_epsilon),
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
        )

    if method == "vmap_cold":
        batched_init = _broadcast_profile_init(init, n_layers)
        return jax.vmap(
            solve_layer,
            in_axes=(
                0,
                0,
                CondensateEquilibriumInit(ln_nk=0, ln_mk=0, ln_ntot=0),
            ),
        )(
            temperatures,
            ln_normalized_pressures,
            batched_init,
        )

    def scan_body(carry_init, layer_inputs):
        temperature, ln_normalized_pressure = layer_inputs
        result = solve_layer(temperature, ln_normalized_pressure, carry_init)
        return result.to_init(), result

    def run_scan(
        temperatures_scan: Array,
        ln_pressures_scan: Array,
        init0: CondensateEquilibriumInit,
        *,
        reverse_output: bool,
    ) -> CondensateEquilibriumResult:
        _, result_seq = lax.scan(scan_body, init0, (temperatures_scan, ln_pressures_scan))
        if reverse_output:
            return _flip_condensate_profile_result(result_seq)
        return result_seq

    if method == "scan_hot_from_top":
        return run_scan(
            temperatures,
            ln_normalized_pressures,
            _profile_init_at(init, n_layers, 0),
            reverse_output=False,
        )

    return run_scan(
        jnp.flip(temperatures, axis=0),
        jnp.flip(ln_normalized_pressures, axis=0),
        _profile_init_at(init, n_layers, n_layers - 1),
        reverse_output=True,
    )


__all__ = [
    "CondensateEquilibriumDiagnostics",
    "CondensateEquilibriumInit",
    "CondensateProfileMethod",
    "CondensateEquilibriumResult",
    "minimize_gibbs_cond",
    "minimize_gibbs_cond_profile",
    "minimize_gibbs_cond_core",
    "minimize_gibbs_cond_with_diagnostics",
]
