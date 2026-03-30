"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import tree_util

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_rgie_cond import (
    minimize_gibbs_cond_with_diagnostics as _minimize_gibbs_cond_with_diagnostics_raw,
)

Array = jax.Array


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


__all__ = [
    "CondensateEquilibriumDiagnostics",
    "CondensateEquilibriumInit",
    "CondensateEquilibriumResult",
    "minimize_gibbs_cond",
    "minimize_gibbs_cond_core",
    "minimize_gibbs_cond_with_diagnostics",
]
