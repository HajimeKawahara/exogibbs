"""One-layer gas-equilibrium application service."""

from typing import Mapping, Optional, Tuple, Union

import jax.numpy as jnp

from exogibbs.equilibrium.gas.initialization import (
    prepare_init,
    resolve_initial_guess,
)
from exogibbs.equilibrium.gas.kernel.diagnostics import (
    minimize_gibbs_with_diagnostics,
)
from exogibbs.equilibrium.gas.kernel.solver import minimize_gibbs
from exogibbs.equilibrium.gas.types import (
    Array,
    EquilibriumInit,
    EquilibriumInitializer,
    EquilibriumInitRequest,
    EquilibriumOptions,
    EquilibriumResult,
    ThermoState,
)
from exogibbs.thermo.models import ChemicalSetup


def ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    """Return log pressure normalized by the reference pressure."""

    return jnp.log(pressure / reference_pressure)


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
    """Compute gas-only equilibrium at one temperature and pressure."""

    opts = options or EquilibriumOptions()
    formula_matrix = setup.formula_matrix
    species_count = int(formula_matrix.shape[1])
    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if b.shape[0] != formula_matrix.shape[0]:
        raise ValueError(
            f"b has length {b.shape[0]} but A expects "
            f"{formula_matrix.shape[0]} elements."
        )

    solver_init = resolve_initial_guess(
        initializer,
        EquilibriumInitRequest(
            setup=setup,
            T=T,
            P=P,
            b=b,
            K=species_count,
            user_init=init,
        ),
    )
    ln_nk_init, ln_ntot_init = prepare_init(
        solver_init,
        b,
        species_count,
    )
    state = ThermoState(T, ln_normalized_pressure(P, Pref), b)
    if return_diagnostics:
        ln_n, diagnostics = minimize_gibbs_with_diagnostics(
            state,
            ln_nk_init,
            ln_ntot_init,
            formula_matrix,
            setup.hvector_func,
            epsilon_crit=opts.epsilon_crit,
            max_iter=opts.max_iter,
        )
    else:
        ln_n = minimize_gibbs(
            state,
            ln_nk_init,
            ln_ntot_init,
            formula_matrix,
            setup.hvector_func,
            epsilon_crit=opts.epsilon_crit,
            max_iter=opts.max_iter,
        )
        diagnostics = None

    amounts = jnp.exp(ln_n)
    total_amount = jnp.asarray(jnp.sum(amounts))
    result = EquilibriumResult(
        ln_n=ln_n,
        n=amounts,
        x=amounts / jnp.clip(total_amount, 1.0e-300),
        ntot=total_amount,
        iterations=None,
        metadata=None,
    )
    if return_diagnostics:
        return result, diagnostics
    return result


solve = equilibrium


__all__ = ("equilibrium", "solve")
