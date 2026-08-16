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
from exogibbs.thermo.fugacity import (
    LogFugacityCoefficientFunction,
    effective_gas_hvector,
)
from exogibbs.thermo.models import ChemicalSetup


def ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    """Return log pressure normalized by the reference pressure."""

    return jnp.log(pressure / reference_pressure)


def _effective_equilibrium_tolerance(
    requested: float,
    dtype: jnp.dtype,
) -> float:
    """Return a convergence tolerance resolvable in the solver dtype."""

    return max(requested, 8.0 * float(jnp.finfo(dtype).eps))


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
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None,
) -> Union[EquilibriumResult, Tuple[EquilibriumResult, Mapping[str, Array]]]:
    """Compute gas-only equilibrium at one temperature and pressure.

    ``lnphi_func`` supplies pure-component ``ln(phi)`` values in gas-species
    order and is called as ``lnphi_func(T, P, None)`` with pressure in bar.
    """

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
    hvector = effective_gas_hvector(
        setup,
        T,
        P,
        lnphi_func,
        mole_fractions=None,
    )
    solver_dtype = jnp.result_type(
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector,
        b,
        state.temperature,
        state.ln_normalized_pressure,
        jnp.float32,
    )
    epsilon_crit = _effective_equilibrium_tolerance(
        opts.epsilon_crit,
        solver_dtype,
    )
    if return_diagnostics:
        ln_n, diagnostics = minimize_gibbs_with_diagnostics(
            state,
            ln_nk_init,
            ln_ntot_init,
            formula_matrix,
            hvector,
            epsilon_crit=epsilon_crit,
            max_iter=opts.max_iter,
        )
    else:
        ln_n = minimize_gibbs(
            state,
            ln_nk_init,
            ln_ntot_init,
            formula_matrix,
            hvector,
            epsilon_crit=epsilon_crit,
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
