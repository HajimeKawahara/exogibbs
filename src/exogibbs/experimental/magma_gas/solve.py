"""Differentiable MELTYQ-equivalent magma--gas interface solver."""

from __future__ import annotations

from dataclasses import replace
import math
from numbers import Integral
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.experimental.magma_gas.meltyq_basis import (
    co2_mass_fraction_to_mole_ratio,
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
    elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
)
from exogibbs.experimental.magma_gas.setup import (
    CANONICAL_SPECIES,
    PreparedMagmaGasChemistry,
)
from exogibbs.experimental.magma_gas.types import (
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    MagmaAtmosphereInterfaceState,
    MagmaGasRootDiagnostics,
)
from exogibbs.solubility.volatile import (
    ch4_ardia2013,
    co2_lichtenberg2021,
    h2_hirschmann2012,
    h2o_lichtenberg2021,
    ln_co_yoshioka2019,
    ln_n2_dasgupta2022,
)
from exogibbs.thermo.oxygen_fugacity import delta_iw_hirschmann2021
from exogibbs.utils.units import convert_pressure


_SPECIES_INDEX = {
    species: index for index, species in enumerate(CANONICAL_SPECIES)
}
_H2 = _SPECIES_INDEX["H2"]
_HE = _SPECIES_INDEX["He"]
_O2 = _SPECIES_INDEX["O2"]
_H2O = _SPECIES_INDEX["H2O"]
_CO = _SPECIES_INDEX["CO"]
_CO2 = _SPECIES_INDEX["CO2"]
_CH4 = _SPECIES_INDEX["CH4"]
_N2 = _SPECIES_INDEX["N2"]


class _InterfaceParameters(NamedTuple):
    temperature_melt_k: jax.Array
    pressure_melt_bar: jax.Array
    oxygen_fugacity_bar: jax.Array
    co_melt_mole_ratio: jax.Array
    n_melt_mole_ratio: jax.Array


class _InterfaceEvaluation(NamedTuple):
    element_abundances: jax.Array
    gas_log_mole_fractions: jax.Array
    gas_mole_fractions: jax.Array
    log_partial_pressures_bar: jax.Array
    log_fugacities_bar: jax.Array
    partial_pressures_bar: jax.Array
    fugacities_bar: jax.Array
    log_melt_volatile_mole_ratios: jax.Array
    melt_volatile_mole_ratios: jax.Array
    delta_iw: jax.Array


class _RootCarry(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    progressing: jax.Array


class _LineSearchCarry(NamedTuple):
    trial: jax.Array
    alpha: jax.Array
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    accepted: jax.Array


class _OuterRootSolution(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    tolerance: jax.Array
    step_accepted: jax.Array


class _InnerRootDiagnostics(NamedTuple):
    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    tolerance: jax.Array


class _RootSolution(NamedTuple):
    root_variables: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    outer_converged: jax.Array
    inner_converged: jax.Array
    converged: jax.Array
    root_tolerance: jax.Array
    inner_iterations: jax.Array
    inner_residual_norm: jax.Array
    inner_tolerance: jax.Array
    step_accepted: jax.Array


ResidualFunction = Callable[[_InterfaceParameters, jax.Array], jax.Array]
InnerDiagnosticsFunction = Callable[
    [_InterfaceParameters, jax.Array], _InnerRootDiagnostics
]


def _validate_options(options: MagmaAtmosphereInterfaceOptions) -> None:
    if (
        not math.isfinite(options.root_tolerance)
        or options.root_tolerance <= 0.0
    ):
        raise ValueError("root_tolerance must be finite and positive.")
    if (
        isinstance(options.max_iter, bool)
        or not isinstance(options.max_iter, Integral)
        or options.max_iter < 0
    ):
        raise ValueError("max_iter must be a nonnegative integer.")
    if (
        isinstance(options.line_search_steps, bool)
        or not isinstance(options.line_search_steps, Integral)
        or options.line_search_steps <= 0
    ):
        raise ValueError("line_search_steps must be a positive integer.")
    if (
        not math.isfinite(options.backtracking_factor)
        or not 0.0 < options.backtracking_factor < 1.0
    ):
        raise ValueError("backtracking_factor must be between zero and one.")
    if not math.isfinite(options.max_log_step) or options.max_log_step <= 0.0:
        raise ValueError("max_log_step must be finite and positive.")
    if (
        not math.isfinite(options.h2_fraction_in_h_he)
        or not 0.0 < options.h2_fraction_in_h_he < 1.0
    ):
        raise ValueError("h2_fraction_in_h_he must be between zero and one.")
    inner = options.equilibrium_options
    if not math.isfinite(inner.epsilon_crit) or inner.epsilon_crit <= 0.0:
        raise ValueError(
            "equilibrium_options.epsilon_crit must be finite and positive."
        )
    if (
        isinstance(inner.max_iter, bool)
        or not isinstance(inner.max_iter, Integral)
        or inner.max_iter < 0
    ):
        raise ValueError(
            "equilibrium_options.max_iter must be a nonnegative integer."
        )


def _as_scalar(value: ArrayLike, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return array


def _normalize_parameters(
    dtype_source: ArrayLike,
    temperature_melt_k: ArrayLike,
    pressure_melt_bar: ArrayLike,
    oxygen_fugacity_bar: ArrayLike,
    co_melt_mole_ratio: ArrayLike,
    n_melt_mole_ratio: ArrayLike,
) -> _InterfaceParameters:
    raw = (
        _as_scalar(temperature_melt_k, "temperature_melt_k"),
        _as_scalar(pressure_melt_bar, "pressure_melt_bar"),
        _as_scalar(oxygen_fugacity_bar, "oxygen_fugacity_bar"),
        _as_scalar(co_melt_mole_ratio, "co_melt_mole_ratio"),
        _as_scalar(n_melt_mole_ratio, "n_melt_mole_ratio"),
    )
    dtype = jnp.result_type(dtype_source, *raw, jnp.float32)
    return _InterfaceParameters(
        *(jnp.asarray(value, dtype=dtype) for value in raw)
    )


def _element_abundances(root_variables: jax.Array) -> jax.Array:
    """Return ``(H, C, O, N, He)`` abundances in the ``b_H = 1`` gauge."""

    root = jnp.asarray(root_variables)
    return jnp.concatenate(
        [jnp.ones((1,), dtype=root.dtype), jnp.exp(root)]
    )


def _evaluate_lnphi(
    chemistry: PreparedMagmaGasChemistry,
    parameters: _InterfaceParameters,
    dtype: jnp.dtype,
) -> jax.Array:
    if chemistry.lnphi_func is None:
        return jnp.zeros((len(CANONICAL_SPECIES),), dtype=dtype)
    lnphi = jnp.asarray(
        chemistry.lnphi_func(
            parameters.temperature_melt_k,
            parameters.pressure_melt_bar,
            None,
        ),
        dtype=dtype,
    )
    expected_shape = (len(CANONICAL_SPECIES),)
    if lnphi.shape != expected_shape:
        raise ValueError(
            "prepared lnphi_func must return one value per canonical species: "
            f"expected {expected_shape}, got {lnphi.shape}."
        )
    return lnphi


def _effective_equilibrium_options(
    options: MagmaAtmosphereInterfaceOptions,
    dtype: jnp.dtype,
):
    """Use an inner tolerance that is meaningful for the active dtype."""

    roundoff_floor = 8.0 * float(jnp.finfo(dtype).eps)
    epsilon_crit = max(
        options.equilibrium_options.epsilon_crit,
        roundoff_floor,
    )
    return replace(
        options.equilibrium_options,
        epsilon_crit=epsilon_crit,
    )


def _evaluate_interface(
    chemistry: PreparedMagmaGasChemistry,
    options: MagmaAtmosphereInterfaceOptions,
    parameters: _InterfaceParameters,
    root_variables: jax.Array,
) -> _InterfaceEvaluation:
    element_abundances = _element_abundances(root_variables)
    dtype = element_abundances.dtype
    lnphi = _evaluate_lnphi(chemistry, parameters, dtype)

    def evaluated_lnphi_func(temperature, pressure_bar, mole_fractions):
        del temperature, pressure_bar
        if mole_fractions is not None:
            raise ValueError("magma-gas equilibrium requires pure lnphi values.")
        return lnphi

    gas_result = equilibrium(
        chemistry.setup,
        parameters.temperature_melt_k,
        parameters.pressure_melt_bar,
        element_abundances,
        options=_effective_equilibrium_options(options, dtype),
        lnphi_func=evaluated_lnphi_func,
    )
    gas_log_mole_fractions = gas_result.ln_n - logsumexp(gas_result.ln_n)
    log_pressure_bar = jnp.log(parameters.pressure_melt_bar)
    log_partial_pressures_bar = gas_log_mole_fractions + log_pressure_bar
    log_fugacities_bar = log_partial_pressures_bar + lnphi
    log_bar_to_gpa = jnp.log(jnp.asarray(1.0e-4, dtype=dtype))
    log_partial_pressures_gpa = log_partial_pressures_bar + log_bar_to_gpa
    gas_mole_fractions = jnp.exp(gas_log_mole_fractions)
    partial_pressures_bar = jnp.exp(log_partial_pressures_bar)
    fugacities_bar = jnp.exp(log_fugacities_bar)

    pressure_melt_gpa = convert_pressure(
        parameters.pressure_melt_bar,
        from_unit="bar",
        to_unit="GPa",
    )
    partial_pressures_pa = convert_pressure(
        partial_pressures_bar,
        from_unit="bar",
        to_unit="Pa",
    )
    fugacities_gpa = convert_pressure(
        fugacities_bar,
        from_unit="bar",
        to_unit="GPa",
    )
    delta_iw = jnp.asarray(
        delta_iw_hirschmann2021(
            parameters.oxygen_fugacity_bar,
            parameters.temperature_melt_k,
            pressure_melt_gpa,
        ),
        dtype=dtype,
    )

    h2_melt = h2_hirschmann2012(
        fugacities_bar[_H2],
        pressure_melt_gpa,
    )
    h2o_melt = h2o_mass_fraction_to_mole_ratio(
        h2o_lichtenberg2021(partial_pressures_pa[_H2O])
    )
    log_co_melt = elemental_c_ln_mass_fraction_to_ln_mole_ratio(
        ln_co_yoshioka2019(log_fugacities_bar[_CO])
    )
    co_melt = jnp.exp(log_co_melt)
    co2_melt = co2_mass_fraction_to_mole_ratio(
        co2_lichtenberg2021(partial_pressures_pa[_CO2])
    )
    ch4_melt = ch4_ardia2013(
        fugacities_gpa[_CH4],
        pressure_melt_gpa,
    )
    log_n_melt = elemental_n_ln_mass_fraction_to_ln_mole_ratio(
        ln_n2_dasgupta2022(
            log_partial_pressures_gpa[_N2],
            parameters.temperature_melt_k,
            pressure_melt_gpa,
            delta_iw,
        )
    )
    n_melt = jnp.exp(log_n_melt)
    log_melt_volatile_mole_ratios = jnp.asarray(
        jnp.stack(
            (
                jnp.log(h2_melt),
                jnp.log(h2o_melt),
                log_co_melt,
                jnp.log(co2_melt),
                jnp.log(ch4_melt),
                log_n_melt,
            )
        ),
        dtype=dtype,
    )
    melt_volatile_mole_ratios = jnp.asarray(
        jnp.stack(
            (h2_melt, h2o_melt, co_melt, co2_melt, ch4_melt, n_melt)
        ),
        dtype=dtype,
    )
    return _InterfaceEvaluation(
        element_abundances=element_abundances,
        gas_log_mole_fractions=gas_log_mole_fractions,
        gas_mole_fractions=gas_mole_fractions,
        log_partial_pressures_bar=log_partial_pressures_bar,
        log_fugacities_bar=log_fugacities_bar,
        partial_pressures_bar=partial_pressures_bar,
        fugacities_bar=fugacities_bar,
        log_melt_volatile_mole_ratios=log_melt_volatile_mole_ratios,
        melt_volatile_mole_ratios=melt_volatile_mole_ratios,
        delta_iw=delta_iw,
    )


def _inner_root_diagnostics(
    chemistry: PreparedMagmaGasChemistry,
    options: MagmaAtmosphereInterfaceOptions,
    parameters: _InterfaceParameters,
    root_variables: jax.Array,
) -> _InnerRootDiagnostics:
    """Audit the final gas solve without adding a reverse-mode path."""

    audit_parameters = jax.tree_util.tree_map(stop_gradient, parameters)
    audit_root = stop_gradient(root_variables)
    element_abundances = _element_abundances(audit_root)
    dtype = element_abundances.dtype
    lnphi = _evaluate_lnphi(chemistry, audit_parameters, dtype)

    def evaluated_lnphi_func(temperature, pressure_bar, mole_fractions):
        del temperature, pressure_bar
        if mole_fractions is not None:
            raise ValueError("magma-gas equilibrium requires pure lnphi values.")
        return lnphi

    _, diagnostics = equilibrium(
        chemistry.setup,
        audit_parameters.temperature_melt_k,
        audit_parameters.pressure_melt_bar,
        element_abundances,
        options=_effective_equilibrium_options(options, dtype),
        return_diagnostics=True,
        lnphi_func=evaluated_lnphi_func,
    )
    return _InnerRootDiagnostics(
        converged=diagnostics["converged"],
        iterations=diagnostics["n_iter"],
        residual_norm=diagnostics["final_residual"],
        tolerance=diagnostics["epsilon_crit"],
    )


def _interface_residual(
    chemistry: PreparedMagmaGasChemistry,
    options: MagmaAtmosphereInterfaceOptions,
    parameters: _InterfaceParameters,
    root_variables: jax.Array,
) -> jax.Array:
    evaluated = _evaluate_interface(
        chemistry,
        options,
        parameters,
        root_variables,
    )
    h2_fraction = jnp.asarray(
        options.h2_fraction_in_h_he,
        dtype=evaluated.gas_log_mole_fractions.dtype,
    )
    residual = jnp.stack(
        (
            evaluated.log_fugacities_bar[_O2]
            - jnp.log(parameters.oxygen_fugacity_bar),
            evaluated.log_melt_volatile_mole_ratios[2]
            - jnp.log(parameters.co_melt_mole_ratio),
            evaluated.log_melt_volatile_mole_ratios[5]
            - jnp.log(parameters.n_melt_mole_ratio),
            evaluated.gas_log_mole_fractions[_H2]
            - evaluated.gas_log_mole_fractions[_HE]
            - jnp.log(h2_fraction / (1.0 - h2_fraction)),
        )
    )
    return jnp.asarray(residual, dtype=root_variables.dtype)


def _residual_norm(residual: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(residual))


def _effective_root_tolerance(
    options: MagmaAtmosphereInterfaceOptions,
    dtype: jnp.dtype,
) -> jax.Array:
    requested = jnp.asarray(options.root_tolerance, dtype=dtype)
    roundoff_floor = jnp.asarray(64.0 * jnp.finfo(dtype).eps, dtype=dtype)
    return jnp.maximum(requested, roundoff_floor)


def _solve_root_core(
    residual_func: ResidualFunction,
    parameters: _InterfaceParameters,
    root_variables_init: jax.Array,
    options: MagmaAtmosphereInterfaceOptions,
) -> _OuterRootSolution:
    root0 = jnp.asarray(root_variables_init)
    residual0 = residual_func(parameters, root0)
    norm0 = _residual_norm(residual0)
    initial = _RootCarry(
        root_variables=root0,
        residual=residual0,
        residual_norm=norm0,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        progressing=jnp.asarray(True),
    )
    tolerance = _effective_root_tolerance(options, root0.dtype)

    def root_cond(carry: _RootCarry) -> jax.Array:
        return (
            (carry.residual_norm > tolerance)
            & (carry.iterations < options.max_iter)
            & carry.progressing
            & jnp.isfinite(carry.residual_norm)
        )

    def root_body(carry: _RootCarry) -> _RootCarry:
        jacobian = jax.jacrev(lambda root: residual_func(parameters, root))(
            carry.root_variables
        )
        direction = jnp.linalg.solve(jacobian, -carry.residual)
        direction_norm = jnp.max(jnp.abs(direction))
        direction_scale = jnp.minimum(
            jnp.asarray(1.0, dtype=root0.dtype),
            jnp.asarray(options.max_log_step, dtype=root0.dtype)
            / jnp.maximum(
                direction_norm,
                jnp.asarray(jnp.finfo(root0.dtype).tiny, dtype=root0.dtype),
            ),
        )
        direction = direction_scale * direction
        line_initial = _LineSearchCarry(
            trial=jnp.asarray(0, dtype=jnp.int32),
            alpha=jnp.asarray(1.0, dtype=root0.dtype),
            root_variables=carry.root_variables,
            residual=carry.residual,
            residual_norm=carry.residual_norm,
            accepted=jnp.asarray(False),
        )

        def line_cond(line: _LineSearchCarry) -> jax.Array:
            return (
                (line.trial < options.line_search_steps) & (~line.accepted)
            )

        def line_body(line: _LineSearchCarry) -> _LineSearchCarry:
            candidate_root = carry.root_variables + line.alpha * direction
            candidate_residual = residual_func(parameters, candidate_root)
            candidate_norm = _residual_norm(candidate_residual)
            acceptable = (
                jnp.all(jnp.isfinite(candidate_root))
                & jnp.all(jnp.isfinite(candidate_residual))
                & jnp.isfinite(candidate_norm)
                & (candidate_norm < carry.residual_norm)
            )
            return _LineSearchCarry(
                trial=line.trial + 1,
                alpha=line.alpha * jnp.asarray(
                    options.backtracking_factor, dtype=root0.dtype
                ),
                root_variables=jnp.where(
                    acceptable, candidate_root, line.root_variables
                ),
                residual=jnp.where(
                    acceptable, candidate_residual, line.residual
                ),
                residual_norm=jnp.where(
                    acceptable, candidate_norm, line.residual_norm
                ),
                accepted=line.accepted | acceptable,
            )

        line = jax.lax.while_loop(line_cond, line_body, line_initial)
        return _RootCarry(
            root_variables=line.root_variables,
            residual=line.residual,
            residual_norm=line.residual_norm,
            iterations=carry.iterations + 1,
            progressing=line.accepted,
        )

    solved = jax.lax.while_loop(root_cond, root_body, initial)
    converged = (
        jnp.isfinite(solved.residual_norm)
        & (solved.residual_norm <= tolerance)
    )
    return _OuterRootSolution(
        root_variables=solved.root_variables,
        residual=solved.residual,
        residual_norm=solved.residual_norm,
        iterations=solved.iterations,
        converged=converged,
        tolerance=tolerance,
        step_accepted=(solved.iterations > 0) & solved.progressing,
    )


def _complete_root_solution(
    outer: _OuterRootSolution,
    inner: _InnerRootDiagnostics,
) -> _RootSolution:
    return _RootSolution(
        root_variables=outer.root_variables,
        residual=outer.residual,
        residual_norm=outer.residual_norm,
        iterations=outer.iterations,
        outer_converged=outer.converged,
        inner_converged=inner.converged,
        converged=outer.converged & inner.converged,
        root_tolerance=outer.tolerance,
        inner_iterations=inner.iterations,
        inner_residual_norm=inner.residual_norm,
        inner_tolerance=inner.tolerance,
        step_accepted=outer.step_accepted,
    )


def _make_implicit_root_solver(
    residual_func: ResidualFunction,
    inner_diagnostics_func: InnerDiagnosticsFunction,
    options: MagmaAtmosphereInterfaceOptions,
):
    """Return a custom-VJP root whose setup and policy stay static."""

    def solve_with_diagnostics(parameters, root_variables_init):
        outer = _solve_root_core(
            residual_func,
            parameters,
            root_variables_init,
            options,
        )
        inner = inner_diagnostics_func(parameters, outer.root_variables)
        return _complete_root_solution(outer, inner)

    @custom_vjp
    def implicit_root(
        parameters: _InterfaceParameters,
        root_variables_init: jax.Array,
    ) -> _RootSolution:
        return solve_with_diagnostics(parameters, root_variables_init)

    def implicit_root_fwd(parameters, root_variables_init):
        solution = solve_with_diagnostics(parameters, root_variables_init)
        residuals = (
            solution.root_variables,
            parameters,
            solution.converged,
        )
        return solution, residuals

    def implicit_root_bwd(residuals, cotangent):
        root_variables, parameters, converged = residuals
        jacobian = jax.jacrev(
            lambda root: residual_func(parameters, root)
        )(root_variables)
        adjoint = jnp.linalg.solve(
            jacobian.T,
            jnp.asarray(cotangent.root_variables),
        )
        _, parameter_pullback = jax.vjp(
            lambda dynamic_parameters: residual_func(
                dynamic_parameters,
                stop_gradient(root_variables),
            ),
            parameters,
        )
        parameter_cotangent = parameter_pullback(-adjoint)[0]

        def require_converged(value):
            value = jnp.asarray(value)
            return jnp.where(
                converged,
                value,
                jnp.full_like(value, jnp.nan),
            )

        parameter_cotangent = jax.tree_util.tree_map(
            require_converged,
            parameter_cotangent,
        )
        return parameter_cotangent, None

    implicit_root.defvjp(implicit_root_fwd, implicit_root_bwd)
    return implicit_root


def _default_root_variables(
    parameters: _InterfaceParameters,
    options: MagmaAtmosphereInterfaceOptions,
) -> jax.Array:
    """Build a deterministic H-rich proxy for the first outer iteration."""

    dtype = parameters.temperature_melt_k.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    carbon_ratio = jnp.maximum(parameters.co_melt_mole_ratio, tiny)
    nitrogen_ratio = jnp.maximum(parameters.n_melt_mole_ratio, tiny)
    oxygen_proxy = jnp.sqrt(
        jnp.maximum(
            parameters.oxygen_fugacity_bar
            / parameters.pressure_melt_bar,
            tiny,
        )
    )
    oxygen_ratio = jnp.maximum(oxygen_proxy + carbon_ratio, tiny)
    h2_fraction = jnp.asarray(options.h2_fraction_in_h_he, dtype=dtype)
    helium_ratio = 0.5 * (1.0 - h2_fraction) / h2_fraction
    return jnp.log(
        jnp.stack(
            (carbon_ratio, oxygen_ratio, nitrogen_ratio, helium_ratio)
        )
    )


def solve_magma_atmosphere_interface(
    chemistry: PreparedMagmaGasChemistry,
    temperature_melt_k: ArrayLike,
    pressure_melt_bar: ArrayLike,
    oxygen_fugacity_bar: ArrayLike,
    co_melt_mole_ratio: ArrayLike,
    n_melt_mole_ratio: ArrayLike,
    *,
    init: Optional[MagmaAtmosphereInterfaceInit] = None,
    options: Optional[MagmaAtmosphereInterfaceOptions] = None,
) -> MagmaAtmosphereInterfaceState:
    """Solve the experimental MELTYQ-equivalent magma--gas boundary.

    All pressure-like public inputs use bar.  Melt carbon is molecular CO on
    an elemental-C amount basis, and melt nitrogen is an elemental-N dilute
    mole ratio.  The five physical inputs must be positive for the logarithmic
    four-variable solve.  They are not clipped or host-validated so that the
    function remains JIT-compatible; invalid values produce a non-converged
    state containing non-finite values.
    """

    active_options = options or MagmaAtmosphereInterfaceOptions()
    _validate_options(active_options)
    parameters = _normalize_parameters(
        chemistry.setup.formula_matrix,
        temperature_melt_k,
        pressure_melt_bar,
        oxygen_fugacity_bar,
        co_melt_mole_ratio,
        n_melt_mole_ratio,
    )
    if init is None or init.log_element_ratios is None:
        root_variables_init = _default_root_variables(
            parameters,
            active_options,
        )
    else:
        root_variables_init = jnp.asarray(
            init.log_element_ratios,
            dtype=parameters.temperature_melt_k.dtype,
        )
        if root_variables_init.shape != (4,):
            raise ValueError("init.log_element_ratios must have shape (4,).")
    root_variables_init = stop_gradient(root_variables_init)

    def residual_func(dynamic_parameters, root_variables):
        return _interface_residual(
            chemistry,
            active_options,
            dynamic_parameters,
            root_variables,
        )

    def inner_diagnostics_func(dynamic_parameters, root_variables):
        return _inner_root_diagnostics(
            chemistry,
            active_options,
            dynamic_parameters,
            root_variables,
        )

    implicit_root = _make_implicit_root_solver(
        residual_func,
        inner_diagnostics_func,
        active_options,
    )
    root_solution = implicit_root(parameters, root_variables_init)
    evaluated = _evaluate_interface(
        chemistry,
        active_options,
        parameters,
        root_solution.root_variables,
    )
    diagnostics = MagmaGasRootDiagnostics(
        converged=root_solution.converged,
        outer_converged=root_solution.outer_converged,
        inner_converged=root_solution.inner_converged,
        iterations=root_solution.iterations,
        inner_iterations=root_solution.inner_iterations,
        residual=root_solution.residual,
        residual_norm=root_solution.residual_norm,
        root_tolerance=root_solution.root_tolerance,
        inner_residual_norm=root_solution.inner_residual_norm,
        inner_tolerance=root_solution.inner_tolerance,
        step_accepted=root_solution.step_accepted,
    )
    return MagmaAtmosphereInterfaceState(
        element_abundances=evaluated.element_abundances,
        gas_log_mole_fractions=evaluated.gas_log_mole_fractions,
        gas_mole_fractions=evaluated.gas_mole_fractions,
        partial_pressures_bar=evaluated.partial_pressures_bar,
        fugacities_bar=evaluated.fugacities_bar,
        melt_volatile_mole_ratios=evaluated.melt_volatile_mole_ratios,
        delta_iw=evaluated.delta_iw,
        root_variables=root_solution.root_variables,
        diagnostics=diagnostics,
    )


__all__ = ("solve_magma_atmosphere_interface",)
