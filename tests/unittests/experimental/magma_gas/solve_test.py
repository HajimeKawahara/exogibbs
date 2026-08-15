"""Integration and differentiation tests for the magma--gas interface."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.experimental.magma_gas import (
    CANONICAL_ELEMENTS,
    CANONICAL_SPECIES,
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
    prepare_meltyq_chemistry,
    solve_magma_atmosphere_interface,
)
from exogibbs.experimental.magma_gas.meltyq_basis import (
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
)
from exogibbs.solubility import co_yoshioka2019, n2_dasgupta2022
from exogibbs.solubility.volatile import ln_co_yoshioka2019
from exogibbs.equilibrium.gas.types import EquilibriumOptions
from exogibbs.presets.ykb4 import chemsetup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.oxygen_fugacity import delta_iw_hirschmann2021


jax.config.update("jax_enable_x64", True)


_FORMULA_MATRIX = jnp.asarray(
    [
        [2, 0, 0, 2, 0, 0, 4, 0, 3],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 2, 1, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=jnp.float64,
)
_TEMPERATURE_K = 1700.0
_PRESSURE_BAR = 7000.0
_OXYGEN_FUGACITY_BAR = 5.938980509016222e-11
_O2_MOLE_FRACTION = _OXYGEN_FUGACITY_BAR / _PRESSURE_BAR
_H2_HE_MOLE_FRACTION = 0.53 - _O2_MOLE_FRACTION
_KNOWN_GAS_MOLE_FRACTIONS = jnp.asarray(
    [
        0.84 * _H2_HE_MOLE_FRACTION,
        0.16 * _H2_HE_MOLE_FRACTION,
        _O2_MOLE_FRACTION,
        0.20,
        0.08,
        0.07,
        0.04,
        0.05,
        0.03,
    ],
    dtype=jnp.float64,
)


class _SyntheticCase(NamedTuple):
    chemistry: object
    root_variables: jax.Array
    element_abundances: jax.Array
    oxygen_fugacity_bar: jax.Array
    co_melt_mole_ratio: jax.Array
    n_melt_mole_ratio: jax.Array
    options: MagmaAtmosphereInterfaceOptions


@pytest.fixture(scope="module")
def synthetic_case() -> _SyntheticCase:
    hvector = -jnp.log(_PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS)
    source_setup = ChemicalSetup(
        formula_matrix=_FORMULA_MATRIX,
        hvector_func=lambda temperature: hvector,
        elements=CANONICAL_ELEMENTS,
        species=CANONICAL_SPECIES,
    )
    chemistry = prepare_meltyq_chemistry(source_setup)
    element_abundances = _FORMULA_MATRIX @ _KNOWN_GAS_MOLE_FRACTIONS
    element_abundances = element_abundances / element_abundances[0]
    root_variables = jnp.log(element_abundances[1:])
    oxygen_fugacity_bar = jnp.asarray(_OXYGEN_FUGACITY_BAR)
    pressure_gpa = _PRESSURE_BAR * 1.0e-4
    delta_iw = delta_iw_hirschmann2021(
        oxygen_fugacity_bar,
        _TEMPERATURE_K,
        pressure_gpa,
    )
    co_melt_mole_ratio = elemental_c_mass_fraction_to_mole_ratio(
        co_yoshioka2019(
            _PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS[4]
        )
    )
    n_melt_mole_ratio = elemental_n_mass_fraction_to_mole_ratio(
        n2_dasgupta2022(
            pressure_gpa * _KNOWN_GAS_MOLE_FRACTIONS[7],
            _TEMPERATURE_K,
            pressure_gpa,
            delta_iw,
        )
    )
    return _SyntheticCase(
        chemistry=chemistry,
        root_variables=root_variables,
        element_abundances=element_abundances,
        oxygen_fugacity_bar=oxygen_fugacity_bar,
        co_melt_mole_ratio=co_melt_mole_ratio,
        n_melt_mole_ratio=n_melt_mole_ratio,
        options=MagmaAtmosphereInterfaceOptions(
            root_tolerance=1.0e-10,
            max_iter=20,
        ),
    )


@pytest.fixture(scope="module")
def solved_state(synthetic_case):
    return solve_magma_atmosphere_interface(
        synthetic_case.chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        synthetic_case.oxygen_fugacity_bar,
        synthetic_case.co_melt_mole_ratio,
        synthetic_case.n_melt_mole_ratio,
        init=MagmaAtmosphereInterfaceInit(
            synthetic_case.root_variables
            + jnp.asarray([0.5, -0.3, 0.4, -0.2])
        ),
        options=synthetic_case.options,
    )


def test_solver_recovers_a_known_tce_interface(synthetic_case, solved_state):
    assert bool(solved_state.diagnostics.converged)
    assert bool(solved_state.diagnostics.outer_converged)
    assert bool(solved_state.diagnostics.inner_converged)
    assert int(solved_state.diagnostics.iterations) < 10
    assert int(solved_state.diagnostics.inner_iterations) < 1000
    assert float(solved_state.diagnostics.residual_norm) < 1.0e-9
    assert (
        float(solved_state.diagnostics.inner_residual_norm)
        <= float(solved_state.diagnostics.inner_tolerance)
    )
    np.testing.assert_allclose(
        solved_state.root_variables,
        synthetic_case.root_variables,
        rtol=0.0,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        solved_state.element_abundances,
        synthetic_case.element_abundances,
        rtol=0.0,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        solved_state.gas_mole_fractions,
        _KNOWN_GAS_MOLE_FRACTIONS,
        rtol=0.0,
        atol=2.0e-9,
    )


def test_state_preserves_pressure_fugacity_and_melt_contracts(
    synthetic_case,
    solved_state,
):
    np.testing.assert_allclose(
        jnp.sum(solved_state.partial_pressures_bar),
        _PRESSURE_BAR,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        solved_state.fugacities_bar,
        solved_state.partial_pressures_bar,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        solved_state.melt_volatile_mole_ratios[2],
        synthetic_case.co_melt_mole_ratio,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        solved_state.melt_volatile_mole_ratios[5],
        synthetic_case.n_melt_mole_ratio,
        rtol=1.0e-9,
    )
    h2_he_fraction = solved_state.gas_mole_fractions[0] / (
        solved_state.gas_mole_fractions[0]
        + solved_state.gas_mole_fractions[1]
    )
    np.testing.assert_allclose(h2_he_fraction, 0.84, rtol=1.0e-10)


def test_nonideal_lnphi_is_used_end_to_end():
    lnphi_at_point = jnp.linspace(-0.08, 0.08, len(CANONICAL_SPECIES))
    hvector_temperature_slope = jnp.linspace(
        -0.03,
        0.04,
        len(CANONICAL_SPECIES),
    )
    hvector = (
        -jnp.log(_PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS)
        - lnphi_at_point
    )

    def hvector_func(temperature):
        return hvector + hvector_temperature_slope * (
            temperature / _TEMPERATURE_K - 1.0
        )

    source_setup = ChemicalSetup(
        formula_matrix=_FORMULA_MATRIX,
        hvector_func=hvector_func,
        elements=CANONICAL_ELEMENTS,
        species=CANONICAL_SPECIES,
    )

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        assert mole_fractions is None
        return (
            lnphi_at_point
            + 1.0e-5 * (temperature - _TEMPERATURE_K)
            + 1.0e-8 * (pressure_bar - _PRESSURE_BAR)
        )

    chemistry = prepare_meltyq_chemistry(
        source_setup,
        lnphi_func=lnphi_func,
    )
    element_abundances = _FORMULA_MATRIX @ _KNOWN_GAS_MOLE_FRACTIONS
    element_abundances = element_abundances / element_abundances[0]
    root_variables = jnp.log(element_abundances[1:])
    fugacities_bar = (
        _PRESSURE_BAR
        * _KNOWN_GAS_MOLE_FRACTIONS
        * jnp.exp(lnphi_at_point)
    )
    oxygen_fugacity_bar = fugacities_bar[2]
    pressure_gpa = _PRESSURE_BAR * 1.0e-4
    delta_iw = delta_iw_hirschmann2021(
        oxygen_fugacity_bar,
        _TEMPERATURE_K,
        pressure_gpa,
    )
    co_melt = elemental_c_mass_fraction_to_mole_ratio(
        co_yoshioka2019(fugacities_bar[4])
    )
    n_melt = elemental_n_mass_fraction_to_mole_ratio(
        n2_dasgupta2022(
            pressure_gpa * _KNOWN_GAS_MOLE_FRACTIONS[7],
            _TEMPERATURE_K,
            pressure_gpa,
            delta_iw,
        )
    )
    state = solve_magma_atmosphere_interface(
        chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        oxygen_fugacity_bar,
        co_melt,
        n_melt,
        init=MagmaAtmosphereInterfaceInit(root_variables + 0.1),
        options=MagmaAtmosphereInterfaceOptions(root_tolerance=1.0e-10),
    )

    assert bool(state.diagnostics.converged)
    np.testing.assert_allclose(
        state.gas_mole_fractions,
        _KNOWN_GAS_MOLE_FRACTIONS,
        rtol=0.0,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(state.fugacities_bar, fugacities_bar, rtol=1e-9)

    root_weights = jnp.asarray([0.3, -0.2, 0.5, 0.1])

    def state_loss(log_scales):
        scales = jnp.exp(log_scales)
        varied = solve_magma_atmosphere_interface(
            chemistry,
            _TEMPERATURE_K * scales[0],
            _PRESSURE_BAR * scales[1],
            oxygen_fugacity_bar * scales[2],
            co_melt * scales[3],
            n_melt * scales[4],
            init=MagmaAtmosphereInterfaceInit(root_variables + 0.1),
            options=MagmaAtmosphereInterfaceOptions(root_tolerance=1.0e-10),
        )
        return (
            jnp.vdot(root_weights, varied.root_variables)
            + 0.03 * varied.gas_log_mole_fractions[4]
            - 0.02 * jnp.log(varied.melt_volatile_mole_ratios[1])
            + 0.01 * varied.delta_iw
            + 0.02 * jnp.log(varied.fugacities_bar[7])
        )

    origin = jnp.zeros(5)
    calculated_gradient = jax.jit(jax.grad(state_loss))(origin)
    epsilon = 1.0e-4
    finite_difference = jnp.stack(
        [
            (
                state_loss(origin + epsilon * direction)
                - state_loss(origin - epsilon * direction)
            )
            / (2.0 * epsilon)
            for direction in jnp.eye(5)
        ]
    )

    assert jnp.all(jnp.isfinite(calculated_gradient))
    np.testing.assert_allclose(
        calculated_gradient,
        finite_difference,
        rtol=5.0e-6,
        atol=5.0e-7,
    )


def test_default_h_rich_initialization_converges(synthetic_case):
    state = solve_magma_atmosphere_interface(
        synthetic_case.chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        synthetic_case.oxygen_fugacity_bar,
        synthetic_case.co_melt_mole_ratio,
        synthetic_case.n_melt_mole_ratio,
        options=synthetic_case.options,
    )

    assert bool(state.diagnostics.converged)
    np.testing.assert_allclose(
        state.root_variables,
        synthetic_case.root_variables,
        rtol=0.0,
        atol=2.0e-9,
    )


def test_co_constraint_remains_in_log_space_at_extreme_dilution():
    co_mole_fraction = 1.0e-313
    h2_he_mole_fraction = 0.58 - co_mole_fraction
    gas_mole_fractions = np.asarray(
        [
            0.84 * h2_he_mole_fraction,
            0.16 * h2_he_mole_fraction,
            0.03,
            0.20,
            co_mole_fraction,
            0.07,
            0.04,
            0.05,
            0.03,
        ]
    )
    hvector = jnp.asarray(-np.log(_PRESSURE_BAR * gas_mole_fractions))
    chemistry = prepare_meltyq_chemistry(
        ChemicalSetup(
            formula_matrix=_FORMULA_MATRIX,
            hvector_func=lambda temperature: hvector,
            elements=CANONICAL_ELEMENTS,
            species=CANONICAL_SPECIES,
        )
    )
    element_abundances = np.asarray(_FORMULA_MATRIX) @ gas_mole_fractions
    element_abundances = element_abundances / element_abundances[0]
    root_variables = jnp.asarray(np.log(element_abundances[1:]))
    oxygen_fugacity_bar = _PRESSURE_BAR * gas_mole_fractions[2]
    pressure_gpa = _PRESSURE_BAR * 1.0e-4
    delta_iw = delta_iw_hirschmann2021(
        oxygen_fugacity_bar,
        _TEMPERATURE_K,
        pressure_gpa,
    )
    log_co_melt = elemental_c_ln_mass_fraction_to_ln_mole_ratio(
        ln_co_yoshioka2019(
            np.log(_PRESSURE_BAR) + np.log(co_mole_fraction)
        )
    )
    co_melt_mole_ratio = jnp.exp(log_co_melt)
    n_melt_mole_ratio = elemental_n_mass_fraction_to_mole_ratio(
        n2_dasgupta2022(
            pressure_gpa * gas_mole_fractions[7],
            _TEMPERATURE_K,
            pressure_gpa,
            delta_iw,
        )
    )

    state = solve_magma_atmosphere_interface(
        chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        oxygen_fugacity_bar,
        co_melt_mole_ratio,
        n_melt_mole_ratio,
        init=MagmaAtmosphereInterfaceInit(root_variables),
        options=MagmaAtmosphereInterfaceOptions(
            root_tolerance=1.0e-10,
            max_iter=0,
            equilibrium_options=EquilibriumOptions(
                epsilon_crit=1.0e-12,
                max_iter=1000,
            ),
        ),
    )

    assert bool(state.diagnostics.converged)
    assert jnp.all(jnp.isfinite(state.diagnostics.residual))
    assert (
        state.gas_log_mole_fractions[4]
        < jnp.log(jnp.finfo(state.gas_log_mole_fractions.dtype).tiny)
    )
    np.testing.assert_allclose(
        state.melt_volatile_mole_ratios[2],
        co_melt_mole_ratio,
        rtol=1.0e-10,
    )


def test_mixed_source_dtype_is_normalized_for_float32_root(synthetic_case):
    hvector_float64 = -jnp.log(
        jnp.asarray(_PRESSURE_BAR, jnp.float64)
        * jnp.asarray(_KNOWN_GAS_MOLE_FRACTIONS, jnp.float64)
    )
    source_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(_FORMULA_MATRIX, jnp.float32),
        hvector_func=lambda temperature: hvector_float64,
        elements=CANONICAL_ELEMENTS,
        species=CANONICAL_SPECIES,
    )
    chemistry = prepare_meltyq_chemistry(source_setup)
    state = solve_magma_atmosphere_interface(
        chemistry,
        jnp.asarray(_TEMPERATURE_K, jnp.float32),
        jnp.asarray(_PRESSURE_BAR, jnp.float32),
        jnp.asarray(synthetic_case.oxygen_fugacity_bar, jnp.float32),
        jnp.asarray(synthetic_case.co_melt_mole_ratio, jnp.float32),
        jnp.asarray(synthetic_case.n_melt_mole_ratio, jnp.float32),
        init=MagmaAtmosphereInterfaceInit(
            jnp.asarray(synthetic_case.root_variables + 0.1, jnp.float32)
        ),
    )

    assert state.root_variables.dtype == jnp.float32
    assert state.melt_volatile_mole_ratios.dtype == jnp.float32
    assert bool(state.diagnostics.converged)


def test_implicit_gradient_matches_central_difference(synthetic_case):
    init = MagmaAtmosphereInterfaceInit(
        synthetic_case.root_variables
        + jnp.asarray([0.2, -0.2, 0.15, -0.1])
    )

    root_weights = jnp.asarray([0.3, -0.2, 0.5, 0.1])

    def weighted_root(log_oxygen_scale):
        state = solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            synthetic_case.oxygen_fugacity_bar
            * jnp.exp(log_oxygen_scale),
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            init=init,
            options=synthetic_case.options,
        )
        return jnp.vdot(root_weights, state.root_variables)

    calculated = jax.jit(jax.grad(weighted_root))(jnp.asarray(0.0))
    epsilon = 1.0e-4
    finite_difference = (
        weighted_root(epsilon) - weighted_root(-epsilon)
    ) / (2.0 * epsilon)

    assert jnp.isfinite(calculated)
    np.testing.assert_allclose(calculated, finite_difference, rtol=2.0e-6)


def test_initial_root_vector_shape_is_validated(synthetic_case):
    with pytest.raises(ValueError, match="shape \\(4,\\)"):
        solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            synthetic_case.oxygen_fugacity_bar,
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            init=MagmaAtmosphereInterfaceInit(jnp.zeros((3,))),
            options=synthetic_case.options,
        )


def test_inner_nonconvergence_rejects_outer_root(synthetic_case):
    options = MagmaAtmosphereInterfaceOptions(
        root_tolerance=1.0e-9,
        max_iter=20,
        equilibrium_options=EquilibriumOptions(
            epsilon_crit=1.0e-12,
            max_iter=20,
        ),
    )
    state = solve_magma_atmosphere_interface(
        synthetic_case.chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        synthetic_case.oxygen_fugacity_bar,
        synthetic_case.co_melt_mole_ratio,
        synthetic_case.n_melt_mole_ratio,
        init=MagmaAtmosphereInterfaceInit(
            synthetic_case.root_variables
            + jnp.asarray([0.5, -0.3, 0.4, -0.2])
        ),
        options=options,
    )

    assert bool(state.diagnostics.outer_converged)
    assert not bool(state.diagnostics.inner_converged)
    assert not bool(state.diagnostics.converged)
    assert (
        float(state.diagnostics.residual_norm)
        <= float(state.diagnostics.root_tolerance)
    )
    assert (
        float(state.diagnostics.inner_residual_norm)
        > float(state.diagnostics.inner_tolerance)
    )

    def root_sum(log_oxygen_scale):
        failed = solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            synthetic_case.oxygen_fugacity_bar
            * jnp.exp(log_oxygen_scale),
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            init=MagmaAtmosphereInterfaceInit(
                synthetic_case.root_variables
                + jnp.asarray([0.5, -0.3, 0.4, -0.2])
            ),
            options=options,
        )
        return jnp.sum(failed.root_variables)

    assert jnp.isnan(jax.grad(root_sum)(jnp.asarray(0.0)))


def test_unattempted_root_step_and_gradient_fail_closed(synthetic_case):
    options = MagmaAtmosphereInterfaceOptions(max_iter=0)
    init = MagmaAtmosphereInterfaceInit(synthetic_case.root_variables + 0.2)

    def root_sum(log_oxygen_scale):
        state = solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            synthetic_case.oxygen_fugacity_bar
            * jnp.exp(log_oxygen_scale),
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            init=init,
            options=options,
        )
        return jnp.sum(state.root_variables)

    state = solve_magma_atmosphere_interface(
        synthetic_case.chemistry,
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        synthetic_case.oxygen_fugacity_bar,
        synthetic_case.co_melt_mole_ratio,
        synthetic_case.n_melt_mole_ratio,
        init=init,
        options=options,
    )

    assert not bool(state.diagnostics.converged)
    assert not bool(state.diagnostics.step_accepted)
    assert jnp.isnan(jax.grad(root_sum)(jnp.asarray(0.0)))


@pytest.mark.parametrize(
    "options",
    (
        MagmaAtmosphereInterfaceOptions(root_tolerance=jnp.inf),
        MagmaAtmosphereInterfaceOptions(line_search_steps=1.5),
        MagmaAtmosphereInterfaceOptions(backtracking_factor=jnp.nan),
        MagmaAtmosphereInterfaceOptions(
            equilibrium_options=EquilibriumOptions(epsilon_crit=jnp.inf)
        ),
    ),
)
def test_nonfinite_or_nonintegral_options_are_rejected(
    synthetic_case,
    options,
):
    with pytest.raises(ValueError):
        solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            synthetic_case.oxygen_fugacity_bar,
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            options=options,
        )


def test_invalid_traced_physical_input_fails_closed(synthetic_case):
    @jax.jit
    def convergence_status(oxygen_fugacity_bar):
        state = solve_magma_atmosphere_interface(
            synthetic_case.chemistry,
            _TEMPERATURE_K,
            _PRESSURE_BAR,
            oxygen_fugacity_bar,
            synthetic_case.co_melt_mole_ratio,
            synthetic_case.n_melt_mole_ratio,
            options=synthetic_case.options,
        )
        return state.diagnostics.converged, state.diagnostics.residual_norm

    converged, residual_norm = convergence_status(jnp.asarray(0.0))

    assert not bool(converged)
    assert not jnp.isfinite(residual_norm)


@pytest.mark.smoke
def test_ykb4_reduced_chemistry_solves_documented_point():
    chemistry = prepare_meltyq_chemistry(
        chemsetup(),
        species_map={
            "He": "He1",
            "H2O": "H2O1",
            "CO": "C1O1",
            "CO2": "C1O2",
            "CH4": "C1H4",
            "NH3": "H3N1",
        },
    )
    state = solve_magma_atmosphere_interface(
        chemistry,
        temperature_melt_k=1700.0,
        pressure_melt_bar=7000.0,
        oxygen_fugacity_bar=1.0e-10,
        co_melt_mole_ratio=5.0e-5,
        n_melt_mole_ratio=1.0e-4,
    )

    assert bool(state.diagnostics.converged)
    assert float(state.diagnostics.residual_norm) < 1.0e-8
    np.testing.assert_allclose(
        jnp.sum(state.gas_mole_fractions),
        1.0,
        rtol=1.0e-12,
    )
