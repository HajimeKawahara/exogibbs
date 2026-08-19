"""Contracts for the stable MELTYQ preset and its legacy facade."""

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.magma_gas import MagmaGasInit, MagmaGasOptions, solve
from exogibbs.experimental.magma_gas import (
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    prepare_meltyq_chemistry,
    solve_magma_atmosphere_interface,
)
from exogibbs.magma_gas.models import meltyq as meltyq_implementation
from exogibbs.magma_gas.models.meltyq.basis import (
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
)
from exogibbs.presets import magma_gas as magma_gas_preset
from exogibbs.presets.magma_gas import (
    MELTYQ_ELEMENTS,
    MELTYQ_SPECIES,
    MeltyqMagmaGasInputs,
    prepare_meltyq_problem,
)
from exogibbs.solubility import co_yoshioka2019, n2_dasgupta2022
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.oxygen_fugacity import delta_iw_hirschmann2021


jax.config.update("jax_enable_x64", True)

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


def test_stable_preset_matches_legacy_wrapper_for_known_composition() -> None:
    hvector = -jnp.log(_PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS)
    source_setup = ChemicalSetup(
        formula_matrix=_FORMULA_MATRIX,
        hvector_func=lambda temperature: hvector,
        elements=MELTYQ_ELEMENTS,
        species=MELTYQ_SPECIES,
    )
    element_abundances = _FORMULA_MATRIX @ _KNOWN_GAS_MOLE_FRACTIONS
    element_abundances = element_abundances / element_abundances[0]
    root_variables = jnp.log(element_abundances[1:])
    pressure_gpa = _PRESSURE_BAR * 1.0e-4
    delta_iw = delta_iw_hirschmann2021(
        _OXYGEN_FUGACITY_BAR,
        _TEMPERATURE_K,
        pressure_gpa,
    )
    co_melt_mole_ratio = elemental_c_mass_fraction_to_mole_ratio(
        co_yoshioka2019(_PRESSURE_BAR * _KNOWN_GAS_MOLE_FRACTIONS[4])
    )
    n_melt_mole_ratio = elemental_n_mass_fraction_to_mole_ratio(
        n2_dasgupta2022(
            pressure_gpa * _KNOWN_GAS_MOLE_FRACTIONS[7],
            _TEMPERATURE_K,
            pressure_gpa,
            delta_iw,
        )
    )
    perturbed_root = root_variables + jnp.asarray([0.5, -0.3, 0.4, -0.2])

    stable = solve(
        prepare_meltyq_problem(source_setup),
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        MeltyqMagmaGasInputs(
            oxygen_fugacity_bar=_OXYGEN_FUGACITY_BAR,
            co_melt_mole_ratio=co_melt_mole_ratio,
            n_melt_mole_ratio=n_melt_mole_ratio,
        ),
        init=MagmaGasInit(root_variables=perturbed_root),
        options=MagmaGasOptions(root_tolerance=1.0e-10, max_iter=20),
    )
    legacy = solve_magma_atmosphere_interface(
        prepare_meltyq_chemistry(source_setup),
        _TEMPERATURE_K,
        _PRESSURE_BAR,
        _OXYGEN_FUGACITY_BAR,
        co_melt_mole_ratio,
        n_melt_mole_ratio,
        init=MagmaAtmosphereInterfaceInit(perturbed_root),
        options=MagmaAtmosphereInterfaceOptions(
            root_tolerance=1.0e-10,
            max_iter=20,
        ),
    )

    assert bool(stable.diagnostics.converged)
    assert bool(legacy.diagnostics.converged)
    np.testing.assert_allclose(
        jnp.exp(stable.gas.log_mole_fractions),
        _KNOWN_GAS_MOLE_FRACTIONS,
        rtol=0.0,
        atol=2.0e-9,
    )
    for stable_value, legacy_value in (
        (stable.element_abundances, legacy.element_abundances),
        (stable.root_variables, legacy.root_variables),
        (stable.gas.equilibrium.ln_n, legacy.gas_ln_n),
        (stable.gas.log_mole_fractions, legacy.gas_log_mole_fractions),
        (stable.gas.partial_pressures_bar, legacy.partial_pressures_bar),
        (stable.gas.fugacities_bar, legacy.fugacities_bar),
        (
            stable.model_state.melt_volatile_mole_ratios,
            legacy.melt_volatile_mole_ratios,
        ),
        (stable.model_state.delta_iw, legacy.delta_iw),
        (stable.diagnostics.residual, legacy.diagnostics.residual),
    ):
        np.testing.assert_allclose(stable_value, legacy_value, rtol=0.0, atol=0.0)


def test_preset_exports_are_owned_by_the_stable_implementation() -> None:
    assert all(
        getattr(magma_gas_preset, name)
        is getattr(meltyq_implementation, name)
        for name in magma_gas_preset.__all__
    )
