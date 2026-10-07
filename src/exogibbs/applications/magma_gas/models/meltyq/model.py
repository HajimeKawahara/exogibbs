"""Built-in MELTYQ magma--gas boundary model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exogibbs.applications.magma_gas.models.meltyq.basis import (
    co2_mass_fraction_to_mole_ratio,
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
    elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
)
from exogibbs.applications.magma_gas.models.meltyq.setup import (
    MELTYQ_SPECIES,
    prepare_meltyq_chemistry,
)
from exogibbs.applications.magma_gas.models.meltyq.types import (
    MeltyqMagmaGasInputs,
    MeltyqMagmaGasState,
)
from exogibbs.applications.magma_gas.types import (
    MagmaGasConditions,
    MagmaGasEquilibriumState,
    MagmaGasModelEvaluation,
    MagmaGasProblem,
)
from exogibbs.solubility.volatile import (
    ch4_ardia2013,
    co2_lichtenberg2021,
    h2_hirschmann2012,
    h2o_lichtenberg2021,
    ln_co_yoshioka2019,
    ln_n2_dasgupta2022,
)
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.oxygen_fugacity import delta_iw_hirschmann2021
from exogibbs.utils.units import convert_pressure


_SPECIES_INDEX = {
    species: index for index, species in enumerate(MELTYQ_SPECIES)
}
_H2 = _SPECIES_INDEX["H2"]
_HE = _SPECIES_INDEX["He"]
_O2 = _SPECIES_INDEX["O2"]
_H2O = _SPECIES_INDEX["H2O"]
_CO = _SPECIES_INDEX["CO"]
_CO2 = _SPECIES_INDEX["CO2"]
_CH4 = _SPECIES_INDEX["CH4"]
_N2 = _SPECIES_INDEX["N2"]


class _MeltyqParameters(NamedTuple):
    temperature_k: jax.Array
    pressure_bar: jax.Array
    oxygen_fugacity_bar: jax.Array
    co_melt_mole_ratio: jax.Array
    n_melt_mole_ratio: jax.Array
    h2_fraction_in_h_he: jax.Array


def _as_scalar(value: ArrayLike, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return array


def _parameters(conditions: MagmaGasConditions) -> _MeltyqParameters:
    inputs = conditions.model_inputs
    if not isinstance(inputs, MeltyqMagmaGasInputs):
        raise TypeError("MELTYQ requires MeltyqMagmaGasInputs.")
    raw = (
        _as_scalar(conditions.temperature_k, "temperature_k"),
        _as_scalar(conditions.pressure_bar, "pressure_bar"),
        _as_scalar(inputs.oxygen_fugacity_bar, "oxygen_fugacity_bar"),
        _as_scalar(inputs.co_melt_mole_ratio, "co_melt_mole_ratio"),
        _as_scalar(inputs.n_melt_mole_ratio, "n_melt_mole_ratio"),
        _as_scalar(inputs.h2_fraction_in_h_he, "h2_fraction_in_h_he"),
    )
    dtype = jnp.result_type(*raw, jnp.float32)
    return _MeltyqParameters(
        *(jnp.asarray(value, dtype=dtype) for value in raw)
    )


@dataclass(frozen=True)
class MeltyqMagmaGasModel:
    """MELTYQ element parameterization and volatile boundary residual."""

    def initial_root(self, conditions: MagmaGasConditions) -> jax.Array:
        parameters = _parameters(conditions)
        dtype = parameters.temperature_k.dtype
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
        carbon_ratio = jnp.maximum(parameters.co_melt_mole_ratio, tiny)
        nitrogen_ratio = jnp.maximum(parameters.n_melt_mole_ratio, tiny)
        oxygen_proxy = jnp.sqrt(
            jnp.maximum(
                parameters.oxygen_fugacity_bar / parameters.pressure_bar,
                tiny,
            )
        )
        oxygen_ratio = jnp.maximum(oxygen_proxy + carbon_ratio, tiny)
        h2_fraction = parameters.h2_fraction_in_h_he
        helium_ratio = 0.5 * (1.0 - h2_fraction) / h2_fraction
        return jnp.log(
            jnp.stack(
                (carbon_ratio, oxygen_ratio, nitrogen_ratio, helium_ratio)
            )
        )

    def element_abundances(
        self,
        conditions: MagmaGasConditions,
        root_variables: jax.Array,
    ) -> jax.Array:
        del conditions
        root = jnp.asarray(root_variables)
        return jnp.concatenate(
            [jnp.ones((1,), dtype=root.dtype), jnp.exp(root)]
        )

    def evaluate(
        self,
        conditions: MagmaGasConditions,
        root_variables: jax.Array,
        gas: MagmaGasEquilibriumState,
    ) -> MagmaGasModelEvaluation:
        parameters = _parameters(conditions)
        dtype = root_variables.dtype
        pressure_gpa = convert_pressure(
            parameters.pressure_bar,
            from_unit="bar",
            to_unit="GPa",
        )
        partial_pressures_pa = convert_pressure(
            gas.partial_pressures_bar,
            from_unit="bar",
            to_unit="Pa",
        )
        fugacities_gpa = convert_pressure(
            gas.fugacities_bar,
            from_unit="bar",
            to_unit="GPa",
        )
        log_partial_pressures_gpa = gas.log_partial_pressures_bar + jnp.log(
            jnp.asarray(1.0e-4, dtype=dtype)
        )
        delta_iw = jnp.asarray(
            delta_iw_hirschmann2021(
                parameters.oxygen_fugacity_bar,
                parameters.temperature_k,
                pressure_gpa,
            ),
            dtype=dtype,
        )

        h2_melt = h2_hirschmann2012(
            gas.fugacities_bar[_H2],
            pressure_gpa,
        )
        h2o_melt = h2o_mass_fraction_to_mole_ratio(
            h2o_lichtenberg2021(partial_pressures_pa[_H2O])
        )
        log_co_melt = elemental_c_ln_mass_fraction_to_ln_mole_ratio(
            ln_co_yoshioka2019(gas.log_fugacities_bar[_CO])
        )
        co_melt = jnp.exp(log_co_melt)
        co2_melt = co2_mass_fraction_to_mole_ratio(
            co2_lichtenberg2021(partial_pressures_pa[_CO2])
        )
        ch4_melt = ch4_ardia2013(
            fugacities_gpa[_CH4],
            pressure_gpa,
        )
        log_n_melt = elemental_n_ln_mass_fraction_to_ln_mole_ratio(
            ln_n2_dasgupta2022(
                log_partial_pressures_gpa[_N2],
                parameters.temperature_k,
                pressure_gpa,
                delta_iw,
            )
        )
        n_melt = jnp.exp(log_n_melt)
        log_melt = jnp.asarray(
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
        melt = jnp.asarray(
            jnp.stack(
                (h2_melt, h2o_melt, co_melt, co2_melt, ch4_melt, n_melt)
            ),
            dtype=dtype,
        )
        h2_fraction = parameters.h2_fraction_in_h_he
        residual = jnp.asarray(
            jnp.stack(
                (
                    gas.log_fugacities_bar[_O2]
                    - jnp.log(parameters.oxygen_fugacity_bar),
                    log_melt[2] - jnp.log(parameters.co_melt_mole_ratio),
                    log_melt[5] - jnp.log(parameters.n_melt_mole_ratio),
                    gas.log_mole_fractions[_H2]
                    - gas.log_mole_fractions[_HE]
                    - jnp.log(h2_fraction / (1.0 - h2_fraction)),
                )
            ),
            dtype=dtype,
        )
        state = MeltyqMagmaGasState(
            log_melt_volatile_mole_ratios=log_melt,
            melt_volatile_mole_ratios=melt,
            delta_iw=delta_iw,
        )
        return MagmaGasModelEvaluation(residual=residual, state=state)


def prepare_meltyq_problem(
    chemical_setup: ChemicalSetup,
    *,
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None,
    species_map: Optional[Mapping[str, str]] = None,
) -> MagmaGasProblem:
    """Prepare the built-in MELTYQ model from a full gas setup."""

    chemistry = prepare_meltyq_chemistry(
        chemical_setup,
        lnphi_func=lnphi_func,
        species_map=species_map,
    )
    return MagmaGasProblem(
        setup=chemistry.setup,
        model=MeltyqMagmaGasModel(),
        lnphi_func=chemistry.lnphi_func,
    )


__all__ = (
    "MeltyqMagmaGasModel",
    "prepare_meltyq_problem",
)
