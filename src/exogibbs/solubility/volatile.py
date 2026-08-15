"""Empirical volatile-solubility laws used by magma interface models.

The functions use the source-consistent units and fraction bases of the
selected empirical laws.  Where a compilation conflicts with its cited
formulation or experimental source, the discrepancy is recorded in metadata.
Inputs are not clipped to the experimental calibration ranges.  MELTYQ's later
conversion of mass fractions to a common dilute mole-ratio basis is outside the
scope of the individual empirical laws.  Automatic derivatives are finite at
positive interior pressures; fractional-power laws retain their singular
derivative at zero pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike


__all__ = (
    "MELTYQ_SOLUBILITY_METADATA",
    "SolubilityMetadata",
    "ch4_ardia2013",
    "co2_lichtenberg2021",
    "co_yoshioka2019",
    "h2_hirschmann2012",
    "h2o_lichtenberg2021",
    "ln_co_yoshioka2019",
    "ln_n2_dasgupta2022",
    "n2_dasgupta2022",
)


_CO_LOG10_INTERCEPT = -7.2
_CO_FUGACITY_EXPONENT = 0.8
_N_MASS_FRACTION_SCALE = 1.0e-6
_N_REDUCED_PRESSURE_EXPONENT = 0.5
_N_PRESSURE_TEMPERATURE_COEFFICIENT = 5908.0
_N_DELTA_IW_COEFFICIENT = -1.6
_N_COMPOSITION_INTERCEPT = 4.67
_N_SIO2_COEFFICIENT = 7.11
_N_AL2O3_COEFFICIENT = -13.06
_N_TIO2_COEFFICIENT = -120.67


@dataclass(frozen=True)
class SolubilityMetadata:
    """Provenance and calibration range for one solubility law."""

    species: str
    output_quantity: str
    output_basis: str
    calibration_temperature_k: Tuple[float, float]
    calibration_total_pressure_gpa: Tuple[float, float]
    experimental_reference: str
    experimental_doi: str
    formulation_reference: str
    formulation_doi: str
    implementation_reference: str
    implementation_doi: str
    notes: str = ""


MELTYQ_SOLUBILITY_METADATA: Mapping[str, SolubilityMetadata] = MappingProxyType({
    "h2_hirschmann2012": SolubilityMetadata(
        species="H2",
        output_quantity="H2",
        output_basis="mole_fraction",
        calibration_temperature_k=(1673.0, 1773.0),
        calibration_total_pressure_gpa=(0.7, 3.0),
        experimental_reference="Hirschmann et al. (2012)",
        experimental_doi="10.1016/j.epsl.2012.06.031",
        formulation_reference="Seo et al. (2024), Eq. 6",
        formulation_doi="10.3847/1538-4357/ad7461",
        implementation_reference="Ito & Changeat (2026), Eq. A3",
        implementation_doi="10.48550/arXiv.2605.08752",
        notes=(
            "Retained for MELTYQ compatibility. Chaudhari et al. (2025) "
            "report that the older infrared calibration may overestimate H2 "
            "solubility by about one order of magnitude "
            "(doi:10.1007/s00410-025-02272-y)."
        ),
    ),
    "h2o_lichtenberg2021": SolubilityMetadata(
        species="H2O",
        output_quantity="H2O",
        output_basis="mass_fraction",
        calibration_temperature_k=(973.0, 1723.0),
        calibration_total_pressure_gpa=(1.0e-4, 0.8),
        experimental_reference=(
            "Petrological datasets compiled by Lichtenberg et al. (2021), "
            "Table 1"
        ),
        experimental_doi="10.1029/2020JE006711",
        formulation_reference="Lichtenberg et al. (2021), Eq. 9 and Table 1",
        formulation_doi="10.1029/2020JE006711",
        implementation_reference="Ito & Changeat (2026), Eq. A1",
        implementation_doi="10.48550/arXiv.2605.08752",
    ),
    "co2_lichtenberg2021": SolubilityMetadata(
        species="CO2",
        output_quantity="CO2",
        output_basis="mass_fraction",
        calibration_temperature_k=(1123.0, 1923.0),
        calibration_total_pressure_gpa=(1.0e-2, 3.0),
        experimental_reference=(
            "Petrological datasets compiled by Lichtenberg et al. (2021), "
            "Table 1"
        ),
        experimental_doi="10.1029/2020JE006711",
        formulation_reference="Lichtenberg et al. (2021), Eq. 9 and Table 1",
        formulation_doi="10.1029/2020JE006711",
        implementation_reference="Ito & Changeat (2026), Eq. A2",
        implementation_doi="10.48550/arXiv.2605.08752",
    ),
    "co_yoshioka2019": SolubilityMetadata(
        species="CO",
        output_quantity="elemental C dissolved as CO",
        output_basis="mass_fraction",
        calibration_temperature_k=(1523.0, 1873.0),
        calibration_total_pressure_gpa=(0.2, 3.0),
        experimental_reference="Yoshioka et al. (2019)",
        experimental_doi="10.1016/j.gca.2019.06.007",
        formulation_reference=(
            "Yoshioka et al. (2019), Eq. 6; wt.% C converted to mass fraction"
        ),
        formulation_doi="10.1016/j.gca.2019.06.007",
        implementation_reference="Ito & Changeat (2026), Eq. A5",
        implementation_doi="10.48550/arXiv.2605.08752",
        notes=(
            "Uses the corrected -7.2 mass-fraction intercept on an elemental-C "
            "basis with fugacity in bar, following Yoshioka et al. (2019). The "
            "MELTYQ appendix labels fugacity as GPa, while that intercept and "
            "the experimental source require bar."
        ),
    ),
    "ch4_ardia2013": SolubilityMetadata(
        species="CH4",
        output_quantity="CH4",
        output_basis="mole_fraction",
        calibration_temperature_k=(1673.0, 1723.0),
        calibration_total_pressure_gpa=(0.7, 3.0),
        experimental_reference="Ardia et al. (2013)",
        experimental_doi="10.1016/j.gca.2013.03.028",
        formulation_reference="Seo et al. (2024), Eq. 17",
        formulation_doi="10.3847/1538-4357/ad7461",
        implementation_reference="Ito & Changeat (2026), Eq. A4",
        implementation_doi="10.48550/arXiv.2605.08752",
        notes=(
            "Uses fugacity in GPa as in Seo et al. (2024), Eq. 17. The "
            "MELTYQ appendix labels the same term as bar."
        ),
    ),
    "n2_dasgupta2022": SolubilityMetadata(
        species="N2",
        output_quantity="total elemental N",
        output_basis="mass_fraction",
        calibration_temperature_k=(1323.0, 2600.0),
        calibration_total_pressure_gpa=(1.0e-4, 8.2),
        experimental_reference="Dasgupta et al. (2022)",
        experimental_doi="10.1016/j.gca.2022.09.012",
        formulation_reference="Dasgupta et al. (2022), Eq. 10",
        formulation_doi="10.1016/j.gca.2022.09.012",
        implementation_reference="Ito & Changeat (2026), Eq. A6",
        implementation_doi="10.48550/arXiv.2605.08752",
        notes=(
            "The source reports total dissolved N in ppm by mass. Uses "
            "sqrt(melt pressure) as in Dasgupta et al. (2022), Eq. 10; the "
            "MELTYQ appendix typesets this pressure without the square root. "
            "Default oxide mole fractions reproduce its basaltic melt."
        ),
    ),
})


def _valid_nonnegative(*values: jnp.ndarray) -> jnp.ndarray:
    """Return a broadcast validity mask for finite, nonnegative values."""

    valid = jnp.asarray(True)
    for value in values:
        valid = valid & jnp.isfinite(value) & (value >= 0.0)
    return valid


def _valid_ln_nonnegative(value: jnp.ndarray) -> jnp.ndarray:
    """Accept finite logarithms and ``-inf`` as the zero boundary."""

    return jnp.isfinite(value) | jnp.isneginf(value)


def _n2_secondary_terms(
    temperature_k: ArrayLike,
    melt_pressure_gpa: ArrayLike,
    delta_iw: ArrayLike,
    x_sio2: ArrayLike,
    x_al2o3: ArrayLike,
    x_tio2: ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the pressure-independent N terms and their validity mask."""

    temperature = jnp.asarray(temperature_k)
    melt_pressure = jnp.asarray(melt_pressure_gpa)
    delta_iw_array = jnp.asarray(delta_iw)
    silica = jnp.asarray(x_sio2)
    alumina = jnp.asarray(x_al2o3)
    titania = jnp.asarray(x_tio2)

    reduced_exponent = (
        _N_PRESSURE_TEMPERATURE_COEFFICIENT
        * jnp.sqrt(melt_pressure)
        / temperature
        + _N_DELTA_IW_COEFFICIENT * delta_iw_array
    )
    composition_exponent = (
        _N_COMPOSITION_INTERCEPT
        + _N_SIO2_COEFFICIENT * silica
        + _N_AL2O3_COEFFICIENT * alumina
        + _N_TIO2_COEFFICIENT * titania
    )
    valid = _valid_nonnegative(melt_pressure, silica, alumina, titania)
    valid = (
        valid
        & jnp.isfinite(temperature)
        & (temperature > 0.0)
        & jnp.isfinite(delta_iw_array)
        & (silica <= 1.0)
        & (alumina <= 1.0)
        & (titania <= 1.0)
        & (silica + alumina + titania <= 1.0)
    )
    return reduced_exponent, composition_exponent, valid


def h2_hirschmann2012(
    hydrogen_fugacity_bar: ArrayLike,
    melt_pressure_gpa: ArrayLike,
) -> jnp.ndarray:
    """Return dissolved H2 mole fraction from MELTYQ equation A3.

    Non-finite inputs and negative pressures or fugacities return ``nan``.
    """

    fugacity = jnp.asarray(hydrogen_fugacity_bar)
    melt_pressure = jnp.asarray(melt_pressure_gpa)
    result = fugacity * jnp.exp(-11.403 - 0.76 * melt_pressure)
    return jnp.where(
        _valid_nonnegative(fugacity, melt_pressure),
        result,
        jnp.nan,
    )


def h2o_lichtenberg2021(
    water_partial_pressure_pa: ArrayLike,
) -> jnp.ndarray:
    """Return dissolved H2O mass fraction from MELTYQ equation A1.

    Non-finite or negative partial pressures return ``nan``.  The derivative
    with respect to partial pressure is singular at zero.
    """

    partial_pressure = jnp.asarray(water_partial_pressure_pa)
    result = 1.033e-6 * partial_pressure ** (1.0 / 1.747)
    return jnp.where(
        _valid_nonnegative(partial_pressure),
        result,
        jnp.nan,
    )


def co2_lichtenberg2021(
    carbon_dioxide_partial_pressure_pa: ArrayLike,
) -> jnp.ndarray:
    """Return dissolved CO2 mass fraction from MELTYQ equation A2.

    Non-finite or negative partial pressures return ``nan``.
    """

    partial_pressure = jnp.asarray(carbon_dioxide_partial_pressure_pa)
    result = 1.937e-15 * partial_pressure ** (1.0 / 0.714)
    return jnp.where(
        _valid_nonnegative(partial_pressure),
        result,
        jnp.nan,
    )


def co_yoshioka2019(
    carbon_monoxide_fugacity_bar: ArrayLike,
) -> jnp.ndarray:
    """Return elemental-C mass fraction dissolved from CO-bearing fluid.

    Non-finite or negative fugacities return ``nan``.  The derivative with
    respect to fugacity is singular at zero.
    """

    fugacity = jnp.asarray(carbon_monoxide_fugacity_bar)
    result = 10.0**_CO_LOG10_INTERCEPT * fugacity**_CO_FUGACITY_EXPONENT
    return jnp.where(_valid_nonnegative(fugacity), result, jnp.nan)


def ln_co_yoshioka2019(
    ln_carbon_monoxide_fugacity_bar: ArrayLike,
) -> jnp.ndarray:
    """Return ln of elemental-C mass fraction from ``ln(f_CO / 1 bar)``.

    The input and output are natural logarithms.  An input of ``-inf``
    represents zero fugacity and returns ``-inf``; ``nan`` and ``+inf`` are
    invalid and return ``nan``.
    """

    ln_fugacity = jnp.asarray(ln_carbon_monoxide_fugacity_bar)
    result = (
        _CO_LOG10_INTERCEPT * jnp.log(10.0)
        + _CO_FUGACITY_EXPONENT * ln_fugacity
    )
    return jnp.where(_valid_ln_nonnegative(ln_fugacity), result, jnp.nan)


def ch4_ardia2013(
    methane_fugacity_gpa: ArrayLike,
    melt_pressure_gpa: ArrayLike,
) -> jnp.ndarray:
    """Return dissolved CH4 mole fraction from Seo et al. (2024), Eq. 17.

    Non-finite inputs and negative pressures or fugacities return ``nan``.
    """

    fugacity = jnp.asarray(methane_fugacity_gpa)
    melt_pressure = jnp.asarray(melt_pressure_gpa)
    result = fugacity * jnp.exp(-7.63 - 1.9 * melt_pressure)
    return jnp.where(
        _valid_nonnegative(fugacity, melt_pressure),
        result,
        jnp.nan,
    )


def n2_dasgupta2022(
    nitrogen_partial_pressure_gpa: ArrayLike,
    temperature_k: ArrayLike,
    melt_pressure_gpa: ArrayLike,
    delta_iw: ArrayLike,
    *,
    x_sio2: ArrayLike = 0.56,
    x_al2o3: ArrayLike = 0.11,
    x_tio2: ArrayLike = 0.01,
) -> jnp.ndarray:
    """Return total dissolved elemental-N mass fraction.

    ``delta_iw`` is log10 oxygen fugacity relative to the iron-wuestite
    buffer.  The oxide inputs are mole fractions and default to the basaltic
    composition adopted by MELTYQ.  Dasgupta et al. (2022), Eq. 10 reports
    this concentration as nitrogen in ppm by mass.  The function returns
    ``nan`` unless temperature is positive, pressures and oxide fractions are
    nonnegative, each oxide fraction is at most one, their sum is at most one,
    and every input is finite.  The derivative with respect to nitrogen
    partial pressure is singular at zero.
    """

    partial_pressure = jnp.asarray(nitrogen_partial_pressure_gpa)
    reduced_exponent, composition_exponent, valid = _n2_secondary_terms(
        temperature_k,
        melt_pressure_gpa,
        delta_iw,
        x_sio2,
        x_al2o3,
        x_tio2,
    )
    reduced_term = partial_pressure**_N_REDUCED_PRESSURE_EXPONENT * jnp.exp(
        reduced_exponent
    )
    molecular_term = partial_pressure * jnp.exp(composition_exponent)
    result = _N_MASS_FRACTION_SCALE * (reduced_term + molecular_term)

    valid = valid & _valid_nonnegative(partial_pressure)
    return jnp.where(valid, result, jnp.nan)


def ln_n2_dasgupta2022(
    ln_nitrogen_partial_pressure_gpa: ArrayLike,
    temperature_k: ArrayLike,
    melt_pressure_gpa: ArrayLike,
    delta_iw: ArrayLike,
    *,
    x_sio2: ArrayLike = 0.56,
    x_al2o3: ArrayLike = 0.11,
    x_tio2: ArrayLike = 0.01,
) -> jnp.ndarray:
    """Return ln of elemental-N mass fraction from ``ln(P_N2 / 1 GPa)``.

    The pressure input and returned dimensionless mass fraction use natural
    logarithms.  An input of ``-inf`` represents zero partial pressure and
    returns ``-inf``; ``nan`` and ``+inf`` are invalid and return ``nan``.
    Other inputs and validity conditions match :func:`n2_dasgupta2022`.
    """

    ln_partial_pressure = jnp.asarray(ln_nitrogen_partial_pressure_gpa)
    reduced_exponent, composition_exponent, valid = _n2_secondary_terms(
        temperature_k,
        melt_pressure_gpa,
        delta_iw,
        x_sio2,
        x_al2o3,
        x_tio2,
    )
    ln_reduced_term = (
        _N_REDUCED_PRESSURE_EXPONENT * ln_partial_pressure
        + reduced_exponent
    )
    ln_molecular_term = ln_partial_pressure + composition_exponent
    result = jnp.log(_N_MASS_FRACTION_SCALE) + jnp.logaddexp(
        ln_reduced_term,
        ln_molecular_term,
    )

    valid = valid & _valid_ln_nonnegative(ln_partial_pressure)
    return jnp.where(valid, result, jnp.nan)
