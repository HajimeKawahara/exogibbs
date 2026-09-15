"""Young/GCE thermochemistry callbacks
=====================================

JAX thermochemistry for the pinned, full Young/GCE component network.

These callbacks reproduce the source's empirical mass-action model at one
supplied temperature. They retain its four-component Fe-Si-O-H alloy, ideal
silicate and gas assumptions, and the special H2 dissolution pressure in R14.
They do not define a common Gibbs excess energy or select stable phases.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


SOURCE_MODEL_METADATA = {
    "model_id": "gce_young2023_full25_local_isothermal_v1",
    "repository": "https://github.com/ExoInteriors/GlobalChemicalEquilibrium_Release",
    "commit": "31558873d8da460c3cd11986b574cb347621b43d",
    "version": "Young_2023_Version",
    "metal_components": ("Fe", "Si", "O", "H"),
    "activity_convention": "source mole fractions; Fe and H coefficients are one",
    "silicate_and_gas": "ideal mixing as specified by the source",
    "standard_pressure_bar": 1.0,
    "R14_pressure_bar": 1.0e4,
    "limitations": (
        "One supplied temperature replaces the source's two temperatures. "
        "Pressure is supplied locally; no planetary pressure law is used. "
        "Standards use a frozen Shomate branch and a 1 bar approximation. "
        "Both reference temperatures extrapolate the source MgO liquid fit. "
        "No joint calibration domain or liquid-phase stability is established. "
        "Source mass-action reproduction is not a Gibbs-minimum claim."
    ),
}


def make_source_standard_potentials_rt(
    record: dict[str, Any], case: dict[str, Any],
) -> Callable[[ArrayLike, ArrayLike], jax.Array]:
    """Return ``mu0/(RT)(T_K, P_bar)`` in the record's full component order.

    Select ``case`` before tracing: its frozen Shomate coefficient branches
    remain fixed during local temperature differentiation. This is not a
    general temperature-range selector. Pressure is ignored explicitly;
    all standards retain the source's 1 bar approximation. The empirical
    reaction fits reconstruct missing endmember standards without setting
    any standard chemical potential arbitrarily to zero.
    """
    species = tuple(name for phase in record["phases"].values() for name in phase)
    fitted_species = tuple(case["shomate"])
    coefficients = jnp.asarray([case["shomate"][name] for name in fitted_species])
    gas_constant = record["source"]["gas_constant_J_mol_K"]
    log10_to_ln = record["source"]["log10_to_ln"]

    def standard_potentials_rt(
        temperature_k: ArrayLike, pressure_bar: ArrayLike,
    ) -> jax.Array:
        del pressure_bar
        temperature = jnp.asarray(temperature_k)
        if temperature.ndim != 0:
            raise ValueError("temperature_k must be scalar.")
        rt = gas_constant * temperature
        t = temperature / 1000.0
        dh, a, b, c, d, e, f, g, h = coefficients.T
        enthalpy_kj = dh + a*t + b*t**2/2 + c*t**3/3 + d*t**4/4 - e/t + f - h
        entropy = a*jnp.log(t) + b*t + c*t**2/2 + d*t**3/3 - e/(2*t**2) + g
        values = 1000.0 * enthalpy_kj - temperature * entropy
        potentials = dict(zip(fitted_species, values))

        dg_na = rt * log10_to_ln * (-1.33 + 13870.0 / temperature)
        dg_fe = rt * log10_to_ln * (-0.63 + 3103.0 / temperature)
        dg_si = -rt * log10_to_ln * (2.97 - 21800.0 / temperature)
        dg_o = -rt * log10_to_ln * (2.736 - 11439.0 / temperature)
        dg_h2 = rt * (12.5 + 0.76e-4)
        dg_h2o = rt * (14.21 - 2565.0 / temperature)
        dg_co2 = 5200.0 + 119.77 * temperature
        dg_co = dg_co2 + rt * log10_to_ln * jnp.log10(3.0)
        dg_okuchi = 143589.7 - 69.1 * temperature

        potentials["Na2SiO3_silicate"] = (
            potentials["Na2O_silicate"] + potentials["SiO2_silicate"] - dg_na
        )
        potentials["FeSiO3_silicate"] = (
            potentials["FeO_silicate"] + potentials["SiO2_silicate"] - dg_fe
        )
        potentials["H2_silicate"] = potentials["H2_gas"] + dg_h2
        potentials["H2O_silicate"] = potentials["H2O_gas"] + dg_h2o
        potentials["CO_silicate"] = potentials["CO_gas"] + dg_co
        potentials["CO2_silicate"] = potentials["CO2_gas"] + dg_co2
        potentials["Si_metal"] = (
            dg_si - 2 * potentials["FeO_silicate"]
            + 2 * potentials["Fe_metal"] + potentials["SiO2_silicate"]
        )
        potentials["O_metal"] = (
            dg_o + potentials["FeO_silicate"] - potentials["Fe_metal"]
        )
        potentials["H_metal"] = 0.5 * (
            dg_okuchi - potentials["FeO_silicate"]
            + potentials["Fe_metal"] + potentials["H2O_silicate"]
        )
        return jnp.stack([potentials[name] for name in species]) / rt

    return standard_potentials_rt


def source_metal_ln_gamma(
    temperature_k: ArrayLike, pressure_bar: ArrayLike, mole_fractions: ArrayLike,
) -> jax.Array:
    """Return source coefficients in Fe, Si, O, H order, excluding ``ln(x)``.

    Inputs use the entire four-component metal's normalized mole fractions.
    Fe and H coefficients are one by the source's explicit approximation;
    no ternary ExoEOS coefficients are extended to a missing H component.
    Si/O use the GCE oxygen coefficient variant, not Young's printed variant.
    Pressure is not modeled. The caller must supply positive T/P, normalized
    nonnegative fractions and positive Fe; no clipping or traces are added.
    Algebraic zero-solute limits remove the source's ``log(1-x)/x`` quotients.
    """
    del pressure_bar
    temperature = jnp.asarray(temperature_k)
    composition = jnp.asarray(mole_fractions)
    if temperature.ndim != 0 or composition.shape != (4,):
        raise ValueError("Expected scalar temperature_k and shape (4,) in Fe, Si, O, H order.")
    ln_gamma_si, ln_gamma_o = source_solute_ln_gamma(temperature, composition[1], composition[2])
    zero = jnp.zeros_like(ln_gamma_si)
    return jnp.stack((zero, ln_gamma_si, ln_gamma_o, zero))


def source_solute_ln_gamma(
    temperature_k: ArrayLike, silicon_fraction: ArrayLike, oxygen_fraction: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Return the shared GCE Si/O coefficients using full-metal fractions.

    Young, sulfur/nitrogen and carbon sources use the same Si/O equations.
    Additional metal components dilute these fractions without renormalizing
    the Fe/Si/O subset. Network-specific C/S corrections remain separate.
    """
    temperature = jnp.asarray(temperature_k)
    si, oxygen = jnp.asarray(silicon_fraction), jnp.asarray(oxygen_fraction)
    inverse_si = 1 / (1 - si)
    inverse_o = 1 / (1 - oxygen)
    cross = -5.0 * 1873.0 / temperature
    ln_gamma_si = (
        -6.65 * 1873.0 / temperature
        - 12.41 * 1873.0 / temperature * jnp.log1p(-si)
        - cross * (oxygen + jnp.log1p(-oxygen) - oxygen * inverse_si)
        + cross * oxygen**2 * si
        * (inverse_si + inverse_o + si * inverse_si**2 / 2 - 1)
    )
    ln_gamma_o = (
        4.29 - 16500.0 / temperature
        + 1873.0 / temperature * jnp.log1p(-oxygen)
        - cross * (si + jnp.log1p(-si) - si * inverse_o)
        + cross * si**2 * oxygen
        * (inverse_o + inverse_si + oxygen * inverse_o**2 / 2 - 1)
    )
    return ln_gamma_si, ln_gamma_o


def source_reaction_offsets(
    temperature_k: ArrayLike, pressure_bar: ArrayLike,
) -> jax.Array:
    """Add to ordinary reaction residuals to retain the source R14 closure.

    The source fixes the H2 dissolution pressure to 1e4 bar while gas
    chemical potentials use the supplied pressure. Thus R14 alone needs
    ``ln(P_bar / 1e4)``. This is an empirical closure, not an extra phase
    activity coefficient. Order is the complete source reaction basis R0--R17.
    """
    del temperature_k
    pressure = jnp.asarray(pressure_bar)
    if pressure.ndim != 0:
        raise ValueError("pressure_bar must be scalar.")
    offset = jnp.log(pressure / 1.0e4)
    return jnp.zeros((18,), dtype=offset.dtype).at[14].set(offset)
