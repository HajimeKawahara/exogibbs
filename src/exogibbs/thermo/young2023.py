"""Pinned Young/GCE and supplementary NASA7 standard states.

These retain the fixed 2350 K Shomate branches and their declared extrapolation.
No cross-phase calibration, phase recipe or equilibrium solver is supplied.
"""

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

R = 8.31446261815324
DATA = json.loads((Path(__file__).parents[1] / "data/young2023/standard_states.json").read_text())
COMPONENTS = tuple(name for phase in DATA["phases"].values() for name in phase)

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


_BASE = make_source_standard_potentials_rt(DATA, DATA["cases"][0])


def standard_potentials_rt(temperature_k: ArrayLike) -> jax.Array:
    """Return common-R standards in COMPONENTS order; frozen source branch."""
    return _BASE(temperature_k, 1.) * DATA["source"]["gas_constant_J_mol_K"] / R


def supplementary_gas_standard_rt(temperature_k: ArrayLike) -> dict[str, jax.Array]:
    """Return NASA7 H/He/OH/SiH4 standards on the same R and one-bar basis."""
    t = jnp.asarray(temperature_k)
    result = {}
    for name, item in DATA["supplementary_gas"]["species"].items():
        coefficients = jnp.where(t <= 1000., jnp.asarray(item["low"]), jnp.asarray(item["high"]))
        a, b, c, d, e, f, g = coefficients
        h = a+b*t/2+c*t*t/3+d*t**3/4+e*t**4/5+f/t
        s = a*jnp.log(t)+b*t+c*t*t/2+d*t**3/3+e*t**4/4+g
        value = (h-s)*DATA["supplementary_gas"]["source"]["gas_constant_J_mol_K"]/R
        result[name] = jnp.where((t > 200) & (t < 6000), value, jnp.nan)
    return result
