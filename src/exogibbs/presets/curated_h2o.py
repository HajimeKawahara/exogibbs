"""Evaluated 2025 water thermodynamics on the CO2-corrected curated gas model."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_co2
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/h2o_v1.json"
_DATA_SHA256 = "6f79a001d34b1b107a34db322d642f0210ae6218b4f659be1cd8b193cbb1d6d6"


def _polynomial(value, coefficients):
    result = jnp.zeros_like(value)
    for coefficient in coefficients[::-1]:
        result = result * value + coefficient
    return result


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-h2o-v1 with the reviewed evaluated water thermal model.

    Only H2O1 differs from curated-co2-v1. The exact reviewed Cp polynomial
    and analytical H/S integrals retain the ATcT 1.172 total atomization
    anchor and NASA atomic gauge. H2(16O) internal states at aggregate mass
    represent elemental water; isotope fractionation is not modeled.

    The inherited 537-species ion-inclusive or 421-species neutral network,
    budgets, charge treatment, 1 bar standard pressure and 1500--4000 K
    domain are preserved. Scalar/array JAX evaluation and temperature
    derivatives are supported; invalid temperatures return NaN. Enable
    float64 for reviewed precision. Runtime loads packaged numerical data
    only. Scattering-scope and complete physical-accuracy limits remain.
    """
    base = curated_co2.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("H2O adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    dtype = base.formula_matrix.dtype
    thermal = data["thermal"]
    integral_h = jnp.asarray(thermal["enthalpy_integral_u"], dtype=dtype)
    integral_s = jnp.asarray(thermal["entropy_integral_u"], dtype=dtype)
    background = jnp.asarray([data["atomic_background_nasa9"]], dtype=dtype)
    energy = jnp.asarray(data["D0_over_k_K"], dtype=dtype)
    center, scale = thermal["center_K"], thermal["scale_K"]
    anchor = thermal["anchor_temperature_K"]
    u0 = jnp.asarray((anchor - center) / scale, dtype=dtype)
    h0 = thermal["anchor_H_over_R_K"] - scale * _polynomial(u0, integral_h)
    s0 = thermal["anchor_S_over_R"] - _polynomial(u0, integral_s)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, integral_h))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        u = (t - center) / scale
        enthalpy = h0 + scale * _polynomial(u, integral_h)
        entropy = s0 + _polynomial(u, integral_s) + thermal["entropy_log_coefficient"] * jnp.log(t / anchor)
        corrected = enthalpy / t - entropy + _nasa9(t, background)[0][..., 0] - energy / t
        return base.hvector_func(temperature).at[..., index].set(jnp.where(valid, corrected, jnp.nan))

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH,
        "data_sha256": _DATA_SHA256, "base_dataset": data["base_dataset"],
        "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "h2o": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
