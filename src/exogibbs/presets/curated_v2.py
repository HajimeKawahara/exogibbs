"""Reviewed molecular energies and evaluated natural-O2 thermals on curated CH2."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_ch2
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/gas_v2.json"
_DATA_SHA256 = "8096a1dc9c0f6e15e222b7c6c9dc6d5693a1a6904913affb463a144034616261"


def _polynomial(value, coefficients):
    result = jnp.zeros_like(value)
    for coefficient in coefficients[::-1]:
        result = result * value + coefficient
    return result


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-gas-v2 with 28 reviewed molecular corrections.

    Add each physical energy difference once to the inherited reaction
    potential. Only O2 also receives the evaluated 2025 natural-isotope
    molecular thermal function minus its NASA thermal proxy. Atomic thermals
    and the complete inherited fit residual remain unchanged. No additional
    zero-point, spin, symmetry or isotope factor is applied.

    The 537-species ion-inclusive or 421-species neutral network, inventories,
    charge treatment and one-bar standard pressure are inherited from
    curated-ch2-v1. Scalar/array JAX thermodynamics cover 1500--4000 K;
    invalid temperatures return NaN. Enable float64 for reviewed precision.
    Runtime uses packaged numerical data only, without MCEG. Complete physical
    uncertainty and atmospheric accuracy remain unquantified; source-specific
    thermal limitations are recorded.
    """
    base = curated_ch2.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("Curated v2 requires its pinned curated CH2 base")
    shifts = data["energy_shifts_J_mol"]
    names = list(shifts)
    if len(names) != 28 or len(set(names)) != len(names):
        raise ValueError("Curated v2 requires 28 unique energy corrections")
    indices = jnp.asarray([base.species.index(name) for name in names])
    dtype = base.formula_matrix.dtype
    energy = jnp.asarray([shifts[name] / data["gas_constant_J_mol_K"] for name in names], dtype=dtype)
    oxygen = base.species.index("O2")
    thermal = data["thermal"]
    integral_h = jnp.asarray(thermal["enthalpy_integral_u"], dtype=dtype)
    integral_s = jnp.asarray(thermal["entropy_integral_u"], dtype=dtype)
    proxy = jnp.asarray([data["nasa_molecular_thermal_coefficients"]], dtype=dtype)
    center, scale = thermal["center_K"], thermal["scale_K"]
    anchor = thermal["anchor_temperature_K"]
    u0 = jnp.asarray((anchor - center) / scale, dtype=dtype)
    h0 = thermal["anchor_H_over_R_K"] - scale * _polynomial(u0, integral_h)
    s0 = thermal["anchor_S_over_R"] - _polynomial(u0, integral_s)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, energy))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        correction = jnp.where(valid[..., None], energy / t[..., None], jnp.nan)
        result = base.hvector_func(temperature).at[..., indices].add(correction)
        u = (t - center) / scale
        enthalpy = h0 + scale * _polynomial(u, integral_h)
        entropy = (s0 + _polynomial(u, integral_s)
                   + thermal["entropy_log_coefficient"] * jnp.log(t / anchor))
        thermal_delta = enthalpy / t - entropy - _nasa9(t, proxy)[0][..., 0]
        return result.at[..., oxygen].add(jnp.where(valid, thermal_delta, jnp.nan))

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH,
        "data_sha256": _DATA_SHA256, "base_dataset": data["base_dataset"],
        "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {
            **base.metadata["species_provenance"],
            **data["decisions"],
        },
        "sources": {**base.metadata["sources"], "gas_v2": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
