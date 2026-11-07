"""NIST 626 CO2 thermal functions on the versioned curated TiO preset."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_tio
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/co2_v1.json"
_DATA_SHA256 = "dfac2e1ac79ff0c0dadb9010d9f187b7d4a84fd3158916e44cc60e72f94f87ba"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-co2-v1 with the exact reviewed NIST 626 thermal model.

    Only C1O2 differs from curated-tio-v1. The elemental approximation uses
    12C16O2 ground-relative internal functions at NASA molecular mass 44.0095 u,
    retained ATcT 1.172 atomization energy and NASA atomic thermal references.
    Source c2 is preserved, and no extra spin/symmetry factor or ZPE is added.

    A single C2 Gibbs representation preserves H/S/Cp derivatives. Species,
    budgets, charge treatment, 1 bar standard pressure and 1500--4000 K bounds
    are inherited. The JAX function supports scalar/array temperature and
    returns NaN outside the domain. Enable float64 for reviewed precision.
    Complete isotope, continuum and atmospheric physical accuracy remains
    unquantified. Runtime uses only immutable packaged data, without MCEG.
    """
    base = curated_tio.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("CO2 adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    dtype = base.formula_matrix.dtype
    knots = jnp.asarray(data["knots_K"], dtype=dtype)
    polynomial = jnp.asarray(data["coefficients"], dtype=dtype)
    background = jnp.asarray([data["atomic_background_nasa9"]], dtype=dtype)
    energy = jnp.asarray(data["D0_over_k_K"], dtype=dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, polynomial))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        interval = jnp.clip(jnp.searchsorted(knots, t, side="right")-1, 0, len(knots)-2)
        u = (t-knots[interval])/(knots[interval+1]-knots[interval])
        coefficients = polynomial[interval]
        free_energy = coefficients[..., 5]
        for power in range(4, -1, -1):
            free_energy = free_energy*u+coefficients[..., power]
        corrected = free_energy/t+_nasa9(t, background)[0][..., 0]-energy/t
        return base.hvector_func(temperature).at[..., index].set(jnp.where(valid, corrected, jnp.nan))

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH,
        "data_sha256": _DATA_SHA256, "base_dataset": data["base_dataset"],
        "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "co2": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
