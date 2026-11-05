"""NASA HO2 thermal functions with balanced ATcT energy on the curated KH base."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_kh
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/ho2_v1.json"
_DATA_SHA256 = "de1050836d2adfb0853594fb3c7f25253af758a9c35f40458baa32737bbf534b"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-ho2-v1 with the exact reviewed HO2 NASA/ATcT reaction.

    Only HO2 differs from curated-kh-v1. NASA thermal functions retain their
    atomic reference and native dimensionless convention; the balanced ATcT
    1.172 ground-relative energy is inserted once. No new fit is introduced.

    Species, elemental budgets, charge treatment and the one-bar standard
    pressure are inherited. Scalar/array JAX thermodynamics and derivatives
    cover 1500--4000 K; invalid temperatures return NaN. Use float64 for
    reviewed precision. Runtime needs only packaged data, without MCEG.
    Full isotope, continuum and atmospheric physical accuracy is unquantified.
    """
    base = curated_kh.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("HO2 adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    coefficients = jnp.asarray([data["formation_nasa9_coefficients"]], dtype=base.formula_matrix.dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, coefficients))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        corrected = _nasa9(t, coefficients)[0][..., 0]
        return base.hvector_func(temperature).at[..., index].set(jnp.where(valid, corrected, jnp.nan))

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH, "data_sha256": _DATA_SHA256,
        "base_dataset": data["base_dataset"], "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "ho2": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
