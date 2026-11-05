"""Spectroscopic KH energy with unchanged thermal functions on the OH base."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_oh
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup


_DATA_PATH = "curated/kh_v1.json"
_DATA_SHA256 = "d808d0b1e5f65fabcb09759c0961b5974d2de99e7d208b914dc6a73169f52328"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-kh-v1 with the reviewed KH spectroscopic energy.

    Only KH changes from curated-oh-v1. Subtract delta(D0)/(RT) once from
    its formation G/(RT), preserving all inherited thermal conventions,
    entropy and heat capacity. No new fit or additional spin factor is used.
    Species, inventories, charge and one-bar standard pressure are inherited.
    Scalar/array JAX thermodynamics cover 1500--4000 K; invalid temperatures
    return NaN. Reviewed precision requires float64. Runtime uses packaged
    data without MCEG; complete physical uncertainty is not established.
    """
    base = curated_oh.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("KH adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    shift = jnp.asarray(data["delta_D0_over_R_K"], dtype=base.formula_matrix.dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, shift))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        values = base.hvector_func(temperature)
        corrected = values[..., index] - shift / t
        return values.at[..., index].set(jnp.where(valid, corrected, jnp.nan))

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH, "data_sha256": _DATA_SHA256,
        "base_dataset": data["base_dataset"], "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "kh": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
