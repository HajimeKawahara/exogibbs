"""Balanced ATcT CH2 energy with the current thermal fit and source residual."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_c2h2
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup


_DATA_PATH = "curated/ch2_v1.json"
_DATA_SHA256 = "a2d353156b957c38b6f3ff44af1bf735d60e58cf1089f99b7b9358e01ad81536"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-ch2-v1 with the reviewed balanced CH2 energy update.

    Only CH2 changes from curated-c2h2-v1. Subtract delta(D0)/(RT) once
    from its formation G/(RT); the literal reaction thermal fit,
    entropy and heat capacity are retained. There is no new fit or spin,
    isotope or partition-table correction.

    Species, inventories, charge treatment and one-bar standard pressure
    are inherited. Scalar/array JAX thermodynamics cover 1500--4000 K;
    invalid temperatures return NaN. Use float64 for reviewed precision.
    Runtime requires only packaged data, without MCEG. Complete physical
    uncertainty and atmospheric accuracy are unquantified.
    """
    base = curated_c2h2.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("CH2 adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    shift = jnp.asarray(data["delta_D0_over_R_K"], dtype=base.formula_matrix.dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, shift))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        correction = jnp.where(valid, -shift/t, jnp.nan)
        return base.hvector_func(temperature).at[..., index].add(correction)

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH, "data_sha256": _DATA_SHA256,
        "base_dataset": data["base_dataset"], "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "ch2": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
