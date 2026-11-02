"""Opt-in C2H correction of the frozen curated-gas-v1 preset."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/c2h_v1.json"
_DATA_SHA256 = "b2df12855bc20a27e3bed06e8ad4d6a421e5f47a8a682d6e3fdd74e3f698de48"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-c2h-v1 with NASA CCH thermal functions and ATcT D0.

    Only C2H1 differs from curated-gas-v1. Species, elemental budget, charge
    handling and the 1500--4000 K evaluation interval are identical. The
    standard pressure is 1 bar. Outside the interval h is NaN. Source choice
    is conditional; a physical uncertainty for high-temperature thermal
    functions or the complete mixture has not been established.
    """
    base = curated.chemsetup(ions=ions)
    data = json.loads(curated._checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("C2H correction requires its pinned curated base")
    index = base.species.index(data["species"])
    coefficients = jnp.asarray([data["nasa9_reaction_coefficients"]], dtype=base.formula_matrix.dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        t = jnp.asarray(temperature, dtype=jnp.result_type(temperature, coefficients))
        valid = jnp.isfinite(t) & (t >= lower) & (t <= upper)
        corrected = _nasa9(jnp.where(valid, t, lower), coefficients)[0][..., 0]
        return base.hvector_func(t).at[..., index].set(jnp.where(valid, corrected, jnp.nan))

    metadata = {**base.metadata, "dataset": data["id"], "data_file": _DATA_PATH,
                "data_sha256": _DATA_SHA256, "base_dataset": data["base_dataset"],
                "base_data_sha256": data["base_data_sha256"],
                "species_provenance": {**base.metadata["species_provenance"],
                                       data["species"]: data["decision"]},
                "sources": {**base.metadata["sources"], "c2h": data["sources"]},
                "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
                "interpretation": data["interpretation"]}
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
