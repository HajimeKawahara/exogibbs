"""Modern TiO thermal functions and ground energy for the curated gas preset."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_c2h
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/tio_v1.json"
_DATA_SHA256 = "52311c1ad348dc0098a34c17cfced9a175ff0b551bcbfc3819292271762c27d1"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-tio-v1 with TOTO state thermodynamics and Jain D0.

    Only O1Ti1 differs from curated-c2h-v1. The element-aggregated TiO model
    uses 48Ti16O internal levels and NASA elemental translation/atomic thermal
    references; it is not a complete natural-isotope calculation. Its central
    6.824 eV cutoff and energy anchor share the lowest rovibronic zero.

    Species, budgets, charge treatment, 1 bar standard pressure and the
    1500--4000 K domain are inherited. The JAX function accepts scalar or array
    temperatures and returns NaN outside this domain. Enable JAX float64 for
    the reviewed numerical precision. The single C2 Gibbs representation
    preserves thermodynamic derivatives; complete physical accuracy remains
    unquantified. Runtime loads only packaged numerical data.
    """
    base = curated_c2h.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("TiO adoption requires its pinned curated base")
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
        "sources": {**base.metadata["sources"], "tio": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
