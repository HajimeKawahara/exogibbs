"""Measured ground-state TiO2 RRHO thermal correction on the curated PS base."""

from dataclasses import replace
import json
import math

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_ps
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/tio2_v1.json"
_DATA_SHA256 = "7e07c9ed020ef1cd377f00e95eb5bbaee451788a3d75d377d1872fe66ee9aa43"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-tio2-v1 with the reviewed measured TiO2 thermal correction.

    Only TiO2 changes from curated-ps-v1. Add the ground-state RRHO molecular
    G-H0 difference from the NASA thermal proxy, retaining source D0, every
    inherited fit residual and all atomic potentials. No extra energy shift,
    electronic manifold or zero-point correction is introduced.

    The RRHO approximation uses measured principal-isotopologue rotation and
    gas fundamentals as harmonic frequencies, JANAF mean mass, symmetry two
    and spin weight one. It does not establish full high-temperature accuracy.
    Species, inventories, charge treatment and one-bar standard pressure are
    inherited. Scalar/array JAX thermodynamics cover 1500--4000 K; invalid
    temperatures return NaN. Use float64 for reviewed precision. Runtime uses
    packaged data only, without MCEG or external state/source files.
    """
    base = curated_ps.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("TiO2 adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    dtype = base.formula_matrix.dtype
    coefficients = jnp.asarray([data["nasa_molecular_thermal_coefficients"]], dtype=dtype)
    k, h = 1.380649e-23, 6.62607015e-34
    theta_rot_product = math.prod(b*1e6*h/k for b in data["rotational_constants_MHz"])
    mass = data["mass_u"]*1.66053906660e-27
    translation = 2*math.pi*mass*k/h**2
    theta_vib = jnp.asarray(data["frequencies_cm-1"], dtype=dtype)*(h*29979245800/k)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, coefficients))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        t = jnp.where(valid, temperature, lower)
        log_trans = 1.5*jnp.log(translation*t)+jnp.log(k*t/data["standard_pressure_Pa"])
        log_rot = .5*jnp.log(math.pi*t**3/theta_rot_product)-math.log(data["symmetry_number"])
        vibrational = jnp.sum(jnp.log(-jnp.expm1(-theta_vib/t[..., None])), axis=-1)
        rrho = -log_trans-log_rot+vibrational
        nasa = _nasa9(t, coefficients)[0][..., 0]
        correction = jnp.where(valid, rrho-nasa, jnp.nan)
        return base.hvector_func(temperature).at[..., index].add(correction)

    metadata = {
        **base.metadata, "dataset": data["id"], "data_file": _DATA_PATH, "data_sha256": _DATA_SHA256,
        "base_dataset": data["base_dataset"], "base_data_sha256": data["base_data_sha256"],
        "species_provenance": {**base.metadata["species_provenance"], data["species"]: data["decision"]},
        "sources": {**base.metadata["sources"], "tio2": data["sources"]},
        "source_sha256": {**base.metadata["source_sha256"], **data["source_sha256"]},
        "interpretation": data["interpretation"],
    }
    return replace(base, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
