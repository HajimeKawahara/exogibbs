"""Resolved POPS PS thermals with retained energy on the curated CN base."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp

from exogibbs.presets import curated_cn
from exogibbs.presets.curated_v1 import _checked_bytes
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/ps_v1.json"
_DATA_SHA256 = "d070f0fd64fe89039994ff37ec2d1f6586ed66190f97f48c38b6e13f196d90d6"


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-ps-v1 with the reviewed PS molecular thermal difference.

    A degree-18 Chebyshev representation of the fixed-cutoff POPS partition
    sum replaces the NASA/Gurvich molecular thermal contribution. Add this
    difference to the literal inherited fit, retaining the Lodders energy,
    atomic functions and source-fit residual. Runtime loads only packaged
    coefficients; it does not need a state file, SciPy or MCEG.

    The representative-isotope, natural-mass proxy is limited to 1500--4000 K.
    The a4Pi potential and complete physical uncertainty remain unresolved.
    Scalar/array JAX potentials preserve the inherited network and return
    NaN outside the domain. Use float64 for the reviewed numerical precision.
    """
    base = curated_cn.chemsetup(ions=ions)
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    if (base.metadata["dataset"] != data["base_dataset"]
            or base.metadata["data_sha256"] != data["base_data_sha256"]):
        raise ValueError("PS adoption requires its pinned curated base")
    index = base.species.index(data["species"])
    representation = data["log_partition_representation"]
    coefficients = jnp.asarray(representation["coefficients"],dtype=base.formula_matrix.dtype)
    nasa = jnp.asarray([data["NASA_molecular_coefficients"]],dtype=base.formula_matrix.dtype)
    lower, upper = data["temperature_range_K"]
    ratio = data["NASA_native_R_J_mol_K"]/data["gas_constant_J_mol_K"]
    h0_over_r = data["NASA_molecular_H0_gauge_J_mol"]/data["gas_constant_J_mol_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature,dtype=jnp.result_type(temperature,coefficients))
        valid = jnp.isfinite(temperature)&(temperature>=lower)&(temperature<=upper)
        t = jnp.where(valid,temperature,lower)
        log_t = jnp.log(t)
        x = (log_t-representation["ln_temperature_center"])/representation["ln_temperature_half_width"]
        first, second = jnp.zeros_like(t), jnp.zeros_like(t)
        for coefficient in coefficients[:0:-1]:
            first, second = coefficient+2*x*first-second, first
        residual = coefficients[0]+x*first-second
        log_q = representation["leading_ln_T_coefficient"]*log_t+residual
        log_translation = data["translation_log_constant"]+2.5*log_t
        correction = -log_q-log_translation-ratio*_nasa9(t,nasa)[0][...,0]+h0_over_r/t
        return base.hvector_func(temperature).at[...,index].add(jnp.where(valid,correction,jnp.nan))

    metadata = {
        **base.metadata,"dataset":data["id"],"data_file":_DATA_PATH,"data_sha256":_DATA_SHA256,
        "base_dataset":data["base_dataset"],"base_data_sha256":data["base_data_sha256"],
        "species_provenance":{**base.metadata["species_provenance"],data["species"]:data["decision"]},
        "sources":{**base.metadata["sources"],"ps":data["sources"]},
        "source_sha256":{**base.metadata["source_sha256"],**data["source_sha256"]},
        "interpretation":data["interpretation"],
    }
    return replace(base,hvector_func=jax.jit(hvector_func),metadata=metadata)


__all__ = ("chemsetup",)
