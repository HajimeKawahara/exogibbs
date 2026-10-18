"""Versioned gas thermochemistry from the reviewed atomic and molecular choices."""

from __future__ import annotations

from hashlib import sha256
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets import fastchem4
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _nasa9


_DATA_PATH = "curated/gas_v1.json"
_DATA_SHA256 = "7ff5818ed9fd5dd0ab7787a4137893ef2a72cacc7b249df5b640950e93b00a38"
_KB = 1.380649e-23
_PLANCK = 6.62607015e-34
_ELECTRON_MASS = 9.1093837139e-31
_ELECTRON_LOG_FACTOR = (np.log(2.) + 1.5 * np.log(2 * np.pi * _ELECTRON_MASS * _KB / _PLANCK**2)
                        + np.log(_KB / 1.e5))


def _checked_bytes(path: str, expected: str) -> bytes:
    with open(get_data_filepath(path), "rb") as stream:
        content = stream.read()
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"Curated thermochemistry source differs from its pinned version: {path}")
    return content


def chemsetup(*, ions: bool = True) -> ChemicalSetup:
    """Return curated-gas-v1 on 1500--4000 K, with ions included by default.

    The ion-inclusive network has 537 species and 27 elements plus charge.
    ``ions=False`` retains 421 neutral species and removes the charge row
    and electron. Both use the same source choices and normalized Asplund
    2021 reference abundances. Neutral atoms and electrons have zero h;
    formation constants are dimensionless at 1 bar, with h = -ln(Kp).

    The JAX function accepts scalar or array temperatures, with species on
    the last axis. Outside this interval it returns NaN, without fallback
    or extrapolation. Enable JAX float64 before construction for the tested
    equilibrium accuracy. The interval is an evaluation domain, not a
    physical validation of every species or atmospheric application.

    Only packaged numerical data are loaded. Neither MCEG nor the FastChem
    executable is required; source choices are recorded in metadata.
    """
    if not isinstance(ions, bool):
        raise TypeError("ions must be a bool")
    data = json.loads(_checked_bytes(_DATA_PATH, _DATA_SHA256))
    for source in (data["baseline"], data["abundances"]):
        _checked_bytes(source["path"], source["sha256"])
    baseline = fastchem4.chemsetup(path=data["baseline"]["path"], silent=True)
    dtype = baseline.formula_matrix.dtype
    charge_row = baseline.elements.index("e-")
    charges = np.asarray(baseline.formula_matrix)[charge_row]
    keep = [i for i, name in enumerate(baseline.species)
            if name not in data["excluded_species"] and (ions or charges[i] == 0)]
    rows = [i for i in range(len(baseline.elements)) if ions or i != charge_row]
    species = tuple(baseline.species[i] for i in keep)
    elements = tuple(baseline.elements[i] for i in rows)
    keep = jnp.asarray(keep)
    formula_matrix = baseline.formula_matrix[jnp.asarray(rows)[:, None], keep]
    abundance_elements, abundances = fastchem4._read_element_file(data["abundances"]["path"], dtype)
    budget = jnp.asarray([abundances[abundance_elements.index(name)] for name in elements])
    budget = budget / jnp.sum(budget)

    def selected(group):
        records = {name: record for name, record in data[group].items() if name in species}
        return jnp.asarray([species.index(name) for name in records]), list(records.values())

    nasa_indices, nasa_records = selected("nasa9_reactions")
    nasa_coefficients = jnp.asarray(nasa_records, dtype=dtype)
    janaf_indices, janaf_records = selected("janaf_reactions")
    knots = jnp.asarray(janaf_records[0]["knots_K"], dtype=dtype)
    polynomial = jnp.asarray([r["coefficients"] for r in janaf_records], dtype=dtype).transpose(1, 0, 2)
    background = jnp.asarray([r["background_nasa9"] for r in janaf_records], dtype=dtype)
    shift_indices, energy_shifts = selected("logk_energy_shifts_K")
    energy_shifts = jnp.asarray(energy_shifts, dtype=dtype)
    if ions:
        ion_indices, reactions = selected("partition_reactions")
        partition_names = list(data["partition_functions"])
        partitions = list(data["partition_functions"].values())
        width = max(len(r["weights"]) for r in partitions)
        excitation = np.zeros((len(partitions), width))
        log_weights = np.full_like(excitation, -np.inf)
        for i, record in enumerate(partitions):
            count = len(record["weights"])
            excitation[i, :count] = record["excitation_K"]
            log_weights[i, :count] = np.log(record["weights"])
        excitation, log_weights = jnp.asarray(excitation, dtype=dtype), jnp.asarray(log_weights, dtype=dtype)
        neutral_ids = jnp.asarray([partition_names.index(r["neutral"]) for r in reactions])
        ion_ids = jnp.asarray([partition_names.index(r["ion"]) for r in reactions])
        signs = jnp.asarray([r["charge"] for r in reactions], dtype=dtype)
        thresholds = jnp.asarray([r["threshold_K"] for r in reactions], dtype=dtype)
    lower, upper = data["temperature_range_K"]

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, nasa_coefficients))
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        # Select a safe invalid input without halving derivatives at the bounds.
        t = jnp.where(valid, temperature, lower)
        h = baseline.hvector_func(t)[..., keep]
        h = h.at[..., nasa_indices].set(_nasa9(t, nasa_coefficients)[0])
        interval = jnp.clip(jnp.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
        u = ((t - knots[interval]) / (knots[interval + 1] - knots[interval]))[..., None]
        coefficients = polynomial[interval]
        free_energy = coefficients[..., 5]
        for power in range(4, -1, -1):
            free_energy = free_energy * u + coefficients[..., power]
        janaf_h = free_energy / t[..., None] + _nasa9(t, background)[0]
        h = h.at[..., janaf_indices].set(janaf_h)
        h = h.at[..., shift_indices].add(-energy_shifts / t[..., None])
        if ions:
            logq = logsumexp(log_weights - excitation / t[..., None, None], axis=-1)
            electron = _ELECTRON_LOG_FACTOR + 2.5 * jnp.log(t)
            ion_h = (logq[..., neutral_ids] - logq[..., ion_ids]
                     - signs * electron[..., None] + signs * thresholds / t[..., None])
            h = h.at[..., ion_indices].set(ion_h)
        return jnp.where(valid[..., None], h, jnp.nan)

    source_records = baseline.metadata["fastchem_logk_source_records"]
    provenance = {}
    for name in species:
        if name in data["decisions"]:
            provenance[name] = data["decisions"][name]
        elif name in source_records:
            provenance[name] = {"status": "inherited_unreviewed", "source_group": "FastChem4",
                                "record": source_records[name]}
        else:
            provenance[name] = {"status": "reference_gauge", "h": 0.}
    metadata = {
        "source": "ExoGibbs curated gas", "dataset": data["id"], "ions": ions,
        "relative_charge_convergence": ions,
        "data_file": _DATA_PATH, "data_sha256": _DATA_SHA256,
        "temperature_range_K": (lower, upper), "standard_pressure_bar": data["standard_pressure_bar"],
        "abundances": data["abundances"], "baseline": data["baseline"],
        "excluded_species": data["excluded_species"], "species_provenance": provenance,
        "sources": data["sources"], "source_sha256": data["source_sha256"],
        "interpretation": data["interpretation"],
    }
    return ChemicalSetup(formula_matrix, jax.jit(hvector_func), elements, species, budget, metadata)


__all__ = ("chemsetup",)
