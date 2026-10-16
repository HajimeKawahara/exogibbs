"""FastChem4 gas preset with NASA CEA ionization data for Ni+ and Co+ only."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import jax
import jax.numpy as jnp

from exogibbs.presets import fastchem4
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _load_nasa9_data, _nasa9


def chemsetup(
    path: str = "FastChem4/logK/logK.dat",
    *,
    species_default_elements: bool = True,
    element_file: Optional[str] = None,
    silent: bool = False,
) -> ChemicalSetup:
    """Build an ion-inclusive FastChem4 setup with only Ni+/Co+ replaced.

    The default network retains all 539 species. NASA9 reaction Gibbs
    energies replace Ni1+ and Co1+ in the original atomic-reference gauge,
    including the electron term at a standard pressure of 1 bar. Other
    species keep their original FastChem thermochemistry.

    Scalar and array temperatures are supported, with species on the last
    axis. Values outside the NASA interval 1000--6000 K return NaN; there is
    no extrapolation or fallback to the original fits. This interval limits
    the two replacements, not a validation of every other FastChem species.
    The selected network must contain Ni, Ni+, Co, Co+, and the electron.
    """
    baseline = fastchem4.chemsetup(
        path=path,
        species_default_elements=species_default_elements,
        element_file=element_file,
        silent=silent,
    )
    ions = ("Ni1+", "Co1+")
    required = ions + ("Ni1", "Co1", "e1-")
    missing = [name for name in required if name not in baseline.species]
    if missing:
        raise ValueError(f"NASA Ni/Co ionization requires missing species: {', '.join(missing)}")

    data = _load_nasa9_data()
    reference_names = ions + ("Ni", "Co", "e-")
    records = {name: data["records"][name] for name in reference_names}
    coefficients = jnp.asarray(
        [records[name]["coefficients"] for name in reference_names],
        dtype=baseline.formula_matrix.dtype,
    )
    indices = jnp.asarray([baseline.species.index(name) for name in ions])
    lower = max(record["bounds_K"][0] for record in records.values())
    upper = min(record["bounds_K"][1] for record in records.values())

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature, dtype=jnp.result_type(temperature, coefficients))
        g, _, _ = _nasa9(temperature, coefficients)
        # h(M+) = [G(M+) + G(e-) - G(M)] / RT in FastChem's zero-atom gauge.
        ion_h = g[..., :2] + g[..., -1:] - g[..., 2:4]
        h = baseline.hvector_func(temperature).at[..., indices].set(ion_h)
        valid = jnp.isfinite(temperature) & (temperature >= lower) & (temperature <= upper)
        return jnp.where(valid[..., None], h, jnp.nan)

    source_records = baseline.metadata["fastchem_logk_source_records"]
    metadata = {
        **baseline.metadata,
        "source": "FastChem4 with NASA CEA Ni/Co ionization",
        "model": "fastchem4_nasa9_ions",
        "nasa_replacements": {name: records[name]["source_name"] for name in ions},
        "nasa_source_url": data["source_url"],
        "nasa_source_sha256": data["source_sha256"],
        "nasa9_source_records": records,
        "temperature_range_K": (lower, upper),
        "standard_pressure_bar": data["standard_pressure_bar"],
        "fastchem_logk_source_records": {
            name: record for name, record in source_records.items() if name not in ions
        },
        "replaced_fastchem_logk_source_records": {name: source_records[name] for name in ions},
    }
    return replace(baseline, hvector_func=jax.jit(hvector_func), metadata=metadata)


__all__ = ("chemsetup",)
