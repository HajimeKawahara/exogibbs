"""An explicit FastChem4/NASA9 gas model with consistent standard entropy.

This opt-in model changes 34 chemical potentials and excludes diacetyl peroxide.
It does not change the original FastChem preset. See the packaged NASA9 README
for identities, provenance, and the restricted 1500--4500 K evaluation domain.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.thermo.models import ChemicalSetup


_FASTCHEM_PATH = "FastChem4/logK/logK.dat"
_FASTCHEM_SHA256 = "eda9b9d7e62ccf7398036ee84349bd0a721b779acdc9d191cbf9fe1dd01f619c"
_NASA_PATH = "nasa9/fastchem4_hot.json"
_NASA_SHA256 = "420b32fb834ed24f62fe18b63d5879ea287afe36b468203d30af36709f6bcc36"


@dataclass(frozen=True)
class StandardThermodynamics:
    """Prepared standard-state gas properties in the chemical species order.

    Functions accept temperature in K and return arrays with a final species
    axis. Gibbs energy is divided by RT; entropy and heat capacity by R. The
    standard pressure is 1 bar. Functions return NaN outside the declared
    domain, without clipping or extrapolation. Each polynomial is smooth
    throughout this domain. ``hvector_func`` uses the elemental-reference
    gauge; ``standard_gibbs_rt`` restores the NASA atomic reference functions.
    Both give exactly the same equilibrium under elemental conservation.

    Use ``chemical_setup`` for equilibrium: the original 539-species preset
    is not compatible with this model's 538-species entropy.
    """

    chemical_setup: ChemicalSetup
    species: Tuple[str, ...]
    elements: Tuple[str, ...]
    temperature_range: Tuple[float, float]
    standard_pressure_bar: float
    element_masses_u: Mapping[str, float]
    isotope_convention: str
    hvector_func: Callable
    standard_gibbs_rt: Callable
    standard_entropy_r: Callable
    standard_cp_r: Callable
    metadata: Mapping


def _nasa9(temperature, coefficients):
    """Return g/RT, s/R and cp/R from one NASA9 interval."""
    t = jnp.asarray(temperature)[..., None]
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = coefficients.T
    entropy = (
        -a1 / (2 * t**2) - a2 / t + a3 * jnp.log(t) + a4 * t
        + a5 * t**2 / 2 + a6 * t**3 / 3 + a7 * t**4 / 4 + b2
    )
    enthalpy = (
        -a1 / t**2 + a2 * jnp.log(t) / t + a3 + a4 * t / 2
        + a5 * t**2 / 3 + a6 * t**3 / 4 + a7 * t**4 / 5 + b1 / t
    )
    cp = a1 / t**2 + a2 / t + a3 + a4 * t + a5 * t**2 + a6 * t**3 + a7 * t**4
    return enthalpy - entropy, entropy, cp


def _minimum_cp(coefficients, temperature_range):
    """Check all endpoints and real stationary points, using T / 3000."""
    scale = 3000.0
    low, high = np.asarray(temperature_range) / scale
    powers = np.arange(-2, 5)
    minima = []
    for row in coefficients:
        a = row * scale**powers
        derivative = np.array([-2 * a[0], -a[1], 0, a[3], 2 * a[4], 3 * a[5], 4 * a[6]])
        roots = np.polynomial.polynomial.polyroots(derivative)
        candidates = [low, high]
        candidates.extend(r.real for r in roots if abs(r.imag) < 1e-9 and low < r.real < high)
        minima.append(min(np.sum(a * x**powers) for x in candidates))
    return np.asarray(minima)


def prepare_fastchem_thermodynamics(
    *, temperature_range: Tuple[float, float] = (1500.0, 4500.0)
) -> StandardThermodynamics:
    """Prepare the explicitly modified 538-species FastChem4/NASA9 model.

    ``temperature_range`` may restrict, but cannot extend, 1500--4500 K.
    Thirty-four named NASA records replace inconsistent gas fits in both
    chemistry and caloric properties. Diacetyl peroxide (C4H6O4) is omitted
    because no identity-matched NASA record was found. Metadata records both
    changes. This is a defined thermochemical approximation, not a claim that
    the original FastChem4 table has complete high-temperature caloric data.

    Setup validates the pinned source bytes and analytic cp minima for every
    retained species. The floor cp/R >= 2.49 allows a 0.01 fit discrepancy from
    the ideal-gas translational limit 2.5; it never clips heat capacities.
    """
    bounds = np.array(temperature_range, dtype=float, copy=True)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not (1500 <= bounds[0] < bounds[1] <= 4500):
        raise ValueError("temperature_range must be an increasing subset of [1500, 4500] K")
    temperature_range = tuple(float(x) for x in bounds)
    with open(get_data_filepath(_NASA_PATH), "rb") as stream:
        nasa_bytes = stream.read()
    with open(get_data_filepath(_FASTCHEM_PATH), "rb") as stream:
        fastchem_bytes = stream.read()
    if sha256(nasa_bytes).hexdigest() != _NASA_SHA256 or sha256(fastchem_bytes).hexdigest() != _FASTCHEM_SHA256:
        raise ValueError("Thermochemical data differ from the validated pinned sources")
    data = json.loads(nasa_bytes)
    baseline = chemsetup(path=_FASTCHEM_PATH, silent=True)
    elements = baseline.elements
    retained = np.array([i for i, name in enumerate(baseline.species) if name not in data["excluded_species"]])
    species = tuple(baseline.species[i] for i in retained)
    matrix = np.asarray(baseline.formula_matrix)[:, retained].copy()
    records = baseline.metadata["fastchem_logk_source_records"]
    reaction = np.array([records[name]["coefficients"] if name in records else [0.0] * 5 for name in species])
    atom = np.array([data["records"][name]["coefficients"] for name in elements])
    replacements = tuple(data["replacements"])
    replacement_indices = np.array([species.index(name) for name in replacements])
    replacement = np.array([data["records"][name]["coefficients"] for name in replacements])
    for name, index in zip(replacements, replacement_indices):
        composition = data["records"][name]["stoichiometry"]
        expected = np.array([composition.get("E" if e == "e-" else e.upper(), 0) for e in elements])
        if not np.array_equal(matrix[:, index], expected):
            raise ValueError(f"NASA identity has incompatible stoichiometry: {name}")
    for record in data["records"].values():
        if record["bounds_K"] != [1000.0, 6000.0]:
            raise ValueError("Unexpected NASA9 temperature interval")
    cp_coefficients = matrix.T @ atom[:, :7]
    cp_coefficients[:, 2] += reaction[:, 1]
    cp_coefficients[:, 3] += 2 * reaction[:, 3]
    cp_coefficients[:, 4] += 6 * reaction[:, 4]
    cp_coefficients[replacement_indices] = replacement[:, :7]
    minimum_cp = _minimum_cp(cp_coefficients, temperature_range)
    if not np.all(np.isfinite(minimum_cp)) or np.any(minimum_cp < 2.49):
        invalid = [name for name, cp in zip(species, minimum_cp) if not np.isfinite(cp) or cp < 2.49]
        raise ValueError(f"Invalid ideal-gas heat capacities in requested domain: {invalid}")
    dtype = baseline.formula_matrix.dtype
    matrix = jnp.asarray(matrix, dtype=dtype)
    reaction = jnp.asarray(reaction, dtype=dtype)
    atom = jnp.asarray(atom, dtype=dtype)
    replacement = jnp.asarray(replacement, dtype=dtype)
    replacement_indices = jnp.asarray(replacement_indices)

    def properties(temperature):
        t = jnp.asarray(temperature)[..., None]
        a1, a2, a3, a4, a5 = reaction.T
        log_k = a1 / t + a2 * jnp.log(t) + a3 + a4 * t + a5 * t**2
        reaction_s = a2 * (jnp.log(t) + 1) + a3 + 2 * a4 * t + 3 * a5 * t**2
        reaction_cp = a2 + 2 * a4 * t + 6 * a5 * t**2
        atom_g, atom_s, atom_cp = _nasa9(temperature, atom)
        nasa_g, nasa_s, nasa_cp = _nasa9(temperature, replacement)
        g = (atom_g @ matrix - log_k).at[..., replacement_indices].set(nasa_g)
        entropy = (atom_s @ matrix + reaction_s).at[..., replacement_indices].set(nasa_s)
        cp = (atom_cp @ matrix + reaction_cp).at[..., replacement_indices].set(nasa_cp)
        # Keep the elemental gauge used by the original chemical solver.
        h = (-log_k).at[..., replacement_indices].set(nasa_g - (atom_g @ matrix)[..., replacement_indices])
        valid = jnp.isfinite(t) & (t >= bounds[0]) & (t <= bounds[1])
        return tuple(jnp.where(valid, value, jnp.nan) for value in (h, g, entropy, cp))

    hvector = jax.jit(lambda t: properties(t)[0])
    metadata = MappingProxyType({
        "model": "fastchem4_nasa9_hot_538",
        "source_species_count": len(baseline.species),
        "fastchem_logk_file": _FASTCHEM_PATH,
        "fastchem_sha256": _FASTCHEM_SHA256,
        "nasa_source_url": data["source_url"],
        "nasa_source_sha256": data["source_sha256"],
        "nasa_excerpt_sha256": _NASA_SHA256,
        "nasa_replacements": MappingProxyType(data["replacements"]),
        "excluded_species": MappingProxyType(data["excluded_species"]),
        "minimum_standard_cp_r": float(np.min(minimum_cp)),
        "heat_capacity_fit_tolerance_r": 0.01,
        "temperature_range_K": temperature_range,
        "standard_pressure_bar": 1.0,
    })
    setup = ChemicalSetup(
        formula_matrix=matrix,
        hvector_func=hvector,
        species=species,
        elements=elements,
        element_vector_reference=baseline.element_vector_reference,
        metadata=metadata,
    )
    return StandardThermodynamics(
        chemical_setup=setup,
        species=species,
        elements=elements,
        temperature_range=temperature_range,
        standard_pressure_bar=1.0,
        element_masses_u=MappingProxyType({name: data["records"][name]["mass_u"] for name in elements}),
        isotope_convention=data["isotope_convention"],
        hvector_func=hvector,
        standard_gibbs_rt=jax.jit(lambda t: properties(t)[1]),
        standard_entropy_r=jax.jit(lambda t: properties(t)[2]),
        standard_cp_r=jax.jit(lambda t: properties(t)[3]),
        metadata=metadata,
    )
