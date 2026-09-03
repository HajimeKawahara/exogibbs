"""FastChem condensate preset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from exogibbs.equilibrium.condensate.setup import (
    build_condensate_chemical_setup,
)
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem import chemsetup as _base_chemsetup
from exogibbs.presets.fastchem import _print_status
from exogibbs.thermo.stoichiometry import build_formula_matrix
from exogibbs.thermo.models import ChemicalSetup


_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")
_COEFFS_PER_SEGMENT = 5

@dataclass
class _SpeciesEntry:
    name: str
    segments: Sequence[Tuple[float, Sequence[float]]]
    components: Mapping[str, int]


def chemsetup(
    path: str = "fastchem/logK/logK_condensates.dat",
    silent: bool = False,
    *,
    gas_setup: Optional[ChemicalSetup] = None,
) -> ChemicalSetup:
    """Build a ``ChemicalSetup`` from FastChem condensate data."""

    data_path = get_data_filepath(path)
    with open(data_path, "r", encoding="utf-8") as stream:
        text = stream.read()
    entries = _parse_condensate_logk(text)
    species = [entry.name for entry in entries]
    components = {entry.name: dict(entry.components) for entry in entries}
    coeffs, uppers, segment_counts = _prepare_segment_arrays(entries)

    # inherit the element ordering and reference vector from the gas-phase preset
    gas = gas_setup if gas_setup is not None else _base_chemsetup(silent=True)
    if gas.elements is None:
        raise ValueError("fastchem gas preset did not provide an element ordering.")
    elements = list(gas.elements)
    element_vector_ref = gas.element_vector_reference

    _validate_condensate_elements(entries, elements)
    formula_matrix = build_formula_matrix(components, elements)
    if not silent:
        _print_status(species, elements, species, preset_name="fastchem_cond")


    setup = _build_chemical_setup(
        coeffs,
        uppers,
        segment_counts,
        formula_matrix,
        elements,
        species,
        element_vector_ref,
    )

    return setup


def _validate_condensate_elements(
    entries: Sequence[_SpeciesEntry],
    elements: Sequence[str],
) -> None:
    available = set(elements)
    incompatible = [
        (entry.name, sorted(set(entry.components) - available))
        for entry in entries
        if set(entry.components) - available
    ]
    if not incompatible:
        return
    missing = sorted({element for _, values in incompatible for element in values})
    examples = ", ".join(
        f"{name} ({'/'.join(values)})" for name, values in incompatible[:3]
    )
    raise ValueError(
        "Condensate data are incompatible with gas_setup.elements; "
        f"missing elements: {', '.join(missing)}. "
        f"Affected species include: {examples}."
    )


def condensate_chemical_setup(
    *,
    gas_path: str = "fastchem/logK/logK.dat",
    condensate_path: str = "fastchem/logK/logK_condensates.dat",
    species_defalt_elements: bool = True,
    element_file: Optional[str] = None,
    silent: bool = False,
    species_default_elements: Optional[bool] = None,
):
    """Build the production-facing FastChem gas-condensate setup bundle."""

    gas_setup = _base_chemsetup(
        path=gas_path,
        species_defalt_elements=species_defalt_elements,
        element_file=element_file,
        silent=True,
        species_default_elements=species_default_elements,
    )
    condensate_setup = chemsetup(
        path=condensate_path,
        gas_setup=gas_setup,
        silent=True,
    )
    bundle = build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )
    if not silent:
        _print_status(
            list(bundle.condensate_species),
            list(bundle.elements),
            list(bundle.gas_species) + list(bundle.condensate_species),
            preset_name="fastchem_condensate_equilibrium",
        )
    return bundle


def _build_chemical_setup(
    coeffs: jnp.ndarray,
    t_uppers: jnp.ndarray,
    segment_counts: jnp.ndarray,
    formula_matrix: np.ndarray,
    elements: List[str],
    species: List[str],
    element_vector_ref: jnp.ndarray,
) -> ChemicalSetup:
    coeffs = jnp.asarray(coeffs)
    t_uppers = jnp.asarray(t_uppers)
    segment_counts = jnp.asarray(segment_counts, dtype=jnp.int32)
    species_indices = jnp.arange(coeffs.shape[0])
    segment_idx = jnp.arange(t_uppers.shape[1])

    def select_coeffs(temp: jnp.ndarray) -> jnp.ndarray:
        valid = segment_idx[None, :] < segment_counts[:, None]
        candidate = temp <= t_uppers
        mask = jnp.where(valid, candidate, False)
        idx = jnp.argmax(mask, axis=1)
        return coeffs[species_indices, idx, :]

    def logk_at(temp: jnp.ndarray) -> jnp.ndarray:
        c = select_coeffs(temp)
        a1, a2, a3, a4, a5 = [c[:, i] for i in range(_COEFFS_PER_SEGMENT)]
        return a1 / temp + a2 * jnp.log(temp) + a3 + a4 * temp + a5 * temp * temp

    def hvector_func(T: jnp.ndarray) -> jnp.ndarray:
        T = jnp.asarray(T)
        if T.ndim == 0:
            return -logk_at(T)
        flat_t = T.reshape(-1)
        hvector = -vmap(logk_at)(flat_t)
        return hvector.reshape(T.shape + (hvector.shape[-1],))

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=jit(hvector_func),
        elements=tuple(elements),
        species=tuple(species),
        element_vector_reference=element_vector_ref,
        metadata={"source": "fastchem v3.1.3", "dataset": "condensates"},
    )


def _prepare_segment_arrays(
    entries: Sequence[_SpeciesEntry],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.array([len(entry.segments) for entry in entries], dtype=np.int32)
    max_segments = int(counts.max(initial=1))
    coeffs = np.zeros((len(entries), max_segments, _COEFFS_PER_SEGMENT), dtype=np.float64)
    uppers = np.full((len(entries), max_segments), -np.inf, dtype=np.float64)

    for idx, entry in enumerate(entries):
        for seg_idx, (t_upper, coeff) in enumerate(entry.segments):
            coeffs[idx, seg_idx, :] = coeff
            uppers[idx, seg_idx] = t_upper
        uppers[idx, len(entry.segments) - 1] = np.inf

    return coeffs, uppers, counts


def _parse_condensate_logk(text: str) -> List[_SpeciesEntry]:
    entries: List[_SpeciesEntry] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        match = _SPECIES_PATTERN.match(line)
        if not match or ":" not in line:
            i += 1
            continue

        species = match.group(1)
        components = _parse_components(line)

        phase_idx = _next_data_index(lines, i + 1)
        phases = lines[phase_idx].strip()

        temp_idx = _next_data_index(lines, phase_idx + 1)
        temps = [float(x) for x in lines[temp_idx].split()]

        if len(temps) != len(phases):
            raise ValueError(f"{species}: temperature/phase mismatch")

        coeff_idx = temp_idx + 1
        segments: List[Tuple[float, Sequence[float]]] = []
        for phase_pos in range(len(phases)):
            coeff_idx = _next_data_index(lines, coeff_idx)
            coeff_values = np.fromstring(lines[coeff_idx], sep=" ")
            if coeff_values.size != _COEFFS_PER_SEGMENT:
                raise ValueError(f"{species}: expected 5 coefficients")
            segments.append((temps[phase_pos], coeff_values.tolist()))
            coeff_idx += 1

        entries.append(_SpeciesEntry(species, segments, components))
        i = coeff_idx

    return entries


def _parse_components(line: str) -> Dict[str, int]:
    after_colon = line.split(":", 1)[1] if ":" in line else ""
    before_hash = after_colon.split("#", 1)[0]
    tokens = before_hash.strip().split()
    comp: Dict[str, int] = {}
    for k in range(0, len(tokens), 2):
        if k + 1 >= len(tokens):
            break
        element = tokens[k]
        count_str = tokens[k + 1]
        try:
            comp[element] = int(float(count_str))
        except ValueError:
            continue
    return comp


def _next_data_index(lines: Sequence[str], start: int) -> int:
    idx = start
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("#"):
            return idx
        idx += 1
    raise ValueError("unexpected end of file")
