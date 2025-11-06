"""FastChem condensate preset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem import (
    _elements_ref_AAG21,
    _generate_formula_matrix,
    _set_element_species as _base_set_element_species,
    _set_elements as _base_set_elements,
)

_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")
_COEFFS_PER_SEGMENT = 5


@dataclass
class _SpeciesEntry:
    name: str
    segments: Sequence[Tuple[float, Sequence[float]]]
    components: Mapping[str, int]


def chemsetup(path: str = "fastchem/logK/logK_condensates.dat") -> ChemicalSetup:
    """Build a ``ChemicalSetup`` from FastChem condensate data."""

    data_path = get_data_filepath(path)
    text = open(data_path, "r", encoding="utf-8").read()
    entries = _parse_condensate_logk(text)

    species_molecule = [entry.name for entry in entries]
    components_molecule = {entry.name: dict(entry.components) for entry in entries}

    coeffs_molecule, uppers_molecule, segment_counts = _prepare_segment_arrays(entries)

    elements = _base_set_elements(components_molecule)
    element_vector_ref = _elements_ref_AAG21()
    species_element, components_element, acoeff_element = _base_set_element_species(elements)
    coeffs_element = (
        np.array([acoeff_element[spec] for spec in species_element], dtype=np.float64)[
            :, None, :
        ]
    )
    uppers_element = np.full((coeffs_element.shape[0], 1), np.inf, dtype=np.float64)
    counts_element = np.ones(coeffs_element.shape[0], dtype=np.int32)

    coeffs_all, uppers_all, counts_all = _combine_segment_data(
        coeffs_element,
        uppers_element,
        counts_element,
        coeffs_molecule,
        uppers_molecule,
        segment_counts,
    )

    components = {**components_element, **components_molecule}
    species = species_element + species_molecule

    formula_matrix = _generate_formula_matrix(components, elements)
    setup = _build_chemical_setup(
        coeffs_all,
        uppers_all,
        counts_all,
        formula_matrix,
        elements,
        species,
        element_vector_ref,
    )

    return setup


def _build_chemical_setup(
    coeffs: jnp.ndarray,
    t_uppers: jnp.ndarray,
    segment_counts: jnp.ndarray,
    formula_matrix: np.ndarray,
    elements: List[str],
    species: List[str],
    element_vector_ref: jnp.ndarray,
) -> ChemicalSetup:
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
        return -vmap(logk_at)(T)

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=jit(hvector_func),
        elements=tuple(elements),
        species=tuple(species),
        element_vector_reference=element_vector_ref,
        metadata={"source": "fastchem v3.1.3", "dataset": "condensates"},
    )


def _combine_segment_data(
    coeffs_element: np.ndarray,
    uppers_element: np.ndarray,
    counts_element: np.ndarray,
    coeffs_molecule: np.ndarray,
    uppers_molecule: np.ndarray,
    segment_counts_molecule: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    max_segments = max(coeffs_molecule.shape[1], coeffs_element.shape[1])
    coeffs_element, uppers_element = _pad_segments(
        coeffs_element, uppers_element, max_segments
    )
    coeffs_molecule, uppers_molecule = _pad_segments(
        coeffs_molecule, uppers_molecule, max_segments
    )

    coeffs_all = np.concatenate([coeffs_element, coeffs_molecule], axis=0)
    uppers_all = np.concatenate([uppers_element, uppers_molecule], axis=0)
    counts_all = np.concatenate([counts_element, segment_counts_molecule]).astype(np.int32)

    return jnp.asarray(coeffs_all), jnp.asarray(uppers_all), jnp.asarray(counts_all)


def _pad_segments(
    coeffs: np.ndarray, uppers: np.ndarray, target_segments: int
) -> Tuple[np.ndarray, np.ndarray]:
    if coeffs.shape[1] == target_segments:
        return coeffs, uppers

    pad = target_segments - coeffs.shape[1]
    coeffs_padded = np.pad(
        coeffs,
        ((0, 0), (0, pad), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    uppers_padded = np.pad(
        uppers,
        ((0, 0), (0, pad)),
        mode="constant",
        constant_values=-np.inf,
    )
    return coeffs_padded, uppers_padded


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

