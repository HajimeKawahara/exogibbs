"""FastChem4 condensate thermochemistry preset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exogibbs.api.chemistry import ChemicalSetup, setup_float_dtype
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem import _print_status
from exogibbs.presets.fastchem4 import chemsetup as _base_chemsetup


_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")
_COEFFS_PER_SEGMENT = 5


@dataclass(frozen=True)
class _CondensateSpeciesEntry:
    name: str
    segments: Sequence[Tuple[float, Sequence[float]]]
    components: Mapping[str, int]


def chemsetup(
    path: str = "FastChem4/logK/logK_condensates.dat",
    *,
    gas_setup: Optional[ChemicalSetup] = None,
    silent: bool = False,
) -> ChemicalSetup:
    """Build a condensate ``ChemicalSetup`` from packaged FastChem4 logK data."""

    data_path = get_data_filepath(path)
    text = open(data_path, "r", encoding="utf-8").read()
    entries = _parse_condensate_logk(text)
    species = [entry.name for entry in entries]
    validity_upper = [float(entry.segments[-1][0]) for entry in entries]
    coeffs, uppers, segment_counts = _prepare_segment_arrays(entries)

    gas = gas_setup if gas_setup is not None else _base_chemsetup(silent=True)
    if gas.elements is None:
        raise ValueError("FastChem4 gas preset did not provide an element ordering.")
    elements = list(gas.elements)

    formula_matrix = _build_formula_matrix_from_entries(entries, elements)
    if not silent:
        _print_status(species, elements, species, preset_name="fastchem4_cond")

    return _build_chemical_setup(
        coeffs,
        uppers,
        segment_counts,
        formula_matrix,
        elements,
        species,
        gas.element_vector_reference,
        path,
        validity_upper,
    )


def condensate_chemical_setup(
    *,
    gas_path: str = "FastChem4/logK/logK_wo_ions.dat",
    condensate_path: str = "FastChem4/logK/logK_condensates.dat",
    species_defalt_elements: bool = True,
    element_file: Optional[str] = None,
    silent: bool = False,
):
    """Build the production-facing FastChem4 gas-condensate setup bundle."""

    from exogibbs.api.condensate_equilibrium import build_condensate_chemical_setup

    gas_setup = _base_chemsetup(
        path=gas_path,
        species_defalt_elements=species_defalt_elements,
        element_file=element_file,
        silent=True,
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
            preset_name="fastchem4_condensate_equilibrium",
        )
    return bundle


def _build_chemical_setup(
    coeffs: np.ndarray,
    t_uppers: np.ndarray,
    segment_counts: np.ndarray,
    formula_matrix: np.ndarray,
    elements: List[str],
    species: List[str],
    element_vector_ref: jnp.ndarray,
    source_path: str,
    validity_upper: Sequence[float],
) -> ChemicalSetup:
    float_dtype = setup_float_dtype()
    coeff_array = jnp.asarray(coeffs, dtype=float_dtype)
    upper_array = jnp.asarray(t_uppers, dtype=float_dtype)
    count_array = jnp.asarray(segment_counts, dtype=jnp.int32)
    species_indices = jnp.arange(coeff_array.shape[0])
    segment_idx = jnp.arange(upper_array.shape[1])

    def select_coeffs(temp: jnp.ndarray) -> jnp.ndarray:
        valid = segment_idx[None, :] < count_array[:, None]
        candidate = temp <= upper_array
        mask = jnp.where(valid, candidate, False)
        idx = jnp.argmax(mask, axis=1)
        return coeff_array[species_indices, idx, :]

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
        formula_matrix=jnp.asarray(formula_matrix, dtype=float_dtype),
        hvector_func=jit(hvector_func),
        elements=tuple(elements),
        species=tuple(species),
        element_vector_reference=element_vector_ref,
        metadata={
            "source": "FastChem4",
            "dataset": "condensates",
            "fastchem_logk_file": source_path,
            "temperature_validity_upper": tuple(float(value) for value in validity_upper),
        },
    )


def _prepare_segment_arrays(
    entries: Sequence[_CondensateSpeciesEntry],
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


def _build_formula_matrix_from_entries(
    entries: Sequence[_CondensateSpeciesEntry],
    elements: Sequence[str],
) -> np.ndarray:
    matrix = np.zeros((len(elements), len(entries)), dtype=float)
    element_index = {element: idx for idx, element in enumerate(elements)}
    for species_idx, entry in enumerate(entries):
        for element, count in entry.components.items():
            if element in element_index:
                matrix[element_index[element], species_idx] = count
    return matrix


def _parse_condensate_logk(text: str) -> List[_CondensateSpeciesEntry]:
    entries: List[_CondensateSpeciesEntry] = []
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
        temp_idx = _next_data_index(lines, phase_idx + 1)
        temps = [float(x) for x in lines[temp_idx].split()]

        coeff_idx = temp_idx + 1
        segments: List[Tuple[float, Sequence[float]]] = []
        while len(segments) < len(temps):
            coeff_idx = _next_data_index(lines, coeff_idx)
            coeff_values = np.fromstring(lines[coeff_idx], sep=" ")
            if coeff_values.size != _COEFFS_PER_SEGMENT:
                raise ValueError(f"{species}: expected 5 coefficients")
            segments.append((temps[len(segments)], coeff_values.tolist()))
            coeff_idx += 1

        entries.append(_CondensateSpeciesEntry(species, segments, components))
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


__all__ = (
    "chemsetup",
    "condensate_chemical_setup",
)
