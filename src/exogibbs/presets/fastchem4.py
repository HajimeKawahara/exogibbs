"""FastChem4 gas thermochemistry preset.

This module mirrors :mod:`exogibbs.presets.fastchem` while reading the packaged
FastChem4 gas data under ``src/exogibbs/data/FastChem4``. It builds native
``ChemicalSetup`` objects and does not require a FastChem4 runtime.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem import (
    _default_elements,
    _elements_ref_AAG21,
    _parse_fastchem_coeffs_with_metadata,
    _resolve_species_default_elements,
    _restrict_species_to_elements,
    _set_element_species,
    _print_status,
    logk,
)
from exogibbs.thermo.stoichiometry import build_formula_matrix
from exogibbs.thermo.models import ChemicalSetup, setup_float_dtype
from exogibbs.utils.nameparser import set_elements_from_components


def chemsetup(
    path: str = "FastChem4/logK/logK_wo_ions.dat",
    species_defalt_elements: bool = True,
    element_file: Optional[str] = None,
    silent: bool = False,
    *,
    species_default_elements: Optional[bool] = None,
) -> ChemicalSetup:
    """Build a gas-phase ``ChemicalSetup`` from packaged FastChem4 logK data."""

    species_defalt_elements = _resolve_species_default_elements(
        species_defalt_elements=species_defalt_elements,
        species_default_elements=species_default_elements,
    )
    float_dtype = setup_float_dtype()
    data_path = get_data_filepath(path)
    text = open(data_path, "r", encoding="utf-8").read()
    acoeff_molecule, components_molecule, source_records_molecule = (
        _parse_fastchem_coeffs_with_metadata(text, source_file=path)
    )

    if species_defalt_elements:
        if element_file is not None and not silent:
            print("WARNING: element_file is ignored when species_defalt_elements is True.")
        elements = _default_elements()
        element_vector_ref = _elements_ref_AAG21(float_dtype)
    elif element_file is not None:
        elements, element_vector_ref = _read_element_file(element_file, float_dtype)
    else:
        elements = _set_elements_with_adding_Ge(components_molecule)
        element_vector_ref = []

    acoeff_molecule, components_molecule = _restrict_species_to_elements(
        acoeff_molecule,
        components_molecule,
        elements,
    )
    source_records_molecule = {
        species: source_records_molecule[species]
        for species in components_molecule
        if species in source_records_molecule
    }
    species_molecule = list(acoeff_molecule.keys())
    species_element, components_element, acoeff_element = _set_element_species(elements)
    acoeff = {**acoeff_element, **acoeff_molecule}
    species = species_element + species_molecule
    components = {**components_element, **components_molecule}
    formula_matrix = jnp.asarray(build_formula_matrix(components, elements), dtype=float_dtype)

    if not silent:
        _print_status(species_molecule, elements, species, preset_name="fastchem4")

    ccoeff_array = jnp.asarray([acoeff[spec] for spec in species], dtype=float_dtype)
    vmap_logk = vmap(logk, in_axes=(None, 0), out_axes=0)

    def hvector_func(T: Union[float, jnp.ndarray]) -> jnp.ndarray:
        T = jnp.asarray(T)
        hvector = -vmap_logk(T, ccoeff_array)
        if T.ndim == 0:
            return hvector
        return jnp.moveaxis(hvector, 0, -1)

    hvector_func_jit = jit(hvector_func)
    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=hvector_func_jit,
        elements=tuple(elements),
        species=tuple(species),
        element_vector_reference=element_vector_ref,
        metadata={
            "source": "FastChem4",
            "dataset": "gas",
            "fastchem_logk_file": path,
            "fastchem_element_file": (
                element_file
                if element_file is not None
                else "FastChem4/element_abundances/asplund_2021.dat"
            ),
            "fastchem_species_default_elements": str(species_defalt_elements),
            "fastchem_logk_source_records": source_records_molecule,
        },
    )


def _read_element_file(path: str, dtype: jnp.dtype) -> tuple[List[str], jnp.ndarray]:
    import pandas as pd

    element_df = pd.read_csv(
        get_data_filepath(path),
        sep=r"\s+",
        comment="#",
        header=None,
        names=["element", "abundance"],
    )
    elements = element_df["element"].tolist()[1:] + ["e-"]
    abundances = np.asarray(element_df["abundance"].tolist()[1:], dtype=float)
    element_vector_ref = jnp.asarray(
        np.concatenate((10.0 ** (abundances - 12.0), [0.0])),
        dtype=dtype,
    )
    return elements, element_vector_ref


def _set_elements_with_adding_Ge(components: Dict[str, Dict[str, int]]) -> List[str]:
    element_set = set_elements_from_components(components)
    if "Ge" not in element_set:
        return sorted(list(element_set) + ["Ge"])
    return sorted(list(element_set))


__all__ = (
    "chemsetup",
)
