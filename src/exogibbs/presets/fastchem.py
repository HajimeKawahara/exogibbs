import re
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax import jit
from typing import Union
from typing import Dict
from typing import List
from typing import Tuple

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.io.load_data import get_data_filepath

_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")


def chemsetup(path="fastchem/logK/logK.dat") -> ChemicalSetup:
    """
    Prepare a JAX-friendly ChemicalSetup from JANAF-like Gibbs matrices.

    Args:
        path (str): Path to the FastChem logK data file.

    Returns:
        ChemicalSetup: The chemical setup object.

    Notes:
        The species in FastChem consists of element species and molecule species.
        The element species are the reference, therefore its coefficients are all zero.
        The element species are automatically added in this function.
    """

    path_fastchem_data = get_data_filepath(path)
    # molecules species
    acoeff_molecule, components_molecule = _parse_fastchem_coeffs(
        open(path_fastchem_data, "r", encoding="utf-8").read()
    )
    species_molecule = list(acoeff_molecule.keys())

    # elements and element species
    elements = _set_elements(components_molecule)
    element_vector_ref = _elements_ref_AAG21()
    species_element, components_element, acoeff_element = _set_element_species(elements)

    # combine
    acoeff = {**acoeff_element, **acoeff_molecule}
    species = species_element + species_molecule
    components = {**components_element, **components_molecule}

    formula_matrix = _generate_formula_matrix(components, elements)
    _print_status(species_molecule, elements, species)

    ccoeff_array = np.array([acoeff[spec] for spec in species])
    vmap_logk = vmap(logk, in_axes=(None, 0), out_axes=0)

    def hvector_func(T: Union[float, jnp.ndarray]) -> jnp.ndarray:
        T = jnp.asarray(T)
        return -vmap_logk(T, ccoeff_array)

    hvector_func_jit = jit(hvector_func)

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=hvector_func_jit,
        elements=tuple(elements) if elements is not None else None,
        species=tuple(species) if species is not None else None,
        element_vector_reference=element_vector_ref,
        metadata={"source": "fastchem v3.1.3"},
    )


def _print_status(species_molecule, elements, species):
    print("fastchem presets in ExoGibbs")
    print(
        "number of species:",
        len(species),
        "elements:",
        len(elements),
        "molecules:",
        len(species_molecule),
    )


def logk(T, ccoeff):
    a1, a2, a3, a4, a5 = ccoeff
    return a1 / T + a2 * jnp.log(T) + a3 + a4 * T + a5 * T**2


def _set_element_species(elements):
    # Map element symbols to their corresponding species
    species_element = []
    components_element = {}
    acoeff_element = {}
    zerolist = [0.0, 0.0, 0.0, 0.0, 0.0]
    for el in elements:
        if el == "e-":
            species_element.append("e1-")
            components_element["e1-"] = {el: 1}
            acoeff_element["e1-"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            species_element.append(el + "1")
            components_element[el + "1"] = {el: 1}
            acoeff_element[el + "1"] = zerolist

    return species_element, components_element, acoeff_element


def _elements_ref_AAG21():
    # Reference elemental solar abundance b from AAG21 (from exojax.utils.zsol import nsol)
    # AAG21 = Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661
    return jnp.array(
        [
            2.4849887715495676e-06,
            2.214748573895377e-06,
            0.0002662713442014053,
            1.842147625867481e-06,
            1.8850567562968206e-07,
            8.041266040147904e-08,
            3.848791907186885e-07,
            1.3974118788763213e-08,
            2.319126458480869e-08,
            2.662713442014047e-05,
            9.667728169366614e-10,  # Ge
            0.9232608725415705,
            0.07573984826087723,
            1.0847369417204286e-07,
            3.275853193332226e-05,
            2.4284235212154947e-07,
            6.242009577709024e-05,
            1.532231656219368e-06,
            0.00010600453143949772,
            1.4632698717568435e-06,
            0.0004521936201224914,
            2.373145853176355e-07,
            1.2170948666733936e-05,
            2.9876136204576606e-05,
            8.616371800570028e-08,
            7.333721790759454e-09,
            3.352157616477396e-08,
            0.0,
        ]
    )


def _generate_formula_matrix(
    components: Dict[str, Dict[str, int]], elements: List[str]
) -> np.ndarray:
    """
    Generate the formula matrix from the components dict and elements list.

    Args:
        components: mapping ``species -> {element_symbol: count}``
        elements: list of element symbols

    Returns:
        np.ndarray: formula matrix of shape (num_elements, num_species)
    """
    num_species = len(components)
    num_elements = len(elements)
    formula_matrix = np.zeros((num_elements, num_species), dtype=int)
    species_list = list(components.keys())
    for j, spec in enumerate(species_list):
        comp = components[spec]
        for i, el in enumerate(elements):
            if el in comp:
                formula_matrix[i, j] = comp[el]
    return formula_matrix


def _set_elements(components: Dict[str, Dict[str, int]]) -> List[str]:
    """
    element_vector =['Al', 'Ar', 'Ba', 'Be', 'B', 'Ca', 'C', ...]
    """
    element_set = set()
    for spec in components.keys():
        for el in components[spec].keys():
            element_set.add(el)
    elements = sorted(list(element_set) + ["Ge"])
    return elements


def _parse_fastchem_coeffs(
    text: str,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]]]:
    """Parse FastChem logK coefficients and species composition.

    Returns:
        (coeffs, components)
            coeffs: mapping ``species -> [a1, a2, a3, a4, a5]``
            components: mapping ``species -> {element_symbol: count}``
    """
    coeffs: Dict[str, List[float]] = {}
    components: Dict[str, Dict[str, int]] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped[0].isdigit():
            continue

        match = _SPECIES_PATTERN.match(line)
        if not match or ":" not in line:
            continue

        species = match.group(1)
        # Parse element composition from the species line between ':' and '#'
        try:
            after_colon = line.split(":", 1)[1]
        except IndexError:
            after_colon = ""
        before_hash = after_colon.split("#", 1)[0]
        tokens = before_hash.strip().split()
        comp: Dict[str, int] = {}
        for k in range(0, len(tokens), 2):
            if k + 1 >= len(tokens):
                break
            el = tokens[k]
            cnt_str = tokens[k + 1]
            try:
                cnt = int(float(cnt_str))
            except ValueError:
                continue
            comp[el] = cnt
        if comp:
            components[species] = comp
        j = i + 1
        while j < len(lines):
            candidate = lines[j].strip()
            if candidate and not candidate.startswith("#"):
                arr = np.fromstring(candidate, sep=" ")
                if arr.size != 5:
                    raise ValueError(
                        f"{species}: not 5 coefficients (found {arr.size})"
                    )
                coeffs[species] = arr.tolist()
                break
            j += 1
        else:
            raise ValueError(f"{species}: missing coefficient line")

    return coeffs, components
