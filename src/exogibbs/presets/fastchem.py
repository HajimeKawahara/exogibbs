import re
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax import jit
from typing import Union
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.chemistry import setup_float_dtype
from exogibbs.io.load_data import get_data_filepath
from exogibbs.thermo.stoichiometry import build_formula_matrix
from exogibbs.utils.nameparser import set_elements_from_components

_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")



def chemsetup(path="fastchem/logK/logK.dat", species_defalt_elements=True, element_file=None, silent=False) -> ChemicalSetup:

    """
    Prepare a JAX-friendly ChemicalSetup from JANAF-like Gibbs matrices.

    Args:
        path (str): Path to the FastChem logK data file.
        species_defalt_elements (bool): If True, species are restricted to those that include only the 28 default elements in _default_elements(). If False, all elements found in the FastChem data are included.
        element_file (str, optional): Path to an element abundance file. If provided, the abundances will be used as the reference element vector. 
        silent (bool): If True, suppress status output.

    Returns:
        ChemicalSetup: The chemical setup object.

    Notes:
        The species in FastChem consists of element species and molecule species for gas (logK.dat).
        The element species are the reference, therefore its coefficients are all zero.
        The element species are automatically added in this function.
    """
    float_dtype = setup_float_dtype()
    path_fastchem_data = get_data_filepath(path)
    # molecules species
    acoeff_molecule, components_molecule, source_records_molecule = _parse_fastchem_coeffs_with_metadata(
        open(path_fastchem_data, "r", encoding="utf-8").read(),
        source_file=path,
    )
    
    # elements and element species
    if species_defalt_elements:
        print("restricting species to those composed of the default elements only.")
        if element_file is not None:
            print("WARNING: element_file is ignored when species_defalt_elements is True.")
        elements = _default_elements()
        element_vector_ref = _elements_ref_AAG21(float_dtype)
    elif element_file is not None:
        print("setting reference element vector from the provided element file:", element_file)
        import pandas as pd
        element_df = pd.read_csv(get_data_filepath(element_file), sep='\s+', comment="#", header=None, names=["element", "abundance"])
        elements = element_df["element"].tolist()[1:] + ["e-"]
        element_abundances = np.asarray(
            element_df["abundance"].tolist()[1:],
            dtype=float,
        )
        # FastChem element-abundance files use the astronomical A(X)=log10(n_X/n_H)+12 scale.
        # Convert them to linear abundances referenced to hydrogen before storing b_ref.
        element_vector_ref = jnp.asarray(
            np.concatenate((10.0 ** (element_abundances - 12.0), [0.0])),
            dtype=float_dtype,
        )
    else:
        print("setting elements from the species in the logK data.")
        elements = _set_elements_with_adding_Ge(components_molecule)
        element_vector_ref = []
    
    # cleaning species
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
    # combine
    acoeff = {**acoeff_element, **acoeff_molecule}
    species = species_element + species_molecule
    components = {**components_element, **components_molecule}

    formula_matrix = jnp.asarray(build_formula_matrix(components, elements), dtype=float_dtype)
    if not silent:
        _print_status(species_molecule, elements, species)

    ccoeff_array = jnp.asarray([acoeff[spec] for spec in species], dtype=float_dtype)
    vmap_logk = vmap(logk, in_axes=(None, 0), out_axes=0)

    def hvector_func(T: Union[float, jnp.ndarray]) -> jnp.ndarray:
        T = jnp.asarray(T)
        hvector = -vmap_logk(T, ccoeff_array)
        if T.ndim == 0:
            return hvector
        return jnp.moveaxis(hvector, 0, -1)

    hvector_func_jit = jit(hvector_func)
    hvector_logk_source_trace = _build_fastchem_hvector_logk_source_trace_builder(
        species=species,
        source_records=source_records_molecule,
        coefficients=acoeff,
        coefficient_array=ccoeff_array,
        float_dtype=float_dtype,
        provider_path=path,
    )
    try:
        hvector_func_jit.fastchem_hvector_logk_source_trace = hvector_logk_source_trace
    except AttributeError:
        pass

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=hvector_func_jit,
        elements=tuple(elements) if elements is not None else None,
        species=tuple(species) if species is not None else None,
        element_vector_reference=element_vector_ref,
        metadata={
            "source": "fastchem v3.1.3",
            "dataset": "gas",
            "fastchem_logk_file": path,
            "fastchem_element_file": (
                element_file if element_file is not None else "fastchem/element_abundances/asplund_2020.dat"
            ),
            "fastchem_species_default_elements": str(species_defalt_elements),
            "fastchem_logk_source_records": source_records_molecule,
            "fastchem_hvector_logk_source_trace": hvector_logk_source_trace,
            "fastchem_hvector_logk_source_trace_function": (
                "src/exogibbs/presets/fastchem.py::_build_fastchem_hvector_logk_source_trace_builder"
            ),
        },
    )


def _print_status(species_molecule, elements, species, preset_name="fastchem"):
    print(preset_name+" presets in ExoGibbs")
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


def _build_fastchem_hvector_logk_source_trace_builder(
    *,
    species: List[str],
    source_records: Dict[str, Dict[str, object]],
    coefficients: Dict[str, List[float]],
    coefficient_array: jnp.ndarray,
    float_dtype: jnp.dtype,
    provider_path: str,
):
    """Build a default-off FastChem preset hvector/logK source-term trace."""

    provider_function_identity = "src/exogibbs/presets/fastchem.py::chemsetup hvector_func"
    provider_closure_identity = (
        "src/exogibbs/presets/fastchem.py::chemsetup.<locals>.hvector_func"
    )

    def trace(T: Union[float, jnp.ndarray], limit: Optional[int] = None) -> Dict[str, object]:
        temperature_input = jnp.asarray(T)
        temperature_np = np.asarray(temperature_input)
        temperature_float = float(np.ravel(temperature_np.astype(float))[0])
        coeff_np = np.asarray(coefficient_array)
        rows = []
        row_species = species if limit is None else species[:limit]
        for source_index, label in enumerate(row_species):
            coeff = [float(value) for value in coefficients[label]]
            raw_logk = (
                coeff[0] / temperature_float
                + coeff[1] * float(np.log(temperature_float))
                + coeff[2]
                + coeff[3] * temperature_float
                + coeff[4] * temperature_float**2
            )
            record = source_records.get(label, {})
            rows.append(
                {
                    "species_label": label,
                    "molecule_label": label,
                    "source_file": record.get("source_file", provider_path),
                    "source_record": record.get("record_line"),
                    "source_index": int(source_index),
                    "source_record_name": record.get("record_name"),
                    "record_line_number": record.get("record_line_number"),
                    "coefficient_line_number": record.get("coefficient_line_number"),
                    "polynomial_coefficients": coeff,
                    "temperature_segment": {
                        "index": record.get("selected_temperature_segment_index"),
                        "upper_bound": record.get(
                            "selected_temperature_segment_upper_bound"
                        ),
                        "semantics": record.get(
                            "selected_temperature_segment_semantics",
                            "element species or no FastChem molecule source record",
                        ),
                    },
                    "raw_logK_source_term": float(raw_logk),
                    "KL_hvector_value": float(-raw_logk),
                    "sign_convention": "hvector = -logK(T)",
                    "density_gauge_pressure_temperature_correction": (
                        "none in FastChem gas logK provider; pressure term is applied later in _compute_gk"
                    ),
                    "final_mass_action_equivalent_term": float(raw_logk),
                    "native_dtype": "Python float parsed from FastChem logK text",
                    "python_dtype": type(raw_logk).__name__,
                    "jax_dtype": str(coefficient_array.dtype),
                    "cast_to_float64_occurred": str(coefficient_array.dtype) == "float64",
                    "source_stage_label": "FastChem-preset logK coefficient source",
                    "producer_stage_label": "chemsetup hvector_func closure",
                    "provider_function_identity": provider_function_identity,
                    "provider_closure_identity": provider_closure_identity,
                    "coefficient_array_origin": (
                        "ccoeff_array = jnp.asarray([acoeff[spec] for spec in species], dtype=setup_float_dtype())"
                    ),
                    "temperature_input_dtype": str(temperature_np.dtype),
                    "temperature_materialization_site": (
                        "src/exogibbs/presets/fastchem.py::chemsetup hvector_func T = jnp.asarray(T)"
                    ),
                    "output_dtype": str(coefficient_array.dtype),
                    "output_materialization_site": (
                        "src/exogibbs/presets/fastchem.py::chemsetup hvector_func returns -vmap_logk(T, ccoeff_array)"
                    ),
                }
            )
        return {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "FastChem_trace_values_used_as_KL_constructor_inputs": False,
            "provider_function_identity": provider_function_identity,
            "provider_closure_identity": provider_closure_identity,
            "provider_path": provider_path,
            "species_count": len(species),
            "emitted_row_count": len(rows),
            "coefficient_array_dtype": str(coeff_np.dtype),
            "coefficient_array_origin": (
                "ccoeff_array = jnp.asarray([acoeff[spec] for spec in species], dtype=setup_float_dtype())"
            ),
            "temperature_input_dtype": str(temperature_np.dtype),
            "temperature_input_value": temperature_float,
            "output_dtype": str(coefficient_array.dtype),
            "source_stage_label": "FastChem-preset hvector/logK source-term trace",
            "producer_stage_label": "chemsetup hvector_func",
            "rows": rows,
        }

    return trace


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
            acoeff_element["e1-"] = zerolist
        else:
            species_element.append(el + "1")
            components_element[el + "1"] = {el: 1}
            acoeff_element[el + "1"] = zerolist

    return species_element, components_element, acoeff_element


def _default_elements():
    return [
        "Al",
        "Ar",
        "C",
        "Ca",
        "Cl",
        "Co",
        "Cr",
        "Cu",
        "F",
        "Fe",
        "Ge",
        "H",
        "He",
        "K",
        "Mg",
        "Mn",
        "N",
        "Na",
        "Ne",
        "Ni",
        "O",
        "P",
        "S",
        "Si",
        "Ti",
        "V",
        "Zn",
        "e-",
    ]


def _elements_ref_AAG21(dtype):
    # Reference elemental solar abundance b from AAG21 (from exojax.utils.zsol import nsol)
    # AAG21 = Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661
    return jnp.asarray(
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
        ],
        dtype=dtype,
    )


def _set_elements_with_adding_Ge(components: Dict[str, Dict[str, int]]) -> List[str]:
    """set elements adding Ge to the elements from components.

    Args:
        components (Dict[str, Dict[str, int]]): A dictionary mapping species names to their elemental compositions.

    Returns:
        List[str]: A list of unique element symbols including Ge.

    Notes:
        This function extends the set of elements extracted from the components dictionary
        by adding Germanium (Ge) to the list of elements.

    """

    element_set = set_elements_from_components(components)
    if "Ge" not in element_set:
        elements = sorted(list(element_set) + ["Ge"])
    else:
        elements = sorted(list(element_set))
    return elements


def _restrict_species_to_elements(
    coeffs: Dict[str, List[float]],
    components: Dict[str, Dict[str, int]],
    allowed_elements: List[str],
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]]]:
    """Keep only species whose composition uses allowed elements only."""
    allowed = set(allowed_elements)
    filtered_components = {
        species: comp
        for species, comp in components.items()
        if set(comp).issubset(allowed)
    }
    filtered_coeffs = {
        species: coeffs[species]
        for species in filtered_components
        if species in coeffs
    }
    return filtered_coeffs, filtered_components


def _parse_fastchem_coeffs(
    text: str,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]]]:
    """Parse FastChem logK coefficients and species composition.

    Returns:
        (coeffs, components)
            coeffs: mapping ``species -> [a1, a2, a3, a4, a5]``
            components: mapping ``species -> {element_symbol: count}``
    """
    coeffs, components, _metadata = _parse_fastchem_coeffs_with_metadata(text)
    return coeffs, components


def _parse_fastchem_coeffs_with_metadata(
    text: str,
    source_file: Optional[str] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]], Dict[str, Dict[str, object]]]:
    """Parse FastChem gas logK coefficients with source-record metadata."""
    coeffs: Dict[str, List[float]] = {}
    components: Dict[str, Dict[str, int]] = {}
    metadata: Dict[str, Dict[str, object]] = {}
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
        record_name = ""
        name_part = line.split(":", 1)[0].strip().split()
        if len(name_part) > 1:
            record_name = " ".join(name_part[1:])
        j = i + 1
        while j < len(lines):
            candidate = lines[j].strip()
            if not candidate or candidate.startswith("#"):
                j += 1
                continue

            arr = np.fromstring(candidate, sep=" ")
            if arr.size != 5:
                raise ValueError(f"{species}: not 5 coefficients (found {arr.size})")
            coeffs[species] = arr.tolist()
            metadata[species] = {
                "species": species,
                "record_name": record_name,
                "source_file": source_file,
                "record_line_number": i + 1,
                "coefficient_line_number": j + 1,
                "record_line": line,
                "coefficient_line": lines[j],
                "coefficients": arr.tolist(),
                "selected_temperature_segment_index": None,
                "selected_temperature_segment_upper_bound": None,
                "selected_temperature_segment_semantics": (
                    "gas logK.dat has one five-coefficient fit per species record; "
                    "src/exogibbs/presets/fastchem.py::logk performs no temperature segment selection"
                ),
                "parser_file_function": "src/exogibbs/presets/fastchem.py::_parse_fastchem_coeffs_with_metadata",
                "logk_file_function": "src/exogibbs/presets/fastchem.py::logk",
            }
            break
        else:
            raise ValueError(f"{species}: missing coefficient line")

    return coeffs, components, metadata
