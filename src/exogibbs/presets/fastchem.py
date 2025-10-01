from typing import Dict, List
import re
import numpy as np
import jax.numpy as jnp
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.utils.janafinfo import reference_species_dict
from exogibbs.io.load_data import get_data_filepath

_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")


def chemsetup(path="fastchem/logK/logK.dat") -> ChemicalSetup:
    """
    Prepare a JAX-friendly ChemicalSetup from JANAF-like Gibbs matrices.

    Notes
    -----
    * Ensures that all tables (T_table, mu_table) live on device as jnp.arrays.
    * hvector_func(T) stays purely JAX/NumPy to allow grad/jit/vmap through T.
    * formula_matrix is fixed, built from df_molname.
    """

    
    refspecies, nu_ref = reference_species_dict()
    path_fastchem_data = get_data_filepath(path)
    acoeff = _parse_fastchem_coeffs(open(path_fastchem_data, "r", encoding="utf-8").read())
    c_element, element_vector, elements_not_included, element_species = _set_elements(refspecies, nu_ref, acoeff)
    print(element_species)
    set_ccoeff(c_element, element_vector, element_species, refspecies, nu_ref, acoeff)

def logk(T, coeff):
    #log K = a1/T + a2 ln T + a3 + a4 T + a5 T^2 
    a1, a2, a3, a4, a5 = coeff
    return a1 / T + a2 * jnp.log(T) + a3 + a4 * T + a5 * T**2


def _set_elements(refspecies, nu_ref, acoeff):
    c_element = {}
    element_vector = [] 
    elements_not_included = []
    element_species = []
    zerocoeff = np.zeros(5)
    for i in refspecies.keys():
        if nu_ref[i] == 1:
            c_element[i] = zerocoeff
            element_vector.append(i)
            element_species.append(refspecies[i])
        elif refspecies[i] in acoeff:
            c_element[i] = acoeff[refspecies[i]]
            element_vector.append(i)
            element_species.append(refspecies[i])
        else:
            elements_not_included.append(i)

    return c_element, element_vector, elements_not_included, element_species


def set_ccoeff(c_element, element_vector, element_species, refspecies, nu_ref, acoeff):

    ccoeff = {}    
    

def _parse_fastchem_coeffs(text: str) -> Dict[str, List[float]]:
    coeffs: Dict[str, List[float]] = {}
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
        j = i + 1
        while j < len(lines):
            candidate = lines[j].strip()
            if candidate and not candidate.startswith("#"):
                arr = np.fromstring(candidate, sep=" ")
                if arr.size != 5:
                    raise ValueError(f"{species}: not 5 coefficients (found {arr.size})")
                coeffs[species] = arr.tolist()
                break
            j += 1
        else:
            raise ValueError(f"{species}: missing coefficient line")

    return coeffs





if __name__ == "__main__":
    chemsetup()
    