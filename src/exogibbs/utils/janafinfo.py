import re
from typing import Tuple
from exogibbs.io.load_data import get_data_filepath

JANAF_REFERENCE_SPECIES_FILE = "janaf/ref.txt"

def reference_species_dict() -> Tuple[dict[str, str], dict[str, int]]:
    """Load the JANAF reference species dictionary from a text file.
    
    Returns:
        ref: mapping from element symbol to reference species string
        nu: mapping from element symbol to number of atoms (stoichiometric coefficient) in reference species
    """
    
    filename = get_data_filepath(JANAF_REFERENCE_SPECIES_FILE)
    ref = {}
    nu = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            key = re.sub(r"\d+$", "", token)
            m = re.search(r"(\d+)$", token)
            if m:
                num = int(m.group(1))
            else:
                num = 1
            if key == "e1-":
                key = "e-"
                num = 1
            ref[key] = token
            nu[key] = num
    return ref, nu
