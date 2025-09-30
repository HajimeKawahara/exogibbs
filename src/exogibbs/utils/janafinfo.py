import re
from typing import Tuple

def load_ref_dict(filename: str) -> Tuple[dict[str, str], dict[str, int]]:
    """Load the JANAF reference species dictionary from a text file.
    
    Args:
        filename: path to the reference species text file

    Returns:
        ref: mapping from element symbol to reference species string
        nu: mapping from element symbol to number of atoms (stoichiometric coefficient) in reference species
    
        

    """
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


if __name__ == "__main__":
    # Resolve data file relative to this script so it runs from any CWD
    from exogibbs.io.load_data import get_data_filepath

    path_ref = get_data_filepath("janaf/ref.txt")
    ref, nu = load_ref_dict(path_ref)
    print(ref["Al"])
    print(ref["e-"])

