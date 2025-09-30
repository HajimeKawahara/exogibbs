from typing import Dict, List
import re
import numpy as np
import jax.numpy as jnp

# Capture the full species token (including digits, underscores, and charge signs)
# up to the first whitespace or colon, e.g., "Al1Cl1F1+", "Al1H1O1_2".
_SPECIES_PATTERN = re.compile(r"^\s*([^\s:]+)")


def parse_fastchem_coeffs(text: str) -> Dict[str, List[float]]:
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


def logk(T, coeff):
    a1, a2, a3, a4, a5 = coeff
    return a1 / T + a2 * jnp.log(T) + a3 + a4 * T + a5 * T**2

#log K = a1/T + a2 ln T + a3 + a4 T + a5 T^2 

if __name__ == "__main__":
    from pathlib import Path
    # Resolve data file relative to this script so it runs from any CWD
    path_fastchem_data = Path(__file__).with_name("logK.dat")
    txt = path_fastchem_data.read_text(encoding="utf-8")
    fastchem_coeffs = parse_fastchem_coeffs(txt)
    print(fastchem_coeffs["Al1Cl1F1+"])
    print(fastchem_coeffs["Al1Cl1F1"])
