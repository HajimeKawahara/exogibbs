from typing import Dict, List
import re

import numpy as np

_SPECIES_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\b")


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


if __name__ == "__main__":
    from pathlib import Path

    txt = Path("logK.dat").read_text(encoding="utf-8")
    fastchem_coeffs = parse_fastchem_coeffs(txt)
    print(fastchem_coeffs)