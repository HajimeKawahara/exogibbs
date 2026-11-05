"""Package the reviewed KH energy correction without changing thermal functions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/kh_candidate/20261105/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/oh_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/kh_v1.json"


def build_data() -> dict:
    """Transfer the reviewed energy shift onto the immutable OH base."""
    for path, expected in (
        (CANDIDATE, "e6af22cebacac123b0b454807ac7834ec958833636bfbfccd5741baedbb9eb55"),
        (BASE, "f393ed490972af06f39d7804f53a68777967c7e91fda1e8da1ed713faaccd081"),
    ):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("KH adoption requires the reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-kh-v1", "species": "H1K1", "reaction": candidate["reaction"],
        "base_dataset": base["id"], "base_data_sha256": sha256(BASE.read_bytes()).hexdigest(),
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "representation", "D0_cm-1",
            "D0_J_mol", "retained_D0_J_mol", "delta_D0_J_mol", "delta_D0_over_R_K",
            "gas_constant_J_mol_K", "original_logK_coefficients", "candidate_logK_coefficients",
            "source_limit")},
        "decision": {
            "name": "KH", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "Retained JANAF/FastChem five-coefficient fit and atomic convention",
            "energy_source": "Zemke et al. 1988 spectroscopic D0=14282.6 cm-1",
            "reason": "Replace the larger inherited balanced JANAF energy by the inspected spectroscopic D0, retaining the exact current thermal fit. Conserved finite substitutions and smaller NASA thermal-reference controls support the bounded update.",
            "condition": "1500-4000 K and 1 bar standard pressure. The abstract's +/-0.6 cm-1 belongs to De, not a complete D0 uncertainty. Thermal/isotope/continuum and full atmospheric accuracy remain unquantified.",
        },
        "sources": {
            "spectroscopic_energy": "https://doi.org/10.1016/0009-2614(88)87016-7",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/kh_candidate.rst",
            "substitution": "tests/data/validation/kh_equilibrium_impact/20261105/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): sha256(CANDIDATE.read_bytes()).hexdigest(),
                          **candidate["source_sha256"]},
        "interpretation": "KH-only energy adoption on curated-oh-v1, retaining OH, H2O, CO2, TiO and C2H updates. Add the exact reviewed -delta(D0)/(RT) to the inherited reaction once; no thermal refit, spin factor, ZPE or atomic correction. Full physical accuracy remains unknown.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
