"""Package the reviewed COS energy shift without replacing its thermal model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/cos_candidate/20261105/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/ho2_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/cos_v1.json"
CANDIDATE_SHA256 = "ff823f51498eb83614920d60ca11656cebdc5c9c0ab1b14e91ce82d55f1aca94"
BASE_SHA256 = "de1050836d2adfb0853594fb3c7f25253af758a9c35f40458baa32737bbf534b"


def build_data() -> dict:
    """Apply the exact reviewed energy delta to the immutable HO2 base."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("COS adoption requires the reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-cos-v1", "species": "C1O1S1", "reaction": candidate["reaction"],
        "base_dataset": base["id"], "base_data_sha256": BASE_SHA256,
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "gas_constant_J_mol_K",
            "base_D0_J_mol", "D0_J_mol", "delta_D0_J_mol", "delta_D0_over_R_K",
            "atct_version", "same_version_hf0_kJ_mol", "representation")},
        "decision": {
            "name": "COS", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "Retained JANAF molecular thermal fit and NASA atomic background",
            "energy_source": "ATcT 1.228 balanced C+O+S-OCS atomization",
            "reason": "Adopt the consistently balanced evaluated central energy while preserving the reviewed production thermal functions. Exact constant-energy construction and conserved finite substitutions support this bounded update; TIPS/OYT8 thermal replacement is a separate unresolved question.",
            "condition": "1500-4000 K, 1 bar standard pressure. Correlated ATcT reaction uncertainty, isotope approximation, thermal source limitations and complete atmospheric physical accuracy remain unquantified.",
        },
        "sources": {
            "atct": "https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php",
            "janaf": "https://janaf.nist.gov/tables/C-094.html",
            "nasa": "https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/cos_candidate.rst",
            "substitution": "tests/data/validation/cos_equilibrium_impact/20261105/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): CANDIDATE_SHA256,
                          **candidate["source_sha256"]},
        "interpretation": "COS-only energy adoption on curated-ho2-v1, retaining KH, HO2, OH, H2O, CO2, TiO and C2H updates. Subtract the reviewed delta(D0/R)/T exactly once; no thermal fit, isotope/spin change or TIPS crossover. Numerical and scoped scientific support do not establish full physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
