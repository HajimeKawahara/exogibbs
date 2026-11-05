"""Package the reviewed HO2 NASA/ATcT reaction without refitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/ho2_candidate/20261105/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/kh_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/ho2_v1.json"


def build_data() -> dict:
    """Transfer the exact reviewed coefficients onto the immutable KH base."""
    for path, expected in (
        (CANDIDATE, "b04c3146c88d2d688d1ea284815c85f4e1990d1cf8305b2534813a86f45be34d"),
        (BASE, "d808d0b1e5f65fabcb09759c0961b5974d2de99e7d208b914dc6a73169f52328"),
    ):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("HO2 adoption requires the reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-ho2-v1", "species": "H1O2", "reaction": candidate["reaction"],
        "base_dataset": base["id"], "base_data_sha256": sha256(BASE.read_bytes()).hexdigest(),
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "source_interval_K",
            "representation", "formation_nasa9_coefficients", "D0_J_mol", "D0_over_k_K",
            "nasa_D0_J_mol", "ATcT_minus_nasa_D0_over_k_K", "thermal_convention")},
        "decision": {
            "name": "HO2", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "NASA CEA HO2 thermal functions",
            "energy_source": "ATcT 1.172 balanced H+2O-HO2 atomization",
            "reason": "Replace the obsolete JANAF formation energy with the exact reviewed NASA9 reaction and balanced ATcT 1.172 energy. Dense caloric checks, small same-energy JANAF node controls and conserved finite substitutions support this bounded adoption.",
            "condition": "1500-4000 K, 1 bar standard pressure. Shared source ancestry, isotope approximation, continuum and ATcT covariance limits remain; complete atmospheric physical accuracy is unknown.",
        },
        "sources": {
            "nist_evaluation": "https://doi.org/10.6028/jres.121.005",
            "nasa": "https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp",
            "atct": "https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/ho2_candidate.rst",
            "substitution": "tests/data/validation/ho2_equilibrium_impact/20261105/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): sha256(CANDIDATE.read_bytes()).hexdigest(),
                          **candidate["source_sha256"]},
        "interpretation": "HO2-only adoption on curated-kh-v1, retaining KH, OH, H2O, CO2, TiO and C2H updates. Exact reviewed NASA9 reaction coefficients; no fit, extra spin factor, ZPE or atomic correction. Scoped scientific and numerical support does not establish full physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
