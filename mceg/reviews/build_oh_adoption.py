"""Package the reviewed OH NASA/ATcT reaction without refitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/oh_candidate/20261105/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/h2o_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/oh_v1.json"


def build_data() -> dict:
    """Transfer the exact reviewed coefficients onto the immutable H2O base."""
    for path, expected in (
        (CANDIDATE, "5c2eea21282fc1cee77723c0e5774e0b2b52a401e45d62a17a726f76b9c7780d"),
        (BASE, "6f79a001d34b1b107a34db322d642f0210ae6218b4f659be1cd8b193cbb1d6d6"),
    ):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("OH adoption requires the reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-oh-v1", "species": "H1O1", "reaction": candidate["reaction"],
        "base_dataset": base["id"], "base_data_sha256": sha256(BASE.read_bytes()).hexdigest(),
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "source_interval_K",
            "representation", "formation_nasa9_coefficients", "D0_J_mol", "D0_over_k_K",
            "nasa_D0_J_mol", "ATcT_minus_nasa_D0_over_k_K", "thermal_convention")},
        "decision": {
            "name": "OH", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "NASA CEA / Gurvich OH thermal functions",
            "energy_source": "ATcT 1.172 balanced H+O-OH atomization",
            "reason": "Replace the inherited JANAF electronic/caloric defect using the exact reviewed NASA9 reaction and balanced ATcT energy. Dense caloric checks and conserved finite substitutions support the bounded adoption; MYTHOS is a smaller thermal control.",
            "condition": "1500-4000 K, 1 bar standard pressure. Shared source ancestry, isotope approximation, continuum and ATcT covariance limits remain; complete atmospheric physical accuracy is unknown.",
        },
        "sources": {
            "iupac_evaluation": "https://doi.org/10.1063/1.1724828",
            "nasa": "https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp",
            "atct": "https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php",
            "mythos_control": "https://doi.org/10.1093/mnras/stae2803",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/oh_candidate.rst",
            "substitution": "tests/data/validation/oh_equilibrium_impact/20261105/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): sha256(CANDIDATE.read_bytes()).hexdigest(),
                          **candidate["source_sha256"]},
        "interpretation": "OH-only adoption on curated-h2o-v1, retaining H2O, CO2, TiO and C2H updates. Exact reviewed NASA9 reaction coefficients; no fit, extra spin factor, ZPE or atomic correction. Scoped scientific and numerical support does not establish full physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
