"""Package the reviewed NIST 626 CO2 candidate without refitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/co2_candidate/20261104/candidate.json"
CANDIDATE_SHA256 = "9373569f8f6d0a9539f0d8ccf28b137fd0e4eab5b177b764e682aef391065bb2"
BASE = ROOT / "src/exogibbs/data/curated/tio_v1.json"
BASE_SHA256 = "52311c1ad348dc0098a34c17cfced9a175ff0b551bcbfc3819292271762c27d1"
OUTPUT = ROOT / "src/exogibbs/data/curated/co2_v1.json"


def build_data() -> dict:
    """Return the exact primary candidate as an immutable single-species layer."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"CO2 adoption requires its reviewed input: {path}")
    candidate = json.loads(CANDIDATE.read_text())
    model = candidate["models"]["nist626"]
    return {
        "id": "curated-co2-v1", "species": "C1O2", "reaction": "C(g) + 2 O(g) -> CO2(g)",
        "base_dataset": "curated-tio-v1", "base_data_sha256": BASE_SHA256,
        "temperature_range_K": candidate["temperature_range_K"], "standard_pressure_bar": 1.,
        "knots_K": model["knots_K"], "coefficients": model["coefficients"],
        "atomic_background_nasa9": candidate["atomic_background_nasa9"],
        "D0_J_mol": candidate["D0_J_mol"], "D0_over_k_K": candidate["D0_over_k_K"],
        "molecular_mass_u": candidate["molecular_mass_u"], "nuclear_spin_factor": 1,
        "energy_zero": "Lowest 12C16O2 rovibrational level; retained ATcT 1.172 total atomization to C(3P0)+2 O(3P2). No additional ZPE or symmetry correction.",
        "isotope_convention": candidate["conventions"]["main"],
        "source_constants": candidate["conventions"]["source_constants"],
        "representation": candidate["representation"],
        "decision": {
            "name": "CO2", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "NIST 2021 12C16O2 thermodynamic tables, Tashkun-Harvey 2025 evaluation",
            "energy_source": "Retained atom-balanced ATcT 1.172 total atomization",
            "reason": "Adopt the inspected modern effective-Hamiltonian thermodynamic evaluation, with explicit spin/mass/energy conventions, consistent derivatives and verified finite response. Agreement with NASA or a newer opacity release is not the selection criterion.",
            "condition": "1500-4000 K, 1 bar standard pressure; representative 626 internal model at NASA elemental mass. Source covariance does not bound all isotope, continuum, dissociation-cutoff or atmospheric uncertainty.",
            "isotope_convention": candidate["conventions"]["main"],
        },
        "sources": {
            "paper": "https://doi.org/10.1063/5.0276615",
            "dataset": "https://doi.org/10.18434/mds2-2364",
            "source_table": "https://data.nist.gov/od/ds/mds2-2364/tcfun_626.txt",
            "source_table_sha256": candidate["source"]["sha256"]["tcfun_626.txt"],
            "review": "documents/mceg/co2_source_review.rst",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "substitution": "tests/data/validation/co2_equilibrium_impact/20261104/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): CANDIDATE_SHA256},
        "interpretation": "Single-species CO2 adoption on curated-tio-v1. Exact reviewed NIST626 C2 Gibbs coefficients and unchanged ATcT energy/NASA atom reference; fixed-isotope-average diagnostic is not adopted. Numerical and scoped scientific support do not establish complete natural-isotope or planetary physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
