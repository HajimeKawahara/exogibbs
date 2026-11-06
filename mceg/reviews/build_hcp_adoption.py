"""Reproduce the reviewed HCP energy-only production data."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "tests/data/validation/hcp_candidate/20261106/candidate.json"
BASE = "src/exogibbs/data/curated/tio2_v1.json"
CANDIDATE_SHA256 = "8385d71f6cd8d5f3d23d78b718112b2a7c8829a5761a4bc6fa1a1195f8a4d583"
BASE_SHA256 = "7e07c9ed020ef1cd377f00e95eb5bbaee451788a3d75d377d1872fe66ee9aa43"
OUTPUT = ROOT / "src/exogibbs/data/curated/hcp_v1.json"

def build_data() -> dict:
    """Preserve the reviewed energy-only fields on the pinned versioned base."""
    for name, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("HCP adoption requires its reviewed candidate and base bytes")
    candidate = json.loads((ROOT/CANDIDATE).read_text())
    base = json.loads((ROOT/BASE).read_text())
    assert candidate["species"] == "C1H1P1"
    assert base["id"] == "curated-tio2-v1"
    return json.loads(PACKAGED_FIELDS)


PACKAGED_FIELDS = r'''{
  "id": "curated-hcp-v1",
  "species": "C1H1P1",
  "reaction": "C(g) + H(g) + P(g) -> HCP(g)",
  "base_dataset": "curated-tio2-v1",
  "base_data_sha256": "7e07c9ed020ef1cd377f00e95eb5bbaee451788a3d75d377d1872fe66ee9aa43",
  "temperature_range_K": [
    1500,
    4000
  ],
  "standard_pressure_Pa": 100000,
  "gas_constant_J_mol_K": 8.31446261815324,
  "base_D0_J_mol": 1091582,
  "D0_J_mol": 1029274.0,
  "candidate_Hf0_J_mol": 213500.0,
  "base_Hf0_J_mol": 151192,
  "formal_fit_intercept_Ra_J_mol": 1095834.3640026404,
  "energy_source_url": "https://cccbdb.nist.gov/exp2x.asp?casno=6829523",
  "delta_D0_J_mol": -62308,
  "delta_D0_over_R_K": -7493.929898002175,
  "representation": "Add only -delta(D0)/(RT) to the inherited formation G/(RT). Preserve all literal thermal coefficients, atomic gauge and source-fit residual. No thermal refit, added ZPE or extrapolation to zero K.",
  "anchor_scope": "The balanced physical energy difference is separate from the formal R*a fit intercept; preserve their residual rather than relabeling the intercept as D0.",
  "decision": {
    "name": "HCP",
    "status": "prefer_conditionally",
    "source_group": "molecular",
    "thermal_source": "Retained literal JANAF-derived FastChem reaction thermal fit",
    "energy_source": "NIST CCCBDB CP-based adjustment to JANAF HCP formation energy",
    "reason": "Adopt the inspected CCCBDB evaluated formation-energy correction on the unchanged JANAF elemental gauge. The independent 2001 theory is a sensitivity control, not a second additive correction.",
    "condition": "1500-4000 K, one-bar standard pressure. CCCBDB is a CP-based JANAF adjustment, not an independent HCP measurement. Its reported +/-20 kJ/mol is not a complete reaction covariance. Retained thermals and high-temperature physical accuracy remain limited."
  },
  "sources": {
    "candidate": "tests/data/validation/hcp_candidate/20261106/candidate.json",
    "review": "documents/mceg/hcp_candidate.rst",
    "adoption": "documents/mceg/curated_hcp_adoption.rst",
    "substitution": "tests/data/validation/hcp_equilibrium_impact/20261106/summary.json"
  },
  "source_sha256": {
    "tests/data/validation/hcp_candidate/20261106/candidate.json": "8385d71f6cd8d5f3d23d78b718112b2a7c8829a5761a4bc6fa1a1195f8a4d583",
    "src/exogibbs/data/FastChem4/logK/logK.dat": "eda9b9d7e62ccf7398036ee84349bd0a721b779acdc9d191cbf9fe1dd01f619c",
    "tests/data/validation/hcp_source_review/20261106/inputs.json": "f31a84cf62082bee36023a0712069b81c57557cb1154b528849d47f068abd3d0",
    "tests/data/validation/hcp_source_review/20261106/literature.json": "4234122e69f32d4594eab5dde3a873ec6004bba0b9941c2eae077f6143c6d610",
    "tests/data/validation/hcp_source_review/20261106/sources.json": "413990e2a027acd572e74fc8621b435a55eb8f3832c36f0ed73521ddcb5ad1e0",
    "tests/data/validation/hcp_source_review/20261106/C-056.txt": "d9f9b2ca6792df013ed4277ae11aa3756a6aa38ef14d9445d420ba189d629ba7",
    "tests/data/validation/hcp_source_review/20261106/C-003.txt": "df9d51b1fd89984f5c7bce53fb445d34aa688209d15a228c396ed52b793d27d4",
    "tests/data/validation/hcp_source_review/20261106/H-001.txt": "dceb73414db8ba3270e51c70b217570a6a96c8c24be61e4c0ba2ca96a4f084be",
    "tests/data/validation/hcp_source_review/20261106/P-008.txt": "a71761ba86cfb2b728d22642903bbf44eb7cfcf78fbee07a73930ce652cec879"
  },
  "interpretation": "HCP energy-only adoption on curated-tio2-v1. G/H increase by 62308 J/mol; S/Cp and other species potentials remain unchanged. Thermal candidates are not adopted. Numerical verification does not establish complete physical accuracy."
}'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
