"""Reproduce the reviewed CH2 energy-only production data."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "tests/data/validation/ch2_candidate/20261106/candidate.json"
BASE = "src/exogibbs/data/curated/c2h2_v1.json"
CANDIDATE_SHA256 = "8caae4adfbb73910514f5273b012cb857231f14dea06045a72c3392e9495bddb"
BASE_SHA256 = "cdd7f4348e05477c158b9fe0f9e3250b2d3f71d2e6fdd3cb8d051aa147bb169e"
OUTPUT = ROOT / "src/exogibbs/data/curated/ch2_v1.json"

def build_data() -> dict:
    """Preserve the reviewed energy-only fields on the pinned versioned base."""
    for name, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("CH2 adoption requires its reviewed candidate and base bytes")
    candidate = json.loads((ROOT/CANDIDATE).read_text())
    base = json.loads((ROOT/BASE).read_text())
    assert candidate["species"] == "C1H2"
    assert base["id"] == "curated-c2h2-v1"
    return json.loads(PACKAGED_FIELDS)


PACKAGED_FIELDS = r'''{
  "id": "curated-ch2-v1",
  "species": "C1H2",
  "reaction": "C(g) + 2 H(g) -> CH2(g)",
  "base_dataset": "curated-c2h2-v1",
  "base_data_sha256": "cdd7f4348e05477c158b9fe0f9e3250b2d3f71d2e6fdd3cb8d051aa147bb169e",
  "temperature_range_K": [
    1500,
    4000
  ],
  "standard_pressure_Pa": 100000,
  "gas_constant_J_mol_K": 8.31446261815324,
  "base_D0_J_mol": 757340.0,
  "D0_J_mol": 752392.0,
  "candidate_ATcT_version": "1.228",
  "candidate_hf0_kJ_mol": {
    "C": 711.37,
    "H": 216.034,
    "CH2": 391.046
  },
  "formal_fit_intercept_Ra_J_mol": 758027.1860277539,
  "delta_D0_J_mol": -4948,
  "delta_D0_over_R_K": -595.1076127514085,
  "representation": "Add only -delta(D0)/(RT) to the inherited formation G/(RT). Preserve all literal thermal coefficients, atomic gauge and source-fit residual. No thermal refit, added ZPE or extrapolation to zero K.",
  "anchor_scope": "The balanced physical energy difference is separate from the formal R*a fit intercept; preserve their residual rather than relabeling the intercept as D0.",
  "decision": {
    "name": "CH2",
    "status": "prefer_conditionally",
    "source_group": "molecular",
    "thermal_source": "Retained literal JANAF-derived FastChem reaction thermal fit",
    "energy_source": "ATcT 1.228 balanced C+2H-CH2 physical atomization difference",
    "reason": "Adopt the same-version regular CH2 ATcT ground energy while preserving literal thermals and source-fit residual. The independently reconstructed RRHO thermal sensitivity is not adopted because the printed component/equilibrium table discrepancy and high-temperature physical limits remain unresolved.",
    "condition": "1500-4000 K, one-bar standard pressure. Regular equilibrated CH2 is distinct from partial triplet-only species. ATcT covariance, large-amplitude/anharmonic/Renner-Teller thermals, isotope mixtures and planetary physical accuracy remain unquantified."
  },
  "sources": {
    "candidate": "tests/data/validation/ch2_candidate/20261106/candidate.json",
    "review": "documents/mceg/ch2_candidate.rst",
    "adoption": "documents/mceg/curated_ch2_adoption.rst",
    "substitution": "tests/data/validation/ch2_equilibrium_impact/20261106/summary.json"
  },
  "source_sha256": {
    "tests/data/validation/ch2_candidate/20261106/candidate.json": "8caae4adfbb73910514f5273b012cb857231f14dea06045a72c3392e9495bddb",
    "tests/data/validation/ch2_candidate/20261106/thermal_inputs.json": "352823e0983319d7efb8b7851cdacf7b0da588fd224d60fa6e2643b1f67ed659",
    "src/exogibbs/data/FastChem4/logK/logK.dat": "eda9b9d7e62ccf7398036ee84349bd0a721b779acdc9d191cbf9fe1dd01f619c",
    "tests/data/validation/ch2_source_review/20261106/inputs.json": "6ee75797a267bc3f36e7b6c7b19aae3512214669f0d98f68b0921370ddaa43b3",
    "tests/data/validation/ch2_source_review/20261106/comparison.json": "a76bfc9fb587397266dda5714365fc04f654ce66ec95e109c33386cebc8b47d4",
    "tests/data/validation/ch2_source_review/20261106/atct-selected.tsv": "458888d4238bbb2602b058235dbba1f0082658a5c9ea53e907d3387d7819a214",
    "tests/data/validation/ch2_source_review/20261106/nasa-selected.txt": "ff98968625c8341f2b145344e1ca35b64267cee81bb954f1c7d67e91a41af795",
    "tests/data/validation/ch2_source_review/20261106/sources.json": "dc40198155506232d2f5d6ef9eaccd7772d6b3de5ffb235da2125b4720aa1688"
  },
  "interpretation": "CH2 energy-only adoption on curated-c2h2-v1. G/H increase by 4948 J/mol; S/Cp and other species potentials remain unchanged. Thermal candidates are not adopted. Numerical verification does not establish complete physical accuracy."
}'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
