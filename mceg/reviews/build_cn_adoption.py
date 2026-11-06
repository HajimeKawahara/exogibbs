"""Package the reviewed CN energy correction without changing its thermal fit."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/cn_candidate/20261106/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/cos_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/cn_v1.json"
CANDIDATE_SHA256 = "0216d941ccae0422d563be63fc35726f6b218ec6cee909431060d3c0af0bda82"
BASE_SHA256 = "0e8c7801440f1951b7f6d7516fb9cfef42fe11b8b0f4de68a5039917daaa72a2"


def build_data() -> dict:
    """Keep the literal current fit and apply the physical source-energy difference."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("CN adoption requires its reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    assert candidate["base_dataset"] == base["id"] == "curated-cos-v1"
    assert candidate["delta_D0_J_mol"] == -4801
    return {
        "id": "curated-cn-v1", "species": "C1N1", "reaction": candidate["reaction"],
        "base_dataset": base["id"], "base_data_sha256": sha256(BASE.read_bytes()).hexdigest(),
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "gas_constant_J_mol_K",
            "base_D0_J_mol", "D0_J_mol", "delta_D0_J_mol", "delta_D0_over_R_K",
            "candidate_ATcT_version", "candidate_hf0_kJ_mol", "representation",
            "formal_fit_intercept_Ra_J_mol", "anchor_scope")},
        "decision": {
            "name": "CN", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "Retained literal JANAF-derived FastChem reaction thermal fit",
            "energy_source": "ATcT 1.228 balanced C+N-CN physical atomization difference",
            "reason": "Adopt the evaluated same-version gas-atom/molecule energy balance while preserving the supported thermal shape and inherited fit residual. Use only the physical JANAF-to-ATcT delta D0=-4801 J/mol; do not reset the formal fit intercept or add ZPE again.",
            "condition": "1500-4000 K, one-bar standard pressure. Correlated reaction-energy covariance, complete thermal and isotope uncertainty, and planetary accuracy remain unquantified.",
        },
        "sources": {
            "atct": "https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php",
            "janaf": "https://janaf.nist.gov/tables/C-080.html",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/cn_candidate.rst",
            "substitution": "tests/data/validation/cn_equilibrium_impact/20261106/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): sha256(CANDIDATE.read_bytes()).hexdigest(),
                          **candidate["source_sha256"]},
        "interpretation": "CN energy adoption on the unchanged curated-cos-v1 background. G and H increase by 4801 J/mol; S and Cp remain unchanged. The physical source D0 is distinct from the fit intercept, whose 78.141 J/mol residual is retained. Numerical support does not establish full physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
