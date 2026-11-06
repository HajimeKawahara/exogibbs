"""Package the reviewed measured TiO2 thermal correction at retained energy."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/tio2_candidate/20261106/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/ps_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/tio2_v1.json"
CANDIDATE_SHA256 = "81586500b343ddc3e9eb42be2af502d8748bf8f66c99eae931715c77910aa9af"
BASE_SHA256 = "d070f0fd64fe89039994ff37ec2d1f6586ed66190f97f48c38b6e13f196d90d6"


def build_data() -> dict:
    """Retain the exact reviewed thermal inputs and immutable cumulative base."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("TiO2 adoption requires the reviewed candidate and base bytes")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-tio2-v1", "species": "O2Ti1",
        "reaction": "Ti(g) + 2 O(g) -> TiO2(g)",
        "base_dataset": base["id"], "base_data_sha256": BASE_SHA256,
        **{key: candidate[key] for key in (
            "temperature_range_K", "standard_pressure_Pa", "gas_constant_J_mol_K",
            "nasa_native_R_J_mol_K", "original_source_D0_J_mol", "delta_D0_J_mol",
            "nasa_molecular_thermal_coefficients", "rotational_constants_MHz",
            "frequencies_cm-1", "mass_u", "symmetry_number", "electronic_degeneracy",
            "nuclear_spin_weight", "anchor_scope", "thermal_scope", "physical_scope")},
        "representation": "Add the molecular RRHO G-H0 minus the NASA molecular G-H0 to the inherited formation potential. All current-fit residuals to that proxy, source D0 and atomic potentials are retained. No new fit or H298-control energy shift.",
        "decision": {
            "name": "TiO2", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "Measured ground-state rotation and gas fundamentals treated as harmonic frequencies; retained fit residual to NASA thermal proxy",
            "energy_source": "Retained JANAF source D0 = 1267.776 kJ/mol",
            "reason": "Within the declared RRHO approximation, measured rotation and bending are preferable central inputs to the 1973 estimated geometry and bend. Adopt the exact evaluated bounded thermal correction without a separate energy change; fresh cumulative-network validation accompanies the candidate evidence.",
            "condition": "1500-4000 K and one-bar standard pressure. Ground-state RRHO, JANAF mean mass and principal-isotopologue rotation; anharmonicity, full electronic/isotope treatment and complete atmospheric physical accuracy remain unquantified.",
        },
        "sources": {
            "rotation": "https://doi.org/10.1086/528934",
            "bend": "https://doi.org/10.1039/B821849H",
            "gas_spectroscopy": "https://webbook.nist.gov/cgi/cbook.cgi?ID=C13463677&Mask=3EE9&Units=SI",
            "janaf": "https://janaf.nist.gov/tables/O-046.txt",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "review": "documents/mceg/tio2_candidate.rst",
            "substitution": "tests/data/validation/tio2_equilibrium_impact/20261106/summary.json",
        },
        "source_sha256": {str(CANDIDATE.relative_to(ROOT)): CANDIDATE_SHA256,
                          **candidate["source_sha256"]},
        "interpretation": "TiO2-only retained-energy thermal adoption on curated-ps-v1, preserving the cumulative CN, PS and earlier updates. Symmetry two and spin weight one occur once; zero-point energy is not added again. The candidate's H298-fixed and electronic sensitivities are not adopted. Numerical/scoped scientific support is separate from full physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
