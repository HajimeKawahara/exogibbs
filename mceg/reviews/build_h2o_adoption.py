"""Package the reviewed evaluated H2O candidate without any numerical refit."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/h2o_candidate/20261104/candidate.json"
CANDIDATE_SHA256 = "0987628c7105024ad100515a71fda2b3270bd24aa13c8b2f4b7fcbe8d999775a"
BASE = ROOT / "src/exogibbs/data/curated/co2_v1.json"
BASE_SHA256 = "dfac2e1ac79ff0c0dadb9010d9f187b7d4a84fd3158916e44cc60e72f94f87ba"
OUTPUT = ROOT / "src/exogibbs/data/curated/h2o_v1.json"


def build_data() -> dict:
    """Copy the reviewed analytical model onto the pinned CO2 adoption base."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"H2O adoption requires its reviewed input: {path}")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    reference = ROOT / "examples/comparisons/data/major_molecule_reference.json"
    return {
        "id": "curated-h2o-v1", "species": "H2O1", "reaction": "2 H(g) + O(g) -> H2O(g)",
        "base_dataset": base["id"], "base_data_sha256": BASE_SHA256,
        "temperature_range_K": candidate["temperature_range_K"], "standard_pressure_bar": 1.,
        "thermal": candidate["evaluated"],
        "atomic_background_nasa9": candidate["atomic_background_nasa9"],
        "D0_J_mol": candidate["D0_J_mol"], "D0_over_k_K": candidate["D0_over_k_K"],
        "molecular_mass_u": candidate["molecular_mass_u"],
        "isotope_convention": candidate["isotope_internal_states"],
        "spin_and_zero": candidate["spin_and_zero"],
        "evaluated_scope": candidate["evaluated_scope"],
        "scattering_limit": candidate["scattering_limit"],
        "decision": {
            "name": "H2O", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "Furtenbacher, Harvey and Csaszar 2025 evaluated thermochemistry",
            "energy_source": "ATcT 1.172 balanced total atomization, retained",
            "reason": "Replace the inherited NASA thermal functions with the modern evaluated thermochemical source. The exact reviewed continuous model preserves caloric derivatives, and conserved finite substitutions resolve the response; its discrete-state alternative has a much smaller effect on this grid.",
            "condition": "1500-4000 K, 1 bar standard pressure; dominant-isotope internal states at aggregate mass. Scattering interaction assignment and complete atmospheric physical uncertainty remain explicit limitations.",
            "isotope_convention": candidate["isotope_internal_states"],
        },
        "sources": {
            "evaluation_2025": "https://doi.org/10.1063/5.0273661",
            "thermochemical_table": "https://doi.org/10.60893/figshare.jpr.29087342.v1",
            "partition_table": "https://doi.org/10.60893/figshare.jpr.29087339.v1",
            "discrete_alternative": "https://doi.org/10.60893/figshare.jpr.29087330.v1",
            "atct_1_172": "https://doi.org/10.17038/CSE/2440254",
            "original_file_sha256": candidate["source_sha256"],
            "review": "documents/mceg/h2o_source_review.rst",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "substitution": "tests/data/validation/h2o_equilibrium_impact/20261104/summary.json",
        },
        "source_sha256": {
            str(CANDIDATE.relative_to(ROOT)): CANDIDATE_SHA256,
            str(reference.relative_to(ROOT)): sha256(reference.read_bytes()).hexdigest(),
        },
        "interpretation": "H2O-only adoption on curated-co2-v1, inheriting CO2, TiO and C2H updates. The evaluated Cp polynomial and its exact H/S integrals are copied without refitting. Scoped source and numerical support do not establish complete planetary physical accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False) + "\n")
