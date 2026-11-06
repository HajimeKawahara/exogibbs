"""Package the reviewed TiO candidate without fitting or changing its functions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/tio_candidate/20261103/candidate.json"
CANDIDATE_SHA256 = "e4bcb67ff0bee98517d81f784a62a61099553865eedb79955862e5f099577662"
BASE = ROOT / "src/exogibbs/data/curated/c2h_v1.json"
BASE_SHA256 = "b2df12855bc20a27e3bed06e8ad4d6a421e5f47a8a682d6e3fdd74e3f698de48"
OUTPUT = ROOT / "src/exogibbs/data/curated/tio_v1.json"


def build_data() -> dict:
    """Return a single-species extension of the unchanged curated-c2h-v1 data."""
    for path, expected in ((CANDIDATE, CANDIDATE_SHA256), (BASE, BASE_SHA256)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"TiO adoption requires its reviewed input: {path}")
    candidate = json.loads(CANDIDATE.read_text())
    base = json.loads(BASE.read_text())
    return {
        "id": "curated-tio-v1", "species": "O1Ti1", "reaction": "Ti(g) + O(g) -> TiO(g)",
        "base_dataset": base["id"], "base_data_sha256": BASE_SHA256,
        "temperature_range_K": candidate["temperature_range_K"], "standard_pressure_bar": 1.,
        "knots_K": candidate["knots_K"], "coefficients": candidate["coefficients"],
        "atomic_background_nasa9": candidate["atomic_background_nasa9"],
        "D0_eV": candidate["D0_eV"],
        "D0_over_k_K": candidate["D0_eV"]*1.602176634e-19/1.380649e-23,
        "molecular_mass_u": candidate["molecular_mass_u"],
        "nuclear_spin_factor": candidate["nuclear_spin_factor"],
        "energy_zero": candidate["energy_zero"], "isotope_convention": candidate["isotope_convention"],
        "representation": candidate["representation"].replace("; no production update", ""),
        "cutoff": candidate["cutoff"], "completeness": candidate["completeness"],
        "decision": {
            "name": "TiO", "status": "prefer_conditionally", "source_group": "molecular",
            "thermal_source": "ExoMol TOTO 20240509 bound-state moments",
            "energy_source": "Jain et al. 2023 ground-state D0",
            "reason": "Replace the JANAF 1973 low-state estimates with modern explicit levels and a more precise ground-state threshold. The older 6.869 +/- 0.061 eV energy is not disproved by the compatible 6.824 +/- 0.010 eV result.",
            "condition": "1500-4000 K, 1 bar standard pressure, representative-isotope approximation; complete high-temperature and atmospheric physical uncertainty remains unquantified.",
            "isotope_convention": candidate["isotope_convention"],
        },
        "sources": {
            "toto_2019": "https://doi.org/10.1093/mnras/stz1818",
            "toto_2024": "https://doi.org/10.1093/rasti/rzae037",
            "jain_2023": "https://doi.org/10.1063/5.0146365",
            "original_states": candidate["original_states"],
            "D0_source": candidate["D0_source"],
            "review": "documents/mceg/tio_source_review.rst",
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "substitution": "tests/data/validation/tio_equilibrium_impact/20261103/summary.json",
        },
        "source_sha256": {
            str(CANDIDATE.relative_to(ROOT)): CANDIDATE_SHA256,
            **candidate["source_sha256"],
        },
        "interpretation": "Single-species TiO adoption on curated-c2h-v1. The exact reviewed C2 quintic functions use the central-D0 bound-state inventory. Source and numerical support do not establish complete mixture or planetary-atmosphere accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False)+"\n")
