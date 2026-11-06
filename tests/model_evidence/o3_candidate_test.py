"""Replay independently anchored O3 candidates and finite gas responses."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import pending_thermal_gap_inputs as inputs
from pending_equilibrium_impact import replay
from curated_adoption_history import historical_source
DATA = ROOT / "tests/data/validation/o3_candidate/20261106"
IMPACT = ROOT / "tests/data/validation/o3_equilibrium_impact/20261106"


def test_physical_source_cycle_and_continuous_calorics():
    candidate = json.loads((DATA / "candidate.json").read_text())
    assert inputs.build("o3") == candidate
    record = candidate["source_records"]["o3"]
    assert record["janaf_D0_J_mol"] == pytest.approx(595021.0, abs=1e-8, rel=0)
    assert record["atct_D0_J_mol"] == pytest.approx(596120.0, abs=1e-8, rel=0)
    assert inputs.validate(candidate) == json.loads((DATA / "validation.json").read_text())
    chosen = candidate["variants"][0]["replacements"]["O3"]
    old = {**chosen, "delta_GH_J_mol": 0.}
    for t in (1500., 2137.4, 4000.):
        a, b = [list(inputs.model.reaction_properties(t, spec).values()) for spec in (chosen, old)]
        np.testing.assert_allclose(np.subtract(a, b), [-1099.0, -1099.0, 0., 0.], atol=1e-8, rtol=0)
    for t in (1499., 4001., float("nan")):
        with pytest.raises(ValueError):
            inputs.model.formation_h(t, chosen)


def test_construction_inputs_and_outputs_remain_pinned():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["backend"] == "cpu"
    for path, digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest() == digest
    for name, digest in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == digest


def test_finite_response_archives_and_constraints():
    result = replay(DATA / "candidate.json", IMPACT)
    assert result["new_endpoint_solves"] == 300
    assert result["tighter_control_solves"] == 18
    assert result["path_control_solves"] == 10
    assert result["independent_fastchem_runs"] == 6
