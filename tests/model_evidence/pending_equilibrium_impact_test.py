"""Test the shared runner's archived-array checks without new equilibrium solves."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import pending_equilibrium_impact as impact


@pytest.mark.parametrize("network", ["ions", "neutral"])
def test_constraints_accept_real_archive_and_reject_formation_perturbation(network):
    path = ROOT/f"tests/data/validation/c2h2_equilibrium_impact/20261106/{network}.npz"
    with np.load(path) as a:
        species, elements = list(a["species"]), list(a["elements"])
        args = (a["formula_matrix"], elements, species, a["budget"], a["ln_n"])
        result = impact.constraints(*args, a["h"], a["pressure_bar"], ["C", "N"])
        assert result["mass_action_max_abs_ln"].max() < 2e-12
        bad = a["h"].copy()
        bad[1, 5, species.index("C2H2")] += .001
        with pytest.raises(AssertionError):
            impact.constraints(*args, bad, a["pressure_bar"], ["C", "N"])


@pytest.mark.parametrize("targets", [["C2H2"], ["C2H2", "C1H2"]])
def test_case_selection_supports_one_and_multiple_replacement_targets(targets):
    path = ROOT/"tests/data/validation/c2h2_equilibrium_impact/20261106/ions.npz"
    with np.load(path) as a:
        species, elements = list(a["species"]), list(a["elements"])
        carbon = elements.index("C")
        share = np.exp(a["ln_n"])*a["formula_matrix"][carbon]/a["budget"][:, carbon, None]
        logx = impact.log_fractions(a["ln_n"])
        effect, relative, highlight, selected, edges = impact.select_cases(
            logx, {"C": share}, [species.index(s) for s in targets], a["temperature_K"])
        assert effect.shape == (75,)
        assert relative.shape == (2, 75, len(targets))
        assert highlight == 55
        assert len(np.unique(selected)) == 3 and len(np.unique(edges)) == 2
        assert highlight in edges
        assert np.max(a["temperature_K"][selected]) == 4000.


def test_generic_replay_accepts_archived_controls_and_rejects_changed_potential(tmp_path):
    source = ROOT/"tests/data/validation/c2h2_equilibrium_impact/20261106"
    model = json.loads((ROOT/"tests/data/validation/c2h2_candidate/20261106/candidate.json").read_text())
    definition = {"baseline_archive": "tests/data/validation/curated_tio2_adoption/20261106",
        "tracked_elements": ["C"], "variants": [
            {"name": "retained", "replacements": {"C2H2": {"kind": "retained_energy", "coefficients": model["original_logK_coefficients"], "delta_GH_J_mol": 1135.}}},
            {"name": "nasa", "replacements": {"C2H2": {"kind": "nasa9", "coefficients": model["nasa_control_coefficients"]}}}]}
    definition_path = tmp_path/"definition.json"
    definition_path.write_text(json.dumps(definition))
    for network in ("ions", "neutral"):
        with np.load(source/f"{network}.npz") as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        scale = np.zeros((2, 75))
        corrected = arrays["delta_g_corrected"]
        arrays.update(converged=np.ones((3, 75), bool), path_variant_index=1,
            path_case_index=int(arrays["highlighted_case_index"]),
            selected_case_indices=[0, 74], gibbs_numerical_scale=scale,
            gibbs_ordering_resolved=(corrected[:, 0] > scale) & (corrected[:, 1] < -scale))
        np.savez_compressed(tmp_path/f"{network}.npz", **arrays)
    summary = json.loads((source/"summary.json").read_text())
    summary["selected_variant_solves"] = 0
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    for filename in ("diagnostics.json", "selected_diagnostics.json"):
        (tmp_path/filename).write_bytes((source/filename).read_bytes())

    def write_receipt():
        (tmp_path/"receipt.json").write_text(json.dumps({"result": "passed", "selected_file_sha256": {},
            "output_sha256": {f.name: impact.file_hash(f) for f in tmp_path.iterdir() if f.name != "receipt.json"}}))

    write_receipt()
    assert impact.replay(definition_path, tmp_path)["new_endpoint_solves"] == 300
    with np.load(tmp_path/"ions.npz") as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    arrays["h"][1, 5, list(arrays["species"]).index("C2H2")] += .001
    np.savez_compressed(tmp_path/"ions.npz", **arrays)
    write_receipt()
    with pytest.raises(AssertionError):
        impact.replay(definition_path, tmp_path)
