"""Check sampled atom-reference inputs and a primitive equilibrium benchmark."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/ohno_exogibbs_20261031.py"))


def test_sampled_model_rejects_interpolation_and_matches_analytic_hydrogen():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        data = {"elements": ["H"], "species": ["H", "H2"], "formula_matrix": [[1, 2]],
                "budget": [1.], "temperature_K": [1000., 500.], "pressure_bar": .001,
                "lnK_bar": [[0., np.log(1e4)], [0., np.log(1e6)]]}
        case = STUDY["prepare_case"](data)
        setup = STUDY["make_setup"](case)
        assert np.isnan(setup.hvector_func(np.array([499., 501., 750., 1001., np.nan]))).all()
        result = STUDY["solve_case"](case)
        expected_h = 2. / (1. + np.sqrt(1. + 4.*np.array([1e6, 1e4])*.001))
        np.testing.assert_allclose(np.exp(result["log_mole_fractions"])[:, 0], expected_h, rtol=1e-8)
        assert result["cases_passed"] == 2
        assert data["temperature_K"] == [1000., 500.]
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_common_network_preserves_isomers_and_excludes_conflicting_identities():
    elements = ["H", "O", "P"]
    names = ["H", "O", "P", "P4O6", "H3PO4", "HPO2", "H3PO3:1", "H3PO3:2", "conflict"]
    compositions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 6, 4],
                    [3, 4, 1], [1, 2, 1], [3, 3, 1], [3, 3, 1], [1, 0, 1]]
    matrix = np.asarray(compositions).T
    original = {"elements": elements, "species": names, "formula_matrix": matrix.tolist(),
                "budget": [1., .001, .000001], "temperature_K": [500.], "pressure_bar": .001,
                "lnK_bar": [[0., 0., 0., 10., 20., 30., 40., 50., 60.]]}
    first = STUDY["prepare_case"](original)
    changed = {**original, "formula_matrix": matrix.copy().tolist()}
    changed["formula_matrix"][0][-1] = 2
    second = STUDY["prepare_case"](changed)
    cases = STUDY["controlled_cases"](first, second)
    assert "H3PO3:1" in cases["common_janaf"]["species"]
    assert "H3PO3:2" in cases["common_janaf"]["species"]
    assert "conflict" not in cases["common_janaf"]["species"]
    assert "H3PO4" not in cases["burcat_without_H3PO4_HPO2"]["species"]
    assert "H3PO3:1" in cases["burcat_without_H3PO4_HPO2"]["species"]
    assert original["species"] == names


def test_relative_native_input_cli_records_the_executed_file_hash(tmp_path):
    data = {"cases": {"hydrogen": {
        "elements": ["H"], "species": ["H", "H2"], "formula_matrix": [[1, 2]],
        "budget": [1.], "temperature_K": [500.], "pressure_bar": .001,
        "lnK_bar": [[0., np.log(1e6)]]}}}
    source = tmp_path / "native.json"
    source.write_text(json.dumps(data))
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
    subprocess.run([sys.executable, str(ROOT / "mceg/reviews/ohno_exogibbs_20261031.py"),
                    "--native-input", "native.json", "--output-dir", "output", "--native-only"],
                   cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "output/exogibbs_summary.json").read_text())
    assert report["cases_passed"] == 1
    assert report["selected_file_sha256"][str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
