"""Keep the archived curated evidence tied to its inputs and completed run."""

import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path("results/curated_gas_validation")
CONTRACT = Path("mceg/contracts/curated_gas.yaml")
BUILD = Path("results/gas_thermochemistry_impact/fastchem_build.json")
INPUTS = (
    "examples/comparisons/validate_curated_preset.py",
    "examples/comparisons/build_curated_preset.py",
    "src/exogibbs/data/curated/gas_v1.json",
    "examples/comparisons/data/curated_continuous_reference.json",
    "src/exogibbs/presets/curated.py",
    "src/exogibbs/equilibrium/gas/kernel/solver.py",
    "src/exogibbs/equilibrium/gas/solve.py",
    "src/exogibbs/equilibrium/gas/types.py",
)
OUTPUTS = ("thermal_checks.json", "diagnostics.csv", "default_charge.csv",
           "ions.npz", "neutral.npz", "fastchem_parity.csv")


def check_receipt(root: Path) -> None:
    receipt = json.loads((root / RESULTS / "receipt.json").read_text())
    assert receipt["dataset"] == "curated-gas-v1"
    assert receipt["dtype"] == "float64"
    assert receipt["epsilon_crit"] == 1e-13
    assert receipt["temperature_K"] == [1500, 1735, 2000, 2517, 3000, 3973, 4000]
    assert receipt["pressure_bar"] == [1e-6, 1e-4, .01, 1, 100]
    assert receipt["cases_passed"] == receipt["fastchem_cases_passed"] == 70
    assert receipt["default_charge_cases_passed"] == 20
    assert receipt["fastchem"] == "passed"
    assert set(receipt["selected_file_sha256"]) == set(INPUTS)
    assert set(receipt["output_sha256"]) == set(OUTPUTS)
    for key, parent in (("selected_file_sha256", root), ("output_sha256", root / RESULTS)):
        for path, expected in receipt[key].items():
            assert hashlib.sha256((parent / path).read_bytes()).hexdigest() == expected, path
    build = json.loads((root / BUILD).read_text())
    assert receipt["fastchem_binary_sha256"] == build["executable_sha256"]
    revision = re.search(r"^    revision: sha256:([0-9a-f]{64})$",
                         (root / CONTRACT).read_text(), re.MULTILINE)
    assert revision is not None
    assert revision[1] == receipt["selected_file_sha256"]["src/exogibbs/presets/curated.py"]


def test_curated_receipt_matches_current_inputs_outputs_and_contract():
    check_receipt(ROOT)


@pytest.mark.parametrize("change", [
    "input", "output", "missing_output", "not_run", "case_count",
    "missing_input_hash", "missing_output_hash", "revision", "fastchem_binary",
])
def test_stale_or_incomplete_curated_evidence_is_rejected(tmp_path, change):
    paths = [Path(p) for p in INPUTS] + [RESULTS / p for p in OUTPUTS]
    for path in paths + [RESULTS / "receipt.json", CONTRACT, BUILD]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    path = tmp_path / RESULTS / "receipt.json"
    receipt = json.loads(path.read_text())
    if change in ("input", "output", "revision"):
        target = tmp_path / {"input": Path(INPUTS[0]), "output": RESULTS / OUTPUTS[0],
                             "revision": CONTRACT}[change]
        content = target.read_text()
        target.write_text(content.replace("revision: sha256:", "revision: sha256:0")
                          if change == "revision" else content + "\n")
    elif change == "missing_output":
        (tmp_path / RESULTS / OUTPUTS[-1]).unlink()
    elif change == "not_run":
        receipt["fastchem"] = "not_run"
    elif change == "case_count":
        receipt["fastchem_cases_passed"] = 0
    elif change == "missing_input_hash":
        del receipt["selected_file_sha256"][INPUTS[0]]
    elif change == "missing_output_hash":
        del receipt["output_sha256"][OUTPUTS[0]]
    else:
        receipt["fastchem_binary_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises((AssertionError, FileNotFoundError)):
        check_receipt(tmp_path)
