"""Require review when files covered by the H2 evidence snapshot change."""

import hashlib
import json
from pathlib import Path


RECEIPT = "results/mceg/h2_tests_c2d3e8b.json"


def test_h2_evidence_matches_selected_source_files() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / RECEIPT).read_text())
    for name, expected in receipt["selected_file_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Review MCEG evidence after changing {name}"
    report = root / receipt["junit_xml"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == receipt["junit_sha256"]


def test_h2_contract_and_guide_reference_current_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / RECEIPT).read_text())
    contract = (root / "mceg/contracts/h2_solubility.yaml").read_text()
    guide = (root / "mceg/README.md").read_text()
    revision = receipt["audited_revision"]

    # Check this pilot's fixed layout without adding a YAML or MCEG dependency.
    # Full structural validation remains a separate MCEG operation.
    assert f"    revision: {revision}\n" in contract
    evidence = contract.split("\n- id: provider_test_run\n", 1)[1]
    evidence = evidence.split("\n- id:", 1)[0].split("\nlimitations:", 1)[0]
    for name in (RECEIPT, receipt["junit_xml"]):
        assert f"  - {name}\n" in evidence
        assert f"](../{name})" in guide
    assert f"revision {revision}." in evidence
    assert f"`{revision}`" in guide
