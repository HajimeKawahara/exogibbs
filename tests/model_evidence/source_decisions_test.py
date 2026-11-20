"""Keep source-choice records complete without importing optional MCEG."""

from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / "mceg/decisions"


def _decision_id(species):
    species = species.replace("+", "_plus").replace("-", "_minus")
    return "source_choice_" + re.sub(r"[^A-Za-z0-9_.-]", "_", species)


def test_every_frozen_source_choice_has_a_scoped_mceg_decision():
    count = 0
    datasets = sorted((ROOT / "src/exogibbs/data/curated").glob("*.json"))
    assert {p.stem for p in DECISIONS.glob("*.yaml")} == {p.stem for p in datasets}
    assert len(datasets) == 16
    for path in datasets:
        data = json.loads(path.read_text())
        choices = data.get("decisions") or {data["species"]: data["decision"]}
        contract = json.loads((DECISIONS / (path.stem + ".yaml")).read_text())
        assert contract["schema_version"] == "0.4"
        decisions = {item["id"]: item for item in contract["decisions"]}
        assert len(decisions) == len(contract["decisions"]) == len(choices)
        assert set(decisions) == {_decision_id(species) for species in choices}
        for species, choice in choices.items():
            decision = decisions[_decision_id(species)]
            assert decision["revision"] == "1"
            assert decision["reasons"][0] == choice["reason"]
            assert decision["selected"] == "recorded_source"
            assert decision["selected"] in {option["id"] for option in decision["options"]}
            assert decision["scope"] and decision["limitations"]
            if "condition" in choice:
                assert choice["condition"] in decision["limitations"]
            assert decision["basis"]
        count += len(decisions)
    assert count == 124


def test_decisions_pin_existing_evidence_without_rewriting_frozen_artifacts():
    for path in sorted(DECISIONS.glob("*.yaml")):
        contract = json.loads(path.read_text())
        evidence = next(item for item in contract["evidence"] if item["id"] == "archived_review")
        artifacts = {item["source"]: item["sha256"] for item in evidence["artifacts"]}
        dataset = f"src/exogibbs/data/curated/{path.stem}.json"
        assert dataset in artifacts
        assert any(source.startswith("mceg/contracts/") for source in artifacts)
        for source, digest in artifacts.items():
            assert sha256((ROOT / source).read_bytes()).hexdigest() == digest
        assert all(any(pin["kind"] == "evidence" and pin["id"] == evidence["id"]
                       for pin in decision["basis"])
                   for decision in contract["decisions"])
