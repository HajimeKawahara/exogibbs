"""Inspect source choices with MCEG without promoting their scientific support."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mceg.schema_v04")

from mceg import load_contract
from mceg.__main__ import main
from mceg.decisions import decision_basis_checks
from mceg.schema_v04 import decision_basis_digest


ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / "mceg/decisions"


def test_all_decision_basis_pins_resolve_to_the_recorded_review():
    paths = sorted(DECISIONS.glob("*.yaml"))
    assert len(paths) == 16
    count = 0
    for path in paths:
        contract = load_contract(path)
        for decision in contract.decisions:
            assert decision_basis_checks(contract, decision)[0]["status"] == "PASS"
            for pin in decision.basis:
                assert decision_basis_digest(contract, pin.kind, pin.id) == pin.sha256
        count += len(contract.decisions)
    assert count == 124


def test_transcribed_assessments_preserve_original_claims_and_subjects():
    for path in sorted(DECISIONS.glob("*.yaml")):
        contract = load_contract(path)
        evidence = next(item for item in contract.evidence if item.id == "archived_review")
        source = next(item.source for item in evidence.artifacts
                      if item.source.startswith("mceg/contracts/"))
        original = load_contract(ROOT / source)
        if original.schema_version == "0.1":
            # The legacy declaration predates explicit Subjects and Assessments.
            continue
        for assessment in contract.assessments:
            before = next(item for item in original.assessments if item.id == assessment.id)
            for field in ("claim", "claim_revision", "status", "reasons", "context"):
                assert getattr(assessment, field) == getattr(before, field)
            claim = next(item for item in contract.claims
                         if (item.id, item.revision) == (assessment.claim, assessment.claim_revision))
            old_claim = next(item for item in original.claims
                             if (item.id, item.revision) == (before.claim, before.claim_revision))
            for field in ("kind", "statement", "scope"):
                assert getattr(claim, field) == getattr(old_claim, field)
            context = next(item for item in contract.contexts if item.id == assessment.context)
            old_context = next(item for item in original.contexts if item.id == before.context)
            assert context == old_context
            subject = next(item for item in contract.subjects if item.id == assessment.subject)
            old_subject = next(item for item in original.subjects if item.id == before.subject)
            assert subject.implementation == old_subject.implementation
            for source, digest in subject.selected_files.items():
                assert old_subject.selected_files[source] == digest


def test_changed_recorded_premise_requires_review_without_changing_selection():
    contract = load_contract(DECISIONS / "c2h_v1.yaml")
    decision = contract.decisions[0]
    selected = decision.selected
    evidence = next(item for item in contract.evidence if item.id == "archived_review")
    evidence.summary += " Synthetic changed premise."
    assert decision_basis_checks(contract, decision)[0]["status"] == "FAIL"
    assert decision.selected == selected


def test_c2h_cli_exposes_conditional_choice_and_unknown_support(capsys):
    path = DECISIONS / "c2h_v1.yaml"
    contract = load_contract(path)
    decision = contract.decisions[0]
    assert any(a.status == "unknown" for a in contract.assessments)
    assert main(["explain", str(path), "--decision", decision.id,
                 "--investigation", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)["contracts"][0]
    item = report["items"][0]
    assert item["record"]["selected"] == decision.selected
    assert item["basis_checks"][0]["status"] == "PASS"
    assert any(a["status"] == "unknown" for a in item["basis_assessments"])
    assert any(a.status == "unknown" for a in load_contract(path).assessments)
