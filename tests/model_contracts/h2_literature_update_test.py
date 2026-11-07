"""A private-paper review can be replayed without rewriting earlier judgments."""

import json
from pathlib import Path
import runpy

import pytest

pytest.importorskip("mceg")

from mceg import assess_conditions, load_contract, restore_v01, trace_provenance
from mceg.__main__ import main
from mceg.assessment import compare_condition
from mceg.schema_v02 import Context


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "mceg/reviews/h2_literature_update"
RESULT = ROOT / "tests/data/validation/mceg/h2_literature_update.json"
TRIAL = runpy.run_path(str(HERE / "update.py"))


def test_review_preserves_history_and_replays_without_private_material():
    before = load_contract(TRIAL["BASELINE"])
    after = TRIAL["build"]()
    report = TRIAL["summarize"](after)
    assert report == json.loads(RESULT.read_text())
    assert all(report["historical_records_preserved"].values())
    assert restore_v01(after) == restore_v01(before)
    assert after.subjects == before.subjects and after.claims == before.claims
    assert {a.status for a in after.assessments
            if a.kind == "evidence_support" and a.claim == "physical_accuracy"} == {"unknown"}
    assert next(e for e in after.evidence if e.id == "numerical_tests").result == "passed"
    context = next(c for c in after.contexts if c.id == "h2-1750K-7500bar")
    assert assess_conditions(after, context, "imported") == after


def test_new_source_chain_closes_transcription_but_not_calibration_or_material_scope():
    after = TRIAL["build"]()
    assert not trace_provenance(after, "implementation")["recorded_audit_complete"]
    assert trace_provenance(after, "implementation_full_text")["recorded_audit_complete"]
    metadata = trace_provenance(after, "metadata_temperature_full_text")
    assert {i["issue"] for i in metadata["issues"]} == {"comparison_discrepancy"}
    calibration = trace_provenance(after, "calibration_adoption_full_text")
    assert any(i["evidence"] == "original_calibration" and i["issue"] == "source_not_fully_inspected"
               for i in calibration["issues"])
    outcomes = {a.condition: a.classification for a in after.assessments if a.kind == "range_membership"}
    assert outcomes["temperature_basalt_full_text"] == "outside_reported_range"
    assert outcomes["temperature_andesite_full_text"] == "within_reported_range"
    assert outcomes["pressure_basalt_full_text"] == "within_reported_range"
    assert outcomes["pressure_andesite_full_text"] == "outside_reported_range"

    conditions = {c.id: c for c in after.conditions}
    boundary = Context(id="rounded-boundary-probe", scope="Synthetic rounding probe, not an experiment.",
                       quantities={"temperature": {"value": 1723.15, "unit": "K"}})
    assert compare_condition(conditions["temperature_seo"], boundary)[0] == "outside_reported_range"
    assert compare_condition(conditions["temperature_basalt_full_text"], boundary)[0] == "within_reported_range"
    other_material = boundary.model_copy(update={"settings": {"material": "andesite"}})
    # A range checks its numeric coordinate only; prose scope is not a material guard.
    assert compare_condition(conditions["temperature_basalt_full_text"], other_material)[0] == "within_reported_range"


def test_changed_baseline_and_unreviewed_pdf_are_rejected(tmp_path):
    baseline = tmp_path / "changed.yaml"
    baseline.write_bytes(TRIAL["BASELINE"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Baseline changed"):
        TRIAL["build"](baseline=baseline)
    pdf = tmp_path / "different.pdf"
    pdf.write_bytes(b"Synthetic unreviewed bytes, not an article.")
    with pytest.raises(ValueError, match="Private PDF bytes differ"):
        TRIAL["build"](pdf=pdf)


def test_cli_exposes_both_new_inspection_and_historical_gap(tmp_path, capsys):
    output = tmp_path / "reviewed.yaml"
    assert TRIAL["main"](["--output", str(output)]) == 0
    capsys.readouterr()
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        TRIAL["main"](["--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == before
    capsys.readouterr()
    assert main(["check", str(output), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)["contracts"][0]
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["structure"]["status"] == "PASS"
    assert checks["source_audit"]["status"] == "UNKNOWN"
    assert checks["temperature_basalt_full_text"]["status"] == "FAIL"
    for evidence, issues in (("implementation_full_text", False), ("experimental_fit", True)):
        assert main(["explain", str(output), "--evidence", evidence, "--json"]) == 0
        item = json.loads(capsys.readouterr().out)["contracts"][0]["items"][0]
        assert bool(item["issues"]) == issues
        if evidence == "experimental_fit":
            assert item["recorded_follow_ups"] == []
            assert item["resolved_follow_ups"][0]["follow_up"] == "inspect-original-fit"
            assert item["suggested_steps"] == []
