"""Protect chemical identity decisions in the optional source coverage audit."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_source_audit_20261031.py"))
candidate_id = STUDY["candidate_id"]
classify_record = STUDY["classify_record"]


def test_tritium_record_is_not_a_phosphine_candidate():
    classification, note = classify_record("PT", selected=False)
    assert classification == "excluded_isotope"
    assert "not three H atoms" in note
    assert classify_record("PH3", selected=False) == ("alternate_anchor", "PH3")
    assert classify_record("PC", selected=False) == ("alternate_anchor", "CP")


def test_same_formula_isomers_remain_distinct():
    for name in ("H3PO", "C2H7P", "C3H9P"):
        assert classify_record(name, selected=False) == ("candidate", None)
        assert candidate_id(name, 0, 2) != candidate_id(name, 1, 2)
    assert candidate_id("PO3", 0, 1) == "PO3"
