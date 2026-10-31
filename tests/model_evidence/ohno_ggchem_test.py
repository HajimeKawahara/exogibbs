"""Check native record boundaries before forming chemical constraints."""

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/ohno_ggchem_20261031.py"))


def test_native_header_preserves_two_letter_elements_and_two_digit_counts():
    name, composition = STUDY["parse_record_header"]("C12H4CL4O            4  C  H  O  Cl12  4  1  4")
    assert name == "C12H4CL4O"
    assert composition == {"C": 12, "H": 4, "O": 1, "Cl": 4}


def test_native_label_is_not_used_to_infer_stoichiometry():
    name, composition = STUDY["parse_record_header"]("NCCH2OH              3  C  H  N  2  3  1")
    assert name == "NCCH2OH"
    assert composition == {"C": 2, "H": 3, "N": 1}


def test_changed_upstream_archive_is_rejected_before_build(tmp_path, monkeypatch):
    prepare = STUDY["prepare_binary"]
    monkeypatch.setitem(prepare.__globals__, "OUTPUT", tmp_path)
    (tmp_path / "ggchem_upstream.zip").write_bytes(b"Changed source")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        prepare()
