"""Keep the source comparison reproducible and its known differences visible."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/mceg/20261021_volatile_sources/comparison.json"


def test_source_review_pins_and_numerical_replay():
    saved = json.loads(REPORT.read_text())
    for name, expected in saved["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    replay = runpy.run_path(str(ROOT / "mceg/reviews/volatile_source_review_20261021.py"))
    actual = replay["comparisons"]()
    assert len(actual) == len(saved["comparisons"])
    for row, recorded in zip(actual, saved["comparisons"]):
        for key in ("model", "reference", "inputs", "purpose", "output_basis", "agrees"):
            assert row[key] == recorded[key]
        for key in ("reference_value", "implementation_value", "implementation_over_reference"):
            assert row[key] == pytest.approx(recorded[key], rel=2e-12, abs=0.)


def test_source_review_does_not_hide_unit_and_pressure_differences():
    rows = json.loads(REPORT.read_text())["comparisons"]
    assert all(r["agrees"] for r in rows if r["purpose"] == "adopted_expression")
    methane = [r for r in rows if r["reference"] == "meltyq2026_A4"]
    assert len(methane) == 3
    assert all(r["implementation_over_reference"] == pytest.approx(1e-4) for r in methane)
    nitrogen = [r for r in rows if r["reference"] == "meltyq2026_A6"]
    assert [r["agrees"] for r in nitrogen] == [False, True, False]
    assert all(not r["agrees"] for r in rows if r["reference"] == "yoshioka_thesis2018")
