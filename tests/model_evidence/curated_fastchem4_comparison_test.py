"""Protect the updated HCN selection and the frozen formation-constant comparison."""

from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import curated_fastchem4_comparison as review


@pytest.fixture(scope="module")
def replay():
    return review.compute()


def test_official_full_table_contains_the_supplied_hcn_update():
    raw = review.TABLE.read_bytes()
    assert sha256(raw).hexdigest() == "3697abccf664aa5f531103b6c7aac6e77efcd4366f8b69d46b7e6ea29a10e0c5"
    with zipfile.ZipFile(ROOT / "tests/data/validation/new_fastchem_hcn/logK_hcn.zip") as archive:
        assert raw == archive.read("logK.dat")
    assert b"C1H1N1_hcn Hydrogen_Cyanide_HCN" in raw
    assert b"C1H1N1_hnc Hydrogen_Isocyanide_HNC" in raw
    assert b"1.5281916243376781e+05" in raw
    assert raw != (ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat").read_bytes()


def test_changed_input_is_rejected_before_comparison(monkeypatch):
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: original(path) + (b"\n" if path == review.TABLE else b""))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        review.load_inputs()


def test_archive_replays_all_records_without_upgrading_nasa_exclusions(replay):
    archived = json.loads((review.DATA / "comparison.json").read_text())
    def compare(actual, expected):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                compare(actual[key], expected[key])
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                compare(left, right)
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=2e-11, abs=3e-13)
        else:
            assert actual == expected
    compare(replay, archived)
    assert replay["curated_dataset"] == "curated-gas-v2"
    assert replay["molecule_count"] == 108 and replay["sample_count"] == 540
    assert replay["equilibrium_abundance_solves"] == 0
    assert replay["standard_pressure_Pa"] == 100000
    assert replay["native_provider_max_abs_ln_K_error"] < 2e-11
    excluded = [row for row in replay["samples"] if row["scope"] != "main93"]
    assert len(excluded) == 15 * 5
    assert all(row["ln_K_frozen_NASA"] is None and row["nasa_grade"] is None for row in excluded)
    assert all(row["curated_over_fastchem4"] > 0 for row in excluded)
    receipt = json.loads((review.DATA / "receipt.json").read_text())
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    assert receipt["inputs_sha256"] == review.digest(review.DATA / "inputs.json")
    for name, expected in receipt["output_sha256"].items():
        assert review.digest(review.DATA / name) == expected


def test_hcn_independent_decimal_fit_and_ratio_orientation(replay):
    coefficients = ["1.5281916243376781e5", "-3.2553951782995711", "-5.9159043461952807",
                    "1.0101019438732398e-3", "-5.3488185063095000e-8"]
    with localcontext() as context:
        context.prec = 45
        a, b, c, d, e = map(Decimal, coefficients)
        for row in replay["HCN"]:
            t = Decimal(str(row["temperature_K"]))
            expected = a / t + b * t.ln() + c + d * t + e * t * t
            assert row["ln_K_fastchem4"] == pytest.approx(float(expected), abs=3e-13, rel=0)
            ratio = (Decimal(str(row["ln_K_curated"])) - expected).exp()
            assert row["curated_over_fastchem4"] == pytest.approx(float(ratio), rel=3e-13)
    assert replay["HCN"][0]["curated_over_fastchem4"] == pytest.approx(1.0335697183423342, rel=2e-11)
    assert replay["HCN"][-1]["curated_over_fastchem4"] == pytest.approx(1.0141217766979014, rel=2e-11)


def test_formation_ratios_keep_the_frozen_nasa_reference(replay):
    for row in replay["samples"]:
        assert math.log10(row["curated_over_fastchem4"]) == pytest.approx(
            row["log10_curated_over_fastchem4"], abs=3e-14)
        if row["scope"] == "main93":
            assert row["curated_over_frozen_NASA"] == pytest.approx(
                row["curated_over_fastchem4"] * math.exp(row["ln_K_fastchem4"] - row["ln_K_frozen_NASA"]),
                rel=3e-13)
    # Shared ancestry is allowed to give exact agreement, not an accuracy claim.
    vo2 = next(row for row in replay["samples"] if row["molecule"] == "VO2" and row["temperature_K"] == 1500)
    assert vo2["curated_over_fastchem4"] == 1.
