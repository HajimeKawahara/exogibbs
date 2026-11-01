"""Protect the SH source diagnosis and the existing balanced energy correction."""

import json
import math
from pathlib import Path
import re
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/sh_source_investigation/20261102"


@pytest.fixture(scope="module")
def review():
    return runpy.run_path(str(ROOT / "mceg/reviews/sh_source_investigation.py"))


def test_official_nodes_and_frozen_report_replay(review):
    report = review["investigate"]()
    expected = json.loads((DATA / "comparison.json").read_text())
    review["baseline"].check_equal(report, expected)
    for row in report["samples"]:
        assert row["reconstructed_delta_ln_K"] == pytest.approx(
            row["ln_K_janaf_hs"] - row["ln_K_nasa"], abs=2e-13)
    assert report["samples"][2]["raw_minus_reconstructed_dex"] == pytest.approx(2.234327923070567)
    assert report["energy_atct_D0_kJ_mol"] == pytest.approx(350.054)


def test_official_nasa_appendix_coefficients_match_baseline(review):
    excerpt = (DATA / "nasa_sh_appendix_d.txt").read_text()
    coefficients = [float(x.replace("D", "E")) for x in
                    re.findall(r"[+-]?\d+\.\d+D[+-]\d+", excerpt)]
    record = review["baseline"].load_inputs()["nasa9_records"]["SH"]
    assert coefficients == [v for interval in record["intervals"] for v in interval["coefficients"]]
    assert len(coefficients) == 18


def test_existing_curated_correction_is_balanced_and_applied_once(review):
    import jax
    from exogibbs.presets import curated

    jax.config.update("jax_enable_x64", True)
    setup = curated.chemsetup(ions=False)
    report = review["investigate"]()
    index = setup.species.index("H1S1")
    for row in report["samples"]:
        if row["temperature_K"] < 1500:
            assert row["predicted_curated_over_nasa"] is None
            continue
        actual = -float(setup.hvector_func(float(row["temperature_K"]))[index])
        assert actual-row["ln_K_nasa"] == pytest.approx(row["atct_energy_shift_ln_K"], abs=2e-11)
    sources = json.loads((DATA / "sources.json").read_text())
    atoms = sources["atct_1_172"]
    molecular_only = -(atoms["SH"]["hf298_kJ_mol"]-142.135309)*1000/8.31446261815324/1500
    assert not math.isclose(molecular_only, report["samples"][2]["atct_energy_shift_ln_K"])
