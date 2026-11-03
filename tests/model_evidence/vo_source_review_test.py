"""Check VO hybrid reconstruction, spin normalization and vanadium accounting."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import vo_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_native_sources_and_current_hybrid_close_independent_archive(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_native_D0_J_mol"] == pytest.approx({"janaf": 631171, "nasa9": 612000})
    for row in report["source_samples"]:
        if row["baseline_closure_ln_K"] is not None:
            assert max(abs(v) for v in row["baseline_closure_ln_K"].values()) < 3e-11
        components = row["curated_minus_nasa_ln_K_components"]
        assert sum(components.values()) == pytest.approx(row["hybrid_node_ln_K"]-row["nasa_ln_K"], abs=2e-14)
        assert sum(row["raw_janaf_minus_hybrid_ln_K_components"].values()) == pytest.approx(
            row["janaf_ln_K"]-row["hybrid_node_ln_K"], abs=3e-14)
        assert abs(sum(row["native_janaf_minus_nasa_atomic_thermal_g_RT"].values())) < .00032
        if row["temperature_K"] >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["hybrid_node_ln_K"], abs=3e-11)
        else:
            assert row["curated_ln_K"] is None
    low, high = report["source_samples"][2], report["source_samples"][-1]
    assert low["curated_over_nasa"] == pytest.approx(4.742820467397992)
    assert high["same_D0_janaf_over_nasa"] == pytest.approx(.8556781231002328)
    assert report["D0_kelvin_used"]["adopted"]/study.major.EV_K == 6.545
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(5400)


def test_partition_spin_mass_and_version_conventions(report):
    raw = {r["temperature_K"]: r for r in json.loads((study.DATA / "partitions.json").read_text())["rows"]}
    for row in report["partition_samples"]:
        t = row["temperature_K"]
        assert row["hyvo_spin_excluded_Q"] == raw[t]["HyVO"]/8
        assert row["fixed_D0_atoms_zero_assumption_K_ratio"] == pytest.approx(
            row["hyvo_over_janaf_internal_Q"]*report["molecular_translation_mass_factor"])
        # Using full ExoMol degeneracy with spin-excluded atoms creates a false factor eight.
        assert raw[t]["HyVO"]/row["janaf_implied_Q"] == pytest.approx(8*row["hyvo_over_janaf_internal_Q"])
    modern = {r["temperature_K"]: r for r in report["partition_samples"]}
    assert modern[1500]["hyvo_over_janaf_internal_Q"] == pytest.approx(1.0027845374045161)
    assert modern[4000]["hyvo_over_janaf_internal_Q"] == pytest.approx(1.3797981416398608)
    assert modern[4000]["paper_outer_level_fraction"] == pytest.approx(11469/795480)
    assert modern[4000]["downloaded_over_paper_vomyt"] != 1
    assert report["molecular_translation_mass_factor"] == pytest.approx((66.93887412/66.9409)**1.5)
    definition = (study.DATA / "hyvo.def").read_text()
    assert "66.93887412" in definition and "20231218" in definition
    spin_rows = [line.split()[0] for line in definition.splitlines() if "Nuclear spin degeneracy" in line]
    assert spin_rows == ["8", "8"]  # Symmetry labels, not two factors to multiply together.


def test_merriles_neutral_energy_uncertainty_is_separate_from_thermal_diagnostic(report):
    literature = json.loads((study.DATA / "literature.json").read_text())
    measurement = literature["merriles2020"]
    assert measurement["D0_eV"] != measurement["D0_Vplus_O_eV"]
    assert abs(measurement["threshold_cm-1"]-measurement["third_scheme_threshold_cm-1"]) < measurement["assigned_error_cm-1"]
    assert "rendered full text" in measurement["inspection"]
    for row in report["source_samples"]:
        expected = math.exp(.002*study.major.EV_K/row["temperature_K"])
        assert row["quoted_D0_only_multiplicative_error"] == pytest.approx(expected)
    assert "assumed, not established" in report["partition_diagnostic"]
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_archived_grid_conserves_vanadium_and_matches_current_hybrid(report):
    footprint = report["footprint"]
    assert footprint["all_cases_converged"]
    assert footprint["cases"] == 75 and footprint["species_count"] == 537
    point = footprint["maximum_vanadium_share"]
    assert point["case_index"] == 28
    assert point["vanadium_fraction"] == pytest.approx(.5881173506455909)
    with np.load(study.GRID) as grid:
        element = grid["elements"].tolist().index("V")
        molecule = grid["species"].tolist().index("O1V1")
        n = np.exp(grid["ln_n"])
        assert n @ grid["formula_matrix"][element] == pytest.approx(grid["budget"][:, element], rel=1e-7, abs=0)
        i = point["case_index"]
        assert point["mole_fraction"] == pytest.approx(n[i, molecule]/n[i].sum(), rel=2e-12, abs=0)
        for t, h in zip(grid["temperature_K"], grid["h"][:, molecule]):
            assert -h == pytest.approx(study.curated_ln_k(t), abs=3e-11)


def test_review_artifact_hashes():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for source in json.loads((study.DATA / "sources.json").read_text())["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA/source["file"]).read_bytes()).hexdigest() == source["sha256"]
