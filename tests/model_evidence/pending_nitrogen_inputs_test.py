"""Guard physical nitrogen anchors, native NASA R and coupled-source identity."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import pending_nitrogen_inputs as inputs
from tio_candidate import _nasa


@pytest.mark.parametrize("slug,delta", [
    ("no", -1024), ("n2o", -829), ("no2", -1058), ("hno", -7613),
    ("n2h4", -2739), ("no3", -1917), ("n2o4", -1691), ("cis_hno2", 740),
])
def test_physical_source_shift_and_literal_intercept_are_distinct(slug, delta):
    record = inputs.source_record(slug)
    assert record["delta_D0_J_mol"] == delta
    spec = inputs.retained_spec(record)
    assert spec["delta_GH_J_mol"] == -delta
    assert abs(record["formal_fit_intercept_Ra_J_mol"]-record["janaf_D0_J_mol"]) > 1
    for t in (1500., 1937.31, 2500., 3999.91, 4000.):
        selected = inputs.independent_properties(record, t, "retained_energy")
        unchanged = deepcopy(record)
        unchanged["delta_D0_J_mol"] = 0
        old = inputs.independent_properties(unchanged, t, "retained_energy")
        assert selected["G_J_mol"]-old["G_J_mol"] == pytest.approx(-delta, abs=1e-8, rel=0)
        assert selected["H_J_mol"]-old["H_J_mol"] == pytest.approx(-delta, abs=1e-8, rel=0)
        assert selected["S_J_mol_K"] == old["S_J_mol_K"]
        assert selected["Cp_J_mol_K"] == old["Cp_J_mol_K"]


@pytest.mark.parametrize("slug", sorted(inputs.NASA_ALTERNATIVES))
def test_nasa_energy_removed_once_in_native_units_before_si_insertion(slug):
    record = inputs.source_record(slug)
    spec = inputs.nasa_spec(record)
    assert spec["native_energy_removal_D0_over_R_K"] == record["nasa_D0_J_mol"] / inputs.NASA_R
    assert spec["selected_energy_insertion_D0_over_R_K"] == record["atct_D0_J_mol"] / inputs.R
    for t in (1500., 1927.1, 2500., 3999.73, 4000.):
        g, slope, curvature = _nasa(t, spec["coefficients"], np)
        actual = np.asarray([inputs.R*g, inputs.R*(g-t*slope), -inputs.R*slope, -inputs.R*t*curvature])
        independent = np.asarray(list(inputs.independent_properties(record, t, "nasa9").values()))
        np.testing.assert_allclose(actual, independent, rtol=0, atol=1e-8)
        # Removing H0 with SI R would leave a real energy error, not rounding.
        wrong_r = record["nasa_D0_J_mol"] * (1-inputs.R/inputs.NASA_R)
        assert abs(wrong_r) > 1
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(actual + [1e-5, 0, 0, 0], independent, rtol=0, atol=1e-8)


def test_missing_zero_kelvin_and_isomer_mixture_are_not_fabricated():
    pentoxide = inputs.source_record("n2o5")
    assert pentoxide["janaf_D0_J_mol"] is None
    assert pentoxide["atct_D0_J_mol"] == 2151095
    with pytest.raises(ValueError, match="missing JANAF"):
        inputs.retained_spec(pentoxide)
    assert [v["name"] for v in inputs.build("n2o5")["variants"]] == ["nasa"]
    cis = inputs.source_record("cis_hno2")
    assert cis["nasa_same_identity"] is False and cis["nasa_D0_J_mol"] is None
    with pytest.raises(ValueError, match="isomer-mixture"):
        inputs.nasa_spec(cis)


def test_no2_dimer_joint_energy_cycle_uses_both_targets():
    definition = inputs.build("n2o4")
    single, joint = definition["variants"]
    assert set(single["replacements"]) == {"N2O4"}
    assert set(joint["replacements"]) == {"N1O2", "N2O4"}
    shift = joint["replacements"]["N2O4"]["delta_GH_J_mol"]-2*joint["replacements"]["N1O2"]["delta_GH_J_mol"]
    assert shift == definition["special_controls"]["association_delta_GH_J_mol"] == -425


def test_cis_trans_energy_gap_is_not_a_mixture_partition_function():
    definition = inputs.build("cis_hno2")
    cis_only, joint, measured = definition["variants"]
    cis, trans = "H1N1O2cis", "H1N1O2trans"
    assert set(cis_only["replacements"]) == {cis}
    assert set(joint["replacements"]) == set(measured["replacements"]) == {cis, trans}
    gap = definition["special_controls"]["conformer_gap"]
    old_gap = gap["JANAF_gap_J_mol"]
    for variant, desired in [(joint, gap["ATcT_1.228_gap_J_mol"]),
                             (measured, gap["measured_2023_gap_J_mol"])]:
        specs = variant["replacements"]
        actual = old_gap + specs[cis]["delta_GH_J_mol"]-specs[trans]["delta_GH_J_mol"]
        assert actual == pytest.approx(desired, abs=1e-8, rel=0)
        assert specs[cis]["coefficients"] != specs[trans]["coefficients"]
    for edge, sign in zip(definition["selected_variants"], (-1, 1)):
        change = edge["replacements"][cis]["delta_GH_J_mol"]-measured["replacements"][cis]["delta_GH_J_mol"]
        assert change == pytest.approx(sign*gap["quoted_gap_uncertainty_J_mol"], abs=1e-8, rel=0)
        assert edge["replacements"][trans] == measured["replacements"][trans]
