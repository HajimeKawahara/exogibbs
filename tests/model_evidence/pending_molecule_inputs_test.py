"""Test source-energy construction using the already merged acetylene packet."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import pending_molecule_inputs as inputs
import pending_candidate as candidate


def test_atct_cycle_and_native_nasa_alignment_on_merged_source():
    record = inputs.source_record("c2h2")
    assert record["janaf_D0_J_mol"] == 1627153.
    assert record["selected_D0_J_mol"] == 1626018.
    assert record["delta_GH_J_mol"] == 1135.
    for make_spec in (inputs.retained_spec, inputs.nasa_spec):
        spec = make_spec(record)
        for t in (1500., 2314.15, 4000.):
            expected = inputs.independent_properties(record, t, spec["kind"])
            actual = candidate.reaction_properties(t, spec)
            np.testing.assert_allclose(list(actual.values())[:2], list(expected.values())[:2], atol=1e-8, rtol=0)
            np.testing.assert_allclose(list(actual.values())[2:], list(expected.values())[2:], atol=1e-10, rtol=0)


def test_h298_shift_requires_no_invented_zero_kelvin_anchor():
    record = {"original_coefficients": [1., 2., 3., 4., 5.], "delta_GH_J_mol": 56542.,
              "janaf_D0_J_mol": None, "selected_D0_J_mol": None, "nasa_same_identity": True}
    assert inputs.retained_spec(record)["delta_GH_J_mol"] == 56542.
    with pytest.raises(ValueError):
        inputs.nasa_spec(record)
    with pytest.raises(ValueError):
        inputs.nasa_spec({**record, "nasa_same_identity": False, "selected_D0_J_mol": 100.})
