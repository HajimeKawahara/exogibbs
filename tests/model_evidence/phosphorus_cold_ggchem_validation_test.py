"""Check that the new comparison cannot overlook trace gases or changed inputs."""

from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_ggchem_validation_20261031.py"))


def test_trace_abundance_difference_and_nonfinite_closure_fail_comparison():
    metrics = {name: np.zeros(3) for name in STUDY["GATES"]}
    # A difference in a species with arbitrarily small abundance still fails.
    metrics["max_abs_delta_ln_mole_fraction"][1] = 2e-5
    metrics["element_relative_residual"][2] = np.nan
    np.testing.assert_array_equal(STUDY["state_gates"](metrics, np.ones(3, dtype=bool)),
                                  [True, False, False])


def test_intervention_metadata_and_species_order_are_required():
    definition = {"kind": "uniform_energy_probe", "shifted_species": "H3PO4",
                  "offset_kJ_mol": 8.368, "data": {}, "species": ["H", "H3PO4"]}
    case = {key: value for key, value in definition.items() if key != "data"}
    case["intervention"] = "probe"
    assert STUDY["case_inputs"](case, {"probe": definition}, ()) == ({}, ["H", "H3PO4"])
    with pytest.raises(ValueError, match="metadata changed"):
        STUDY["case_inputs"]({**case, "offset_kJ_mol": -8.368}, {"probe": definition}, ())
    with pytest.raises(ValueError, match="identity or order"):
        STUDY["case_inputs"]({**case, "species": ["H3PO4", "H"]}, {"probe": definition}, ())
