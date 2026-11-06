"""Replay the PS compact representation against immutable direct state moments."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import build_ps_adoption as build


@pytest.fixture(scope="module")
def report():
    return build.validate()


def test_dense_off_grid_fit_and_degree_convergence(report):
    assert (report["degree"],report["fit_nodes"],report["dense_points"],report["off_grid_points"])==(18,80,2501,257)
    assert report["validation_points"]==2758 and report["retained_levels"]==223645
    assert report["delta_D0_J_mol"]==0 and report["all_other_potentials_unchanged"]
    for key,limit in report["acceptance_limits"].items():
        assert report["maximum_absolute_fit_errors"][key] < limit
        if key!="ln_Q": assert report["maximum_absolute_runtime_property_errors"][key] < limit
    assert report["maximum_absolute_runtime_h_error"]<2e-12
    by_degree={r["degree"]:r["maximum_absolute_errors"] for r in report["degree_convergence"]}
    assert by_degree[18]["Cp_J_mol_K"] < by_degree[14]["Cp_J_mol_K"] / 100
    assert by_degree[18]["H_J_mol"] < by_degree[14]["H_J_mol"] / 100


def test_packaging_preserves_the_reviewed_energy_cutoff_and_sources():
    model = json.loads(build.OUTPUT.read_text())
    rebuilt=build.build_data()
    a=model["log_partition_representation"].pop("coefficients")
    b=rebuilt["log_partition_representation"].pop("coefficients")
    assert model==rebuilt
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    assert model["original_source_D0_J_mol"]==438000
    assert model["bound_cutoff_cm-1"]==pytest.approx(36613.94086350959,abs=1e-8,rel=0)
    assert model["nuclear_spin_divisor"]==2
    assert model["molecular_mass_u"]==63.038761
    assert model["NASA_native_R_J_mol_K"]==8.314510
    for name,digest in model["source_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def test_fit_receipt_pins_execution_inputs_and_output():
    receipt=json.loads((build.DATA/"fit_receipt.json").read_text())
    assert receipt["result"]=="passed" and receipt["dtype"]=="float64"
    assert receipt["command"][-2:]==["mceg/reviews/build_ps_adoption.py","--validate"]
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((build.DATA/name).read_bytes()).hexdigest()==digest,name
