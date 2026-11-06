"""Check the balanced energy reference, source control and candidate derivatives."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
from curated_adoption_history import historical_source
import koh_candidate as candidate


@pytest.fixture(scope="module")
def report():
    return candidate.validate()


def test_balanced_energy_uses_printed_heat_content_and_atomic_references():
    model = candidate.load_candidate()
    assert model == candidate.build()
    assert model["baseline_dataset"] == "curated-cos-v1"
    assert model["original_source_D0_J_mol"] == pytest.approx(781106.0,abs=1e-8,rel=0)
    assert model["target_D0_J_mol"] == pytest.approx(780727.572,abs=1e-8,rel=0)
    assert model["delta_D0_J_mol"] == pytest.approx(model["atomic_h0_change_J_mol"]-model["molecular_h0_change_J_mol"],abs=1e-8,rel=0)
    assert model["Gurvich_molecular_h0_gauge_J_mol"] == model["Gurvich_molecular_hf298_J_mol"]-model["Gurvich_molecular_heat_content_J_mol"]
    # A molecular Hf margin is not a covariance-aware reaction-energy error bar.
    assert "not an independent Gaussian" in model["uncertainty_scope"]
    for path,digest in model["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    for path,digest in receipt["input_sha256"].items():
        assert sha256(historical_source(ROOT,path,digest).read_bytes()).hexdigest() == digest,path
    for path,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/path).read_bytes()).hexdigest() == digest,path
    assert "src/exogibbs/presets/curated_cos.py" in receipt["input_sha256"]
    assert "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat" in receipt["input_sha256"]


def test_energy_only_preserves_entropy_and_heat_capacity():
    delta = candidate.load_candidate()["delta_D0_J_mol"]
    for t in (1500.,1637.,2500.,3000.,4000.):
        old = candidate.reaction_properties(t,"current")
        new = candidate.reaction_properties(t,"energy")
        for key in ("G_J_mol","H_J_mol"):
            assert new[key]-old[key] == pytest.approx(-delta,abs=1e-8,rel=0)
        for key in ("S_J_mol_K","Cp_J_mol_K"):
            assert new[key] == old[key]
        assert new["G_J_mol"] == pytest.approx(new["H_J_mol"]-t*new["S_J_mol_K"],abs=1e-8,rel=0)
        # Two NASA variants share the exact thermal coefficients and differ only
        # in energy; the native-R anchor must not be silently interpreted as SI.
        a = candidate.reaction_properties(t,"nasa")
        b = candidate.reaction_properties(t,"nasa_retained")
        assert a["G_J_mol"]-b["G_J_mol"] == pytest.approx(-delta,abs=1e-8,rel=0)
        assert a["S_J_mol_K"] == pytest.approx(b["S_J_mol_K"],abs=1e-12,rel=0)
        assert a["Cp_J_mol_K"] == pytest.approx(b["Cp_J_mol_K"],abs=1e-12,rel=0)


def test_dense_reconstruction_and_independent_automatic_derivatives(report):
    assert report["temperature_count"] == 2501
    assert max(report["maximum_absolute_reconstruction_error"].values()) < 2e-12
    for values in report["jax_float64_derivative_errors"].values():
        assert values["G_J_mol"] < 1e-8 and values["H_J_mol"] < 1e-8
        assert values["S_J_mol_K"] < 1e-10 and values["Cp_J_mol_K"] < 1e-10
    frozen = json.loads((candidate.DATA/"validation.json").read_text())
    for actual,expected in zip(report["samples"],frozen["samples"]):
        for variant in candidate.VARIANTS:
            assert actual["K_over_current"][variant] == pytest.approx(expected["K_over_current"][variant],abs=1e-12,rel=0)
    assert report["production_change"] == "none" and not report["OYT_adopted"]


def test_domain_guard_and_named_source_choices():
    import jax
    jax.config.update("jax_enable_x64",True)
    for variant in candidate.VARIANTS:
        for t in (1499.,4001.,np.nan,np.inf):
            with pytest.raises(ValueError,match="domain"):
                candidate.formation_h(t,variant)
        values = np.asarray(jax.jit(lambda t:candidate.formation_h_jax(t,variant))(np.array([1499.,1500.,4000.,4001.,np.nan])))
        assert np.isnan(values[[0,3,4]]).all()
        np.testing.assert_allclose(values[1:3],[candidate.formation_h(t,variant) for t in (1500.,4000.)],atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match="Unknown"):
        candidate.formation_h(2000.,"oyt")
