"""Check retained TiO2 energy, continuous RRHO calorics and source conventions."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import tio2_candidate as candidate


@pytest.fixture(scope="module")
def report():
    return candidate.validate()


def test_source_zero_and_current_fit_are_retained():
    model = candidate.load_candidate()
    assert model == candidate.build()
    assert model["baseline_dataset"] == "curated-cos-v1"
    assert model["original_source_D0_J_mol"] == pytest.approx(1267776.,abs=1e-8,rel=0)
    assert model["delta_D0_J_mol"] == 0
    assert model["nasa_molecular_h0_div_R_K"] == pytest.approx((-305430.-11351.690)/candidate.NASA_R,abs=1e-10,rel=0)
    assert model["new_minus_reference_heat_content_298_J_mol"] == pytest.approx(-169.81843238845067,abs=1e-8,rel=0)
    assert model["symmetry_number"] == 2
    assert model["nuclear_spin_weight"] == model["electronic_degeneracy"] == 1
    assert model["mass_u"] == 79.8788
    assert model["frequencies_cm-1"] == [959,322,944]
    assert "ZPE is already included" in model["physical_scope"]
    for path,digest in model["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    for path,digest in receipt["input_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    for path,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/path).read_bytes()).hexdigest() == digest,path
    assert "src/exogibbs/presets/curated_cos.py" in receipt["input_sha256"]


def test_independent_calorics_and_h298_control():
    inputs = json.loads(candidate.INPUTS[0].read_text())
    for t in (1500.,1637.,2500.,3000.,4000.):
        g,slope,curvature = candidate._rrho(t,np)
        expected = candidate.source.rrho(t,inputs)
        assert candidate.R*(g-t*slope) == pytest.approx(expected["H_minus_H0_J_mol"],abs=1e-8,rel=0)
        assert -candidate.R*slope == pytest.approx(expected["S_J_mol_K"],abs=1e-10,rel=0)
        assert -candidate.R*t*curvature == pytest.approx(expected["Cp_J_mol_K"],abs=1e-10,rel=0)
        a,b = (candidate.reaction_properties(t,v) for v in (candidate.PRIMARY,candidate.CONTROL))
        shift = -candidate.load_candidate()["new_minus_reference_heat_content_298_J_mol"]
        for key in ("G_J_mol","H_J_mol"):
            assert b[key]-a[key] == pytest.approx(shift,abs=1e-8,rel=0)
        for key in ("S_J_mol_K","Cp_J_mol_K"):
            assert b[key] == a[key]
        assert a["G_J_mol"] == pytest.approx(a["H_J_mol"]-t*a["S_J_mol_K"],abs=1e-8,rel=0)


def test_dense_scalar_jax_and_independent_source_reconstruction(report):
    import jax
    jax.config.update("jax_enable_x64",True)
    assert report["temperature_count"] == 2501
    assert max(report["maximum_absolute_reconstruction_error"].values()) < 2e-12
    ts = np.arange(1500.,4001.)
    for variant in candidate.VARIANTS:
        values = np.asarray(jax.jit(lambda t:candidate.formation_h_jax(t,variant))(ts))
        np.testing.assert_allclose(values,[candidate.formation_h(t,variant) for t in ts],atol=2e-12,rtol=0)
    for values in report["jax_float64_derivative_errors"].values():
        assert values["G_J_mol"] < 1e-8 and values["H_J_mol"] < 1e-8
        assert values["S_J_mol_K"] < 1e-10 and values["Cp_J_mol_K"] < 1e-10
    frozen = json.loads((candidate.DATA/"validation.json").read_text())
    for actual,expected in zip(report["samples"],frozen["samples"]):
        for variant in candidate.VARIANTS:
            assert actual["K_over_current"][variant] == pytest.approx(expected["K_over_current"][variant],abs=1e-12,rel=0)
    assert report["production_change"] == "none" and not report["energy_replacement"]


def test_selected_sensitivity_does_not_claim_full_completeness(report):
    row = report["thermal_sensitivity"][-1]
    assert row["bend_316_over_322_K"] > 1 > row["bend_328_over_322_K"]
    assert row["A_singlet_equal_rotation_K_multiplier"] == pytest.approx(1.0088193348108434,abs=1e-12,rel=0)
    assert row["legacy_triplet_equal_rovib_over_ground_weight"] == pytest.approx(.003371442538798663,abs=1e-12,rel=0)
    assert "not a physical uncertainty interval" in report["sensitivity_scope"]


def test_domain_guards():
    import jax
    jax.config.update("jax_enable_x64",True)
    for variant in candidate.VARIANTS:
        for t in (1499.,4001.,np.nan,np.inf):
            with pytest.raises(ValueError,match="domain"):
                candidate.formation_h(t,variant)
        values = np.asarray(jax.jit(lambda t:candidate.formation_h_jax(t,variant))(np.array([1499.,1500.,4000.,4001.,np.nan])))
        assert np.isnan(values[[0,3,4]]).all()
        np.testing.assert_allclose(values[1:3],[candidate.formation_h(t,variant) for t in (1500.,4000.)],atol=2e-12,rtol=0)
    with pytest.raises(ValueError,match="Unknown"):
        candidate.formation_h(2000.,"condensed")
