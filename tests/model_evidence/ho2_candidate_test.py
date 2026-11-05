"""Check HO2 source normalization, calorics, exact nodes and domain offline."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import ho2_candidate as candidate


@pytest.fixture(scope="module")
def report():
    import jax

    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        result = candidate.validate()
        assert not jax.config.x64_enabled
        return result
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_exact_reaction_changes_only_balanced_energy_with_native_R():
    model = candidate.load_candidate()
    assert candidate.build() == model
    assert model["D0_J_mol"] == 694662.
    assert model["nasa_D0_J_mol"] == pytest.approx(694682.762, abs=1e-9, rel=0)
    shift = model["ATcT_minus_nasa_D0_over_k_K"]
    assert shift == pytest.approx(-2.0209623169357656, abs=1e-10, rel=0)
    records = json.loads(candidate.INPUTS[0].read_text())
    coefficients = {s: np.array(next(row["coefficients"] for row in r["intervals"]
                    if row["bounds_K"] == [1000.,6000.])) for s,r in records.items()}
    original = coefficients["HO2"]-coefficients["H"]-2*coefficients["O"]
    difference = np.asarray(model["formation_nasa9_coefficients"])-original
    np.testing.assert_allclose(np.delete(difference,7),0,atol=2e-12,rtol=0)
    assert difference[7] == pytest.approx(-shift,abs=1e-10,rel=0)
    wrong_shift = (model["D0_J_mol"]-model["nasa_D0_J_mol"])/candidate.R
    assert abs(wrong_shift-shift) > .47


def test_dense_independent_calorics_and_derivatives(report):
    assert (report["temperature_count"],report["step_K"],report["temperature_range_K"]) == (2501,1,[1500,4000])
    assert report["maximum_absolute_formation_h_reconstruction_error"] < 1e-12
    errors = report["maximum_absolute_molecular_property_reconstruction_errors"]
    assert errors["G_relative_J_mol"] < 1e-8 and errors["H_relative_J_mol"] < 1e-8
    assert errors["S_J_mol_K"] < 1e-10 and errors["Cp_J_mol_K"] < 1e-10
    errors = report["jax_float64_maximum_absolute_errors"]
    assert errors["formation_h"] < 1e-12 and errors["reaction_H_J_mol"] < 1e-8
    assert errors["reaction_S_J_mol_K"] < 1e-10 and errors["reaction_Cp_J_mol_K"] < 1e-10
    archived = json.loads((candidate.DATA/"validation.json").read_text())
    for row,expected in zip(report["thermal_comparison"],archived["thermal_comparison"]):
        assert row == pytest.approx(expected,abs=1e-12,rel=0)
    assert report["production_change"] == "none"


def test_jax_off_grid_vector_vmap_and_ground_zero_reference():
    import jax

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        temperatures = np.array([1500.,1637.,2253.,3000.,3871.,4000.])
        f = jax.jit(candidate.formation_h_jax)
        expected = np.array([candidate.formation_h(t) for t in temperatures])
        np.testing.assert_allclose(f(temperatures),expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(temperatures),expected,atol=1e-12,rtol=0)
        assert float(f(2500)) == pytest.approx(candidate.formation_h(2500),abs=1e-12,rel=0)
        records = json.loads(candidate.INPUTS[0].read_text())
        for t in temperatures:
            atomic_g = 0
            for s,n in candidate.COMPOSITION.items():
                r = records[s]
                h,entropy,_ = candidate.source.nasa_properties(r,t,candidate.NASA_R)
                h0 = r["hf298_J_mol"]-r["h298_minus_h0_J_mol"]
                atomic_g += n*((h-h0)/t-entropy)/candidate.NASA_R
            molecular = candidate.molecular_properties(t)["G_relative_J_mol"]/(candidate.R*t)
            assert candidate.formation_h(t) == pytest.approx(molecular-atomic_g-candidate.load_candidate()["D0_over_k_K"]/t,abs=1e-12,rel=0)
    finally:
        jax.config.update("jax_enable_x64",previous)


def test_janaf_is_only_original_nodes_with_identical_energy_and_atoms(report):
    previous = json.loads((candidate.source.DATA/"comparison.json").read_text())
    for row in previous["modern_diagnostic"]["samples"]:
        t = row["temperature_K"]
        if t >= 1500:
            assert candidate.formation_h(t) == pytest.approx(-row["nasa_thermal_ATcT_ln_K"],abs=1e-12,rel=0)
            assert candidate.janaf_formation_h_node(t) == pytest.approx(-row["janaf_thermal_ATcT_same_nasa_atoms_ln_K"],abs=1e-12,rel=0)
    for row in report["thermal_comparison"]:
        t = row["temperature_K"]
        assert np.exp(candidate.formation_h(t)-candidate.janaf_formation_h_node(t)) == pytest.approx(row["janaf_same_energy_atoms_over_nasa_K"],rel=1e-12)
    with pytest.raises(ValueError,match="not an archived JANAF knot"):
        candidate.janaf_formation_h_node(1637.)
    assert report["thermal_comparison"][0]["janaf_same_energy_atoms_over_nasa_K"] == pytest.approx(1.0022756535461632)
    assert report["thermal_comparison"][-1]["janaf_same_energy_atoms_over_nasa_K"] == pytest.approx(.9799163021545025)


def test_scalar_and_traced_domain_guards():
    import jax

    invalid = [1499.,4001.,float("nan"),float("inf"),-float("inf")]
    for t in invalid:
        for function in (candidate.formation_h,candidate.molecular_properties,candidate.janaf_formation_h_node):
            with pytest.raises(ValueError,match="outside"):
                function(t)
    assert np.isnan(np.asarray(jax.jit(candidate.formation_h_jax)(np.asarray(invalid)))).all()
    assert np.isfinite(np.asarray(jax.jit(candidate.formation_h_jax)(np.array([1500.,4000.])))).all()


def test_receipt_pins_actual_sources_and_outputs():
    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed"
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/name).read_bytes()).hexdigest() == digest,name
