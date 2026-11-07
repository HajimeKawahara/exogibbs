"""Verify the combined adoption against frozen, independently evaluated candidates."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated_ch2, curated_v1, curated_v2


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "src/exogibbs/data/curated/gas_v2.json"
R = 8.31446261815324


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models(float64):
    return curated_ch2.chemsetup(), curated_v2.chemsetup()


def _selected_references(data):
    specs = {}
    for species, references in data["candidate_references"].items():
        for reference in references:
            raw = (ROOT / reference["path"]).read_bytes()
            assert sha256(raw).hexdigest() == reference["sha256"]
            model = json.loads(raw)
            selected = next(v for v in model["variants"] if v["name"] == reference["variant"])
            spec = selected["replacements"][species]
            if species in specs:
                assert spec == specs[species]
            specs[species] = spec
    return specs


def _reference_properties(t, spec):
    """Evaluate analytic calorics with NumPy, without the runtime/review helpers."""
    if spec["kind"] == "custom":
        p = spec["parameters"]
        old = _reference_properties(t, p["retained"])
        f = p["evaluated"]
        u = (t - f["center_K"]) / f["scale_K"]
        u0 = (f["anchor_temperature_K"] - f["center_K"]) / f["scale_K"]
        poly = np.polynomial.polynomial.polyval
        h = f["anchor_H_over_R_K"] + f["scale_K"] * (
            poly(u, f["enthalpy_integral_u"]) - poly(u0, f["enthalpy_integral_u"]))
        s = (f["anchor_S_over_R"] + poly(u, f["entropy_integral_u"])
             - poly(u0, f["entropy_integral_u"])
             + f["entropy_log_coefficient"] * np.log(t / f["anchor_temperature_K"]))
        cp = poly(u, f["Cp_over_R_polynomial_u"])
        a, b, c, d, e, ff, g, hh, ss = p["nasa_molecular_thermal_coefficients"]
        nasa_h = -a/t + b*np.log(t) + c*t + d*t**2/2 + e*t**3/3 + ff*t**4/4 + g*t**5/5 + hh
        nasa_s = -a/(2*t**2) - b/t + c*np.log(t) + d*t + e*t**2/2 + ff*t**3/3 + g*t**4/4 + ss
        nasa_cp = a/t**2 + b/t + c + d*t + e*t**2 + ff*t**3 + g*t**4
        delta_h, delta_s = h - nasa_h, s - nasa_s
        return old + R * np.stack((delta_h - t*delta_s, delta_h, delta_s, cp - nasa_cp), axis=-1)
    a, b, c, d, e = spec["coefficients"]
    shift = spec["delta_GH_J_mol"]
    g = R * (-a - b*t*np.log(t) - c*t - d*t*t - e*t**3) + shift
    h = R * (-a + b*t + d*t*t + 2*e*t**3) + shift
    s = R * (b*(np.log(t) + 1) + c + 2*d*t + 3*e*t*t)
    cp = R * (b + 2*d*t + 6*e*t*t)
    return np.stack((g, h, s, cp), axis=-1)


def test_only_28_selected_species_change_and_joint_offsets_occur_once(models):
    before, after = models
    data = json.loads(DATA.read_text())
    shifts = data["energy_shifts_J_mol"]
    assert len(shifts) == len(data["decisions"]) == 28
    assert after.metadata["dataset"] == "curated-gas-v2"
    assert after.metadata["base_dataset"] == "curated-ch2-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    t = jnp.array([1500., 1687.5, 2000., 2687.5, 3500., 3937.5, 4000.])
    delta = np.asarray(after.hvector_func(t) - before.hvector_func(t))
    unaffected = [i for i, name in enumerate(before.species) if name not in shifts]
    np.testing.assert_array_equal(delta[:, unaffected], 0)
    for name, shift in shifts.items():
        if name != "O2":
            np.testing.assert_allclose(delta[:, before.species.index(name)], shift/(R*np.asarray(t)), atol=1e-13, rtol=0)
    for name in before.species:
        if name not in shifts:
            assert before.metadata["species_provenance"][name] == after.metadata["species_provenance"][name]
    assert shifts["C1N1O1"] - shifts["C1H1N1O1"] == -14724
    assert shifts["N2O4"] - 2*shifts["N1O2"] == -425
    assert shifts["H1N1O2cis"] - shifts["H1N1O2trans"] == -555
    assert shifts["N2O5"] == 3613 and shifts["P4S3"] == 56542
    assert shifts["O2"] == -107
    assert set(("C2N2", "C1N1Na1", "C1K1N1")).isdisjoint(shifts)


def test_dense_candidate_gibbs_and_caloric_derivatives_match_independent_formulas(models):
    model = models[1]
    data = json.loads(DATA.read_text())
    specs = _selected_references(data)
    names = list(data["energy_shifts_J_mol"])
    indices = jnp.array([model.species.index(name) for name in names])
    # All 2501 published-domain integer nodes plus off-grid points.
    temperatures = np.r_[np.arange(1500., 4001.), [1637.25, 2500.125, 3871.3]]
    expected = np.stack([_reference_properties(temperatures, specs[name]) for name in names], axis=1)

    def potential(t):
        return R*t*model.hvector_func(t)[indices]

    first = jax.jacfwd(potential)
    second = jax.jacfwd(first)
    evaluate = jax.jit(jax.vmap(lambda t: (potential(t), first(t), second(t))))
    g, slope, curvature = [np.asarray(a) for a in evaluate(jnp.asarray(temperatures))]
    actual = np.stack((g, g - temperatures[:, None]*slope, -slope,
                       -temperatures[:, None]*curvature), axis=-1)
    for column, tolerance in enumerate((2e-8, 2e-8, 2e-10, 2e-10)):
        np.testing.assert_allclose(actual[..., column], expected[..., column], atol=tolerance, rtol=0)
    # The 27 energy-only updates retain S/Cp independently of the O2 thermal change.
    for i, name in enumerate(names):
        if name != "O2":
            old = _reference_properties(temperatures, {**specs[name], "delta_GH_J_mol": 0})
            np.testing.assert_allclose(actual[:, i, :2] - old[:, :2], data["energy_shifts_J_mol"][name], atol=2e-8, rtol=0)
            np.testing.assert_allclose(actual[:, i, 2:], old[:, 2:], atol=2e-10, rtol=0)


def test_physical_anchors_and_o2_source_conventions_are_preserved():
    data = json.loads(DATA.read_text())
    specs = _selected_references(data)
    for species in ("Na2", "K2"):
        anchor = data["energy_anchors"][species]
        d0 = anchor["measurement"]["D0_cm_inverse"]*6.62607015e-34*299792458*100*6.02214076e23
        assert anchor["selected_D0_J_mol"] == pytest.approx(d0, abs=2e-11, rel=0)
        assert data["energy_shifts_J_mol"][species] == pytest.approx(anchor["base_D0_J_mol"]-d0, abs=2e-11, rel=0)
        assert anchor["formal_fit_intercept_Ra_J_mol"] != anchor["base_D0_J_mol"]
    for species in ("P4S3", "N2O5"):
        assert data["energy_anchors"][species]["base_D0_J_mol"] is None
        assert data["energy_anchors"][species]["reference_temperature_K"] == 298.15
    assert "NASA" in data["energy_anchors"]["C4N2"]["basis"]
    oxygen = specs["O2"]["parameters"]
    assert data["nasa_molecular_thermal_coefficients"] == oxygen["nasa_molecular_thermal_coefficients"]
    for key, value in data["thermal"].items():
        if key != "species":
            assert value == oxygen["evaluated"][key]
    with np.load(ROOT / "tests/data/validation/o2_candidate/20261106/published_tables.npz") as source:
        assert R*data["thermal"]["anchor_H_over_R_K"] == pytest.approx(source["H_relative_J_mol"][0], abs=1e-10)
        assert R*data["thermal"]["anchor_S_over_R"] == pytest.approx(source["S_J_mol_K"][0], abs=1e-12)
    assert "fit_temperature_K" not in data["thermal"]


def test_batch_domain_jit_and_neutral_projection(models):
    model = models[1]
    t = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(t)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    np.testing.assert_allclose(jax.vmap(model.hvector_func)(t[0]), actual[0], atol=1e-12, rtol=0)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated_v2.chemsetup(ions=False), curated_ch2.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(t), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_v2.chemsetup(ions=1)


def test_modified_packaged_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    changed = tmp_path / "gas_v2.json"
    changed.write_bytes(DATA.read_bytes() + b"\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == curated_v2._DATA_PATH else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated_v2.chemsetup()


def test_builder_reproduces_the_exact_pinned_payload_and_rejects_changed_input(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("build_gas_v2", ROOT / "mceg/reviews/build_gas_v2_adoption.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    encoded = (json.dumps(builder.build_data(), indent=2, allow_nan=False) + "\n").encode()
    assert encoded == DATA.read_bytes()
    assert sha256(encoded).hexdigest() == curated_v2._DATA_SHA256
    path = tmp_path / builder.BASE
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / builder.BASE).read_bytes() + b"\n")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="pinned candidate/base bytes"):
        builder.build_data()
