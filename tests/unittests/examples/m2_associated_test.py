"""Finite multi-atom alloy components retain source budgets and derivatives."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    SOURCE = importlib.import_module("m2_expanded_source")
    LOCAL = importlib.import_module("local")
    COMMON = importlib.import_module("common_gibbs")
    SELECTION = importlib.import_module("phase_selection")
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def eos_checkout():
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    if not (checkout / "examples/m2_material/associate_reference.py").is_file():
        pytest.skip("Requires the associated-metal EOS provider.")
    return checkout


def build(checkout, tmp_path, **kwargs):
    return SOURCE.build_expanded_bse_problem(
        checkout / "examples/m2_material/bse_inventory.json", checkout, tmp_path,
        sys.executable, gas_model="janaf_condensed", metal_model=kwargs.pop("metal_model", "associated"),
        initialization="canonical", **kwargs)


def test_eighteen_species_are_finite_atom_balanced_and_scalar_audited(eos_checkout, tmp_path):
    record, budget, callbacks, initial, metadata = build(eos_checkout, tmp_path)
    names = [name for group in record["phases"].values() for name in group]
    formula = np.array([[record["component_formulas"][name].get(e, 0.) for name in names]
                        for e in record["elements"]])
    np.testing.assert_allclose(formula@initial, budget, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(formula@metadata["numerical_initialization"]["initial_component_amounts_mol"],
                               budget, rtol=1e-12, atol=0.)
    assert len(record["phases"]["metal"]) == 18
    assert record["component_formulas"]["Cr2O_metal"] == {"Cr": 2., "O": 1.}
    lower, upper, curvature = SOURCE.associated_metal_domain(metadata)
    assert lower.shape == upper.shape == (18,)
    assert 1.45 < curvature < 1.46
    n = upper*.1
    n[0] = 1-n[1:].sum()
    state = callbacks["metal"](2173.15, 1., n)
    energy, derivative = callbacks["metal"].energy_value_and_grad_rt(2173.15, 1., n)
    assert state.gibbs_rt == pytest.approx(energy, abs=1e-12)
    np.testing.assert_allclose(derivative, state.mu_rt, atol=1e-12)
    assert float(n@state.mu_rt) == pytest.approx(energy, abs=1e-12)
    assert metadata["associated_metal"]["composition_basis"] == "chemical_species_moles"
    assert "K alloy component" in metadata["atmosphere"]["missing_paths"]
    assert metadata["numerical_execution"]["native_melt_calls"] == 0


def test_named_host_standard_offsets_do_not_recalibrate_independent_associates(eos_checkout, tmp_path):
    base = build(eos_checkout, tmp_path)
    shifted = build(eos_checkout, tmp_path,
                    scenario={"standard_offsets_rt": {"O_metal": 1.6, "H_metal": -.4}})
    _, upper, _ = SOURCE.associated_metal_domain(base[-1])
    n = upper*.1
    n[0] = 1-n[1:].sum()
    a = base[2]["metal"](2173.15, 1., n)
    b = shifted[2]["metal"](2173.15, 1., n)
    offsets = np.zeros(18)
    offsets[[2, 3]] = [1.6, -.4]
    np.testing.assert_allclose(b.mu_rt-a.mu_rt, offsets, atol=1e-13)
    assert b.gibbs_rt-a.gibbs_rt == pytest.approx(offsets@n, abs=1e-13)
    assert shifted[-1]["associated_metal"] == base[-1]["associated_metal"]


def test_closed_fe_mg_o_parcel_solves_association_with_the_original_finite_atoms(eos_checkout, tmp_path):
    record, _, callbacks, _, _ = build(eos_checkout, tmp_path)
    record = copy.deepcopy(record)
    record["phases"] = {"metal": record["phases"]["metal"]}
    budget = np.zeros(len(record["elements"]))
    for element, value in {"Fe": .98, "Mg": .0001, "O": .001}.items():
        budget[record["elements"].index(element)] = value
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(18), phases=("metal",))
    restricted = SELECTION.restrict_phase_callbacks(record, problem, {"metal": callbacks["metal"]})
    result = COMMON.minimize_gibbs(problem, 2173.15, 1., budget, restricted, maxiter=200)
    assert result.accepted, result.audit_reasons
    values = dict(zip(problem.full_species, result.component_amounts_mol))
    assert 0 < values["MgO_metal"] < .0001
    assert values["MgO_metal"]+values["Mg_metal"] == pytest.approx(.0001, abs=1e-14)
    assert values["MgO_metal"]+values["O_metal"] == pytest.approx(.001, abs=1e-14)
    assert max(abs(result.element_residual)) < 1e-12


@pytest.mark.parametrize("potassium", [False, True])
def test_hydrogen_oxygen_option_uses_the_actual_scalar_and_preserves_negative_curvature(
        eos_checkout, tmp_path, potassium):
    options = ({"metal_model": "associated_k", "potassium_standard_offset_rt": 0.}
               if potassium else {})
    base = build(eos_checkout, tmp_path, **options)
    changed = build(eos_checkout, tmp_path,
                    hydrogen_oxygen_model="schenck1961_abstract", **options)
    _, upper, curvature = SOURCE.associated_metal_domain(changed[-1])
    assert curvature < 0
    n = upper*.1
    n[0] = 1-n[1:].sum()
    a = base[2]["metal"](2173.15, 1., n)
    b = changed[2]["metal"](2173.15, 1., n)
    denominator = 1-n[-1] if potassium else 1.
    difference = 52.4*np.log(10.)*n[2]*n[3]/denominator
    assert b.gibbs_rt-a.gibbs_rt == pytest.approx(difference, abs=1e-13)
    energy, gradient = changed[2]["metal"].energy_value_and_grad_rt(2173.15, 1., n)
    assert energy == pytest.approx(b.gibbs_rt, abs=1e-13)
    np.testing.assert_allclose(gradient, b.mu_rt, atol=1e-12)
    assert gradient@n == pytest.approx(energy, abs=1e-12)
    receipt = changed[-1]["associated_metal"]
    assert receipt["interactions"]["hydrogen_oxygen"]["model"] == "schenck1961_abstract"
    assert "examples/m2_material/hydrogen_oxygen_sources.json" in receipt["provider_recipe_file_sha256"]
    assert changed[0] == base[0]
    np.testing.assert_array_equal(changed[1], base[1])
    np.testing.assert_array_equal(changed[3], base[3])


def test_hydrogen_oxygen_selection_rejects_unknown_or_unsupported_models(eos_checkout, tmp_path):
    for options in ({"hydrogen_oxygen_model": "unknown"},
                    {"hydrogen_oxygen_model": "schenck1961_abstract", "metal_model": "ma"},
                    {"hydrogen_oxygen_model": "schenck1961_abstract", "metal_model": "phosphorus"}):
        with pytest.raises(ValueError, match="H-O|hydrogen_oxygen_model"):
            build(eos_checkout, tmp_path, **options)


@pytest.mark.parametrize("upper", [.01, .02, .03])
@pytest.mark.parametrize("kind", ["associated", "associated_k"])
def test_associated_p_bound_preserves_scalar_and_species_basis(eos_checkout, tmp_path, upper, kind):
    options = {"metal_model": kind}
    if kind == "associated_k":
        options["potassium_standard_offset_rt"] = 0.
    base = build(eos_checkout, tmp_path, **options)
    changed = build(eos_checkout, tmp_path, phosphorus_options={"upper_mole_fraction": upper}, **options)
    row = changed[-1]["associated_metal"]
    assert row["composition_basis"] == "chemical_species_moles"
    assert row["upper_species_fractions"][4] == upper
    assert changed[-1]["phosphorus_metal"]["upper_atomic_fractions"][4] == upper
    assert row["standards"] == base[-1]["associated_metal"]["standards"]
    assert row["interactions"] == base[-1]["associated_metal"]["interactions"]
    n = np.asarray(base[-1]["associated_metal"]["upper_species_fractions"])*.1
    n[0] = 1-n[1:].sum()
    a, b = (item[2]["metal"](2173.15, 1., n) for item in (base, changed))
    assert a.gibbs_rt == b.gibbs_rt
    np.testing.assert_array_equal(a.mu_rt, b.mu_rt)
    if upper == .02:
        assert row == base[-1]["associated_metal"]
