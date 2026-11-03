"""Independent source/thermodynamic checks for the bounded NIST CO2 candidate."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import co2_candidate as candidate
from co2_source_review import curated_ln_k


def test_fit_matches_independently_published_off_knot_data():
    validation = candidate.validate()
    expected = json.loads((candidate.DATA/"validation.json").read_text())
    assert validation["curated_change"] == expected["curated_change"] == "none"
    for model, result in validation["models"].items():
        assert result["off_knot_checks"] == 90
        errors = result["maximum_errors"]
        assert errors["ln_K"] < 2.5e-8 and errors["Cp_J_mol_K"] < .00021
        assert errors["H_J_mol"] < .0075 and errors["S_J_mol_K"] < 4.5e-6
        assert result["minimum_Cp_J_mol_K"] > 58
        for key, tolerance in (("ln_K", 1e-12), ("H_J_mol", 1e-8), ("S_J_mol_K", 1e-10), ("Cp_J_mol_K", 1e-10)):
            assert errors[key] == pytest.approx(expected["models"][model]["maximum_errors"][key], abs=tolerance, rel=0)


def test_source_spin_translation_and_average_are_explicit():
    source = json.loads((candidate.DATA/"nist_samples.json").read_text())
    record = candidate.load_candidate()
    assert record["D0_J_mol"] == pytest.approx((711.381+2*246.844+393.111)*1000)
    assert record["molecular_mass_u"] == 44.0095
    nodes = json.loads((candidate.DATA/"nodes.json").read_text())
    for index, t in enumerate(candidate.KNOTS):
        rows = {c: next(r for r in source[c] if r[0] == t) for c in source}
        mono = candidate.molecular_properties(t)
        assert -mono["G_relative_J_mol"]/(candidate.R*t)-candidate.translation(t) == pytest.approx(math.log(rows["626"][1]), abs=2e-14)
        # Independently average log partition functions, not arithmetic Q.
        mean_logq = sum(candidate.ISOTOPE_WEIGHTS[c]*(math.log(r[1])
            -math.log((2 if c[1]=="3" else 1)*(6 if c[0]=="7" else 1)*(6 if c[2]=="7" else 1))) for c,r in rows.items())
        average = candidate.molecular_properties(t, "isotope_average")
        assert average["G_relative_J_mol"] == pytest.approx(-candidate.R*t*(mean_logq+candidate.translation(t)), abs=2e-9, rel=0)
        assert average["H_relative_J_mol"] == pytest.approx(candidate.R*t*sum(candidate.ISOTOPE_WEIGHTS[c]*r[7] for c,r in rows.items()), abs=2e-9, rel=0)
        assert average["Cp_J_mol_K"] == pytest.approx(nodes["isotope_average"][index]["Cp_J_mol_K"], abs=2e-12)
        assert math.exp(-candidate.formation_h(t)-curated_ln_k(t)) > 1


def test_all_calorics_share_one_C2_gibbs_and_bounded_jax_model():
    jax.config.update("jax_enable_x64", True)
    record = candidate.load_candidate()
    for model, polynomial in record["models"].items():
        def g(t):
            return candidate.R*candidate._polynomial(t, polynomial, jnp)[0]
        for t in (1500., 1623., 2000., 2741., 3500., 3879., 4000.):
            props = candidate.molecular_properties(t, model)
            slope, curvature = float(jax.grad(g)(t)), float(jax.grad(jax.grad(g))(t))
            assert -slope == pytest.approx(props["S_J_mol_K"], abs=3e-12)
            assert -t*curvature == pytest.approx(props["Cp_J_mol_K"], abs=3e-12)
            assert float(g(t))-t*slope == pytest.approx(props["H_relative_J_mol"], abs=2e-9)
            assert float(jax.jit(candidate.formation_h_jax, static_argnames="model")(t, model=model)) == pytest.approx(candidate.formation_h(t, model), abs=3e-13)
        for i,t in enumerate(candidate.KNOTS[1:-1],1):
            c0,c1=np.array(polynomial["coefficients"])[i-1:i+1]
            for order in (0,1,2):
                left=np.polynomial.polynomial.polyval(1.,np.polynomial.polynomial.polyder(c0,order))/250**order
                right=np.polynomial.polynomial.polyval(0.,np.polynomial.polynomial.polyder(c1,order))/250**order
                assert left == pytest.approx(right, abs=3e-10, rel=0)
    for t in (1499.,4001.,float('nan')):
        with pytest.raises(ValueError,match='outside'):
            candidate.molecular_properties(t)
        assert np.isnan(float(candidate.formation_h_jax(t)))


def test_candidate_receipt_and_source_scope_are_pinned():
    receipt=json.loads((candidate.DATA/'receipt.json').read_text())
    for group in ('selected_input_sha256','output_sha256'):
        for name,digest in receipt[group].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    scope=json.loads((candidate.DATA/'scientific_scope.json').read_text())
    assert scope['isotope_mixing_entropy_difference_over_R']==pytest.approx(
        -scope['heteronuclear_oxygen_pair_probability']*math.log(2),abs=1e-15)
    assert scope['natural_reaction_constructed'] is False


def test_source_validation_detects_a_material_zero_offset(monkeypatch):
    from copy import deepcopy
    shifted = deepcopy(candidate.load_candidate())
    for record in shifted["models"].values():
        for coefficients in record["coefficients"]:
            coefficients[0] += .01  # +0.083 J/mol G and H, above source-fit residuals.
    monkeypatch.setattr(candidate, "load_candidate", lambda: shifted)
    for result in candidate.validate()["models"].values():
        assert result["maximum_errors"]["H_J_mol"] > .07
        assert result["maximum_errors"]["ln_K"] > 2e-6
