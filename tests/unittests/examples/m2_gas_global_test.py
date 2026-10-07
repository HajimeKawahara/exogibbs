"""Gas certificates bind one potential, full species basis and exact atoms."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    G = importlib.import_module("m2_gas_global")
    C = importlib.import_module("m2_common_plane")
    A = importlib.import_module("m2_gas_eos")
    F = importlib.import_module("m2_finite_gas")
finally:
    sys.path.pop(0)


def parameters():
    return {"schema": G.SCHEMA, "species": ["H2", "He1", "H2O1", "trace"],
            "temperature_K": 2173.15, "pressure_Pa": 2.674e7,
            "gas_constant_J_mol_K": 8.31446261815324,
            "coefficients_m3_mol": [[14e-6, 10e-6, 15e-6, 0.],
                                     [10e-6, 8e-6, 15e-6, 0.],
                                     [15e-6, 15e-6, 8e-6, 0.], [0., 0., 0., 0.]]}


def independent_state(p, amounts):
    """130-digit direct Helmholtz/Legendre calculation independent of _I."""
    with localcontext() as context:
        context.prec = 130
        cast = lambda value: Decimal(value.numerator)/Decimal(value.denominator)
        n = list(map(Fraction, amounts))
        total = sum(n)
        x = [cast(value/total) for value in n]
        matrix = [[Decimal.from_float(value) for value in row]
                  for row in p["coefficients_m3_mol"]]
        bx = [sum(a*b for a, b in zip(row, x)) for row in matrix]
        b = sum(a*v for a, v in zip(x, bx))
        ideal = (Decimal.from_float(p["pressure_Pa"])
                 /(Decimal.from_float(p["gas_constant_J_mol_K"])
                   *Decimal.from_float(p["temperature_K"])))
        density = 2*ideal/(1+(1+4*ideal*b).sqrt())
        z = ideal/density
        residual = density*b+z-1-z.ln()
        chemical = [2*density*value-z.ln() for value in bx]
        return cast(total)*residual, chemical, x, z


def test_outward_square_root_and_exact_zero():
    value = G._I(Fraction(2)).sqrt()
    with localcontext() as context:
        context.prec = 130
        exact = Decimal(2).sqrt()
    assert value.lo <= exact <= value.hi
    assert G._I(0).sqrt().lo == G._I(0).sqrt().hi == 0
    with pytest.raises(ArithmeticError, match="nonnegative"):
        G._I(-1).sqrt()


@pytest.mark.parametrize("amounts", [
    [Fraction(7, 10), Fraction(1, 5), Fraction(9, 100), Fraction(1, 100)],
    [Fraction(11, 7), Fraction(13, 11), Fraction(0), Fraction(1, 10**150)],
    [Fraction(0), Fraction(0), Fraction(0), Fraction(1)],
])
def test_exact_primal_and_trace_fugacity_enclose_independent_helmholtz_state(amounts):
    p = parameters()
    energy, chemical, _, z = independent_state(p, amounts)
    result = G.gas_residual_energy(p, amounts)
    assert result.lo <= energy <= result.hi
    point = G._residual_state(p, [v/sum(amounts) for v in amounts])
    for value, reference in zip(point["log_fugacity_coefficients"], chemical):
        assert value.lo <= reference <= value.hi
    assert point["compressibility_factor"].lo <= z <= point["compressibility_factor"].hi
    if amounts[0]:
        assert point["log_fugacity_coefficients"][-1].hi < 0
    assert G.gas_residual_energy(p, [0]*4).lo == 0


def test_global_entropy_bound_is_tight_at_stationarity_and_below_every_trial():
    p = parameters()
    anchor = [Fraction(7, 10), Fraction(1, 5), Fraction(9, 100), Fraction(1, 100)]
    _, chemical, fractions, _ = independent_state(p, anchor)
    with localcontext() as context:
        context.prec = 130
        costs = [-x.ln()-mu for x, mu in zip(fractions, chemical)]
    lower, proof = G.gas_insertion_lower_bound(p, costs, anchor, [True]*4)
    assert proof["convexity"]["accepted"]
    assert lower.lo <= 0 <= lower.hi
    assert lower.hi-lower.lo < Decimal("1e-43")
    trials = [[Fraction(int(i == j)) for i in range(4)] for j in range(4)]
    trials += [[Fraction(i, 10), Fraction(10-i, 20), Fraction(10-i, 40), Fraction(10-i, 40)]
               for i in range(1, 10)]
    for trial in trials:
        value = C.ideal_energy(trial)+C._dot(trial, costs)+G.gas_residual_energy(p, trial)
        assert value.lo >= lower.lo


def test_zero_physical_amounts_and_unavailable_elements_remain_exact_zero():
    p = parameters()
    amounts = [Fraction(4), Fraction(1), Fraction(0), Fraction(0)]
    support = G.gas_element_support([[2, 0, 2, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]],
                                    ["H", "He", "O", "C"], ["H", "He", "O"], [8, 1, 2])
    assert support == [True, True, True, False]
    before = amounts[:]
    lower, proof = G.gas_insertion_lower_bound(p, [0]*4, amounts, support)
    assert amounts == before
    assert proof["reference_regularized_zero_indices"] == [2]
    assert Fraction(proof["exact_reference_fractions"][-1]) == 0
    assert proof["entire_nonnegative_supported_simplex"]
    assert not proof["reference_regularization_changes_primal"]
    assert lower.lo.is_finite()
    assert G.gas_residual_energy(p, amounts).lo > 0
    with pytest.raises(ValueError, match="unsupported"):
        G.gas_insertion_lower_bound(p, [0]*4, [4, 1, 0, 1], support)


@pytest.mark.parametrize("matrix", [[[.001]*4 for _ in range(4)], [[-.001]*4 for _ in range(4)]])
def test_global_nonconvex_or_mechanically_unstable_domain_is_rejected(matrix):
    p = parameters()
    p["coefficients_m3_mol"] = matrix
    assert not G.gas_convexity_certificate(p)["accepted"]
    with pytest.raises(ValueError, match="curvature"):
        G.gas_insertion_lower_bound(p, [0]*4, [1]*4, [True]*4)


def test_zero_pair_limit_retains_the_full_ideal_analytic_minimum():
    p = parameters()
    p["coefficients_m3_mol"] = [[0.]*4 for _ in range(4)]
    costs = [G._I(0), G._I(2), G._I(-1), G._I(3)]
    lower, proof = G.gas_insertion_lower_bound(p, costs, [7, 2, 1, 0], [True]*4)
    ideal = C.ideal_minimum(costs)
    assert max(lower.lo, ideal.lo) <= min(lower.hi, ideal.hi)
    assert lower.hi-lower.lo < Decimal("1e-43")
    assert Decimal(proof["convexity"]["entropy_curvature_lower_bound"]) == 1


@pytest.fixture
def saved_source():
    exoeos = pytest.importorskip("exoeos")
    root = Path(exoeos.__file__).resolve().parents[2]
    if not (root/"examples/m2_material/major_gas_eos.py").exists():
        pytest.skip("Requires the selected major-gas provider checkout.")
    species = list(F.build_atmosphere_setup("janaf_condensed").gas_species)
    options = {"h2_he_cm3_mol": 10., "water_cross_temperature_policy": "extrapolate",
               "trace_pair_policy": "zero"}
    p = A.build_gas_eos(root, species, options).parameters(2173.15, 267.4*1e5)
    source = {"gas_model": "janaf_condensed", "gas_eos_options": options,
              "temperature_K": 2173.15, "pressure_bar": 267.4,
              "source_metadata": {"gas_eos": deepcopy(p), "atmosphere": {"gas_model": "janaf_condensed"}},
              "source_internal_record": {"atmosphere_gas_model": "janaf_condensed"},
              "source_atmosphere_parcel": {"gas_eos": deepcopy(p), "gas_species": species,
                  "gas_eos_convexity": G.gas_convexity_certificate(p)}}
    return source, root


def test_receipt_replay_binds_the_actual_full_catalog_and_preserves_ideal(saved_source):
    source, root = saved_source
    result = G.require_saved_gas_eos(source, root)
    assert len(result["species"]) == 76
    assert result == source["source_metadata"]["gas_eos"]
    ideal = {"source_metadata": {}, "source_atmosphere_parcel": {}}
    assert G.require_saved_gas_eos(ideal, root) is None


@pytest.mark.parametrize("change", [
    lambda s: s.update(gas_model="janaf"),
    lambda s: s.update(temperature_K=2174.),
    lambda s: s.update(pressure_bar=268.),
    lambda s: s.update(gas_eos_options=None),
    lambda s: s["gas_eos_options"].update(h2_he_cm3_mol=20.),
    lambda s: s["source_metadata"]["gas_eos"]["coefficients_m3_mol"][0].__setitem__(0, .1),
    lambda s: s["source_atmosphere_parcel"]["gas_species"].reverse(),
    lambda s: s["source_atmosphere_parcel"]["gas_eos_convexity"].update(accepted=False),
])
def test_changed_source_or_receipt_cannot_inherit_gas_proof(saved_source, change):
    source, root = saved_source
    change(source)
    with pytest.raises(ValueError):
        G.require_saved_gas_eos(source, root)


def test_matching_forged_receipts_still_require_exact_provider_replay(saved_source):
    source, root = saved_source
    forged = source["source_metadata"]["gas_eos"]
    forged["potential_expression"]["residual_gibbs_over_nRT"] = "0"
    source["source_atmosphere_parcel"]["gas_eos"] = deepcopy(forged)
    with pytest.raises(ValueError, match="provider replay"):
        G.require_saved_gas_eos(source, root)
