"""Global alloy bounds must cover unsampled minima and preserve the full box."""
from decimal import Decimal
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module("m2_associated_global")
finally:
    sys.path.pop(0)


def solve(**options):
    return M.certify_associated_insertion(lambda x: 100*x[0]*x[1],
        [0., -2., -2.], [.4, 0., 0.], [1., .3, .3], 3., [100., 100.],
        **options)


def test_nonconvex_symmetric_minima_are_globally_bounded():
    result = solve(max_nodes=501, maxiter=100)
    assert result["minimum_certified"]
    assert result["nodes_evaluated"] > 1
    assert Decimal(result["uncertainty_upper_rt"]) <= Decimal("1e-8")
    # Both separated basins and the unstable central state remain covered.
    for y in ([.3, 2.058054413e-11], [2.058054413e-11, .3], [.15, .15]):
        value, _ = M._scalar_gradient(lambda x: 100*x[0]*x[1],
                                     list(map(M._I, [0., -2., -2.])),
                                     list(map(Fraction.from_float, y)), intervals=True)
        assert Decimal(result["lower_bound_rt"]) <= value.lo
    assert not result["empirical_material_certified"]


def test_one_box_does_not_turn_a_finite_search_into_a_certificate():
    result = solve(max_nodes=1)
    assert not result["minimum_certified"]
    assert result["nodes_evaluated"] == 1
    assert result["frontier"][0]["lower"] == [0., 0.]
    assert result["frontier"][0]["upper"] == [.3, .3]


def test_interval_gradient_and_exact_dependent_coordinate():
    costs = list(map(M._I, [0., -2., -2.]))
    y = [Fraction(1, 10), Fraction(1, 20)]
    value, gradient = M._scalar_gradient(lambda x: 100*x[0]*x[1], costs, y,
                                         intervals=True)
    expected = 100*.1*.05-.3+.85*np.log(.85)+.1*np.log(.1)+.05*np.log(.05)
    assert abs(float(value.lo)-expected) < 1e-14
    reference = [np.log(.1/.85)-2+5, np.log(.05/.85)-2+10]
    assert np.max(np.abs(np.array([float(g.lo) for g in gradient])-reference)) < 1e-14


@pytest.mark.parametrize("mutation", [
    {"costs": [0., np.nan, 0.]}, {"costs": [0., 1.]},
    {"curvature": 0.}, {"curvature": np.inf},
    {"shifts": [-1., 100.]}, {"shifts": [100.]},
    {"lower": [.6, 0., 0.]}, {"upper": [1., np.inf, .3]},
    {"tolerance": np.nan}, {"tolerance": 0.}, {"max_nodes": 0},
])
def test_invalid_or_uncovered_inputs_fail_closed(mutation):
    args = dict(excess=lambda x: 100*x[0]*x[1], costs=[0., -2., -2.],
                lower=[.4, 0., 0.], upper=[1., .3, .3], curvature=3.,
                shifts=[100., 100.], max_nodes=1)
    args.update(mutation)
    with pytest.raises(ValueError):
        M.certify_associated_insertion(**args)


def test_box_quadratic_lower_uses_gradient_error_on_both_sides():
    result = M._quadratic_lower(M._I(1), [M._I(-2., Decimal(3))],
                                [Fraction(0)], [-1.], [1.], 2.)
    # For d<0 the larger slope is the smaller product; its constrained
    # minimum is -2. For d>=0 the minimum is -1. Add the anchor value 1.
    assert result <= -1
    assert result > Decimal('-1.00000000000000000001')


def test_zero_inventory_face_keeps_its_exact_continuous_boundary():
    result = M.certify_associated_insertion(lambda x: 100*x[0]*x[1],
        [0., -2., -2.], [.4, 0., 0.], [1., .3, 0.], 3., [100., 100.], max_nodes=501)
    assert result['minimum_certified']
    assert result['composition'][-1] == 0.
    assert all(row['upper'][-1] == 0. for row in result['frontier'])


def test_singleton_domain_needs_no_numeric_search():
    result = M.certify_associated_insertion(lambda x: 0*x[0],
        [0., 0.], [1., 0.], [1., 0.], 1., [0.], max_nodes=1)
    assert result['minimum_certified']
    assert result['composition'] == [1., 0.]
    assert Decimal('-1e-45') < Decimal(result['lower_bound_rt']) <= 0


@pytest.fixture
def factory_fixture(monkeypatch, tmp_path):
    from types import SimpleNamespace as S
    import hashlib
    names = ['Fe','Si','O','H','P','Mg','Ca','Al','Cr','Ti','MgO','CaO','AlO','CrO','TiO','Al2O','Cr2O','Ti2O']
    matrix = np.zeros((18,18)); matrix[2,3] = matrix[3,2] = 100.
    interactions = {'additional_matrix': matrix.tolist(),
                    'temperature_policy_for_extra_P_H_cross_terms': 'constant'}
    host = S(host=S(dry_model=S(interaction_K=np.zeros(3))), epsilon=np.zeros(4))
    model = S(reference_model_id='fixture', host=host, additional_matrix=matrix)
    provider = S(COMPONENTS=names, FORMULAS=[{n:1.} for n in names],
                 make_associated_model=lambda *a, **k:(model, interactions),
                 associated_curvature_lower_bound=lambda *a: 1.,
                 associated_excess=lambda t,d,e,m,x:100*x[1]*x[2])
    monkeypatch.setattr(M,'_load_provider',lambda *a:provider)
    path = tmp_path/'recipe.py'; path.write_text('# fixed test expression\n')
    standards = np.zeros(18); standards[[2,3]] = -2.
    lo=np.zeros(18);lo[0]=.4
    hi=np.zeros(18);hi[0]=1.;hi[[2,3]]=.3
    metadata={'metal_model':'associated','input':{'elements':names},
              'associated_metal':{'model_id':'fixture','interactions':interactions,
                'component_order':names,'component_formulas':provider.FORMULAS,
                'composition_basis':'chemical_species_moles',
                'standards':{'temperature_K':2000.,'standard_potentials_rt':standards.tolist()},
                'lower_species_fractions':lo.tolist(),'upper_species_fractions':hi.tolist(),
                'provider_recipe_file_sha256':{'recipe.py':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    def callback(t,p,x):
        assert (t,p)==(2000.,1.)
        x=np.asarray(x);positive=x>0
        return S(gibbs_rt=float(standards@x+np.dot(x[positive],np.log(x[positive]))+100*x[2]*x[3]))
    return metadata,callback,tmp_path,lo,hi


def test_factory_returns_bound_provenance_and_exact_element_order(factory_fixture):
    metadata,callback,path,lo,hi=factory_fixture
    fn=M.make_associated_insertion_minimizer(metadata,callback,path)
    result=fn(2000.,1.,np.eye(18),np.zeros(18),lo,hi,maxiter=100)
    assert result.minimum_certified
    assert result.global_certificate['binding']['element_order']==metadata['input']['elements']
    assert result.global_certificate['binding']['formula_matrix']==np.eye(18).tolist()
    changed=np.eye(18);changed[[0,1]]=changed[[1,0]]
    with pytest.raises(ValueError,match='formula'):
        fn(2000.,1.,changed,np.zeros(18),lo,hi)


def test_factory_rejects_changed_scalar_or_provider(factory_fixture):
    from types import SimpleNamespace
    metadata,callback,path,lo,hi=factory_fixture
    bad=lambda t,p,x:SimpleNamespace(gibbs_rt=callback(t,p,x).gibbs_rt+.01)
    fn=M.make_associated_insertion_minimizer(metadata,bad,path)
    with pytest.raises(ValueError,match='source scalar'):
        fn(2000.,1.,np.eye(18),np.zeros(18),lo,hi,maxiter=100)
    (path/'recipe.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='recipe'):
        M.make_associated_insertion_minimizer(metadata,callback,path)
