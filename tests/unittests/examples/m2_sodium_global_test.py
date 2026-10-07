"""An analytic Fe/Na image must enclose the full physical species domain."""
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/'examples/metal_silicate'
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module('m2_sodium_global')
finally:
    sys.path.pop(0)


def solve(*, sodium_cost=4., sodium_upper=.1, sodium_lower=0., max_nodes=2000):
    lo, hi = np.zeros(20), np.zeros(20)
    lo[0], hi[0], hi[1], hi[-1], lo[-1] = .7, 1., .2, sodium_upper, sodium_lower
    costs = np.zeros(20); costs[1], costs[-1] = 2., sodium_cost
    return M.certify_sodium_insertion(lambda x: 0*x[0], costs, lo, hi,
        1., np.zeros(18), max_nodes=max_nodes)


def test_full_ideal_simplex_minimum_and_separate_sodium_restoration():
    result = solve()
    with localcontext() as context:
        context.prec = 70
        expected = -(1+Decimal(-2).exp()+Decimal(-4).exp()).ln()
    assert Decimal(result['lower_bound_rt']) <= expected <= Decimal(result['upper_bound_rt'])
    assert result['minimum_certified']
    composition = list(map(Fraction, result['exact_composition']))
    assert sum(composition) == 1
    assert 0 < composition[-1] < Fraction(.1)
    assert abs(float(composition[-1]/composition[0])-np.exp(-4)) < 1e-15


def test_active_sodium_cap_retains_the_original_domain():
    result = solve(sodium_cost=-5.)
    assert result['minimum_certified']
    assert Fraction(result['exact_composition'][-1]) == Fraction(.1)
    assert result['composition'][0] >= .7
    assert result['analytic_elimination']['restored_Na_fraction'] == str(Fraction(.1))


def test_one_convex_minorant_can_certify_the_exact_constrained_split():
    result = solve(sodium_cost=-5., max_nodes=1)
    assert result['minimum_certified']
    assert result['nodes_evaluated'] == 1


def test_zero_sodium_face_removes_the_unavailable_split():
    result = solve(sodium_cost=-100., sodium_upper=0.)
    assert result['minimum_certified']
    assert result['composition'][-1] == 0.
    with localcontext() as context:
        context.prec = 70
        expected = -(1+Decimal(-2).exp()).ln()
    assert Decimal(result['lower_bound_rt']) <= expected <= Decimal(result['upper_bound_rt'])


def test_positive_lower_sodium_constraint_is_retained_by_upper_point():
    result = solve(sodium_cost=100., sodium_lower=.02)
    assert result['minimum_certified']
    assert Fraction(result['exact_composition'][-1]) == Fraction(.02)


def test_coupled_fe_lower_bound_is_preserved_without_shrinking_other_solutes():
    lo, hi = np.zeros(20), np.zeros(20)
    lo[0], hi[0], hi[1], hi[-1] = .75, 1., .2, .1
    costs = np.zeros(20); costs[1], costs[-1] = -4., -5.
    result = M.certify_sodium_insertion(lambda x: 0*x[0],costs,lo,hi,
                                        1.,np.zeros(18),max_nodes=1)
    assert not result['minimum_certified']
    x = list(map(Fraction,result['exact_composition']))
    assert x[0] >= Fraction(.75) and x[-1] <= Fraction(.1)
    assert result['analytic_elimination']['parent_domain_upper'][1] == .2
    # A single minorant at the nonsmooth Fe/Na cap intersection is not
    # silently promoted to a converged minimum certificate.
    assert Decimal(result['uncertainty_upper_rt']) > Decimal('1e-8')


def test_parent_image_does_not_shrink_the_original_species_box():
    lo, hi = np.zeros(20), np.zeros(20)
    lo[0], hi[0], hi[1], hi[-1] = .7, 1., .2, .1
    parent_lo, parent_hi = M.parent_domain(lo, hi)
    for silicon in (0., .1, .2):
        for sodium in (0., .05, .1):
            fe = 1-Fraction(silicon)-Fraction(sodium)
            combined = fe+Fraction(sodium)
            assert Fraction(float(parent_lo[0])) <= combined <= Fraction(float(parent_hi[0]))


@pytest.mark.parametrize('kind', ['short', 'nan', 'coupled_box'])
def test_malformed_or_uncovered_sodium_domains_fail_closed(kind):
    lo, hi = np.zeros(20), np.zeros(20)
    lo[0], hi[0], hi[1], hi[-1] = .7, 1., .2, .1
    if kind == 'short':lo, hi = lo[:-1], hi[:-1]
    elif kind == 'nan':hi[-1] = np.nan
    else:lo[0] = .9
    with pytest.raises(ValueError):M.parent_domain(lo, hi)


def test_sodium_factory_binds_actual_provider_energy_and_all_twenty_potentials():
    import hashlib
    from types import SimpleNamespace
    import jax
    exoeos = pytest.importorskip("exoeos")
    associated = importlib.import_module('m2_associated_global')
    root = Path(exoeos.__file__).resolve().parents[2]
    recipe = root/'examples/m2_material/sodium_metal.py'
    if not recipe.exists():pytest.skip('The pinned sodium provider is unavailable.')
    from exoeos import total_solution_state
    jax.config.update('jax_enable_x64', True)
    provider = associated._load_provider(root, True, True)
    model, interactions = provider.make_associated_model(2173.15)
    hi = np.array([1.,.02,.01,.04,.02,.002,.00002,.0001,.12,.00001,
                   .003,.00002,.0001,.005,.00001,.000001,.002,.000001,.02,.02])
    lo = np.r_[.75, np.zeros(19)]
    x = hi*.05; x[0] = 1-x[1:].sum()
    bare = total_solution_state(model,2173.15,2.7e7,x,np.zeros(20))
    standards = -np.asarray(bare.mu_RT)
    elements = list(dict.fromkeys(e for formula in provider.FORMULAS for e in formula))
    atoms = np.array([[formula.get(e,0.) for formula in provider.FORMULAS] for e in elements])
    metadata = {'metal_model':'associated_k_na', 'input':{'elements':elements},
        'associated_metal':{'model_id':model.reference_model_id,
            'interactions':interactions, 'component_order':list(provider.COMPONENTS),
            'component_formulas':list(provider.FORMULAS),
            'composition_basis':'chemical_species_moles',
            'standards':{'temperature_K':2173.15,'standard_potentials_rt':standards.tolist()},
            'lower_species_fractions':lo.tolist(), 'upper_species_fractions':hi.tolist(),
            'provider_recipe_file_sha256':{'examples/m2_material/sodium_metal.py':
                hashlib.sha256(recipe.read_bytes()).hexdigest()}}}
    def callback(t,p,n):
        state = total_solution_state(model,t,p*1e5,n,standards)
        return SimpleNamespace(gibbs_rt=float(state.gibbs_RT),mu_rt=np.asarray(state.mu_RT))
    fn = associated.make_associated_insertion_minimizer(metadata,callback,root)
    result = fn(2173.15,270.,atoms,np.zeros(len(elements)),lo,hi,maxiter=100)
    assert result.minimum_certified
    assert result.composition.shape == (20,)
    assert abs(result.upper_bound_rt) < 1e-10
    assert result.global_certificate['binding']['component_order'][-1] == 'Na'
    def wrong(t,p,n):
        state = callback(t,p,n)
        shift = np.zeros(20); shift[-1] = .001
        return SimpleNamespace(gibbs_rt=state.gibbs_rt+shift@n,mu_rt=state.mu_rt+shift)
    wrong_fn = associated.make_associated_insertion_minimizer(metadata,wrong,root)
    with pytest.raises(ValueError,match='insertion gradient'):
        wrong_fn(2173.15,270.,atoms,np.zeros(len(elements)),lo,hi,maxiter=100)
