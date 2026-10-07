"""Analytic Fe/Na splitting for the declared pseudo-Fe excess model.

The provider maps Fe+Na to the parent's Fe coordinate only in the excess
energy. Physical Fe and Na atom columns and their linear costs stay distinct.
The constrained ideal split retains all original Fe and Na bounds, including
the coupled Fe-rich face. Its remaining one-dimensional correction is convex,
so the parent's proved curvature and global minimizer remain applicable.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np
from scipy.special import expit

from m2_associated_global import (certify_associated_insertion, _First, _scalar_gradient,
                                  _validate_domain)
from m2_common_plane import ideal_minimum, _exp
from m2_liquid_global import _I, _outward_float


def parent_domain(lower, upper):
    """Enclose the entire Fe+Na image by the parent's independent box."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (lo.shape != (20,) or hi.shape != (20,)
            or not all(np.all(np.isfinite(v)) for v in (lo, hi))
            or np.any(lo < 0) or np.any(lo > hi) or np.any(hi > 1)
            or lo[0] <= 0 or sum(map(Fraction, lo)) > 1 or sum(map(Fraction, hi)) < 1):
        raise ValueError("Require finite feasible twenty-species sodium bounds.")
    parent_lo, parent_hi = lo[:-1].copy(), hi[:-1].copy()
    # Fe+Na is exactly the dependent complement of the eighteen parent
    # solutes. Outward rounding must not remove any boundary point.
    f_min, f_max = 1-sum(map(Fraction, hi[1:-1])), 1-sum(map(Fraction, lo[1:-1]))
    if (f_min < Fraction(float(lo[0]))+Fraction(float(lo[-1]))
            or f_max > Fraction(float(hi[0]))+Fraction(float(hi[-1]))):
        raise ValueError("The parent rectangle needs a feasible Fe/Na split at every point.")
    parent_lo[0] = _outward_float(_I(f_min).lo, True)
    parent_hi[0] = _outward_float(_I(f_max).hi, False)
    _validate_domain(parent_lo, parent_hi)
    return parent_lo, parent_hi


def _split_penalty(f, fe_cost, na_cost, folded, ratio, lower, upper):
    """Enclose a convex constrained-split correction and its first derivative.

    Inside the free split the correction is zero. When one component is
    fixed at a bound c, its second derivative is 1/(f-c)-1/f >= 0.
    At a free-to-bound switch the derivative is continuous. At an intersection
    of two active bounds it jumps upwards, preserving convexity. Interval
    comparisons retain every potentially active subgradient branch.
    """
    interval = isinstance(f.value, _I)
    cast = _I if interval else float
    fv = f.value
    a, b, folded, ratio = map(cast, (fe_cost, na_cost, folded, ratio))
    target = fv*ratio
    lo_na, hi_na, lo_fe, hi_fe = map(cast, (lower[-1], upper[-1], lower[0], upper[0]))
    def possible(left, right):
        return left.lo <= right.hi if interval else left <= right
    branches = []
    # The unconstrained minimizer may lie inside both component bounds.
    if (possible(lo_na, target) and possible(target, hi_na)
            and possible(lo_fe, fv-target) and possible(fv-target, hi_fe)):
        branches.append((cast(0), cast(0)))
    def xlogx(value):
        zero = value.lo == value.hi == 0 if interval else value == 0
        return cast(0) if zero else value*value.log() if interval else value*np.log(value)
    # A fixed Na endpoint is active only when it is the effective endpoint
    # of the split interval. The analogous Fe endpoint is handled separately.
    for fixed_na, bound, active in (
            (True, lo_na, possible(target, lo_na) and possible(fv-hi_fe, lo_na)),
            (True, hi_na, possible(hi_na, target) and possible(hi_na, fv-lo_fe)),
            (False, hi_fe, possible(target, fv-hi_fe) and possible(lo_na, fv-hi_fe)),
            (False, lo_fe, possible(fv-lo_fe, target) and possible(fv-lo_fe, hi_na))):
        if not active:
            continue
        if not fixed_na and lower[-1] == upper[-1]:
            continue
        fe, na = (fv-bound, bound) if fixed_na else (bound, fv-bound)
        moving, cost = (fe, a) if fixed_na else (na, b)
        if (moving.lo <= 0 if interval else moving <= 0):
            raise ValueError("A split endpoint has no finite one-sided parent derivative.")
        value = xlogx(fe)+xlogx(na)+fe*a+na*b-xlogx(fv)-fv*folded
        derivative = (moving/fv).log()+cost-folded if interval else np.log(moving/fv)+cost-folded
        branches.append((value, derivative))
    if not branches:
        raise ValueError("No feasible branch encloses the constrained Fe/Na minimum.")
    if interval:
        value = _I(min(v.lo for v, _ in branches), max(v.hi for v, _ in branches))
        derivative = _I(min(d.lo for _, d in branches), max(d.hi for _, d in branches))
    else:
        value, derivative = branches[0]
    return _First(value, [derivative*g for g in f.gradient])


def certify_sodium_insertion(parent_excess, costs, lower, upper, curvature,
                              shifts, *, tolerance=1e-8, max_nodes=2000,
                              maxiter=500):
    """Bind parent global bounds to a feasible full-20 upper evaluation.

    At fixed f=x_Fe+x_Na, unconstrained ideal splitting changes the parent
    Fe linear cost to -log(exp(-a_Fe)+exp(-a_Na)). All other parent costs
    and the provider's parent curvature/convexification bounds are unchanged.
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    parent_lo, parent_hi = parent_domain(lo, hi)
    costs = tuple(_I(value) for value in costs)
    if (type(max_nodes) is not int or max_nodes < 1
            or len(costs) != 20 or not all(value.lo.is_finite() and value.hi.is_finite()
                                    for value in costs)):
        raise ValueError("Require twenty finite linear costs and a positive integer node budget.")
    folded = costs[0] if hi[-1] == 0 else ideal_minimum([costs[0], costs[-1]])
    parent_costs = [folded, *costs[1:-1]]
    ratio = _I(0) if hi[-1] == 0 else _exp(folded-costs[-1])
    def constrained_excess(xs):
        interval = isinstance(xs[0].value, _I)
        midpoint = lambda value: float((value.lo+value.hi)/2)
        values = (costs[0], costs[-1], folded, ratio)
        if not interval:values = tuple(map(midpoint, values))
        return parent_excess(xs)+_split_penalty(1-sum(xs), *values, lo, hi)
    result = certify_associated_insertion(constrained_excess, parent_costs,
        parent_lo, parent_hi, curvature, shifts, tolerance=tolerance,
        max_nodes=max_nodes, maxiter=maxiter)
    parent = [Fraction(value) for value in result['exact_composition']]
    combined = parent[0]
    sodium_lo = max(Fraction(float(lo[-1])), combined-Fraction(float(hi[0])))
    sodium_hi = min(Fraction(float(hi[-1])), combined-Fraction(float(lo[0])))
    if sodium_lo > sodium_hi:
        raise ValueError("The relaxed parent anchor has no feasible Fe/Na split.")
    a_fe, a_na = (float((cost.lo+cost.hi)/2) for cost in (costs[0], costs[-1]))
    sodium = Fraction(float(combined)*float(expit(a_fe-a_na)))
    # A strictly positive finite trial keeps the free Na chemical potential
    # finite. This floor never enters the lower-bound domain or its proof.
    if sodium_lo == 0 < sodium_hi:
        sodium = max(sodium, min(sodium_hi/100, Fraction(1, 10**100)))
    sodium = min(sodium_hi, max(sodium_lo, sodium))
    full = [combined-sodium, *parent[1:], sodium]
    value, _ = _scalar_gradient(lambda xs: parent_excess(xs[:-1]), costs,
                                 full[1:], intervals=True)
    lower_value = Decimal(result['lower_bound_rt'])
    gap = value-_I(lower_value)
    if gap.hi < 0:
        raise ArithmeticError("The Fe/Na relaxation has inconsistent lower and upper bounds.")
    result.update(upper_bound_rt=str(value.hi), uncertainty_upper_rt=str(gap.hi),
        minimum_certified=gap.hi <= Decimal.from_float(float(tolerance)),
        composition=[float(x) for x in full], exact_composition=[str(x) for x in full],
        domain_lower=lo.tolist(), domain_upper=hi.tolist(),
        analytic_elimination={'model':'sodium_pseudoiron_ideal_split_v1',
            'parent_domain_lower':parent_lo.tolist(), 'parent_domain_upper':parent_hi.tolist(),
            'folded_Fe_cost_interval_rt':[str(folded.lo), str(folded.hi)],
            'original_Na_fraction_bounds':[float(lo[-1]), float(hi[-1])],
            'restored_Na_fraction':str(sodium),
            'curvature_coordinate_basis':'Eighteen parent solutes after constrained Fe/Na minimization; not the full twenty-species Hessian.',
            'scope':'The constrained split retains every original Fe/Na bound. Its convex piecewise correction preserves the parent curvature. Physical atom columns remain separate.'})
    return result
