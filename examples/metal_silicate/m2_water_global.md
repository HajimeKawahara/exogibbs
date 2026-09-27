# Reconstructed-water common-plane bound

`run_m2_water_global.py` verifies the declared dry-MELTS plus reconstructed
water scalar against the elemental potentials of one accepted saved pressure
root. It does not reuse a hydrated-MELTS tangent-plane certificate. The
provider's saved `water_reconstruction.expression` and `dry_properties`
define the complete dry-component simplex. A component can be excluded only
by the declared unsupported provider domain or a zero total elemental
inventory; a zero parent amount alone never justifies exclusion.

At fixed dry amounts, write `N=sum(n_dry)`,
`S=sum(W_i n_i)/W_water`, and `log C=log_a+(w.n)/(T k.n)` using the
**actual binary64 coefficients supplied by ExoEOS**. Let `q_H2` be the
dissolved-H2 standard minus twice the saved hydrogen potential. Its source
scenario offset remains a separate constant. For `q_H2>0`, minimization over
all nonnegative dissolved H2 adds

```text
a*(N+h),  a = log(1-exp(-q_H2)).
```

Let `q_water` be the saved gas H2O standard, plus the separate `h2o_melts`
scenario shift, minus the common H/O elemental plane. For a dry composition
`r=n/N`, set `c(r)=q_water-2 log C(r)+a`. If `c(r)>0` on the whole dry
simplex, minimizing the water amount gives

```text
h/N = (S/N)/(exp(c/2)-1)
G_volatile_min/(RT N) = a + 2*(S/N)*log(1-exp(-c/2)).
```

This leaves a dry simplex with its original ideal/quadratic MELTS terms and
an explicit smooth rational water term. Outward Decimal Hessian intervals,
verified LDL curvature shifts and convex alpha-BB minorants bound every box.
Exact rational simplex anchors and exact box/simplex affine minimization
provide the lower bounds. Floating-point optimization supplies anchors only.
The complete frontier remains covered even when the node budget is reached;
meeting the requested lower-bound tolerance is a separate flag.

The provider's oxygen-capacity constraint `h <= oxygen.n_dry` is relaxed
only for the lower bound. A negative trial is an admissible counterexample
only when its eliminated water amount satisfies the original capacity.
The positive-cost preconditions are checked on the entire dry domain and
fail closed if unavailable. Bounds are per dry native-component mole;
the exact minimum atoms in that unit connects them to the finite-inventory
common-plane gap. Added water and H2 cannot reduce that atom count.

The source/audit binding checks the accepted final root, T/P, clean executed
provider pins, component formulas, amount scale, all source host amounts,
element budget and external plane. The gas water anchor must equal the
source's actual water-standard receipt. `source_hydrogen_standard_receipt`
replays the lower builder's common-H2/Hirschmann recipe, independently of
the retained-gas catalog, and records the separate H2 scenario term. Recipe,
thermochemical-data and imported package hashes are checked. Source and proof
files are checked again after the bound finishes; existing outputs are never
overwritten.

```bash
PYTHONPATH="$GIBBS_CHECKOUT/src:$EOS_CHECKOUT/src" \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/metal_silicate/run_m2_water_global.py \
  --saved-closure /path/to/accepted_closure.json \
  --saved-physical-audit /path/to/fresh_physical.json \
  --output /path/to/new_water_bound.json
```

The runner reports a strict nonnegative flag only when its actual outward
lower bound is nonnegative. The requested numerical tolerance and empirical
material acceptance are separate. An allowed negative bound can still enter
the finite-inventory common-plane correction with its sign unchanged.
New standard scenarios require new bounds. The result concerns the declared
constitutive expression and full supported domain, not empirical calibration
or a Gibbs minimum of the nonisothermal planet.

`run_m2_solid_global.py` can separately reevaluate the same reconstructed
host, including its saved water standard shift, while acquiring native
competing-phase standards from the native provider. The two providers and
the unchanged water gas anchor are preserved explicitly in the receipts.
