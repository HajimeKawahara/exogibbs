# Complete-domain bounds for a second regular liquid

`m2_liquid_global.py` bounds the tangent-plane distance of the explicit
quadratic-plus-ideal mixing expression supplied by ExoEOS. This is stronger
than a local Hessian or a finite set of trial compositions: every feasible
box is either bounded or retained as unresolved.

For the parent fractions `q`, candidate fractions `x`, native water indices
`w`, and `alpha = R_native/R`, the dimensionless distance is

```text
D(x) = alpha sum_i [x_i log(x_i/q_i) - x_i + q_i]
       + 0.5 (x-q)' W (x-q)
       + alpha [x_w log(x_w/q_w)
                + (1-x_w) log((1-x_w)/(1-q_w))].
```

The standard-state terms cancel. Parent fractions in the verifier are exact
ratios of the supplied finite component amounts; the provided floating-point
coefficients define the mathematical expression being certified. The source
expression and its native compatibility checks are separate evidence.

## Bounds and boundaries

On a box `[l,u]`, add `rho/2 sum_i (x_i-l_i)(x_i-u_i)` to obtain a lower
relaxation. Its Hessian is bounded below by

```text
W + diag(alpha/u_i + rho) + 4 alpha e_w e_w'.
```

Interval LDL decomposition verifies positive definiteness of this matrix.
An optimizer supplies an interior anchor only. The tangent at that anchor
is a global affine lower bound for the relaxation, and a box-constrained
simplex linear program gives a lower bound for every composition in the box.
All energies, gradients, LDL operations and reported lower bounds use
50-digit outward Decimal intervals. Decimal logarithms are enclosed by
neighbors of the correctly rounded result. Simplex propagation and its
linear program use exact finite decimal sums at a larger precision.

The branch-and-bound construction follows the convex-underestimator principle
of [Adjiman et al. (1998)](https://doi.org/10.1016/S0098-1354(98)00027-1).
Here entropy derivatives are unbounded at zero, but their positive Hessian
lower bound and continuous `x log x` energy limit extend each lower affine
bound to the closed simplex. No finite derivative at zero is assumed.
The parent-containing box can close at zero when convexity is verified;
the parent supplies an exact stationary supporting plane there.
Numerical anchor failures and degenerate boxes remain unresolved, and a
node budget never counts as certification.

The search first restricts to positive parent components. The result counts
as the complete element-supported provider domain only when every excluded
component either requires an element absent from the parent or is explicitly
unsupported by this provider model. A zero amount alone is insufficient.
Signed-coordinate liquid extensions are not part of the supplied native
evaluator's declared nonnegative-component model.

## Dissolved H2 and other phases

For parent dissolved-H2 fraction `z0`, minimizing over the candidate H2
fraction gives `-log[z0+(1-z0) exp(-D)]`. Its sign is the sign of `D`, and
any nonpositive lower bound for `D` is also a conservative bound for this
augmented distance. A nonnegative supporting plane therefore excludes every
finite two-liquid split with the same ideal H2 law. The shared linear H2
standard cancels; the H2O component already remains in the native expression.

This result does not establish the stability of the 20 native solid solution
models, the Fe-Si-O-H source alloy, an empirical BSE applicability domain, or
a planetary pressure closure. The native Fe-Ni `alloy-liquid` candidate and
the selected Fe-Si-O-H source alloy are distinct property models.

## Reproduction

```bash
PYTHONPATH=src python examples/metal_silicate/run_m2_liquid_global.py \
  --saved-physical-audit physical_audit.json \
  --exoeos-checkout /path/to/exoeos \
  --runtime /path/to/pinned/alphamelts-runtime \
  --python /path/to/native-worker-python \
  --output-directory new-liquid-bound
```

The runner preserves the input bytes, reevaluates the host and near-vertex
native compositions, records compatibility independently, and saves the
complete box proof. It never solves pressure again. The callable API is
`assess_liquid_global_tangent_plane(properties, dissolved_h2_moles,
mixing_model=exoeos_example_module, tolerance_rt=1e-8, max_nodes=20000)`.
Only `formal_two_liquid_bound_accepted` combines the formal bound and complete
supported domain. `native_binary_error_bound_certified` and
`global_empirical_stability_certified` remain false.
