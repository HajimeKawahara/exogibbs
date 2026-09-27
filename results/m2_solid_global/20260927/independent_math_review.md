# Independent read-only review of the declared solid global bounds

Reviewed Gibbs commit `3ca51168a6a08d293491d61166fb5fa05d088ead` at
`/tmp/stage2-solid-bounds-20260927` and EOS commit
`38475d56a7cbf0b00aa1e51ef6d6e61e4c4291fc` at
`/tmp/stage2-solid-eos-20260927`. Neither checkout was modified. No native job,
full suite, or model sweep was run. Two small analytic examples were evaluated.

## Findings

1. **An accepted tolerance test is not strict nonnegativity.**
   `m2_solid_global.py:451` accepts a leaf when its verified lower bound is at
   least `-tolerance_rt`; line 471 then reports
   `formal_global_insertion_bound_accepted=True`. With a constant declared
   insertion energy of `-5e-9 RT` and the default tolerance `1e-8`, a one-node
   analytic case returns accepted with the correctly reported lower bound
   `-5.000000000000001e-9`. This is consistent with the explicit tolerance
   contract, not an erroneous lower bound. Strictly nonnegative claims must
   use the reported lower bound as well as the acceptance flag. It does not
   affect cases whose reported global lower bounds are strictly positive.

2. **The curvature anchor should be explicitly confined to its verified box.**
   Line 171 computes `(v.lo+v.hi)/2` in the ambient Decimal context, normally
   28 digits. For a singleton at binary64 `0.1`, the exact endpoint is
   `0.1000000000000000055511151231257827021181583404541015625`, while the computed
   center is `0.1000000000000000055511151232`, outside that singleton.
   The interval Hessian is verified over the original box, so the generic
   supporting-plane argument lacks an explicit anchor-in-box guarantee in
   this case. Clamping the computed midpoint to `[v.lo,v.hi]` would establish
   that precondition directly. No invalid numerical lower bound was found
   from this example, and fixed endpoints 0 and 1 do not exhibit this issue.
   This is a narrow generic proof-precondition gap, not demonstrated failure
   of the current 20-model result set.

Both findings were sent to the root and phase-stability agents before any
checkout changes. The second was proposed as a follow-up correction rather
than applied during the fixed run.

## Mathematical paths checked

- **LP dual bound (`m2_solid_global.py:126-150`).** With constraints
  `b+A*x>=0` and any nonnegative multipliers `lambda`, the affine objective is
  at least `c0-lambda*b+(c-A.T*lambda)*x`. The implementation clips multipliers
  to nonnegative values and interval-evaluates this Lagrangian on the complete
  box. It does not trust primal feasibility, the solver's objective value,
  or exact stationarity of the returned duals. Failure falls back to zero
  multipliers. Constraint rows entering this LP are checked for exact
  binary64 representability.
- **Positive entropy (`:185-196`).** For `f(q)=c*q*ln(q)`, `c>=0`, and
  `q in [0,U]`, `f''(q)>=c/U`. The selected curvature is rounded downward.
  The tangent plus that quadratic is a lower bound, with the zero endpoint
  justified by continuity. The entropy anchor is clipped to the occupation
  interval and replaced by a positive interior value if necessary.
- **Negative entropy (`:197-208`).** For `c<0`, the function is concave.
  The line joining downward-rounded endpoint values is below its true
  endpoint chord and hence below the function on the closed interval.
  Signed interval multiplication is retained.
- **Interval Hessian and LDL (`:209-235`, `m2_liquid_global.py:106-122`).**
  Eigenvalues and row bounds only propose a shift. Acceptance requires
  positive interval LDL pivots for the symmetric polynomial/entropy Hessian
  plus that shift. The added term
  `rho/2*sum((x_i-l_i)*(x_i-u_i))` is nonpositive on the box and shifts the
  Hessian by `rho*I`; it therefore gives a convex lower relaxation. The
  support value and gradient are reevaluated by intervals before the verified
  affine LP. The anchor-in-box qualification is the finding above.
- **Coverage and exclusions (`:101-123`, `:391-469`).** Linear tightening uses
  outward interval bounds with the correct sign for each coefficient.
  Nonlinear domain rows remain in interval exclusion checks. Bisection covers
  both closed children. A negative unresolved queue cannot produce an
  accepted result. Positive singular barriers use their correct lower limit.
- **Pure references (`:238-279`, `:355-386`).** The one-dimensional reference
  domain is covered by 256 closed subintervals; lower bounds cover all states,
  and point evaluations supply upper witnesses only. Reference intervals are
  subtracted before multiplying each signed endmember polynomial. Correlation
  loss can loosen the result but does not clip or omit negative fractions.
- **Support and units (`:287-386`; EOS `melts_solid_mixing.py`).** Host reactions
  use rational elimination on the supplied binary64 oxide basis. Missing or
  zero-support endpoint potentials raise instead of becoming zero. Fe/Ni
  singleton restrictions require exact absent-element budgets. Standard
  energies are J/mol divided by common `R*T`; source entropy uses the explicit
  native-R/common-R ratio. Source pressure terms use bar minus 1, with Pa
  converted by `1e5`. The H2 host correction is the solvent log fraction.

## Provider and scope checks

The committed EOS Cpx/Opx pure-reference arrays are identical, retaining the
source's `clino=TRUE` pure-standard convention. Rhm pure reference native
indices are correctly remapped from retained local indices `[0,1,2,3]` to
`[0,1,2,4]`. The plagioclase model is labeled as a pinned native-binary
instruction transcription, and the standard provider's binary hash is checked.
The runner preserves `native_binary_error_bound_certified=False` and
`global_empirical_stability_certified=False`; it does not promote finite native
agreement or the declared mathematical bound to a source-alloy or empirical
stability claim.

Apart from the two qualifications above, this review found no error in the
stronger-bound algebra or its treatment of signed references that would raise
a computed lower bound above the declared model's true minimum. This is an
independent code/mathematics review, not a replacement for the saved complete
proof partitions or a theorem about every possible provider input.
