# Saved-alloy insertion bound

`run_m2_alloy_insertion_bound.py` supplements a saved finite-inventory pressure
root with an outward interval bound for its Fe–Si–O–H alloy against the saved
elemental potential. It does not rerun equilibrium, call the native MELTS worker,
change the selected alloy, or overwrite phase-selection and material flags.

The accompanying ExoEOS `ma_alloy_insertion_lower_bound` evaluates the exact
declared scalar and its reduced gradient at a positive solute point. Fe is
defined by the real expression `1 - Si - O - H`, and interval arithmetic must
establish that the point is inside the saved hard composition domain. For the
existing positive global curvature bound `kappa`, strong convexity gives

```text
min q >= q(a) - ||gradient q(a)||² / (2 kappa),
q(x) = g(x)/(RT) - sum_i x_i lambda_i.
```

Only free reduced coordinates enter the gradient norm. The domain includes
zero-solute faces by continuity, although the evaluation point must be positive.
The returned bound has units of RT per mole of atomic components. Negative
bounds are retained and do not receive a strict nonnegative certificate through
a tolerance or rounding to zero. This supplements the alloy's existing exact
self-tangent convexity proof; it does not prove every phase against one common
plane or certify the empirical composition domain.
A negative lower bound alone also does not prove a negative true minimum.

The runner requires the exact final closure bytes named by the physical audit,
accepted numerical closure/contact, matching T/P/layers, the primitive alloy
composition, and the same scenario values and input hash in both records.
Atomic unit formula columns select the four saved potentials directly, avoiding
a rounded matrix multiplication. All recorded Gibbs recipe files must match;
the source ExoEOS commit's Ma model and array helpers must match the proof
checkout. The recorded JAX/NumPy versions are also required.

The shared input binding also supports the associated-alloy common-plane path.
Its 18-, 19-, and 20-species composition domains use chemical-species mole
fractions, including multi-atom oxide associates. The saved domain and provider
metadata must agree on that basis, every component's atom count, and the exact
species-fraction limits. Atomic-fraction aliases are rejected; the atomic
four-component and phosphorus-alloy contracts remain unchanged.

All four base standards are reconstructed with that pinned source recipe.
Fe/Si standards and the formal shifts are checked against the saved independent
extraction. The reconstructed binary64 values define exact constants for this
proof; it does not interval-certify the original Shomate evaluation or measured
standard uncertainty. O/H scenario terms remain separate constants and are
added inside interval arithmetic. No standard is inferred from a saved chemical
potential by subtracting a mixing term.

Run with the matching ExoEOS proof branch. Its Git object database must contain
the source ExoEOS commit recorded in the closure; fetch that specific published
commit first if needed. Supply the original physical audit and a fresh output:

```bash
env JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  PYTHONPATH=src:/path/to/exoeos/src \
  python examples/metal_silicate/run_m2_alloy_insertion_bound.py \
  --saved-closure /path/to/closure.json \
  --saved-physical-audit /path/to/physical.json \
  --exoeos-checkout /path/to/exoeos \
  --output /path/to/new_alloy_bound.json
```

The output retains the source/audit hashes, selected root, source and proof
revisions, complete reconstructed constants, scenario terms, domain, curvature,
lower bound and all relevant file hashes. Keep it beside the final-root evidence
and preserve the original generic physical-audit flags.

The documented-example harness registers `metal_m2_alloy_insertion_bound`.
Pass its explicit `--m2-closure` and `--m2-physical-audit` inputs when selecting
this case. It requires fresh proof output tied to those exact bytes, while
allowing a finite negative bound without claiming strict nonnegativity.

The [finite-source common-plane gap](m2_common_plane.md) combines this unchanged
bound with all other declared phase domains and exact atom feasibility. Its
tolerance-based acceptance leaves this original fixed-plane strict flag intact.
