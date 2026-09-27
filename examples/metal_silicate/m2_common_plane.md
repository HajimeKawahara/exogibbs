# Finite-source common-plane Gibbs gap

`run_m2_common_plane.py` connects the saved global phase bounds to **one
finite, exactly conserved elemental inventory at the saved source T/P**.
It does not change the equilibrium, physical models, native property records,
or the original strict-sign flags. In particular, a tiny negative value of
the original fixed elemental plane remains negative.

For phase `p`, let `tau_p` be a verified global insertion lower bound per
phase unit, and let `a_p > 0` bound the atoms in that unit from below. Set

```text
delta = max_p max(0, -tau_p/a_p)
lambda_corrected = lambda_saved - delta * ones(number_of_elements)
L = lambda_corrected . b
```

Only negative phase bounds need a positive atom-count lower bound. Every
admissible phase then satisfies `G_p/(RT) >= lambda_corrected . atoms_p`,
including arbitrary amounts and splits. Summing over phases and using exact
atom conservation proves `G_min/(RT) >= L`. The correction is evaluated
outwards; it is an explicit change to the comparison plane, not a change to
any Gibbs function or standard state.

The primal construction replaces the atmosphere's atom-carrier coordinates
with its saved primitive gas/cloud species. Rational elimination repairs a
well-populated, independent set of columns to match the original thirteen
binary64 elemental budgets **exactly as rational numbers**. Negative
repaired amounts, an unsupported element, or departure from the alloy box
cause failure. Interval evaluation of this feasible state gives an upper
bound `U`. Therefore `0 <= G_primal/(RT) - G_min/(RT) <= U-L`.

The new model acceptance requires `(U-L)/sum(b) <= 1e-9`, the M2 energy
comparison tolerance. Source atom and chemistry residuals retain their
`1e-9` and `1e-8` gates. The selected positive alloy's global-minimum interval,
composition-search uncertainty, and complementarity normalized by total
inventory atoms must also be within `1e-8`. None of these tolerance checks
promotes the original plane to a strictly nonnegative certificate.

The phase catalog is covered as follows:

| Phase family | Global lower bound against the common plane |
| --- | --- |
| Declared liquid, with arbitrary dissolved H2 | Transfer the saved full-simplex self-tangent bound to the common elemental plane using an outward affine correction. Minimize the ideal host/H2 dilution analytically with log-sum-exp. |
| Twenty competing solution models, including Fe–Ni alloys | Translate the saved full-domain bounds with exact signed host reaction coefficients and interval bounds over their complete site boxes. Reconstruct every declared expression/domain from the exact pinned ExoEOS git blobs; require the same native standard-state receipt. |
| Thirteen remaining native pure candidates | Evaluate the fixed recorded oxide-mass/Gibbs unit with rational stoichiometry. Its domain is that composition and every nonnegative amount of it; no other composition or native build is certified. |
| Source Fe–Si–O–H alloy | Replay the existing outward curvature/insertion calculation on the same full declared box, standards, separate offsets and original saved elemental plane. |
| Ideal gas | Use the analytic full-simplex minimum `-log(sum(exp(-cost_i)))`. |
| Eligible pure clouds | Evaluate every scalar insertion cost on the retained temperature-valid catalog. Ineligible catalog entries are listed explicitly. |

The source gas standards are reconstructed as the original raw standard plus
the elemental gauge, rather than treating their rounded sum as a different
exact standard. The source log-pressure value and dissolved-H2 standard are
reconstructed from the pinned recipe. Liquid intervals retain the declared
`mu0_J/(R*T)` and also enclose the provider's binary64 conversion to `mu0_RT`;
the latter's rounding is bounded rather than silently changing the former.
An independent repaired-primal energy check against the saved total source
energy must meet the same `1e-9 RT` per inventory atom tolerance.

The runner requires an accepted final pressure root bound to a fresh physical
audit and the original phase-proof inputs. Exact mathematical input equality
permits previously recorded proof reuse. Runtime receipt hashes may differ
between native calls, but all liquid numerical fields, basis, expression,
standards and dissolved H2 must agree; both records remain hashed. The alloy
interval bound is independently reproduced. Every read input is checked
again for changes before the output is written.

```bash
PYTHONPATH="$GIBBS_CHECKOUT/src:$EOS_CHECKOUT/src" \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/metal_silicate/run_m2_common_plane.py \
  --evidence-binding /path/to/OH_evidence_binding.json \
  --alloy-bound /path/to/OH_alloy_insertion_bound.json \
  --exoeos-checkout "$EOS_CHECKOUT" \
  --output /path/to/new_common_plane.json
```

Alternatively, provide the verified five-root aggregate as
`--evidence-binding` and choose one case with `--case M1` (or `GAS`, `CLOUD`,
`OXYGEN`, `OH`). The ExoEOS repository must contain the recorded solution-model
commit as a git object; its checkout is not changed. Existing outputs are
never overwritten. `benchmarks/documented_examples/run_all.py` exposes the
single-root version as `metal_m2_common_plane` with resources
`--m2-evidence-binding` and `--m2-alloy-bound`.

This certificate covers the declared constitutive expressions, finite source
inventory, eligible phases and fixed composition domains. It does not certify
empirical applicability, native-binary equivalence, a larger alloy domain,
a metal-free boundary, omitted transport, or a Gibbs minimum for the
nonisothermal planet. Those scientific and numerical questions retain their
own evidence and acceptance conditions.

The [five saved-case assessment](../../results/m2_common_plane/20260928/five_saved_cases/README.md)
archives the original executions: all five normalized gap upper bounds are
between `1.08e-13` and `1.50e-13`, below the `1e-9` design tolerance. Each
case retains its original strict-sign and empirical-material flags.
