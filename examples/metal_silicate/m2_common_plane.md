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

The reconstructed-water model uses `--water-proof` instead of the original
native-water liquid proof. Its coefficients, elemental plane, external water
standard shift, H2 standard and exact source/audit hashes must match the
fresh audit. The complete dry simplex and all nonnegative water/H2 amounts
have an analytic volatile minimum; only the lower bound relaxes the original
oxygen-capacity constraint. The atom-repaired primal retains that constraint
and evaluates the unreduced extensive water/H2 scalar. A negative lower bound
remains usable, with its explicit contribution to the common-plane correction.

For `metal_model='phosphorus'|'associated'|'associated_k'`, omit `--alloy-bound`.
The runner freshly reconstructs the provider-owned scalar and standards from
the actual saved P gas anchor, original Ma host, association/K recipe and
separate scenario offsets. All selected reduced chemical potentials are checked
against the saved primitive ledger. Positive-curvature reference bounds or the
H–O alphaBB bound cover the entire declared domain; search error remains an
explicit acceptance condition. P's coupled Fe constraint is enclosed by a
larger rectangle only for the lower bound, with curvature recomputed on that
rectangle. The feasible primal always obeys the original domain.
Associated components count chemical species moles: each component's complete
atom column enters the plane, exact budget repair and atom normalization.
This extension does not certify a zero-metal branch or uncalibrated material
properties.

With reconstructed water and an explicit dry-host He scalar, the lower bound
also eliminates every nonnegative dissolved-He amount analytically:
`min_nHe (G_He/RT - lambda_He*nHe) = -M_dry*a*exp(lambda_He-muHe0)`.
This is linear in the declared dry-host masses and only changes the dry
component costs. The provider receipt, actual retained-gas standard/gauge,
primitive He amount and independent host correction must all match. Water
and H2 have zero He-host mass, and the original H2 denominator is unchanged.
The feasible upper bound explicitly reverses that elimination and evaluates
the unreduced scalar at its exact atom-repaired He amount. A minimized He
energy must never be used as a feasible upper bound. Other liquid models with
He currently fail closed in this runner.

`--alloy-tolerance-rt` (default `1e-10`) and `--alloy-max-nodes` (default
`20000`) control fresh extended-alloy bounds. The stricter search target helps
resolve the `1e-9` common-plane energy gate; neither option changes that gate
or the chemical acceptance tolerance. A finite budget that does not resolve
the bound remains explicitly unaccepted.

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
