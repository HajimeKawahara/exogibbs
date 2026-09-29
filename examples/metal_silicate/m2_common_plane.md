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

Post-hoc proof code can differ from the executed source's construction and
selection wrappers. The runner verifies **every** recorded source-recipe hash
against its original git blob. Only `melts_coupled.py`, `m2_expanded_source.py`,
`m2_scenarios.py` and `phase_selection.py` may differ in the current checkout;
all other recorded files, including the thermochemical replay and gas data,
must remain byte-identical. Source-building control flow is not rerun. Saved
scenario values/domains, constitutive coefficients, standards and the exact
primitive energy are bound separately. Original blob hashes and current proof
hashes are both retained. The standalone four-component alloy runner offers
this same policy only with `--allow-historical-builders`; its default remains
strict byte equality. Missing commits, changed historical blobs or modified
replayed standards fail closed.

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


## Fixed-pressure internal sources

The common-plane and reconstructed-water runners remain final-pressure-root
only by default. `--allow-fixed-pressure-source` explicitly permits the separate
`m2_fixed_pressure_internal_source_v1` envelope after a fresh physical audit
labels it `fixed_pressure_internal_source`. This envelope preserves one original
`source_state`, its inventory, scenario, clean provider records, and primitive
source ledger. It must not contain `runs` or `roots`, and its pressure-closure
flag must remain false. The source atoms, contact, column diagnostics, exact
T/P, and every existing constitutive-expression/standard binding still apply.

Both runners reuse the same global bounds and primal-dual arithmetic. A
successful fixed-pressure result certifies only the declared finite source at
that T/P. Its `pressure_closure_performed: false` and explicit source kind do
not establish a bottom-pressure root, global column closure, or empirical
material applicability. Earlier root-only artifacts and their input bytes are
unchanged. Solid-model bounds use the exact same fresh physical audit; they do
not require or imply a pressure solve.
