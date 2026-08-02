# FastChem-Parity KL Semantic Design Note

This note defines the production-design semantics implied by the Round 9
FastChem Cond KL parity diagnostics. It is not a production behavior change.

## Scope

The design target is a future KL semantic interface that can name and test the
state basis used by donor, molecule, inventory, and removed-condensate
comparisons. The current evidence is strong enough for design work, but not for
promotion into solver behavior.

## Milestone 20 KL-Native Reconstruction Addendum

Milestone 20 attempts a KL-native reconstruction using the established semantic
interface. The ladder covers baseline KL, physical donor, molecule-cache vector,
fixed/condensed overwrite boundary, inventory/atom state, removed correction,
post-PMI tau/complementarity replay, projection coefficient response, and a
best native coherent reconstruction. Full FastChem coherent RHS is kept as a
reference only.

The result is negative for production: no KL-native variant closes current-five
or the existing `45:-5` pilot. The reference FastChem coherent RHS closes all
cases, so the remaining missing object is not one of the named interface fields
closed by Milestones 15-19. The design blocker is hidden full coherent source
state outside the KL-native reconstruction.

Milestone 20 decision: KL-native reconstruction blocked by hidden full coherent
source.

## Milestone 19 Projection/Tau Closure Addendum

Milestone 19 restores the direct-broad raw artifacts needed by the patched
Python diagnostics and reruns the current-five broad replay plus the existing
`45:-5` pilot. Projection coefficients now emit as explicit unit-RHS linear
responses; no aggregate residual inference is used.

The post-PMI tau/complementarity ladder also emits as a diagnostic replay. This
closes the review field, but it does not create a standalone tau rule.

The layer-45 coefficient comparison is important: `45:-5` and `45:-10` have
identical outside-selected unit-response coefficients for the common row pairs.
The already known `45` difference is therefore not coefficient geometry; it
remains a source residual/projection-content difference, consistent with the
Milestone 13 sign/magnitude result.

Milestone 19 decision: projection coefficients closed; post-PMI tau replay
closed.

## Milestone 18 Projection/Tau Patch Addendum

Milestone 18 implements the Python diagnostic patch sites identified by
Milestones 16 and 17. The direct-broad diagnostic script now has a first-class
projection-coefficient emitter for `solve(J, unit_outside_row)` into selected
rows, and a diagnostic post-PMI tau/complementarity residual ladder.

The patch does not close projection coefficients yet because the raw
direct-broad snapshot and trace artifacts required to construct numeric cases
are not present in this workspace. The existing direct-broad compact predates
the new emitted fields and must not be used to infer coefficients from aggregate
outside-selected residuals.

Design impact: tau/complementarity baseline consumption is acceptable design
review evidence, but projection coefficient closure requires restoring or
regenerating the direct-broad raw objects and rerunning the patched diagnostic.
Production remains not promotable.

Milestone 18 decision: projection coefficients remain blocked by exact missing
direct-broad objects.

## Milestone 17 Python Gap-Closure Addendum

Milestone 17 re-checks the two remaining Python diagnostic fields from the
design-review package using only existing artifacts.

Projection coefficients remain unavailable. The intended diagnostic object is
the selected-row response to `solve(J, unit_outside_row)` for each
outside-selected row. Existing compacts provide outside-selected RHS
differences and selected-row summaries, but not the reduced Jacobian `fc_j`,
`row_to_result`, or emitted unit-response coefficients.

Post-PMI tau/complementarity replay also remains unavailable as a separate
ladder. The available Round 8/9 fields prove tau/complementarity is already
consumed in the coherent baseline for the audited rows, but they do not emit a
standalone tau RHS vector before the removed-tail replay.

Design impact: the semantic-interface package is design-review hardened, not
production-promotable. The review can either accept the baseline-consumption
proof for tau/complementarity or request the Python ladder patch; projection
coefficients require the direct-broad Python compact patch before production
candidacy.

Milestone 17 decision: both remain Python diagnostic gaps with exact blockers.

## Milestone 14 Production-Gap Update

Milestone 14 packages the Milestone 6-13 findings into a production-readiness
gap scorecard. The current state is:

- Design-note ready: physical donor comparability, coherent molecule+inventory
  state interface, removed-condensate provenance boundary, and fixed/condensed
  overwrite plus molecule-cache boundary naming.
- Guarded diagnostic only: a default-off semantic-interface prototype that
  emits state boundaries, row-wise projection/sign checks, and reconstruction
  gates.
- Not promotable: Al4C3 removed-tail rules, tau/complementarity standalone
  rules, full coherent FastChem RHS transplant, isolated molecule/inventory or
  donor/removed transplants, and fixed-overwrite/cache transplants.

Two low-cost fields remain nonblocking but useful before design review:
explicit projection coefficients from outside-selected free-element rows into
selected condensate rows, and a separate post-PMI tau/complementarity tail
replay or explicit baseline-consumed marker. The current artifacts place both
patches in Python diagnostic compact builders, not production C++ solver logic.

## Milestone 15 Prototype Contract

Milestone 15 implements the recommended semantic interface as a compact-only,
default-off diagnostic prototype. It is ready for design review and is still
not a production behavior.

The prototype contract requires explicit fields for:

- normalized donor basis,
- physical donor basis,
- molecule-cache full-element vector,
- fixed/condensed overwrite boundary,
- `correctValues` overwrite records,
- reduced Newton RHS/Jacobian/result-slot labels,
- inventory/atom state,
- removed-condensate correction,
- tau/complementarity sensitivity,
- row-wise projection/sign audit,
- metric-family lineage.

The interface deliberately records unavailable exact projection coefficients
and standalone post-PMI tau replay as diagnostic field gaps. This preserves the
production boundary: no full coherent RHS transplant, Al4C3 rule,
tau/complementarity rule, fallback metric, legacy conversion, or row/species
dropping is part of the design.

## Milestone 16 Design-Review Package

Milestone 16 packages the guarded semantic interface for design review. The
package is diagnostic-only and production remains not promotable.

The review package adds:

- source artifacts for each interface field,
- invariant gates carried forward from the prototype,
- known unavailable fields,
- forbidden production shortcuts,
- evidence still required before production.

The low-cost field audit did not close projection coefficients. Current
compacts expose outside-selected RHS differences and selected-row summaries,
but not the coefficient matrix from each outside-selected free-element row into
each selected condensate row. The Python patch site is the direct broad eval
projection path after `common['fc_j']`, `common['row_to_result']`, and selected
rows are available.

The audit does prove that tau/complementarity is already consumed in the
coherent baseline for the current projected ladders: Round 8 records
`tau_contribution=null` on all projected rows with the baseline-consumed status,
and Round 9 repeats that status for the Al4C3 removed-tail rows. A standalone
post-PMI tau replay remains an unavailable Python diagnostic field.

## Required State Interfaces

1. Normalized donor state
   - KL raw `gas_only_final` is normalized.
   - It must not be compared directly to FastChem physical donor values.

2. Physical donor state
   - A diagnostic physical conversion is required before FastChem donor
     comparison.
   - The conversion removes the inflated `C1H4` / `H2O1` donor gap.

3. Coherent molecule + inventory state
   - Molecule-only and inventory-only replays are destructive.
   - The design interface must treat molecule and inventory rows as a coherent
     state bundle when parity diagnostics rely on their cancellation.

4. Removed-condensate correction state
   - The remaining `45:-10` PMI residual is fully explained by the emitted
     `Al4C3(s)` removed-condensate analytic correction.
   - This is a provenance boundary, not a production removed-tail patch.

## Candidate Guarded KL Option

A future prototype may add a default-off diagnostic option that emits:

- normalized donor rows,
- physical donor rows,
- molecule-cache rows,
- inventory/atom rows,
- removed-condensate correction rows,
- metric-family labels for focused and broad scorecards.

The option must not change presets, defaults, production solver behavior,
active-selection behavior, row selection, row scaling, lifecycle handling, or
FastChem compatibility behavior.

## Non-Promotion Rules

The following are explicitly not production-promotable from the current evidence:

- CH4 data-validity masking as a KL default.
- MgCO3/SiC FastChem donor snapshot transplant.
- Group-B reduced-solve exclusion as a selected-row rule.
- Legacy KL-reference burden-ratio conversion.
- Full-vector infinity-norm fallback.
- Molecule-only, inventory-only, donor-only, or removed-tail transplants.
- Broad projection as focused regression.
- Repaired alpha/beta production behavior.

## Production Candidate Gate

A future production candidate must pass the invariant checklist in
`results/fastchem_cond_kl_production_readiness_compact.json`, regenerate or
explicitly account for broad cases beyond the current five-case set, and show
that any guarded option improves the relevant metric without dropping rows,
species, or cases.

Current conclusion: semantic levers ready for production design note but not
promotable.

## Milestone 2 Generalization Readiness

Milestone 2 did not run extra broad cases. The phase2 source script accepts
`--cases`, but the downstream compact stack is not yet generalized enough to
make an extra source case interpretable. Several required compacts still use the
current five-case broad set or focused `45/60` layers, and the current broad
source artifacts are already multi-GB.

The next broad regeneration campaign must produce:

- expanded phase2 broad source and trace snapshots,
- broad selected-row delta/raw-result provenance,
- broad reduced-Newton result-slot trace inputs,
- broad FastChem electron-solver trace wrapper,
- broad `rhs_terms_fresh` and exact total-inventory/removed source artifacts,
- regenerated broad smoke, direct broad eval, Round 8 ladder, and Round 9-style
  removed-tail locality compact.

The guarded KL semantic state-interface remains a default-off prototype design
only. Its state contract is:

- emit normalized donor rows,
- emit physical donor rows with conversion provenance,
- emit molecule-cache rows with stage labels,
- emit inventory rows with normalized/physical basis labels,
- emit removed-correction rows with source function and case locality,
- attach metric-family labels to every scorecard.

The option must not change production solver behavior, presets, defaults,
active selection, row selection, row scaling, lifecycle behavior, or FastChem/KL
runtime semantics. It must also reject legacy burden-ratio conversion,
infinity-norm fallback, silent row/species/case dropping, and broad projection
as focused regression.

Milestone 2 decision: broad generalization requires regeneration campaign
before next decision.

## Milestone 3 Manifest Contract

Milestone 3 introduces a broad case manifest as the required entry point for
future broad generalization:

- `results/fastchem_cond_kl_broad_case_manifest.json`

Each case entry must declare the case key, layer, epsilon, source artifacts,
trace artifacts, downstream compact availability, physical donor availability,
molecule/inventory ladder availability, and removed-tail locality availability.
This prevents source-only pilot runs from being interpreted without the
downstream replay artifacts needed for a semantic decision.

The manifest-driven compact builder is:

- `examples/comparisons/fastchem_cond_kl_milestone3_broad_generalization_infrastructure.py`

It replays the current five broad cases from the manifest and preserves the
same design constraints: no production behavior change, no default-on guarded
mode, no legacy burden-ratio conversion, no infinity-norm fallback, no broad
projection as focused regression, and no silent row/species/case dropping.

Milestone 3 decision: manifest-driven broad generalization infrastructure
ready; pilot case not yet run.

## Milestone 4 Pilot-Ready State Interface Gate

Milestone 4 makes the downstream diagnostic stack pilot-ready without adding a
production rule or changing the guarded KL option. The current-five replay must
enter through `results/fastchem_cond_kl_broad_case_manifest.json`; diagnostic
scripts must fail rather than silently fall back to a hard-coded case list.

The manifest-only gate covers:

- repaired alpha/beta broad smoke,
- repaired alpha/beta direct broad eval,
- Round 8 physical donor / molecule / inventory ladder,
- Round 9 removed-tail locality,
- Milestone 3 infrastructure replay.

The Round 9 `45:-10` constant is allowed only as a localized removed-tail
decomposition target. It is not a broad replay default and not a production
lifecycle rule.

Pilot execution remains separate from design readiness. The one-case pilot was
not run because source+trace generation is still estimated at about `0.8G`
before downstream compacts. A pilot case can be interpreted only after it has a
manifest entry, source artifacts, trace artifacts, and regenerated downstream
compact rows for normalized donor, physical donor, molecule cache, inventory,
and removed-correction comparisons.

Milestone 4 decision: pilot-ready manifest-only downstream stack passes; pilot
case not yet run.

## Milestone 5 Pilot Design Implication

The single pilot `45:-5` was regenerated and added to the manifest. It should
not change the guarded KL state-interface design, but it tightens the required
state contract:

- source-level removed-correction rows must be emitted for every pilot case,
- selected-row mapping must be emitted before physical+molecule+inventory
  residual claims are made,
- same-boundary KL non-fixed vector must be emitted before donor closure is
  claimed,
- gauge-normalized inventory/atom rows must be emitted before coherent bundle
  claims are made,
- removed-tail locality must distinguish source provenance from projected PMI
  closure.

The pilot source trace contains `Al4C3(s)` removed-correction provenance at
`45:-5`. That is a design signal, not a production rule: projected PMI
materiality is not available until the missing downstream fields are generated.

Milestone 5 decision: one pilot broad case regenerated but downstream
interpretation remains incomplete.

## Milestone 6 Pilot Closure Design Implication

Milestone 6 closes the `45:-5` pilot interpretation from existing source and
trace artifacts. The guarded state-interface design should retain separate
channels for:

- normalized donor,
- physical donor,
- selected-row mapping,
- molecule cache,
- inventory/atom rows,
- removed-correction rows,
- projected residual metrics.

The design reason is now stronger: `45:-5` shows that source-level
`Al4C3(s)` removed-correction provenance is not enough to infer projected
closure. Physical donor conversion closes the dominant donor gap, but the
coherent molecule+inventory+removed bundle does not reproduce the `45:-10`
closure behavior. The interface must therefore expose both provenance and
projection results, with basis labels, before any semantic option can be
reviewed.

Milestone 6 decision: pilot 45:-5 downstream interpretation reveals a new
blocker.

## Milestone 7 Blocker Attribution Design Implication

Milestone 7 attributes the `45:-5` blocker without running another broad case.
The partial physical+molecule+inventory plus `Al4C3(s)` removed replay does not
close the projected PMI residual, while the full FastChem coherent RHS closes
all selected rows.

The attribution is not a corrected removed-tail rule. The available
delta-to-full decomposition shows the remaining gap is dominated by neutral
molecule full-vector source-state provenance outside the selected rows, which
couples through the coherent solve. The selected-row RHS gap between the best
partial replay and the full FastChem RHS is roundoff-scale, so the design
interface must treat the coherent RHS source-state bundle as the comparison
unit when this boundary is audited.

The guarded diagnostic interface should therefore continue to emit separate
normalized donor, physical donor, molecule cache, inventory/atom, removed
correction, selected-row mapping, and metric-family fields, but it must also
label whether a comparison is using a partial replay or the full coherent RHS
source-state bundle. This remains design-only; no default-on guarded option,
production rule, or solver behavior change is justified.

Milestone 7 decision: 45:-5 blocker requires full coherent RHS source-state
bundle.

## Milestone 8 Full-Bundle Decomposition Design Implication

Milestone 8 decomposes the `45:-5` full coherent RHS bundle requirement one
level deeper from the existing pilot artifacts. The smaller local blocker is
neutral molecule full-vector source-state coupling, not charged molecules,
inventory/atom, removed correction, row scaling, or row/Jacobian
materialization.

The design interface must therefore preserve full-vector molecule source-state
provenance outside the selected rows. For `45:-5`, the affected outside-selected
rows are all free-element rows; the largest rows are `Mn`, `S`, `K`, `Na`,
`Mg`, `Fe`, `Cl`, and `Cu`. Species-level attribution is available from the
pilot molecule-cache records and shows new neutral species such as `H1Mn1`,
`Cl2Na2`, `Fe1H2O2`, `H2Mg1O2`, `Cl2K2`, `H2S1`, `Al1F2O1`, and `O2V1`
dominate the coupling. `C1H4` and `H2O1` remain non-dominant after physical
donor conversion.

The top species records have matching mass-action / hvector and density-gauge
fields; the leading residual comes from the full element vector consumed by the
neutral molecule cache. A future default-off diagnostic interface should label
that source-state snapshot explicitly, while still keeping partial replay and
full coherent RHS comparisons separate.

Milestone 8 decision: 45:-5 full bundle blocker is neutral molecule full-vector
source-state coupling.

## Milestone 9 Neutral Source-State Decomposition Design Implication

Milestone 9 decomposes the `45:-5` neutral molecule source-state coupling one
level deeper. The blocker is the molecule-cache full-element vector consumed at
`iter1_full_reduced_system`, not hvector/logK and not density gauge. The
dominant top species remain non-`C1H4`/`H2O1`: `H1Mn1`, `Cl2Na2`,
`Fe1H2O2`, `H2Mg1O2`, `Cl2K2`, `H2S1`, `Al1F2O1`, and `O2V1`.

The available trace reconstructs per-species log densities from the physical
donor vector and from the molecule-cache full-element vector. The named
additive split between physical donor and hidden source-state budget is not
emitted; the exact diagnostic patch site is the molecule cache trace emitter for
per-neutral-species donor/source-state budgets. Reconstruction from the emitted
full element vector is sufficient for the current local verdict, but not for a
production rule.

Milestone 9 decision: 45:-5 neutral source blocker is molecule-cache
full-element vector.

## Milestone 10 Cache-Vector Provenance Design Implication

Milestone 10 traces the `45:-5` molecule-cache full-element vector to the
cache-side post-`correctValues` molecule refresh record. The first matching
stage is `molecule_refresh_record_0`, emitted as
`iter0_post_correctValues_full_element_vector_before_molecule_refresh`; the same
top-element vector is then consumed by the iter1 RHS molecule-density
provenance and the cache-side iter1 full reduced system.

For the top elements (`Mn`, `S`, `K`, `Na`, `Mg`, `Fe`, `Cl`, `Cu`, plus
`Al`, `F`, `O`, `V`, and `H` from the dominant species), the emitted
`value_source_mode` is `reduced_overwrite_from_correctValues` with a fixed
element overwrite component. Candidate-vector replay verifies that the physical
donor vector does not recover cache density, while the post-refresh cache vector
and iter1 RHS vector do.

The next missing diagnostic field is inside that overwrite: per-element additive
budget for old element density, solver delta, clipping/cap, condensate-coupled,
and inventory/removal components. That deeper split requires a C++ trace patch;
Python reconstruction is sufficient for the current local verdict but not for a
production rule.

Milestone 10 decision: 45:-5 cache-vector source is fixed/condensed overwrite
component.

## Milestone 11 Fixed/Condensed Overwrite Budget Design Implication

Milestone 11 decomposes the fixed/condensed overwrite one level deeper using a
fresh diagnostic-only `45:-5` raw FastChem trace. The local `correctValues`
budget is now source-checked: old element density, raw solver delta, clipped
solver delta, and the final reduced new number density reconstruct the cache
overwrite values to relative roundoff. That reconstructed vector also matches
the molecule-refresh cache vector and recovers the Milestone 10 top-species
molecule-density replay.

This is still not a production rule. The physically meaningful additive split
is not emitted: retained-condensate, removed-condensate, condensate-coupled, and
inventory/removal contributions are folded into `result(i + nb_cond_jac)`.
Design work should therefore treat `correctValues` closure as a diagnostic
identity gate, not as a promotable handoff or source-state policy.

Milestone 11 decision: 45:-5 overwrite budget remains unresolved due missing
additive trace fields.

## Milestone 12 Reduced Newton Result-Slot Design Implication

Milestone 12 moves the missing-field boundary upstream into the reduced
Newton solve that feeds `correctValues`. The diagnostic trace now emits the
`newton_iter=0` full reduced system, RHS term families, retained/Jac and
removed-active condensate burden splits, row scaling, and Jacobian subblocks.
Using the traced global result scaling factor, the result slot reconstructs the
Milestone 11 raw solver delta and preserves the overwrite/cache/top-species
closure gate.

This is still not a production rule. The attribution is a fixed-J
linear-sensitivity budget over the traced RHS families. For the `45:-5` top
elements, tau/complementarity rows propagated through the coupled reduced
linear solve dominate the result slot. That establishes a diagnostic source
family, not a transferable update policy.

Milestone 12 decision: 45:-5 result slot is coupled-linear-solve dominated.

## Milestone 13 Layer-45 Projection/Sensitivity Design Implication

Milestone 13 resolves the layer-45 comparison at the row/projection level. The
aggregate Milestone 12 tau/complementarity dominance is not enough by itself;
the decisive difference is how the Al4C3 removed projection lands on the
projected condensate rows.

For `45:-10`, Al4C3 is opposite-sign to the PMI residual on all eight projected
rows and closes the residual to tolerance. For `45:-5`, Al4C3 is mixed-sign
against PMI: four rows improve and four rows worsen, leaving a material
residual until the full coherent RHS replay brings in the outside-selected
neutral molecule full-vector coupling.

This remains diagnostic-only. The result is a sign/magnitude explanation for
the existing projection behavior, not a production projection or removed-tail
rule.

Milestone 13 decision: 45 comparison resolved by Al4C3 projection mismatch
against tau/complementarity sensitivity.

## Milestone 21 Design Implication

Milestone 21 audits the Milestone 20 ladder before interpreting it. D is only a
semantic boundary label for C, and H/I are semantic labels for G; those compact
aliases must not be counted as independent reconstruction evidence.

The new vector-level diagnostic reconstructs the I-versus-J gap explicitly:
`KL_native_rhs + hidden_rhs_delta == full_FastChem_rhs` to roundoff, and solving
the hidden RHS delta through the same FastChem Jacobian reconstructs the
J-minus-I solution delta. This makes the hidden coherent source an RHS source
state problem. Jacobian and row scaling are not implicated by the available
evidence.

Design review can accept the current diagnostic interface as a non-production
evidence package. Production still needs a KL-native source-state contract; it
must not transplant the full FastChem coherent RHS or promote Al4C3/tau rules.

Milestone 21 decision: hidden coherent source is RHS-side.
## Milestone 22 Design Implication

Milestone 22 decomposes the Milestone 21 hidden RHS vector by RHS term family. Decision: hidden RHS delta is molecule-RHS dominated.

- The additive budget closes for all requested cases: current-five plus the existing `45:-5` pilot.
- The closing source is `full FastChem molecule RHS - reconstructed candidate molecule RHS`; no separate remaining charged/electron additive hidden family is needed for closure.
- Inventory/atom, removed-condensate, tau/complementarity, activity burden, and fixed/condensed overwrite families have zero remaining additive hidden RHS after the Milestone 21 KL-native RHS assembly.
- Outside-selected free-element rows remain the dominant carriers, but they are a row-location carrier view of the molecule RHS residual rather than a separate additive source.
- Production remains not promotable; KL-native reconstruction is blocked on a coherent molecule RHS parity contract, not on a new production rule.
## Milestone 23 Design Implication

Milestone 23 tests the coherent molecule RHS parity contract directly. Decision: coherent molecule RHS parity holds at matched source state.

- Full FastChem molecule RHS is reproduced to roundoff by all-molecule RHS at matched source state.
- The M22 molecule delta is therefore not a request to transplant FastChem RHS; it identifies the missing semantic contract for source-state plus row-scaling/RHS convention parity.
- Source-vector, hvector/lnK, density-gauge, cache timing, and neutral/charged branches are emitted as diagnostic variants.
- KL-native reconstruction remains blocked until the semantic interface exposes a matched molecule source state and RHS convention contract.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 24 Design Implication

Milestone 24 attempts to construct the matched coherent molecule source state from KL-native semantic-interface fields. Decision: matched coherent molecule source state blocked by hidden coherent source.

- The matched source-state contract is explicit: e-first physical density vector, FastChem row-scaling convention, molecule species order, hvector/lnK, and density gauge.
- No KL-native candidate among physical donor, molecule-cache, fixed/overwrite, correctValues, or best repaired same-boundary vector constructs the matched source state in all requested cases.
- Tau/complementarity, inventory/atom, removed-condensate, and reduced-slot adjusted all-element molecule-source vectors are not emitted as source-state fields.
- Row scaling is explicit and can be tested separately; it is not sufficient to construct the missing source state.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 25 Design Implication

Milestone 25 attempts to materialize the missing all-element molecule source vectors from existing artifacts. Decision: source-vector materialization blocked by exact missing artifacts.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and e-first same-boundary KL candidates were materialized and rerun.
- Tau/complementarity, inventory/atom, removed-condensate, and broad reduced-slot all-element molecule source vectors are not present in existing artifacts; only RHS terms or focused/layer-limited reduced-slot records are available.
- Combination search cannot close because every requested adjusted-vector combination depends on an unmaterialized all-element source vector.
- FastChem row scaling remains explicit and is not used silently; materialized source-state residuals remain nonzero under that explicit convention.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 26 Design Implication

Milestone 26 implements diagnostic emitter attempts for the missing all-element source vectors. Decision: all-element source-vector emission blocked by exact trace architecture gap.

- Python emitter attempts were added at the latest diagnostic sites where RHS terms, source traces, row scaling, and molecule labels are live.
- Current artifacts expose RHS term contributions for tau/complementarity, inventory/atom, and removed-condensate paths, but not the adjusted all-element molecule-source vectors.
- The reduced-slot compact remains focused/layer-limited and does not expose a broad case-keyed canonical e-first all-element molecule-source vector.
- Matched-source construction was rerun with all emitted/materialized vectors; no non-hidden candidate closes.
- Production remains not promotable; a future diagnostic trace must emit these all-element source vectors before another construction attempt can close.
## Milestone 27 Design Implication

Milestone 27 patched diagnostic C++ trace schema and emitted the reduced-slot all-element source vector. Decision: matched coherent molecule source state still blocked by hidden coherent source.

- The emitted vector is FastChem reference-only and uses the hidden coherent FastChem source state.
- It reproduces the matched coherent molecule source state to roundoff, proving the trace architecture can carry the needed vector shape.
- It does not make the matched source KL-native constructible; KL adjusted all-element tau/inventory/removed vectors remain missing.
- Production remains not promotable; the patch is trace-only and inactive unless diagnostic tracing is enabled.
## Milestone 28 Design Implication

Milestone 28 compares KL-native source-vector candidates against the M27 emitted reduced-slot FastChem reference target. Decision: KL-native source-vector reconstruction blocked by exact KL-side trace fields.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and prior same-boundary KL-native vectors were compared directly against the M27 28-element target.
- No non-hidden candidate or requested staged combination closes the source vector or molecule RHS in all cases.
- Tau/complementarity, inventory/atom, removed-condensate, and KL reduced-slot mapped all-element source vectors remain exact missing fields.
- Production remains not promotable; this milestone is diagnostic-only and does not change presets, defaults, or solver behavior.
## Milestone 29 Design Implication

Milestone 29 closes the M28 Python-owned KL-side source-vector fields and reruns reconstruction. Decision: M27 target vector still requires hidden FastChem coherent source.

- All eight M28 fields were ownership-audited and emitted in the M29 diagnostic compact.
- The two fields previously labelled C++-required are KL-side source-state fields after ownership audit, so no C++ trace patch or rebuild was required.
- Emitted non-hidden KL-side vectors still do not reconstruct the M27 FastChem reference target or matched coherent molecule RHS.
- Production remains not promotable.
## Milestone 30 Design Implication

Milestone 30 decomposes the value delta between the M27 target and emitted KL reduced-slot source vector. Decision: M27 target blocked by missing semantic transform.

- Element-order, log/linear, density-gauge, and row-scaling verifier branches do not close the residual.
- Fixed/non-fixed and inert/electron splits show the largest differences are not explained by a single bookkeeping subset.
- Least-squares fits can reduce residual diagnostically but require non-production coefficients, indicating a missing semantic transform rather than a direct emitted-vector selection.
- Production remains not promotable.
## Milestone 31 Design Implication

Milestone 31 synthesizes diagnostic semantic transforms from emitted KL-side vectors to the M27 target. Decision: inert/carrier transform partially closes but not all cases.

- Inert pass-through from the KL physical donor stage removes the dominant He/Ne/Ar carrier residuals diagnostically.
- Source-vector and molecule-RHS closure still fail after all A-H transforms, so the transform is only partial and remains non-production.
- LS-guided class coefficients do not collapse to a global, ntot, or density-gauge scalar across cases.
- No production solver, preset, default, or guarded mode changed.
## Milestone 32 Design Implication

Milestone 32 prototypes broader diagnostic source-state transforms after M31 inert pass-through. Decision: broader transform partially closes but fixed-element source-state remains.

- Gen1 attempted A2-H2; Gen2 attempted targeted top-class, metal/minor, fixed/condensed, and constrained-scalar overlays.
- No non-hidden transform reconstructs the M27 target or matched molecule RHS across current-five plus `45:-5`.
- The balanced best transform remains conservative and the remaining dominant residual class is fixed/condensed source-state material.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 33 Design Implication

Milestone 33 decomposes the fixed-element source-state blocker. Decision: fixed-element transform rejected due molecule-RHS sign amplification.

- A2, F2, and K2 were decomposed by condensation class and residual class for current-five plus `45:-5`.
- A3-G3 RHS-aware fixed-element transforms were attempted; no transform reconstructs the M27 target and molecule RHS across all cases.
- Source-vector improvements can worsen molecule RHS because fixed-row changes are amplified by molecule stoichiometry and FastChem row scaling.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 34 Design Implication

Milestone 34 audits the source-to-molecule-RHS operator. Decision: sign amplification due row scaling.

- R(x) was evaluated directly from diagnostic source vectors through molecule reconstruction and FastChem row scaling.
- Finite-difference sensitivity on top fixed elements attributes the worsening RHS behavior to row-scaled fixed-element molecule-burden sensitivity.
- RHS-space stage/scalar/finite-difference candidates do not close all cases without hidden coherent source state.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 35 Design Implication

Milestone 35 decomposes row-scaling amplification into numerator and scaling terms. Decision: row-scaling high-gain fixed rows require M27 source parity.

- The compact emits `N(x)`, `R(x)`, `Delta N`, `Delta R`, row-scaling signs/magnitudes, and amplification factors for each covered case.
- Row-scaling-aware candidates A-F do not close all cases, current-five, or `45:-5` without hidden coherent source state.
- High-gain fixed rows remain the practical inverse-problem blocker; production remains not promotable.
## Milestone 36 Design Implication

Milestone 36 traces high-gain fixed-row source provenance. Decision: high-gain fixed-row parity insufficient; full M27 source vector remains required.

- High-gain fixed rows were selected from M35 row-scaling amplification and RHS contribution budgets.
- FastChem and KL ladders show the M27 reduced-system assembly source value is not reproduced by emitted KL-native stages.
- High-gain-row-only replays are diagnostic-only and do not close all cases; full M27 source-vector parity remains required.
- No production solver, preset, default, or C++ trace changed.
## Milestone 37 Design Implication

Milestone 37 decomposes source-vector support. Decision: sparse support partially closes but not all cases.

- Row classes and overlaps were emitted for high-gain fixed, inert, electron, volatile, reactive, metal/minor, top-species, outside-selected, and row-scaling high-gain rows.
- Replays A-K and greedy support search found a stable outside-selected support branch that partially closes, including `45:-5`, but not all current-five cases.
- Top-species support rows improve attribution but do not replace full M27 source-vector parity for non-closing cases.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 38 Design Implication

Milestone 38 decomposes residuals after the M37 sparse support. Decision: 45:-5 closes by cancellation; current-five requires broader support.

- The best sparse support residual was decomposed for H/K, and `45:-5` was compared against non-closing current-five cases.
- Outside-selected internal ablations A-G and a second-stage greedy search were attempted.
- The 45:-5 closure is not a general support sufficiency result; current-five still requires broader support.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 39 Design Implication

Milestone 39 decomposes the remaining `30:-10` sparse-support residual. Decision: 30:-10 residual is numerical/tolerance-scale but not production-promotable.

- H/B/G/K support residuals were decomposed for `30:-10` and compared with closing cases.
- Internal add/remove ablations A-H and tolerance/scale audit classify the remaining residual as strict-tolerance scale.
- Closure under looser diagnostic tolerance is not production-promotable; production remains not promotable.
## Milestone 40 Design Gate

Milestone 40 campaign decision: strict tolerance residual is diagnostic-only; production requires source-state contract.

- Track 1: strict 30:-10 residual closes only at diagnostic 1e-5 and is not production-promotable.
- Track 2: emitted non-hidden KL full-vector candidates do not close source, numerator, or row-scaled RHS parity.
- Track 3: production readiness now depends on a formal source-state semantic contract, not another support subset.
- Track 4: no additional broad pilot is justified until the contract gap is addressed.
## Milestone 41 Source-State Contract Design

Milestone 41 created the default-off source-state contract prototype. Decision: default-off source-state contract prototype complete; KL-native implementation remains blocked.

- The schema separates FastChem/M27 reference records, best non-hidden KL candidates, sparse support overlays, tolerance-only closures, and the unavailable production-ready KL-native source state.
- The acceptance gate requires source parity, numerator parity, row-scaled RHS parity, no hidden FastChem source, strict tolerance, complete coverage, explicit row scaling, and preserved lineage.
- Current-five plus `45:-5` are instantiated from existing artifacts; no new broad pilot was requested.
- All best non-hidden KL candidates fail the default-off production gate because the source-state constructor and numerator contract are still missing.
## Milestone 43 Synthesis Result

Milestone 43 synthesized gate-driven constructors from non-hidden KL-side basis vectors. Decision: free fit also fails; hidden source-state information structurally absent.

- Generations G1-G3 attempted source-, numerator-, RHS-, joint-, class-wise, constrained, and free diagnostic fits.
- M41 gate evaluation remained default-off and diagnostic-only; production behavior and presets were unchanged.
- The free diagnostic fit also failed source, numerator, and RHS gate closure, classifying the residual as structurally absent from the non-hidden KL basis.
## Milestone 44 Primitive Source-State Map

Milestone 44 expanded the non-hidden KL primitive basis and emitted structural span diagnostics. Decision: diagnostic C++ trace required for missing primitive source-state fields.

- Current and expanded basis rank/projection residuals were emitted for source vector, unscaled numerator, and row-scaled RHS spaces.
- Primitive branches A-G were attempted; exact number_density_min/maj and epsilon/phi/degree transforms remain unavailable without diagnostic trace fields.
- Expanded basis gate rerun did not close M41; next work requires diagnostic C++ trace for missing primitive source-state fields.
## Milestone 45 Trace Primitive Closure

Milestone 45 added env-gated C++ primitive source-state trace fields. Decision: traced primitives improve but remain FastChem reference-only.

- The M45 marker emits fixed-row pre/post overwrite values, number_density_min/maj, gas solver path, epsilon, phi, and degree_of_condensation.
- The traced primitive basis reruns the M41 gate but remains diagnostic FastChem reference-only, not KL-native production logic.
- Production remains not promotable; the next implementation step is coding the semantic source-state algorithm on the KL side.
## Milestone 46 Algorithm Extraction

Milestone 46 implemented a default-off KL-native semantic source-state algorithm prototype. Decision: KL lacks required lifecycle input fields for semantic source-state algorithm.

- Variants A-D compute primitive fields from KL/public vectors only; M45 trace values are diagnostic reference targets only.
- The M41 gate rerun preserves no-hidden-source, not-reference-only, and KL-native-constructible checks, but source/N/R parity remains open.
- Missing KL lifecycle inputs are now narrowed to molecule contribution order, backup branch lifecycle, condensation-degree transform inputs, and reduced correctValues assembly semantics.
## Milestone 47 Lifecycle Input Map

Milestone 47 localized semantic source-state gaps by FastChem function. Decision: FastChem semantic source-state functions must be ported before production.

- Four default-off Python function-port prototypes were implemented and compared against M45 trace references without using traced values as constructor inputs.
- Three M47 source-state candidates were gated through M41 over current-five plus 45:-5.
- Remaining blockers are exact lifecycle fields: molecule order/accumulators, backup/intertSol branch state, condensation-stage inputs, and correctValues reduced result/clipping state.
## Milestone 48 Lifecycle State Schema

Milestone 48 emitted default-off KL lifecycle state records and reran five M48 candidates through M41. Decision: M48 improves but molecule order state remains missing.

- Implemented lifecycle emitters: condensation_stage_state, correctValues_reduced_result_state, gas_solver_branch_state, minor_major_accumulator_state.
- Unavailable lifecycle emitters: molecule_order_state.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem reference-source transplant was changed.
## Milestone 49 Molecule Order State

Milestone 49 emitted KL molecule-order state proxies and reran minor/major accumulator replay. Decision: exact molecule_order_state requires FastChem-specific ordering not present in KL.

- KL-native molecule-order emitters: A_KL_gas_species_order, B_FastChem_label_aligned_KL_density, D_best_non_hidden_molecule_order.
- Exact element-specific FastChem minor/major molecule order remains unavailable from KL artifacts and would need a diagnostic trace for validation.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem source-vector transplant was changed.
## Milestone 50 Trace And Port Readiness

Milestone 50 added the v6 molecule-order trace and assessed KL port readiness. Decision: FastChem createMoleculeLists algorithm must be ported before production.

- v6 trace marker verified: `True`.
- Trace closes exact FastChem molecule order as reference evidence, but KL-native production needs createMoleculeLists and staged molecule-density lifecycle ports.
- No production solver behavior, presets, defaults, tolerance, coverage, or coherent source-vector transplant was changed.
## Milestone 51 CreateMoleculeLists Port

Milestone 51 implemented default-off KL-native createMoleculeLists port variants. Decision: M51 improves but molecule abundance lifecycle remains missing.

- Implemented four molecule abundance candidates and five createMoleculeLists port variants.
- Reran six M51 M41 candidates with best-selection tie-break by gate/pass policy and residuals.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or trace-as-constructor path was changed.
## Milestone 52 Molecule Abundance Lifecycle Emitters

Milestone 52 implemented default-off KL lifecycle state carriers for molecule abundance, ordered molecule density, and n_major accumulator replay. Decision: M52 improves but abundance state stage mismatch remains.

- Implemented five KL-native molecule abundance lifecycle emitters and ordered-density stage replay.
- Implemented four n_major accumulator lifecycle replay variants.
- Reran seven M52 M41 candidates over current-five plus 45:-5.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or trace-as-constructor path was changed.
## Milestone 53 Abundance Stage Alignment

Milestone 53 implemented default-off abundance stage alignment and accumulator path search. Decision: M53 improves but abundance values mismatch after stage search.

- Implemented ten KL-native abundance-stage candidates A-J and ordered-density alignment replay.
- Implemented six n_major accumulator path variants A-F.
- Reran eleven M53 M41 candidates over current-five plus 45:-5, including continuation verifier branches.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or trace-as-constructor path was changed.
## Milestone 54 Dominance Split Source/N/R Conflict

Milestone 54 implemented default-off dominance split and source/N/R conflict diagnostics. Decision: case-family-specific abundance-stage contract required.

- Ran 30:-10, layer-45, and source/N/R Pareto conflict branches.
- Implemented six 30:-10 branch candidates, six layer-45 branch candidates, and seven source/N/R conflict candidates plus a stage-contract verifier.
- Reran twenty M54 M41 candidates over current-five plus 45:-5.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or trace-as-constructor path was changed.
## Milestone 55 Case-Family Abundance-Stage Contract

Milestone 55 emitted the default-off case-family abundance-stage contract and reran M41 for 48 contract records.

- Implemented eight contract prototypes across thirty_m10, layer45, standard current-five, and global fallback families.
- Contract-only records did not pass strict source/N/R checks, so five broader semantic source-state port verifier candidates were run.
- The diagnostic package identifies the fixed/high-gain source-state port as the next implementation target and keeps production behavior unchanged.
## Milestone 56 Fixed/High-Gain Source-State Port

Milestone 56 implemented the default-off fixed/high-gain source-state lifecycle carrier and gated 42 primary records.

- Added an opt-in carrier emitter in `src/exogibbs/optimize/pipm_rgie_cond.py`.
- Added an opt-in correctValues/condensation diagnostic record emitter in `src/exogibbs/optimize/minimize_cond.py`.
- Ran three continuation verifier branches after the primary M56 candidates failed strict source/N/R checks.
- Production defaults and solver behavior remain unchanged.
## Milestone 57 CorrectValues Condensation Source-State Carrier

Milestone 57 implemented a default-off case-keyed correctValues/condensation source-state carrier and gated 42 primary records.

- M56 depth audit classified the previous metric path as decorator-only for M41 source/N/R metrics.
- M57 source vectors are materialized from carrier overwrite/result-slot/clip fields, then passed through the M41 source-to-N/R operator.
- Six continuation branches were run after the primary carrier candidates failed strict source/N/R checks.
- Production defaults and solver behavior were not changed.
## Milestone 58 Source N R Conflict Resolution

Milestone 58 gated 108 default-off diagnostic records through the M41 source/N/R checks.

- Static decision audit found no fixed M58 final-decision constant.
- Pairwise objective candidates, carrier-parameter variants, operator-contract probes, and case-family branches were run.
- Computed decision: 30:-10 drives objective incompatibility under carrier space.
- Production defaults and solver behavior were not changed.
## Milestone 59 30 m10 Source N R Driver Resolution

Milestone 59 gated 25 30:-10 driver-specific diagnostic records.

- Static audit prevents a driver label from being a terminal decision.
- 30:-10 bridge, carrier-space expansion, broader correctValues/condensation, global replacement, and continuation branches were run.
- Computed decision: 30:-10 conflict is dominated by source-vector contract.
- Production defaults and solver behavior were not changed.
## Milestone 60 Source Vector Contract Carrier

Milestone 60 gated 11 source-contract carrier records after patching the default-off carrier helper.

- Source-contract bridge fields were added as no-op-by-default diagnostic carrier fields.
- Source/N/R were recomputed from carrier-owned M60 values and M41 was rerun for 30:-10 and global replacement branches.
- Computed decision: M60 source contract is hidden source-state not expressible by current KL carriers.
- Production defaults and solver behavior were not changed.
## Milestone 61 Hidden Source State Ownership

Milestone 61 gated 12 hidden-source ownership bridge records.

- M27 reduced-slot mapped all-element source vector closes the 30:-10 source/N/R diagnostic reference checks.
- M27-aligned exact records are hidden/reference-only and are not production candidates.
- Computed decision: M61 identifies reduced-slot solve-state owner with exact port target.
- Production defaults and solver behavior were not changed.
## Milestone 62 Reduced Slot Solve State Port

Milestone 62 gated 11 reduced-slot solve-state carrier records.

- A default-off reduced-slot solve-state source carrier helper was added.
- Source/N/R were recomputed from carrier-owned M62 values and global replacement gates were run.
- Computed decision: M62 reduced-slot port improves source/N/R but result-slot primitive field blocks closure.
- Production defaults and solver behavior were not changed.
## Milestone 64 Raw Reduced Solver Result Reconstruction

Milestone 64 gated 15 raw result-slot reconstruction records.

- The reduced-slot carrier now records raw slot vectors, slot basis, nb_cond_jac, element slot indices, solve convention, backend, global scaling, scaled slots, and correctValues bridges.
- v7 values were used only as reference targets and mapping evidence.
- Decision: M64 blocked by exact ownership gap.
- Production defaults and solver behavior were not changed.

<!-- fastchem-cond-kl-m65-exact-same-iteration-kl-trace-alignment -->
## Milestone 65 Exact Same-Iteration KL Trace Alignment

- Artifact: `results/fastchem_cond_kl_milestone65_exact_same_iteration_kl_trace_alignment_compact.json`
- Decision: M65 exact same-iteration KL state reconstruction is blocked with exact field target.
- FastChem v8 is sufficient for J/RHS/raw-result/backend reference targets; no new C++ trace is currently indicated.
- KL `include_system_trace=True` emits the required field schema, but the exact `30:-10` / `newton_iter=0` input bundle is not present in KL-owned artifacts.
- Exact next target: src/exogibbs/optimize/pipm_rgie_cond.py::diagnose_reduced_solver_backend_experiments must emit the exact case-scoped newton_iter=0 input bundle: ln_nk, ln_mk, ln_ntot, formula_matrix, formula_matrix_cond, b, gk, bk, hvector_cond, sk, condensate row labels, and element row labels before _update_all_with_metrics.

<!-- fastchem-cond-kl-m66-exact-kl-input-bundle-emission -->
## Milestone 66 Exact KL Input Bundle Emission

- Artifact: `results/fastchem_cond_kl_milestone66_exact_kl_input_bundle_emission_compact.json`
- Decision: M66 blocked by exact code ownership gap.
- `diagnose_reduced_solver_backend_experiments` now has a default-off exact input bundle emitter before `_update_all_with_metrics`.
- The live 30:-10 `newton_iter=0` call-site still lacks ownership of case labels, condensate indices, and reduced element labels needed to activate the emitter on the target state.
- Next target: src/exogibbs/optimize/pipm_rgie_cond.py::minimize_gibbs_cond_core or its caller must pass case_key, condensates_jac_indices, condensate_labels_jac_order, and element_labels_reduced_order into diagnose_reduced_solver_backend_experiments for 30:-10 iter 0.

<!-- fastchem-cond-kl-m67-live-callsite-exact-bundle-wiring -->
## Milestone 67 Live Call-Site Exact Bundle Wiring

- Artifact: `results/fastchem_cond_kl_milestone67_live_callsite_exact_bundle_wiring_compact.json`
- Decision: M67 live call-site wiring blocked at caller context ownership with exact target.
- Caller-level context plumbing is now default-off and wired through `trace_condensate_reduced_solver_backends` plus `_update_all_with_metrics` diagnostics.
- The remaining missing owner is a KL-owned 30:-10 init/context carrier with the 106-row label state.
- NRKG readiness: READY.

<!-- fastchem-cond-kl-m67-plus-kl-owned-30-m10-carrier -->
## Milestone 67+ KL-Owned 30:-10 Carrier

- Artifact: `results/fastchem_cond_kl_milestone67_plus_kl_owned_30_m10_carrier_compact.json`
- Decision: M67+ live KL-owned 30:-10 bundle emitted; row/slot mapping mismatch blocks direct v8/KL comparison.
- The default-off diagnostic caller now receives KL-owned layer/profile state and row-label context.
- The emitted KL bundle solves in the element-plus-ntot basis; the v8 target remains a condensate-slot plus selected-element-slot basis.
- NRKG readiness: READY.
