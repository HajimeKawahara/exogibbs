# RGIE/PIPM Transfer Notes

These notes separate transferable semantics from FastChem-specific numerical mechanics.

## Latest KL-Native Reconstruction Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone20_kl_native_reconstruction_attempt_compact.json`
- `results/fastchem_cond_kl_milestone20_kl_native_reconstruction_attempt_compact.md`

Transfer boundary:

- Transfer the negative result: the explicit semantic interface is sufficient
  for design review but not sufficient to reconstruct the FastChem closing
  direction natively.
- Do not transfer full FastChem coherent RHS into RGIE/PIPM production logic.
- Do not turn tau/complementarity, Al4C3, removed correction, or projection
  response into standalone rules.
- Further transfer work should decompose hidden coherent source state, not add
  another pilot case.

Current transfer decision: KL-native reconstruction blocked by hidden full
coherent source.

## Latest Direct-Broad Projection/Tau Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone19_direct_broad_regeneration_projection_tau_compact.json`
- `results/fastchem_cond_kl_milestone19_direct_broad_regeneration_projection_tau_compact.md`

Transfer boundary:

- Transfer projection coefficients only as diagnostic linear-response evidence,
  not as a production correction.
- Transfer the fact that layer-45 coefficient geometry is shared between
  `45:-5` and `45:-10`; it does not explain the Al4C3/projection mismatch.
- Transfer the post-PMI tau ladder as diagnostic provenance only. Do not create
  a standalone RGIE/PIPM tau rule.
- The next transfer work should be KL-native reconstruction, not another pilot
  case.

Current transfer decision: projection coefficients closed; post-PMI tau replay
closed.

## Latest Projection/Tau Patch Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone18_projection_tau_patch_compact.json`
- `results/fastchem_cond_kl_milestone18_projection_tau_patch_compact.md`

Transfer boundary:

- Transfer the patched diagnostic interface shape, not coefficient values: the
  current workspace cannot emit coefficients until the direct-broad raw
  snapshot and trace JSON files are restored or regenerated.
- Do not infer outside-selected projection coefficients from aggregate compact
  residuals.
- Treat tau/complementarity baseline consumption as accepted design-review
  provenance only. It is not an RGIE/PIPM production correction.
- The next step is a KL-native reconstruction or restored direct-broad raw
  diagnostic rerun, not one more pilot case.

Current transfer decision: projection coefficients remain blocked by exact
missing direct-broad objects.

## Latest Python Gap Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone17_python_gap_closure_compact.json`
- `results/fastchem_cond_kl_milestone17_python_gap_closure_compact.md`

Transfer boundary:

- Transfer the exact blocker, not a coefficient estimate: current artifacts do
  not contain the `fc_j` and `row_to_result` fields needed for
  `solve(J, unit_outside_row)` projected into selected rows.
- Transfer the tau/complementarity baseline-consumption proof as diagnostic
  provenance only. A separate post-PMI tau ladder is still not emitted.
- Do not transfer full coherent FastChem RHS, Al4C3, or tau/complementarity as
  standalone RGIE/PIPM production rules.
- No new pilot case is justified by this audit; the next step is design-review
  acceptance or a Python compact patch.

Current transfer decision: both remain Python diagnostic gaps with exact
blockers.

## Latest Design Review Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone16_design_review_and_field_closure_compact.json`
- `results/fastchem_cond_kl_milestone16_design_review_and_field_closure_compact.md`

Transfer boundary:

- Transfer the design-review interface definitions and invariant gates.
- Do not infer outside-selected projection coefficients from aggregate
  residuals. The required transfer object is a per-outside-row linear response
  coefficient into each selected condensate row, and it is not emitted.
- Treat the tau/complementarity baseline-consumption proof as a diagnostic
  provenance statement only. It is not a standalone RGIE/PIPM correction.
- Production transfer remains blocked until projection coefficients or
  accepted nonmateriality proof, post-PMI tau semantics, and regenerated broad
  scorecards are reviewed.

Current transfer decision: design review package complete; projection/tau
fields remain Python diagnostic gaps.

## Latest Guarded Interface Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone15_guarded_semantic_interface_prototype_compact.json`
- `results/fastchem_cond_kl_milestone15_guarded_semantic_interface_prototype_compact.md`

Transfer boundary:

- Transfer the interface contract, not FastChem state values: RGIE/PIPM should
  name donor basis, molecule-cache source, overwrite boundary, reduced Newton
  labels, inventory/atom state, removed correction, tau sensitivity, projection
  signs, and metric lineage before comparing directions.
- Do not transfer full coherent FastChem RHS as an implementation source.
- Do not transfer Al4C3 or tau/complementarity as standalone production rules.
- Treat unavailable outside-selected projection coefficients and post-PMI tau
  replay as explicit diagnostic gaps, not reasons to infer hidden production
  mappings.

Current transfer decision: guarded diagnostic semantic-interface prototype
complete; design review is next.

## Latest Production-Gap Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone14_production_promotability_gap_compact.json`
- `results/fastchem_cond_kl_milestone14_production_promotability_gap_compact.md`

Transfer boundary:

- Transfer only the semantic-interface requirement: explicitly name normalized
  donor, physical donor, molecule cache, inventory, removed correction,
  fixed/condensed overwrite, metric family, and row mapping boundaries.
- Do not transfer Al4C3 or tau/complementarity as standalone RGIE/PIPM rules.
  Milestone 13 showed row-wise sign/magnitude mismatch at `45:-5`.
- Do not transfer full coherent FastChem RHS, fixed overwrite, or molecule-cache
  vectors as hidden production inputs.
- Projection coefficients from outside-selected free-element rows and separate
  post-PMI tau tail replay remain diagnostic compact gaps; they can improve
  reviewability but do not justify production behavior.

Current transfer decision: production gap package complete; guarded diagnostic
interface is next.

## Latest Layer-45 Projection/Sensitivity Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone13_45_comparison_projection_sensitivity_compact.json`
- `results/fastchem_cond_kl_milestone13_45_comparison_projection_sensitivity_compact.md`

Transfer boundary:

- Do not transfer Al4C3 removal as a general correction rule. It closes
  `45:-10` because the projected signs and magnitudes cancel the PMI residual
  on all eight projected rows.
- The same projection does not generalize to `45:-5`: four projected rows
  improve and four worsen, and full coherent RHS is still required.
- RGIE/PIPM must preserve row-wise sign and projection geometry when comparing
  reduced directions. Aggregate RHS-family dominance is insufficient.
- Outside-selected neutral molecule full-vector coupling remains a diagnostic
  requirement for the `45:-5` path; explicit projection coefficients are not
  emitted and should not be inferred as a production rule.

Current transfer decision: `45` comparison resolved by Al4C3 projection
mismatch against tau/complementarity sensitivity.

## Latest Pilot 45:-5 Result-Slot Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone12_reduced_newton_result_slot_budget_compact.json`
- `results/fastchem_cond_kl_milestone12_reduced_newton_result_slot_budget_compact.md`

Transfer boundary:

- Do not transfer the reduced result slot as a production shortcut. Milestone
  12 attributes the traced `45:-5` slot by fixed-J RHS-family sensitivity only.
- The closure gate is stronger: the traced, globally scaled slot reconstructs
  the Milestone 11 raw delta, then the overwrite, molecule-cache vector, and
  top-species replay.
- The dominant focused source family is tau/complementarity propagated through
  the coupled reduced linear solve, not an isolated retained/removed
  condensate or molecule-RHS rule.
- RGIE/PIPM work should treat this as a diagnostic coupling requirement:
  tau/complementarity rows, row scaling, and the reduced solve backend must be
  audited before any reduced-direction parity rule is considered.

Current transfer decision: `45:-5` result slot is coupled-linear-solve
dominated.

## Latest Pilot 45:-5 Overwrite-Budget Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone11_fixed_condensed_overwrite_budget_compact.json`
- `results/fastchem_cond_kl_milestone11_fixed_condensed_overwrite_budget_compact.md`

Transfer boundary:

- Do not transfer the fixed/condensed overwrite as a production shortcut.
  Milestone 11 only proves the local `correctValues` overwrite identity for the
  focused `45:-5` trace.
- The cache-vector replay gate remains diagnostic-only: reconstructed overwrite
  values match the molecule-refresh vector and recover the Milestone 10
  top-species density replay.
- The source split needed for RGIE/PIPM remains missing. Retained-condensate,
  removed-condensate, condensate-coupled, and inventory/removal terms are folded
  into `result(i + nb_cond_jac)`.
- A transferable rule would require a diagnostic-only upstream expansion of the
  reduced element result into those additive terms. Until then, do not port a
  `correctValues`, fixed-overwrite, molecule-cache, or inventory/removed rule.

Current transfer decision: `45:-5` overwrite budget remains unresolved due
missing additive trace fields.

## Latest Pilot 45:-5 Neutral Source-State Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone10_pilot_45_m5_cache_vector_provenance_compact.json`
- `results/fastchem_cond_kl_milestone10_pilot_45_m5_cache_vector_provenance_compact.md`

Transfer boundary:

- Do not transfer the cache vector or fixed/condensed overwrite as a production
  shortcut. Milestone 10 identifies the local source as the cache-side
  `reduced_overwrite_from_correctValues` component before molecule refresh.
- Candidate-vector replay is diagnostic-only: physical donor does not recover
  the top-species cache densities, while the post-refresh cache vector and iter1
  RHS vector do.
- The deeper additive budget inside `correctValues` is not emitted. RGIE/PIPM
  transfer work needs old element density, solver delta, clipping/cap,
  condensate-coupled, and inventory/removal components before any source rule is
  portable.
- `45:-10` still lacks the same cache-vector provenance ladder in the listed
  artifacts. Al4C3 projection closure there cannot be used to infer an
  analogous cache-vector cancellation.
- Continue to reject legacy KL-reference burden-ratio conversion,
  full-vector infinity-norm fallback, broad projection as focused regression,
  and silent row/species/element/case dropping.

Current transfer decision: `45:-5` cache-vector source is fixed/condensed
overwrite component and remains diagnostic-only.

## Prior Pilot 45:-5 Neutral Source-State Transfer Boundary

The prior compact audit is:

- `results/fastchem_cond_kl_milestone9_pilot_45_m5_neutral_source_state_decomposition_compact.json`
- `results/fastchem_cond_kl_milestone9_pilot_45_m5_neutral_source_state_decomposition_compact.md`

Transfer boundary:

- Do not transfer the full coherent RHS bundle as a production shortcut. The
  smaller local blocker is the molecule-cache full-element vector consumed at
  iter1 RHS assembly, and it remains diagnostic provenance only.
- Do not transfer individual top neutral species as rules. `H1Mn1`, `Cl2Na2`,
  `Fe1H2O2`, `H2Mg1O2`, `Cl2K2`, `H2S1`, `Al1F2O1`, and `O2V1` identify the
  `45:-5` source-state gap, not a generalized production correction.
- Keep selected-row and full-vector metrics separate. The selected-row RHS gap
  after partial replay is roundoff-scale, while the material residual lives in
  outside-selected free-element rows.
- Treat mass-action / hvector and density-gauge matching as negative evidence
  for those subterms in this pilot. The residual source is the full element
  vector consumed by the neutral molecule cache.
- Keep the additive donor/source-state split as a missing diagnostic field. The
  emitted full element vector is sufficient for the local verdict, but not for a
  production rule.
- Continue to reject legacy KL-reference burden-ratio conversion,
  full-vector infinity-norm fallback, broad projection as focused regression,
  and silent row/species/element/case dropping.

Current transfer decision: `45:-5` neutral source blocker is molecule-cache
full-element vector and remains diagnostic-only.

## Prior Pilot 45:-5 Full-Bundle Transfer Boundary

The prior compact audit is:

- `results/fastchem_cond_kl_milestone8_pilot_45_m5_full_bundle_decomposition_compact.json`
- `results/fastchem_cond_kl_milestone8_pilot_45_m5_full_bundle_decomposition_compact.md`

Milestone 8 decomposed the full coherent RHS bundle to neutral molecule
full-vector source-state coupling, dominated by outside-selected free-element
rows. Milestone 9 supersedes only the smaller-source label; the production
guardrails remain unchanged.

## Latest Pilot 45:-5 Transfer Boundary

The latest compact audit is:

- `results/fastchem_cond_kl_milestone7_pilot_45_m5_new_blocker_attribution_compact.json`
- `results/fastchem_cond_kl_milestone7_pilot_45_m5_new_blocker_attribution_compact.md`

Transfer boundary:

- Do not port the `Al4C3(s)` removed-tail correction as a standalone RGIE/PIPM
  rule. It recurs at `45:-5`, but projected closure is epsilon-dependent.
- Do not port a partial molecule, inventory, removed-tail, row-scaling,
  Jacobian, or tau/complementarity rule from the `45:-5` pilot. The full
  selected-row closure appears only when the full FastChem coherent RHS
  source-state bundle is used.
- Treat the neutral molecule full-vector provenance outside the selected rows
  as a source-state coupling requirement, not as a focused regression target.
  The `embedded_broad_10row_projection` remains separate from focused
  regression.
- Keep missing-field reporting explicit. Do not use legacy KL-reference
  burden-ratio conversion, full-vector infinity-norm fallback, or silent
  row/species/case dropping to force a transfer claim.

Current transfer decision: `45:-5` requires full coherent RHS source-state
bundle provenance and remains diagnostic-only.

## Latest Broad Projection Transfer Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.md`

Transfer boundary:

- Keep focused and broad metric families separate. The focused closure belongs to `focused_raw_result_provenance_metric`; the broad residual probe is `embedded_broad_10row_projection` and must not become a focused regression target.
- Treat the closed focused one-sided frontier as diagnostic provenance only: CH4 data-validity mask, MgCO3/SiC FC donor snapshot, and Group-B intentional reduced-solve exclusion explain the focused frontier but are not RGIE/PIPM rules.
- Do not treat the broad residual as reduced by the focused closure. The compact records `40` broad numeric failures, `12` focused-frontier annotations, and `38` still-failing broad rows. The remaining rows are retained, not silently dropped.
- Full-vector term differences are now emitted for all five broad cases and point to neutral molecule full-vector provenance. This is still a diagnostic source-state attribution, not a production molecule rule.
- The broad replay protocol preserves `embedded_broad_10row_projection` and reduces residual only under diagnostic combined available-term alignment. Beta/electron-only and row-scaling/Jacobian-only broad replays remain missing-field records.
- Do not fall back to full-vector infinity norms or legacy KL-reference burden-ratio conversion. The compact reports l2/mean/max where available and records missing terms explicitly.
- Current transfer decision: broad projection residual is dominated by neutral molecule full-vector source state.

## Latest Neutral Molecule Transfer Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.md`

Transfer boundary:

- Treat the neutral-molecule result as a coherent bundle requirement, not as a molecule-only transfer rule. Molecule RHS alignment alone worsens all `40` broad rows, inventory/atom alignment alone worsens all `40` rows, and only combined available-term alignment improves all `40`.
- Do not port top molecule species conclusions yet. The current broad artifacts do not emit per-projected-row neutral species contributors, molecule densities, mass-action terms, element input vectors, or density-gauge terms.
- Do not assign an earliest species-stage divergence. The source-stage lineage for top neutral molecule contributors is missing from the broad replay artifact and is recorded as a Python audit patch requirement.
- Current transfer decision: neutral molecule residual requires coherent molecule+inventory+removed/tau bundle.

## Latest Fixed-Element Source-Decomposition Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.md`

Transfer boundary:

- Treat molecule cache as downstream symptom only. The current source-clean boundary is the FastChem cached full-element vector before molecule refresh.
- The transferable provenance result is narrower than a physical additive budget split. At that stage FastChem exposes an exact overwrite/carry-forward decomposition: fixed/reduced elements come from the `correctValues` overwrite path, while non-reduced elements are carried forward in the full vector.
- Do not transfer guessed `free_atomic_gas_component`, condensed correction, inventory, or electron subcomponents from this audit. Those fields are still not explicit in source at the cached-vector stage and are recorded as missing with the exact source function instead of being inferred.
- Treat fixed-element overwrite values as the dominant component mismatch in the focused smoke. Overwrite-only replay materially improves molecule reconstruction, but it does not produce a coherent selected-row RHS replay, so it is still not a transferable production rule.
- Electron is secondary, not dominant. Fixed-element overwrite plus electron improves molecule density further, but the selected-row replay still fails by many orders of magnitude relative to the full FC upper bound.
- Keep the existing guardrail: no fixed-element handoff rule, membership rule, electron rule, `phi` rule, or molecule rule should be ported to RGIE/PIPM from this audit. Full FC cached input vector remains diagnostic-only.

## Latest Fixed-Element Overwrite Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.md`

Transfer boundary:

- Treat the fixed-element mismatch specifically as an overwrite-value provenance problem, not as a molecule-cache timing problem and not as a proven overwrite-mask rule.
- The FastChem overwrite source is exact and narrow: `elem_densities_new[i]` fully replaces the current fixed-element value through `elements_cond[i]->number_density = elem_densities_new[i]` in `CondensedPhase::calculate`.
- Do not transfer guessed prior-value or timing semantics. The exact prior full-element carry-forward numeric value at the write point is still missing from the compact artifact, and the audit records the precise missing local variable instead of inferring it.
- Overwrite-only replay is the decisive single-component candidate, but it is still diagnostic-only because selected-row closure still requires the full FC cached vector. Overwrite + electron helps further without producing a coherent RHS replay.
- Keep the production guardrail unchanged: do not port overwrite values, overwrite mask, overwrite timing, electron, molecule, or RHS behavior from this audit. Full FC cached vector remains a diagnostic upper bound only.

## Latest Fixed-Element Materialization Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.md`

Transfer boundary:

- Treat the next blocker as a missing materialization boundary, not as a gas-recoupling success criterion. The KL gas-recoupling output exists, but the later molecule-input vector is rebuilt inline before iter1 RHS/Jacobian assembly.
- Do not infer an exact pre-gas vector from the compact artifact. The audit records the exact missing field:
  `gas_trace.post_condensed_phase_fixed_atomic_element_species_state`
  from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::actual_fastchem_like_coupled_loop`.
- The first later consumer is `_second_post_seed_update_actualization_solve`, which reads `gas_only["ln_nk"][:n_elem]` and passes that `u` into `_assemble_fastchem_reduced_update`.
- `_assemble_fastchem_reduced_update` reconstructs the atom/full-element vector inline from current `u`. No explicit fixed-element overwrite consumer is exposed before that reconstruction. The audit records the exact missing object instead of guessing it:
  `iter1_molecule_input.fixed_element_bookkeeping_consumer`
  from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- Gas-recoupling carry alone is not transferable. On the focused layers it does not improve the molecule replay, while FC fixed overwrite values still dominate the improvement and full FC cached input is still required for the upper bound.
- Keep the production guardrail unchanged: do not port a gas-recoupling carry rule, fixed-overwrite materialization rule, fixed-mask rule, molecule rule, or RHS rule from this audit. The boundary is source-proven but remains diagnostic-only.

## Latest Synthetic Fixed-Overwrite Consumer Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.md`

Transfer boundary:

- Treat this as a negative synthetic-consumer replay, not as a new KL carry rule. Gas-recoupling output adoption still exists in the diagnostic gas replay path, but the later consumer boundary remains the inline-recomputed `u` inside `_assemble_fastchem_reduced_update`.
- Keep the missing-consumer result explicit. The compact audit still cannot point to any emitted `iter1_molecule_input.fixed_element_bookkeeping_consumer` before molecule reconstruction, so no transferable fixed-overwrite consumer exists yet.
- Do not port the gas-recoupling fixed subset as an overwrite surrogate. On both focused layers the best KL-side synthetic rung is still the unchanged current KL vector, and splicing gas fixed values into that boundary does not improve the selected-row replay.
- The fixed-subset adoption result is narrower than the earlier overwrite-provenance result: FastChem overwrite values still matter, but gas-recoupling fixed values do not explain them. Electron is negligible in this synthetic ladder.
- Keep the guardrail explicit: synthetic fixed-overwrite consumer is informative but not promotable. Full FC cached input remains required for the coherent upper bound, so no RGIE/PIPM production rule should be ported from this audit.

## Latest Elem Densities New Source Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.md`

Transfer boundary:

- Treat this as a narrower overwrite-source result, not as a new additive source split. The focused reduced-branch trace now exposes the exact local construction chain for fixed rows in `CondensedPhase::correctValues`, but it still does not expose separate additive carry-forward, condensed/fixed correction, or electron-specific component locals.
- The transferable provenance result is now source-clean and specific: fixed rows later written through `elem_densities_new[i]` are constructed as `elem_number_dens_old[i] -> result(i + nb_cond_jac) -> delta_n_elem -> update_factor -> elem_number_dens_new[i]`, and every focused fixed row classifies as `pure overwrite`.
- Do not port a free-gas carry-forward, condensed/fixed correction, or electron-specific rule from this audit. Those components remain un-emitted as dedicated locals and are reported as exact missing fields rather than inferred.
- Overwrite-term-only replay is still the only source-clean component candidate, and it still fails the selected-row closure by many orders of magnitude relative to the full FC cached-input upper bound.
- Keep the guardrail explicit: `elem_densities_new` source is further resolved but remains diagnostic-only. No RGIE/PIPM production rule should be ported from this audit.

## Latest Reduced-Newton Result Slot Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.md`

Transfer boundary:

- Treat this as a narrower post-solve bridge result, not as a solved subterm split. The fixed-row overwrite site is now tied back to the exact same-iteration reduced-Newton slot, but the trace still does not expose a finer retained-condensate, removed-condensate, fixed/condensed correction, or electron-coupling local at that same iteration.
- The transferable provenance result is now: fixed rows depend on the same-iteration reduced solver result slot plus an explicit caller-side `global_scaling_factor` in `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`. The reduced-newton anatomy also exposes slot index, scaled RHS entry, row scaling factor, and reduced Jacobian diagonal.
- Do not port a retained-condensate, removed-condensate, fixed/condensed correction, electron-coupling, or row-scaling bridge rule from this audit. Those finer source components remain unresolved from emitted same-iteration locals, and the only successful bridge found here is still diagnostic-only.
- The key negative result is equally important: the unbridged pre-global-scaling result slot is materially worse than the bridged overwrite replay on both focused layers, so the raw solver slot itself is not a transferable fixed-row update rule.
- Keep the guardrail explicit: reduced-Newton result slot provenance is further resolved but remains diagnostic-only. No RGIE/PIPM production rule should be ported from this audit.

## Latest Fixed-Row Solve-Space Boundary

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.md`

Transfer boundary:

- Treat the reduced-Newton bridge as resolved provenance, not as the final frontier. The next unresolved object is the fixed-row solve-space equation that produces `z_i` before caller/global scaling.
- The transferable provenance result is now narrower and explicit:
  same-iteration slot index, `b_i`, `z_i`, `g`, `g*z_i`, row scaling factor, and `J_ii` are source-proven
  same-iteration labelled matrix rows are still missing from emitted diagnostics
- Do not port a same-iteration solve-space matrix-row rule from this audit. The exact missing same-iteration locals remain `jacobian` and `rhs` in `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep`, so the off-diagonal decomposition still relies on the emitted labelled iter1 analogue.
- The labelled analogue is still useful diagnostically: the row equations close to roundoff on a relative basis, and Schur-style attribution shows RHS-only, condensate-coupling, and other-element-coupling pieces. But no single piece closes the replay, and the bridged full slot remains better than any isolated solve-space component on both focused layers.
- Keep the guardrail explicit: fixed-row solve-space provenance is further resolved but remains diagnostic-only. No RGIE/PIPM production rule should be ported from this audit.

## Latest Coherent-Bundle Transfer Boundary

The coherent gas-state bundle compact audit tightens the transfer boundary further:

- Keep the selected-row mapping metric attached to every compact residual report. The new coherent-bundle ladder explicitly reports selected-row mean, selected-row max, and full-vector norm separately; do not compare those metrics interchangeably.
- Treat inventory gauge normalization as source-proven provenance only. It remains necessary to recover the molecule/inventory cancellation, but it is still not a production inventory rule.
- Treat the coherent bundle as the current diagnostic closure boundary, not as a transfer candidate. In the mapped selected rows, only the full FC coherent gas-state bundle closes both focused layers, with exact removed correction needed only for layer `45:-10`.
- Do not transfer KL-native molecule-cache or mass-action reconstruction from this smoke. The iter1 RHS-entry trace still lacks a symmetric KL full element vector and per-molecule mass-action constant ledger, so KL-native coherent bundle replay is not source-proven.
- RGIE/PIPM should carry forward the bundle fields and stage labels as provenance requirements: atom rows, molecule-cache provenance, normalized and physical inventory rows, row scaling, active-burden rows, removed rows, and the exact post-refresh / iter1 RHS-entry stage boundary.
- Production implication is unchanged: full FastChem gas-state bundle remains diagnostic-only; no smaller KL production rule is promotable from the current coherent-bundle audit.

## Latest KL-Native Molecule Reconstruction Boundary

Refresh-timing / FC mass-action transfer boundary:

- FastChem mass-action provenance is now field-complete for the focused smoke. The emitted FC ledger self-closes molecule cache reconstruction from the FC full element vector and per-molecule mass-action constants to `5.551115123125783e-17`.
- This removes FC mass-action availability as the blocker, but it does not create a transferable rule. KL vector + FC mass action still does not recover molecule/inventory cancellation, and FC vector + KL mass action is destructive.
- The remaining transfer stop is the missing distinct KL cached-refresh snapshot. KL currently exposes only the inline RHS/Jacobian molecule-cache construction; FastChem exposes a cached molecule vector carried from post-iter0 refresh into iter1 assembly.
- RGIE/PIPM should not port a FastChem cached molecule state or a mass-action/hvector convention from this audit. Require a stage-labelled KL refresh snapshot before deciding whether timing alone, a hidden coupled snapshot, or another source-stage mismatch is transferable.

Decision: KL-native molecule timing provenance remains mixed or inconclusive.

Fresh field-completion transfer boundary:

- The KL-side molecule source ledger is now available for the focused entrance smoke. It records the RHS-entry full element vector, physical atom basis, per-molecule `mass_action_constant=-hmol`, RHS-consumed molecule cache, and the inline refresh-equivalent stage in `_assemble_fastchem_reduced_update`.
- KL molecule formula closure is no longer the transfer blocker: the emitted formula closes to roundoff on both focused layers.
- The transfer blocker moves to stage semantics. KL recomputes the molecule cache inline at RHS/Jacobian assembly, while FastChem uses a cached molecule vector stable from the post-iter0 molecule-density refresh through iter1 RHS/Jacobian assembly.
- KL-native source-clean molecule reconstructions do not recover molecule/inventory cancellation, with or without the proven density-gauge bridge. Direct FC cache replay still reaches the known cancellation boundary, so this remains diagnostic provenance rather than a transferable molecule rule.
- RGIE/PIPM should require both sides to emit stage-labelled molecule-cache refresh snapshots, per-molecule mass-action/hvector ledgers, and the exact full element vector consumed by the cache before interpreting molecule-cache differences as a production convention.

Decision: molecule mismatch is dominated by molecule-cache refresh timing.

The KL-native molecule reconstruction compact audit tightens the transfer boundary again:

- Treat the molecule-cache formula itself as source-proven but not transferable. Earlier molecule audits already show that both FastChem and KL self-close their own molecule-density formulas to roundoff; that is not the current blocker.
- Treat the missing KL refresh / RHS-entry fields as the hard transfer stop. The focused smoke still does not emit a symmetric KL full element vector or per-molecule mass-action ledger at the molecule-cache refresh boundary or iter1 RHS entry.
- Treat the FastChem cache as a coupled snapshot requirement, not as a KL rule. The FastChem cached molecule vector remains stable from `after_iter0_calcNumberDensity_refresh` through iter1 `assembleJacobian`, and the only molecule replay that reaches the coherent closure boundary still uses that direct FC cache.
- Do not transfer a KL-native molecule-cache, mass-action, hvector, or density-gauge rule from this smoke. Without the missing KL fields, RGIE/PIPM cannot separate full-element-vector mismatch from hidden refresh-state mismatch cleanly enough for a production rule.
- Production implication stays unchanged: FastChem hidden/coupled molecule snapshot remains diagnostic-only; no KL production rule is promotable.

## Latest Baseline / Gauge Transfer Boundary

The baseline-reconciliation and inventory-gauge compact audit tightens the transfer boundary:

- Treat the fresh baseline_ABC mismatch as an audit extractor issue, not as a new physical replay fact. The fresh compact dropped the prior selected-row metric and reported a full-vector infinity norm instead. Any RGIE/PIPM comparison that mixes these metrics will generate a false baseline mismatch.
- Treat KL budget inventory and FastChem physical inventory as different gauges of the same source quantity in this smoke. KL budget matches FastChem normalized epsilon rows to roundoff, and KL budget physicalized by FastChem total-element density matches FastChem physical total inventory to roundoff.
- Do not transfer that gauge conversion as a production rule. It is source-proven for diagnostic comparison only.
- After baseline/gauge reconciliation, the prior cancellation structure returns: molecule + inventory + atom closes layer `60:-5`, and exact removed replay is only the remaining layer-45 tail. This is still not a transferable smaller rule; it remains a diagnostic decomposition of the coherent RHS state.
- Layer `60:-5` removed replay is `not_applicable` when both removed sets are empty; do not record it as unresolved in downstream transfer ledgers.

## Latest Source-Provenance Boundary

The latest compact exact total-inventory / removed-source audit now has a fresh field-complete artifact. Transfer implication:

- Treat exact total-inventory closure as source-proven on each side, but treat FastChem-vs-KL total-inventory provenance as still mixed because KL exposes the consumed `budget[element]` entry rather than a separate `total_element_density * epsilon` factorization.
- Do not substitute shared gas diagnostics for source comparison once exact RHS-consumed rows are present; use the emitted row payloads instead.
- Treat the layer-45 removed tail as source-proven and separate from the total-inventory mismatch story.
- Keep the full gas-state bundle diagnostic-only; no smaller molecule/inventory or removed-tail production rule is transferable from the current artifact set.

## Transfers Cleanly

- Physical density gauge conversion before comparing or initializing condensed states.
- Full candidate row materialization before thresholding or sparse filtering.
- Total-inventory upper-bound semantics for candidate `maxDensity`.
- Explicit lifecycle seed state for newly active condensates.

## Transfers With Care

- FastChem `correctValues` clipping is useful for parity audits, but RGIE/PIPM should treat it as a trust-region analogue, not as a required production formula.
- Retained/removed partitioning can inform working-set construction, but RGIE/PIPM should keep its own KKT/complementarity interpretation.
- FastChem row scaling and reduced RHS/Jacobian anatomy are diagnostic references for solver parity, not direct algorithm requirements.
- Negative iter-1 delta replay means RGIE/PIPM should audit state handoff, cap status, and row mapping before comparing reduced directions.
- The shared-row four-way replay validates the `correctValues` algebra, so RGIE/PIPM transfer work should focus on state refresh: old log activity, old atomic element densities, and inherited post-iter0 condensate state.
- The mapped FastChem iter1 old-state trace proves FastChem-side log-activity closure for focused shared rows, but KL/RGIE/PIPM still need an explicit global-basis element-density ledger before donor-refresh conclusions are portable.
- Row coverage limitations remain: FastChem iter1 raw rows do not cover KL-only candidates such as `CH4(s,l)` in the focused smoke, so candidate-wide replay conclusions require more trace coverage.
- The iter1 reduced-system audit shows KL-only rows can enter RHS/Jacobian/row scaling and change the focused update. RGIE/PIPM comparisons should align the row universe before assigning blame to RHS/Jacobian formulas or solver tolerances.
- The iter1 row-universe replay confirms row-universe alignment can materially reduce the shared-row mismatch, but the mixed layer response means RGIE/PIPM should treat row filtering as a provenance diagnostic, not as a standalone solver rule.
- The iter1 full reduced-system trace now emits full labelled RHS/Jacobian/scaling/result arrays for the two-layer smoke. RGIE/PIPM transfer should compare labelled row/column systems and explicitly account for unmatched rows/columns before replaying RHS, Jacobian, scaling, or solver-result components.
- Projection alone is not enough in the current smoke: drop/freeze/source-labelled projections are solvable but worsen focused density mismatch. Scaling and conditioning must be audited alongside any row/column isomorphism.
- The row-scaling audit identifies the current reduced-system frontier as scaling plus conditioning context. FastChem and KL audit both scale Jacobian rows and RHS entries by row factors, but fixed-scaling swaps show the projected Jacobian/RHS assembly still controls much of the remaining reduced-direction difference.
- Solver backend details matter for portability: FastChem uses Eigen LU variants with optional configured fallbacks, while the KL audit uses NumPy dense solves with least-squares fallback. RGIE/PIPM transfer should record factorization, pivoting, conditioning, and fallback rules before accepting parity of a Newton direction.
- The reduced block decomposition shows the current dominant shared-system mismatch is the element-row / element-column Jacobian block, not the retained condensate diagonal or retained condensate stoichiometry rows. RGIE/PIPM transfer should separately audit atom diagonal terms, molecular stoichiometric outer products, removed-condensate fold-in, and element-basis mapping.
- The element-element subterm audit narrows the current dominant source term to the molecule stoichiometric outer-product contribution. A simple global physical-density scalar does not close the block, so RGIE/PIPM transfer must audit molecule-density gauge, element variable basis, and old molecular-density construction before copying reduced-system assembly formulas.
- Per-molecule provenance is now required before transferring the element-element molecule outer-product. In the focused smoke, `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2` dominate the high-residual entries, and both FastChem and KL close their own molecule-density formulas to roundoff. The mismatch is therefore a state/gauge/hvector provenance problem, not an absent molecule-density trace.
- The top-molecule factorization audit shows that direct `J_mol` replay closes the molecule outer-product subterm but leaves a mixed reduced-system residual. RGIE/PIPM transfer must keep source `logK.dat` record/temperature-segment provenance and stage-labelled molecule densities, otherwise hvector, gauge, and old-atomic-state explanations cannot be separated.
- The cached molecule-vector audit shows that FastChem refreshes `Molecule::number_density` from the full element vector after iter0 and then reuses that cached value in iter1 `assembleJacobian`. The cache is not stale within the traced solve; the remaining source-level mismatch is the full element vector used to reconstruct the molecule densities.

## Does Not Transfer Directly

- Replaying traced FastChem `delta_n_cond` is a provenance diagnostic only.
- Replaying FastChem old state is also diagnostic-only unless a source-proven handoff rule closes the focused residual.
- Excluding KL-only rows is diagnostic-only until a FastChem-equivalent row-universe rule is proven from source and survives an extra trace.
- Replaying the FastChem iter1 active row universe through KL is diagnostic-only; it exposes remaining RHS/Jacobian/scaling/result differences and must not be reused as a broad lifecycle tweak.
- Replaying FastChem RHS/Jacobian/scaling/result components is not yet available because the current labelled systems are not isomorphic; KL has unmatched reduced rows/columns at both focused layers.
- Reusing a FastChem-style row projection without the matching scaling/conditioning semantics is not transferable. The current audit shows KL row scaling materially changes projected solver results.
- Reusing FastChem row scaling as a standalone RGIE/PIPM rule is not transferable. The latest diagnostic did not prove a scaling-only transplant that closes the focused residual; it instead exposes the need to audit reduced RHS/Jacobian assembly under a fixed, well-conditioned scaling.
- Replaying the whole FastChem reduced Jacobian, or even the dominant element-row / element-column block, is not transferable yet. The focused smoke shows the block is dominant but not sufficient alone to close the density residual.
- Replaying only the FastChem molecule outer-product element-element subterm is not transferable yet. It materially improves the projected solver result, but the focused density residual does not close and the exact physical-density or variable-basis rule is not yet proven.
- Replaying FastChem traced molecule densities in the molecule outer-product is also not transferable yet. It closes the molecule block by construction, but the tested source-derived transforms remain mixed and the focused density residual does not close.
- Replaying only top molecule contributors such as `H2`, `C1H4`, or `H2O1`, or replaying cumulative top-k molecule contributions, is not transferable. These diagnostics reduce selected entry residuals but do not generalize across both layers and often worsen the projected solver result.
- Recomputing molecule densities from a KL reduced-state atom vector is not interchangeable with FastChem cached molecule densities unless the full element vector, hvector/mass-action convention, and refresh stage are source-aligned. The current audit proves the FastChem cache semantics, but not a transferable KL production rule.
- The full element-vector audit narrows the current molecule outer-product frontier: the FastChem full element vector is traced, but H/C/O molecule errors do not close from full-vector offsets alone. Top rows are dominated by mass-action/hvector gauge/provenance residuals, so RGIE/PIPM must keep the gas thermo source record, no-segment gas-record semantics, and gauge convention attached to each molecule-density reconstruction.
- Replaying a FastChem full element vector without a source-proven mass-action/hvector alignment rule is not transferable. In the focused smoke, FastChem full vector plus KL hvector worsens `J_mol`, while direct cached replay closes only the molecule subterm and leaves the reduced-system residual elsewhere.
- The source-level FastChem/KL gas thermo relationship is now proven for the focused top molecules: gas `logK.dat` has one 5-coefficient record per species in this path, and FastChem evaluates `k_FC = raw_logK(T) + (sum_nu - 1)*ln(1e-6*k_B*T)`. With KL source convention `h_source=-raw_logK(T)`, this is `k_FC = -h_source - (sum_nu - 1)*ln(1e6/(k_B*T))`.
- This provenance does not transfer as a standalone RGIE/PIPM solver rule. The source formula closes the common-convention mass-action comparison, but diagnostic `J_mol` reconstruction with current KL/full-vector states worsens the focused result, so RGIE/PIPM must keep source-record gauge conversion separate from element-state, projection, and reduced-system assembly checks.
- The KL hvector plumbing "unknown convention" result was an audit attribution problem for the five focused top molecules. Molecule-column provenance must use `gas_setup.species[n_elem:]`; the old VMR comparison slice `gas_setup.species[29:]` shifted the names by one. After corrected labels, `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2` all have `h_current=h_source` in both focused layers.
- The density-basis candidate `h_needed=h_source+(sigma_m-1)*ln(1e6/(k_B*T))` remains source-proven, but it does not reduce `J_mol` or improve focused iter1 density in the corrected-label smoke. RGIE/PIPM transfer should keep hvector conversion diagnostic-only until it closes both molecule-block and update metrics.
- Full all-species molecule-column proof remains blocked by local formula-parser limits for alias/suffix gas species and one `F2Si1` name/formula discrepancy. RGIE/PIPM transfer should treat top-molecule attribution as proven, but all-species gas-name parsing as a separate ledger task.
- After FastChem cached `J_mol` replay closes the molecule outer-product term, full `J_ee` and full element-row/element-column replays do not further reduce the focused projected residual, while full FastChem Jacobian replay closes the projected linear solve. The next transferable audit frontier is therefore the element-row / condensate-column Jacobian block.
- This post-`J_mol` block classification is not a transferable solver rule yet. RGIE/PIPM should first source-prove the element-row/condensate-column coupling and show focused density improvement across both layers before adopting any analogous block actualization.
- The `J_ec` source decomposition proves that matched element-row/condensate-column entries use the same positive `stoich*n_old` variable basis, and the largest entry residuals are explained by old condensate density differences rather than stoichiometry, sign, or retained-column mapping.
- The old-density provenance is not yet transferable: the focused smoke still lacks FastChem post-seed condensate `n`, separately labelled post-iter0 condensate `n`, and FastChem-side cap/maxDensity/tau fields for the retained columns. RGIE/PIPM should treat `J_ec` old-density handoff as a trace frontier, not as a production rule.
- Replaying FastChem `J_ec` can close the block and, with cached `J_mol`, nearly close the projected linear solve, but it does not improve focused iter1 density across both layers. A transferable rule needs both source-level old-density provenance and two-layer density improvement.
- The widened old-state ladder removes the previous missing-field blocker for shared retained columns. It shows the old `n_c` handoff is dominated by seed/maxDensity/cap state, not by `J_ec` sign, stoichiometry, or retained-column mapping.
- Coherent replay of FC old `n_c`, lambda, tau, and `J_ec` is still not transferable: it improves the layer `60:-5` focused density but worsens layer `45:-10`. RGIE/PIPM should not port old-condensate handoff rules until cap/maxDensity state propagation closes both layers.
- The iter0 cap truth table shows that the Eq.13 total-inventory maxDensity source reaches the focused cap state for all shared retained rows, but the remaining old-state mismatch is mixed: retained/carryover rows dominate, with both-capped maxDensity-value and one-sided cap rows still present.
- Cap/maxDensity replay remains diagnostic-only. FC delta/maxDensity/post-iter0 variants improve layer `60:-5` but worsen or fail layer `45:-10`, so RGIE/PIPM should not port a seed, cap, maxDensity, or delta handoff rule without a single branch proof across both layers.
- The coherent iter0-to-iter1 transition audit shows that per-row cap provenance is not enough. RGIE/PIPM transfer must audit the full active row universe, all-active RHS burden, removed-condensate fold-in, global result scaling, refreshed activity/molecule state, and partition mapping as one transition.
- The all-active burden residual is dominated by KL-only `CH4(s,l)` in both focused layers, but coherent full-state replay is still layer-mixed: it worsens `45:-10` and improves only `60:-5`. This makes active-burden alignment a diagnostic frontier, not a production rule.
- A distinct FastChem post-`correctValues` refreshed all-active snapshot is still missing from the labelled trace; until that source stage is exposed, RGIE/PIPM should treat iter0-to-iter1 transition replay as incomplete provenance.
- The CH4 lifecycle audit narrows the active-burden signal: FastChem sees `CH4(s,l)` as a candidate but does not select it active, while KL keeps it as retained/Jac. Dropping CH4 removes most H/C burden residual but does not move focused iter1 density, so RGIE/PIPM should not port CH4 eviction without a source-proven update improvement.
- FastChem iter0 raw result scaling is now traced and differs from KL, but scaling remains a diagnostic signal because the CH4/burden replays are neutral for focused density.
- The exact FastChem post-`correctValues` refreshed all-active snapshot is now traced after element-density refresh, active-condensate `calcActivity`, and molecule `calcNumberDensity`, before old-state assignment and the next reduced Newton setup.
- The exact refreshed snapshot proves the previous iter1 pre-reduced proxy was numerically stale: exact-vs-proxy residuals are large in log-`n`, log-lambda, and log-activity even though the FastChem active row universe is unchanged.
- Exact refreshed-state replay is still not transferable. It worsens focused density at `45:-10` and improves only `60:-5`, and exact all-active burden residuals are larger than the previous proxy. RGIE/PIPM should treat the exact refreshed boundary as required provenance, not as a production handoff rule.
- The compact one-step residual is delta dominated after the branch/cap formula closes to roundoff. The delta-provenance compact decomposes selected rows into raw-result, global-scaling, local-clipping, mapping/index, removed analytic, and projected/focused coverage terms; available compact fields classify the selected rows mainly as raw-result dominated (`89`) with mapping/index secondary (`3`).
- This does not transfer as a solver rule. Standalone global-scaling and max-raw fields remain missing in the compact source rows, and old/delta compensation exists on `21` selected rows. RGIE/PIPM should audit raw directions, scaling, clipping, mapping, removed-row analytic deltas, and row coverage separately before comparing one-step residuals.
- The compact raw-result provenance audit confirms the raw reduced direction is the next meaningful frontier and rules out the audit solver backend: FC projected `J` + FC projected RHS reproduces FastChem raw results to mean absolute residuals `5.50e-12` and `7.12e-12`.
- This still does not transfer as a RHS, Jacobian, row-universe, or block-level rule. Row-level classification is mixed (`45` Jacobian-dominated, `41` mixed/unresolved, `3` mapping/index, `3` requiring both RHS and Jacobian), per-term RHS vectors are not emitted, and `J_mol` subterm arrays are not isomorphic to the shared projected element block.
- The compact Jacobian-subterm audit now projects `J_mol`/molecule outer-product and other `J_ee` subterms into the shared labelled system exactly enough for replay. Additivity closes to roundoff in both focused layers.
- This still does not transfer as a molecule, `J_ee`, or `J_ec` rule. Full FC `J` with KL RHS is the aggregate improvement, but `J_ee` and molecule-only swaps reduce raw residual only slightly, `J_ec` worsens, and most selected rows remain not explained by single-block swaps. RGIE/PIPM transfer needs full labelled block projection plus RHS-term trace completion before interpreting reduced directions.
- The coherent element-row audit shows the relevant aggregate Jacobian object is `[J_ec,J_ee]`, not either block alone. In the shared projection, replacing `[J_ec,J_ee]` exactly reproduces the full-FC-J + KL-RHS replay because `J_cc` and `J_ce` already match.
- The Schur-complement audit confirms the same coupling: replacing both `C=J_ec` and `D=J_ee` recovers the FC-like Schur matrix, but the effective RHS still differs and separated RHS term vectors are missing. RGIE/PIPM should treat this as a provenance boundary, not a Schur or element-row update rule.
- RHS-term tracing has been added as diagnostic-only instrumentation on both sides, but the current compact artifact predates those fields. Full-vector replay confirms the remaining post-coherent-Jacobian residual is RHS-side, while termwise attribution remains unavailable until the entrance smoke is regenerated with `condensate_rhs_terms` and `element_rhs_terms`.
- RGIE/PIPM should not transfer any RHS burden, inventory, activity, or removed-correction rule from the current result. Require labelled RHS term additivity and term-swap sensitivity before interpreting the remaining reduced direction.
- The fresh RHS-term entrance smoke removes the stale-artifact field blocker: FastChem and KL both expose 4 condensate RHS terms and 6 element RHS terms for `45:-10` and `60:-5`, with row-label counts matching the full RHS vector.
- Fresh scaled RHS additivity closes to roundoff, but unscaled RHS additivity fails on both sides. The compact audit therefore correctly refuses term sensitivity and Schur effective RHS term decomposition. RGIE/PIPM should not transfer an RHS inventory, atom, molecule-burden, all-active-burden, removed-correction, or condensate-row rule until the diagnostic decomposition closes both unscaled and scaled RHS identities.
- The scaled RHS is the reduced solve-space object in FastChem: `assembleRightHandSide` receives the row scaling factors and `solveSystem` consumes the scaled `rhs`. A scaled-term audit is therefore valid even while the unscaled bookkeeping remains unresolved.
- Scaled RHS term replay remains non-transferable as a single rule. The molecule-burden term helps layer `45:-10`, log-activity helps layer `60:-5`, all-active burden dominates Schur effective RHS norms but worsens direct replay, and cumulative closure requires multiple RHS terms. RGIE/PIPM should carry scaled solve-space RHS provenance, Schur amplification diagnostics, and source-space bookkeeping checks together rather than porting one RHS subterm.
- The RHS interaction/minimal-subset audit confirms that scaled term contributions add back to the full solve-space RHS contribution to roundoff, so the remaining residual is term-interaction/cancellation structure rather than missing term mapping.
- RGIE/PIPM should not transfer the best cancelling pair `log_activity + all_active_condensate_burden` as a rule. It improves both focused layers but does not close them, and its usefulness comes from cancellation between aligned condensate-row activity and anti-aligned element-row active-burden contributions.
- The only common closing subset is the full nonzero coherent RHS state: condensate-row `log_activity` and `log_tau_log_n_log_lambda` plus element-row total inventory, atom, molecule burden, all-active condensate burden, and removed-condensate correction. That is provenance for full RHS-state replay, not a reduced production rule.
- Source-state transfer implication: any RGIE/PIPM RHS comparison must keep condensate activity/tau state, gas molecule-density burden, active-condensate burden, removed-row correction, atom density, and total inventory together until a smaller source-coherent group closes both layers. Single-term and family-level RHS rules are rejected by the current smoke.
- The RHS source-state provenance audit shows the remaining RHS issue is not a formula mismatch for the terms whose source variables are present. Reconstructed scaled RHS terms close to roundoff on FastChem and KL in both focused layers, while expanded per-active-condensate burden, per-removed-correction rows, and separated total-density/epsilon fields remain trace-frontier items.
- RGIE/PIPM transfer should therefore treat RHS formula parity and RHS source-state handoff separately. Formula parity is sufficient for diagnostic interpretation; it is not a production rule because only the full coherent RHS source state closes both layers.
- The minimal source-state group that improves both layers is molecule state, but it does not close and should not be ported. The best mixed cancellation pair is condensate activity plus active-condensate burden state, but it also does not close. Full-source-state closure is diagnostic-only.
- The RHS lineage audit places the source-state handoff mismatch across multiple upstream stages, not at one promotable boundary. Condensate activity/complementarity and active-row universe differences are already present before candidate selection, removed-correction becomes material at RHS assembly / partition, and molecule/inventory lineage cannot yet be separated before RHS assembly because symmetric full element-vector, molecule-cache, total-density/epsilon, and atom-density fields are missing.
- RGIE/PIPM should not transfer an active-selection, post-refresh, partition, molecule-cache, hvector, or RHS-assembly rule from this smoke. The transferable work item is better instrumentation: emit symmetric post-refresh/pre-RHS source snapshots and expanded element source vectors before comparing partial handoffs.
- The preselection closure audit shows the full condensate catalog is materialized on both sides, so preselection divergence is not simply a full-row materialization issue. It is dominated by log-activity value differences with mixed donor and lnK/hvector components; threshold/candidate differences are material but not sufficient as a standalone rule.
- RGIE/PIPM should treat pre-seed complementarity as undefined for production transfer. Entry-seed and later complementarity can be audited, but pre-seed placeholder `n_c/lambda/tau` values should not be used as a solver-rule source.
- Inventory/atom and molecule lineage still need symmetric per-stage source fields before transfer: total-density/epsilon, atom density, full element vectors, molecule-cache refresh vectors, and mass-action constants. Removed correction should remain a partition/RHS-assembly provenance item with per-removed-row contribution expansion.
- FastChem final-tail or gas-refresh lifecycle details should not be ported without separate source and metric proof.
- Any donor or lnK adjustment not proven by source-level trace remains out of scope.
- The preselection activity-value audit closes the activity formula itself on both FastChem and KL. RGIE/PIPM should therefore separate formula parity from source-state parity: donor and lnK values can be wrong or mismatched even when `log_activity = lnK + donor_sum` closes exactly.
- Donor/lnK counterfactuals are not transferable as standalone rules. FC lnK with KL donor and KL lnK with FC donor both reduce candidate-set agreement against FastChem, while FC lnK plus FC donor is just full FastChem activity-state replay.
- Active-condensate burden should be decomposed by exact shared, FC-only, and KL-only row sets before interpreting magnitudes. In this smoke both shared-row `n_c` values and KL-only universe terms are large, and active-burden replay variants worsen the reduced residual, so RGIE/PIPM should not port an active-universe or shared-value rule.
- Removed correction remains a later partition/RHS-assembly provenance item and should not be mixed into preselection activity or active-burden causal claims.
- The candidate-to-active audit shows no separate selected-active filter beyond the activity threshold in the compact trace. FastChem and KL each have candidate rows equal to selected-active rows in the focused smoke; the mismatch rows are KL-only candidates, not phase/rank/order rejections after candidate agreement.
- RGIE/PIPM should not port a FastChem selected-active set or invent a KL eviction rule from this result. Removing KL-only selected rows matches FastChem row sets but worsens the coherent-Jacobian raw replay, so selected-row set parity alone is not a transferable solver rule.
- Future RGIE/PIPM selection comparisons need explicit phase/rank/order traces only if the source path actually uses them. For this compact FastChem path, the next provenance dependency remains donor/lnK activity-state alignment before the threshold.
- The coherent activity/burden pair is now source-visible in the compact solve. Activity-only and active-burden-only replays both fail, but the pair `log_activity + all_active_condensate_burden` reduces the raw residual sharply in both focused layers.
- This still does not transfer as a production rule. KL-only-row-only activity/burden edits fail, and the residual after the pair clearly persists into the other RHS source groups. RGIE/PIPM should treat the pair as a boundary condition for the next audit stage, not as a standalone transplant.
- If RGIE/PIPM wants to compare this frontier, it should keep activity and active burden coupled, then continue immediately into molecule, inventory/atom, removed-correction, and condensate complementarity rather than trying to freeze the pair as a solver rule.
- The post-activity-burden audit sharpens that frontier. With `baseline_AB = KL RHS + FC log_activity + FC all_active_condensate_burden`, condensate complementarity is the smallest remaining group that improves both focused layers, but closure still requires the full remaining quartet: complementarity + molecule + inventory/atom + removed-correction.
- RGIE/PIPM should not port complementarity alone even though it is the smallest common improver. Molecule and inventory/atom are almost perfectly cancelling after `baseline_AB`, removed correction is layer-specific, and the minimal common closing subset is still the full remaining RHS source state.
- The complementarity-provenance follow-up narrows that result further. Within the comparable compact rows, `activity_correction` is neutral, `log_tau` is the strongest single improving complementarity subcomponent, `-log_n` is secondary, and `-log_lambda` is neutral/worse. Full emitted complementarity still helps most as a one-group replay, but it is not source-clean enough to transfer.
- RGIE/PIPM should treat complementarity source formulas and complementarity replay separately. In the current compact smoke, source reconstruction does not self-close on comparable rows, and cross-state comparison is classified as row mapping mismatch because `Al(s)` and/or `CH4(s,l)` are missing at iter1 RHS entry and KL does not expose exact condensate rows for `post_correctValues_update` or `post_correctValues_refreshed_all_active_state`.
- After full complementarity, molecule-only and inventory-only replays each worsen badly, but their paired cancellation closes layer `60:-5` and nearly closes `45:-10`. The remaining transferable lesson is therefore not "use log_tau" or "use complementarity", but "keep complementarity coupled to the later molecule/inventory provenance until a smaller source-proven rule closes both layers."
- The source-closure follow-up resolves the formula question. Using exact RHS-entry `n_old`, `activity_correction_old`, `tau`, and `log_tau`, FastChem and KL complementarity formulas now self-close to roundoff. The dominant exact divergence is no longer generic complementarity replay but `tau_seed_rule_mismatch`, already visible at reset / immediate entry seed on the common exact-mapping rows.
- RGIE/PIPM still should not port a tau-seed or standalone `log_tau` rule. Even after source-clean complementarity closure, `log_tau` alone leaves residuals `0.86793` and `0.99908`; layer `60:-5` still needs the later molecule/inventory cancellation pair, and layer `45:-10` still needs an additional removed-correction tail.
- The transferable result is narrower and cleaner: use exact complementarity source closure and tau-seed lineage as provenance checks, then continue directly into molecule/inventory and removed-correction coupling. Do not promote a complementarity or `log_tau` rule from this compact smoke.
- The tau-seed formula audit makes the source split explicit. FastChem seed tau is support/reference-element aware and density-scaled: `cond_tau * total_element_density * epsilon(reference_element)`. KL seed tau is schedule-only: `tau_scale * exp(epsilon)`.
- This still does not transfer as a rule. Condensate-scalar-only, total-density-only, and reference-element-only replays do not close, and the full FC tau replay still stops at `0.86793` / `0.99908` before the later complementarity, molecule/inventory, and removed tails.
- RGIE/PIPM should therefore transfer only the source-formula checklist: verify whether the target branch wants a schedule-only tau, a density-scaled tau, or a support/reference-element-aware tau. Do not port a FastChem tau seed rule from this audit.
- The post-complementarity tail audit tightens the remaining transfer boundary. After fixing `baseline_ABC = KL RHS + FC log_activity + FC all_active_condensate_burden + FC full complementarity`, molecule-only and inventory-only replays each worsen badly, while the paired `molecule + inventory_atom` replay closes `60:-5` and reduces `45:-10` to `0.08475871276818528`.
- Molecule and inventory/atom should therefore be treated as a cancellation pair, not as standalone transferable rules. Their contribution vectors remain almost perfectly anti-aligned after complementarity (cosines `-0.9997438633784276` / `-0.9997471855058516`, cancellation indices `0.9886099574272349` / `0.9870268652530958`).
- Removed correction is only the remaining layer-45 tail after that pair. Adding removed closes `45:-10` to `8.199347373900107e-12` while leaving `60:-5` at `1.2971603589242228e-11`.
- RGIE/PIPM should not port the molecule/inventory pair or removed tail as production behavior from this smoke. Carry forward only the provenance checks: exact molecule-burden aggregate closure from `molecule_density_provenance`, exact atom-density closure, the unresolved total-inventory source split, and the missing per-removed contribution vectors.
- The stricter molecule/inventory/removed source audit sharpens that boundary. Molecule burden closes exactly from the emitted iter1 molecule caches, and atom density closes exactly from the emitted iter1 element state, but the independently emitted gas diagnostics do not reconstruct the RHS total-inventory term.
- RGIE/PIPM should therefore not treat the post-complementarity cancellation as a standalone molecule rule or a standalone total-inventory rule. At strict source-clean granularity it requires the full coherent gas-state bundle.
- The layer-45 removed tail also remains unresolved at strict source-clean granularity. The current compact trace still lacks an exact emitted per-removed RHS contribution rule, so RGIE/PIPM should keep removed-tail replay diagnostic-only.
- The molecule timing-resolution audit reconciles FastChem replay before interpreting KL timing. FastChem reconstructed molecule cache from the emitted full element vector, hvector/mass-action ledger, gauge, and clipping fields matches the direct FastChem cache to roundoff at both post-iter0 refresh and iter1 RHS/Jacobian entry, so the prior reconstructed-vs-direct discrepancy was a compact replay mapping bug rather than a FastChem formula or gauge failure.
- Exposed KL cached-stage snapshots from `gas_only_final`, `post_initial_activity_maxdensity_scan`, and `post_selectActiveCondensates_reset` do not recover the molecule/inventory cancellation; later KL post-correctValues and pre-partition molecule-stage vectors remain unexposed, and KL computes the cache inline at RHS/Jacobian assembly.
- RGIE/PIPM should transfer this as a guardrail only: require a source-clean cached gas/molecule snapshot before comparing molecule timing. Do not port a FastChem cached molecule state, hvector conversion, density gauge, inventory normalization, or removed-tail rule from this result.
- The later KL molecule snapshot audit now emits the previously missing `post_correctValues_update`, exact post-refresh, `iter1_pre_partition`, and `iter1_RHS_assembly_entry` source bundles. Availability is no longer the blocker, but the result remains negative: all later KL molecule snapshots replay at the KL inline RHS residuals (`36.907067382036104` at `45:-10`, `50.419424423672645` at `60:-5`) and do not recover molecule/inventory cancellation.
- RGIE/PIPM transfer implication is unchanged but sharper: KL inline RHS recomputation is the best KL-native exposed stage and is still insufficient. Keep FastChem cached molecule state diagnostic-only; no molecule timing, hvector, density-gauge, inventory, removed-tail, guarded-mode, or production solver rule should be ported from this audit.
- The later-stage distinctness audit resolves why the later KL carries are all equivalent: the emitted `post_correctValues_update`, exact post-refresh, `iter1_pre_partition`, and `iter1_RHS_assembly_entry` molecule source bundles are numerically identical on both focused layers. Pairwise differences in `u`, atom/full-element vector, and molecule cache are all zero.
- Freeze-and-carry does not create a useful diagnostic candidate. Carrying any later KL cache to iter1 RHS exactly reproduces inline RHS residuals, remains far from the FastChem cached molecule state, and does not recover cancellation. RGIE/PIPM should interpret this as a collapsed KL-native snapshot, not as a timing lever.
- The gas-refresh snapshot audit finds distinct KL diagnostic gas states, but they are not transferable. `post_gas_recoupling_atomic_element_species_state` and the proxy `gas_replay_final_atomic_element_species_state` differ from the collapsed later KL snapshot, yet they do not improve the molecule/inventory replay and remain far from the FastChem cached molecule state.
- The requested iter1-pre-partition gas-recompute proxy remains unavailable because `fastchem_target_donor_replay_from_gas_replay_final` contains a non-finite `e-` atomic entry. RGIE/PIPM should treat this as a trace frontier, not as a solver rule.
- Transfer implication: a distinct gas refresh alone is insufficient. A target branch must expose a source-clean coupled molecule snapshot that actually improves the selected-row replay before any molecule-cache, coupled-loop, inventory, or removed-tail rule is considered.
- The molecule input-vector provenance audit moves the blocker upstream of the molecule cache. Reusing direct FastChem mass-action constants with KL input vectors still fails at the KL residual scale, while FC input-vector reconstruction reproduces the direct FC molecule boundary.
- Top atomic/full-element input-vector residuals are dominated by FastChem fixed-by-condensation elements, with `e-` also a large non-fixed residual. RGIE/PIPM should therefore treat molecule-cache mismatch as a downstream symptom of fixed-element/input-vector handoff mismatch, not as a standalone molecule timing or hvector rule.
- Transfer implication: before comparing molecule caches, a target branch must source-prove the fixed-element/full-element vector handoff that feeds molecule reconstruction. Do not port molecule timing, gas-refresh, or fixed-element behavior from this diagnostic result.
- The compact fixed-element handoff audit sharpens that blocker: the earliest exposed fixed-subset divergence is already present before `post_selectActiveCondensates_reset`, and the dominant component is the fixed-element values themselves rather than KL non-fixed values or standalone electron replay.
- Fixed-element-only value substitution strongly improves molecule reconstruction, but it does not produce a coherent closing RHS replay. RGIE/PIPM should treat this as source provenance only, not as permission to transplant a partial fixed-element rule.
- `e-` remains a large residual and FC fixed+electron replay improves molecule density further, but electron-only replay is not competitive and no separable KL mask-consumption or `phi` / `degree_of_condensation` handoff field is emitted at the molecule input-vector stage. Transfer implication remains unchanged: do not port fixed-mask, electron, `phi`, or degree rules from this audit.
- Full FC input vector / direct FC molecule replay remains the only molecule-side upper bound. Therefore the transferable lesson is still the guardrail: source-prove the full fixed-element handoff before comparing molecule caches, and keep the result diagnostic-only until a coherent smaller rule closes without the FC upper bound.
- The overwrite-source follow-up audit moves one step deeper without changing that guardrail. At the cached full-element boundary, fixed-by-condensation rows are full overwrites from local `elem_densities_new[i]` in `CondensedPhase::calculate`, with the focused reduced branch pointing to `CondensedPhase::correctValues`.
- This is still not transferable as a production handoff rule. Overwrite-only replay remains the decisive improver, but additive source components are still not emitted cleanly (`free_atomic_gas_component`, `condensed_or_fixed_correction_component`, `total_inventory_component`, `electron_specific_component`), and the prior carry-forward local `full_element_densities_before_write[i]` is not yet surfaced in the compact artifact. FC overwrite plus FC electron still fails the selected-row replay catastrophically.
- RGIE/PIPM should therefore carry only the narrower provenance result: mismatch is dominated by overwrite values themselves, overwrite source is further resolved, and the result remains diagnostic-only until a smaller source-clean component closes both focused layers without the full FC cached vector.
- The KL materialization-boundary audit clarifies the next missing object. The diagnostic gas replay does produce a true post-recoupling atomic state, but the later molecule input used at `iter1_RHS_assembly_entry` is rebuilt inline from `_assemble_fastchem_reduced_update`, not carried from the gas-recoupling output.
- This is source-visible and measurable: gas-recoupling `u` differs materially from iter1 molecule-input `u`, while the pre-gas current-state proxy matches the iter1 molecule input exactly on both focused layers. The iter1 cache is marked `cache_is_computed_inline = true` and `cache_is_carried_from_earlier_stage = false`.
- RGIE/PIPM should therefore treat the missing transfer object as an explicit materialization boundary. Do not port gas-recoupling output, fixed-element gas replay, or fixed-mask semantics into later molecule reconstruction unless the target branch proves that boundary with an emitted carried full-element vector and a real molecule-input consumer.
- The exact fixed-row subspace audit did not pass its trace gate. Diagnostic-only `CondPhaseSolver::newtonStep` trace emission for `exact_same_iteration_fixed_row_reduced_system` was added, but the rebuilt entrance-smoke trace still has no `condensed_phase_exact_fixed_row_reduced_system` records for layers `45:-10` or `60:-5`.
- RGIE/PIPM should not use the older emitted labelled analogue as a production transfer basis for fixed-row closure or Schur reasoning. The missing exact local remains `jacobian` in `CondPhaseSolver::newtonStep`, plus exact same-iteration row and column labels.
- Transfer implication: fixed-row scalar provenance through `b_i`, `z_i`, global scaling, row scaling, and `J_ii` remains useful diagnostic provenance, but coherent fixed-row subspace replay is blocked until exact same-iteration labelled matrix rows are available. Do not port fixed-row RHS, Jacobian, outside-coupling, Schur-complement, full-system, guarded-mode, or production solver rules from this audit.
- The trace-repair follow-up adds a v2 probe and exact fixed-row compact matrix emitter at the post-`solveSystem` source point, and a clean import check proves those records can be emitted for the exact focused layer inputs. The requested phase2 entrance-smoke raw traces still lack the v2 records while retaining the older iter1 full-system analogue.
- RGIE/PIPM transfer implication: treat this run as a trace/build-path failure, not a chemistry or solver conclusion. Do not run or interpret coherent fixed-row subspace replay until the requested entrance smoke itself contains the v2 probe, exact fixed-row matrix record, non-empty fixed submatrices, fixed RHS entries, solver result, and exact labels/indices.
- The import-repair follow-up fixes the phase2 build/import path: the requested
  phase2 smoke now loads the rebuilt `fastchem/python` extension with checksum
  matching the root extension, and the raw traces carry
  `exact_fixed_row_subspace_trace_v2` on both legacy iter1 full-system records
  and v2 exact fixed-row records.
- The v2 gate now passes on both focused layers. Exact fixed rows-by-all-columns,
  all-rows-by-fixed-columns, fixed RHS entries, solver result, and row/column
  labels are present for `45:-10` and `60:-5`; fixed-row equation closure is
  roundoff-level in the compact replay.
- RGIE/PIPM transfer implication: exact same-iteration fixed-row matrix closure
  is now available as a diagnostic provenance check. It is not a transferable
  production rule because coherent subspaces S0-S5 do not produce a promotable
  two-layer closure; keep fixed-row RHS, Jacobian, outside-coupling, Schur, and
  full-system coherence claims diagnostic-only.
- The fixed-row subspace tail-context reconciliation proves that this remains
  true after restoring the known post-complementarity molecule/inventory context.
  The common selected-row code path reproduces `baseline_ABC`, the destructive
  molecule-only and inventory-only legs, the molecule+inventory closure at
  `60:-5`, and the molecule+inventory+removed closure at `45:-10`.
- RGIE/PIPM transfer implication: do not replace the exact emitted FC molecule
  RHS term with fixed-row subspace molecule sources. S0-S5 remain numerically
  indistinguishable from the KL-current molecule source in the proven tail
  context, while the FC cached input-vector reconstruction remains destructive.
  Carry the molecule RHS term, gauge-normalized inventory/atom, and removed tail
  as one diagnostic provenance bundle only.
- The molecule RHS artifact reconciliation resolves why the cached-input path
  looked destructive. The emitted FastChem molecule RHS term is reconstructable
  to roundoff from the enriched FastChem molecule cache and full-element vectors
  if the burden is converted directly into FastChem scaled solve space. The
  destructive result comes from the legacy KL-reference burden-ratio conversion,
  a source-space versus solve-space convention mismatch.
- RGIE/PIPM transfer implication: source-prove molecule burden in the target
  branch in the same solve-space convention before comparing or replaying it.
  Do not port the legacy ratio conversion, and do not treat the exact FastChem
  molecule RHS term as a production rule.
- The convention-safe S0-S5 rerun applies that requirement directly. The
  FastChem-scaled builder closes the emitted molecule RHS identity gate on both
  focused layers, and the legacy KL-reference ratio remains catastrophic as a
  negative control.
- RGIE/PIPM transfer implication: even after using the correct solve-space
  molecule RHS convention, fixed-row subspace molecule sources remain
  indistinguishable from the KL-current molecule source. A target branch should
  carry this as a diagnostic convention check, not as a fixed-row subspace or
  molecule-source rule.
- The full-element subset audit attributes the FC cached-input molecule RHS
  recovery to a broad fixed-elements plus electron subset. Fixed elements
  without the FastChem electron are destructive, electron-only is insufficient,
  and non-fixed elements do not move the replay off the KL-current scale.
- RGIE/PIPM transfer implication: if a target branch audits this path, it must
  keep fixed-element values and electron state coupled in the cached molecule
  input. This is still diagnostic-only because it is a broad cached-vector
  provenance bundle and layer `45:-10` still needs the removed tail.
- The charged-vs-neutral molecule group audit refines that attribution without
  promoting a rule. Neutral molecules dominate the selected-row fixed+electron
  recovery: the neutral fixed+electron term reproduces the FC molecule/inventory
  behavior on both focused layers, and layer `45:-10` closes when exact removed
  is added. Charged molecules explain the catastrophic fixed-only with
  KL-electron replay, so electron remains an essential secondary state, but the
  charged group is not the selected-row recovery carrier.
- RGIE/PIPM transfer implication: keep fixed elements, electron state, neutral
  molecule burden, inventory/atom, and the layer-45 removed tail as a diagnostic
  provenance bundle. No small charged, neutral, ion-support, electron, or
  fixed+electron production rule is promotable from this pass, and the KL-native
  fixed/electron materialization fields remain missing.
- The KL-native fixed+electron materialization audit confirms that missing
  boundary. No emitted KL source reproduces the FC fixed+electron bundle:
  current/post-correctValues/RHS-entry fixed values stay at the KL-current scale,
  gas-recouping fixed values stay non-closing even with the FC electron, and FC
  fixed values paired with the best KL-native electron remain catastrophic.
- The best KL-native electron is gas-recouping, but its log residual from the FC
  electron is still `59.59` / `74.88`, and it does not reach the FC
  fixed+electron residual scale. KL fixed candidates remain tens of log units
  from the FC fixed bundle and still lack fixed-element bookkeeping plus
  `degree_of_condensation` / `phi` analogues.
- RGIE/PIPM transfer implication: the FC fixed+electron bundle is source-proven
  as a diagnostic upper bound only. A target branch would need to emit a real
  same-boundary fixed+electron materialization object before any fixed,
  electron, gas-recouping, overwrite-derived, or molecule-input rule can be
  considered.
- The reduced-Newton fixed-vector follow-up materializes the missing fixed-side
  object from existing diagnostic write-site fields. For every fixed element on
  the focused layers, the compact records the old fixed value, reduced Newton
  result slot, row/global scaling, clipped delta, update factor, and
  `elem_number_dens_new` value from `CondensedPhase::correctValues`.
- RGIE/PIPM transfer implication: the fixed side is no longer the blocker at
  this boundary. KL reduced-Newton fixed values plus the FC electron reproduce
  the FC fixed+electron upper bound, and layer `45:-10` closes with exact
  removed. Same-boundary KL reduced fixed plus post-`correctValues` electron
  remains catastrophic, while gas-recouping electron improves but remains far
  from closure. The remaining transfer frontier is therefore electron
  materialization, not fixed-value materialization.
- The electron materialization provenance audit confirms that frontier. The
  FC cached electron is source-marked at
  `iter0_post_correctValues_full_element_vector_before_molecule_refresh` and is
  consumed by molecule reconstruction, but the focused trace does not emit the
  FastChem electron equation branch locals (`alpha`, `beta`,
  `positive_ion_density`, `negative_ion_density`, branch choice, or Newton
  electron result). The best KL-native electron is still gas-recouping /
  post-adoption gas recompute, and it remains tens of log units from FC.
- RGIE/PIPM transfer implication: do not port an electron carry, scalar gauge
  offset, charge-neutrality reconstruction, gas-solver replay, or clipping rule
  from this audit. A target branch must first emit a same-boundary,
  source-proven electron materialization object; until then the FC electron is
  a diagnostic upper bound only.
- The FastChem electron-solver trace now source-proves the FC side of that
  object. The cached electron is produced by the singly-ion analytic equation
  in `GasPhase::calculateSinglyIonElectrons`, with emitted `alpha` and `beta`
  reconstructing the FC cached electron exactly on both focused layers.
- RGIE/PIPM transfer implication: the equation itself is not the missing FC
  provenance anymore; the transfer blocker is carrying the source-proven gas
  electron to the same fixed+electron materialization boundary. Do not port a
  carry rule until the target branch emits the same-boundary electron lineage
  and proves the KL candidate closes.
- The KL-side FastChem-style reconstruction applies that exact singly-ion
  equation to emitted KL candidate source states with the FastChem ion list.
  Current/post-`correctValues` and gas-recouping/post-adoption reconstructions
  remain far from the FC electron, and the selected-row replay with KL reduced
  fixed does not close. The alpha/beta decomposition attributes the dominant
  residual to the beta / ion-correction source state on both focused layers.
- RGIE/PIPM transfer implication: do not port a beta correction, alpha/beta
  reconstruction, or electron materialization rule from this pass. A target
  branch should first emit same-boundary KL alpha/beta inputs with complete
  ion-support vector logs and prove that the reconstructed electron reaches the
  FC fixed+electron residual without borrowing FC non-fixed inputs.
- The beta ion-correction attribution compact expands the negative-ion side
  directly. The FastChem identity gate closes for all `50` beta-side ions on
  both focused layers: `sum beta_i` matches beta and
  `sqrt(alpha / (1 + beta))` reproduces the cached FC electron with zero log
  residual. `KL alpha + FC beta` reaches the known FC fixed+electron residual
  scale, while `FC alpha + KL beta` remains catastrophic, so beta is the
  controlling electron term.
- Per-ion residuals have large dominant contributors (`Al1F4-`, `F6S1-`,
  `F5S1-`, plus layer-dependent metal/carbon ions), but the closing behavior is
  not a small KL-native rule. FC non-fixed and full FC cached-input hybrids are
  diagnostic upper bounds only. Same-boundary reduced-fixed alpha/beta
  candidates still lack non-fixed support logs for global element indices `2`,
  `13`, and `19` (`Ar`, `He`, `Ne`) from the diagnostic KL-native
  fixed+electron bundle overlay.
- RGIE/PIPM transfer implication: do not port a beta correction,
  support-element hybrid, alpha/beta swap, guarded mode, or electron rule. A
  transfer candidate must first source-prove a KL-native same-boundary
  non-fixed/electron state that closes without FC borrowing.
- The same-boundary vector repair compact fixes the diagnostic construction
  bug behind the missing `Ar`, `He`, and `Ne` support logs. KL candidate
  vectors were complete but in KL element order with `e-` last; FastChem-style
  alpha/beta stoichiometry expects FastChem global indices with `e-` at index
  `0`. Canonicalizing by element label repairs all `28` element logs for
  candidates `A-G`, and `Ar`, `He`, and `Ne` are present and finite on both
  focused layers.
- After repair, KL reduced-fixed plus KL current/gas/post-adoption non-fixed
  candidates all match the prior KL reduced-fixed plus FC non-fixed diagnostic
  upper-bound residuals, not the old catastrophic index-order result: beta
  log1p residuals are `0.6005995827798536` and `0.8753728757415544`, and
  electron log residuals are `0.2025060253072546` and
  `0.32999751763544793`. Full FC cached input still closes beta to roundoff.
- RGIE/PIPM transfer implication: the missing-field blocker is repaired, but
  replacing `Ar`, `He`, `Ne`, beta-supporting non-fixed elements, or all
  non-fixed elements does not move the repaired residual. Do not port a beta
  correction, non-fixed support hybrid, alpha/beta swap, guarded mode, or
  electron rule from this result; beta attribution remains mixed or
  inconclusive.
- The repaired alpha/beta tail replay resolves the selected-row relevance of
  that residual. With the canonical gate passing, the repaired same-boundary
  KL reduced-fixed + FastChem-style alpha/beta electron reconstructs a molecule
  RHS that matches the exact emitted FC molecule RHS on selected rows. In the
  common post-complementarity tail, candidate molecule RHS plus
  gauge-normalized inventory/atom gives `0.08475871276713746` at `45:-10` and
  `1.3947630928999967e-11` at `60:-5`; exact removed then closes layer
  `45:-10` to `9.001471173380398e-12`.
- Raw-result implication: molecule-only remains destructive in the coherent
  FC-Jacobian context, but candidate molecule plus inventory/atom recovers the
  same selected-row cancellation. The exact beta residual is nonblocking for
  selected-row closure but remains a provenance residual. Keep this as
  diagnostic provenance only; do not port an electron/beta rule, molecule RHS
  rule, inventory rule, removed-tail rule, or guarded mode.
- The integrated repaired-alpha/beta ladder extends that result beyond the
  common post-complementarity tail. The canonical alpha/beta gate passes in
  FastChem order with `e-` at index `0`, while raw KL source vectors are
  explicitly recorded as ExoGibbs-order inputs with `e-` last before
  conversion. In the full semantic ladder, repaired molecule plus
  gauge-normalized inventory/atom closes layer `60:-5` and leaves layer
  `45:-10` at the known removed-tail residual; exact removed closes layer
  `45:-10`. In the labelled reduced-system solve context, molecule plus
  inventory alone does not close (`2.9375647869000923` / `1.9877020617705434`);
  closure requires the full tail bundle with tau/complementarity.
- RGIE/PIPM transfer implication: the repaired same-boundary alpha/beta
  electron is useful as a diagnostic provenance check, not a production source
  rule. Exact beta mismatch is selected-row nonblocking but full-vector
  relevant, so target branches should keep canonical index-order checks,
  molecule/inventory cancellation, tau/complementarity state, and removed-tail
  provenance coupled until a smaller source-clean rule closes both selected-row
  and raw labelled solve contexts.
- The positional-boundary follow-up resolves the four remaining unknown
  diagnostic matrix reads. The fixed-row result-entry and reduced-Newton
  result-slot compacts were reading element-element Jacobian subterm diagonals;
  those matrices are ordered by emitted `element_labels` in
  `element_element_jacobian_subterms` local element-label order, not by reduced
  solve-space slot order and not by a global element index. The old
  `element_index` name was ambiguous; the helper now uses
  `subterm_element_pos` and emits basis-guard metadata.
- RGIE/PIPM transfer implication: keep this as a diagnostic indexing guardrail.
  The resolution removes an audit ambiguity but does not justify porting a
  Jacobian diagonal, fixed-row, subterm-order, guarded-mode, or production
  solver rule.
- The repaired alpha/beta coherent-bundle audit shows the remaining useful
  signal is bundle coherence, not a promotable beta/electron source rule.
  Repaired molecule RHS is destructive alone, inventory/atom is destructive
  alone, and the pair recovers the selected-row tail only inside the existing
  coherent activity/burden/complementarity context. Layer `45:-10` still needs
  exact removed, while layer `60:-5` closes with molecule plus inventory/atom.
- Raw solve implication: the full tail bundle with removed/tau/complementarity
  is nonblocking relative to full FC at roundoff-scale l2 differences, but the
  beta mismatch still affects full-vector provenance outside selected rows.
  RGIE/PIPM should keep alpha/beta electron reconstruction, molecule RHS,
  inventory/atom, complementarity/tau, and removed-tail provenance coupled
  until a broader smoke and source-clean smaller rule are available.
- The phase2 broad-smoke generation blocker was repaired diagnostically. The
  failing assertion had compared raw element labels (`Al`, ..., `e-`) to
  formula-like element species (`Al1`, ..., `e1-`). The guard now parses species
  labels and checks formula equivalence plus the ExoGibbs e-last identity block
  in `formula_matrix_gas[:, :n_elem]`. Running the phase2 driver on `30:-10`,
  `45:-10`, `60:-5`, `75:-5`, and `90:-5` now emits the broad `/tmp`
  artifacts.
- The repaired alpha/beta compact remains limited by focused specialized
  source-state dependencies. The broad dependency graph identifies the current
  blockers: focused selected-row delta/raw provenance, reduced-slot provenance
  with only `--trace-45` / `--trace-60`, and focused-only molecule-vector,
  coherent-bundle, and electron-reconstruction compacts. No broad-layer
  repaired-alpha/beta conclusion is drawn from the multi-GB phase2 artifacts.
- The direct broad evaluator reads `/tmp/exogibbs_phase2_broad_smoke.json`
  without those focused compacts. It finds broad FastChem reduced-system,
  electron alpha/beta, molecule, inventory/removed, row scaling, RHS/Jacobian,
  and solver-result fields. The phase2 driver now has a diagnostic
  `repaired_alpha_beta_source_state_snapshot` hook for future broad runs, but
  the existing broad artifact and derived snapshot still lack the same-boundary
  KL non-fixed vector and selected-row delta/raw-result mapping. Snapshot
  emission is therefore incomplete.
- RGIE/PIPM transfer implication: do not generalize the repaired alpha/beta
  coherent bundle from two layers. Transfer the formula-equivalent species
  boundary guard, but wait for regenerated broad selected-row delta/raw,
  RHS/source, molecule, reduced-slot, electron, and coherent-bundle compacts
  under the same canonical vector gate before considering any production rule.
- A fresh broad phase2 rerun with the embedded diagnostic
  `repaired_alpha_beta_source_state_snapshot` now emits the same-boundary KL
  non-fixed values for all five broad cases from `iter1_RHS_assembly_entry`.
  Canonical mapping, reduced fixed values, molecule RHS inputs,
  inventory/atom inputs, exact removed inputs, and raw labelled solve fields are
  present in the fresh artifact. The remaining transfer blocker is the missing
  selected-row mapping (`row_position`, `result_index`, row label, and delta
  classification) used by the primary selected-row metric.
- RGIE/PIPM transfer implication: the same-boundary non-fixed source hook is
  useful diagnostic plumbing, but broad repaired-alpha/beta replay remains
  incomplete until selected-row mapping is emitted. Do not port an electron,
  beta, molecule/inventory, selected-row, guarded-mode, or solver rule from the
  broad smoke yet.
- The selected-row mapping is now emitted for all five broad cases. It is
  sourced from the existing focused row definition
  `PRESELECTION_ACTIVITY_FOCUSED_NAMES` and labelled reduced-system
  solver-result mappings, not from a new row definition. Focused validation
  against the raw-result provenance compact passes for `45:-10` and `60:-5`.
- RGIE/PIPM transfer implication: selected-row provenance is now available as a
  diagnostic bridge, but broad repaired-alpha/beta replay remains mixed because
  the direct compact still lacks the convention-safe molecule-density and
  molecule/inventory replay implementation outside the focused compact stack.
  Do not promote an electron, beta, molecule/inventory, selected-row, guarded
  mode, or solver rule.
- The contracted formula-matrix boundary audit adds a diagnostic guard for a
  future regression class: if `contract_formula_matrix` returns fewer formula
  rows, then `formula_matrix_gas.shape[0]` is not a valid species or hvector
  molecule boundary. The current phase2 parity paths are guarded: species and
  hvector boundary uses are covered by the full element-count assertion, while
  contracted-basis uses propagate `element_mask` and `element_names`.
- RGIE/PIPM transfer implication: keep contracted formula rows and species
  boundaries separate. Port the guardrail and metadata requirement only; do not
  port a production formula-matrix, species-boundary, molecule, RHS, electron,
  inventory, guarded-mode, or solver rule from this audit.
- The direct repaired alpha/beta broad replay is now implemented for all five
  broad source-state cases. It constructs the repaired candidate in FastChem
  canonical e-first order, runs FastChem-style alpha/beta electron
  reconstruction, builds molecule densities, applies the convention-safe
  scaled molecule RHS, and replays molecule/inventory/removed tails in the
  labelled raw solve context.
- RGIE/PIPM transfer implication: this is still a non-promotion result. The
  broad embedded selected-row mapping is a 10-row diagnostic projection and
  does not reproduce the older focused selected-row reference metric; the
  direct regression gate records `selected-row mapping` as the differing term.
  Full-vector outside-selected residuals remain material, and removed tails
  remain separate by layer. Do not port an alpha/beta electron rule,
  molecule/inventory pair, selected-row mapping, removed-tail rule, guarded
  mode, or solver behavior from this replay.
- The selected-row reconciliation compact separates the focused reference
  metric from the embedded broad 10-row projection. Focused 45/60 references
  pass under the focused metric, while the broad projection remains
  non-closing. Focused-compatible mappings cannot be built for `30:-10`,
  `75:-5`, or `90:-5` because the raw-result provenance source is focused-only.
- RGIE/PIPM transfer implication: no selected-row rule should be transferred.
  Carry metric lineage with any future comparison, require a broad
  focused-compatible raw-result provenance source before generalization, and
  keep outside-selected/full-vector residual attribution as the remaining
  blocker.
- The broad raw-result provenance compact now generalizes the focused metric
  construction where the required delta-provenance rows exist. It validates
  `45:-10` and `60:-5` against the older focused raw-result compact, but it
  cannot emit focused-compatible mappings for `30:-10`, `75:-5`, or `90:-5`
  because the delta-provenance compact lacks selected rows for those cases.
- RGIE/PIPM transfer implication: broad repaired alpha/beta remains
  metric-inconclusive. Do not transfer an electron, beta, molecule/inventory,
  selected-row, removed-tail, guarded-mode, or solver rule until broad
  delta-provenance inputs exist and focused-compatible selected-row closure
  holds across all requested broad cases.
- A broad delta-provenance candidate now exists for all five broad cases from
  labelled broad solver-result mappings, but it is not accepted as
  focused-compatible. It fails focused `45:-10`/`60:-5` validation against the
  old delta compact, so it cannot be used to generalize the focused selected
  row metric.
- RGIE/PIPM transfer implication: solver-result mappings alone are not enough
  to transfer the focused selected-row metric. Future broad transfer needs the
  one-step old/new `correctValues` source rows and focused delta-selection
  context, or an explicitly validated equivalent, before any repaired
  alpha/beta replay conclusion can be generalized.
- A broad one-step compact extract with the focused schema now exists for all
  five cases. Rebuilding broad delta provenance from that extract still fails
  focused `45:-10`/`60:-5` validation, and the extract reports missing
  FastChem raw-result fields on rows outside the emitted one-step trace subset.
- RGIE/PIPM transfer implication: keep the broad one-step extract as
  diagnostic provenance only. Do not transfer repaired alpha/beta conclusions
  until the broad one-step source validates against the focused delta metric or
  the stale-artifact difference is source-proven.
- The stale-artifact difference is now source-proven. A fresh focused one-step
  extract generated with the same broad extractor path differs from the old
  focused one-step compact, and the repaired diagnostic FastChem row choice
  exposes retained iter1 raw-result rows instead of later eliminated
  same-condensate records. Broad delta provenance validates against the fresh
  focused delta reference, and raw focused-compatible mapping rows are available
  for all five broad cases.
- RGIE/PIPM transfer implication: do not transfer the historical focused
  repaired-alpha/beta closure values into RGIE/PIPM. They belong to the stale
  focused metric artifact. A current focused rebaseline is required before any
  broad selected-row closure or repaired alpha/beta rule can be interpreted for
  transfer. The embedded broad 10-row projection remains an outside-selected
  probe only.
- The direct broad evaluator now consumes the accepted broad raw-result
  provenance mapping under the current fresh focused-compatible metric. All
  five broad cases have mapping rows, but the selected rows are one-sided or
  mapping/index-only, leaving `0` shared-projected numeric rows in each case.
- RGIE/PIPM transfer implication: broad repaired-alpha/beta generalization is
  partially metric-inconclusive, not promotable. Carry the one-sided row
  attribution and stale-focused rebaseline result as diagnostics only. Do not
  transfer an electron, beta, molecule/inventory, selected-row, removed-tail,
  guarded-mode, row-scaling, lifecycle, RHS, KL/FastChem behavior, preset, or
  solver rule until current fresh focused-compatible numeric closure is defined
  and passes on shared selected rows and the outside-selected broad projection
  residual is explained.
- One-sided attribution now classifies the current focused metric as
  row-universe/mapping dominated: all `12` selected rows are one-sided and no
  shared numeric focused rows exist. Where focused source-stage lineage is
  available, `CH4(s,l)` traces to candidate/activity-threshold mismatch.
- RGIE/PIPM transfer implication: the next transfer-relevant diagnostic is
  row-universe/mapping lineage, not repaired alpha/beta. Do not port repaired
  alpha/beta as a current blocker solution; first source-prove candidate,
  active-set, partition/reset/update, and labelled reduced-system mapping
  behavior under the fresh focused-compatible metric.
- Deep attribution now rules out label normalization as the sole blocker and
  classifies the frontier as mixed row-universe/mapping provenance:
  partition-split divergence, focused-layer activity-threshold crossing, and
  missing broad case-keyed candidate/active/reset traces all remain in play.
- RGIE/PIPM transfer implication: carry the attribution as a non-promotion
  diagnostic. The next broad artifact should emit case-keyed candidate,
  active-set, reset, lifecycle/update, and labelled reduced-system row tables
  before any RGIE/PIPM transfer decision is revisited.
- The latest broad case-keyed attribution emits selected-row lineage tables for
  all current one-sided rows, but it also proves the blocker is still missing
  trace coverage: broad `candidate_set`, `active_set`,
  `post_selectActiveCondensates_reset`, and `partition_split_before` row tables
  are unavailable. Broad-only `CH4(s,l)` rows have KL activity/threshold
  evidence but lack FastChem activity/candidate evidence, and partition rows
  lack the pre-partition active table needed to prove their cause.
- RGIE/PIPM transfer implication: do not port repaired alpha/beta, row
  selection, lifecycle, labelled reduced-system, or partition behavior. The
  next transfer-relevant artifact is broad case-keyed candidate/active/reset
  and pre-partition lineage. Current decision: one-sided row attribution
  remains unresolved due to missing traces.
- The latest full-row attribution resolves the missing-trace blocker without a
  C++ trace patch. Broad candidate, active, reset, pre-partition,
  post-partition, and materialization tables are emitted from existing
  diagnostic artifacts. The selected rows remain one-sided, with earliest
  divergence split between `activity_threshold_crossing_mismatch=7` and
  `result_index_mapping_mismatch=5`.
- RGIE/PIPM transfer implication: the frontier is still mixed
  row-universe/mapping provenance. Broad-only `CH4(s,l)` is now value-driven at
  activity threshold, but exact KL labelled reduced-system arrays are still not
  emitted, and no repaired alpha/beta, partition, lifecycle, result-index, or
  materialization rule is promotable.
- The latest split attribution separates the one-sided blocker into two
  transfer-relevant tracks. Track A has `7` activity-threshold crossing rows:
  `30:-10:MgCO3(s)`, `30:-10:SiC(s)`, and `CH4(s,l)` at all five broad cases.
  The available counterfactuals attribute `CH4(s,l)` to a thermo/lnK
  reference-state component with a FastChem `-10` clipping/sentinel or
  display-floor trace, while `MgCO3(s)` and `SiC(s)` are attributed to atomic
  gas element density snapshot mismatch. Track B has `5` result-index mapping
  rows: `30:-10:Al(s)`, `30:-10:K3AlF6(s)`,
  `30:-10:Na3AlF6(s,l)`, `30:-10:Na5Al3F14(s,l)`, and `45:-10:Al(s)`.
  Those rows remain present but without result indices after label
  normalization recheck. Exact KL labelled RHS/Jacobian row/column arrays, row
  scaling, and solver result vectors by label are still not emitted.
- RGIE/PIPM transfer implication: do not transfer repaired alpha/beta as a
  solution to the current focused blocker. The next transfer frontier is split
  between activity-threshold source-state decomposition and exact KL labelled
  reduced-system/result-index materialization. Current decision: next blocker
  is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus
  result-index mapping.
- The split frontier now has source-state detail. For `CH4(s,l)`, the
  diagnostic-only FastChem whitebox trace emits the raw mass-action
  `raw_log_activity_before_floor_clip` before the data-validity floor, while
  the stored `displayed_log_activity_after_floor_clip` remains `-10` with
  `data_validity_floor=true` and `clipped_or_floored=true` from
  `Condensate<double_type>::calcActivity` lines `77-92`. The raw pre-floor
  values are `58.169686102954905`, `-4.530329499225015`,
  `-15.485824491588767`, `-19.05959499690444`, and `-21.452325720828508`
  for `30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`. The stored `-10`
  is not a true computed activity. KL CH4 activities are positive in all five
  cases, and the decomposition points to thermo/lnK reference-state convention
  with phase segment `l`, density-gauge/standard-state terms, and formula row
  `C + 4H` recorded.
- For `30:-10:MgCO3(s)` and `30:-10:SiC(s)`, the activity-threshold source
  state is donor snapshot mismatch: FastChem uses post-condensation fixed
  atomic element-species donor terms while KL uses gas-only `ln_nk` donor
  terms. The per-element post-`correctValues` contribution before FastChem
  `calcActivity` is still a missing diagnostic field.
- For Track B, KL exact labelled RHS/Jacobian row/column arrays, row scaling,
  and solver result vectors by label remain unavailable. KL split-history
  indices reconstruct labels, positions, and result indices, but the exact
  labelled arrays are still required before result-index mapping can be
  promoted beyond diagnostic provenance.
- RGIE/PIPM implication: the next transferable comparison must stay split.
  Activity rows require activity source-state traces, and result-index rows
  require KL exact labelled reduced-system materialization. Repaired
  alpha/beta cannot affect absent or unprojectable one-sided rows, and no
  electron, beta, molecule, inventory, removed-tail, selected-row,
  row-scaling, lifecycle, labelled-system, guarded-mode, or production solver
  rule is promotable.

## Split Frontier Display-Floor and Materialization Audit

The split frontier remains diagnostic-only under `focused_raw_result_provenance_metric`: `12` selected rows, `0` shared numeric rows, and `12` one-sided rows. Track A keeps the `7` activity-threshold rows, and Track B keeps the `5` result-index rows. No one-sided row is reinterpreted as shared numeric, and the embedded broad 10-row projection remains a separate non-closing diagnostic rather than a focused regression.

For `CH4(s,l)`, the diagnostic-only FastChem trace from `Condensate<double_type>::calcActivity` now emits the exact display-floor condition flags. All five broad cases have `data_validity_floor=true`, finite raw/stored values, valid species/phase and density/maxDensity flags, and `candidate_absence_display_flag=false`. The threshold input used by `selectActiveCondensates` is the stored `log_activity`, not the raw pre-floor value. At `30:-10`, raw pre-floor `58.169686102954905` would pass, but the data-validity floor stores `-10`, so the stored threshold fails and the display-floor path affects candidate selection. At `45:-10`, `60:-5`, `75:-5`, and `90:-5`, both raw pre-floor and stored threshold fail on the FastChem side. KL CH4 remains threshold-positive in all five cases, and the thermo/lnK side-by-side record remains emitted for source comparison.

For `30:-10:MgCO3(s)` and `30:-10:SiC(s)`, the compact now emits a per-element C/Mg/O/Si stage table. The available earliest divergent stage is the fixed-element/full-element donor vector consumed by FastChem `calcActivity`: FastChem uses post-condensation fixed full-element values, while KL uses gas-only `ln_nk` values. Per-element post-`correctValues` and per-element density-gauge transformed atomic values are still reported as missing diagnostic fields. Counterfactuals show `FC/KL thermo + FC full-element vector` pass and `FC/KL thermo + KL gas-only vector` fail, so the atomic vector is the threshold-crossing component.

For Track B, exact KL labelled RHS row labels, Jacobian row labels, Jacobian column labels, row scaling by label, and solver result vector by label remain unavailable. The compact reports the exact missing Python locals / trace records and patch site, while split-history materialization still reconstructs labels, positions, and result indices. In the reconstructed materialization all five Group B rows are absent before reduced-system assembly and have no result slot, but exact KL arrays are still required before this can be promoted to a final basis claim.

Repaired alpha/beta remains irrelevant unless shared numeric selected rows appear. No production electron rule, guarded mode, solver behavior, preset, RHS, molecule, inventory, removed-tail, selected-row, row-scaling, active-selection, lifecycle, labelled-system, maxDensity, or density-gauge bridge behavior is promotable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Focused Frontier Closure and Broad Projection Transfer Pivot

For RGIE/PIPM transfer, the focused one-sided selected-row blocker is now
closed as diagnostic provenance: CH4 `5/5`, MgCO3/SiC `2/2`, and Group-B
`5/5` have explicit explanations, leaving `0` unresolved focused one-sided
rows. Exact KL labelled arrays are available for all broad cases, and the
Group-B rows are classified as `intentionally_excluded_from_reduced_solve`
rather than shared numeric rows.

The transfer implication is deliberately narrow. CH4 data-validity masking,
FastChem donor snapshot attribution, and Group-B reduced-solve exclusion are
not promotable RGIE/PIPM rules. They only close the focused diagnostic
frontier and preserve existing production behavior.

The remaining transfer frontier is the broad diagnostic projection /
outside-selected residual. The embedded broad 10-row projection remains a
separate non-closing probe, not a focused regression. Focused-frontier levers
annotate overlapping broad rows, but the remaining broad residual is still
outside-selected/full-vector source state dominated.

Decision: broad projection residual remains dominated by outside-selected/full-vector source state.

## Integrated Split-Frontier Counterfactual

The RGIE/PIPM transfer implication is now narrower. A diagnostic-only
integrated counterfactual applies the FastChem CH4 data-validity candidate mask
to KL CH4 rows and the FastChem fixed/full-element donor snapshot to
`30:-10:MgCO3(s)` and `30:-10:SiC(s)`. The CH4 mask removes all five CH4
one-sided rows; the donor snapshot explains the two atomic-source rows. These
transforms are source-state probes only and do not define a production
thermochemistry, density-gauge, maxDensity, donor, active-selection, or solver
rule.

The remaining transferable blocker is Group-B result-index materialization:
`30:-10:Al(s)`, `30:-10:K3AlF6(s)`, `30:-10:Na3AlF6(s,l)`,
`30:-10:Na5Al3F14(s,l)`, and `45:-10:Al(s)`. Exact KL labelled RHS/Jacobian
arrays, row scaling by label, and solver result vector by label are still
required. Repaired alpha/beta remains irrelevant until shared numeric selected
rows reappear, and the broad 10-row projection remains a separate non-closing
diagnostic.

Decision: integrated counterfactual reduces blocker to Group-B result-index mapping.

## Group-B Transfer Gate

The Group-B audit leaves the RGIE/PIPM transfer gate at exact KL labelled
materialization. FastChem provides exact labelled reduced-system rows and
columns for the comparison cases, while KL still provides only reconstructed
split-history indices from `condensates_jac`. Without exact KL RHS/Jacobian
labels, row scaling by label, and solver result vector by label, the five
Group-B rows cannot be proven shared numeric, intentionally excluded, or compact
artifacts.

The remaining transfer blocker is therefore diagnostic materialization, not
alpha/beta, donor state, CH4 thermochemistry, electron handling, selected-row
rules, row scaling, lifecycle, or production solver behavior.

Decision: Group-B result-index blocker remains blocked by missing KL exact labelled arrays.

## Exact KL Array Transfer Update

The transfer gate moved from missing KL materialization to exact lifecycle
classification. The existing broad diagnostic snapshot already contains exact
KL reduced-system arrays under
`actual_true_kl_atomic_branch_exact_second_post_seed_update_proven`; the
Group-B compact now consumes that record. All five rows are absent before
candidate selection in the exact lifecycle table, so they do not become shared
numeric rows and do not require a production result-index or selected-row rule.

The current focused one-sided blocker is therefore reduced by diagnostic
source-state and exact-materialization provenance: CH4 `5/5`, MgCO3/SiC `2/2`,
and Group-B `5/5` are all classified without promoting behavior.

Decision: Group-B result-index blocker reduces after exact labelled materialization.

For RGIE/PIPM transfer, the exact KL array wiring resolves the Group-B
materialization gap. The five rows are classified as
`intentionally_excluded_from_reduced_solve` with lifecycle root cause
`absent_before_candidate_selection`, so no shared numeric row or production
result-index rule is introduced.

Decision: Group-B result-index blocker is fully explained as intentional reduced-solve exclusion.

## Static-Code Split Frontier Sharpening

Static code confirms the CH4 branch: FastChem `Condensate<double_type>::calcMassActionConstant` evaluates `log_K = a1/T + a2*log(T) + a3 + a4*T + a5*T*T`, applies `density_correction = -sigma * log(1.0e6 / (k_B*T))`, and uses `mass_action_constant + sum_i nu_i log(n_i)` as raw activity. `Condensate<double_type>::calcActivity` then enforces `if use_data_validity_limits && temperature > fit_coeff_limits.back()` by tracing the raw value, storing `log_activity = -10`, setting `data_validity_floor=true`, and returning. `selectActiveCondensates` consumes stored `log_activity >= 0`, not the raw pre-floor value.

The CH4 compact now records temperature, selected segment index, finite FastChem data-validity upper, the `temperature > fit_coeff_limits.back()` predicate, raw/stored activity, and stored-threshold candidate decision for all five broad cases. `30:-10:CH4(s,l)` is raw-positive (`58.169686102954905`) but rejected because the data-validity branch writes stored `-10`. The other CH4 cases are raw-negative and also stored at `-10` by the data-validity path. KL uses `compute_kl_condensate_log_activity = formula_cond.T @ u - hcond`; `fastchem_cond.py` prepares the final segment upper as `inf`, so KL extrapolates the CH4 final segment beyond FastChem's finite validity upper. CH4 is therefore classified as FastChem data-validity floor versus KL extrapolation, not a stored true-activity mismatch.

MgCO3/SiC remain fixed/full-element donor snapshot mismatches. Their per-element donor tables show FastChem `calcActivity` consumes fixed/post-condensation full-element donor values while KL consumes gas-only `ln_nk`; source swaps confirm the donor vector drives threshold pass/fail. Group B remains blocked by missing exact KL labelled arrays: split-history reconstructs index materialization, but exact RHS/Jac labels, row scaling by label, and solver result vector by label are still unavailable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Diagnostic Counterfactual Split Frontier

The CH4 counterfactual audit is diagnostic-only and does not alter KL thermochemistry or presets. Current KL activity stays positive for all five CH4 rows. Applying a FastChem-style data-validity candidate mask, `T > FastChem fit_coeff_limits.back() -> threshold value -10`, makes CH4 fail the KL candidate threshold in all five broad cases and removes the CH4 one-sided membership. The finite-upper-without-floor KL selector is not emitted as an exact artifact and is reported as a missing diagnostic selector rather than inferred. FastChem thermo/density/formula with KL atomic state fails, while KL final-segment extrapolated thermo with FastChem donor state passes, confirming the CH4 frontier is the data-validity floor versus KL final-segment extrapolation.

MgCO3/SiC remain donor-snapshot driven: FastChem full-element donor values from the fixed overwrite path make both rows pass, while KL gas-only donor values make them fail under both FC and KL thermo terms. Group B remains blocked by missing KL exact labelled RHS/Jacobian/scaling/result arrays; split-history is still the closest materialization.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Production-Readiness Transfer Boundary

The production-readiness package adds a transfer boundary, not a transfer rule:

- `results/fastchem_cond_kl_production_readiness_compact.json`
- `docs/condensates/fastchem_parity_kl_semantic_design_note.md`

Transfer implication:

- RGIE/PIPM may adopt the semantic vocabulary of normalized donor state,
  physical donor state, coherent molecule + inventory state, and
  removed-condensate provenance state.
- RGIE/PIPM must not adopt the diagnostic FastChem CH4 data-validity mask,
  MgCO3/SiC donor snapshot, Group-B exclusion, full FastChem coherent replay,
  or `Al4C3(s)` removed-tail replay as production behavior.
- Physical donor conversion is a comparison requirement, not a default donor
  rule. The legacy KL-reference burden-ratio conversion remains forbidden.
- Molecule and inventory must be treated coherently when the diagnostic result
  relies on their cancellation. Standalone molecule or inventory transfer is
  not supported.
- The `45:-10` removed-tail closure is localized until extra broad cases are
  regenerated. Do not project it as a focused regression or general broad
  lifecycle rule.

Decision: semantic levers ready for production design note but not promotable.

## Milestone 2 Transfer Readiness

Milestone 2 adds a costed transfer blocker, not a new transfer rule:

- `results/fastchem_cond_kl_milestone2_generalization_readiness_compact.json`

Transfer status:

- The available five-case broad set supports the semantic vocabulary:
  normalized donor, physical donor, coherent molecule + inventory, and removed
  provenance.
- The `Al4C3(s)` removed tail remains local to `45:-10` until extra broad
  regeneration proves otherwise.
- RGIE/PIPM should not treat the current five-case result as broad
  generalization beyond the represented cases.
- Before any transfer decision changes, the broad regeneration campaign must
  emit expanded source/traces, selected-row provenance, same-boundary KL
  non-fixed vectors, and regenerated broad smoke/direct/round compacts.

Decision: broad generalization requires regeneration campaign before next
decision.

## Milestone 3 Transfer Infrastructure

RGIE/PIPM transfer work can now consume a broad case manifest:

- `results/fastchem_cond_kl_broad_case_manifest.json`

The manifest-driven replay preserves the prior transfer boundary. The current
five cases still support physical donor comparability and coherent
molecule/inventory semantics, while the `Al4C3(s)` removed tail remains local
to `45:-10`.

No new transfer rule is introduced. A future pilot case should first be added
to the manifest with explicit source/trace shard paths, downstream compact
availability, and missing-field blockers. It must not be interpreted from a
source snapshot alone.

Decision: manifest-driven broad generalization infrastructure ready; pilot case
not yet run.

## Milestone 4 Transfer Gate

RGIE/PIPM transfer work can now rely on a manifest-only current-five replay
gate for the broad diagnostic stack:

- current-five broad replay is driven by
  `results/fastchem_cond_kl_broad_case_manifest.json`,
- downstream diagnostic compacts no longer use a hidden hard-coded current-five
  fallback for replay,
- the Round 9 `45:-10` decomposition remains a localized tail target and not a
  general broad lifecycle rule,
- no rows, species, or cases may be dropped silently.

The one-case pilot remains unrun because the expected source+trace artifact
cost is about `0.8G` before downstream compacts. A future transfer decision
requires the pilot case to be added to the manifest with source paths, trace
paths, downstream compact availability, and exact missing-field blockers.

Decision: pilot-ready manifest-only downstream stack passes; pilot case not yet
run.

## Milestone 5 Transfer Boundary

The one-case pilot `45:-5` has been regenerated and registered in the manifest.
It is useful for transfer planning because it shows `Al4C3(s)` removed
correction in the source trace outside the original `45:-10` case.

Transfer boundary:

- RGIE/PIPM must not treat this as projected removed-tail closure.
- RGIE/PIPM must not promote a layer-wide `Al4C3(s)` rule.
- Physical donor and coherent molecule+inventory transfer remain design
  vocabulary only for the pilot, because selected-row and same-boundary fields
  are missing for projection.
- The next transfer-ready artifact must include pilot selected-row mapping,
  same-boundary KL non-fixed vector, and gauge-normalized inventory/atom rows.

Decision: one pilot broad case regenerated but downstream interpretation
remains incomplete.

## Milestone 6 Transfer Boundary

Milestone 6 closes the `45:-5` pilot interpretation from existing artifacts.
Transfer consumers may now rely on these diagnostic facts:

- the raw normalized KL donor gap is closed by physical donor conversion,
- selected-row mapping is explicit and includes projected versus one-sided
  membership,
- inventory/atom terms use the same current convention as the five-case stack,
- `Al4C3(s)` removed correction appears at `45:-5`,
- projected `Al4C3(s)` removed correction does not close the `45:-5` PMI
  residual.

Transfer consumers must not infer a general removed-tail correction from
`45:-10`. The source provenance recurs by layer, but projected closure is
epsilon-dependent and currently weakens broad generalization.

Decision: pilot 45:-5 downstream interpretation reveals a new blocker.

## Milestone 21 Transfer Boundary

RGIE/PIPM transfer work may use the Milestone 21 compact as evidence that the
remaining KL-native reconstruction gap is RHS-side. The same Jacobian and row
scaling are used for variant I and reference J; the hidden RHS delta alone
reconstructs the solution gap. Top hidden carriers are outside-selected
free-element rows in the covered cases.

Transfer consumers must keep this diagnostic-only. It is not permission to
copy the full FastChem coherent RHS, to use broad projection as focused
regression, or to create standalone Al4C3/tau rules. The transferable object is
the source-state contract requirement, not a correction formula.

Decision: hidden coherent source is RHS-side.
## Milestone 22 Transfer Boundary

Milestone 22 decomposes the Milestone 21 hidden RHS vector by RHS term family. Decision: hidden RHS delta is molecule-RHS dominated.

- The additive budget closes for all requested cases: current-five plus the existing `45:-5` pilot.
- The closing source is `full FastChem molecule RHS - reconstructed candidate molecule RHS`; no separate remaining charged/electron additive hidden family is needed for closure.
- Inventory/atom, removed-condensate, tau/complementarity, activity burden, and fixed/condensed overwrite families have zero remaining additive hidden RHS after the Milestone 21 KL-native RHS assembly.
- Outside-selected free-element rows remain the dominant carriers, but they are a row-location carrier view of the molecule RHS residual rather than a separate additive source.
- Production remains not promotable; KL-native reconstruction is blocked on a coherent molecule RHS parity contract, not on a new production rule.
## Milestone 23 Transfer Boundary

Milestone 23 tests the coherent molecule RHS parity contract directly. Decision: coherent molecule RHS parity holds at matched source state.

- Full FastChem molecule RHS is reproduced to roundoff by all-molecule RHS at matched source state.
- The M22 molecule delta is therefore not a request to transplant FastChem RHS; it identifies the missing semantic contract for source-state plus row-scaling/RHS convention parity.
- Source-vector, hvector/lnK, density-gauge, cache timing, and neutral/charged branches are emitted as diagnostic variants.
- KL-native reconstruction remains blocked until the semantic interface exposes a matched molecule source state and RHS convention contract.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 24 Transfer Boundary

Milestone 24 attempts to construct the matched coherent molecule source state from KL-native semantic-interface fields. Decision: matched coherent molecule source state blocked by hidden coherent source.

- The matched source-state contract is explicit: e-first physical density vector, FastChem row-scaling convention, molecule species order, hvector/lnK, and density gauge.
- No KL-native candidate among physical donor, molecule-cache, fixed/overwrite, correctValues, or best repaired same-boundary vector constructs the matched source state in all requested cases.
- Tau/complementarity, inventory/atom, removed-condensate, and reduced-slot adjusted all-element molecule-source vectors are not emitted as source-state fields.
- Row scaling is explicit and can be tested separately; it is not sufficient to construct the missing source state.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 25 Transfer Boundary

Milestone 25 attempts to materialize the missing all-element molecule source vectors from existing artifacts. Decision: source-vector materialization blocked by exact missing artifacts.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and e-first same-boundary KL candidates were materialized and rerun.
- Tau/complementarity, inventory/atom, removed-condensate, and broad reduced-slot all-element molecule source vectors are not present in existing artifacts; only RHS terms or focused/layer-limited reduced-slot records are available.
- Combination search cannot close because every requested adjusted-vector combination depends on an unmaterialized all-element source vector.
- FastChem row scaling remains explicit and is not used silently; materialized source-state residuals remain nonzero under that explicit convention.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 26 Transfer Boundary

Milestone 26 implements diagnostic emitter attempts for the missing all-element source vectors. Decision: all-element source-vector emission blocked by exact trace architecture gap.

- Python emitter attempts were added at the latest diagnostic sites where RHS terms, source traces, row scaling, and molecule labels are live.
- Current artifacts expose RHS term contributions for tau/complementarity, inventory/atom, and removed-condensate paths, but not the adjusted all-element molecule-source vectors.
- The reduced-slot compact remains focused/layer-limited and does not expose a broad case-keyed canonical e-first all-element molecule-source vector.
- Matched-source construction was rerun with all emitted/materialized vectors; no non-hidden candidate closes.
- Production remains not promotable; a future diagnostic trace must emit these all-element source vectors before another construction attempt can close.
## Milestone 27 Transfer Boundary

Milestone 27 patched diagnostic C++ trace schema and emitted the reduced-slot all-element source vector. Decision: matched coherent molecule source state still blocked by hidden coherent source.

- The emitted vector is FastChem reference-only and uses the hidden coherent FastChem source state.
- It reproduces the matched coherent molecule source state to roundoff, proving the trace architecture can carry the needed vector shape.
- It does not make the matched source KL-native constructible; KL adjusted all-element tau/inventory/removed vectors remain missing.
- Production remains not promotable; the patch is trace-only and inactive unless diagnostic tracing is enabled.
## Milestone 28 Transfer Boundary

Milestone 28 compares KL-native source-vector candidates against the M27 emitted reduced-slot FastChem reference target. Decision: KL-native source-vector reconstruction blocked by exact KL-side trace fields.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and prior same-boundary KL-native vectors were compared directly against the M27 28-element target.
- No non-hidden candidate or requested staged combination closes the source vector or molecule RHS in all cases.
- Tau/complementarity, inventory/atom, removed-condensate, and KL reduced-slot mapped all-element source vectors remain exact missing fields.
- Production remains not promotable; this milestone is diagnostic-only and does not change presets, defaults, or solver behavior.
## Milestone 29 Transfer Boundary

Milestone 29 closes the M28 Python-owned KL-side source-vector fields and reruns reconstruction. Decision: M27 target vector still requires hidden FastChem coherent source.

- All eight M28 fields were ownership-audited and emitted in the M29 diagnostic compact.
- The two fields previously labelled C++-required are KL-side source-state fields after ownership audit, so no C++ trace patch or rebuild was required.
- Emitted non-hidden KL-side vectors still do not reconstruct the M27 FastChem reference target or matched coherent molecule RHS.
- Production remains not promotable.
## Milestone 30 Transfer Boundary

Milestone 30 decomposes the value delta between the M27 target and emitted KL reduced-slot source vector. Decision: M27 target blocked by missing semantic transform.

- Element-order, log/linear, density-gauge, and row-scaling verifier branches do not close the residual.
- Fixed/non-fixed and inert/electron splits show the largest differences are not explained by a single bookkeeping subset.
- Least-squares fits can reduce residual diagnostically but require non-production coefficients, indicating a missing semantic transform rather than a direct emitted-vector selection.
- Production remains not promotable.
## Milestone 31 Transfer Boundary

Milestone 31 synthesizes diagnostic semantic transforms from emitted KL-side vectors to the M27 target. Decision: inert/carrier transform partially closes but not all cases.

- Inert pass-through from the KL physical donor stage removes the dominant He/Ne/Ar carrier residuals diagnostically.
- Source-vector and molecule-RHS closure still fail after all A-H transforms, so the transform is only partial and remains non-production.
- LS-guided class coefficients do not collapse to a global, ntot, or density-gauge scalar across cases.
- No production solver, preset, default, or guarded mode changed.
## Milestone 32 Transfer Boundary

Milestone 32 prototypes broader diagnostic source-state transforms after M31 inert pass-through. Decision: broader transform partially closes but fixed-element source-state remains.

- Gen1 attempted A2-H2; Gen2 attempted targeted top-class, metal/minor, fixed/condensed, and constrained-scalar overlays.
- No non-hidden transform reconstructs the M27 target or matched molecule RHS across current-five plus `45:-5`.
- The balanced best transform remains conservative and the remaining dominant residual class is fixed/condensed source-state material.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 33 Transfer Boundary

Milestone 33 decomposes the fixed-element source-state blocker. Decision: fixed-element transform rejected due molecule-RHS sign amplification.

- A2, F2, and K2 were decomposed by condensation class and residual class for current-five plus `45:-5`.
- A3-G3 RHS-aware fixed-element transforms were attempted; no transform reconstructs the M27 target and molecule RHS across all cases.
- Source-vector improvements can worsen molecule RHS because fixed-row changes are amplified by molecule stoichiometry and FastChem row scaling.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 34 Transfer Boundary

Milestone 34 audits the source-to-molecule-RHS operator. Decision: sign amplification due row scaling.

- R(x) was evaluated directly from diagnostic source vectors through molecule reconstruction and FastChem row scaling.
- Finite-difference sensitivity on top fixed elements attributes the worsening RHS behavior to row-scaled fixed-element molecule-burden sensitivity.
- RHS-space stage/scalar/finite-difference candidates do not close all cases without hidden coherent source state.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 35 Transfer Boundary

Milestone 35 decomposes row-scaling amplification into numerator and scaling terms. Decision: row-scaling high-gain fixed rows require M27 source parity.

- The compact emits `N(x)`, `R(x)`, `Delta N`, `Delta R`, row-scaling signs/magnitudes, and amplification factors for each covered case.
- Row-scaling-aware candidates A-F do not close all cases, current-five, or `45:-5` without hidden coherent source state.
- High-gain fixed rows remain the practical inverse-problem blocker; production remains not promotable.
## Milestone 36 Transfer Boundary

Milestone 36 traces high-gain fixed-row source provenance. Decision: high-gain fixed-row parity insufficient; full M27 source vector remains required.

- High-gain fixed rows were selected from M35 row-scaling amplification and RHS contribution budgets.
- FastChem and KL ladders show the M27 reduced-system assembly source value is not reproduced by emitted KL-native stages.
- High-gain-row-only replays are diagnostic-only and do not close all cases; full M27 source-vector parity remains required.
- No production solver, preset, default, or C++ trace changed.
## Milestone 37 Transfer Boundary

Milestone 37 decomposes source-vector support. Decision: sparse support partially closes but not all cases.

- Row classes and overlaps were emitted for high-gain fixed, inert, electron, volatile, reactive, metal/minor, top-species, outside-selected, and row-scaling high-gain rows.
- Replays A-K and greedy support search found a stable outside-selected support branch that partially closes, including `45:-5`, but not all current-five cases.
- Top-species support rows improve attribution but do not replace full M27 source-vector parity for non-closing cases.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 38 Transfer Boundary

Milestone 38 decomposes residuals after the M37 sparse support. Decision: 45:-5 closes by cancellation; current-five requires broader support.

- The best sparse support residual was decomposed for H/K, and `45:-5` was compared against non-closing current-five cases.
- Outside-selected internal ablations A-G and a second-stage greedy search were attempted.
- The 45:-5 closure is not a general support sufficiency result; current-five still requires broader support.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 39 Transfer Boundary

Milestone 39 decomposes the remaining `30:-10` sparse-support residual. Decision: 30:-10 residual is numerical/tolerance-scale but not production-promotable.

- H/B/G/K support residuals were decomposed for `30:-10` and compared with closing cases.
- Internal add/remove ablations A-H and tolerance/scale audit classify the remaining residual as strict-tolerance scale.
- Closure under looser diagnostic tolerance is not production-promotable; production remains not promotable.
## Milestone 40 Transfer Boundary

Milestone 40 campaign decision: strict tolerance residual is diagnostic-only; production requires source-state contract.

- Track 1: strict 30:-10 residual closes only at diagnostic 1e-5 and is not production-promotable.
- Track 2: emitted non-hidden KL full-vector candidates do not close source, numerator, or row-scaled RHS parity.
- Track 3: production readiness now depends on a formal source-state semantic contract, not another support subset.
- Track 4: no additional broad pilot is justified until the contract gap is addressed.
## Milestone 41 Transfer Boundary

Milestone 41 created the default-off source-state contract prototype. Decision: default-off source-state contract prototype complete; KL-native implementation remains blocked.

- The schema separates FastChem/M27 reference records, best non-hidden KL candidates, sparse support overlays, tolerance-only closures, and the unavailable production-ready KL-native source state.
- The acceptance gate requires source parity, numerator parity, row-scaled RHS parity, no hidden FastChem source, strict tolerance, complete coverage, explicit row scaling, and preserved lineage.
- Current-five plus `45:-5` are instantiated from existing artifacts; no new broad pilot was requested.
- All best non-hidden KL candidates fail the default-off production gate because the source-state constructor and numerator contract are still missing.
## Milestone 43 Transfer Boundary

Milestone 43 synthesized gate-driven constructors from non-hidden KL-side basis vectors. Decision: free fit also fails; hidden source-state information structurally absent.

- Generations G1-G3 attempted source-, numerator-, RHS-, joint-, class-wise, constrained, and free diagnostic fits.
- M41 gate evaluation remained default-off and diagnostic-only; production behavior and presets were unchanged.
- The free diagnostic fit also failed source, numerator, and RHS gate closure, classifying the residual as structurally absent from the non-hidden KL basis.
## Milestone 44 Transfer Boundary

Milestone 44 expanded the non-hidden KL primitive basis and emitted structural span diagnostics. Decision: diagnostic C++ trace required for missing primitive source-state fields.

- Current and expanded basis rank/projection residuals were emitted for source vector, unscaled numerator, and row-scaled RHS spaces.
- Primitive branches A-G were attempted; exact number_density_min/maj and epsilon/phi/degree transforms remain unavailable without diagnostic trace fields.
- Expanded basis gate rerun did not close M41; next work requires diagnostic C++ trace for missing primitive source-state fields.
## Milestone 45 Transfer Boundary

Milestone 45 added env-gated C++ primitive source-state trace fields. Decision: traced primitives improve but remain FastChem reference-only.

- The M45 marker emits fixed-row pre/post overwrite values, number_density_min/maj, gas solver path, epsilon, phi, and degree_of_condensation.
- The traced primitive basis reruns the M41 gate but remains diagnostic FastChem reference-only, not KL-native production logic.
- Production remains not promotable; the next implementation step is coding the semantic source-state algorithm on the KL side.
## Milestone 46 Transfer Boundary

Milestone 46 implemented a default-off KL-native semantic source-state algorithm prototype. Decision: KL lacks required lifecycle input fields for semantic source-state algorithm.

- Variants A-D compute primitive fields from KL/public vectors only; M45 trace values are diagnostic reference targets only.
- The M41 gate rerun preserves no-hidden-source, not-reference-only, and KL-native-constructible checks, but source/N/R parity remains open.
- Missing KL lifecycle inputs are now narrowed to molecule contribution order, backup branch lifecycle, condensation-degree transform inputs, and reduced correctValues assembly semantics.
## Milestone 47 Transfer Boundary

Milestone 47 localized semantic source-state gaps by FastChem function. Decision: FastChem semantic source-state functions must be ported before production.

- Four default-off Python function-port prototypes were implemented and compared against M45 trace references without using traced values as constructor inputs.
- Three M47 source-state candidates were gated through M41 over current-five plus 45:-5.
- Remaining blockers are exact lifecycle fields: molecule order/accumulators, backup/intertSol branch state, condensation-stage inputs, and correctValues reduced result/clipping state.
## Milestone 48 Transfer Boundary

Milestone 48 emitted default-off KL lifecycle state records and reran five M48 candidates through M41. Decision: M48 improves but molecule order state remains missing.

- Implemented lifecycle emitters: condensation_stage_state, correctValues_reduced_result_state, gas_solver_branch_state, minor_major_accumulator_state.
- Unavailable lifecycle emitters: molecule_order_state.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem reference-source transplant was changed.
## Milestone 49 Transfer Boundary

Milestone 49 emitted KL molecule-order state proxies and reran minor/major accumulator replay. Decision: exact molecule_order_state requires FastChem-specific ordering not present in KL.

- KL-native molecule-order emitters: A_KL_gas_species_order, B_FastChem_label_aligned_KL_density, D_best_non_hidden_molecule_order.
- Exact element-specific FastChem minor/major molecule order remains unavailable from KL artifacts and would need a diagnostic trace for validation.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem source-vector transplant was changed.
## Milestone 50 Transfer Boundary

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
