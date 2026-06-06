# FastChem/KL Condensate Semantics Ledger

This ledger freezes the semantic levers already proven by audit-only FastChem/KL comparisons. It is not a production change plan by itself.

## Milestone 20 KL-Native Reconstruction Attempt

The latest KL-native reconstruction compact is:

- `results/fastchem_cond_kl_milestone20_kl_native_reconstruction_attempt_compact.json`
- `results/fastchem_cond_kl_milestone20_kl_native_reconstruction_attempt_compact.md`

Current status from that audit:

- The ladder uses explicit semantic-interface fields only: physical donor,
  molecule-cache vector, fixed/condensed overwrite boundary, inventory/atom
  state, removed correction, post-PMI tau/complementarity replay, and projection
  coefficient response. Full FastChem coherent RHS appears only as reference.
- KL-native reconstruction does not close any covered case. The best native
  max residuals remain nonzero for current-five and for the existing `45:-5`
  pilot.
- Full FastChem coherent RHS reference closes all covered cases, so the current
  blocker is not a missing projection coefficient, tau replay, overwrite,
  inventory, or removed-correction field. It is a hidden full coherent source
  that is not represented by the KL-native semantic interface.
- The layer-45 coefficient map remains identical between `45:-5` and `45:-10`;
  variant-I residuals are also effectively identical. The known layer-45
  difference therefore remains a source residual/projection-content issue, not
  coefficient geometry.
- Production remains not promotable.
- Decision: KL-native reconstruction blocked by hidden full coherent source.

## Milestone 19 Direct-Broad Regeneration Projection/Tau Closure

The latest direct-broad regeneration compact is:

- `results/fastchem_cond_kl_milestone19_direct_broad_regeneration_projection_tau_compact.json`
- `results/fastchem_cond_kl_milestone19_direct_broad_regeneration_projection_tau_compact.md`

Current status from that audit:

- The exact missing current-five direct-broad source and trace artifacts were
  regenerated. The existing `45:-5` pilot source/traces were reused; no new
  broad case was introduced.
- The patched direct-broad evaluator now emits explicit
  `solve(J, unit_outside_row)` projection coefficients for current-five cases
  and the `45:-5` pilot.
- The post-PMI tau/complementarity diagnostic ladder is emitted for all covered
  cases. Tau/complementarity is separable as a diagnostic RHS vector in this
  replay, but remains non-promotable as a standalone production rule.
- Layer `45:-5` and `45:-10` have identical outside-selected unit-response
  coefficients for common layer-45 row pairs. Therefore coefficient geometry
  does not explain the known projection mismatch; the mismatch remains in the
  source residual/projection content already resolved by Milestone 13.
- Production remains not promotable.
- Decision: projection coefficients closed; post-PMI tau replay closed.

## Milestone 18 Projection/Tau Python Patch

The latest projection/tau patch compact is:

- `results/fastchem_cond_kl_milestone18_projection_tau_patch_compact.json`
- `results/fastchem_cond_kl_milestone18_projection_tau_patch_compact.md`

Current status from that audit:

- The direct-broad Python diagnostic script now has an explicit
  `solve(J, unit_outside_row)` projection-coefficient emitter and a diagnostic
  post-PMI tau/complementarity ladder emitter.
- Regeneration cannot emit coefficient rows in this workspace because the raw
  direct-broad snapshot and trace objects are absent:
  `results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke.json`
  and
  `results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke_traces.json`.
- The existing stored direct-broad compact predates the new coefficient/tau
  fields. It is useful for lineage but cannot be used to infer coefficients
  from aggregate outside-selected residuals.
- The Round 8/9 tau/complementarity baseline-consumption proof is accepted as
  sufficient for design review. A standalone tau ladder will emit when the
  patched direct-broad numeric cases are constructible, but it is not a
  production rule.
- Decision: projection coefficients remain blocked by exact missing
  direct-broad objects.

## Milestone 17 Python Gap Closure

The latest Python gap-closure compact is:

- `results/fastchem_cond_kl_milestone17_python_gap_closure_compact.json`
- `results/fastchem_cond_kl_milestone17_python_gap_closure_compact.md`

Current status from that audit:

- No new broad case was run, and no production solver behavior, preset,
  default, or production rule changed.
- Projection coefficient reconstruction was attempted as
  `solve(J, unit_outside_row)` projected into selected rows. Existing artifacts
  contain outside-selected row summaries, but the source snapshot is absent and
  the compact does not contain `fc_j`, `row_to_result`, or unit-response
  coefficients. The coefficient solve is therefore blocked exactly at the
  Python compact boundary.
- The post-PMI tau/complementarity replay remains unavailable as a separate
  ladder. Milestone 17 formalizes the baseline-consumption proof from Round
  8/9: tau fields are null/statused as already consumed in the coherent
  baseline, while the standalone tau RHS vector and post-PMI ladder are not
  emitted.
- Both blockers remain Python diagnostic patch-site gaps; no C++ trace patch is
  required by the current evidence.
- Decision: both remain Python diagnostic gaps with exact blockers.

## Milestone 16 Design Review And Field Closure

The latest design-review compact is:

- `results/fastchem_cond_kl_milestone16_design_review_and_field_closure_compact.json`
- `results/fastchem_cond_kl_milestone16_design_review_and_field_closure_compact.md`

Current status from that audit:

- The guarded semantic-interface package is design-review ready and remains
  diagnostic-only. Production remains not promotable.
- Interface field definitions are now packaged with source artifacts for donor
  basis, molecule-cache vector, fixed/condensed overwrite, `correctValues`,
  reduced Newton labels, inventory/atom state, removed correction,
  tau/complementarity, row-wise projection/sign audit, and metric lineage.
- Existing compacts do not emit explicit outside-selected to selected-row
  projection coefficients. They emit outside-selected RHS differences and
  selected-row residual summaries, but not the response matrix
  `solve(J, unit_outside_row)` projected into each selected row.
- Round 8/9 provide an explicit baseline-consumption proof for
  tau/complementarity: Round 8 has `tau_contribution=null` on all 40 projected
  rows with status that tau is already in the coherent baseline, and Round 9
  repeats this status for the 8 Al4C3 removed-tail rows.
- The separate post-PMI tau replay vector is still not emitted. Both remaining
  fields are Python diagnostic patch-site gaps, not C++ trace requirements.
- Decision: design review package complete; projection/tau fields remain Python
  diagnostic gaps.

## Milestone 15 Guarded Semantic-Interface Prototype

The latest semantic-interface compact is:

- `results/fastchem_cond_kl_milestone15_guarded_semantic_interface_prototype_compact.json`
- `results/fastchem_cond_kl_milestone15_guarded_semantic_interface_prototype_compact.md`

Current status from that audit:

- The prototype is a default-off diagnostic compact, not a runtime solver mode.
  No production behavior, presets, defaults, or production rules changed.
- The unified interface records name normalized donor, physical donor,
  molecule-cache full-element vector, fixed/condensed overwrite boundary,
  `correctValues` overwrite records, reduced Newton RHS/Jacobian/result-slot
  labels, inventory/atom state, removed-condensate correction,
  tau/complementarity sensitivity, row-wise projection/sign audit, and metric
  lineage.
- Existing artifacts support current-five manifest records for `30:-10`,
  `45:-10`, `60:-5`, `75:-5`, and `90:-5`, plus the focused layer-45
  `45:-5`/`45:-10` comparison.
- The `45:-10` Al4C3 projection remains local evidence only: it opposes PMI on
  all eight rows. At `45:-5`, the same row set remains mixed-sign, four
  opposing and four reinforcing PMI. No Al4C3 or tau/complementarity production
  rule is created.
- Low-cost projection/tau closure was attempted from existing artifacts.
  Exact outside-selected projection coefficients and a standalone post-PMI
  tau/complementarity replay are still unavailable; both are declared
  diagnostic Python patch-site fields and do not require a C++ trace patch for
  this prototype.
- Decision: guarded diagnostic semantic-interface prototype complete; design
  review is next.

## Milestone 14 Production-Promotability Gap Package

The latest production gap compact is:

- `results/fastchem_cond_kl_milestone14_production_promotability_gap_compact.json`
- `results/fastchem_cond_kl_milestone14_production_promotability_gap_compact.md`

Current status from that audit:

- No new broad case was generated. The package reads the Milestone 6-13
  compacts, Round 8/9 broad diagnostics, and the existing production-readiness
  scorecard.
- Design-note-ready levers are physical donor comparability, coherent
  molecule+inventory state interface, removed-condensate provenance boundary,
  and fixed/condensed overwrite plus molecule-cache boundary naming.
- Guarded diagnostic mode is the only prototype class currently justified:
  explicit semantic-state-interface trace, row-wise projection/sign audit, and
  source-stage labels for `correctValues` overwrite and reduced Newton slots.
- Not-promotable levers remain Al4C3 removed-tail rules, tau/complementarity
  standalone rules, full coherent FastChem RHS transplant, isolated
  molecule/inventory/donor/removed transplants, and fixed-overwrite/cache
  transplants.
- Low-cost missing fields were checked. Explicit outside-selected projection
  coefficients are not emitted, and a separate post-PMI tau/complementarity
  tail replay is not emitted; both are Python diagnostic patch-site gaps, not
  C++ trace requirements for the current package.
- Required production evidence now includes a default-off semantic-interface
  contract, broad-compatible projection coefficients, tau baseline/tail
  semantics, regenerated broad scorecards, and KL-native reconstruction without
  hidden full coherent RHS transplant.
- Decision: production gap package complete; guarded diagnostic interface is
  next.

## Milestone 13 Layer-45 Projection/Sensitivity Comparison Addendum

The latest comparison compact is:

- `results/fastchem_cond_kl_milestone13_45_comparison_projection_sensitivity_compact.json`
- `results/fastchem_cond_kl_milestone13_45_comparison_projection_sensitivity_compact.md`

Current status from that audit:

- No new broad case was generated. The audit reruns only the focused `45:-5`
  and `45:-10` traces needed to attach Milestone 12 row-wise sensitivities.
- The projected row set is the same eight condensate rows: `SiC(s)`,
  `Cr23C6(s)`, `Cr7C3(s)`, `Al6Si2O13(s)`, `MgCO3(s)`, `K3AlF6(s)`,
  `Na3AlF6(s,l)`, and `Na5Al3F14(s,l)`.
- At `45:-10`, the Al4C3 removed projection is opposite-sign to the PMI
  residual on all eight rows and closes the maximum residual to
  `2.91e-11`.
- At `45:-5`, the Al4C3 removed projection has mixed sign: four rows oppose
  PMI and four rows reinforce it. The maximum residual after the Al4C3
  projection remains `1.515`, while full FastChem coherent RHS closes the rows.
- The remaining `45:-5` closure path is consistent with Milestone 8: a
  material outside-selected neutral molecule full-vector coupling over 22
  free-element rows remains before the full coherent RHS replay.
- Decision: `45` comparison resolved by Al4C3 projection mismatch against
  tau/complementarity sensitivity.
- Guardrail: diagnostic-only. No production behavior, preset, projection,
  source-state, burden-ratio, infinity-norm, row-scaling, or row/species/case
  dropping rule is promotable.

## Milestone 12 Pilot 45:-5 Reduced Newton Result-Slot Budget Addendum

The latest result-slot compact is:

- `results/fastchem_cond_kl_milestone12_reduced_newton_result_slot_budget_compact.json`
- `results/fastchem_cond_kl_milestone12_reduced_newton_result_slot_budget_compact.md`

Current status from that audit:

- No new broad case was generated. The audit reruns the existing `45:-5`
  focused FastChem trace path and the cheap focused `45:-10` mirror only.
- The diagnostic C++ trace now emits the `newton_iter=0` full reduced system
  that feeds `correctValues`, plus retained/Jac and removed-active
  condensate burden RHS families. Production solver arithmetic and presets are
  unchanged.
- The traced result slot, after FastChem's global result scaling, reconstructs
  the Milestone 11 raw solver delta. The scaled slot plus clipping
  reconstructs the overwrite, the overwrite reconstructs the molecule-cache
  vector, and the cache vector recovers the Milestone 10 top-species molecule
  density replay.
- Fixed-J RHS-family sensitivity attributes the top-element slots primarily to
  tau/complementarity rows propagated through the coupled reduced linear solve;
  retained condensate, molecule RHS, and total inventory are smaller in the
  focused aggregate, and removed-condensate terms are zero for the traced
  `45:-5` slot.
- Decision: `45:-5` result slot is coupled-linear-solve dominated.
- Guardrail: diagnostic-only. No production behavior, preset, overwrite,
  source-state, selected-row projection, burden-ratio, infinity-norm, or
  row/species/element/case dropping rule is promotable.

## Milestone 11 Pilot 45:-5 Fixed/Condensed Overwrite Budget Addendum

The latest overwrite-budget compact is:

- `results/fastchem_cond_kl_milestone11_fixed_condensed_overwrite_budget_compact.json`
- `results/fastchem_cond_kl_milestone11_fixed_condensed_overwrite_budget_compact.md`

Current status from that audit:

- No new broad case was generated. The audit reruns only the existing `45:-5`
  focused FastChem trace path after verifying the local pyfastchem trace emits
  `correctvalues_element_update` records.
- The emitted local `correctValues` identity closes for the top affected
  elements: old element density plus solver delta plus clipping/cap residual
  reconstructs the overwrite/cache value to relative roundoff.
- The reconstructed overwrite vector matches the molecule-refresh cache vector
  and recovers the top-species molecule-density replay through the Milestone 10
  identity gate.
- The physical additive terms remain unavailable: retained-condensate,
  removed-condensate, condensate-coupled, and inventory/removal contributions
  are folded into `result(i + nb_cond_jac)` and are not emitted as separate
  per-element budget fields.
- Decision: `45:-5` overwrite budget remains unresolved due missing additive
  trace fields.
- Guardrail: diagnostic-only. No production behavior, preset, overwrite,
  source-state, selected-row projection, burden-ratio, infinity-norm, or
  row/species/element/case dropping rule is promotable.

## Milestone 10 Pilot 45:-5 Cache-Vector Provenance Addendum

The latest cache-vector provenance compact is:

- `results/fastchem_cond_kl_milestone10_pilot_45_m5_cache_vector_provenance_compact.json`
- `results/fastchem_cond_kl_milestone10_pilot_45_m5_cache_vector_provenance_compact.md`

Current status from that audit:

- No new broad case was generated. The audit consumes existing `45:-5` pilot
  artifacts and the existing Round 9 `45:-10` comparison compact.
- The first emitted stage matching the molecule-cache vector is
  `molecule_refresh_record_0`, the cache-side
  `iter0_post_correctValues_full_element_vector_before_molecule_refresh`
  record.
- The iter1 RHS molecule-density provenance and cache-side iter1 full reduced
  system are identical to that cache vector for the top elements.
- The top elements are classified as `fixed/condensed overwrite` via
  `value_source_mode=reduced_overwrite_from_correctValues` and emitted fixed
  overwrite components.
- Candidate-vector replay verifies the source: physical donor vectors fail to
  recover cache density and carry the M9 RHS contribution; post-refresh cache,
  iter1 RHS, and full coherent vectors recover cache density with zero
  contribution versus cache.
- The `45:-10` listed artifact does not emit the same cache-vector provenance
  ladder, so the comparison remains unresolved beyond the known Al4C3 projected
  closure.
- The exact deeper missing field is the additive budget inside
  `reduced_overwrite_from_correctValues`: old element density, solver delta,
  clipping/cap, condensate-coupled, and inventory/removal components. A C++
  trace patch is required for that deeper split.
- Decision: `45:-5` cache-vector source is fixed/condensed overwrite component.
- Guardrail: diagnostic-only. No production behavior, preset, molecule-cache,
  fixed-overwrite, source-state, selected-row projection, burden-ratio,
  infinity-norm, or row/species/element/case dropping rule is promotable.

## Milestone 9 Pilot 45:-5 Neutral Source-State Decomposition Addendum

The latest neutral source-state decomposition compact is:

- `results/fastchem_cond_kl_milestone9_pilot_45_m5_neutral_source_state_decomposition_compact.json`
- `results/fastchem_cond_kl_milestone9_pilot_45_m5_neutral_source_state_decomposition_compact.md`

Current status from that audit:

- No new broad case was generated. The audit consumes existing `45:-5` pilot
  artifacts and current-five comparison artifacts.
- The neutral source-state blocker decomposes to the molecule-cache
  full-element vector consumed at `iter1_full_reduced_system`.
- The top-species factorization is element-vector dominated for `H1Mn1`,
  `Cl2Na2`, `Fe1H2O2`, `H2Mg1O2`, `Cl2K2`, `H2S1`, `Al1F2O1`, and `O2V1`.
  hvector/logK and density-gauge deltas are zero for these species.
- The physical donor vector differs from the cache vector, while the iter1 RHS
  entry vector and full FastChem coherent RHS vector match the cache vector for
  the reported elements.
- `45:-10` has existing neutral species projection evidence and Al4C3 removed
  projection closure, but no emitted M8-style outside-selected full-vector
  decomposition. It remains a local projection closure case, not a source rule.
- The exact missing field is a per-neutral-species additive split into
  physical-donor, hidden-source-state, hvector/logK, and density-gauge budgets.
  The emitted full element vector is sufficient for the local verdict but not
  for promotion.
- Decision: `45:-5` neutral source blocker is molecule-cache full-element
  vector.
- Guardrail: diagnostic-only. No production behavior, preset, molecule-cache,
  source-state, hvector/logK, density-gauge, selected-row projection,
  burden-ratio, infinity-norm, or row/species/element/case dropping rule is
  promotable.

## Milestone 8 Pilot 45:-5 Full-Bundle Decomposition Addendum

The latest pilot full-bundle decomposition compact is:

- `results/fastchem_cond_kl_milestone8_pilot_45_m5_full_bundle_decomposition_compact.json`
- `results/fastchem_cond_kl_milestone8_pilot_45_m5_full_bundle_decomposition_compact.md`

Current status from that audit:

- No new broad case was generated. The audit consumes the existing `45:-5`
  pilot source and trace artifacts.
- The gap from the best partial replay (`PMI + Al4C3 removed`) to the full
  FastChem coherent RHS is neutral-molecule dominated: neutral molecule
  full-vector L2 `1.6367459746586315`, charged molecule L2
  `5.139755216542042e-14`, and unexplained residual after the molecule gap L2
  `4.8166936300605893e-14`.
- Selected-row RHS metrics and full-vector metrics remain separate. The
  selected-row max RHS gap after the partial replay is roundoff-scale
  (`7.105427357601002e-15`), while the full-vector neutral molecule gap is
  material.
- All `22` affected outside-selected rows are free-element rows. The largest
  rows are `Mn`, `S`, `K`, `Na`, `Mg`, `Fe`, `Cl`, and `Cu`.
- Species attribution is available from the pilot molecule-cache records. The
  largest neutral species are `H1Mn1`, `Cl2Na2`, `Fe1H2O2`, `H2Mg1O2`,
  `Cl2K2`, `H2S1`, `Al1F2O1`, and `O2V1`. `C1H4` and `H2O1` do not dominate
  the `45:-5` neutral gap after physical donor conversion.
- The top species have matching mass-action / hvector and density-gauge fields.
  The residual is traced to the full element source-state snapshot consumed by
  the neutral molecule cache.
- Decision: `45:-5` full bundle blocker is neutral molecule full-vector
  source-state coupling.
- Guardrail: diagnostic-only. No production behavior, preset, selected-row
  projection, donor, molecule, inventory, removed-tail, tau/complementarity,
  row-scaling, Jacobian, guarded-mode, burden-ratio, infinity-norm, or
  row/species/element/case dropping rule is promotable.

## Milestone 7 Pilot 45:-5 Blocker Attribution Addendum

The latest pilot attribution compact is:

- `results/fastchem_cond_kl_milestone7_pilot_45_m5_new_blocker_attribution_compact.json`
- `results/fastchem_cond_kl_milestone7_pilot_45_m5_new_blocker_attribution_compact.md`

Current status from that audit:

- No new broad case was generated. The audit uses the existing `45:-5` pilot
  source and trace artifacts.
- The row-wise ladder keeps the `embedded_broad_10row_projection` separate from
  focused regression. It reports `8` projected rows, `4` PMI rows better than
  baseline, `4` PMI rows worse than baseline, and `8` rows closed by the full
  FastChem coherent RHS.
- The remaining partial-replay gap is not explained by promoting the `Al4C3(s)`
  removed correction. The `Al4C3(s)` tail recurs, but projected closure is
  epsilon-dependent.
- The available delta-to-full decomposition attributes the blocker to a full
  coherent RHS source-state bundle requirement, dominated by neutral molecule
  full-vector provenance outside the selected rows that couples through the
  coherent solve.
- Decision: `45:-5` blocker requires full coherent RHS source-state bundle.
- Guardrail: diagnostic-only. No production behavior, preset, selected-row
  projection, donor, molecule, inventory, removed-tail, tau/complementarity,
  row-scaling, Jacobian, guarded-mode, burden-ratio, infinity-norm, or
  row/species/case dropping rule is promotable.

## Broad Projection Residual Decomposition Addendum

The latest compact broad residual decomposition is:

- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.md`

Current status from that audit:

- Metric-family lineage is explicit: focused closure remains tied to `focused_raw_result_provenance_metric`, while the broad replay remains `embedded_broad_10row_projection`.
- The embedded broad 10-row projection is not used as a focused regression. The broad projection remains a separate outside-selected/full-vector diagnostic probe.
- The focused one-sided frontier is closed diagnostically: original rows `12`, CH4 data-validity rows `5`, MgCO3/SiC FC-donor rows `2`, Group-B intentionally excluded rows `5`, final unresolved count `0`.
- Broad projection rows originally failing remain `40`. Focused-frontier levers annotate `12` rows, but only `2` are numeric broad failures; `38` broad rows still fail and are retained in the compact.
- Full-vector term differences are now emitted for all five broad cases. Each case is neutral-molecule dominated in the outside-selected/full-vector source-state attribution.
- The diagnostic broad replay protocol preserves `embedded_broad_10row_projection` and is not a focused regression. Combined available-term alignment reduces the broad projection residual in all five cases; beta/electron-only and row-scaling/Jacobian-only broad replays remain unavailable with exact missing fields recorded.
- Focused closure did not numerically reduce broad residual by itself because the focused levers close one-sided selected-row diagnostics, while the broad residual is an outside-selected/full-vector source-state probe.
- Decision: broad projection residual is dominated by neutral molecule full-vector source state.
- Guardrail: CH4 data-validity mask, FC donor snapshot, Group-B intentional exclusion, and repaired alpha/beta are diagnostic-only and non-promotable. No broad projection definition, selected-row rule, result-index rule, donor rule, electron rule, or production solver behavior changed.

## Neutral Molecule Full-Vector Provenance Addendum

The latest compact neutral-molecule provenance audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.md`

Current status from that audit:

- All `40` failing `embedded_broad_10row_projection` rows are retained and remain separate from the focused `focused_raw_result_provenance_metric`.
- Molecule RHS alignment alone worsens all `40` rows. Inventory/atom alignment alone also worsens all `40` rows. The removed/tau bundle reduces `3` cases, and combined available-term alignment improves all `40` rows.
- The current broad artifacts do not emit per-row/per-species neutral molecule contributors. The compact records the exact missing field `direct_numeric_broad_replay_cases[].neutral_molecule_species_contributions_by_projected_row` from the broad smoke artifact and marks it as a Python audit patch, not a C++ trace patch.
- Top-species source-stage lineage is also not emitted; the compact records `direct_numeric_broad_replay_cases[].neutral_molecule_species_stage_lineage` as missing.
- Decision: neutral molecule residual requires coherent molecule+inventory+removed/tau bundle.
- Guardrail: this is diagnostic-only. No molecule transplant rule, donor transplant rule, production electron rule, broad projection redefinition, selected-row rule change, or result-index rule change is promotable.

## Fixed-Element Source Decomposition Addendum

The latest compact fixed-element source-decomposition audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.md`

Current status from that audit:

- Molecule cache remains a downstream symptom only. The audit stays at the FastChem cached full-element vector consumed before molecule refresh.
- The cached FastChem vector remains source-proven at `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`, stage `iter0_post_correctValues_full_element_vector_before_molecule_refresh`.
- The exact source decomposition available from code is narrower than a physical additive budget split: each full-element entry is either a reduced-element overwrite coming from `correctValues` / `correctValuesFull` or a carry-forward full-element value. On that explicit source decomposition, component sum to final cached value is exact.
- The requested additive subcomponents are still not explicit in source at that stage and are recorded as missing rather than guessed: `free_atomic_gas_component`, `condensed_or_fixed_correction_component`, `total_inventory_component`, and `electron_specific_component`, all from `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`.
- On the KL side, the analogue stages remain available for `gas_only_final`, `post_selectActiveCondensates_reset`, `post_correctValues_update`, `exact_postCorrectValues_refreshed_all_active_state`, `gas_recoupling`, `gas_replay_final_proxy`, and `iter1_RHS_entry`, but KL still does not emit a separable fixed-element membership consumer or a source-clean `phi` / degree handoff component at this molecule-input boundary.
- The fixed-only FastChem overwrite replay is still the only single-component handoff that materially improves molecule reconstruction on both focused layers:
  `45:-10`: `147.92838359730413 -> 15.226673958408794`
  `60:-5`: `140.48218322609824 -> 19.407748535067963`
  Electron-only replay remains near baseline, while overwrite + electron improves molecule density further but still does not produce a coherent selected-row RHS replay.
- The fixed-subset divergence still appears before `post_selectActiveCondensates_reset` on both focused layers, and the iter1 RHS-entry residual concentration remains fixed-element dominated with `e-` as the largest single non-fixed residual.
- Decision: mismatch is dominated by fixed-element overwrite values.
- Guardrail: fixed-element handoff is source-proven but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Fixed-Element Overwrite Provenance Addendum

The latest compact fixed-element overwrite provenance audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.md`

Current status from that audit:

- The exact FastChem overwrite source is still `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` at `iter0_post_correctValues_full_element_vector_before_molecule_refresh`.
- For fixed-by-condensation elements, the cached full-element value is a full overwrite from `elem_densities_new[i]` via `elements_cond[i]->number_density = elem_densities_new[i]`. In the focused smoke the overwrite path is the reduced solver path, so the overwrite originates from `correctValues`, not from a later molecule refresh or another state update.
- The overwrite is source-proven as a full replacement, not a partial modification. The numeric prior full-element carry-forward value at the exact write point is still not emitted in the compact artifact, so the audit records the exact missing local variable instead of guessing it: `full_element_densities_before_write[i]` in `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`.
- KL analogue stages remain available at `gas_only_final`, `post_selectActiveCondensates_reset`, `post_correctValues_update`, `exact_postCorrectValues_refreshed_all_active_state`, `gas_recoupling`, `gas_replay_final_proxy`, and `iter1_RHS_entry`, but KL still does not expose an overwrite-like fixed-element consumer, explicit fixed-index bookkeeping, or a carry-forward vs overwrite distinction at this boundary.
- Overwrite-only replay remains the decisive single-component ladder rung:
  `45:-10`: `147.92838359730413 -> 15.226673958408794`
  `60:-5`: `140.48218322609824 -> 19.407748535067963`
  Overwrite + electron improves molecule density further (`0.4347374576125725`, `0.565493860944129`) but still does not produce a coherent selected-row RHS replay, while full FC cached vector remains the closing upper bound.
- Earliest overwrite divergence remains `before post_selectActiveCondensates_reset` on both focused layers.
- Decision: mismatch is dominated by overwrite values themselves.
- Guardrail: overwrite provenance is source-proven but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Fixed-Element Materialization Boundary Addendum

The latest compact fixed-element materialization-boundary audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.md`

Current status from that audit:

- The KL gas-recoupling output is present and source-visible, but it is not the vector consumed later for molecule reconstruction on the focused layers. The exact pre-gas vector is still not emitted as a standalone compact field, so the audit records the exact missing field instead of inferring it: `gas_trace.post_condensed_phase_fixed_atomic_element_species_state` from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::actual_fastchem_like_coupled_loop`.
- The diagnostic gas replay path explicitly adopts `gas_result.ln_nk` into both `result.ln_nk` and `gas_only["ln_nk"]`, but the first later molecule consumer is `_second_post_seed_update_actualization_solve`, which forwards `gas_only["ln_nk"][:n_elem]` into `_assemble_fastchem_reduced_update`.
- `_assemble_fastchem_reduced_update` then rebuilds the atom/full-element vector inline from its current `u` by `atom = exp(clip(u))`. The focused entrance-smoke trace shows this inline molecule-input vector matches the later `post_correctValues_update` proxy exactly and differs materially from the KL gas-recoupling output on both focused layers.
- No explicit fixed-element overwrite consumer is exposed before molecule reconstruction. The audit records the exact missing boundary object instead of guessing it: `iter1_molecule_input.fixed_element_bookkeeping_consumer` from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- The counterfactual ladder keeps the selected-row metric and the established overwrite boundary:
  `45:-10`: current `147.92838359730413`, gas recoupling `149.20929529868224`, current + FC fixed overwrite `15.226673958408794`, current + FC fixed overwrite + electron `0.4347374576125725`, full FC cached input `0.08475871276711094`, direct FC molecule replay `0.08475871276711094`.
  `60:-5`: current `140.48218322609824`, gas recoupling `140.07824601545863`, current + FC fixed overwrite `19.407748535067963`, current + FC fixed overwrite + electron `0.565493860944129`, full FC cached input `1.3724132941206335e-11`, direct FC molecule replay `1.3724132941206335e-11`.
- A direct gas-recoupling carry and synthetic fixed-overwrite ladders remain diagnostic-only. Full FC cached input is still required for the known upper bound.
- Decision: KL explicit fixed-overwrite consumer is missing before molecule reconstruction.
- Guardrail: KL fixed-element materialization boundary is source-proven but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Synthetic Fixed-Overwrite Consumer Addendum

The latest compact synthetic fixed-overwrite consumer audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.md`

Current status from that audit:

- The gas-recoupling output adoption result is unchanged: the diagnostic gas replay path adopts `gas_result.ln_nk` into `result.ln_nk` and `gas_only["ln_nk"]`, but the later molecule-input boundary that matters is still the inline-recomputed `u` consumed by `_assemble_fastchem_reduced_update`.
- The missing consumer result is also unchanged and remains explicit: `iter1_molecule_input.fixed_element_bookkeeping_consumer` is still absent from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- The synthetic materialization ladder now tests the gas-recoupling fixed subset directly at that later molecule boundary. On both focused layers, the best KL-side synthetic rung remains the unchanged current KL vector `A`, and the gas-fixed variants do not improve the selected-row replay:
  `45:-10`: `A=36.90705832854512`, `C=36.90706738203272`, `D=36.90706738203608`, `H=0.08475871276711094`.
  `60:-5`: `A=50.41942341750188`, `C=50.41942442202005`, `D=50.41942442400552`, `H=1.3724132941206335e-11`.
- The fixed-subset adoption result is therefore negative for gas-recoupling values. The emitted classifications are the same on both focused layers:
  fixed values = `gas-recoupling fixed values do not explain the FC overwrite values`
  electron = `electron is negligible`
- Synthetic consumer replay remains informative because it separates the gas-recoupling fixed subset from the already-proven FastChem overwrite boundary, but it does not create a closer KL-side carry or overwrite rule. Full FC cached input remains the only coherent upper bound.
- Decision: gas-recoupling fixed values are insufficient even with a synthetic consumer.
- Guardrail: synthetic fixed-overwrite consumer is informative but not promotable. No guarded mode, production solver fix, or promotable rule was added.

## Elem Densities New Source Addendum

The latest compact `elem_densities_new` source audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.md`

Current status from that audit:

- The synthetic fixed-overwrite consumer result remains the entry condition for this step: gas-recoupling fixed values are still insufficient, so the next object is the FastChem source construction of the overwrite values themselves.
- The focused `CondensedPhase::correctValues` trace now emits the exact per-element construction chain used to write the later overwrite values for fixed rows:
  `elem_number_dens_old[i]`
  `result(i + nb_cond_jac)`
  `delta_n_elem`
  `update_factor`
  `elem_number_dens_new[i]`
  with caller storage later observed as `elem_densities_new[i]`.
- On both focused layers, every fixed row emitted at that stage classifies as `pure overwrite`, with no `carry-forward only`, `overwrite + correction`, or `mixed` fixed rows:
  `45:-10`: `22` pure-overwrite fixed rows
  `60:-5`: `23` pure-overwrite fixed rows
- No dedicated additive component locals are emitted for:
  `free_gas_carry_forward_term`
  `condensed_or_fixed_correction_term`
  `electron_specific_term`
  The audit records those as exact missing locals rather than inferring them numerically.
- The component ladder therefore stays source-clean and narrow. Overwrite-term-only replay still gives the already-known molecule improvement but not selected-row closure:
  `45:-10`: current `147.92838359730413`, overwrite-term-only `15.226673958408794`, full FC cached input `0.08475871276711094`
  `60:-5`: current `140.48218322609824`, overwrite-term-only `19.407748535067963`, full FC cached input `1.3724132941206335e-11`
- The dominant source decision is now stricter than the earlier overwrite-source wording: mismatch is dominated by the overwrite term itself.
- The earliest divergence for that dominant source component remains `before post_selectActiveCondensates_reset` on both focused layers.
- Guardrail: `elem_densities_new` source is further resolved but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Reduced-Newton Result Slot Addendum

The latest compact reduced-Newton result-slot audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.md`

Current status from that audit:

- The write-site provenance result is unchanged upstream: the overwrite term itself still dominates `elem_densities_new[i]`, so the next unresolved object is the fixed-row reduced-Newton slot that eventually appears at `result(i + nb_cond_jac)`.
- For each fixed row the audit now emits the exact same-iteration reduced-system slot index, scaled RHS entry, solver result entry before any caller-side/global rescaling, row scaling factor, reduced Jacobian diagonal, and the later `correctValues` write-site chain:
  solver result slot before global scaling
  `result(i + nb_cond_jac)`
  `delta_n_elem`
  `update_factor`
  `elem_number_dens_new[i]`
- Exact same-iteration row and column labels still are not emitted in the reduced-newton anatomy trace. The audit records those exact missing fields and uses the nearest labelled iter1 analogue only for structural classification.
- No explicit same-iteration retained-condensate, removed-condensate, fixed/condensed correction, or electron-coupling local exists at the fixed-row write site. The audit records the exact missing locals:
  `retained_condensate_delta_local_variable`
  `removed_condensate_delta_local_variable`
  `fixed_or_condensed_correction_local_variable`
  `electron_specific_local_variable`
- A same-iteration caller-side bridge does exist in source: `global_scaling_factor` in `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`. Replaying the exact same-iteration solver result slot with that bridge reproduces the raw-result-derived overwrite rung on both focused layers, while the unbridged pre-global-scaling slot is materially worse.
- The structural replay ladder therefore narrows the dominant component further:
  `45:-10`: molecule mean log residual `147.92838359730413 -> 15.226673958408796` after the explicit global-scaling bridge, versus `22.443213083775767` from the unbridged slot
  `60:-5`: molecule mean log residual `140.48218322609824 -> 19.40774853506796` after the explicit global-scaling bridge, versus `26.494334739330682` from the unbridged slot
  Full FC cached input / direct FC replay remain the selected-row upper bound:
  `0.08475871276711094`
  `1.3724132941206335e-11`
- Decision: mismatch is dominated by missing global-scaling bridge after solve.
- Guardrail: reduced-Newton result slot provenance is further resolved but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Fixed-Row Solve-Space Addendum

The latest compact fixed-row solve-space audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.md`

Current status from that audit:

- The reduced-Newton global-scaling bridge result remains the immediate upstream fact: the caller-side `global_scaling_factor` reproduces the write-site `result(i + nb_cond_jac)` exactly on both focused layers.
- The next unresolved object is the fixed-row solve-space equation itself. The audit now emits, for each fixed row, the exact same-iteration slot index, scaled RHS entry `b_i`, unbridged solve result `z_i`, caller bridge `g`, bridged value `g*z_i`, row scaling factor, and solve-space diagonal `J_ii`.
- Exact same-iteration labelled matrix rows and row/column labels are still not emitted from `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep`. The audit records the exact missing locals and fields:
  `jacobian`
  `rhs`
  same-iteration labelled row label
  same-iteration labelled column label
- The emitted labelled iter1 full reduced-system analogue is sufficient to decompose the fixed-row equation structurally. On the focused layers the row equations close to roundoff on a relative basis, and the Schur-style `z_i` attribution can be reported as:
  RHS-row-only contribution
  condensate-coupling-induced contribution
  other-element-coupling-induced contribution
- No single solve-space component closes the replay. The bridged full fixed-row slot still beats any isolated solve-space component, while the best isolated additive candidate is the RHS-row-only contribution:
  `45:-10`: bridged slot `15.226673958408796`, RHS-only `15.977102788049098`, condensate `16.19561867754053`, other-element `15.924659915242676`
  `60:-5`: bridged slot `19.40774853506796`, RHS-only `20.027064857591796`, condensate `20.28568059122682`, other-element `20.357508000322714`
- Because the bridged full slot is still materially better than any single RHS/coupling component, the solve-space decision remains mixed rather than collapsing to one component family.
- Decision: mismatch is dominated by mixed solve-space coupling in fixed rows.
- Earliest divergence: inside reduced Newton solve-space assembly.
- Guardrail: fixed-row solve-space provenance is further resolved but remains diagnostic-only. No guarded mode, production solver fix, or promotable rule was added.

## Coherent Gas-State Bundle Addendum

The latest coherent-bundle compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_coherent_gas_state_bundle_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_coherent_gas_state_bundle_compact.md`

Current status from that audit:

- The baseline extractor bug stays resolved. The new ladder keeps the selected-row mapping metric and reproduces the prior `baseline_ABC` values at layer `45:-10` = `0.5620237706783301` and layer `60:-5` = `0.914249889596486`.
- Inventory gauge normalization remains source-proven but diagnostic-only. Molecule + inventory + atom still recovers the prior cancellation after gauge normalization, and layer-45 removed correction remains source-proven and separate.
- Under the coherent FC Jacobian and scaled RHS framework, the compact ladder closes in the established order: activity + burden helps, complementarity tightens to `baseline_ABC`, the FC molecule term alone is destructive, the FC inventory/atom bundle plus molecule closes layer `60:-5`, and exact removed closes layer `45:-10`.
- The iter1 RHS-entry bundle is now emitted compactly on both sides for atom rows, molecule-cache provenance, normalized/physical inventory rows, total-element-density scalar, row scaling, active-burden rows, and removed rows. FastChem carries a full element vector used for molecule reconstruction; the KL entrance-smoke trace still does not emit a symmetric full element vector or per-molecule mass-action constant ledger.
- Bundle self-consistency is mixed exactly where expected: FastChem and KL molecule-burden cache closure each self-close to roundoff; all bundle terms add back to the element RHS rows; inventory gauge identities are exact on the emitted basis where both factors are present; but KL-native molecule reconstruction remains blocked by missing symmetric full-element-vector / mass-action fields, and removed-per-condensate expansion stays diagnostic-only.
- The current coherent-bundle decision is therefore still conservative: the full FC coherent gas-state bundle closes, but KL-native reconstruction is not source-proven from the emitted entrance-smoke fields. Full FastChem gas-state bundle remains diagnostic-only; no smaller KL production rule is promotable.

## KL-Native Molecule Reconstruction Addendum

Molecule refresh-timing and FastChem mass-action ledger audit:

- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_entrance_smoke.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_entrance_smoke_traces.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_compact.md`

Current refresh-timing status:

- The FastChem per-molecule mass-action ledger is now emitted at refresh-stage cached-density records and at iter1 RHS/Jacobian entry. The source path is `Molecule::calcMassActionConstant`, carried through `diagnostic_trace.h::append_molecule_density_vector` and `append_iter1_full_reduced_system`.
- FastChem molecule formula self-closure now succeeds from emitted FC full element vector plus FC mass action: max absolute closure error `5.551115123125783e-17` on both focused layers.
- KL-side fields still pass the prior gate and KL formula self-closure remains exact enough. The new unresolved field is narrower: a distinct KL cached-refresh molecule snapshot is still not emitted because `_assemble_fastchem_reduced_update` computes the molecule cache inline at RHS/Jacobian assembly.
- Complete FC-ledger cross-state tests do not produce a KL-native closing rule. KL full vector + FC mass-action remains non-closing (`36.90705832854512` / `50.41942341750188` when paired with gauge-normalized inventory/atom), FC full vector + KL mass-action is destructive, and the source-clean FC formula reconstruction closes molecule log densities but is not a robust solve-space replay because exponentiated burden differences are numerically amplified.
- Direct FC cache remains the only robust molecule vector that reaches the known cancellation boundary (`0.08475871276711094` on layer `45:-10` before exact removed and `1.3724132941206335e-11` on layer `60:-5`). The KL cached-refresh variants are unavailable, not promotable.

Decision: KL-native molecule timing provenance remains mixed or inconclusive.

Fresh field-completion audit:

- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_entrance_smoke.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_entrance_smoke_traces.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_compact.md`

Fresh status:

- The previous KL-native molecule reconstruction audit was blocked by missing symmetric KL fields. The fresh instrumentation now emits the KL RHS-entry full element vector, per-molecule `mass_action_constant=-hmol`, RHS-consumed molecule cache, and an explicit inline refresh-equivalent record from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- KL has no discrete molecule-cache-refresh stage in this audit path. The cache is computed inline at RHS/Jacobian assembly from `u`, `A_mol = formula_matrix_gas[:, n_elem:]`, and `hmol = hvector_gas[n_elem:]`.
- The fresh field-presence gate passes on both focused layers. KL molecule formula self-closure is exact enough for replay: max absolute closure error `2.2737367544323206e-13` on both `45:-10` and `60:-5`.
- Cross-state replay does not recover molecule/inventory cancellation from KL-native fields. KL full vector + KL mass-action leaves paired selected-row residuals `36.907067382036104` at `45:-10` and `50.419424423672645` at `60:-5`. Applying the proven density-gauge bridge does not close either layer.
- Direct FC molecule cache remains the only tested molecule vector that recovers the known cancellation boundary with gauge-normalized inventory/atom: `0.08475871276711094` at `45:-10` before exact removed and `1.3724132941206335e-11` at `60:-5`. Variants requiring FC per-molecule mass-action constants remain unavailable because the fresh trace still does not emit `FastChem.iter1_RHS_assembly_entry.mass_action_constants_by_molecule`.
- The molecule boundary is now source-clean on the KL side but still diagnostic-only. No KL-native molecule reconstruction rule is promotable, and no guarded mode or production solver fix is justified.

Decision: molecule mismatch is dominated by molecule-cache refresh timing.

The latest compact KL-native molecule reconstruction audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_reconstruction_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_reconstruction_compact.md`

Current status from that audit:

- The next unresolved object stays exactly where the coherent-bundle audit left it: KL-native molecule cache / mass-action / full-element-vector reconstruction. Inventory gauge normalization, molecule/inventory cancellation after gauge normalization, and the layer-45 removed tail stay source-proven provenance only.
- FastChem still exposes the iter1 RHS-entry molecule cache and the full element vector consumed by that cache. The FastChem cache remains stable from `after_iter0_calcNumberDensity_refresh` through iter1 `assembleJacobian`, so there is no new evidence for an internal FastChem cache drift within the focused entrance smoke.
- KL still does not emit the symmetric fields needed for a source-clean replay at the required stage. The missing fields are:
  `KL.iter1_RHS_assembly_entry.full_element_vector_used_for_molecule_reconstruction`
  `KL.iter1_RHS_assembly_entry.mass_action_constants_by_molecule`
  `KL.full_element_vector_at_molecule_cache_refresh`
  `KL.mass_action_constants_by_molecule_at_molecule_cache_refresh`
  all from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- Prior molecule-density provenance rows remain intact: both sides still self-close their own molecule-density formulas to roundoff in the earlier per-molecule audits, and both sides still self-close molecule-burden aggregation from the emitted iter1 molecule caches. What remains unresolved is not the local formula algebra, but the cross-state reconstruction inputs at the refresh / RHS-entry boundary.
- In the coherent selected-row replay, the direct FC molecule cache paired with gauge-normalized inventory/atom still gives the known post-complementarity closure boundary:
  layer `45:-10` = `0.08475871276711094`, then exact removed closes to `0`
  layer `60:-5` = `1.3724132941206335e-11`
  No KL-native branch is available from the emitted fields, and no smaller KL production rule is promotable.
- The compact audit therefore upgrades the molecule decision from bundle-level ambiguity to a sharper molecule-level boundary: the currently closing molecule replay requires a FastChem-only cache / full-element-vector snapshot that KL does not emit symmetrically in this smoke.

Decision: molecule state requires FastChem hidden/coupled snapshot.

## Baseline Reconciliation / Inventory Gauge Addendum

The compact baseline-reconciliation and inventory-gauge audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_inventory_gauge_baseline_reconciliation_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_inventory_gauge_baseline_reconciliation_compact.md`

New source-proven status from that audit:

- Fresh exact total-inventory rows remain present on both sides and both focused layers, so the prior missing-row blocker is resolved for this entrance-smoke audit path.
- The fresh baseline_ABC mismatch is not a new physical RHS effect. Common-code reconstruction from the fresh entrance-smoke artifact reproduces the prior post-complementarity selected-row residuals exactly:
  layer `45:-10` = `0.5620237706783301`
  layer `60:-5` = `0.914249889596486`
- The larger fresh compact baseline values (`3.298714708385887`, `10.29909704108286`) came from an extractor/metric mismatch: selected-row mapping was dropped and the compact fell back to a full-vector infinity norm. The exact mismatch reason is therefore `extractor_bug`, not a changed baseline physics term.
- Inventory provenance is now split cleanly by gauge. KL `explicit_budget_vector` rows match FastChem normalized inventory (`epsilon`) to roundoff, and physicalizing KL budget rows by the FastChem `total_element_density` reproduces the FastChem physical total-inventory rows to roundoff. The apparent FastChem/KL inventory disagreement is therefore a gauge/normalization issue in this audit, not a newly proven epsilon drift.
- Solve-space comparison says the scaled inventory mismatch is explained by gauge normalization, but that gauge conversion is diagnostic-only. It does not become a production solver rule.
- Post-complementarity replay returns to the prior pattern once the baseline metric and inventory gauge are handled consistently:
  layer `60:-5` closes with molecule + inventory + atom,
  layer `45:-10` nearly closes with molecule + inventory + atom and then closes to roundoff after exact removed contribution replay.
- Layer `45:-10` removed tail is therefore source-proven and separate in the fresh exact-source audit path, while layer `60:-5` removed replay is `not_applicable` because both removed sets are empty.

Production implication:

- No guarded mode, solver fix, row-scaling rule, inventory rule, or smaller replay rule is promotable.
- Gauge conversion is source-proven but diagnostic-only.
- Full gas-state bundle remains diagnostic-only; no smaller production rule is promotable.

| Lever | FastChem source path | KL audit mode | Equation or rule | Metric impact | KL production candidate | RGIE/PIPM transfer | Confidence |
|---|---|---|---|---|---|---|---|
| Exact total-inventory / removed-source trace frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleRightHandSide`; KL analogue in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` | `phase2_total_inventory_removed_source_compact` | Emit exact per-element total-inventory RHS rows and per-removed per-element analytic correction rows from the local RHS builder variables, without changing solver behavior | Fresh entrance-smoke artifacts now carry the emitted rows. Field-presence gate passes: FastChem total-inventory rows are `22/23` on the focused layers, KL total-inventory rows are `28/28`, and layer `45:-10` FastChem removed rows are present and self-close. Exact total-inventory formula closure succeeds to roundoff on both sides, but FastChem-vs-KL provenance remains mixed because KL exposes exact `budget[element]` rows rather than separate `total_element_density` and `epsilon` factors. Layer-45 removed-tail provenance is source-proven and separate. | No; audit-only. Exact source closure improves provenance, but the remaining closing replay still requires the full coherent gas-state bundle | Transfers as a stronger provenance boundary: use exact emitted RHS-consumed rows, not shared gas diagnostics, and keep the full gas-state bundle diagnostic-only | High for field presence and self-closure; medium-high for provenance because cross-side inventory factorization is still mixed |
| maxDensity total-inventory source | `fastchem/fastchem_src/condensed_phase/condensed_phase.cpp::CondensedPhase::selectActiveCondensates` and `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` | `actual_condensed_phase_exact_maxdensity_total_inventory_proven` | `maxDensity_c = min_j n_elem,total_j / nu_cj` over the physical total inventory used by FastChem, not a normalized KL-only gas inventory | Proven material upstream alignment lever before seed/update attribution | Yes, after isolating from audit-only scaffolding | Transfers as candidate upper-bound semantics for RGIE/PIPM working-set initialization, not as a barrier update | High |
| Full row materialization before thresholding | `fastchem/fastchem_src/condensed_phase/condensate_struct.cpp::Condensate::calcActivity` and prescan/candidate trace path | `actual_condensed_phase_exact_prescan_full_row_materialization_proven` | Materialize all candidate rows and their activity/source state before threshold or support pruning | Removes false attribution from absent rows and makes raw-sequence comparison possible | Yes, as deterministic candidate-row construction | Transfers as full support ledger before sparse RGIE/PIPM filtering | High |
| Pressure/density gauge bridge | `fastchem/fastchem_src/calc_densities.cpp` gas-density and total-density trace path | `actual_true_kl_atomic_branch_exact_density_gauge_bridge_proven` | Convert KL normalized atomic branch to the FastChem physical density gauge before comparing donor, activity, and maxDensity quantities | Exposed the donor scalar as the next meaningful source mismatch | Yes, as an explicit unit/gauge conversion boundary | Transfers as a required physical-density bridge before RGIE/PIPM comparisons | High |
| Physical atomic donor scalar conversion | FastChem gas-only density output and ExoGibbs `gas_only['ln_nk']` audit comparison | `actual_true_kl_atomic_branch_exact_physical_atomic_donor_from_gas_only_proven` | Convert normalized gas-only atomic densities to exact physical atomic element-species densities before condensed candidate evaluation | Removed the prior donor-scale mismatch and allowed seed/update attribution | Yes, if implemented as an explicit input conversion, not hidden in solver logic | Transfers as a pre-solver state conversion for RGIE/PIPM, not as a changed objective | High |
| Immediate seed `n=maxDensity`, `lambda=1` | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` entry seeding records | `actual_true_kl_atomic_branch_exact_first_post_seed_update_proven` upstream seed diagnostics | `n_old = maxDensity`, `lambda_old = 1` for newly active condensates at condensed-solve entry | Immediate seed now matches raw FastChem sequence | Yes, as lifecycle initialization semantics | Transfers as initialized active-set state for RGIE/PIPM warm starts | High |
| First correctValues update | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate -> solver.newtonStep -> CondensedPhase::correctValues` | `actual_true_kl_atomic_branch_exact_first_post_seed_update_proven` | Retained rows take `delta_n_cond` from the reduced result; removed rows use the analytic eliminated-row formula; clip `delta_n_cond`; cap `n`; update `lambda` with clipped `delta_lambda` | Materially reduced the first post-seed mismatch and moved the exposed blocker to the next condensed update | Candidate only after the reduced Newton assembly is fully reconciled | Transfers as a diagnostic update formula; RGIE/PIPM should not inherit FastChem clipping blindly | High |
| Second correctValues update status | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate -> solver.newtonStep -> CondensedPhase::correctValues` | `actual_true_kl_atomic_branch_exact_second_post_seed_update_proven` | Same algebra as iter 0; iter 1 changes the old state and reduced Newton inputs before calling `correctValues` again | CorrectValues/cap/clipping/lambda algebra is validated for shared rows: `FC_old + FC_delta` through KL correctValues closes to machine precision, and the iter-1 lambda gap is eliminated | Not yet. CorrectValues algebra is captured, but iter-1 state refresh and reduced Newton provenance remain the blocker | Transfers only as a proven audit invariant: the formula is reused, but the old-state refresh and reduced step must be solved natively | High for formula, medium for full density outcome |
| Iter-1 delta replay diagnostic | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` and `calculate.cpp::CondensedPhase::correctValues` trace records | `actual_true_kl_atomic_branch_exact_iter1_delta_n_replay_proven` | Replay FastChem traced iter-1 `delta_n_cond` through the existing guarded KL correctValues formula | Negative diagnostic result: focused mean log-density mismatch worsened from `0.485277` to `0.732594`, while lambda mismatch stayed `0`; delta formula alone is not sufficient | No direct production transplant; it must not become a solver rule | Transfers as a debugging method for reduced-step anatomy, not as an RGIE/PIPM update rule | High for negative result |
| Iter-1 old-state / cap / mapping frontier | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` and `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` | no production mode; old-state provenance audit only | Decompose `log(n_new_FC)-log(n_new_KL)` into `log(n_old_FC)-log(n_old_KL)`, `delta_n_FC-delta_n_KL`, and cap/mapping residual | Current frontier after negative delta replay; focused rows classify mostly as old-state residual, with row coverage limits on KL-only candidates | Not yet; requires a proven old-state handoff rule before any guarded mode | Transfers to RGIE/PIPM as state-handoff provenance checks before comparing reduced directions | Medium |
| Iter-1 state-refresh frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` `condensed_phase_reduced_newton_anatomy`, `condensed_phase_iter1_old_element_mapping`, and `condensed_phase_iter1_old_log_activity_closure`; `calculate.cpp::CondensedPhase::correctValues` raw `post_correctValues_update` | no production mode; state-refresh provenance audit only | Trace `post iter0 correctValues -> pre iter1 newtonStep` for `n_old`, `lambda_old`, `log_activity`, mapped full-global old element densities, partition, and row mapping | FastChem iter1 element-density mapping is now available for the focused smoke: 22/28 mapped elements at `45:-10`, 23/28 at `60:-5`; FastChem old log-activity closure is available for 13 focused rows with max closure error `0`; dominant old-state counts remain `log_activity_old_mismatch=12`, `inherited_iter0_n_old_mismatch=5`, `row_mapping_or_alignment_mismatch=3`; KL/FastChem donor decomposition remains limited for rows whose KL contracted element basis does not align candidate-wide | No; audit-only until the exact state refresh rule is source-proven and candidate-wide | Transfers as a warning that RGIE/PIPM must verify refreshed log activity and atomic element-density handoff before comparing reduced directions | Medium; FastChem-side closure is high-confidence, candidate-wide KL alignment remains limited |
| Iter-1 reduced-system row-universe frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` `condensed_phase_reduced_newton_anatomy`; KL audit reduced iteration diagnostics in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py` | no production mode; reduced-system row-universe audit only | Compare full iter1 active row sets, shared rows, retained/Jac rows, removed rows, reduced unknown ordering, and projected reduced RHS/Jacobian/scaling/result vectors | FastChem has `219` total active rows across the focused smoke and KL has `234`; all `15` KL-only rows participate in RHS/Jacobian/row scaling. Excluding KL-only rows improves layer `45:-10` mean log-n mismatch from `1.031812` to `0.451817`; adding the FC partition improves it to `0.354056`. Layer `60:-5` does not improve under shared-only restriction. | No; audit-only until the exact row-universe and reduced-system assembly rule is source-proven and closes focused mismatch | Transfers as a candidate-row-universe warning: RGIE/PIPM must align active candidate sets before interpreting reduced directions, partitions, or row scaling | Medium; row-universe effect is proven in layer 45, but it is insufficient |
| Iter-1 row-universe replay diagnostic | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` prescan candidate trace and `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` reduced-system anatomy | `actual_true_kl_atomic_branch_exact_iter1_row_universe_replay` | At iter1 reduced Newton only, restrict the KL diagnostic row universe to the FastChem iter1 active row set; preserve seed, first update, and second `correctValues` algebra | Replay improves aggregate shared-row mean log-n mismatch from `0.6477089522` to `0.4598570508` with lambda mismatch `0`; layer `45:-10` improves from `1.031812` to `0.451817`, while layer `60:-5` worsens from `0.263606` to `0.467898`. All `15` KL-only rows classify as `never_candidate_in_fastchem` from the prescan candidate trace and `missing_fastchem_eviction_rule` on the KL side. Projected RHS, row-scaling, and solver-result differences remain large. | No; diagnostic-only. The row universe helps but is not sufficient and must not be promoted as a naive row drop. | Transfers as a strict audit guard: align row universes first, then audit RHS/Jacobian/scaling rather than conflating row contamination with assembly differences | Medium-high for row presence/exclusion provenance; medium for reduced-system causality because full global RHS/Jacobian/scaling replay is still unavailable |
| Iter-1 full reduced-system assembly trace | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` `condensed_phase_iter1_full_reduced_system`; KL trace in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_second_post_seed_update_actualization_solve` | no production mode; full reduced-system trace and labelled alignment only | Emit labelled row/column bases, RHS before/after row scaling, row scaling, full Jacobian before/after scaling, solver result, and solver-result-to-`delta_n_cond` mapping for iter1 | Full matrices are now emitted for the two-layer entrance smoke. Label alignment finds no FastChem-only reduced rows/columns, but KL has `13` unmatched rows/columns at `45:-10` and `14` at `60:-5`. Matched-entry residuals are large: layer 45 RHS mean abs `1.384750`, row-scaling mean abs `7.3375e16`, solver-result mean abs `1807.030749`; layer 60 remains non-isomorphic as well. Exact RHS/Jacobian/scaling/result replay is therefore still blocked by a missing source-proven projection/isomorphism rule, not by absent matrix dumps. | No; audit-only until the labelled reduced systems are source-proven isomorphic or an exact projection rule closes focused mismatch | Transfers as a reduced-system provenance requirement: RGIE/PIPM must compare labelled systems, not positional arrays, and must treat unmatched rows/columns as first-class blockers | Medium-high for trace availability; medium for causal attribution because component replay remains trace/projection limited |
| Iter-1 projection/isomorphism and scaling frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` full reduced-system trace; KL labelled full reduced-system trace in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py` | no production mode; projection/isomorphism audit only | Build shared labelled subspaces; test drop-only, freeze, Schur complement, and source-labelled KL-only-condensate projections; compare scaling variants by solving the projected systems | Projection is safe but not sufficient. Drop/freeze/source-labelled projections solve with condition numbers `1.3433e4` at `45:-10` and `1.4351e4` at `60:-5`, but focused log-n mismatch worsens to `1.5367` and `2.5021`; Schur complement is solvable but worse (`2.4337`, `2.1889`) and more conditioned (`1.5121e6`, `4.4298e6`). Row scaling is numerically active: FastChem/common scaling reproduces FastChem solver results to about `1e-11`, while KL scaling differs by `776.7` and `1168.2` in mean solver-result residual. | No; audit-only. A projection rule alone is not a solver fix, and row scaling/conditioning needs source proof before actualization | Transfers as a warning that row/column isomorphism must be paired with scaling/conditioning checks before comparing Newton directions | Medium; projection and scaling diagnostics are clear for the focused smoke, but no exact guarded mode closes the mismatch |
| Iter-1 row-scaling and solver-conditioning frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`, `assembleRightHandSide`, and `solveSystem`; KL audit `_safe_row_scaling`, `_solve_projected_system`, and projected replay helpers in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py` | no production mode; row-scaling audit only | FastChem uses `scaling_factors = jacobian.rowwise().maxCoeff()`, then divides both Jacobian rows and RHS entries by that factor. KL audit uses row max with an absolute-row fallback when the signed max is too small, then divides both Jacobian rows and RHS entries. FastChem solves with `Eigen::PartialPivLU` by default; KL audit uses `numpy.linalg.solve` with `lstsq` fallback. | The source formulas are now recorded. On projected shared systems, FastChem scaling, common FastChem-derived scaling, no scaling, diagonal-equilibration scaling, and row-norm scaling reproduce the FastChem projected solve to roughly `1e-11`; the unscaled systems are highly conditioned (`1.4484e18`, `1.8776e20`) but still reproduce FastChem in this diagnostic. Fixed-scaling swaps show RHS changes alone are smaller than Jacobian changes, while KL Jacobian under good scaling reproduces the prior solver-result residual scale (`~776.7`, `~1168.2`). The focused log-lambda mismatch remains `0`. | No; audit-only. A FastChem-equivalent row-scaling rule was not promoted because the projected solve still carries assembly/Jacobian residuals and does not close focused density mismatch. | Transfers as a reduced complementarity conditioning requirement: RGIE/PIPM must audit row scaling, conditioning, and solver backend before treating reduced directions as comparable | Medium-high for source formula and scaling replay; medium for causality because row scaling exposes, but does not alone close, the remaining projected density residual |
| Iter-1 reduced Jacobian/RHS block assembly frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian` and `assembleRightHandSide`; KL audit `_assemble_fastchem_reduced_update` and projected block replay helpers in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py` | no production mode; labelled block decomposition only | Decompose projected shared iter1 reduced systems into retained condensate diagonal, condensate-row/element-column stoichiometry, element-row/condensate-column density derivatives, and element-row/element-column atom+molecule+removed-condensate fold-in blocks. Solve block swaps under fixed FastChem row scaling. | Condensate-row/condensate-column and condensate-row/element-column blocks match exactly in the focused smoke. The dominant block residual is element-row/element-column: layer `45:-10` FC norm `9.0946e18`, KL norm `9.9625`, diff `9.0946e18`; layer `60:-5` FC norm `1.2651e18`, KL norm `9056.686`, diff `1.2651e18`. Element-row/condensate-column is secondary. Under fixed FastChem scaling, replacing the KL element-row/element-column block with FC improves solver-result residual from `779.20` to `500.24` at layer 45 and from `1170.12` to `625.56` at layer 60, but does not close the focused density mismatch. | No; audit-only. No exact assembly-block transplant was promoted because the block replay materially helps the linear result but does not close the iter1 density residual. | Transfers as a requirement to audit the element-element complementarity block, molecular terms, removed-condensate fold-in, and element-density basis before porting reduced Newton directions to RGIE/PIPM | Medium-high for block attribution on shared rows; medium for final fix because element-row/element-column is dominant but not sufficient alone |
| Iter-1 element-element Jacobian subterm frontier | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`; KL audit `_assemble_fastchem_reduced_update` and element-subterm replay helpers in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py` | no production mode; element-element subterm decomposition only | Split `J_ee` into atom diagonal `n_i`, molecule stoichiometric outer product `sum_m nu_mi nu_mj n_m`, removed-condensate fold-in `sum_r nu_ri nu_rj n_r/lambda_r`, and zero residual `other` term; compare physical-density scalar and variable-basis diagnostics. | The subterm trace is available for the two-layer entrance smoke. The molecule outer-product term dominates the element-element residual: layer `45:-10` molecule diff Frobenius `9.0946e18` versus atom diagonal `9.1798e11` and removed fold-in `7.6350e13`; layer `60:-5` molecule diff `1.2651e18` versus atom diagonal `7.3695e10` and no removed fold-in. Focused high-residual entries `H/H`, `C/H`, `H/C`, `H/O`, and `O/H` are all explained by the molecule outer-product term in both layers. Physical-density scalar tests do not close by a simple global multiplier (`relative_difference` about `0.815` and `0.9998`). Subterm replay improves the solver result when the FC molecule term is inserted into KL (`779.20 -> 499.65`, `1170.12 -> 625.70`) but does not close focused density residuals. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_element_element_jacobian_subterm_proven` mode was added because the exact source/basis rule is not yet proven as a closing transplant. | Transfers as a requirement to audit molecule-density gauge and variable basis in element-element complementarity blocks before porting reduced Newton assembly to RGIE/PIPM | Medium-high for identifying the dominant subterm on shared rows; medium for final fix because the density/basis rule remains to be proven |
| Iter-1 molecule outer-product density provenance | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`; FastChem trace `diagnostic_trace::append_iter1_full_reduced_system`; KL audit `_assemble_fastchem_reduced_update` | no production mode; per-molecule density provenance and transformation replay only | Decompose `J_mol[j,k] = sum_m nu_mj nu_mk n_m` molecule-by-molecule; trace FastChem `Molecule::number_density`, KL `mol = exp(A_mol.T @ u - hmol)`, top molecule contributions, self-closure, and candidate density transforms. | FastChem and KL each close their own molecule-density formula to roundoff: FastChem max log closure `2.13e-14`/`5.68e-14`, KL max log closure `5.68e-14`/`5.68e-14`. Dominant entries are molecule-density driven: `H/H` is mostly `H2`, `C/H` and `H/C` are almost entirely `C1H4`, and `H/O`/`O/H` are mostly `H2O1`, `Fe1H2O2`, and `H2Mg1O2`. Direct FastChem molecule-density replay closes the molecule outer-product residual but only improves solver-result residual to the same subterm-replay level (`779.20 -> 499.65`, `1170.12 -> 625.70`) and does not close focused density. Tested KL transforms are not sufficient: raw-to-physical scalar, physical atomic old densities plus KL hvector, gas gauge bridge, FastChem old atoms plus KL hvector, and KL old atoms plus FastChem-implied hvector all fail or worsen. Classification remains mixed: `molecule_density_gauge_mismatch=25`, `molecule_lnk_hvector_mismatch=18`, `molecule_atomic_donor_old_state_mismatch=6` in each focused layer. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_molecule_outer_product_density_proven` mode was added because no single source-proven reconstruction rule closes the molecule outer-product residual and focused density outcome. | Transfers as a warning that RGIE/PIPM must trace per-molecule gas densities and hvector/gauge/state provenance before adopting element-element molecule outer-product assembly | Medium-high for per-molecule attribution and self-closure; medium-low for a fix because provenance remains mixed |
| Iter-1 top-molecule factorization and post-`J_mol` residual | `fastchem/fastchem_src/species_struct.h::Molecule::mass_action_coeff`, `Molecule::mass_action_constant`, and `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`; KL audit `_assemble_fastchem_reduced_update` | no production mode; top-molecule factorization, stage-lag audit, and post-`J_mol` residual attribution only | For top contributors, decompose `delta_ln_n_m = delta_lnK + delta_atomic_donor + delta_gauge + delta_stage + residual_unexplained`; then replay current KL, FastChem traced `J_mol`, full FastChem `J_ee`, full element-row/element-column block, full FastChem Jacobian, and RHS/Jacobian swaps under fixed FastChem scaling. | Top contributors are stable across layers: `H2` explains about `99.6%` of `H/H`, `C1H4` explains essentially all `C/H` and `H/C`, and `H2O1` dominates `H/O` and `O/H` with `Fe1H2O2` and `H2Mg1O2` secondary. The later source-proven mass-action/hvector row now resolves the formerly missing `logK.dat` record provenance for these gas molecules; gas records have no temperature segment in this path. Stage-lag diagnostics are still incomplete: molecule densities are not traced at FastChem gas solve exit/post-initial-gas/iter0 stages or analogous KL gas-only/physical-donor/iter0 stages. Direct `J_mol` replay closes the molecule term but leaves solver residual `499.65`/`625.70`; full FastChem Jacobian replay closes the projected linear solve, so the remaining residual moves to RHS, element-row/condensate-column, projection/row-universe, partition/mapping, and conditioning. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_molecule_density_rule_proven` mode was added because no single source-proven molecule-density reconstruction rule generalizes and improves focused density across both layers. | Transfers as a requirement to carry source-record/no-segment provenance and stage-labelled gas molecule densities before using molecule outer-products in RGIE/PIPM reduced assembly comparisons | Medium for attribution; low for production readiness because source-record provenance is solved but stage/state reconstruction remains unresolved |
| Iter-1 cached molecule-vector provenance | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`, `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`, and `fastchem/fastchem_src/gas_phase/molecule_struct.cpp::Molecule<double_type>::calcNumberDensity`; KL audit `_assemble_fastchem_reduced_update` | no production mode; cached molecule-vector stage trace and reconstruction replay only | FastChem refreshes cached molecules with `number_density = exp(mass_action_constant + sum_i nu_i log(elements[i].number_density))` using the full `elements` vector, then `assembleJacobian` consumes cached `Molecule::number_density`; KL recomputes `mol = exp(A_mol.T @ u - hmol)`. | Raw-sequence stage trace shows the FastChem cached molecule vector is stable from `after_iter0_calcNumberDensity_refresh` to `immediately_before_iter1_newtonStep` and `inside_iter1_assembleJacobian_J_mol_cached_vector`. `calcNumberDensity` applies no `checkN`/floor/cap internally and uses `mass_action_constant` directly. FastChem cached density closes to FastChem full element logs plus FastChem mass-action constant at roundoff. KL current density remains far away, while FastChem full elements + KL hvector and KL elements + FastChem hvector do not close. Cached classification is dominated by `full_element_vector_mismatch` (`47` rows per focused layer, `2` mixed). Direct cached molecule replay closes `J_mol` by construction but leaves solver residual `499.65`/`625.70` and focused log-density mismatch `1.5367`/`2.2604`; no top-k or reconstructed rule generalizes. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_cached_molecule_vector_proven` mode was added because the cached vector is traced but not a closing source-proven KL reconstruction rule. | Transfers as a requirement to preserve full-element vector provenance for cached gas molecule refresh before comparing molecule outer-product reduced Jacobian terms in RGIE/PIPM | Medium-high for FastChem cached-vector source semantics; medium-low for production because the closing rule is still not proven |
| Iter-1 full element-vector and mass-action/hvector provenance | `fastchem/fastchem_src/gas_phase/init_read_files.cpp::GasPhase::readSpeciesData`, `fastchem/fastchem_src/gas_phase/molecule_struct.cpp::Molecule<double_type>::calcMassActionConstant`, `fastchem/fastchem_src/gas_phase/molecule_struct.cpp::Molecule<double_type>::calcNumberDensity`, and KL parser/audit sources `src/exogibbs/presets/fastchem.py::_parse_fastchem_coeffs_with_metadata`, `logk`, `hvector_func`, and `_assemble_fastchem_reduced_update` | no production mode; source-record provenance, common-convention comparison, full element-vector factorization, and mass-action/hvector replay only | Gas `logK.dat` records have one 5-coefficient fit and no temperature segment. FastChem computes `k_FC = raw_logK(T) + (sum_nu - 1)*ln(1e-6*k_B*T)`, equivalently `k_FC = -h_source_KL - (sum_nu - 1)*ln(1e6/(k_B*T))` when KL `h_source=-raw_logK(T)` from the same source record. | Full element-vector mismatch alone does not explain top molecule errors: `H2`, `C1H4`, and `H2O1` factorization residuals remain large in both layers. FastChem and KL source records now match for `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2`; the source-derived mass-action formula has residual `<=6e-14` across both focused layers. The current KL hvector entries used in the reduced update remain gauge/source-convention mismatched, so all top rows still classify as `mass_action_hvector_gauge_mismatch` relative to the old KL entry. Source-proven mass-action reconstruction closes the common-convention comparison, but diagnostic `J_mol` reconstruction with that conversion and current KL/full-vector candidates worsens the molecule residual and focused iter1 density, while direct FastChem cached replay closes only `J_mol` by construction. | No; audit-only. No guarded `actual_true_kl_atomic_branch_exact_iter1_mass_action_hvector_proven` mode was added because the proven source conversion does not materially improve focused iter1 density across both layers and does not close the wider reduced-system residual. | Transfers as a requirement to carry gas `logK.dat` source record, no-segment semantics, hvector/mass-action gauge conversion, and full element-vector provenance separately before reusing molecule outer-products in RGIE/PIPM reduced assembly | High for source-record and formula provenance; medium-low for production because the exact formula is proven but is not a closing KL solver rule |
| Iter-1 KL hvector plumbing frontier | `src/exogibbs/presets/fastchem.py::hvector_func`, `src/exogibbs/optimize/pipm_rgie_cond.py::gas_molecule_density_gauge_bridge`, and `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` | no production mode; corrected-column KL hvector path trace and hvector-only `J_mol` candidates | Trace `h_source=-L_m(T)`, bridge `(sigma_m-1)*ln(1e6/(k_B*T))`, `h_needed=h_source+bridge`, audit copy, and `h_current=hgas[n_elem+mol_i]` keyed by `gas_setup.species[n_elem:]`; test current/source/needed/wrong-sign/cached replays under fixed FastChem scaling. | Supersedes the earlier unknown-convention result: the prior classification came from using the VMR comparison slice `gas_setup.species[29:]` as molecule-column labels. With corrected labels, `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2` all classify as `current_is_source_hvector` with `h_current-h_source=0` in both focused layers. `h_needed` remains source-proven but diagnostic-only: it does not reduce `J_mol` or improve focused iter1 density. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_hvector_plumbing_proven` mode was added because corrected attribution proves a label issue, not a safe hvector bridge. | Transfers as a requirement to keep `h_source`, bridge, `h_needed`, and molecule-column labels separate; key hvector provenance by formula-matrix column, not by VMR comparison slices | High for corrected top-molecule attribution; medium-low for production because no hvector conversion closes the reduced update |
| Iter-1 molecule-column label alignment | `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` and `gas_setup.species[n_elem:]` column provenance | no production mode; audit attribution fix only | Keep `gas_species_compare=gas_setup.species[29:]` for VMR comparison, but use `gas_molecule_species_by_col=gas_setup.species[n_elem:]` for molecule-column records, `J_mol` provenance, hvector plumbing, and factorization. | In the focused smoke `n_elem=28`; the old `[29:]` labels were shifted by one. Required top molecules reclassify from old-label `current_is_unknown_convention` to corrected-column `current_is_source_hvector` with `h_current-h_source=0` in both layers. Current/source `J_mol` residual remains `9.0946e18` / `1.2651e18`; `h_needed` does not reduce `J_mol` or improve iter1 density, while FastChem cached replay closes only `J_mol` by construction. Full all-column stoichiometry proof remains blocked for 18 parser-unparseable alias/suffix species and one `F2Si1` name/formula discrepancy, but the five required top molecules are column-proven. | No; audit-only. No guarded hvector mode was added because corrected attribution proves a label bug, not a safe density-basis hvector conversion. | Transfers as a provenance rule: molecule thermo/hvector rows must be keyed by formula-matrix column index, not by VMR comparison species slices. | High for the five top molecules; medium for all-species alignment because alias/suffix stoichiometry parsing is incomplete |
| Iter-1 post-`J_mol` residual block attribution | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian` full reduced-system trace and KL audit fixed-scaling block replays | no production mode; post-`J_mol` residual decomposition only | After FastChem cached molecule replay closes `J_mol`, compare current KL, cached `J_mol`, full `J_ee`, full element-row/element-column, RHS-only, element-row/condensate-column-only, partition/mapping availability, and full FastChem Jacobian upper bound under fixed FastChem scaling. | Corrected-label smoke keeps `J_mol` closed by construction in the cached replay but leaves solver residuals `499.65`/`625.70` and focused log-density mismatch `1.5367`/`2.2604`. Full `J_ee` and full element-row/element-column replays remain essentially at the cached-`J_mol` residual (`500.24`/`625.56`), RHS-only does not improve (`779.20`/`1170.12`), while full FastChem Jacobian closes the projected linear solve (`~1e-11`). With `J_mol` removed, the next attributable Jacobian frontier is element-row/condensate-column. Partition/mapping-only post-`J_mol` replay is unavailable in the current labelled trace. | No; audit-only. No post-`J_mol` guarded mode was added because no exact source rule for the next block was proven to improve focused density across both layers. | Transfers as a reduced-system block ledger: close molecule outer-products first, then audit element-row/condensate-column coupling before treating RHS, projection, partition, or scaling as primary | Medium-high for focused block attribution; medium for production because it is an attribution by labelled replay/elimination, not a source-proven block rule |
| Iter-1 element-row / condensate-column `J_ec` provenance | `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`; KL audit `_assemble_fastchem_reduced_update` and `element_condensate_jacobian_audit` | no production mode; source-level `J_ec` decomposition and fixed-scaling replay only | Compare matched projected entries with `J_ec_FC[j,c]=stoich_cj*n_old_FC` and `J_ec_KL[j,c]=A_jac[j,c]*n_old_KL`; decompose into density, stoichiometry, mapping/sign, cap/maxDensity, and unexplained terms. | Entry-level residuals are mostly old-condensate-density driven: layer `45:-10` has `2161` `old_condensate_density_mismatch` entries and `83` `cap_or_maxDensity_state_mismatch` entries; layer `60:-5` has `2552` and `116`. Stoichiometry and sign/basis do not dominate the matched entries: top residuals close as `stoich_shared*(n_old_FC-n_old_KL)` to roundoff-level unexplained terms. Old-density provenance in this row was trace-limited at the time; the later old-condensate handoff row resolves the retained-column missing fields and classifies the handoff as seed/maxDensity/cap dominated. `J_ec` replay alone closes the block but worsens layer 45 focused log-density (`1.5367 -> 2.1344`) and improves layer 60 only (`2.5021 -> 1.7909`); `J_mol+J_ec` reduces solver-result mismatch to `4.35`/`0.77` but leaves focused density at the cached-`J_mol` values (`1.5367`/`2.2604`). | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_element_condensate_jacobian_proven` mode was added because no single source-proven `J_ec` rule improves focused iter1 density across both layers. | Transfers as a warning that element-row/condensate-column coupling must carry retained old-condensate density provenance, cap/maxDensity handoff, and retained-column mapping before being used as an RGIE/PIPM production rule. | Medium-high for entry formula attribution; medium-low for production because old-density provenance is trace-limited and replay does not improve both focused layers |
| Iter-1 old-condensate state handoff | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`, `CondensedPhase::correctValues`, and `fastchem/fastchem_src/diagnostic_trace.h::append_iter1_full_reduced_system`; KL audit `_second_post_seed_update_actualization_solve` | no production mode; coherent old-state ladder and replay only | Trace retained/Jac shared condensates through seed, iter0 `correctValues`, iter1 `assembleJacobian`, and iter1 `correctValues`; replay FC old `n_c`, lambda, tau, and `J_ec` coherently in audit closure. | The widened FastChem trace exposes all requested retained-column old-state fields in both focused layers. Ladder classification is cap/maxDensity/seed handoff dominated: `45:-10` has `102` shared retained columns classified as `inherited_from_seed_or_maxDensity_mismatch`; `60:-5` has `116`. `J_ec` entry residuals remain density-driven (`2161`/`2552` old-density entries) and old-density provenance collapses to `cap_or_maxDensity_mismatch` for `102`/`116` retained columns. Coherent FC old-state replay is not promotable: FC old `n_c` + `J_ec` + `correctValues` worsens layer 45 focused log-n (`1.7620 -> 2.2474`) while improving layer 60 (`1.9461 -> 1.2958`); adding FC lambda/tau does not change those values, and adding FC `J_mol` leaves layer 45 worse (`1.9536`) while layer 60 improves only to `1.7079`. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_old_condensate_state_handoff_proven` mode was added because no minimal coherent old-state rule improves focused iter1 density across both layers. | Transfers as a cap/maxDensity handoff frontier: RGIE/PIPM needs coherent seed/maxDensity, iter0 cap, old `n_c`, lambda, and tau provenance before porting the retained-condensate old-state handoff. | High for trace completeness and cap/maxDensity classification on shared retained columns; medium-low for production because replay is mixed across the two layers |
| Iter-0 cap/maxDensity provenance | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::correctValues` raw `post_correctValues_update` and `fastchem/fastchem_src/diagnostic_trace.h`; KL audit iter0 cap truth table and fixed-scaling cap replay | no production mode; iter0 cap/maxDensity truth table and replay only | Decompose `n_uncapped = n_seed*exp(delta_n_cond_clipped)` and `n_new=min(n_uncapped,maxDensity)` for each shared retained/Jac column; classify both-capped, one-side-capped, neither-capped, retained/carryover, and missing-trace rows; replay seed, delta, and maxDensity/cap variants without changing production behavior. | The Eq.13 total-inventory maxDensity source is present in the focused iter0 cap state for all shared retained rows (`102/102` at `45:-10`, `116/116` at `60:-5`). The remaining truth table is mixed: layer 45 classifies `70` rows as `retained_carryover_state_mismatch`, `27` as `both_capped_maxDensity_value_mismatch`, and `5` as `kl_only_capped`; layer 60 classifies `72`, `42`, and `2`. Source/stage decomposition splits between limiting-element and maxDensity-value differences (`80/22` at layer 45, `86/30` at layer 60). Replay does not generalize: FC delta/seed/maxDensity variants worsen layer 45 focused log-n, while FC delta/maxDensity/post-iter0 variants improve only layer 60; the upper-bound FC post-iter0 old-state replay remains the prior layer-mixed result (`2.2474` at 45, `1.2958` at 60). | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter0_cap_maxdensity_handoff_proven` mode was added because no single seed, delta, cap, maxDensity, or retained/carryover rule improves focused iter1 density across both layers. | Transfers as a mixed cap/maxDensity and retained/carryover provenance warning: RGIE/PIPM must prove the consumed iter0 old state, cap branch, and stage-copy path together before porting a cap handoff. | High for branch-count provenance and Eq.13 propagation; medium-low for production because replay is layer-mixed and retained/carryover rows dominate |
| Iter-0 to iter-1 coherent full active-state transition | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` result scaling, `correctValues`, and refresh sequence; `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleRightHandSide` and `newtonStep`; KL audit coherent transition summary | no production mode; coherent full active-state transition audit only | Track all active condensates through iter0 pre reduced Newton, iter0 post `correctValues`, nearest post-refresh iter1 pre state, iter1 pre partition, and iter1 pre reduced Newton. Compare all-active burden `B_cond[j]=sum_active nu_cj*n_c`, removed fold-in `F_rem[j,k]=sum_removed nu_rj*nu_rk*n_r/lambda_r`, and global result scaling `result_scaled=result*min(1,limit/max(abs(result)))`. | The coherent transition smoke remains mixed/inconclusive. Eq.13 propagation is not the blocker. All-active burden residuals are large and mostly KL-only `CH4(s,l)`: `B_cond` norms `5.22e15`/`7.37e14`, KL-only norms `5.40e15`/`7.51e14`. Removed fold-in is layer-specific (`7.63e13` at `45:-10`, zero at `60:-5`). Global scaling raw results differ. Coherent full-state proxies worsen layer 45 focused log-n (`1.7620 -> 2.2474`) but improve layer 60 (`1.9461 -> 1.2958`), so no minimal component generalizes. A distinct FastChem post-`correctValues` refreshed all-active snapshot is still missing; the audit uses iter1 pre reduced state as the nearest labelled snapshot. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter0_to_iter1_coherent_transition_proven` mode was added because no coherent transition component improves focused iter1 density across both focused layers. | Transfers as a full-transition provenance requirement: RGIE/PIPM must align full active row universe, all-active RHS burden, removed fold-in, global result scaling, refreshed activity/molecule state, and partition before porting iter0-to-iter1 handoff rules. | Medium-high for audit attribution; medium-low for production because replay is layer-mixed and one refreshed-stage trace remains missing |
| Iter-1 KL-only `CH4(s,l)` burden lifecycle | `fastchem/fastchem_src/condensed_phase/condensed_phase.cpp::CondensedPhase::selectActiveCondensates`, `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleRightHandSide`, and FastChem reduced-iteration records from `calculate.cpp::CondensedPhase::calculate`; KL audit CH4 lifecycle and burden replay | no production mode; CH4 lifecycle, all-active burden isolation, and targeted RHS replay only | Trace `CH4(s,l)` from all rows through candidate threshold, active lifecycle, seed, iter0 update, iter1 partition, all-active burden, RHS, and `correctValues`; compare `B_cond` with current KL, without CH4, without all KL-only rows, FastChem active rows, and shared rows. | `CH4(s,l)` is a FastChem candidate but not selected active in both focused layers, while KL keeps it as retained/Jac, classifying as `candidate_but_not_selected_active` versus `missing_fastchem_eviction_rule`. Dropping only CH4 removes most all-active burden norm (`87.0%` at `45:-10`, `93.5%` at `60:-5`) and almost all H/C burden, but not O burden. FastChem iter0 raw scaling is now exposed: raw max `217.40`/`303.91`, factors `2.30e-2`/`1.65e-2`; KL iter0 raw max `119.64`/`198.65`, factors `4.18e-2`/`2.52e-2`. CH4/all-KL-only burden RHS replays do not change focused log-n (`1.7620`/`1.9461`), and the upper-bound `J_mol+J_ec` with CH4 removal improves only layer 60. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter1_ch4_or_burden_lifecycle_proven` mode was added because CH4 explains burden magnitude but not focused iter1 density across both layers. | Transfers as a lifecycle warning: RGIE/PIPM must distinguish candidate-but-not-selected condensates from stale active carryover, and must not port CH4 eviction or KL-only row eviction until the RHS burden change also moves the reduced update in both layers. | High for CH4 burden attribution and FastChem iter0 scaling trace; medium-low for production because targeted replay is neutral for focused density |
| Exact post-`correctValues` refreshed all-active snapshot | `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` after `correctValues`, element-density refresh, active-condensate `calcActivity`, and molecule `calcNumberDensity`, before objective evaluation and old-state assignment; KL closest proxy is the post-iter0 update state in the phase-2 audit | no production mode; exact refreshed-state trace and proxy comparison only | Emit all active condensate rows, refreshed element densities, refreshed molecule densities, and all-active burden at `post_correctValues_refreshed_all_active_state`; compare to the previous iter1 pre-reduced proxy and KL closest refreshed proxy; replay exact refreshed-state candidates without altering production behavior. | The exact FastChem snapshot is now available for both focused layers (`103` rows at `45:-10`, `116` at `60:-5`). It has the same active row universe as the previous FastChem proxy and still excludes `CH4(s,l)`, but its state is not the same as the proxy: exact-vs-proxy mean absolute log-`n` residual is `29.679`/`29.867`, mean absolute log-lambda residual is `10.575`/`10.650`, and mean absolute log-activity residual is `123.141`/`176.517`. Exact all-active burden residuals versus KL are `6.112e15`/`9.019e14`, larger than the previous proxy residuals `5.222e15`/`7.369e14`; CH4 remains absent in FastChem and present in KL, but CH4 alone no longer explains the exact burden residual (`62.1%`/`56.9%` removed). Exact refreshed-state replays remain layer-mixed: full condensate state worsens layer 45 focused log-n (`1.7620 -> 2.2474`) and improves only layer 60 (`1.9461 -> 1.2958`); full state plus `J_mol+J_ec` gives `1.9536`/`1.7079`; full FastChem Jacobian closes the projected solve but not focused density. | No; audit-only. No `actual_true_kl_atomic_branch_exact_iter0_post_refresh_state_proven` mode was added because no exact refreshed-state component improves focused iter1 density across both layers. | Transfers as a refreshed-state boundary requirement: RGIE/PIPM must trace the post-update refreshed active state separately from the next reduced-Newton proxy before treating lifecycle, burden, or old-state handoff as production semantics. | High for trace location and exact snapshot availability; medium-low for production because exact replay remains mixed |

## Compact Delta Provenance Addendum

The compact one-step residual is now recorded as delta dominated, with cap branch formulas closed to roundoff in the prior compact extract:

- Layer `45:-10`: `62` rows included, `56` closed by branch formula, mean absolute observed residual `0.5661896008239972`, mean absolute prediction error `1.2011720982942263e-15`, dominant counts `delta=23`, `old_state=17`, `old_delta_compensation=14`, `mapping_or_missing=6`, `cap_branch=2`.
- Layer `60:-5`: `54` rows included, `53` closed by branch formula, mean absolute observed residual `0.7292492655416573`, mean absolute prediction error `1.4385439904274839e-15`, dominant counts `delta=33`, `old_state=11`, `old_delta_compensation=7`, `mapping_or_missing=1`, `cap_branch=2`.

The new compact delta-provenance artifact `results/actual_fastchem_gas_phase_transplant_phase2_delta_provenance_compact.json` selects focused rows, top residual rows, delta-dominated rows, and old/delta compensation rows. It reports zero delta-path self-closure error on both FastChem and KL rows available in the compact source. The available compact fields classify `89` selected rows as `delta_raw_result_dominated` and `3` as `delta_mapping_or_index_dominated`, with mean absolute delta residual `0.6146803422146688`.

Standalone row fields `global_scaling_factor_FC`, `global_scaling_factor_KL`, `max_abs_raw_result_FC`, and `max_abs_raw_result_KL` are still absent from the compact source rows. Effective row scaling was inferred only from `delta_scaled/raw` where possible. The source trace target remains `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` at `result_scaled = result`, `max_value = result_scaled.cwiseAbs().maxCoeff()`, optional `result_scaled *= limit/max_value`, and `correctValues(...)`.

KL production implication: no solver or guarded-mode change is justified. The next audit target is raw solver-result provenance, with source-level global scaling fields populated before promoting any interpretation. Old state is secondary but compensation exists and can make a delta-only replay look worse on individual rows.

RGIE/PIPM transfer implication: treat one-step delta provenance as a checklist, not a rule. Separate raw direction, global result scaling, local clipping, solver-result-to-row mapping, removed-row analytic deltas, and projected/focused coverage before comparing reduced directions.

## Compact Raw Solver Result Provenance Addendum

The raw-result compact audit `results/actual_fastchem_gas_phase_transplant_phase2_raw_result_provenance_compact.json` reuses only the labelled entrance-smoke reduced systems for `45:-10` and `60:-5`. It does not change production solvers, presets, row materialization, lifecycle, `correctValues`, schedules, or CH4 lifecycle.

Field availability:

- FastChem and KL labelled Jacobian matrices, RHS vectors, row scaling vectors, solver result vectors, and solver-result-to-row mappings are available for both focused layers in `results/actual_fastchem_gas_phase_transplant_phase2_one_step_closure_entrance_smoke.json`.
- Full unprojected component solves are not emitted; the compact audit solves labelled projected shared systems only.
- Separated per-term RHS vectors are still missing: the labelled trace exposes full RHS before/after scaling but not standalone RHS terms for condensate activity residuals, element conservation burden, molecule burden, removed-condensate correction, old activity correction, or tau/log-tau.

Mapping and row-universe result:

- Layer `45:-10`: `47` selected rows, `45` label matches, `2` missing/mismatched labels, `47` vector-index mismatches, `124` matched projected rows/columns, `13` KL-only projected rows/columns, and `2` focused rows outside the projected solve.
- Layer `60:-5`: `45` selected rows, `44` label matches, `1` missing/mismatched label, `44` vector-index mismatches and `1` vector-index match, `139` matched projected rows/columns, `14` KL-only projected rows/columns, and `1` focused row outside the projected solve.

Counterfactual solve result:

- Re-solving the same FC projected `J` and FC projected RHS through the audit backend reproduces the FastChem raw result to roundoff: mean absolute residuals `5.50e-12` at `45:-10` and `7.12e-12` at `60:-5`; solver backend is non-primary.
- FC `J` + KL RHS strongly reduces raw-result residual relative to KL `J` + KL RHS, while KL `J` + FC RHS does not. FC `J` + FC RHS closes the projected raw vector. However, row-level classification remains mixed: `raw_result_Jacobian_dominated=45`, `raw_result_mixed_or_unresolved=41`, `raw_result_mapping_or_index_dominated=3`, and `raw_result_requires_RHS_and_Jacobian=3`.
- The compact block isolation does not provide a promotable block rule. `J_ec`-only replacement worsens the mean raw-result residual on selected rows (`5505.13` at `45:-10`, `6090.82` at `60:-5`), and `J_mol` subterm replacement is unavailable because the emitted element-element subterm arrays are not isomorphic to the shared projected element block.

KL production implication: no production solver or guarded-mode change is justified. The next blocker is still source-level raw-result provenance, with a more exact RHS/Jacobian/block decomposition and explicit row-universe projection rule before any solver behavior can be considered.

RGIE/PIPM transfer implication: labelled projected systems and backend guardrails are required before using reduced Newton directions. The current result proves the raw-vector mismatch is real and not a backend artifact, but does not isolate a single production-transferable RHS, Jacobian, row-universe, mapping, or block-level rule.

Decision: raw solver result provenance remains mixed or inconclusive.

## Compact Jacobian Subterm Sensitivity Addendum

The Jacobian-block/subterm compact audit `results/actual_fastchem_gas_phase_transplant_phase2_jacobian_subterm_sensitivity_compact.json` projects the existing labelled entrance-smoke matrices for `45:-10` and `60:-5` onto the same shared row/column ordering used by the raw-result compact audit. It is diagnostic-only and does not change production solvers, guarded modes, `J_mol`, `J_ec`, row universe, row scaling, lifecycle, schedules, CH4 lifecycle, or `correctValues`.

Projected Jacobian availability:

- Full projected blocks `J_cc`, `J_ce`, `J_ec`, and `J_ee` are available from the labelled shared projected `J`.
- `J_ee` subterms are now projected by element labels/global element indices into the shared projected shape: `J_atom`, `J_molecule`, `J_removed`, and `J_other`.
- Additivity closes to roundoff. Layer `45:-10` reconstruction max errors are `2.22e-16` for FastChem and `4.81e-35` for KL. Layer `60:-5` max errors are `2.22e-16` for FastChem and `0` for KL.
- RHS term vectors remain missing. The trace exposes only `rhs_vector_before_scaling` and `rhs_vector_after_scaling`, not separated condensate-row or element-row RHS terms.

Block sensitivity result:

- Full FC `J` + KL RHS remains the aggregate Jacobian-side improvement: raw residuals are `22.539650451705914` at `45:-10` and `258.23728172185525` at `60:-5`.
- FC `J_cc` only and FC `J_ce` only do not move the residual from current KL in this compact projection.
- FC `J_ec` only worsens raw residuals to `5505.130855573409` and `6090.823096193314`.
- FC `J_ee` only improves only slightly: `1343.587761911473` versus current `1395.0214581827588` at `45:-10`, and `2296.453339585019` versus current `2360.305920367364` at `60:-5`.
- FC molecule outer-product only is nearly the same as FC `J_ee` only (`1343.8112139591194` and `2296.3289997189245`) but does not close the raw direction or focused one-step metric.
- Row-level attribution remains unresolved: `raw_result_not_explained_by_J_blocks=80`, `raw_result_dominated_by_J_ec=7`, `raw_result_mapping_or_index_issue=3`, and `raw_result_requires_multiple_J_blocks=2`.

KL production implication: no production solver, guarded mode, block transplant, RHS change, row-universe change, or lifecycle change is justified. The raw-result mismatch is aggregate Jacobian-side, but exact subblock attribution is not isolated by single-block or `J_ee` subterm swaps.

RGIE/PIPM transfer implication: `J_mol` can now be projected and tested in the shared system, but neither `J_mol` nor `J_ee` is a transferable rule. Any transfer must keep full labelled Jacobian projection, RHS-term trace completion, and row-universe guardrails attached to the reduced direction comparison.

Decision: raw result mismatch is Jacobian-side but exact subblock remains unresolved.

## Compact Coherent Element-Row / Schur Addendum

The coherent element-row compact audit `results/actual_fastchem_gas_phase_transplant_phase2_element_row_schur_compact.json` reuses the same labelled entrance-smoke shared projection for `45:-10` and `60:-5`. It is diagnostic-only and does not change production solvers, guarded modes, `J_mol`, `J_ec`, row universe, row scaling, lifecycle, schedules, CH4 lifecycle, or `correctValues`.

Coherent block replay:

- The coherent FC element-row block `[J_ec,J_ee]` inserted into KL `J` exactly equals the full FC `J` replay on the shared projected system, because `J_cc` and `J_ce` are identical across FC and KL in this projection.
- Layer `45:-10`: current KL raw residual `1395.0214581827588`; coherent FC `[J_ec,J_ee]` + KL RHS `22.539650451705914`; FC full `J` + KL RHS `22.539650451705914`; FC full `J` + FC RHS `0`.
- Layer `60:-5`: current KL raw residual `2360.305920367364`; coherent FC `[J_ec,J_ee]` + KL RHS `258.23728172185525`; FC full `J` + KL RHS `258.23728172185525`; FC full `J` + FC RHS `0`.
- Single-block failure is now coherent: FC `J_ec` alone worsens, FC `J_ee` alone is weak, but FC `[J_ec,J_ee]` together gives the full aggregate Jacobian-side improvement.

Schur-complement result:

- In both layers, replacing both `C=J_ec` and `D=J_ee` recovers the FC-like Schur complement exactly in Frobenius norm, while replacing `D` only leaves a small but nonzero Schur residual and replacing `C` only remains ill-conditioned.
- Layer `45:-10`: `D_KL` condition number `1.32e5`, `S_KL` condition number `1.32e22`; `D_FC` condition number `52.4`, `S_FC` condition number `7.98e4`; mixed `C+D` has `S_frobenius_norm_vs_FC=0` but effective RHS still differs by `97.465`.
- Layer `60:-5`: `D_KL` condition number `1.29e10`, `S_KL` condition number `4.77e23`; `D_FC` condition number `54.6`, `S_FC` condition number `1.23e5`; mixed `C+D` has `S_frobenius_norm_vs_FC=0` but effective RHS still differs by `136.883`.

RHS status:

- Separated labelled RHS term vectors remain missing. The trace exposes `rhs_vector_before_scaling` and `rhs_vector_after_scaling`, but not condensate-row terms (`log_activity`, activity correction, log-tau/log-n/log-lambda) or element-row terms (total inventory, atom, molecule burden, all-active condensate burden, removed-condensate correction).
- Therefore the residual after coherent Jacobian alignment is not term-attributable yet: `22.539650451705914` at `45:-10` and `258.23728172185525` at `60:-5`.

KL production implication: no production solver, guarded mode, coherent block transplant, RHS change, row-universe change, or lifecycle change is justified. The coherent element-row block is the aggregate Jacobian-side object, but it does not close without the remaining RHS/vector provenance.

RGIE/PIPM transfer implication: use coherent element-row and Schur diagnostics as labelled provenance checks. Do not transfer `J_ec`, `J_ee`, `J_mol`, or Schur-complement substitutions as standalone rules until RHS term vectors and row-universe coupling are source-proven.

Decision: coherent element-row block helps but residual remains mixed.

## Compact RHS Term Decomposition Addendum

The RHS-term compact audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_term_decomposition_compact.json` follows the coherent element-row result and keeps the same entrance-smoke scope (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, `J_mol`, `J_ec`, row universe, row scaling, lifecycle, schedules, CH4 lifecycle, or `correctValues`.

Instrumentation status:

- FastChem diagnostic tracing now reconstructs RHS term vectors in `fastchem/fastchem_src/diagnostic_trace.h::append_iter1_full_reduced_system` using the `CondPhaseSolver::assembleRightHandSide` formula and emits `condensate_rhs_terms` and `element_rhs_terms` in the same order as `rhs_vector_before_scaling`.
- KL audit assembly now builds matching common RHS term vectors in `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` and attaches them to `full_reduced_system_trace`.
- The current source artifact `results/actual_fastchem_gas_phase_transplant_phase2_one_step_closure_entrance_smoke.json` predates those fields. Therefore the compact RHS audit reports the exact missing fields instead of guessing: `FastChem full_reduced_system_trace.condensate_rhs_terms.terms`, `FastChem full_reduced_system_trace.element_rhs_terms.terms`, `KL full_reduced_system_trace.condensate_rhs_terms.terms`, and `KL full_reduced_system_trace.element_rhs_terms.terms`.

Current compact result:

- RHS additivity cannot be checked in the existing artifact because term vectors are absent.
- Termwise RHS sensitivity and Schur effective RHS decomposition are unavailable for the same reason.
- Full-vector sensitivity still confirms the remaining residual is RHS-side after coherent Jacobian alignment: FC `J` + KL full RHS leaves `22.539650451705914` at `45:-10` and `258.23728172185525` at `60:-5`; FC `J` + FC full RHS closes both to `0`.
- Row-level RHS attribution is therefore `rhs_mixed_or_unresolved=92`.

Guardrail: coherent `[J_ec,J_ee]` remains the proven aggregate Jacobian-side object, but no broad block transplant is promoted. The next valid audit step is to regenerate the entrance smoke with RHS term traces and then run the same compact RHS audit for additivity and term sensitivity.

Decision: RHS term decomposition remains mixed or inconclusive.

## Fresh RHS-Term Smoke Addendum

The fresh RHS-term entrance smoke was regenerated from the diagnostic-only trace fields and written to `results/actual_fastchem_gas_phase_transplant_phase2_rhs_terms_fresh_entrance_smoke.json`, with traces in `results/actual_fastchem_gas_phase_transplant_phase2_rhs_terms_fresh_entrance_smoke_traces.json`. The compact rerun is `results/actual_fastchem_gas_phase_transplant_phase2_rhs_term_decomposition_fresh_compact.json`.

Fresh field availability:

- FastChem and KL now both expose `full_reduced_system_trace.condensate_rhs_terms.terms` and `full_reduced_system_trace.element_rhs_terms.terms` for both focused layers.
- Each side has 4 condensate RHS terms (`log_activity`, `activity_correction`, `log_tau_log_n_log_lambda`, `other`) and 6 element RHS terms (`total_inventory`, `atom`, `molecule_burden`, `all_active_condensate_burden`, `removed_condensate_correction`, `other`).
- Term row-label counts match `rhs_vector_before_scaling` in the fresh artifact.

Fresh additivity result:

- Scaled RHS term reconstruction closes to roundoff on both sides: max scaled errors are `1.42e-14` for FastChem and KL in both focused layers.
- Unscaled RHS term reconstruction does not close. Layer `45:-10` unscaled max errors are `295.0` for FastChem and `1.0` for KL. Layer `60:-5` unscaled max errors are `82.15625` for FastChem and `0.03125` for KL.
- Because the audit requires both unscaled and scaled additivity before term sensitivity, termwise RHS swaps and Schur effective RHS term decomposition were not run.

Full-vector replay still confirms the residual after coherent Jacobian alignment is RHS-side: FC `J` + KL full RHS leaves raw residuals `22.53965045170672` and `258.2372817218515`, while FC `J` + FC full RHS closes both to `0`. Row-level RHS attribution remains `rhs_mixed_or_unresolved=92`.

KL production implication: no production solver, guarded mode, coherent block transplant, RHS term, row-universe, lifecycle, schedule, `J_mol`, or `J_ec` change is justified. The next blocker is diagnostic RHS term additivity, not solver behavior.

RGIE/PIPM transfer implication: RHS term containers are now available, but no inventory, atom, molecule burden, all-active burden, removed-correction, or condensate-row RHS rule is transferable until the term decomposition adds back to both unscaled and scaled RHS vectors and term-swap sensitivity is run.

Decision: RHS term decomposition remains mixed or inconclusive.

## Scaled RHS-Term Decomposition Addendum

The scaled RHS-term compact audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_scaled_term_decomposition_compact.json` reuses the fresh RHS-term entrance smoke and the coherent/full FastChem Jacobian replay. It is diagnostic-only and does not change production solvers, guarded modes, row universe, row scaling, lifecycle, schedules, CH4 lifecycle, `J_mol`, `J_ec`, `correctValues`, or presets.

Solve-space convention:

- `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` calls `assembleJacobian`, which returns `scaling_factors`.
- `assembleRightHandSide` receives those `scaling_factors` and divides the RHS rows by them.
- `solveSystem(jacobian, rhs, result)` consumes the scaled RHS, represented in the trace by `rhs_vector_after_scaling`.
- `rhs_vector_before_scaling` remains a diagnostic source-space/bookkeeping vector; its term mismatch is not a scaled replay blocker.

Scaled additivity result:

- FastChem and KL scaled RHS term vectors sum to the solve-space RHS at roundoff in both layers. Max scaled reconstruction error is `1.4210854715202004e-14`.
- Unscaled bookkeeping remains unresolved. Reconstructing source-space RHS from scaled full RHS by multiplying by row scaling nearly closes the full vector on KL and to `0.125`/`0.03125` on FastChem, but reconstructing from scaled term sums still leaves residuals. The compact classifies the unscaled failure as `unscaled_failure_unresolved` and keeps it diagnostic-only.

Scaled sensitivity result:

- Full-vector baseline is unchanged: FC `J` + KL scaled full RHS leaves raw residuals `22.53965045170672` and `258.2372817218515`; FC `J` + FC scaled full RHS closes to `0`.
- No single RHS term closes both layers. At `45:-10`, the molecule-burden term gives the only meaningful single-term improvement (`22.53965045170672` to `20.416053154422197`), while log-activity, total inventory, and all-active burden worsen. At `60:-5`, log-activity improves strongly (`258.2372817218515` to `101.08227213770724`) and molecule burden improves moderately (`207.817857298179`), while all-active burden and total inventory worsen.
- Cumulative best scaled terms eventually close both layers (`8.21e-12` at `45:-10`, `1.30e-11` at `60:-5`), but the required ordered term set is multi-term and layer-mixed.
- Row-level attribution is mixed/multi-term: `rhs_requires_multiple_terms=51`, `rhs_element_molecule_burden_dominated=23`, `rhs_element_removed_correction_dominated=8`, `rhs_mixed_or_unresolved=5`, `rhs_element_atom_dominated=3`, `rhs_element_total_inventory_dominated=1`, and `rhs_condensate_tau_n_lambda_dominated=1`.

Schur effective scaled RHS result:

- The largest effective condensate RHS residual comes from `element_rhs_terms.all_active_condensate_burden` in both layers, with amplification through `solve(D, rhs_e)` of about `10.73` at `45:-10` and `11.06` at `60:-5`.
- This Schur norm signal is not by itself a closing term replay: all-active burden single-term replacement worsens the raw residual in both layers.

KL production implication: no production solver, guarded mode, coherent block transplant, row-universe change, row-scaling change, lifecycle change, schedule change, `J_mol`, `J_ec`, `correctValues`, or RHS term rule is promotable. The scaled solve-space result requires multiple RHS terms and remains diagnostic-only.

RGIE/PIPM transfer implication: use scaled solve-space RHS terms for reduced-direction replay, but do not transfer a single RHS burden, inventory, activity, molecule, atom, removed-correction, or Schur-effective rule from this smoke. The source-space/unscaled bookkeeping mismatch should be repaired as diagnostic provenance, not treated as solver behavior.

Decision: remaining residual after coherent Jacobian alignment requires multiple RHS terms.

## Compact RHS Term Interaction / Minimal-Subset Addendum

The RHS-term interaction compact audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_term_interaction_compact.json` follows the scaled solve-space decomposition for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Contribution-vector additivity:

- Each scaled term contribution was solved as `c_t = solve(J_FC, RHS_t_FC - RHS_t_KL)`, with `c_full = solve(J_FC, RHS_FC - RHS_KL)`.
- Layer `45:-10`: max raw-result-space additivity error `6.05e-12`, L2 error `1.60e-11`, selected-row mean abs error `1.24e-12`, focused-row error `1.92e-12`.
- Layer `60:-5`: max raw-result-space additivity error `1.93e-11`, L2 error `5.21e-11`, selected-row mean abs error `4.09e-12`, focused-row error `5.52e-12`.

Term geometry and cancellation:

- `log_activity` is strongly aligned with the full RHS residual (`cos=0.861` at layer 45, `0.999` at layer 60).
- `molecule_burden` is also aligned (`cos=0.795`, `0.985`).
- `all_active_condensate_burden` is strongly anti-aligned (`cos=-0.785`, `-0.985`) even though it has the largest Schur effective RHS norm. This explains why it is a Schur/cancellation signal but worsens direct single-term replay.
- The strongest cancelling pairs are `molecule_burden + total_inventory` and `log_activity + all_active_condensate_burden`. The latter pair is the minimal common subset that improves both layers, reducing raw residuals to `3.0603` and `2.5286`, but it does not close either layer.

Minimal subset and coherent groups:

- The minimal common closing subset is effectively the full nonzero scaled RHS state: `log_activity`, `log_tau_log_n_log_lambda`, `all_active_condensate_burden`, `atom`, `molecule_burden`, `removed_condensate_correction`, and `total_inventory`. It closes to `8.55e-12` at layer 45 and `1.31e-11` at layer 60.
- The layer-specific minimal closing subset omits `removed_condensate_correction` at layer 60, so the exact numerical minimum is layer-dependent.
- The tested coherent groups do not provide a smaller production rule. `condensate_row_all` improves only layer 60; `element_row_all`, `burden_family`, and `removed_and_burden_family` worsen both layers. The previous cumulative best/full nonzero subset closes both layers only because it carries the coupled RHS state.
- Row-level interaction attribution is dominated by coupled molecule/log-activity structure: layer 45 classifies `molecule_plus_log_activity=45`; layer 60 classifies `molecule_plus_log_activity=43` and `burden_plus_log_activity=1`.

Source-state interpretation: the closing subset spans condensate-row log activity/tau state plus element-row total inventory, atom, molecule burden, all-active condensate burden, and removed-condensate correction. This is not a single RHS term rule, not a condensate-only rule, and not an element-burden-only rule. It is the full coherent RHS state for the nonzero terms in this solve-space replay.

KL production implication: no production solver, guarded mode, RHS term rule, row-universe rule, row-scaling rule, lifecycle rule, `J_mol`, `J_ec`, `correctValues`, or preset change is justified. Single-term RHS rules are rejected, and no RHS term rule is promotable; full coherent RHS state remains diagnostic-only.

RGIE/PIPM transfer implication: transfer the audit pattern, not a term rule. RGIE/PIPM comparisons should carry scaled solve-space RHS provenance, term geometry, cancellation, and source-state grouping together; a single activity, burden, inventory, atom, or removed-correction rule must not be ported.

Decision: RHS residual requires full coherent RHS state.

## Compact RHS Source-State Provenance Addendum

The RHS source-state provenance compact audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_source_state_provenance_compact.json` maps the full coherent RHS-state result back to source variables at iter1 RHS assembly entry for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Source-state table status:

- Condensate-row tables are emitted for both FastChem and KL. They include species, global condensate index, active local index, retained/Jac labels, log activity, activity correction/lambda, log lambda, `n_c`, log `n_c`, tau/log-tau, max density, row scaling, and RHS labels where the trace exposes or allows exact per-row inference.
- KL lacks a standalone `old_log_activity_by_active_condensate` vector and `log_tau_values`; the audit infers per-row log activity/log-tau only from emitted RHS source terms and records those fields as inferred/missing.
- FastChem layer `45:-10` exposes `log_tau_values` in a non-isomorphic compact indexing for the full table; the audit infers per-row log-tau from `condensate_rhs_terms.log_tau_log_n_log_lambda` and records that inference.
- Element-row tables include total inventory product, atom density, top molecule burden contributors from molecule provenance, total molecule burden, active-condensate burden aggregate, removed-correction aggregate, row scaling, and RHS labels.
- Expanded per-condensate active-burden and removed-correction source rows are still missing on both sides, as are separated `total_element_density` and `epsilon`; the audit records the exact missing fields.

Formula self-closure:

- Reconstructing scaled RHS terms from emitted/inferred source-state variables closes to roundoff on both sides and layers.
- Layer `45:-10`: FastChem max reconstructed scaled-RHS error `1.42e-14`; KL max `1.42e-14`.
- Layer `60:-5`: FastChem max `2.84e-14`; KL max `1.42e-14`.
- Cross-state formula tests classify the formulas as equivalent given the same state for the terms with complete source variables; aggregate-only terms are blocked only for per-source expansion, not by formula mismatch.

Source-state residual decomposition:

- The largest scaled source residual is condensate `log_activity`: norm `15.8673` at layer 45 and `58.2247` at layer 60.
- The next largest is `all_active_condensate_burden`: norm `9.1743` and `11.1502`.
- Total inventory, molecule burden, atom, removed correction, and tau/log-n/log-lambda are smaller in direct scaled RHS norm, but interaction replay still requires the coupled state.

Source-state group replay:

- `molecule_state` is the minimal source group that improves both layers (`20.4161` and `207.8179`) but does not close.
- `condensate_activity` improves only layer 60 and worsens layer 45.
- `condensate_complementarity`, `inventory_atom`, `active_condensate_burden_state`, and `removed_correction_state` do not provide a two-layer closing rule.
- The best two-group cancellation remains `condensate_activity + active_condensate_burden_state`, reducing residuals to `3.0603` and `2.5286`, but it still does not close.
- The minimal common closing group is the full coherent RHS source state: condensate activity, condensate complementarity, inventory/atom, molecule state, active-condensate burden state, and removed-correction state. It closes to `8.56e-12` and `1.31e-11`.

KL production implication: no source-state group, RHS term, standard physical group, guarded mode, or production solver rule is promotable. Formula mismatch is not the dominant classification; the residual requires the full coherent RHS source state and remains diagnostic-only.

RGIE/PIPM transfer implication: transfer the source-state provenance checklist, not a rule. RGIE/PIPM should keep condensate activity, complementarity, inventory/atom, molecule burden, active-condensate burden, and removed-correction state together until a smaller source-proven group closes both layers.

Decision: RHS residual requires full coherent RHS source state.

## Compact RHS Source-State Lineage Addendum

The RHS source-state lineage compact audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_source_state_lineage_compact.json` follows the full coherent RHS source-state result upstream through the entrance-smoke lineage stages (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Stage availability and source-state residuals:

- FastChem and KL both expose compact condensate source-state snapshots for post-initial scan, active selection reset, entry seed, iter0 pre-reduced Newton, iter1 pre-partition, iter1 RHS assembly entry, and iter1 reduced-Newton solve entry.
- FastChem also exposes `post_correctValues_update` and `post_correctValues_refreshed_all_active_state`; KL does not expose symmetric exact post-refresh stages. The audit uses KL `post_partition_split` as the nearest proxy for those two stages and marks `log_activity`, tau/log-tau, old number density, activity correction, molecule cache, and element inventory/atom state as stale or recomputed later.
- Condensate activity, condensate complementarity, and active-row universe/burden state already diverge before candidate selection in both focused layers. Layer `45:-10` first activity norm is `73.529`; layer `60:-5` first activity norm is large as well, and both layers retain KL-only active rows after selection.
- Removed-correction state first becomes material at RHS assembly / retained-removed split, with KL-only rows feeding the split (`7` KL-only active rows at layer 45 and `9` at layer 60 in the shared RHS projection).
- Inventory/atom and molecule-state upstream lineage remain unresolved at pre-RHS stages because the compact lineage lacks symmetric total-density/epsilon, atom-density, full-element-vector, mass-action, and molecule-cache handoff fields.

Consistency and replay:

- RHS formula/source consistency still closes at RHS assembly: scaled reconstructed RHS max errors are `1.42e-14`/`1.42e-14` for FastChem/KL at layer 45 and `2.84e-14`/`1.42e-14` at layer 60.
- Current KL RHS assembly state leaves residuals `22.53965045170672` and `258.2372817218515`.
- FC iter1 RHS assembly / full RHS state closes to `8.556197725637401e-12` and `1.3137289236262653e-11`.
- Earlier-stage full-source replays are blocked by exact missing fields: KL post-refresh full RHS source terms, KL post-refresh molecule cache, KL post-refresh element inventory/atom state, FC post-refresh full scaled RHS vector, FC post-refresh element terms, and FC pre-partition element-row source aggregates.

Active-row and molecule split:

- Active-row universe membership is a real contributor but cannot be separated from state values with the current compact fields. The blocked tests are `FC active universe + KL n_c`, `KL active universe + FC n_c`, KL-only rows removed only, and FC removed set applied only.
- Direct FC molecule cache replay is the already-known `molecule_state` source group: it improves both layers (`20.4161`, `207.8179`) but does not close. Cache/vector/mass-action split remains blocked by missing cross-state full element vectors and mass-action constants at the lineage stage; no hvector bridge is promoted.

KL production implication: RHS formulas are equivalent for the emitted source variables, and the remaining mismatch is source-state handoff distributed across multiple upstream stages. No active-selection, post-refresh, partition, RHS-assembly, molecule-cache, hvector, guarded-mode, or production solver rule is promotable. Full coherent RHS state remains diagnostic-only.

RGIE/PIPM transfer implication: transfer the lineage checklist, not a handoff rule. RGIE/PIPM should emit symmetric post-refresh and pre-RHS source snapshots plus expanded element source vectors before adopting any partial RHS state handoff.

Decision: full RHS source-state mismatch is distributed across multiple upstream stages.

## Compact Preselection Lineage Closure Addendum

The preselection-divergence and missing-field closure audit `results/actual_fastchem_gas_phase_transplant_phase2_rhs_preselection_lineage_closure_compact.json` follows the distributed RHS lineage result back through the full condensate catalog at post-initial activity/maxDensity scan for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Preselection decomposition:

- The full-catalog table covers `186` condensates in each focused layer. FastChem and KL both materialize all catalog rows at the preselection table level; the divergence is therefore not dominated by full-catalog row materialization.
- Layer `45:-10` row classifications are `log_activity_value_mismatch=139`, `donor_term_mismatch=34`, `lnK_or_hvector_mismatch=7`, and `threshold_decision_mismatch=6`.
- Layer `60:-5` row classifications are `log_activity_value_mismatch=90`, `donor_term_mismatch=70`, `lnK_or_hvector_mismatch=17`, and `threshold_decision_mismatch=9`.
- For paired rows, the audit decomposes `delta_log_activity = delta_lnK + delta_donor_sum + residual`. Value mismatch dominates over row-universe mismatch, but value mismatch is mixed between donor and lnK/hvector components, not a single source rule.
- Threshold/candidate differences remain material because they create the later KL-only selected-active rows (`6` in the stage-level layer-45 lineage table and `9` in layer 60), but they do not explain the full RHS residual alone.

Complementarity and active burden:

- Pre-seed complementarity is not a physically meaningful `n_c/lambda/tau` comparison. The audit reclassifies the earliest meaningful complementarity divergence to `entry_seed`.
- Active-condensate burden can be split by universe counts: full materialized rows match (`186`/`186`), while candidate/selected/RHS-active universes differ (`103` vs `109` at layer 45; `116` vs `125` at layer 60). Value dominance remains blocked by missing per-condensate burden contributions and cross-universe `n_c` values.

Inventory, molecule, and removed correction:

- Preselection rows already expose total-density/epsilon provenance for conserved-inventory donor checks (`425` per-element entries in each focused layer), and RHS assembly exposes total-inventory and atom term residuals. Stage-by-stage inventory/atom classification remains unresolved because symmetric FastChem/KL total-element-density-by-element and atomic-density-by-element fields are still missing at every lineage stage.
- RHS assembly exposes `495` FastChem and `495` KL molecule cache records plus molecule burden vectors. Direct FC molecule cache replay remains the known molecule-state improvement (`20.4161`, `207.8179`) but does not close. Full element-vector/cache/mass-action split remains blocked by missing cross-state full element vectors and lineage-stage mass-action constants; no hvector bridge is promotable.
- Removed correction remains a later partition/RHS-assembly issue. Pre-partition removed sets are empty on both sides; post-partition FastChem has removed rows while KL has none in the compact split, and layer 45 has a nonzero removed-correction RHS residual while layer 60 is zero. Per-removed-row source expansion is still missing, but formula mismatch is not indicated.

Stage replay: newly available fields do not create an earlier closing state. FC exact post-refresh and pre-partition full RHS source replays remain blocked by missing full scaled RHS vectors and element-row source aggregates. FC iter1 RHS assembly/full RHS state still closes (`8.56e-12`, `1.31e-11`); current KL RHS assembly state remains open (`22.5397`, `258.2373`).

KL production implication: no preselection row-universe, donor, lnK/hvector, threshold, active-selection, inventory/atom, molecule-cache, removed-partition, guarded-mode, or production solver rule is promotable. The next frontier is diagnostic field closure, not a solver change.

RGIE/PIPM transfer implication: transfer the preselection and missing-field checklist, not a rule. RGIE/PIPM should keep full-catalog preselection rows, donor/lnK decomposition, active-universe counts, symmetric element vectors, molecule caches, and removed-row source expansion together before adopting any partial handoff.

Decision: full RHS source-state mismatch remains distributed across active selection, refresh, partition, and RHS assembly.

## Compact Preselection Activity-Value Addendum

The preselection activity-value and active-burden decomposition audit `results/actual_fastchem_gas_phase_transplant_phase2_preselection_activity_value_compact.json` closes the full-catalog activity formula and expands the active-burden universe/value split for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Preselection activity closure:

- Both focused layers cover all `186` paired condensate rows.
- FastChem reconstructs `log_activity = lnK_final_density_basis + donor_sum` to max residual `1.14e-13` at layer `45:-10` and `4.55e-13` at layer `60:-5`.
- KL reconstructs the same identity to exact compact precision (`0.0` max residual in both focused layers).
- Formula closure therefore does not indicate an activity formula mismatch. The remaining preselection mismatch is value/provenance, not an absent formula term.

Donor/lnK counterfactuals and provenance:

- Current KL preselection activity has candidate-set Jaccard `0.945` at layer 45 and `0.928` at layer 60 against FastChem.
- Replacing only lnK with FastChem worsens candidate agreement (`0.617`, `0.689`), and replacing only donor state with FastChem also worsens candidate agreement (`0.565`, `0.635`).
- Replacing both lnK and donor with FastChem recovers the FastChem candidate set by construction, but this is the full FastChem activity state and is not a smaller source rule.
- Row classifications remain dominated by donor/lnK cancellation (`145` rows at layer 45 under the mixed counterfactuals), with donor-dominated and lnK-dominated rows still present. Top donor residuals come from atomic element density stage differences, especially O/F/Cr/Cl/H at layer 45 and Cr/O/F/Cl/Al at layer 60.
- Large lnK rows retain pressure-density gauge/source-vs-final-lnK provenance signals, but no hvector or density-gauge bridge is promoted.

Active-burden decomposition:

- Candidate-stage burden values are not physically meaningful before seeded `n_c`; the audit records this missing-stage condition rather than using it as a causal rule.
- At selected-active and RHS-active stages, layer 45 has `103` shared active rows and `6` KL-only rows; layer 60 has `116` shared active rows and `9` KL-only rows.
- The exact set split shows both shared-row value residuals and KL-only universe residuals are large. Layer 45 selected/RHS-active norms are `4.064e15` shared-value and `5.400e15` KL-only universe; layer 60 norms are `5.580e14` and `7.522e14`.
- Active-burden replay variants do not improve either focused layer. Removing KL-only active-burden contribution or substituting shared FastChem `n_c` worsens the raw residuals (`94.63`/`95.82` at layer 45 and `360.31`/`362.52` at layer 60), compared with current KL RHS residuals `22.54` and `258.24`.
- Removed correction remains a later partition/RHS-assembly issue and is excluded from the preselection causal classification.

KL production implication: full-catalog materialization is ruled out as the dominant cause, and activity formulas close on both sides. The remaining preselection activity mismatch is donor/lnK mixed and the active-burden split is mixed universe/value. No donor, lnK/hvector, gauge, threshold, candidate-selection, active-burden, guarded-mode, or production solver rule is promotable.

RGIE/PIPM transfer implication: transfer the formula-closure table, donor/lnK counterfactuals, source-record provenance, and exact active-burden set split together. Do not port a standalone donor, lnK, candidate, CH4, or active-burden rule from this smoke.

Decision: preselection/activity lineage remains mixed or inconclusive.

## Compact Candidate-to-Active Selection Addendum

The candidate-to-active selection provenance audit `results/actual_fastchem_gas_phase_transplant_phase2_candidate_active_selection_compact.json` follows the selected-active row mismatch after the preselection activity-value audit for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Selection trace availability:

- FastChem exposes `post_selectActiveCondensates_reset` rows from `fastchem/fastchem_src/condensed_phase/condensed_phase.cpp::CondensedPhase::selectActiveCondensates`.
- The visible FastChem source rule is `log_activity >= 0` and `is_calculated == false`; the function then resets condensate density and activity correction for selected rows. No phase-rule rank, linear-dependence rank test, replacement, or eviction decision is present in the compact trace.
- KL exposes candidate and active rows through `row_universe_presence_KL` and lifecycle source-stage rows. The visible KL candidate rule is also a `log_activity >= 0` mask from `build_kl_atomic_candidate_masks`; the compact trace does not expose a separate phase/rank eviction rule.

Row-level result:

- Layer `45:-10`: FastChem has `103` candidates and `103` selected-active rows; KL has `109` candidates and `109` selected-active rows. There are `103` candidate-and-selected rows on both sides, `77` noncandidate rows on both sides, and `6` KL-only candidate/selected rows.
- Layer `60:-5`: FastChem has `116` candidates and `116` selected-active rows; KL has `125` candidates and `125` selected-active rows. There are `116` candidate-and-selected rows on both sides, `61` noncandidate rows on both sides, and `9` KL-only candidate/selected rows.
- There are no rows where the candidate flag agrees but selected-active status differs. The selected-active mismatch is therefore the candidate-threshold mismatch in the compact trace.
- Layer 45 has `6` candidate-threshold mismatches (`4` near-threshold, `2` large-margin). Layer 60 has `9` candidate-threshold mismatches (`1` near-threshold, `8` large-margin).

Counterfactual and active burden replay:

- Current KL candidate plus KL selection gives the already-known selected-set Jaccard values `0.944954128440367` and `0.928`, with raw residuals `22.53965045170672` and `258.2372817218515`.
- Replacing the KL selected-active set with the FastChem selected-active set, or removing rows FastChem rejects at the threshold, gives selected-set Jaccard `1.0` but worsens raw residuals to `95.81632051178373` and `362.52332123938066`.
- Active-burden decomposition remains mixed: layer 45 selected/RHS-active shared-value norm is `4.064e15` and KL-only universe norm is `5.400e15`; layer 60 norms are `5.580e14` and `7.522e14`.
- The KL-only selected rows explain the KL-only burden rows, but removing them is not a two-layer replay improvement and does not close the full coherent RHS source state.

KL production implication: full-catalog materialization and activity formula closure are ruled out as blockers, and candidate-to-active selection does not expose an independent phase/rank/order rule in the current compact trace. The active selection mismatch is coupled to the donor/lnK activity state that creates the candidate-threshold differences. No selected-row transplant, missing-eviction rule, threshold rule, guarded mode, or production solver rule is promotable.

RGIE/PIPM transfer implication: transfer the selected-active provenance table and the explicit missing-fields list. Do not port a FastChem selected-active row set, KL eviction, or threshold/candidate rule without source-proven donor/lnK state alignment and two-layer solve-space replay improvement.

Decision: active-selection mismatch is coupled to donor/lnK activity-state and cannot be isolated.

## Compact Activity-Burden Pair Addendum

The coherent activity-state plus active-burden audit `results/actual_fastchem_gas_phase_transplant_phase2_activity_burden_pair_compact.json` follows the candidate-to-active result one step further: it keeps condensate activity state and active-condensate burden state together in solve space under the coherent FastChem Jacobian for the entrance smoke only (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Focused row tables and gauge pairs:

- The audit emits focused row tables for all KL-only candidate/selected rows plus the top shared-value burden rows. The KL-only rows are:
  - layer `45:-10`: `114`, `125`, `139`, `180`, `56`, `7`
  - layer `60:-5`: `12`, `120`, `125`, `180`, `3`, `52`, `58`, `73`, `77`
- Within each side, gauge-pair bookkeeping closes: density-gauge and source-gauge reconstructions preserve the emitted log activity, so the current problem is not an activity formula bug.
- The KL-only candidate rows are mostly classified by isolated lnK-side movement in this compact causality test, with one donor-driven row in each layer:
  - layer `45:-10`: `KL-only due to lnK state = 5`, `KL-only due to donor state = 1`
  - layer `60:-5`: `KL-only due to lnK state = 8`, `KL-only due to donor state = 1`
- This does not overturn the earlier donor/lnK coupling result. The isolated flip test is local to KL-only threshold rows; globally, single-side donor or lnK substitution still worsens candidate agreement.

Active-burden decomposition:

- Active burden remains mixed universe/value at the selected-active and RHS-active stages.
- Layer `45:-10`: shared-value norm `4.064e15`, KL-only universe norm `5.400e15`.
- Layer `60:-5`: shared-value norm `5.580e14`, KL-only universe norm `7.522e14`.
- KL-only rows explain the KL-only burden universe component, but the largest total contributors still include large shared-row value residuals. CH4 remains the top KL-only species contributor, but it is not a standalone rule.

Coherent solve-space replay:

- Current KL RHS leaves raw residuals `22.53965045170672` and `258.2372817218515`.
- `FC log_activity` alone worsens layer `45:-10` to `74.1879` and improves only layer `60:-5` to `101.0823`.
- `FC all_active_condensate_burden` alone worsens both layers to `94.2256` and `359.5332`.
- KL-only-row-only edits fail as well:
  - KL-only activity-only replay is neutral in the current compact solve (`22.5397`, `258.2373`)
  - KL-only burden-only replay worsens to `99.6415` and `365.3090`
  - KL-only activity + KL-only burden replay is the same worsening object in the current compact solve
- The coherent pair `FC log_activity + FC all_active_condensate_burden` is the useful mixed replay:
  - layer `45:-10`: `22.5397 -> 3.0603`
  - layer `60:-5`: `258.2373 -> 2.5286`
- The pair improves both layers but does not close them. Full coherent RHS source state still closes to `8.48e-12` and `1.28e-11`.

Interpretation:

- The known useful pair from the earlier RHS term interaction audit is therefore explained by mixed activity/burden source state, not by KL-only universe only, not by shared-row `n_c` only, and not by donor or lnK alone.
- Once the coherent activity/burden pair is applied, the remaining residual sits in the other RHS source groups required by the full coherent RHS state: molecule, inventory/atom, removed-correction, and condensate complementarity.

KL production implication: active-selection mismatch still equals candidate-threshold mismatch, selected-row transplant still worsens, donor/lnK activity state remains coupled, and active burden remains mixed universe/value. The coherent activity/burden pair is a strong diagnostic replay but not a promotable source-state rule.

RGIE/PIPM transfer implication: transfer the pairwise provenance result, not a rule. Do not port selected-row parity, burden-only, activity-only, CH4-only, or KL-only-row-only actualization. If RGIE/PIPM wants to reuse the pair insight, it must keep activity and active burden coupled and then continue into molecule/inventory/removed source-state diagnostics.

Decision: activity-burden replay helps but residual moves to molecule/inventory/removed source state.

## Compact Post-Activity-Burden Residual Addendum

The post-activity-burden compact audit `results/actual_fastchem_gas_phase_transplant_phase2_post_activity_burden_residual_compact.json` reuses the same entrance-smoke scope (`45:-10`, `60:-5`), the coherent FastChem Jacobian, and the scaled RHS convention. It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, or RHS behavior.

Post-activity-burden baseline:

- `baseline_AB = KL RHS + FC condensate_rhs_terms.log_activity + FC element_rhs_terms.all_active_condensate_burden`.
- The baseline reproduces the established residuals within `1e-9`: layer `45:-10` gives `3.0603000036721433` and layer `60:-5` gives `2.5285823085810524`.

Remaining-group replay result:

- The smallest remaining source-state group that improves both layers is condensate complementarity. `baseline_AB + complementarity` reduces the residual to `0.5620` at `45:-10` and `0.9142` at `60:-5`.
- Molecule-only, inventory/atom-only, and removed-only replays do not improve both layers from `baseline_AB`.
- `molecule + inventory_atom` improves both layers but does not close (`2.9376`, `1.9877`).
- `complementarity + molecule + inventory_atom` closes layer `60:-5` to `1.30e-11` and reduces layer `45:-10` to `0.08476`, but does not close both.
- The minimal common subset that closes both layers is the full remaining source-state quartet on top of `baseline_AB`: complementarity + molecule + inventory/atom + removed-correction. That gives `8.20e-12` at `45:-10` and `1.30e-11` at `60:-5`.
- Full FC coherent RHS still closes exactly in the replay audit (`0.0`, `0.0`) because it also carries `element_rhs_terms.other`.

Residual-after-AB geometry and attribution:

- Relative to the residual after `baseline_AB`, condensate complementarity is strongly anti-aligned in both layers (`cos=-0.956` and `-0.897`), which is why it is the smallest common improving group.
- Molecule and inventory/atom contributions are almost perfectly cancelling against each other (`cos=-0.99974` in both layers, cancellation index about `0.989` and `0.987`), so neither is a standalone rule even though they are jointly needed for closure.
- Removed correction is tiny at `45:-10` and exactly zero in the focused `60:-5` replay, but it is still part of the minimal common closing subset because layer `45:-10` does not close without it.
- Row-level residual-after-AB attribution is mostly complementarity-dominated (`37/45` residual rows at layer 45 and `31/44` at layer 60), with the remaining rows requiring multiple remaining groups. There is no row-level evidence for a standalone molecule, inventory/atom, or removed-correction production rule.

Source-state provenance and promotion result:

- The winning closing subset is physically mixed: condensate complementarity (`tau/log_tau`, `n_c/log_n`, `lambda/log_lambda`, `activity_correction`) plus molecule state (molecule density cache, full element vector, mass-action constants) plus inventory/atom state (total inventory, atom density, epsilon) plus removed correction (removed set and removed `n/lambda/log_activity/log_tau`).
- Formula mismatch is ruled out for these replayed groups in the compact source-state path. The remaining mismatch is source-state value/stage/partition mixed, not a standalone formula bug.
- KL production implication: no post-activity-burden source-state rule is promotable. Activity-only, burden-only, selected-row, KL-only-row, donor-only, and lnK-only rules remain rejected, and even after taking the proven activity+burden pair as baseline, closure still requires the full remaining RHS source state.
- RGIE/PIPM transfer implication: treat the activity+active-burden pair only as the new provenance baseline. The next transferable checklist is complementarity plus molecule plus inventory/atom plus removed-correction together; do not port any smaller post-activity-burden rule from this smoke.

Decision: post-activity-burden residual requires full remaining RHS source state.

## Compact Complementarity Provenance Addendum

The compact complementarity-provenance audit `results/actual_fastchem_gas_phase_transplant_phase2_complementarity_provenance_compact.json` starts from the proven post-activity-burden baseline `baseline_AB = KL RHS + FC condensate_rhs_terms.log_activity + FC element_rhs_terms.all_active_condensate_burden` and keeps the same entrance-smoke scope (`45:-10`, `60:-5`), coherent FastChem Jacobian, and scaled RHS convention. It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, RHS behavior, or complementarity behavior.

Complementarity source-term and missing-field status:

- The compact source-term table emits iter1 RHS-entry values for selected condensate rows, including `n_c/log_n`, `lambda/log_lambda`, `tau/log_tau`, `activity_correction`, row scaling, and scaled complementarity subterms.
- Exact missing fields are now explicit instead of inferred:
  - layer `45:-10`: `FastChem.iter1_RHS_assembly_entry.row_label.Al(s)`, `FastChem.iter1_RHS_assembly_entry.row_label.CH4(s,l)`, `KL.iter1_RHS_assembly_entry.row_label.Al(s)`.
  - layer `60:-5`: `FastChem.iter1_RHS_assembly_entry.row_label.CH4(s,l)`.
- The lineage audit also records missing KL post-`correctValues` source fields because `actual_source_state_by_stage` does not expose exact KL rows for `post_correctValues_update` or `post_correctValues_refreshed_all_active_state`. The compact report lists the exact missing stage fields: `KL.post_correctValues_update.{number_density_old,activity_correction_old,tau,log_tau}` and `KL.post_correctValues_refreshed_all_active_state.{number_density_old,activity_correction_old,tau,log_tau}`.

Formula and cross-state provenance result:

- Complementarity source formulas do not self-close in the current comparable compact rows. The reconstructed-vs-emitted closure errors are large:
  - layer `45:-10`: max abs error `9.99` for `activity_correction`, `44.54` for `log_tau_log_n_log_lambda`, and `41.39` after row scaling for full complementarity.
  - layer `60:-5`: max abs error `9.99` for `activity_correction`, `37.61` for `log_tau_log_n_log_lambda`, and `34.46` after row scaling for full complementarity.
- Cross-state formula classification is `row mapping mismatch` in both layers, not a clean formula mismatch, because the comparable iter1 RHS-entry rows are incomplete (`Al(s)` and/or `CH4(s,l)` missing as above). Formula parity is therefore not promotable from this compact audit.

Subcomponent replay from `baseline_AB`:

- The baseline is verified again at `3.0603000036721433` and `2.5285823085810524`.
- `activity_correction` alone is neutral in both layers.
- `log_tau` is the strongest single improving subcomponent:
  - layer `45:-10`: `3.0603 -> 2.0848`
  - layer `60:-5`: `2.5286 -> 0.9991`
- `-log_n` improves both layers but more weakly (`2.7133`, `2.4276`).
- `-log_lambda` worsens or is neutral (`3.3082`, `2.5286`).
- The emitted full `log_tau_log_n_log_lambda` term is the smallest common improving complementarity group and reproduces the known full-complementarity replay:
  - layer `45:-10`: `0.5620237706752297`
  - layer `60:-5`: `0.914249889603709`
- Adding molecule plus inventory/atom after full complementarity closes layer `60:-5` to `1.21e-11` and reduces layer `45:-10` to `0.0847587127674602`.
- Adding removed correction closes both layers to `9.02e-12` and `1.21e-11`.

Row-level attribution and interaction result:

- Residual-after-`baseline_AB` row attribution is dominated by `log_tau` in the compact comparable rows:
  - layer `45:-10`: `log_tau_dominated=16`, `log_n_dominated=9`, `tau_n_lambda_coupled=9`, `log_lambda_dominated=8`, `requires molecule/inventory coupling=3`, `row_mapping_or_absence=2`.
  - layer `60:-5`: `log_tau_dominated=31`, `log_n_dominated=6`, `requires molecule/inventory coupling=5`, `tau_n_lambda_coupled=2`, `row_mapping_or_absence=1`.
- After full complementarity, molecule-only and inventory-only replays each worsen dramatically (`36.61`/`36.91` at layer 45 and `49.73`/`50.42` at layer 60), while the paired `molecule + inventory_atom` replay closes layer 60 and nearly closes layer 45. This reaffirms that the remaining complementarity-corrected residual still requires molecule/inventory coupling.

Promotion result:

- Condensate complementarity remains the strongest common improving direction after `baseline_AB`, but no complementarity subcomponent is promotable.
- `activity_correction` is neutral, `log_tau` is the strongest single improving subcomponent, formula/self-closure does not hold on the comparable compact rows, and closure still requires the post-complementarity molecule/inventory pair plus removed correction.
- KL production implication: no complementarity subcomponent, no full complementarity replay, no guarded mode, and no production solver rule is promotable from this audit.
- RGIE/PIPM transfer implication: treat complementarity as a diagnostic frontier only. Keep the activity+burden baseline, then carry full complementarity provenance together with molecule/inventory coupling and removed correction; do not port a standalone complementarity or `log_tau` rule.

Decision: post-activity-burden residual is dominated by log_tau.

## Compact Complementarity Source-Closure Addendum

The compact complementarity source-closure audit `results/actual_fastchem_gas_phase_transplant_phase2_complementarity_source_closure_compact.json` re-runs the same entrance-smoke scope (`45:-10`, `60:-5`) with exact iter1 RHS-entry source variables. It remains diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, RHS behavior, or complementarity behavior.

Exact source-variable and missing-field result:

- Exact FastChem RHS-entry `n_old`, `activity_correction_old`, `tau`, and `log_tau` now come from `fastchem_internal_trace.entry_seeding_record_sequence_by_stage.post_partition_split` with `newton_iter=1`. Exact KL RHS-entry old-state values come from `reduced_update_diagnostics.full_reduced_system_trace.{old_condensate_number_densities,old_activity_corrections,tau_values}`.
- Layer `45:-10` has `102` exact-mapping rows, `7` `FastChem_absent_KL_present` rows, and `1` unresolved mapping row. Layer `60:-5` has `116` exact-mapping rows and `9` `FastChem_absent_KL_present` rows.
- KL still does not expose exact per-condensate post-`correctValues` rows for `post_correctValues_update` or `post_correctValues_refreshed_all_active_state`. The exact missing fields remain `KL.post_correctValues_update.{number_density_old,activity_correction_old,tau,log_tau}` and `KL.post_correctValues_refreshed_all_active_state.{number_density_old,activity_correction_old,tau,log_tau}`. The audit records the nearest proxy stages and which fields are stale vs recomputed later.

Formula self-closure result:

- Complementarity source reconstruction now closes to roundoff on both sides using only exact RHS-entry variables.
- Layer `45:-10` FastChem/KL max scaled full-complementarity closure error is `2.78e-17` / `5.55e-17`. Layer `60:-5` FastChem/KL max scaled full-complementarity closure error is `2.78e-17` / `5.55e-17`.
- The emitted split vectors `condensate_rhs_terms.terms.{log_tau,minus_log_n,minus_log_lambda}` are still not stored separately in the compact artifact, but the reconstructed split terms sum exactly to the emitted `log_tau_log_n_log_lambda` and full complementarity vectors. Formula mismatch is therefore not the remaining blocker.

Source-clean subcomponent replay from `baseline_AB`:

- `baseline_AB` re-verifies at `3.0603000036721433` and `2.5285823085810524`.
- `activity_correction` remains neutral in both layers.
- Source-clean `log_tau` is the strongest single improving complementarity subcomponent, reducing the residual to `0.8679295995767341` at `45:-10` and `0.9990771116715287` at `60:-5`.
- Source-clean `-log_n` is weaker and mixed: `3.496131654576106` at `45:-10`, `2.4276070575558273` at `60:-5`.
- `-log_lambda` remains neutral at the baseline residuals.
- Full source-clean `log_tau_log_n_log_lambda` reproduces full complementarity at `0.5620237706752316` and `0.9142498896037093`.
- After clean complementarity, `molecule + inventory_atom` closes `60:-5` to `1.21e-11` but leaves `45:-10` at `0.0847587127674602`; adding `removed` closes both layers to `9.01e-12` and `1.21e-11`.

Tau provenance and promotion result:

- With exact source closure in place, the dominant remaining provenance class is now `tau_seed_rule_mismatch`, not generic `log_tau` only. Layer `45:-10` counts are `tau_seed_rule_mismatch=102`, `tau_row_mapping_or_absence=8`. Layer `60:-5` counts are `tau_seed_rule_mismatch=116`, `tau_row_mapping_or_absence=9`.
- The dominant tau divergence is already present at `post_selectActiveCondensates_reset` / immediate entry seed, before later partition or RHS-entry handoff. Later KL post-`correctValues` per-row gaps remain missing, but they are no longer required to show that the earliest common exact divergence is seed-stage tau state.
- Layer `60:-5` remaining residual after clean complementarity is a molecule/inventory cancellation pair; layer `45:-10` keeps only a removed-correction tail.
- KL production implication: `log_tau` is source-clean as a diagnostic signal, but no complementarity subcomponent closes or nearly closes both layers. No `log_tau` rule, complementarity rule, guarded mode, or production solver change is promotable.
- RGIE/PIPM transfer implication: transfer the exact complementarity source-closure checklist and tau-seed provenance, not a rule. RGIE/PIPM should not port a standalone `log_tau` or complementarity rule; it must keep clean complementarity coupled to the later molecule/inventory pair and layer-45 removed tail.

Decision: post-activity-burden residual is dominated by tau seed rule.

## Compact Tau Seed-Rule Addendum

The compact tau seed-rule audit `results/actual_fastchem_gas_phase_transplant_phase2_tau_seed_rule_compact.json` follows the source-clean complementarity closure result and keeps the same entrance-smoke scope (`45:-10`, `60:-5`). It is diagnostic-only and does not change production solvers, guarded modes, presets, maxDensity, row materialization, density-gauge bridge, hvector convention, `J_mol`, `J_ec`, `correctValues`, lifecycle, row scaling, schedules, CH4 lifecycle, row universe, RHS behavior, or complementarity behavior.

Exact tau seed formulas:

- FastChem source is now formula-proven from source plus trace. `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` sets `tau = options.cond_tau * epsilon(reference_element) * total_element_density`, and `fastchem/fastchem_src/condensed_phase/condensate_struct.cpp::Condensate::findReferenceElement` chooses the smallest-abundance-per-stoich support element as `reference_element`.
- The source-level `cond_tau` option default is `1e-15`. Reconstructing `total_element_density = tau / (cond_tau * epsilon(reference_element))` from the exact seed trace is constant across active rows to roundoff: `4.91590875160372e+18` at `45:-10` and `6.838184386244998e+17` at `60:-5`.
- FastChem seed-formula closure is exact enough for audit purposes: max seed reconstruction residual `6.94e-18` at layer 45 and `6.78e-21` at layer 60.
- KL source is also formula-proven from `src/exogibbs/optimize/pipm_rgie_cond.py::build_internal_complementarity_tau`: `tau = tau_scale * exp(epsilon)` with `tau_scale = 1.0`. KL seed closure is exact to roundoff.

Tau source comparison result:

- Exact complementarity source closure remains succeeded, so formula mismatch inside RHS assembly is still ruled out.
- The seed-stage tau mismatch is now source-proven as a formula mismatch upstream of RHS assembly: FastChem uses `cond_tau * total_element_density * epsilon(reference_element)` while KL uses `tau_scale * exp(epsilon)`.
- On the exact mapped rows, the dominant rowwise classification is `tau_support_or_limiting_element_mismatch`: `102` rows at `45:-10` and `116` rows at `60:-5`. The remaining rows are `tau_row_mapping_or_absence` (`7` and `9`).
- Scalar-only or partial tau swaps do not explain the result. Replaying only `cond_tau` or only `total_element_density` worsens badly (`22.26`/`21.84` at layer 45 and `18.39`/`24.55` at layer 60), while a reference-element-only replay still remains weak (`4.47` and `3.85`). Full FC tau is the smallest tau-only replay that helps both layers (`0.86793`, `0.99908`), but it still does not close.

Relation to old entry-seeding result:

- This tau source mismatch is independent of the older `n_seed = max_number_density`, `lambda_seed = 1.0` result. FastChem seed `n`/`lambda` are still not tau-regularized.
- Tau depends on a separate source formula and is therefore not implied by the seed `n`/`lambda` manifold.
- FastChem tau is refreshed later before RHS assembly, but the dominant mismatch already exists at reset / seed. KL tau is effectively carried unchanged from the schedule builder into RHS assembly.

Promotion result:

- The tau source formula is now proven, but no tau rule is promotable.
- `FC tau + KL n/lambda` gives the known source-clean `log_tau` replay (`0.86793`, `0.99908`). Full complementarity remains better (`0.56202`, `0.91425`), and closure still requires the later `molecule + inventory` pair plus the layer-45 removed tail.
- KL production implication: no tau seed rule, no `cond_tau` transplant, no total-density transplant, no epsilon-element rule, no guarded mode, and no production solver change is justified.
- RGIE/PIPM transfer implication: carry the exact tau source-formula checklist only. The transferable result is that the FastChem seed tau source is support/reference-element aware and density-scaled, while KL is schedule-only. Do not port a tau rule from this audit.

Decision: tau source rule is proven but not promotable.

## Compact Post-Complementarity Tail Addendum

The compact post-complementarity tail audit `results/actual_fastchem_gas_phase_transplant_phase2_post_complementarity_tail_compact.json` fixes the source-clean baseline

`baseline_ABC = KL RHS + FC log_activity + FC all_active_condensate_burden + FC full complementarity`

and verifies it at `0.5620237706783301` for `45:-10` and `0.914249889596486` for `60:-5`.

Tail replay result:

- Molecule-only and inventory/atom-only replays are each strongly destructive after complementarity:
  - `45:-10`: `36.6124` and `36.9071`
  - `60:-5`: `49.7254` and `50.4194`
- The paired `molecule + inventory_atom` replay closes `60:-5` to `1.2971603589242228e-11` and reduces `45:-10` to `0.08475871276818528`.
- Adding removed correction closes `45:-10` to `8.199347373900107e-12` while leaving `60:-5` at `1.2971603589242228e-11`.

Contribution geometry and provenance:

- Molecule and inventory/atom are an inseparable cancellation pair after complementarity:
  - layer `45:-10`: cosine `-0.9997438633784276`, cancellation index `0.9886099574272349`
  - layer `60:-5`: cosine `-0.9997471855058516`, cancellation index `0.9870268652530958`
- Molecule-burden aggregate self-closure succeeds to roundoff in scaled solve space from `molecule_density_provenance`.
- Atom-density closure is exact from the iter1 RHS-entry old element state, but total-inventory remains coupled to shared gas-diagnostic `total_element_density_fastchem_trace` and `epsilon_by_element_fastchem_trace` fields rather than an independently emitted KL inventory source vector.
- Removed-correction remains a layer-45 tail. FastChem has one removed condensate row in the focused `45:-10` compact projection, KL has none, and exact per-removed contribution vectors are still not emitted (`FastChem.iter1_RHS_assembly_entry.removed_correction.per_removed_condensate_source_terms`, `KL.iter1_RHS_assembly_entry.removed_correction.per_removed_condensate_source_terms`).

Promotion result:

- The post-complementarity tail is source-proven but still diagnostic-only.
- No molecule-state, inventory/atom-state, removed-correction, guarded-mode, or production solver rule is promotable from this audit.
- KL production implication: keep the source-clean tail attribution as provenance only; do not port the molecule/inventory pair or the removed tail as production behavior.
- RGIE/PIPM transfer implication: carry forward the cancellation-pair diagnosis plus the layer-45 removed tail check, not a solver rule.

Decision: post-complementarity tail requires molecule+inventory pair plus layer-45 removed tail.

## Compact Molecule/Inventory/Removed Source Addendum

The compact molecule/inventory/removed source audit `results/actual_fastchem_gas_phase_transplant_phase2_molecule_inventory_removed_source_compact.json` keeps the same entrance-smoke scope (`45:-10`, `60:-5`) and asks a narrower source-closure question after the post-complementarity tail result.

Source-closure result:

- Molecule burden self-closes in scaled solve space from the exact iter1 RHS-entry molecule caches on both sides. Direct FC molecule replay and FC molecule reconstructed from per-molecule contributions are identical (`36.6124` at `45:-10`, `49.7254` at `60:-5`), so molecule burden itself is not blocked by missing term closure.
- Atom density also self-closes exactly on both sides, and the atom-only replay stays near the baseline (`0.5719` and `0.9261`).
- Independent total-inventory source closure fails badly. Reconstructing `total_inventory_by_element = total_element_density * epsilon` from the emitted gas diagnostics does not reproduce the RHS total-inventory term, and FC/KL side-specific `total_element_density` / `epsilon` fields consumed by RHS assembly are not emitted. The total-inventory-only or full inventory/atom replays become huge (`6.40e4` / `8.65e4` scale raw residuals).
- The factorized molecule/inventory source replay therefore does not reproduce the earlier cancellation pair. At this stricter source-clean level, the pair requires the full coherent gas-state bundle rather than a promotable smaller rule.

Removed-tail result:

- Layer `45:-10` still has a visible FastChem removed row while KL has none, but the exact FastChem per-removed RHS contribution rule is not emitted separately enough to self-close from the current compact source variables.
- The removed-tail classification therefore remains unresolved rather than source-proven. Missing or insufficient fields remain `FastChem.iter1_RHS_assembly_entry.removed_correction.per_removed_condensate_source_terms`, `KL.iter1_RHS_assembly_entry.removed_correction.per_removed_condensate_source_terms`, `KL.iter1_RHS_assembly_entry.old_log_activity_by_active_condensate`, and `KL.iter1_RHS_assembly_entry.removed_condensate_labels`.

Promotion result:

- No molecule-state, inventory/atom-state, removed-tail, guarded-mode, or production solver rule is promotable from this audit.
- KL production implication: the post-complementarity tail remains diagnostic-only. The current source-closure boundary is a coherent gas-state bundle, not a smaller production rule.
- RGIE/PIPM transfer implication: carry forward the exact molecule-cache closure, the exact atom closure, the independent total-inventory source failure, and the unresolved removed-tail source rule as provenance checks only.

Decision: molecule/inventory cancellation requires full coherent gas-state bundle.

## Molecule Timing Resolution Addendum

The compact molecule timing resolution audit
`results/actual_fastchem_gas_phase_transplant_phase2_molecule_timing_resolution_compact.json`
first reconciles the FastChem reconstructed-cache replay path before interpreting KL timing.

FastChem replay consistency:

- FastChem per-molecule mass-action ledgers are present at the post-iter0 cached molecule refresh and iter1 RHS/Jacobian entry.
- Reconstructed FastChem molecule caches match direct cached densities to roundoff:
  - layer `45:-10`: iter1 max log residual `1.1102230246251565e-16`
  - layer `60:-5`: iter1 max log residual `3.552713678800501e-15`
- The prior `FC full element vector + FC mass-action` vs direct-FC discrepancy was a replay-stage/RHS-burden mapping issue in the compact reconstruction path, not a FastChem formula failure. `D_clean` now equals `K_direct` on both layers.

KL cached-stage result:

- KL early atomic stage vectors are available and diagnostic molecule caches were reconstructed for `gas_only_final`, `post_initial_activity_maxdensity_scan`, and `post_selectActiveCondensates_reset`.
- Later requested KL stage vectors remain absent for `post_correctValues_update`, exact `post_correctValues_refreshed_all_active_state`, and `iter1 pre partition`; the KL branch computes the molecule cache inline at RHS/Jacobian assembly.
- No exposed KL cached stage recovers the molecule/inventory cancellation. The KL inline and early cached-stage paired residuals remain destructive:
  - layer `45:-10`: `36.907067382036104`
  - layer `60:-5`: `50.419424423672645` to `50.419424423866325`
- Clean reconstructed FC molecule plus gauge-normalized inventory/atom reproduces direct FC molecule replay:
  - layer `45:-10`: `0.08475871276711094`, then exact removed correction closes to `9.230739629452324e-12`
  - layer `60:-5`: `1.3724132941206335e-11`

Promotion result:

- FastChem cached molecule snapshot remains diagnostic-only; no KL production rule is promotable.
- No guarded mode, production solver fix, molecule-cache timing rule, hvector/mass-action rule, density-gauge rule, row-universe rule, inventory rule, or removed-tail rule is justified.
- KL production implication: the exposed KL stages cannot reconstruct the FastChem cached molecule state that participates in the recovered molecule/inventory cancellation.
- RGIE/PIPM transfer implication: carry the FastChem reconstructed-cache gate and KL cached-stage failure as provenance checks only. A target branch must expose a source-clean cached gas/molecule snapshot before any transfer rule is considered.

Decision: molecule state requires FastChem hidden/coupled snapshot not reconstructable from exposed KL stages.

## KL Later Molecule Snapshot Addendum

The later-stage KL molecule snapshot audit
`results/actual_fastchem_gas_phase_transplant_phase2_kl_later_molecule_snapshots_compact.json`
regenerated the focused entrance smoke for `45:-10` and `60:-5` with diagnostic-only molecule source snapshots at `post_correctValues_update`, `exact_post_correctValues_refreshed_all_active_state`, `iter1_pre_partition`, and `iter1_RHS_assembly_entry`.

Result:

- The field-presence gate passes for all four later KL stages on both focused layers.
- FastChem reconstructed-cache vs direct-cache still passes, and `D_clean == K_direct` remains reconciled:
  - layer `45:-10`: `0.08475871276711094`, closing to `9.230739629452324e-12` only after exact removed.
  - layer `60:-5`: `1.3724132941206335e-11`.
- Early exposed KL stages still fail (`36.907067382036104` at `45:-10`, about `50.4194244239` at `60:-5`).
- Later KL snapshots do not recover molecule/inventory cancellation. `post_correctValues_update`, exact post-refresh, `iter1_pre_partition`, and inline RHS entry all replay at the same failing KL-native residuals:
  - layer `45:-10`: `36.907067382036104`.
  - layer `60:-5`: `50.419424423672645`.

Promotion result:

- Later KL molecule snapshots are now emitted, but they are diagnostic-only.
- No molecule-cache timing, hvector, density-gauge, inventory, removed-tail, guarded-mode, or production solver rule is promotable.
- KL production implication: KL inline RHS recomputation is the best KL-native exposed stage in this audit, but it is still insufficient.
- RGIE/PIPM transfer implication: require a source-clean cached gas/molecule snapshot that actually recovers the cancellation before considering a transfer rule.

Decision: KL inline RHS recomputation is the best KL-native stage but still insufficient.

## KL Later-Stage Distinctness Addendum

The distinctness/freeze-carry audit
`results/actual_fastchem_gas_phase_transplant_phase2_kl_later_stage_distinctness_compact.json`
answers what "KL inline RHS recomputation is the best KL-native stage but still insufficient" means.

Distinctness result:

- On both focused layers, all later KL molecule stages collapse to one effective snapshot.
- Pairwise differences between `post_correctValues_update`, `exact_post_correctValues_refreshed_all_active_state`, `iter1_pre_partition`, and `iter1_RHS_assembly_entry` are exactly zero for:
  - `u`
  - atom / full-element vector
  - molecule cache
- Therefore exact post-refresh does not materially differ from RHS entry, iter1 pre-partition does not materially differ from exact post-refresh, and inline RHS recomputation reproduces an already fixed diagnostic snapshot.

Freeze-and-carry result:

- Carrying any later KL cache to iter1 RHS is exactly equal to the inline RHS molecule cache and gives no improvement:
  - layer `45:-10`: `36.907067382036104`
  - layer `60:-5`: `50.419424423672645`
- Direct FastChem molecule cache remains the diagnostic cancellation boundary:
  - layer `45:-10`: `0.08475871276711094`, then exact removed closes to `9.230739629452324e-12`
  - layer `60:-5`: `1.3724132941206335e-11`
- The frozen KL caches remain far from FastChem RHS-entry cache:
  - layer `45:-10`: max log cache residual `66.90247419848673`
  - layer `60:-5`: max log cache residual `210.94452675909793`

Promotion result:

- No KL later-stage carry is a diagnostic candidate for promotion.
- Exposed KL later stages are insufficient; FastChem cached molecule snapshot remains diagnostic-only.
- KL production implication: the exposed KL later-stage molecule state is static across the audited boundary, so there is no hidden KL-native timing lever in these emitted stages.
- RGIE/PIPM transfer implication: require a genuinely distinct source-clean cached gas/molecule snapshot before considering transfer.

Decision: KL later stages collapse to one effective molecule snapshot.

## KL Gas-Refresh Snapshot Addendum

The gas-refresh snapshot audit
`results/actual_fastchem_gas_phase_transplant_phase2_kl_gas_refresh_snapshot_compact.json`
tests whether a KL diagnostic gas-refresh or coupled-loop atomic state can reproduce the missing FastChem cached molecule state after the later-stage collapse result.

Candidate stages:

- `KL_gas_recoupling_after_postCorrectValues` is available from `actual_fastchem_like_coupled_loop` post-gas-recoupling `ln_nk` capture.
- `KL_exact_post_refresh_then_gas_recompute` is available only as the proxy `gas_replay_final_atomic_element_species_state` from `exact_fastchem_gas_phase_newton_replay`.
- `KL_iter1_pre_partition_then_gas_recompute` cannot be constructed: `fastchem_target_donor_replay_from_gas_replay_final` has a non-finite `e-` entry from `fastchem_narrow_post_gas_solve_donor_transform applied to gas replay final atomic ln_n`.
- `KL_RHS_entry_inline_current` remains the inline reference.

Distinctness and replay result:

- The gas-recoupling and gas-replay-final snapshots are materially distinct from the collapsed later KL snapshot:
  - layer `45:-10`: max `u` difference `14.87958673430007`, max molecule-cache difference `1.2775306763590613`
  - layer `60:-5`: max `u` difference `8.370219832642732`, max molecule-cache difference `455.59137915527765`
- Despite being distinct, they do not improve the selected-row replay:
  - layer `45:-10`: gas-refresh variants remain `36.907067382036104`
  - layer `60:-5`: gas-refresh variants are `50.419424423951064`, slightly worse than inline `50.419424423672645`
- They remain far from the FastChem RHS-entry molecule cache:
  - layer `45:-10`: max/mean log residual `72.80738647765477` / `41.74146353890789`
  - layer `60:-5`: max/mean log residual `156.7200546573116` / `32.70926008936871`

Promotion result:

- No gas-refresh snapshot recovers molecule/inventory cancellation.
- No gas-refresh snapshot is a diagnostic candidate for promotion.
- FastChem hidden/coupled molecule snapshot remains diagnostic-only; no KL production rule is promotable.

Decision: KL gas-refresh snapshots are materially distinct but still insufficient.

## Molecule Input-Vector Provenance Addendum

The input-vector provenance audit
`results/actual_fastchem_gas_phase_transplant_phase2_molecule_input_vector_provenance_compact.json`
tests whether the FastChem cached molecule snapshot failure is already present in the atomic/full-element vector that feeds molecule reconstruction.

FastChem input-vector source:

- FastChem post-iter0 molecule refresh input vector is emitted by `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` at `iter0_post_correctValues_full_element_vector_before_molecule_refresh`.
- FastChem iter1 RHS-entry molecule input is represented by the same refreshed full-element vector source and the iter1 RHS molecule provenance. It includes element/global indices, element names, number densities, log values, fixed-by-condensation flags, `degree_of_condensation`, `phi`, and `epsilon`.

KL input-vector result:

- KL candidate vectors were emitted/reconstructed for gas-only, post-selection reset, post-correctValues, exact post-refresh, gas recoupling, gas-replay final proxy, iter1 pre-partition, and RHS entry.
- Reconstructing molecules with direct FastChem mass-action constants but KL input vectors still fails:
  - layer `45:-10`: KL input-vector variants remain about `36.90706` to `36.90707`; FC input reconstruction is `0.08475871276711094`, closing with exact removed to `9.230739629452324e-12`.
  - layer `60:-5`: KL input-vector variants remain about `50.41942`; FC input reconstruction is `1.3724132941206335e-11`.
- KL input vectors are far from the FastChem iter1 RHS-entry input vector:
  - layer `45:-10`: best mean log residual is still `43.619896422039254` from gas-only; later KL RHS-entry mean is `43.87686375096906`.
  - layer `60:-5`: best mean log residual is still `42.19254824596088` from gas-only; later KL RHS-entry mean is `42.75542280753954`.

Fixed-element / coupled-loop provenance:

- Top residual elements are dominated by FastChem fixed-by-condensation rows, although `e-` is also the largest single residual.
- Across the top residual tables, fixed elements dominate the mismatch counts (`75` vs `15` at `45:-10`; `73` vs `17` at `60:-5`).
- The molecule cache is therefore a downstream symptom of upstream atomic/full-element vector and fixed-element handoff mismatch, not a direct molecule timing lever.

Promotion result:

- No KL molecule timing, gas-refresh, fixed-element handoff, inventory, removed-tail, guarded-mode, or production solver rule is promotable.

Decision: molecule mismatch is dominated by fixed-element handoff mismatch.

## Fixed-Element Handoff Compact Addendum

The fixed-element handoff audit
`results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_handoff_compact.json`
decomposes the FastChem cached full-element vector by source, emits KL per-stage analogues, splits residuals by the FastChem fixed mask, and tests diagnostic-only fixed-element counterfactuals.

FastChem source decomposition:

- The cached FastChem full-element vector remains source-proven at `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`, stage `iter0_post_correctValues_full_element_vector_before_molecule_refresh`.
- Per-element emitted fields are limited to value/log, fixed flag, `degree_of_condensation`, `phi`, and `epsilon`. Separate `total_inventory_contribution`, `free_atomic_gas_contribution`, and `condensed/fixed correction contribution` are still not emitted by that source stage and are therefore reported as missing fields rather than inferred values.

KL analogue decomposition:

- KL analogue vectors are emitted for `gas_only_final`, `post_initial_activity_maxdensity_scan`, `post_selectActiveCondensates_reset`, `post_correctValues_update`, exact post-refresh, gas recoupling, gas-replay final proxy, `iter1_pre_partition`, and `iter1_RHS_entry`.
- Later KL stages still collapse numerically, but the fixed-subset divergence already exists before `post_selectActiveCondensates_reset`, so the earliest exposed mismatch stays upstream of the later refresh / RHS-entry bundle.

Fixed vs non-fixed split:

- At `iter1_RHS_entry`, fixed elements still dominate the top residual set on both focused layers, while `e-` remains the largest single non-fixed residual.
  - layer `45:-10`: fixed mean log residual `43.03152208289549`, non-fixed mean `43.03900830273183`, electron residual `66.66365768977406`, top-count split `7` fixed vs `0` non-fixed.
  - layer `60:-5`: fixed mean log residual `41.28868465004005`, non-fixed mean `41.06646883523925`, electron residual `83.24621631922903`, top-count split `7` fixed vs `0` non-fixed.

Handoff counterfactuals:

- Current KL input vector remains at the KL destructive scale:
  - layer `45:-10`: molecule mean log residual `147.92838359730413`, selected-row residual `36.90705832854512`.
  - layer `60:-5`: molecule mean log residual `140.48218322609824`, selected-row residual `50.41942341750188`.
- Replacing FC fixed-element values alone strongly improves molecule reconstruction but does not produce a usable RHS transplant:
  - layer `45:-10`: fixed-only molecule mean log residual `15.226673958408794`.
  - layer `60:-5`: fixed-only molecule mean log residual `19.407748535067963`.
- Non-fixed-only and electron-only replays do not explain the mismatch:
  - layer `45:-10`: non-fixed-only `148.18922607187318`, electron-only `147.5263302999384`.
  - layer `60:-5`: non-fixed-only `140.73107091601747`, electron-only `138.4526607344315`.
- Adding FC electron on top of FC fixed values improves molecule residual further (`0.4347374576125725`, `0.565493860944129`) but still does not create a safe closing RHS replay; the selected-row residuals blow up because the partial handoff is not a coherent RHS source-state bundle.
- Full FC input vector and direct FC molecule replay remain the only source-proven molecule-side upper bounds:
  - layer `45:-10`: selected-row residual `0.08475871276711094`.
  - layer `60:-5`: selected-row residual `1.3724132941206335e-11`.

Promotion result:

- Molecule cache remains a downstream symptom only.
- Fixed-element handoff is source-proven but remains diagnostic-only.
- No fixed-element value, fixed-mask, `phi` / `degree_of_condensation`, electron, molecule timing, gas-refresh, inventory, removed-tail, guarded-mode, or production solver rule is promotable.

Decision: mismatch is dominated by fixed-element values themselves.

## Fixed-Element Overwrite Source Addendum

The overwrite-source audit
`results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_source_compact.json`
keeps the focus at the FastChem cached full-element vector boundary and asks whether the dominant fixed-element mismatch is in the overwrite values themselves or in a later additive source split.

FastChem overwrite-source result:

- The cached fixed-element rows are still source-proven at `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`, stage `iter0_post_correctValues_full_element_vector_before_molecule_refresh`.
- For fixed-by-condensation elements, the cached value is a full overwrite from local `elem_densities_new[i]` via `elements_cond[i]->number_density = elem_densities_new[i]`.
- In the focused reduced branch this overwrite origin remains `CondensedPhase::correctValues`, not `correctValuesFull`.
- The overwrite/carry-forward split remains the only source-clean decomposition currently available at that boundary. The audit continues to report `carry_forward_prior_value_if_available`, `free_atomic_gas_component_if_explicitly_available`, `condensed_or_fixed_correction_component_if_explicitly_available`, `total_inventory_component_if_explicitly_available`, and `electron_specific_component_if_explicitly_available` as missing rather than inferred. The exact missing local variable for prior carry-forward is `full_element_densities_before_write[i]` in `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`.

KL analogue result:

- KL analogue vectors are emitted for `gas_only_final`, `post_selectActiveCondensates_reset`, `post_correctValues_update`, exact post-refresh, gas recoupling, gas-replay final proxy, and `iter1_RHS_entry`.
- No source-clean overwrite-like KL operation is exposed on these stages. `fixed_element_bookkeeping`, `degree_of_condensation_or_phi_analogue`, and carry-forward vs overwrite distinction remain missing; gas-recoupling also lacks source-clean inventory/budget closure.

Counterfactual result:

- Overwrite-only replay remains the decisive single-component improvement:
  - layer `45:-10`: `147.92838359730413 -> 15.226673958408794`
  - layer `60:-5`: `140.48218322609824 -> 19.407748535067963`
- Electron-only replay stays near the destructive KL baseline:
  - layer `45:-10`: `147.5263302999384`
  - layer `60:-5`: `138.4526607344315`
- Overwrite plus electron improves molecule reconstruction further but still does not yield a coherent selected-row RHS replay:
  - layer `45:-10`: molecule mean `0.4347374576125725`, selected-row residual `64072.83656065602`
  - layer `60:-5`: molecule mean `0.565493860944129`, selected-row residual `86511.80029512591`
- Full FC cached input vector and direct FC molecule replay remain the only coherent molecule-side upper bounds:
  - layer `45:-10`: selected-row residual `0.08475871276711094`
  - layer `60:-5`: selected-row residual `1.3724132941206335e-11`

Promotion result:

- Overwrite source is further resolved but remains diagnostic-only.
- No overwrite, condensed/fixed correction, total-inventory, electron, guarded-mode, or production solver rule is promotable.

Decision: mismatch is dominated by overwrite values themselves.

## Fixed-Element Materialization Boundary Addendum

The KL materialization-boundary audit
`results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.json`
tests whether the diagnostic gas-recoupling output is actually carried into the later molecule-refresh input vector, or whether KL reconstructs molecules from a different current-`u` state.

Boundary result:

- The gas-recoupling output exists as a true atomic state from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::actual_fastchem_like_coupled_loop post gas recoupling ln_nk capture`.
- The later molecule input used at `iter1_RHS_assembly_entry` is emitted by `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- On both focused layers, the gas-recoupling output and the iter1 molecule input are not the same:
  - layer `45:-10`: max `u` difference `14.87958673430007`
  - layer `60:-5`: max `u` difference `8.370219832642732`
- The pre-gas current-state proxy `post_correctValues_update` is identical to the iter1 molecule input and iter1 RHS-entry input on both layers to tolerance (`0.0` max `u` and full-element-vector difference).
- The iter1 molecule cache is source-marked as computed inline and not carried from an earlier refresh stage (`cache_is_computed_inline = true`, `cache_is_carried_from_earlier_stage = false`).

Counterfactual result:

- Using the gas-recoupling output directly as molecule input does not improve selected-row replay and is slightly worse on layer `45:-10`:
  - layer `45:-10`: current KL `147.92838359730413`, gas-recoupling direct `149.20929529868224`
  - layer `60:-5`: current KL `140.48218322609824`, gas-recoupling direct `140.07824601545863`
- Full FC cached input vector / direct FC replay remain the only closing upper bounds:
  - layer `45:-10`: selected-row residual `0.08475871276711094`
  - layer `60:-5`: selected-row residual `1.3724132941206335e-11`

Missing fields:

- The exact pre-gas boundary vector is not separately emitted in the fresh artifact. The exact missing field is `gas_recoupling_diagnostics.post_condensed_phase_fixed_atomic_element_species_state` from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::actual_fastchem_like_coupled_loop`.
- KL still does not expose the exact fixed-mask consumer at the molecule-input side. The missing field remains `iter1_molecule_input.fixed_element_bookkeeping_consumer` from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.

Promotion result:

- The missing object is now explicit: KL fixed-aware gas replay exists, but the later molecule input is rebuilt from a different current-`u` state instead of carrying the gas-recoupling output.
- This boundary is source-proven but remains diagnostic-only.
- No materialization, gas-recoupling adoption, fixed-mask, guarded-mode, or production solver rule is promotable.

Decision: KL molecule input vector is recomputed from a different current-u state.

## Exact Fixed-Row Subspace Trace Addendum

The exact fixed-row subspace compact audit
`results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_compact.json`
was run on layers `45:-10` and `60:-5` with entrance-smoke scope only.

Trace availability:

- Diagnostic-only newton-step trace emission was added for an
  `exact_same_iteration_fixed_row_reduced_system` stage in
  `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep`.
- The rebuilt entrance-smoke trace still does not contain
  `condensed_phase_exact_fixed_row_reduced_system` records for either focused
  layer.
- The exact missing object remains the same-iteration labelled matrix rows
  sourced from local `jacobian` in `CondPhaseSolver::newtonStep`, together with
  exact same-iteration row and column labels.

Closure and replay result:

- Fixed-row equation closure `J_i,: z = b_i` was not evaluated because the exact
  same-iteration labelled matrix rows were unavailable in the regenerated trace.
- Coherent subspaces `S0` through `S5` were defined in the compact audit, but
  their condition numbers, outside coupling, Schur residuals, and replay metrics
  were not evaluated because the exact matrix trace gate failed.
- No fallback to the previous emitted labelled analogue was promoted for this
  audit.

Promotion result:

- No coherent fixed-row subspace is a diagnostic candidate from this run.
- No fixed-row RHS, Jacobian, outside-coupling, Schur-complement, full-system,
  guarded-mode, or production solver rule is promotable.
- Fixed-row solve-space provenance is source-proven through the scalar bridge,
  but exact matrix provenance remains unavailable and diagnostic-only.

Decision: exact fixed-row solve-space provenance remains mixed or inconclusive.

## Exact Fixed-Row Trace Repair Addendum

The trace-repair entrance smoke
`results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_trace_repair_entrance_smoke.json`
was run on layers `45:-10` and `60:-5` after adding a diagnostic-only v2
probe and compact fixed-row matrix emitter in
`fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep`.

Source inspection:

- `jacobian` is local to `CondPhaseSolver::newtonStep`.
- `rhs` is local to `CondPhaseSolver::newtonStep`.
- `result` and `scaling_factors` are output references passed into
  `CondPhaseSolver::newtonStep`.
- `assembleJacobian` has finished immediately after assigning
  `scaling_factors = assembleJacobian(..., jacobian, diagnostic_newton_iter)`.
- `assembleRightHandSide` has finished immediately after assigning
  `objective_function = assembleRightHandSide(..., scaling_factors, rhs)`.
- `solveSystem` has consumed the scaled `jacobian` and `rhs` immediately after
  `const bool jacobian_is_invertible = solveSystem(jacobian, rhs, result)`.
- Condensate and element row/column labels are constructible at that post-solve
  point from `condensates_jac`, `condensates`, `elements`, and
  `diagnostic_all_elements`.

Repair status:

- A clean rebuild/import identity check loaded
  `/home/kawahara/exogibbs/fastchem/pyfastchem.cpython-310-x86_64-linux-gnu.so`
  and a minimal trace at the focused layer inputs emitted both
  `condensed_phase_exact_fixed_row_reduced_system_probe` and
  `condensed_phase_exact_fixed_row_reduced_system` with build marker
  `exact_fixed_row_subspace_trace_v2`.
- The requested full phase2 entrance smoke still failed the raw field-presence
  gate. Its preserved raw traces contain `condensed_phase_iter1_full_reduced_system`
  records but no v2 probe or exact fixed-row matrix records.
- Preserved raw failing traces:
  `results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_trace_repair_layer45_raw_fastchem_cond_trace.jsonl`
  and
  `results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_trace_repair_layer60_raw_fastchem_cond_trace.jsonl`.

Gate result:

- Probe records: absent in the requested entrance-smoke raw traces.
- Exact fixed-row matrix records: absent in the requested entrance-smoke raw
  traces.
- Fixed-row submatrices, fixed RHS entries, solver result, and exact labels:
  unavailable from the requested entrance smoke.
- Coherent subspace replay did not run.

Promotion result:

- No physical conclusion is allowed from this run.
- No coherent fixed-row diagnostic candidate exists from the requested smoke.
- No fixed-row RHS, Jacobian, outside-coupling, Schur-complement, full-system,
  guarded-mode, or production solver rule is promotable.

Decision: build/import path mismatch.

## Exact Fixed-Row Import Repair Addendum

The import-repair entrance smoke
`results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_import_repair_entrance_smoke.json`
was run on layers `45:-10` and `60:-5` after cleaning both candidate
`pyfastchem` shared objects, rebuilding, and copying the rebuilt extension into
`fastchem/python`.

Import identity and marker status:

- The rebuilt root extension and the `fastchem/python` copy have matching
  checksum
  `7493746969d5ce442beaa2339e8dc4e35d7d7446ef3f006038b2012623d126ee`.
- The phase2 audit process loaded
  `/home/kawahara/exogibbs/fastchem/python/pyfastchem.cpython-310-x86_64-linux-gnu.so`.
- The phase2 audit reports expected build marker
  `exact_fixed_row_subspace_trace_v2`; `FASTCHEM_COND_TRACE_JSONL` is false at
  import time but true during FastChem trace calls.
- Legacy `condensed_phase_iter1_full_reduced_system`, v2 probe, and v2 exact
  fixed-row reduced-system records all carry the marker in the fresh raw trace.

Raw trace gate:

- Layer `45:-10`: `2` legacy iter1 full-system records, `4` v2 probe records,
  and `4` v2 exact fixed-row reduced-system records. The exact record contains
  `125` row labels, `125` column labels, `22` fixed rows, non-empty fixed
  rows-by-all-columns and all-rows-by-fixed-columns submatrices, fixed RHS
  entries, and a `125`-entry solver result.
- Layer `60:-5`: `2` legacy iter1 full-system records, `4` v2 probe records,
  and `4` v2 exact fixed-row reduced-system records. The exact record contains
  `139` row labels, `139` column labels, `23` fixed rows, non-empty fixed
  rows-by-all-columns and all-rows-by-fixed-columns submatrices, fixed RHS
  entries, and a `139`-entry solver result.
- The exact same-iteration matrix rows are now available in the requested phase2
  entrance smoke. No fallback to the older labelled analogue was used.

Compact replay result:

- The compact audit
  `results/actual_fastchem_gas_phase_transplant_phase2_exact_fixed_row_subspace_import_repair_compact.json`
  ran only after the v2 gate passed.
- Fixed-row equation closure `J_i,: z = b_i` closes to roundoff in both focused
  layers: max relative error `7.284557345559698e-16` at `45:-10` and
  `1.2551568802448063e-15` at `60:-5`.
- Coherent subspaces `S0` through `S5` were evaluated with exact same-iteration
  matrix data. No coherent subspace gives a promotable two-layer closing rule;
  the compact audit classifies the solve-space provenance as diagnostic-only and
  mixed/inconclusive.
- Current KL and bridged/full reduced-system variants remain close to each other
  on the selected-row metric, while full FC cached input and direct FC molecule
  replay are destructive under this compact selected-row replay.

Promotion result:

- The import-path mismatch is repaired for the requested phase2 smoke.
- Exact same-iteration fixed-row matrix availability is now proven.
- No coherent fixed-row subspace is promotable.
- No fixed-row RHS, fixed-row Jacobian, outside-coupling, Schur-complement,
  full-system coherence, guarded-mode, or production solver rule is promotable.
- RGIE/PIPM implication: fixed-row matrix closure can now be used as a
  diagnostic provenance check, but the result must not be transferred as a solver
  rule. Keep exact matrix trace, closure, and coherent-subspace comparison as
  diagnostic-only requirements.

Decision: exact matrix trace gate passes and subspace replay completed.

## Fixed-Row Subspace Tail-Context Reconciliation

The compact tail-context reconciliation audit
`results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_subspace_tail_context_compact.json`
uses the repaired exact fixed-row matrix trace only after first reproducing the
known post-complementarity molecule/inventory ladder in one selected-row code
path.

Tail ladder gate:

- Layer `45:-10`: `baseline_ABC` reproduces at
  `0.5620237706782278` selected-row mean residual. FC molecule alone is
  destructive (`36.61241184552874`), FC inventory/atom alone is destructive
  (`36.907067382036104`), the paired molecule+inventory/atom replay nearly
  closes (`0.08475871276711094`), and adding exact removed closes to
  `9.230739629452324e-12`.
- Layer `60:-5`: `baseline_ABC` reproduces at
  `0.9142498895939201`. FC molecule alone is destructive
  (`49.72541494970198`), FC inventory/atom alone is destructive
  (`50.419424423672645`), and the paired molecule+inventory/atom replay closes
  to `1.3724132941206335e-11`; removed is neutral.
- The gate passes without falling back to a full-vector norm.

I/J discrepancy reconciliation:

- The post-complementarity closing object is the exact emitted FC molecule RHS
  term paired with gauge-normalized inventory/atom, plus exact removed for layer
  45.
- The catastrophic exact-subspace compact `I/J` path is the FC cached
  input-vector molecule reconstruction path, not the exact emitted FC molecule
  RHS term. It remains catastrophic in the tail context as variant `I`
  (`1.714610488868347e45` at layer 45 and `3.835950895723594e55` at layer 60).
- The exact emitted direct FC molecule cache path closes in the same tail context
  as variant `J`/`K`, so the discrepancy is classified as
  `source_artifact_mismatch`, not as a selected-row mapping, matrix trace, or row
  scaling failure.

Fixed-row subspace replay inside the tail context:

- S0-S5 fixed-row subspace molecule sources paired with FC inventory/atom remain
  indistinguishable from the KL-current molecule source at the selected-row
  scale.
- Layer `45:-10`: KL-current and S0/S3 are `36.907067382036104`; S1/S2/S4/S5
  are `36.90706739639015`.
- Layer `60:-5`: KL-current is `50.419424423672645`; S0/S3 are
  `50.41942442367235`; S1/S2/S4/S5 are `50.41941899584381`.
- Exact direct FC molecule+inventory/atom still closes the molecule/inventory
  cancellation, and exact removed remains the layer-45 tail.

Promotion result:

- Exact matrix trace gate pass remains valid.
- Known tail ladder reconciliation is source-consistent.
- Fixed-row subspace molecule sources do not recover the molecule/inventory
  cancellation and are not diagnostic candidates for promotion.
- No fixed-row subspace, molecule-cache reconstruction, inventory/atom,
  removed-tail, guarded-mode, production solver, or row-scaling rule is
  promotable.
- RGIE/PIPM implication: carry the exact emitted FC molecule RHS term,
  gauge-normalized inventory/atom pair, and layer-45 removed tail as a
  diagnostic provenance bundle only. Do not substitute fixed-row subspace
  molecule sources for the proven molecule/inventory cancellation.

Decision: fixed-row subspace remains indistinguishable from KL current molecule source.

## Molecule RHS Artifact Reconciliation

The compact molecule RHS artifact reconciliation audit
`results/actual_fastchem_gas_phase_transplant_phase2_molecule_rhs_artifact_reconciliation_compact.json`
keeps the repaired exact matrix trace and the proven tail context fixed, then
compares the exact emitted FastChem molecule RHS term against molecule burdens
reconstructed from cached full-element vectors and molecule caches.

Artifact identity:

- The exact emitted FastChem molecule term is
  `element_rhs_terms.molecule_burden` in scaled solve space, sourced from
  `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleRightHandSide`.
- The traceable FastChem molecule density cache is sourced from
  `CondPhaseSolver::assembleJacobian` at `iter1_full_reduced_system`, with `495`
  molecule records in each focused layer.
- The enriched import-repair trace carries per-molecule mass-action/logK,
  density-gauge, stoichiometry, and full-element-vector provenance. The older
  RHS-term fresh artifact has the emitted RHS term but not all reconstruction
  source fields, so the reconciliation uses both source artifacts explicitly.

Direct term versus reconstructed burden:

- Direct FastChem scaled reconstruction from cached full-element vector `G`
  matches the emitted RHS term to roundoff:
  - layer `45:-10`: max element-RHS residual `1.5432100042289676e-14`
  - layer `60:-5`: max element-RHS residual `2.1316282072803006e-14`
- Reconstruction from the post-`correctValues` vector `E` gives the same
  roundoff match.
- Reconstruction from the direct iter1 RHS-entry molecule cache `F` also matches
  to roundoff:
  - layer `45:-10`: max `1.2212453270876722e-15`
  - layer `60:-5`: max `7.771561172376096e-16`
- The catastrophic previous `I/J` path is reproduced only when the cached-input
  burden is converted through the legacy KL-reference burden-ratio convention:
  - layer `45:-10`: selected-row mean `1.714610488868347e45`
  - layer `60:-5`: selected-row mean `3.835950895723594e55`

Tail replay with explicit molecule artifacts:

- Layer `45:-10`: exact emitted term, direct reconstruction from `G`, direct
  reconstruction from `E`, direct reconstruction from `F`, and direct FC
  molecule cache all give the same near-closure with inventory/atom
  (`0.084758712767...`); adding exact removed closes to
  `9.230739629452324e-12`.
- Layer `60:-5`: the same artifact family closes with inventory/atom at
  roundoff scale (`3.148e-10` in this direct selected-row solve path), and full
  FC coherent RHS is exactly zero.
- Therefore the molecule source artifact itself is now reconciled. The remaining
  discrepancy is not molecule density, stoichiometry, sign, row scaling, or row
  mapping; it is the source-space versus solve-space conversion used by the
  legacy cached-input replay.

S0-S5 implication:

- S0-S5 did not help because they modify the cached-input/reconstructed molecule
  source path. The closing object in the proven tail context is the emitted
  solve-space RHS molecule_burden term, or an exactly equivalent direct
  FastChem-scaled reconstruction of it.
- No fixed-row subspace, molecule RHS, cached-input reconstruction, hvector,
  density-gauge, inventory/atom, removed-tail, row-scaling, guarded-mode, or
  production solver rule is promotable.

Decision: source_artifact_mismatch_due_to_source_space_vs_solve_space.

## Convention-Safe Fixed-Row Subspace Molecule Replay

The convention-safe subspace replay compact
`results/actual_fastchem_gas_phase_transplant_phase2_convention_safe_subspace_molecule_replay_compact.json`
reruns the fixed-row S0-S5 molecule-source replay using the same FastChem scaled
solve-space convention as the exact emitted `element_rhs_terms.molecule_burden`
term. The legacy KL-reference burden-ratio path is retained only as a negative
control.

Builder identity:

- Diagnostic helper formula:
  `scaled_rhs_element[j] = -sum_m stoich[j,m] * n_molecule[m] / FastChem_row_scaling[j]`.
- The helper uses FastChem reduced element row labels, FastChem row scaling, and
  the selected-row mapping from `raw_result_provenance_compact`; it does not use
  any KL-reference burden ratio.
- Identity gate passes:
  - layer `45:-10`: `G`/`E` max residual `1.5432100042289676e-14`, `F` max
    residual `1.2212453270876722e-15`
  - layer `60:-5`: `G`/`E` max residual `2.1316282072803006e-14`, `F` max
    residual `7.771561172376096e-16`

Negative control:

- The legacy KL-reference burden-ratio path remains catastrophic:
  - layer `45:-10`: selected-row mean `1.714610488868347e45`
  - layer `60:-5`: selected-row mean `3.835950895723594e55`
- This confirms the earlier source-space versus solve-space convention mismatch
  and is not used for S0-S5.

Convention-safe S0-S5 replay:

- Layer `45:-10`: KL-current molecule replay is `36.907064016566785`; S0/S3 are
  `36.9070673431478`; S1/S2/S4/S5 are `36.90706738135855`. Full FC cached-input
  reconstruction gives `0.08475871276713746`, direct FC molecule gives
  `0.08475871276711094`, and direct FC molecule plus exact removed closes at
  `9.230739629452324e-12`.
- Layer `60:-5`: KL-current molecule replay is `50.41942391450421`; S0/S3 are
  `50.41942441776398`; S1/S2/S4/S5 are `50.41942442367564`. Full FC cached-input
  reconstruction gives `3.14866355211052e-10`, and direct FC molecule gives
  `3.1480369826071677e-10`.
- After fixing the convention, the cached-input FC path is no longer
  catastrophic, but fixed-row subspace molecule sources remain indistinguishable
  from the KL-current molecule source and do not recover the molecule/inventory
  cancellation.

Promotion result:

- The FastChem-scaled molecule RHS builder is source-proven and diagnostic-only.
- S0-S5 do not become diagnostic candidates after the convention fix.
- No fixed-row subspace, molecule RHS, cached-input reconstruction, row-scaling,
  inventory/atom, removed-tail, hvector, density-gauge, guarded-mode, or
  production solver rule is promotable.

Decision: fixed-row subspace remains indistinguishable from KL current after convention fix.

## Full-Element Subset Molecule RHS Attribution

The full-element subset compact
`results/actual_fastchem_gas_phase_transplant_phase2_full_element_subset_molecule_rhs_compact.json`
keeps the convention-safe molecule RHS builder fixed and asks which part of the
FastChem cached full-element input vector is needed to reproduce the exact
FastChem molecule RHS term. This pass does not continue S0-S5 decomposition and
does not use the legacy KL-reference burden-ratio conversion.

Identity gate:

- The convention-safe builder still reconstructs the exact emitted FastChem
  molecule RHS term from the full FC cached input vector to roundoff:
  - layer `45:-10`: max `1.5432100042289676e-14`, mean
    `4.358913594348756e-16`
  - layer `60:-5`: max `2.1316282072803006e-14`, mean
    `7.02777899153347e-16`
- The selected-row mapping residual is zero in both layers; the sign convention
  remains negative stoichiometric molecule burden divided by FastChem row
  scaling.

Subset replay:

- Layer `45:-10`:
  - KL current molecule RHS: `36.907064016566785`
  - FC fixed-only with KL electron: catastrophic
    (`25211452426707.906`)
  - FC non-fixed-only: unchanged from KL current
  - FC electron-only: `36.907067382036104`
  - FC fixed+electron: `0.08475871276713746`
  - FC fixed+electron plus exact removed: `9.230739629452324e-12`
- Layer `60:-5`:
  - KL current molecule RHS: `50.41942391450421`
  - FC fixed-only with KL electron: catastrophic
    (`5273914743321.05`)
  - FC non-fixed-only: unchanged from KL current
  - FC electron-only: `50.419424423672645`
  - FC fixed+electron: `3.14866355211052e-10`
- Top-k RHS-residual and top-k log-residual hybrids for `k <= 15` do not recover
  the cancellation. The smallest audited subset that improves by at least 50%
  and reaches within 10x of the full FC cached-input residual is fixed elements
  plus electron: `23` elements at layer 45 and `24` elements at layer 60.

Interpretation:

- Electron is an essential secondary term because fixed-only is catastrophic and
  electron-only is insufficient, while fixed+electron recovers the full FC
  cached-input molecule RHS behavior.
- Fixed+electron is still not a production rule: layer 45 still needs the exact
  removed tail, and the subset is broad, source-coherent cached-vector
  provenance rather than a small KL solver rule.
- No fixed-only, non-fixed-only, electron-only, top-k, full cached-vector,
  molecule RHS, inventory/atom, removed-tail, row-scaling, guarded-mode, or
  production solver rule is promotable.

Decision: electron is essential secondary term but not sufficient.

## Fixed+Electron Molecule Group Attribution

The molecule-group compact
`results/actual_fastchem_gas_phase_transplant_phase2_fixed_electron_molecule_group_compact.json`
keeps the same convention-safe solve-space builder and partitions the FastChem
molecule cache into neutral, positive, negative, charged, and electron-related
groups. This pass is diagnostic-only, uses only layers `45:-10` and `60:-5`,
does not continue S0-S5, and does not use the legacy KL-reference burden-ratio
conversion.

Identity gate:

- The convention-safe builder again reproduces the exact emitted FastChem
  molecule RHS term from the full FC cached input vector:
  - layer `45:-10`: max `1.5432100042289676e-14`, mean
    `4.358913594348756e-16`
  - layer `60:-5`: max `2.1316282072803006e-14`, mean
    `7.02777899153347e-16`
- The selected-row mapping residual is zero on both layers. Row scaling is the
  FastChem row-scaling vector projected to common FC row positions; the sign is
  negative stoichiometric molecule burden divided by row scaling.

Group result:

- The FC molecule set splits into `380` neutral, `65` positive, and `50`
  negative molecules on both focused layers.
- Neutral molecule replay with FC fixed elements recovers the selected-row
  molecule/inventory behavior: `0.08475871276713746` at `45:-10` before removed
  and `3.14866355211052e-10` at `60:-5`.
- Charged molecule replay is the electron-sensitive failure mode. FC fixed-only
  with the KL electron is catastrophic through the positive-ion term
  (`25211452426691.668` at `45:-10`, `5273914743307.41` at `60:-5`), while FC
  electron on the fixed subset suppresses that charged contribution back to the
  FastChem group term.
- Electron log blending confirms the sensitivity: moving the electron toward the
  FC value reduces the fixed-only catastrophe sharply, but no KL-native exposed
  electron source equals the FC fixed+electron bundle.
- Layer `45:-10` removed correction pairs with the neutral molecule term on the
  selected-row metric: neutral-only fixed+electron plus exact removed closes,
  while charged-only fixed+electron plus removed remains at the KL-current
  scale.

Support/provenance:

- Small top charged/neutral molecule supports do not recover cancellation. The
  useful subset is still broad fixed-element support plus electron; no small
  physically interpretable subset is promotable.
- KL-native fixed-element and electron stages are carried/recomputed/overwritten
  diagnostics only. They remain tens of log units away from the FC
  fixed+electron bundle and lack `fixed_element_bookkeeping`,
  `degree_of_condensation_or_phi_analogue`, and a same-boundary carried
  fixed+electron materialization field.
- Diagnostic candidate exists only as a broad cached-vector attribution; no
  production solver, guarded mode, molecule, inventory, removed, row-scaling,
  hvector, cached-vector, fixed-element, or electron rule is promotable.

Decision: neutral molecules dominate the fixed+electron recovery.

## KL-Native Fixed+Electron Materialization Boundary

The KL-native materialization compact
`results/actual_fastchem_gas_phase_transplant_phase2_kl_native_fixed_electron_materialization_compact.json`
tests whether any emitted KL fixed/electron source boundary can reproduce the
FastChem fixed+electron molecule RHS bundle. This pass is diagnostic-only,
entrance-smoke only, uses layers `45:-10` and `60:-5`, preserves the selected-row
metric, and does not use the legacy KL-reference burden-ratio conversion.

Identity gate:

- Full FC cached input and the FC fixed+electron bundle both reproduce the exact
  emitted FC molecule RHS term to roundoff:
  - layer `45:-10`: max `1.5432100042289676e-14`, mean
    `4.358913594348756e-16`
  - layer `60:-5`: max `2.1316282072803006e-14`, mean
    `7.02777899153347e-16`
- Selected-row identity residual is zero in both cases. The row-scaling and sign
  conventions are unchanged: FastChem row-scaling vector in the common row basis
  and negative stoichiometric burden divided by row scaling.

KL-native candidates:

- KL current fixed values with KL current/post-`correctValues`/RHS-entry
  electron remain at the KL-current selected-row scale:
  `36.907064016566785` at layer 45 and `50.41942391450421` at layer 60.
- KL gas-recouping fixed plus gas-recouping electron remains non-closing:
  `36.907067382036104` and `50.41942442367563`.
- KL gas-recouping fixed plus FC electron is still non-closing, so fixed values
  remain a blocker even when the electron is supplied from the FC upper bound.
- FC fixed values plus the best KL-native electron candidate are catastrophic:
  `21430726675.379684` and `1221779458.584635`.
- FC fixed values plus FC electron reproduce the known upper bound:
  `0.08475871276713746` at layer 45 before removed and
  `3.14866355211052e-10` at layer 60.
- The reduced-Newton overwrite-derived fixed bundle candidates are unavailable
  because the compact trace does not emit
  `KL.reduced_Newton_overwrite_derived_fixed_full_element_vector` from
  `_assemble_fastchem_reduced_update / reduced_update_diagnostics`.

Materialization boundaries:

- Best KL-native electron is gas-recouping, but it is still far from the FC
  electron: log residual `59.593429776301384` at layer 45 and
  `74.8759964865863` at layer 60. It does not reach within 10x of the FC
  fixed+electron residual.
- KL fixed candidates are tens of log units away from the FC fixed bundle:
  current/post/RHS-entry fixed mean residuals are `43.03152208289549` and
  `41.28868465004004`; gas-recouping fixed mean residuals are
  `44.10696218661984` and `41.39784705435456`.
- KL fixed candidates still lack fixed-element bookkeeping and
  `degree_of_condensation` / `phi` analogues at the materialization boundary.
- Layer `45:-10` exact removed pairs only with the FC fixed+electron / exact
  emitted FC molecule path, closing to about `9e-12`; no KL-native candidate
  reaches that pairing frontier.

Decision: FC fixed+electron bundle is required; no KL-native materialization candidate closes.

## KL Reduced-Newton Fixed+Electron Bundle

The reduced-Newton fixed+electron compact
`results/actual_fastchem_gas_phase_transplant_phase2_kl_reduced_newton_fixed_electron_bundle_compact.json`
materializes the previously missing KL reduced-Newton overwrite-derived fixed
full-element vector from the diagnostic result-slot compact. It uses the
existing `CondensedPhase::correctValues` write-site chain fields:
`old_element_density`, reduced Newton result slot, row scaling, global scaling,
`delta_log_n_fixed`, clipped delta, update factor, and
`final_elem_number_dens_new`.

Availability:

- The vector is available for all fixed elements in the focused layers:
  `22` rows at `45:-10` and `23` rows at `60:-5`.
- The materialized values are diagnostic-only and are not consumed by molecule
  reconstruction in the KL run.
- Exact same-iteration reduced-system row/column labels remain missing from
  `CondPhaseSolver::newtonStep`, but the slot index, scaling, result value, and
  write-site chain are present.

Candidate replay:

- KL reduced-Newton fixed + KL current/post/RHS-entry electron remains
  catastrophic:
  `25211452426707.906` at layer 45 and `5273914743321.05` at layer 60.
- KL reduced-Newton fixed + gas-recouping electron improves relative to the KL
  electron catastrophe but remains non-closing:
  `21430726675.379684` and `1221779458.584635`.
- KL reduced-Newton fixed + FC electron reproduces the FC fixed+electron upper
  bound:
  `0.08475871276713746` at layer 45 before removed and
  `3.14866355211052e-10` at layer 60.
- Layer `45:-10` exact removed then closes the reduced-fixed + FC-electron path
  to `9.001471173380398e-12`.

Boundary result:

- Same-boundary KL reduced fixed + post-`correctValues` electron is still
  catastrophic because the post-`correctValues` electron equals the KL-current
  electron value.
- Mixed-boundary KL reduced fixed + FC electron closes the fixed side, proving
  the remaining KL-native blocker is electron materialization.
- FC fixed + gas-recouping electron remains catastrophic, so the best KL-native
  electron is still insufficient even when fixed values are FC-quality.

Decision: KL-native fixed materialization works but electron source remains blocker.

## Electron Materialization Provenance

The electron materialization compact
`results/actual_fastchem_gas_phase_transplant_phase2_electron_materialization_provenance_compact.json`
keeps the reduced-Newton fixed vector fixed and audits only electron source
provenance.

Source table result:

- The FC cached electron in the fixed+electron bundle is
  `9.284083575104927e-13` at layer `45:-10` and
  `3.4061311979326554e-29` at layer `60:-5`.
- The FC cached electron is emitted at
  `iter0_post_correctValues_full_element_vector_before_molecule_refresh` from
  `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`
  and is consumed by molecule reconstruction.
- The focused trace does not emit the prior FastChem gas-only, activity-scan,
  reset, or gas-solver charge-neutrality electron stage values for this cached
  electron. Those fields are recorded as missing rather than inferred.
- On KL, the best native electron remains the gas-recouping / post-adoption gas
  recompute electron, with log residuals `59.593429776301384` and
  `74.8759964865863` from the FC electron. It improves the charged-ion
  catastrophe but remains non-closing.

Equation provenance:

- FastChem source contains a special electron solver path in
  `fastchem/fastchem_src/gas_phase/calc_electron_densities.cpp`, including the
  singly-ion analytic branch (`alpha`, `beta`) and the multi-ion charge
  conservation / Newton branch (`positive_ion_density`,
  `negative_ion_density`, `electron_density`, `delta`,
  `newtonSolElectron`).
- The focused diagnostic trace does not emit the branch choice or those local
  variables for the cached FC electron. Therefore no source-proven
  FastChem-style charge-neutrality reconstruction candidate is available.
- No scalar electron gauge-offset field is emitted, so an electron gauge/basis
  repair is not source-proven.

Replay result:

- KL reduced fixed + KL current/post/RHS-entry electron remains catastrophic:
  `25211452426707.906` and `5273914743321.05`.
- KL reduced fixed + gas-recouping or post-adoption electron improves but does
  not close: `21430726675.379684` and `1221779458.584635`.
- KL reduced fixed + FC electron remains the only closing electron replay
  candidate, reproducing the upper bound and pairing with exact removed on
  layer `45:-10`.

Decision: no KL-native electron candidate reaches the FastChem electron; FC electron remains required.

## FastChem Electron Solver Trace

The FastChem electron-solver trace compact
`results/actual_fastchem_gas_phase_transplant_phase2_fastchem_electron_solver_trace_compact.json`
adds diagnostic-only source records from
`fastchem/fastchem_src/gas_phase/calc_electron_densities.cpp`.

Trace availability:

- Layer `45:-10` emits `32` electron solver trace records. The cached electron
  matches call order `24`.
- Layer `60:-5` emits `19` electron solver trace records. The cached electron
  matches call order `43`.
- Both matching calls use `GasPhase::calculateSinglyIonElectrons` with branch
  `singly_ion_analytic`; no electron Newton-history records are expected for
  the matching branch.

Source equation:

- Layer `45:-10`: `alpha = 5.442750603286076e-15`,
  `beta = 6314520127.835389`, and
  `sqrt(alpha / (1 + beta)) = 9.284083575104927e-13`.
- Layer `60:-5`: `alpha = 3.2674752491051913e-40`,
  `beta = 2.8163690441222026e17`, and
  `sqrt(alpha / (1 + beta)) = 3.4061311979326554e-29`.
- The reconstructed logs match the cached FC electron exactly in the compact:
  `-27.70530471849309` and `-65.5493905956773`.

Lineage:

- The FC gas electron solver output is carried into
  `iter0_post_correctValues_full_element_vector_before_molecule_refresh` with
  `value_source_mode = carry_forward_full_element_value`.
- The electron row is not present in `elements_cond`, is not overwritten by
  `correctValues`, and is consumed by molecule reconstruction.
- KL gas-recouping / post-adoption remains the best KL-native electron, but it
  is mixed-boundary relative to the KL reduced fixed vector and remains far from
  the FC electron (`59.593429776301384` / `74.8759964865863` log units).

Decision: electron boundary/carry mismatch is the remaining blocker.

## KL FastChem-Style Electron Reconstruction

The KL-side FastChem-style electron reconstruction compact applies the
source-proven singly-ion equation to KL candidate source states using the
emitted FastChem ion list and coefficient definitions. The FastChem identity
gate closes exactly on both focused layers: `sqrt(alpha / (1 + beta))`
reproduces the cached electron with zero log residual and the multi-ion branch
is not used.

KL current/post-`correctValues` and gas-recouping/post-adoption alpha/beta
reconstructions do not approach the FC electron. When paired with the KL
reduced fixed vector, the reconstructed KL electrons do not close the selected
row replay; only FC electron or FC non-fixed diagnostic upper-bound input
reaches the known fixed+electron residual scale. The log-electron decomposition
attributes the dominant KL-vs-FC mismatch to the beta / ion-correction term on
both layers.

Reduced-fixed plus current/gas non-fixed alpha/beta reconstruction is recorded
as unavailable where the candidate vector lacks required ion-support log fields
for global element indices `2`, `13`, and `19`. No same-boundary KL-native
reconstruction reaches within `10x` of the FC fixed+electron residual.

Decision: beta / ion-correction source mismatch is the remaining blocker.

## Beta Ion-Correction Attribution

The beta ion-correction attribution compact
`results/actual_fastchem_gas_phase_transplant_phase2_beta_ion_correction_attribution_compact.json`
expands the FastChem negative-ion side without changing production behavior.
The FastChem beta identity gate passes on both focused layers with `50`
beta-side negative-ion entries. Layer `45:-10` has
`alpha = 5.442750603286076e-15`, `beta = 6314520127.835389`,
`sum beta_i = 6314520127.835388`, and zero electron log residual. Layer
`60:-5` has `alpha = 3.2674752491051913e-40`,
`beta = 2.8163690441222026e17`, `sum beta_i = 2.8163690441221955e17`,
and zero electron log residual.

Beta residual attribution is broad but has large leading contributors. The top
log-beta residual rows include `F6S1-`, `Al1F4-`, `F5S1-`, and layer-specific
`Ni1-` / `Cu1-` or `C2-` / `F4S1-` rows. The alpha-beta swap replay separates
the source: `KL alpha + FC beta` reaches the known diagnostic FC
fixed+electron residual scale, while `FC alpha + KL beta` remains far from FC.
Top-k FC beta swaps improve only by borrowing FC beta support and are not a
KL-native closure.

Support-element hybrids confirm the same boundary. KL reduced fixed plus FC
non-fixed reduces beta/electron residuals as a diagnostic upper bound, and full
FC cached input closes beta, but no KL-native beta/electron candidate closes.
Same-boundary reduced-fixed alpha/beta reconstruction still lacks non-fixed
support logs for global element indices `2`, `13`, and `19` (`Ar`, `He`,
`Ne`) from the diagnostic KL-native fixed+electron bundle overlay.

Guardrail: no production electron rule, beta correction, guarded mode, or
support-element hybrid is promotable. The beta source is a diagnostic upper
bound only.

Decision: FC non-fixed source state is required; no KL-native beta candidate closes.

## Beta Same-Boundary Vector Repair

The same-boundary beta repair compact
`results/actual_fastchem_gas_phase_transplant_phase2_beta_same_boundary_vector_repair_compact.json`
fixes the diagnostic full-vector construction bug without changing solver
behavior. The prior missing `Ar`, `He`, and `Ne` fields were caused by mixing
KL element row order with FastChem global stoichiometry indices. The repaired
constructor canonicalizes every candidate onto the FastChem full-element order
by element label, so candidates `A` through `G` all emit `28` element logs and
`Ar`, `He`, and `Ne` are present and finite on both focused layers.

The FastChem beta identity still closes. The repaired KL reduced-fixed plus KL
current/gas/post-adoption non-fixed candidates are now constructible and no
longer show the previous catastrophic index-order beta. They all reconstruct
the same alpha/beta as the KL reduced-fixed plus FC non-fixed diagnostic upper
bound:

- Layer `45:-10`: alpha `4.4758533510576745e-15`, beta
  `3463404901.445013`, alpha log residual `0.19558753216534797`, beta
  log1p residual `0.6005995827798536`, electron log residual
  `0.2025060253072546`.
- Layer `60:-5`: alpha `2.6343586377808056e-40`, beta
  `1.1735995994752506e17`, alpha log residual `0.21537784047065145`, beta
  log1p residual `0.8753728757415544`, electron log residual
  `0.32999751763544793`.

Full FC cached input still closes beta to roundoff, but replacing `Ar`, `He`,
`Ne`, requested chemically named non-fixed subsets, beta-supporting non-fixed
elements, or all non-fixed elements does not move the repaired beta residual.
The remaining beta residual is therefore not attributable to the repaired
missing inert support rows or to the requested non-fixed ablations. It remains
diagnostic-only and not a production electron rule.

Decision: beta attribution remains mixed or inconclusive.

## Repaired Alpha/Beta Tail Replay

The repaired alpha/beta tail replay compact
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_tail_replay_compact.json`
keeps the canonical FastChem index-order gate in front of the replay. The gate
passes with `e-` at global index `0`, all `28` element logs finite, and
zero residual against the expected FastChem canonical mapping.

The repaired same-boundary candidate uses KL reduced-Newton overwrite-derived
fixed values, KL same-boundary non-fixed values, and a FastChem-style
alpha/beta electron reconstructed from that same canonical vector. It gives:

- Layer `45:-10`: beta log1p residual `0.6005995827798536`, electron log
  residual `0.2025060253072546`.
- Layer `60:-5`: beta log1p residual `0.8753728757415544`, electron log
  residual `0.32999751763544793`.

Despite the remaining exact beta residual, the convention-safe molecule RHS
from the repaired candidate matches the exact emitted FC molecule RHS on the
selected rows. In the common post-complementarity tail:

- Baseline `ABC` remains `0.5620237706782278` / `0.9142498895939201`.
- Repaired candidate molecule RHS plus gauge-normalized inventory/atom gives
  `0.08475871276713746` / `1.3947630928999967e-11`.
- Adding exact removed closes layer `45:-10` to `9.001471173380398e-12`.

The coherent FC-Jacobian raw-result replay shows the same cancellation
structure: molecule-only is destructive, while repaired molecule plus
inventory/atom recovers the selected-row cancellation. The remaining beta
residual is therefore nonblocking for selected-row closure but remains a
provenance residual. This is diagnostic-only; no production electron/beta rule
or guarded mode is promotable, and the layer-45 removed tail remains separate.

Decision: repaired same-boundary KL alpha/beta electron recovers molecule/inventory cancellation.

## Repaired Alpha/Beta Integrated Ladder

The integrated ladder compact
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_integrated_ladder_compact.json`
keeps the repaired same-boundary alpha/beta candidate inside the full
diagnostic RHS semantic ladder. It remains entrance-smoke-only, selected-row
metric only, and diagnostic-only.

The canonical index-order gate passes for every vector consumed by
FastChem-style alpha/beta stoichiometry: vectors are in FastChem canonical
order with `e-` at index `0`, all `28` element logs are finite, and the mapping
from the KL source vectors is recorded by element label. The raw KL source
vectors are explicitly logged as ExoGibbs-order inputs with `e-` last before
conversion; they are not used directly for alpha/beta indexing.

The repaired candidate reconstruction is unchanged:

- Layer `45:-10`: alpha `4.4758533510576745e-15`, beta
  `3463404901.445013`, electron `1.136805826048949e-12`, beta log1p
  residual `0.6005995827798536`, electron log residual
  `0.2025060253072546`.
- Layer `60:-5`: alpha `2.6343586377808056e-40`, beta
  `1.1735995994752506e17`, electron `4.7378081767088225e-29`, beta
  log1p residual `0.8753728757415544`, electron log residual
  `0.32999751763544793`.

The full ladder preserves the selected-row metric. `baseline_AB` gives
`3.0603000036721433` / `2.5285823085810524`; `baseline_ABC` gives
`0.5620237706782278` / `0.9142498895939201`. Repaired molecule-only remains
destructive (`36.612411845528754` / `49.72541494970158`), and inventory/atom
only remains destructive (`36.907067382036104` / `50.419424423672645`). The
repaired molecule plus gauge-normalized inventory/atom cancellation gives
`0.08475871276713746` / `1.3947630928999967e-11`; exact removed then closes
layer `45:-10` to `9.001471173380398e-12`. Full FC coherent RHS closes to
`0.0` on both layers.

In the labelled reduced-system solve context beyond the common
post-complementarity tail, molecule plus inventory alone does not close
(`2.9375647869000923` / `1.9877020617705434`). The full tail bundle including
tau/complementarity closes to `9.001471173380398e-12` /
`1.3947630928999967e-11`. The exact beta residual does not affect the raw
solve context at selected rows, but it remains full-vector relevant: the full
element RHS l2 differences are `1.6836286868830854` and
`2.2365923396683853`.

The positional-boundary regression audit passes for alpha/beta and molecule
RHS reconstruction: `23` occurrences were classified, `4` remain unknown in
legacy/static contexts, and none of the unknowns touch alpha/beta or molecule
RHS reconstruction.

Decision: exact beta mismatch is nonblocking for selected-row closure but remains full-vector provenance residual.

## Positional Boundary Unknown Resolution

The positional-boundary compact
`results/actual_fastchem_gas_phase_transplant_phase2_positional_boundary_assumptions_compact.json`
now resolves the four previously unknown matrix diagonal reads from:

- `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2_fixed_row_result_entry_provenance_compact.py`
- `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.py`

The basis is not global ExoGibbs element order and not FastChem canonical
full-element order. The matrices are
`element_element_jacobian_subterms` local element-label order: the local
position is built from `element_labels.index(element)`. The ambiguous helper
argument has been renamed to `subterm_element_pos`, and each diagonal is read
by expected element label rather than by a direct `element_index`. The
diagnostic output records basis, expected element, element-label position,
label at that position, global element index when available, matrix dimensions,
and a per-subterm `basis_guard.safe` flag.

The regenerated positional audit reports `19` classified occurrences and `0`
unknown occurrences. The repaired alpha/beta canonical gate remains unchanged
and safe.

Decision: all positional-boundary unknowns resolved as safe element-label-local subterm indexing.

## Repaired Alpha/Beta Coherent Bundle

The coherent bundle compact
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_coherent_bundle_compact.json`
evaluates the repaired same-boundary KL alpha/beta molecule RHS candidate as
part of the coherent diagnostic RHS source-state bundle. The canonical vector
and source-boundary gate passes on layers `45:-10` and `60:-5`: all 28
elements are finite, FastChem canonical order is used with `e-` at index `0`,
and no legacy KL-reference burden-ratio conversion is used.

The repaired candidate reconstruction is unchanged. Layer `45:-10` has alpha
`4.4758533510576745e-15`, beta `3463404901.445013`, electron
`1.136805826048949e-12`, beta log1p residual `0.6005995827798536`, and
electron log residual `0.2025060253072546`. Layer `60:-5` has alpha
`2.6343586377808056e-40`, beta `1.1735995994752506e17`, electron
`4.7378081767088225e-29`, beta log1p residual `0.8753728757415544`, and
electron log residual `0.32999751763544793`.

The coherent ladder confirms that the repaired molecule RHS is not useful as an
isolated swap, but it is coherent with the inventory/atom and tail bundle:

- Layer `45:-10`: `baseline_AB=3.0603000036721433`,
  `baseline_ABC=0.5620237706782278`, repaired molecule-only
  `36.612411845528754`, inventory/atom-only `36.907067382036104`, repaired
  molecule plus inventory/atom `0.08475871276713746`, and exact removed closes
  to `9.001471173380398e-12`.
- Layer `60:-5`: `baseline_AB=2.5285823085810524`,
  `baseline_ABC=0.9142498895939201`, repaired molecule-only
  `49.72541494970158`, inventory/atom-only `50.419424423672645`, and repaired
  molecule plus inventory/atom closes to `1.3947630928999967e-11`.

In the labelled raw solve context, the full tail bundle with
removed/tau/complementarity remains nonblocking relative to full FC: raw solve
l2 differences are `1.1301087716753843e-10` and
`2.0481871868101534e-10`. Exact beta is therefore nonblocking for selected-row
and raw-solve closure, but not eliminated as provenance: full element RHS l2
differences are `1.6836286868830854` and `2.2365923396683853`, with `22`
outside-selected rows affected on each focused layer.

Broader smoke generalization was not run because the existing repaired
alpha/beta artifacts expose only the focused layers `45:-10` and `60:-5`.

Decision: exact beta mismatch is nonblocking for selected-row and raw-solve closure but remains full-vector provenance residual.

## Repaired Alpha/Beta Broad Smoke Attempt

The broad-smoke compact
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_smoke_compact.json`
attempts to generalize the repaired same-boundary alpha/beta coherent bundle
beyond the focused layers. The requested compact case set was
`30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`.

The blocking assertion was diagnosed as over-strict, not as a real species
boundary failure. `gas_setup.elements` uses raw element labels (`Al`, `Ar`,
..., `e-`), while `gas_setup.species[:n_elem]` uses formula-like one-atom
species (`Al1`, `Ar1`, ..., `e1-`). Parsing those labels with
`sanitize_formula` / `parse_formula_with_charge` shows each prefix species is
formula-equivalent to one atom of the corresponding ExoGibbs e-last element,
and `formula_matrix_gas[:, :n_elem]` is the identity block. The assertion now
checks formula equivalence plus the identity block instead of literal string
equality.

After the diagnostic assertion repair, the phase2 broad smoke command completed
for the requested case set and emitted `/tmp/exogibbs_phase2_broad_smoke.json`,
`/tmp/exogibbs_phase2_broad_smoke_traces.json`, and
`/tmp/exogibbs_phase2_broad_smoke.md`. The run still prints the JAX CUDA
warning in this environment, but it is nonblocking.

The follow-up dependency graph compact
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_dependency_graph_compact.json`
shows the specialized repaired-alpha/beta replay dependencies are still
focused-only. The raw-result compact depends on a focused selected-row delta
input; the reduced-Newton result-slot compact has `LAYERS=(45, 60)` and only
`--trace-45` / `--trace-60` inputs; the molecule-vector, coherent-bundle, and
electron-reconstruction compacts also emit only focused layers. Therefore the
compact carries forward only available focused cases `45:-10` and `60:-5`.
On those cases the canonical gate, selected-row tail closure, and raw solve
nonblocking checks pass, but outside-selected residuals remain material (`22`
outside rows affected on each focused layer), and no broad repaired-alpha/beta
values are inferred.

Promotion readiness remains negative for the repaired alpha/beta rule: phase2
broad smoke is unblocked, but specialized broad dependencies were not
regenerated. No broad-layer alpha/beta, RHS-tail, raw-solve, or
outside-selected residual is inferred from the multi-GB phase2 artifacts.

Decision: dependency regeneration incomplete; repaired candidate remains focused-only.

## Direct Broad Repaired Alpha/Beta Extraction

The direct broad evaluator
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_direct_broad_eval_compact.json`
reads `/tmp/exogibbs_phase2_broad_smoke.json` directly instead of using the
focused compact dependency chain. It confirms the broad artifact contains
FastChem internal reduced-system records, alpha/beta electron solver traces,
molecule provenance, exact inventory/removed traces, row labels, row scaling,
RHS vectors, Jacobians, and solver result vectors for all five requested cases.

The phase2 driver now has a diagnostic-only
`repaired_alpha_beta_source_state_snapshot` hook for future broad runs, and the
source-snapshot compact summarizes the existing broad artifact. Snapshot
emission is still incomplete for the existing run: the broad artifact does not
emit the same-boundary KL non-fixed vector under the required
`iter1_RHS_assembly_entry` source boundary, and it does not emit the
selected-row delta/raw-result mapping used as the primary repaired replay
metric. Because those fields are absent, no canonical repaired alpha/beta
candidate, molecule+inventory replay, raw solve context, or outside-selected
attribution is inferred for the broad cases.

Decision: repaired source-state snapshot emission incomplete.

## Fresh Broad Source-State Snapshot Rerun

The phase2 driver was rerun on the broad case set `30:-10`, `45:-10`,
`60:-5`, `75:-5`, and `90:-5` with the diagnostic-only
`repaired_alpha_beta_source_state_snapshot` embedded in the case payload. The
fresh artifacts are
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke.json`,
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke_traces.json`,
and
`results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke.md`.

The fresh snapshot now emits
`same_boundary_KL_non_fixed_values` for every requested broad case from the
`iter1_RHS_assembly_entry` source boundary, with ExoGibbs e-last labels and
FastChem canonical e-first converted entries. Canonical mapping inputs,
reduced fixed values, molecule RHS inputs, inventory/atom inputs, exact removed
inputs, and raw labelled reduced-system fields are also present for all five
cases.

The remaining incomplete field is `selected_row_mapping`: the fresh case
payload still does not expose `selected_rows[].row_position`,
`selected_rows[].result_index`, row labels, or delta classifications for the
primary selected-row metric. The direct broad evaluator therefore records the
narrower blocker `repaired source-state snapshot emission incomplete`; no broad
canonical repaired candidate replay, molecule+inventory tail replay, raw-solve
comparison, outside-selected attribution, or production rule is inferred.

## Broad Selected-Row Mapping Emission

The broad source-state snapshot now emits `selected_row_mapping` for all five
requested cases. The mapping uses the same diagnostic focused row definition as
the one-step/raw-result provenance path: `PRESELECTION_ACTIVITY_FOCUSED_NAMES`,
projected through labelled FastChem reduced-system columns plus the emitted
FastChem/KL `solver_result_to_delta_n_cond_mapping` records. Rows outside the
projected shared solve are retained with null result indices and explicit
`mapping_status`, matching the focused raw-result provenance behavior.

Focused validation passes for `45:-10` and `60:-5`: selected row counts,
row-label sets, and FastChem/KL solver-result indices match the existing
focused raw-result provenance compact. The broad source snapshot is therefore
complete for same-boundary non-fixed values and selected-row mapping. The
remaining blocker is no longer a missing broad trace field; it is that the
direct broad compact has not yet implemented the convention-safe
alpha/beta-derived molecule-density replay without the focused compact stack.

Decision: repaired candidate remains mixed or inconclusive.

## Split Diagnostic Frontier: CH4 Floor, Atomic Snapshot, and Result Index

The one-sided selected-row frontier remains split under
`focused_raw_result_provenance_metric`: `12` selected rows, `0` shared numeric
rows, and `12` one-sided rows. Track A has `7` activity-threshold rows:
`30:-10:MgCO3(s)`, `30:-10:SiC(s)`, and `CH4(s,l)` at `30:-10`, `45:-10`,
`60:-5`, `75:-5`, and `90:-5`. Track B has `5` result-index rows:
`30:-10:Al(s)`, `30:-10:K3AlF6(s)`, `30:-10:Na3AlF6(s,l)`,
`30:-10:Na5Al3F14(s,l)`, and `45:-10:Al(s)`.

For `CH4(s,l)`, a diagnostic-only FastChem whitebox trace patch now resolves
the `-10` path without changing solver behavior. It emits the raw mass-action
`raw_log_activity_before_floor_clip` before the data-validity floor, while the
stored `displayed_log_activity_after_floor_clip` remains `-10` with
`clipped_or_floored = true` and `data_validity_floor = true` from
`Condensate<double_type>::calcActivity` lines `77-92`. The raw pre-floor values
are `58.169686102954905`, `-4.530329499225015`, `-15.485824491588767`,
`-19.05959499690444`, and `-21.452325720828508` for `30:-10`, `45:-10`,
`60:-5`, `75:-5`, and `90:-5`. The stored `-10` is therefore a
sentinel/display-floor value, not a true computed activity. KL CH4 activity is
positive in all five cases. The CH4 decomposition
keeps phase segment `l`, the logK source record, selected temperature
interval, density-gauge/standard-state corrections, formula row `C + 4H`,
atomic contributions, and final activity in the compact artifact. CH4 is
classified as `clipping_or_sentinel_display_floor_mismatch` with thermo/lnK
reference-state counterfactuals recorded.

For `30:-10:MgCO3(s)` and `30:-10:SiC(s)`, the selected threshold crossing is
classified as atomic gas element density snapshot mismatch. The compact emits
FastChem post-condensation fixed atomic element-species donor terms, KL
gas-only `ln_nk` donor terms, conserved-inventory terms that would be wrong if
used as the donor, hvector/lnK terms, density-gauge terms, formula rows, and
source-state swaps. The per-element post-`correctValues` contribution before
`calcActivity` is not separately emitted and is reported as an exact missing
diagnostic field.

For Track B, KL exact labelled RHS/Jacobian row labels, Jacobian column labels,
row scaling by label, and solver result vector by label remain unavailable.
The closest materialization is
`current_best_upstream_kl_branch.split_history[0].condensates_jac`, which
reconstructs KL labels, positions, and result indices by label, but not exact
labelled arrays or row scaling/result vectors. This is a Python audit trace
availability gap, not a production solver rule.

Repaired alpha/beta remains irrelevant to the current focused blocker because
there are no shared numeric selected rows. It cannot affect absent or
unprojectable one-sided rows. No production electron rule, guarded mode,
selected-row rule, row-scaling rule, lifecycle rule, labelled-system rule, or
solver behavior is promotable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Focused Frontier Closure and Broad Projection Pivot

The current focused one-sided selected-row frontier is diagnostically closed
under the fresh focused metric. The original `12` one-sided rows reduce to
zero unresolved focused rows: CH4 data-validity masking accounts for `5/5`
CH4 rows, FastChem fixed/full-element donor snapshot attribution accounts for
`2/2` MgCO3/SiC rows, and exact KL labelled materialization classifies the
five Group-B rows as `intentionally_excluded_from_reduced_solve`.

This closure is a diagnostic ledger entry only. The CH4 data-validity mask is
not a production thermochemistry rule, the FastChem donor snapshot is not a
production donor transplant, and Group-B intentional exclusion does not change
selected-row or result-index semantics. Repaired alpha/beta remains irrelevant
until shared numeric selected rows appear.

The remaining work is the embedded broad 10-row diagnostic projection and
outside-selected/full-vector residual. That projection keeps
`metric_id=embedded_broad_10row_projection` and
`purpose=outside-selected/full-vector residual probe`; it is not used as a
focused regression. Replaying the focused diagnostic levers annotates the
broad rows, but numeric broad residual closure is not recomputed and the broad
projection remains non-closing. The residual remains dominated by
outside-selected/full-vector source state, with neutral molecule full-vector
provenance still material in the direct broad compact.

Decision: broad projection residual remains dominated by outside-selected/full-vector source state.

## Integrated Split-Frontier Counterfactual

The integrated counterfactual remains diagnostic-only. It applies two
source-proven diagnostic transforms to the current `12` one-sided selected rows
without changing production FastChem, KL, presets, thermochemistry defaults,
row selection, active selection, row scaling, lifecycle, labelled-system,
maxDensity, density-gauge bridge, or solver behavior.

First, applying the FastChem CH4 data-validity candidate mask to KL CH4 rows
uses threshold value `-10` whenever `T > 190.6`, matching FastChem's finite
thermo validity branch. This makes all five KL CH4 rows fail the diagnostic
candidate threshold and removes their one-sided membership. Second, applying
the FastChem fixed/full-element donor snapshot to the `30:-10:MgCO3(s)` and
`30:-10:SiC(s)` KL activity rows makes both rows pass, matching the
threshold-crossing donor source state and explaining their one-sided
membership. These are counterfactual source-state diagnostics only, not
promotable production rules.

The remaining five rows are exactly Group B:
`30:-10:Al(s)`, `30:-10:K3AlF6(s)`, `30:-10:Na3AlF6(s,l)`,
`30:-10:Na5Al3F14(s,l)`, and `45:-10:Al(s)`. They are held out as
result-index mapping unresolved because exact KL labelled RHS/Jacobian
row/column arrays, row scaling by label, and solver result vector by label are
still unavailable. The current focused blocker therefore reduces from `12`
one-sided rows to `5` Group-B result-index mapping rows.

Decision: integrated counterfactual reduces blocker to Group-B result-index mapping.

## Group-B Result-Index Exact Materialization

The Group-B materialization audit keeps all five rows as one-sided selected
rows and does not reinterpret them as shared numeric rows. FastChem exact
labelled RHS rows, Jacobian rows, Jacobian columns, row positions, column
positions, row scaling by label, solver result slots, and result indices are
available for the `30:-10` and `45:-10` comparison cases from
`CondPhaseSolver::newtonStep` / `iter1_full_reduced_system_records`.

KL still does not emit exact labelled RHS row labels, Jacobian row labels,
Jacobian column labels, row scaling by label, or solver result vector by label.
The closest current materialization remains
`current_best_upstream_kl_branch.split_history[0].condensates_jac`, which
reconstructs labels and positions but is not the exact labelled reduced-system
array set. The required patch is a Python audit trace/materialization patch;
no C++ trace patch is indicated by the current artifacts.

The exact no-result-index cause for all five Group-B rows is therefore
`missing_KL_exact_labelled_arrays`. None of the rows becomes shared numeric,
none is proven intentionally excluded from the reduced solve, and none is
proven to be a compact mapping artifact. The final remaining blocker count is
`5`.

Decision: Group-B result-index blocker remains blocked by missing KL exact labelled arrays.

## Group-B Exact KL Array Wiring

Static plumbing found the exact KL reduced-system arrays already present in the
broad snapshot at
`cases[].modes.actual_true_kl_atomic_branch_exact_second_post_seed_update_proven.stage_aligned_diagnostics.newly_active_lifecycle_diagnostics.reduced_update_diagnostics.full_reduced_system_trace`.
That record emits exact KL row labels, column labels, row positions, column
positions, row scaling, solver result vector, solver-result mapping,
`condensates_jac_indices`, and `condensates_rem_indices` for all five broad
cases. No C++ patch was required.

After wiring those arrays into the Group-B compact, the exact no-result-index
cause for all five rows is `absent_before_candidate_selection`. They do not
become shared numeric rows, are not promoted as an intentional reduced-solve
exclusion, and are not silently dropped. The missing-KL-array blocker is
removed; the final remaining blocker count for the integrated diagnostic
frontier is `0` under the current focused one-sided blocker accounting.

Decision: Group-B result-index blocker reduces after exact labelled materialization.

The final requested taxonomy maps the exact lifecycle root cause
`absent_before_candidate_selection` to
`intentionally_excluded_from_reduced_solve`: the rows have no exact KL result
slot because they never enter the KL candidate/reduced-system path. They are
not shared numeric rows and no result-index semantics were changed.

Decision: Group-B result-index blocker is fully explained as intentional reduced-solve exclusion.

## Split One-Sided Provenance Tracks

The current fresh focused-compatible selected-row metric is now split into two
diagnostic tracks without changing production behavior. All `12` selected rows
remain one-sided and there are still `0` shared numeric rows, so repaired
alpha/beta replay is not a solution to the current focused blocker.

Track A contains the `7` `activity_threshold_crossing_mismatch` rows:
`30:-10:MgCO3(s)`, `30:-10:SiC(s)`, and `CH4(s,l)` at all five broad cases.
The decomposition compact records FastChem/KL log activity, threshold result,
candidate/active membership, density-gauge/source-state terms, and
diagnostic-only counterfactuals where the source terms are available. The
dominant available components are `thermo_lnK_reference_state_mismatch` for
the five `CH4(s,l)` rows and
`atomic_gas_element_density_snapshot_mismatch` for `MgCO3(s)` and `SiC(s)`.
The `CH4(s,l)` audit records the FastChem stored `-10` value as a
data-validity clipping/sentinel or display-floor trace and emits the raw
pre-floor mass-action value separately. The KL value is positive in all five
selected cases, with KL threshold pass and FastChem threshold fail.

Track B contains the `5` `result_index_mapping_mismatch` rows:
`30:-10:Al(s)`, `30:-10:K3AlF6(s)`, `30:-10:Na3AlF6(s,l)`,
`30:-10:Na5Al3F14(s,l)`, and `45:-10:Al(s)`. Label normalization was retried
and does not explain the blocker. The compact classifies all five as `row
present but no result index`. FastChem exact labelled arrays remain available;
KL split-history index materialization is available, but exact KL labelled
RHS/Jacobian row/column arrays, row scaling, and solver result vectors by label
are still not emitted. No labelled reduced-system materialization mismatch is
therefore promoted from this evidence.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Current Focused-Compatible Broad Raw-Result Replay

The direct broad evaluator now consumes
`results/actual_fastchem_gas_phase_transplant_phase2_raw_result_provenance_broad_compact.json`
as the accepted focused-compatible selected-row source. The metric id is
`focused_raw_result_provenance_metric`; the current validation reference is the
fresh focused metric, and the old focused artifact is recorded only as stale
historical context. The embedded broad 10-row projection remains a separate
`embedded_broad_10row_projection` outside-selected/full-vector residual probe
and is not used as focused regression.

Mapping is available and consumed for all broad cases: `30:-10`, `45:-10`,
`60:-5`, `75:-5`, and `90:-5`. The selected-row counts are `7`, `2`, `1`, `1`,
and `1`. Every selected row is one-sided (`missing_on_one_side`, `FC-only`,
`KL-only`, or focused-only), so the direct evaluator reports row-presence and
mapping/index attribution for the full selected-row set but has zero
shared-projected numeric rows. As a result, current fresh focused-compatible
numeric closure is undefined for the full selected-row set. For `45:-10` and
`60:-5`, the stale historical closure values are preserved in the report, but
the current fresh values are `null` because the selected rows are one-sided;
stale closure therefore does not carry as a current validation claim.

The repaired alpha/beta replay still runs diagnostically for all cases and the
broad 10-row projection still fails as an outside-selected/full-vector probe.
No production electron rule, guarded mode, KL-reference burden-ratio
conversion, solver behavior, selected-row rule, row scaling, molecule,
inventory, removed-tail, lifecycle, RHS, KL, FastChem, or preset behavior is
changed.

Decision: current focused-compatible metric is available but not numerically
projectable for one-sided rows; broad generalization remains partially
metric-inconclusive.

## One-Sided Selected-Row Attribution

The one-sided attribution compact traces the current
`focused_raw_result_provenance_metric` rows without reinterpreting them as
numeric shared rows. Across the five broad cases there are `12` selected rows,
`0` shared-projected numeric rows, and `12` one-sided row-universe/mapping
rows. Counts by one-sided type are `mapping_absent=4`, `FastChem_only=2`,
`KL_only=3`, and `label_mismatch=3`.

Focused-layer source evidence is specific for `CH4(s,l)`: at both `45:-10`
and `60:-5`, candidate active-selection provenance shows a FastChem false /
KL true candidate flag split with activity-threshold mismatch, so the earliest
source-backed stage is candidate selection. For broad-only layers, the older
source-stage lineage compacts are focused-only; those rows remain attributed to
raw-result row-universe/mapping evidence unless a regenerated broad lineage
trace is produced.

This makes repaired alpha/beta irrelevant to the current focused blocker. The
candidate can still be replayed diagnostically in the direct broad evaluator,
but the current selected-row blocker is row-universe/mapping provenance, not
same-boundary repaired alpha/beta source-state. The broad 10-row projection
remains separate and non-closing as an outside-selected/full-vector residual
probe.

Decision: repaired alpha/beta is irrelevant to the current focused blocker;
next blocker is row-universe/mapping provenance.

## Deep One-Sided Lineage Attribution

The deep attribution keeps the current focused metric family
`focused_raw_result_provenance_metric` and does not reinterpret one-sided rows
as shared numeric rows. Broad raw/delta provenance provides full-catalog
identity, selected-row raw-result indices, partition labels, correctValues
snapshot presence, and selected-row activity/maxDensity values. Candidate and
active-set source tables remain focused-layer only; broad case-keyed
candidate/active/reset tables are explicitly recorded as missing lineage
inputs with patch sites.

Earliest divergence is now classified row by row: `5` rows are observed first
at `partition_split_mismatch`, `2` focused `CH4(s,l)` rows are
`activity_threshold_crossing_mismatch`, `4` broad focused-only rows are
`missing_trace`, and one `Al(s)` row remains `mixed_or_unresolved`. Label
normalization was attempted for all label/mapping rows and explains zero rows;
index mapping still fails after normalization. The compact therefore records a
mixed row-universe/mapping frontier, not a label-normalization-only or
alpha/beta source-state frontier.

Decision: next blocker is mixed row-universe/mapping provenance.

## Broad Case-Keyed One-Sided Lineage Update

The latest one-sided attribution compact now emits broad case-keyed full row
tables for all five broad cases. It materializes FastChem candidate, active,
post-`selectActiveCondensates` reset, pre-partition, post-partition, and exact
labelled reduced-system rows from `source_state_by_stage` and
`iter1_full_reduced_system_records`. It materializes KL candidate/active and
pre/post partition rows from the preselection census and split history.

The row-by-row split remains `12` selected rows, `0` shared numeric rows, and
`12` one-sided rows. The previous `missing_trace` rows are resolved. Current
earliest divergence counts are `activity_threshold_crossing_mismatch=7` and
`result_index_mapping_mismatch=5`. The broad-only `CH4(s,l)` rows at `30:-10`,
`75:-5`, and `90:-5` now have FastChem log activity `-10`, KL positive log
activity, FastChem threshold fail, and KL threshold pass; they are therefore
value-driven threshold crossings. The former partition rows now prove an
earlier activity-threshold crossing, so partition is not the earliest blocker.

Reduced-system materialization mismatch is not conclusively proven from the
current compacts. FastChem exact labelled arrays and KL index materialization
are available, but exact KL labelled RHS/Jacobian row/column arrays, KL row
scaling, and KL solver-result vectors by label are not emitted as local arrays.
No selected row is promoted to a shared numeric row, and repaired alpha/beta
remains irrelevant to the current focused blocker.

Decision: next blocker is mixed row-universe/mapping provenance.

## Focused One-Step Staleness Gate For Broad Delta Provenance

The broad one-step extractor now proves the previous broad delta validation
failure was caused by a stale focused artifact plus a diagnostic row-choice bug,
not by missing C++ FastChem trace fields. The current broad/fresh-focused
extractor uses the same schema as the focused one-step compact and now prefers
retained `correctValues_rule` records with `raw_solver_result_value` when
multiple FastChem iter1 records exist for the same condensate. The patch site
is diagnostic-only:
`examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_fastchem_iter1_one_step_records`.
No C++ rebuild is required.

Fresh focused one-step extraction for `45:-10` and `60:-5` no longer matches
the old `actual_fastchem_gas_phase_transplant_phase2_one_step_compact_extract`
row universe. The old focused artifact is therefore recorded as stale relative
to the current code path, and it is not silently used as the current validation
reference. Broad delta provenance rebuilt from the repaired broad one-step
extract validates against the fresh focused delta reference for `45:-10` and
`60:-5`, and raw-result broad mapping rows are available for all five broad
cases. However, the historical focused closure values still belong to the old
focused metric and require a current focused rebaseline before they can be used
for broad generalization.

The embedded broad 10-row projection remains a separate outside-selected probe
and is not a focused regression metric. The direct broad replay records the
current stop condition as stale focused artifact mismatch rather than accepting
the old focused closure under a changed metric.

Decision: stale focused artifact mismatch proven; broad generalization requires
current focused rebaseline.

## Broad Focused-Compatible Raw-Result Provenance Metric

The focused raw-result provenance selected-row metric is now documented and
emitted through a broad diagnostic compact. The selection is not a new rule and
does not hard-code layer-specific row labels: it reuses delta-provenance rows
whose `decomposition.classification` is `delta_raw_result_dominated` or
`delta_mapping_or_index_dominated`, carries the condensate row identity and
mapping status, and attaches the labelled FastChem/KL raw-result vector
indices when those focused source rows exist.

The regenerated broad provenance compact validates `45:-10` and `60:-5`
against the older focused raw-result compact: selected row counts, row labels,
FastChem result indices, mapping statuses, and delta classifications match
with classification validation sourced from the delta-provenance compact. The
same focused-compatible mapping cannot be emitted for `30:-10`, `75:-5`, or
`90:-5` because
`results/actual_fastchem_gas_phase_transplant_phase2_delta_provenance_compact.json`
has no selected delta-provenance rows for those cases. The embedded broad
10-row projection remains a separate outside-selected diagnostic scorecard and
is not used as the focused regression metric.

The direct broad replay was rerun with this focused-compatible provenance
compact. Focused 45/60 references still pass under the focused reference metric,
the broad diagnostic projection still does not close, and broad generalization
remains metric-inconclusive because three broad cases lack the source delta rows
needed to construct the focused-compatible selected-row metric. This is
diagnostic-only; no electron, beta, molecule/inventory, selected-row, removed,
guarded-mode, or solver rule is promoted.

Decision: focused-compatible broad mapping cannot be constructed; broad
generalization remains metric-inconclusive.

## Broad One-Step Extract Validation Gate

The broad one-step compact extract has now been generated from the fresh broad
phase2 artifact using the same upstream one-step `correctValues` row-state
schema consumed by the focused delta-provenance compact. The extract covers
`30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`, and preserves the focused
schema keys for row identity, old/new `correctValues` state, raw result,
scaled/clipped delta, cap branch, mapping status, dominant term, and top-row
summaries.

The regenerated broad delta compact now reads this broad one-step extract and
uses the same delta-provenance classification logic as the focused compact.
That is still not accepted as focused-compatible: validation against the old
focused delta compact fails for `45:-10` and `60:-5` on selected row counts,
row labels, classifications, result indices, and mapping status. The broad
one-step extract also reports missing row-level fields, especially FastChem raw
result fields on rows outside the emitted FastChem one-step trace subset.

The stop gate therefore remains active. The broad delta provenance is recorded
as diagnostic but rejected, the raw-result broad compact does not consume it for
focused-compatible mappings, and the repaired alpha/beta broad replay remains
metric-inconclusive. No production rule is promoted.

Decision: broad delta provenance fails focused validation; broad generalization
remains metric-inconclusive.

## Broad Delta-Provenance Candidate Audit

The broad delta-provenance compact now derives candidate condensate delta rows
for all five broad cases from the labelled broad phase2 solver-result mappings.
It applies the same diagnostic decomposition classes used by the focused delta
compact and marks rows selected by the focused raw-result provenance metric only
when the classification is `delta_raw_result_dominated` or
`delta_mapping_or_index_dominated`.

This candidate broad delta source is not accepted as focused-compatible. It
emits selected rows for all five cases, but validation against the old focused
delta provenance fails on `45:-10` and `60:-5`: selected row counts, row labels,
classifications, result indices, and mapping statuses do not match. The broad
snapshot has solver-result mappings and delta values, but not the
`one_step_compact_extract`-style per-row old/new `correctValues` state,
focused-source row membership, dominant-term selection, and top-residual row
context that produced the focused delta provenance rows.

The raw-result provenance broad compact therefore keeps the validated focused
45/60 mapping and withholds focused-compatible mappings for `30:-10`, `75:-5`,
and `90:-5`. The embedded broad 10-row projection remains only a diagnostic
outside-selected probe. No production rule is promoted.

Decision: focused-compatible broad mapping cannot be constructed; broad
generalization remains metric-inconclusive.

## Selected-Row Metric Reconciliation for Direct Broad Replay

The direct broad evaluator now emits selected-row metric lineage for every
scorecard. The focused reference metric is explicitly identified as
`focused_raw_result_provenance_metric`, sourced from
`results/actual_fastchem_gas_phase_transplant_phase2_raw_result_provenance_compact.json`
and the focused repaired tail compact. The embedded broad projection is
separately identified as `embedded_broad_10row_projection`, sourced from the
embedded `repaired_alpha_beta_source_state_snapshot.selected_row_mapping`.
The broad 10-row projection is not used as the focused regression metric.

For `45:-10` and `60:-5`, the focused reference scorecard reproduces the
focused repaired-alpha/beta references exactly: `45:-10` molecule+inventory is
`0.08475871276713746`, exact removed gives
`9.001471173380398e-12`, and `60:-5` molecule+inventory is
`1.3947630928999967e-11`. Recomputing the same focused row set against the
regenerated broad raw fields does not reproduce those values, so that object
is recorded separately as a raw-field/source-artifact comparison, not as the
focused regression. The embedded broad projection also remains non-closing.

Focused-compatible mapping cannot be constructed for `30:-10`, `75:-5`, or
`90:-5` because the focused raw-result provenance compact only contains the
focused layers. The exact missing source is
`raw_result_provenance_compact.layers[].rows[]` for those case keys. Broad
generalization therefore remains metric-inconclusive rather than promoted.

Decision: focused-compatible broad mapping cannot be constructed; broad
generalization remains metric-inconclusive.

## Contracted Formula-Matrix Boundary Guard Audit

The positional-boundary compact now includes a diagnostic-only guard audit for
contracted formula-matrix boundaries in the phase2 FastChem parity scripts. The
scan covers `contract_formula_matrix(...)`, `formula_matrix_gas.shape[0]` used
as `n_elem`, `species[n_elem:]`, `hvector[n_elem:]`, `n_elem + mol_i`,
`element_vector_full[element_mask]`, and `element_names` derived from
`element_mask`.

The regenerated compact reports `24` contracted-boundary occurrences:
`17` are full-element matrix boundaries guarded by the phase2 full element
count assertion, and `7` are contracted-matrix uses that do not slice species
and explicitly propagate `element_mask` / `element_names`. There are `0`
unsafe occurrences and `0` unknown occurrences.

The audit verifies that any species/hvector boundary use requires
`n_elem == len(gas_setup.elements)` or `len(chem.elements)`, while contracted
formula rows remain labelled by `element_mask` and `element_names` when they are
used as contracted element bases. This is a diagnostic regression guard only;
no numerical replay semantics, solver behavior, molecule behavior, electron
behavior, inventory behavior, row scaling, hvector convention, lifecycle, or
production rule changed.

Decision: contracted-matrix boundary assumptions are all guarded.

## Direct Numeric Broad Repaired Alpha/Beta Replay

The direct broad evaluator now implements the diagnostic-only numeric replay
from the completed broad source-state snapshot for all five cases:
`30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`. It constructs the
FastChem-canonical e-first vector from reduced-Newton overwrite-derived fixed
values plus same-boundary KL non-fixed values, reconstructs the electron with
the FastChem alpha/beta formula, rebuilds molecule densities, and uses the
convention-safe solve-space molecule RHS formula
`scaled_rhs_element[j] = -sum_m stoich[j,m] * n_molecule[m] / row_scaling[j]`.
No production electron rule, guarded mode, legacy KL-reference burden-ratio
conversion, or solver behavior changed.

The canonical vector gate passes for all five broad cases and the repaired
candidate molecule RHS matches the full FastChem molecule RHS on the embedded
selected rows. Full-vector molecule RHS residuals remain material outside the
selected projection: the maximum full-element RHS differences are about
`1251.63` at `30:-10`, `0.66057` at `45:-10`, `0.51809` at `60:-5`,
`0.51597` at `75:-5`, and `0.51558` at `90:-5`. The direct compact reports
full-vector l2/mean/max only as secondary metrics and does not use an infinity
norm fallback.

The coherent molecule+inventory tail replay runs for every broad case, but the
focused regression gate does not pass under the embedded broad selected-row
mapping. The broad snapshot selected-row definition has `10` diagnostic rows
per case, with `5`, `8`, `9`, `9`, and `9` projected result rows respectively,
whereas the older focused tail compact used the focused selected-row solve
metric that produced `0.08475871276713746`, `9.001471173380398e-12`, and
`1.3947630928999967e-11`. The direct broad replay therefore records the
differing term as `selected-row mapping` and stops broad generalization
classification instead of comparing non-identical metrics as if they were the
same.

Promotion readiness remains negative. Broad source-state cases are available
and direct numeric replay is implemented, but selected-row closure does not
hold under the embedded broad metric, outside-selected residuals remain
material, and removed-tail handling remains a separate layer-dependent tail.

Decision: repaired candidate remains mixed or inconclusive.

## Split Frontier Display-Floor and Materialization Audit

The split frontier remains diagnostic-only under `focused_raw_result_provenance_metric`: `12` selected rows, `0` shared numeric rows, and `12` one-sided rows. Track A keeps the `7` activity-threshold rows, and Track B keeps the `5` result-index rows. No one-sided row is reinterpreted as shared numeric, and the embedded broad 10-row projection remains a separate non-closing diagnostic rather than a focused regression.

For `CH4(s,l)`, the diagnostic-only FastChem trace from `Condensate<double_type>::calcActivity` now emits the exact display-floor condition flags. All five broad cases have `data_validity_floor=true`, finite raw/stored values, valid species/phase and density/maxDensity flags, and `candidate_absence_display_flag=false`. The threshold input used by `selectActiveCondensates` is the stored `log_activity`, not the raw pre-floor value. At `30:-10`, raw pre-floor `58.169686102954905` would pass, but the data-validity floor stores `-10`, so the stored threshold fails and the display-floor path affects candidate selection. At `45:-10`, `60:-5`, `75:-5`, and `90:-5`, both raw pre-floor and stored threshold fail on the FastChem side. KL CH4 remains threshold-positive in all five cases, and the thermo/lnK side-by-side record remains emitted for source comparison.

For `30:-10:MgCO3(s)` and `30:-10:SiC(s)`, the compact now emits a per-element C/Mg/O/Si stage table. The available earliest divergent stage is the fixed-element/full-element donor vector consumed by FastChem `calcActivity`: FastChem uses post-condensation fixed full-element values, while KL uses gas-only `ln_nk` values. Per-element post-`correctValues` and per-element density-gauge transformed atomic values are still reported as missing diagnostic fields. Counterfactuals show `FC/KL thermo + FC full-element vector` pass and `FC/KL thermo + KL gas-only vector` fail, so the atomic vector is the threshold-crossing component.

For Track B, exact KL labelled RHS row labels, Jacobian row labels, Jacobian column labels, row scaling by label, and solver result vector by label remain unavailable. The compact reports the exact missing Python locals / trace records and patch site, while split-history materialization still reconstructs labels, positions, and result indices. In the reconstructed materialization all five Group B rows are absent before reduced-system assembly and have no result slot, but exact KL arrays are still required before this can be promoted to a final basis claim.

Repaired alpha/beta remains irrelevant unless shared numeric selected rows appear. No production electron rule, guarded mode, solver behavior, preset, RHS, molecule, inventory, removed-tail, selected-row, row-scaling, active-selection, lifecycle, labelled-system, maxDensity, or density-gauge bridge behavior is promotable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Static-Code Split Frontier Sharpening

Static code confirms the CH4 branch: FastChem `Condensate<double_type>::calcMassActionConstant` evaluates `log_K = a1/T + a2*log(T) + a3 + a4*T + a5*T*T`, applies `density_correction = -sigma * log(1.0e6 / (k_B*T))`, and uses `mass_action_constant + sum_i nu_i log(n_i)` as raw activity. `Condensate<double_type>::calcActivity` then enforces `if use_data_validity_limits && temperature > fit_coeff_limits.back()` by tracing the raw value, storing `log_activity = -10`, setting `data_validity_floor=true`, and returning. `selectActiveCondensates` consumes stored `log_activity >= 0`, not the raw pre-floor value.

The CH4 compact now records temperature, selected segment index, finite FastChem data-validity upper, the `temperature > fit_coeff_limits.back()` predicate, raw/stored activity, and stored-threshold candidate decision for all five broad cases. `30:-10:CH4(s,l)` is raw-positive (`58.169686102954905`) but rejected because the data-validity branch writes stored `-10`. The other CH4 cases are raw-negative and also stored at `-10` by the data-validity path. KL uses `compute_kl_condensate_log_activity = formula_cond.T @ u - hcond`; `fastchem_cond.py` prepares the final segment upper as `inf`, so KL extrapolates the CH4 final segment beyond FastChem's finite validity upper. CH4 is therefore classified as FastChem data-validity floor versus KL extrapolation, not a stored true-activity mismatch.

MgCO3/SiC remain fixed/full-element donor snapshot mismatches. Their per-element donor tables show FastChem `calcActivity` consumes fixed/post-condensation full-element donor values while KL consumes gas-only `ln_nk`; source swaps confirm the donor vector drives threshold pass/fail. Group B remains blocked by missing exact KL labelled arrays: split-history reconstructs index materialization, but exact RHS/Jac labels, row scaling by label, and solver result vector by label are still unavailable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Diagnostic Counterfactual Split Frontier

The CH4 counterfactual audit is diagnostic-only and does not alter KL thermochemistry or presets. Current KL activity stays positive for all five CH4 rows. Applying a FastChem-style data-validity candidate mask, `T > FastChem fit_coeff_limits.back() -> threshold value -10`, makes CH4 fail the KL candidate threshold in all five broad cases and removes the CH4 one-sided membership. The finite-upper-without-floor KL selector is not emitted as an exact artifact and is reported as a missing diagnostic selector rather than inferred. FastChem thermo/density/formula with KL atomic state fails, while KL final-segment extrapolated thermo with FastChem donor state passes, confirming the CH4 frontier is the data-validity floor versus KL final-segment extrapolation.

MgCO3/SiC remain donor-snapshot driven: FastChem full-element donor values from the fixed overwrite path make both rows pass, while KL gas-only donor values make them fail under both FC and KL thermo terms. Group B remains blocked by missing KL exact labelled RHS/Jacobian/scaling/result arrays; split-history is still the closest materialization.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Production-Readiness Semantic Package

The production-readiness compact and design note are:

- `results/fastchem_cond_kl_production_readiness_compact.json`
- `results/fastchem_cond_kl_production_readiness_compact.md`
- `docs/condensates/fastchem_parity_kl_semantic_design_note.md`

Ledger update:

- KL raw `gas_only_final` is a normalized state. A physical donor conversion is
  required before comparing to FastChem physical donor values.
- The physical donor conversion removes the inflated `C1H4` / `H2O1` donor
  gap, but it is not a donor-snapshot transplant rule.
- Physical donor plus molecule plus inventory is the coherent comparison
  boundary. Molecule-only and inventory-only replays remain non-promotable.
- The remaining PMI residual is localized to `45:-10` and closes with the
  emitted `Al4C3(s)` removed-condensate analytic correction. This is
  provenance, not a production removed-tail patch.
- The all-neutral species table is nonblocking for the current
  inventory/removed verdict.
- Extra broad cases beyond `30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`
  were not regenerated.

Lever classes:

- Ready for design note: physical donor comparability, coherent physical
  molecule + inventory interface, and removed-tail provenance boundary.
- Diagnostic-only: CH4 data-validity mask, MgCO3/SiC donor snapshot, Group-B
  reduced-solve exclusion, full FastChem coherent gas-state replay,
  `Al4C3(s)` removed-tail replay, and all-neutral post-donor species ranking.
- Candidate guarded KL option: a default-off semantic state-interface prototype
  that emits normalized donor, physical donor, molecule cache, inventory rows,
  and removed-correction rows without changing defaults.
- Not promotable: legacy burden-ratio conversion, infinity-norm fallback,
  molecule-only/inventory-only/donor-only/removed-tail transplants,
  row/species/case dropping, broad projection as focused regression, and
  repaired alpha/beta production behavior.

Decision: semantic levers ready for production design note but not promotable.

## Milestone 2 Generalization Readiness

The Milestone 2 compact is:

- `results/fastchem_cond_kl_milestone2_generalization_readiness_compact.json`
- `results/fastchem_cond_kl_milestone2_generalization_readiness_compact.md`

No extra broad cases were run. The reason is not missing source CLI support:
`audit_actual_fastchem_gas_phase_transplant_phase2.py` accepts `--cases`. The
blocker is interpretability. The downstream compact stack still hard-codes the
current five broad cases or focused `45/60` layers, and the current five-case
source/trace artifacts are already `4.0G` combined.

Available-case scorecard:

- Physical donor conversion closes the dominant donor gap.
- Physical + molecule + inventory reduces broad residual in all five available
  cases.
- Removed tail is material only at `45:-10`.
- `Al4C3(s)` removed tail remains localized in the available broad set.
- Metric lineage is preserved; `embedded_broad_10row_projection` is not used as
  focused regression.

Regeneration dependency map:

- Expand the phase2 broad source and trace snapshots.
- Generalize selected-row delta/raw-result provenance to extra case keys.
- Generalize reduced-Newton, molecule-input, electron, and coherent-bundle
  support compacts away from focused `45/60` assumptions.
- Regenerate broad smoke, direct broad eval, Round 8 ladder, and Round 9-style
  removed-tail locality compacts.

Decision: broad generalization requires regeneration campaign before next
decision.

## Milestone 3 Manifest-Driven Infrastructure

The Milestone 3 artifacts are:

- `results/fastchem_cond_kl_broad_case_manifest.json`
- `results/fastchem_cond_kl_milestone3_broad_generalization_infrastructure_compact.json`
- `results/fastchem_cond_kl_milestone3_broad_generalization_infrastructure_compact.md`

The manifest records the current broad cases, source/trace artifacts,
downstream compact availability, physical donor availability,
molecule/inventory ladder availability, and removed-tail locality availability.

The current-five replay is now manifest-driven through
`examples/comparisons/fastchem_cond_kl_milestone3_broad_generalization_infrastructure.py`.
It reproduces the Milestone 2 scorecard:

- physical donor conversion closes dominant donor gaps,
- physical + molecule + inventory improves all five available cases,
- removed tail is material only at `45:-10`,
- `Al4C3(s)` closes the localized `45:-10` PMI tail,
- `embedded_broad_10row_projection` is not used as focused regression.

A one-case pilot was not run. The manifest-driven reader is ready, but the
legacy full replay scripts still need case-key generalization before a pilot
source shard can support a semantic decision.

Decision: manifest-driven broad generalization infrastructure ready; pilot case
not yet run.

## Milestone 4 Pilot-Ready Manifest Gate

The Milestone 4 artifacts are:

- `results/fastchem_cond_kl_milestone4_pilot_ready_generalization_compact.json`
- `results/fastchem_cond_kl_milestone4_pilot_ready_generalization_compact.md`

Milestone 4 removes the hidden current-five fallback from the downstream
diagnostic broad stack. The repaired alpha/beta broad smoke, direct broad eval,
Round 8 ladder, Round 9 locality, and Milestone 3 infrastructure scripts now
require manifest case input for current-five replay. Historical current-five
lists remain only as provenance fields, and the Round 9 `45:-10` tail target is
retained only as the localized material PMI-tail decomposition target.

The manifest-only gate passes on the current five cases and reproduces the
scorecard:

- physical donor conversion closes dominant donor gaps,
- physical + molecule + inventory improves all five available cases,
- removed tail is material only at `45:-10`,
- `Al4C3(s)` closes the localized `45:-10` PMI tail,
- `embedded_broad_10row_projection` is not used as focused regression.

A one-case pilot was not run. The compact records the exact command and
expected artifacts, but source+trace cost remains about `0.8G` before
downstream compacts, so this milestone avoids generating orphan source shards.

Decision: pilot-ready manifest-only downstream stack passes; pilot case not yet
run.

## Milestone 5 One-Case Pilot

The Milestone 5 artifacts are:

- `results/fastchem_cond_kl_milestone5_pilot_45_m5_phase2_source.json`
- `results/fastchem_cond_kl_milestone5_pilot_45_m5_phase2_traces.json`
- `results/fastchem_cond_kl_milestone5_pilot_45_m5_phase2.md`
- `results/fastchem_cond_kl_milestone5_pilot_45_m5_direct_broad_eval_compact.json`
- `results/fastchem_cond_kl_milestone5_one_case_pilot_generalization_compact.json`
- `results/fastchem_cond_kl_milestone5_one_case_pilot_generalization_compact.md`

The single pilot case is `45:-5`. It was selected because it shares the layer
of the localized `45:-10` `Al4C3(s)` removed-tail case while changing epsilon,
making it the highest-value one-case disconfirmation test for case-local versus
layer-wide removed-tail behavior.

The phase2 source generation completed. The pilot was added to
`results/fastchem_cond_kl_broad_case_manifest.json` with source and trace
artifact paths. Direct broad eval can read the pilot source, but full downstream
interpretation remains incomplete because the pilot lacks the selected-row
mapping, same-boundary KL non-fixed vector, and gauge-normalized
inventory/atom fields needed for Round 8/9 projection.

Pilot scorecard:

- physical donor closure: not downstream-evaluable,
- physical + molecule + inventory improvement: not downstream-evaluable,
- removed-tail source correction is material,
- `Al4C3(s)` removed-correction provenance appears at `45:-5`,
- no other removed-tail species were observed in the source removed trace,
- `embedded_broad_10row_projection` was not used as focused regression,
- no rows, species, or cases were dropped.

This changes the locality statement: `Al4C3(s)` removed-correction provenance is
not unique to `45:-10` at the source-trace level. Projected PMI materiality for
`45:-5` is still incomplete until the missing downstream fields are emitted.

Decision: one pilot broad case regenerated but downstream interpretation
remains incomplete.

## Milestone 6 Pilot 45:-5 Interpretation Closure

The Milestone 6 artifacts are:

- `results/fastchem_cond_kl_milestone6_pilot_45_m5_interpretation_closure_compact.json`
- `results/fastchem_cond_kl_milestone6_pilot_45_m5_interpretation_closure_compact.md`

No additional broad case was run. Milestone 6 consumed only the existing
`45:-5` pilot source and trace artifacts.

Missing-field closure:

- same-boundary KL non-fixed vector: present in the pilot source embedded
  `repaired_alpha_beta_source_state_snapshot`; absent from the trace-only file;
  closed from source with e-last to e-first identity gate,
- selected-row mapping: present in the pilot source embedded snapshot; absent
  from the trace-only file; closed from source with explicit projected and
  one-sided row membership,
- gauge-normalized inventory/atom inputs: present in source and trace; closed
  with the current inventory convention
  `rhs_element_total_inventory = total_element_density * element.epsilon` and
  `row_scaling_factor`.

Pilot replay scorecard:

- physical donor conversion closes the dominant raw donor gap,
- physical + molecule does not improve residual,
- physical + molecule + inventory improves relative to molecule-only but not
  relative to baseline,
- `Al4C3(s)` removed tail is material in the source trace,
- `Al4C3(s)` removed correction does not close the projected PMI residual,
- no other removed-tail species were observed,
- `embedded_broad_10row_projection` remains separate from focused regression,
- no rows, species, or cases were dropped.

Comparison with `45:-10`: `Al4C3(s)` removed correction is layer-recurring at
the source-provenance level, but the projected effect is epsilon-dependent.
The `45:-10` tail closure does not generalize to `45:-5`.

Decision: pilot 45:-5 downstream interpretation reveals a new blocker.

## Milestone 21 Hidden Coherent Source Decomposition

Milestone 21 supersedes the Milestone 20 hidden-source label with a vector-level
budget. The direct-broad compact now emits
`milestone21_hidden_coherent_source_decomposition` while `fc_j`, `fc_rhs`, the
KL-native RHS components, and selected-row mapping are live.

For current-five plus the existing `45:-5` pilot, variant I plus the hidden RHS
delta reconstructs the full FastChem RHS to roundoff, and
`solve(fc_j, hidden_rhs_delta)` reconstructs the J-minus-I solution gap. The
same Jacobian and row scaling are used for I and J, so the remaining coherent
source is RHS-side, not Jacobian-side. Top carriers are outside-selected
free-element rows, but no production transplant is implied.

Milestone 21 decision: hidden coherent source is RHS-side.
## Milestone 22 Hidden RHS Term-Family Budget

Milestone 22 decomposes the Milestone 21 hidden RHS vector by RHS term family. Decision: hidden RHS delta is molecule-RHS dominated.

- The additive budget closes for all requested cases: current-five plus the existing `45:-5` pilot.
- The closing source is `full FastChem molecule RHS - reconstructed candidate molecule RHS`; no separate remaining charged/electron additive hidden family is needed for closure.
- Inventory/atom, removed-condensate, tau/complementarity, activity burden, and fixed/condensed overwrite families have zero remaining additive hidden RHS after the Milestone 21 KL-native RHS assembly.
- Outside-selected free-element rows remain the dominant carriers, but they are a row-location carrier view of the molecule RHS residual rather than a separate additive source.
- Production remains not promotable; KL-native reconstruction is blocked on a coherent molecule RHS parity contract, not on a new production rule.
## Milestone 23 Coherent Molecule RHS Parity

Milestone 23 tests the coherent molecule RHS parity contract directly. Decision: coherent molecule RHS parity holds at matched source state.

- Full FastChem molecule RHS is reproduced to roundoff by all-molecule RHS at matched source state.
- The M22 molecule delta is therefore not a request to transplant FastChem RHS; it identifies the missing semantic contract for source-state plus row-scaling/RHS convention parity.
- Source-vector, hvector/lnK, density-gauge, cache timing, and neutral/charged branches are emitted as diagnostic variants.
- KL-native reconstruction remains blocked until the semantic interface exposes a matched molecule source state and RHS convention contract.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 24 Matched Source-State Construction

Milestone 24 attempts to construct the matched coherent molecule source state from KL-native semantic-interface fields. Decision: matched coherent molecule source state blocked by hidden coherent source.

- The matched source-state contract is explicit: e-first physical density vector, FastChem row-scaling convention, molecule species order, hvector/lnK, and density gauge.
- No KL-native candidate among physical donor, molecule-cache, fixed/overwrite, correctValues, or best repaired same-boundary vector constructs the matched source state in all requested cases.
- Tau/complementarity, inventory/atom, removed-condensate, and reduced-slot adjusted all-element molecule-source vectors are not emitted as source-state fields.
- Row scaling is explicit and can be tested separately; it is not sufficient to construct the missing source state.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 25 Source-Vector Materialization

Milestone 25 attempts to materialize the missing all-element molecule source vectors from existing artifacts. Decision: source-vector materialization blocked by exact missing artifacts.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and e-first same-boundary KL candidates were materialized and rerun.
- Tau/complementarity, inventory/atom, removed-condensate, and broad reduced-slot all-element molecule source vectors are not present in existing artifacts; only RHS terms or focused/layer-limited reduced-slot records are available.
- Combination search cannot close because every requested adjusted-vector combination depends on an unmaterialized all-element source vector.
- FastChem row scaling remains explicit and is not used silently; materialized source-state residuals remain nonzero under that explicit convention.
- Production remains not promotable; no C++ trace or production rule was added.
## Milestone 26 All-Element Source-Vector Emitters

Milestone 26 implements diagnostic emitter attempts for the missing all-element source vectors. Decision: all-element source-vector emission blocked by exact trace architecture gap.

- Python emitter attempts were added at the latest diagnostic sites where RHS terms, source traces, row scaling, and molecule labels are live.
- Current artifacts expose RHS term contributions for tau/complementarity, inventory/atom, and removed-condensate paths, but not the adjusted all-element molecule-source vectors.
- The reduced-slot compact remains focused/layer-limited and does not expose a broad case-keyed canonical e-first all-element molecule-source vector.
- Matched-source construction was rerun with all emitted/materialized vectors; no non-hidden candidate closes.
- Production remains not promotable; a future diagnostic trace must emit these all-element source vectors before another construction attempt can close.
## Milestone 27 Trace Architecture Source Vectors

Milestone 27 patched diagnostic C++ trace schema and emitted the reduced-slot all-element source vector. Decision: matched coherent molecule source state still blocked by hidden coherent source.

- The emitted vector is FastChem reference-only and uses the hidden coherent FastChem source state.
- It reproduces the matched coherent molecule source state to roundoff, proving the trace architecture can carry the needed vector shape.
- It does not make the matched source KL-native constructible; KL adjusted all-element tau/inventory/removed vectors remain missing.
- Production remains not promotable; the patch is trace-only and inactive unless diagnostic tracing is enabled.
## Milestone 28 KL-Native Source-Vector Reconstruction

Milestone 28 compares KL-native source-vector candidates against the M27 emitted reduced-slot FastChem reference target. Decision: KL-native source-vector reconstruction blocked by exact KL-side trace fields.

- Physical donor, molecule-cache/correctValues, fixed overwrite, and prior same-boundary KL-native vectors were compared directly against the M27 28-element target.
- No non-hidden candidate or requested staged combination closes the source vector or molecule RHS in all cases.
- Tau/complementarity, inventory/atom, removed-condensate, and KL reduced-slot mapped all-element source vectors remain exact missing fields.
- Production remains not promotable; this milestone is diagnostic-only and does not change presets, defaults, or solver behavior.
## Milestone 29 Exact KL-Side Trace Field Closure

Milestone 29 closes the M28 Python-owned KL-side source-vector fields and reruns reconstruction. Decision: M27 target vector still requires hidden FastChem coherent source.

- All eight M28 fields were ownership-audited and emitted in the M29 diagnostic compact.
- The two fields previously labelled C++-required are KL-side source-state fields after ownership audit, so no C++ trace patch or rebuild was required.
- Emitted non-hidden KL-side vectors still do not reconstruct the M27 FastChem reference target or matched coherent molecule RHS.
- Production remains not promotable.
## Milestone 30 Target-vs-KL Value Delta

Milestone 30 decomposes the value delta between the M27 target and emitted KL reduced-slot source vector. Decision: M27 target blocked by missing semantic transform.

- Element-order, log/linear, density-gauge, and row-scaling verifier branches do not close the residual.
- Fixed/non-fixed and inert/electron splits show the largest differences are not explained by a single bookkeeping subset.
- Least-squares fits can reduce residual diagnostically but require non-production coefficients, indicating a missing semantic transform rather than a direct emitted-vector selection.
- Production remains not promotable.
## Milestone 31 Semantic Transform Synthesis

Milestone 31 synthesizes diagnostic semantic transforms from emitted KL-side vectors to the M27 target. Decision: inert/carrier transform partially closes but not all cases.

- Inert pass-through from the KL physical donor stage removes the dominant He/Ne/Ar carrier residuals diagnostically.
- Source-vector and molecule-RHS closure still fail after all A-H transforms, so the transform is only partial and remains non-production.
- LS-guided class coefficients do not collapse to a global, ntot, or density-gauge scalar across cases.
- No production solver, preset, default, or guarded mode changed.
## Milestone 32 Broader Source-State Prototype

Milestone 32 prototypes broader diagnostic source-state transforms after M31 inert pass-through. Decision: broader transform partially closes but fixed-element source-state remains.

- Gen1 attempted A2-H2; Gen2 attempted targeted top-class, metal/minor, fixed/condensed, and constrained-scalar overlays.
- No non-hidden transform reconstructs the M27 target or matched molecule RHS across current-five plus `45:-5`.
- The balanced best transform remains conservative and the remaining dominant residual class is fixed/condensed source-state material.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 33 Fixed-Element Decomposition

Milestone 33 decomposes the fixed-element source-state blocker. Decision: fixed-element transform rejected due molecule-RHS sign amplification.

- A2, F2, and K2 were decomposed by condensation class and residual class for current-five plus `45:-5`.
- A3-G3 RHS-aware fixed-element transforms were attempted; no transform reconstructs the M27 target and molecule RHS across all cases.
- Source-vector improvements can worsen molecule RHS because fixed-row changes are amplified by molecule stoichiometry and FastChem row scaling.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 34 Molecule-RHS Operator Sensitivity

Milestone 34 audits the source-to-molecule-RHS operator. Decision: sign amplification due row scaling.

- R(x) was evaluated directly from diagnostic source vectors through molecule reconstruction and FastChem row scaling.
- Finite-difference sensitivity on top fixed elements attributes the worsening RHS behavior to row-scaled fixed-element molecule-burden sensitivity.
- RHS-space stage/scalar/finite-difference candidates do not close all cases without hidden coherent source state.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 35 Row-Scaling Amplification Budget

Milestone 35 decomposes row-scaling amplification into numerator and scaling terms. Decision: row-scaling high-gain fixed rows require M27 source parity.

- The compact emits `N(x)`, `R(x)`, `Delta N`, `Delta R`, row-scaling signs/magnitudes, and amplification factors for each covered case.
- Row-scaling-aware candidates A-F do not close all cases, current-five, or `45:-5` without hidden coherent source state.
- High-gain fixed rows remain the practical inverse-problem blocker; production remains not promotable.
## Milestone 36 High-Gain Fixed-Row Provenance

Milestone 36 traces high-gain fixed-row source provenance. Decision: high-gain fixed-row parity insufficient; full M27 source vector remains required.

- High-gain fixed rows were selected from M35 row-scaling amplification and RHS contribution budgets.
- FastChem and KL ladders show the M27 reduced-system assembly source value is not reproduced by emitted KL-native stages.
- High-gain-row-only replays are diagnostic-only and do not close all cases; full M27 source-vector parity remains required.
- No production solver, preset, default, or C++ trace changed.
## Milestone 37 Source-Vector Support Decomposition

Milestone 37 decomposes source-vector support. Decision: sparse support partially closes but not all cases.

- Row classes and overlaps were emitted for high-gain fixed, inert, electron, volatile, reactive, metal/minor, top-species, outside-selected, and row-scaling high-gain rows.
- Replays A-K and greedy support search found a stable outside-selected support branch that partially closes, including `45:-5`, but not all current-five cases.
- Top-species support rows improve attribution but do not replace full M27 source-vector parity for non-closing cases.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 38 Sparse-Support Residual Decomposition

Milestone 38 decomposes residuals after the M37 sparse support. Decision: 45:-5 closes by cancellation; current-five requires broader support.

- The best sparse support residual was decomposed for H/K, and `45:-5` was compared against non-closing current-five cases.
- Outside-selected internal ablations A-G and a second-stage greedy search were attempted.
- The 45:-5 closure is not a general support sufficiency result; current-five still requires broader support.
- Production remains not promotable; no C++ trace, preset, default, or production rule changed.
## Milestone 39 30:-10 Sparse-Support Residual

Milestone 39 decomposes the remaining `30:-10` sparse-support residual. Decision: 30:-10 residual is numerical/tolerance-scale but not production-promotable.

- H/B/G/K support residuals were decomposed for `30:-10` and compared with closing cases.
- Internal add/remove ablations A-H and tolerance/scale audit classify the remaining residual as strict-tolerance scale.
- Closure under looser diagnostic tolerance is not production-promotable; production remains not promotable.
## Milestone 40 Production-Path Decision Campaign

Milestone 40 campaign decision: strict tolerance residual is diagnostic-only; production requires source-state contract.

- Track 1: strict 30:-10 residual closes only at diagnostic 1e-5 and is not production-promotable.
- Track 2: emitted non-hidden KL full-vector candidates do not close source, numerator, or row-scaled RHS parity.
- Track 3: production readiness now depends on a formal source-state semantic contract, not another support subset.
- Track 4: no additional broad pilot is justified until the contract gap is addressed.
## Milestone 41 Default-Off Source-State Contract Prototype

Milestone 41 created the default-off source-state contract prototype. Decision: default-off source-state contract prototype complete; KL-native implementation remains blocked.

- The schema separates FastChem/M27 reference records, best non-hidden KL candidates, sparse support overlays, tolerance-only closures, and the unavailable production-ready KL-native source state.
- The acceptance gate requires source parity, numerator parity, row-scaled RHS parity, no hidden FastChem source, strict tolerance, complete coverage, explicit row scaling, and preserved lineage.
- Current-five plus `45:-5` are instantiated from existing artifacts; no new broad pilot was requested.
- All best non-hidden KL candidates fail the default-off production gate because the source-state constructor and numerator contract are still missing.
## Milestone 43 Gate-Driven Constructor Synthesis

Milestone 43 synthesized gate-driven constructors from non-hidden KL-side basis vectors. Decision: free fit also fails; hidden source-state information structurally absent.

- Generations G1-G3 attempted source-, numerator-, RHS-, joint-, class-wise, constrained, and free diagnostic fits.
- M41 gate evaluation remained default-off and diagnostic-only; production behavior and presets were unchanged.
- The free diagnostic fit also failed source, numerator, and RHS gate closure, classifying the residual as structurally absent from the non-hidden KL basis.
## Milestone 44 Structural Basis Expansion

Milestone 44 expanded the non-hidden KL primitive basis and emitted structural span diagnostics. Decision: diagnostic C++ trace required for missing primitive source-state fields.

- Current and expanded basis rank/projection residuals were emitted for source vector, unscaled numerator, and row-scaled RHS spaces.
- Primitive branches A-G were attempted; exact number_density_min/maj and epsilon/phi/degree transforms remain unavailable without diagnostic trace fields.
- Expanded basis gate rerun did not close M41; next work requires diagnostic C++ trace for missing primitive source-state fields.
## Milestone 45 Trace Primitive Source-State Fields

Milestone 45 added env-gated C++ primitive source-state trace fields. Decision: traced primitives improve but remain FastChem reference-only.

- The M45 marker emits fixed-row pre/post overwrite values, number_density_min/maj, gas solver path, epsilon, phi, and degree_of_condensation.
- The traced primitive basis reruns the M41 gate but remains diagnostic FastChem reference-only, not KL-native production logic.
- Production remains not promotable; the next implementation step is coding the semantic source-state algorithm on the KL side.
## Milestone 46 KL-Native Semantic Algorithm

Milestone 46 implemented a default-off KL-native semantic source-state algorithm prototype. Decision: KL lacks required lifecycle input fields for semantic source-state algorithm.

- Variants A-D compute primitive fields from KL/public vectors only; M45 trace values are diagnostic reference targets only.
- The M41 gate rerun preserves no-hidden-source, not-reference-only, and KL-native-constructible checks, but source/N/R parity remains open.
- Missing KL lifecycle inputs are now narrowed to molecule contribution order, backup branch lifecycle, condensation-degree transform inputs, and reduced correctValues assembly semantics.
## Milestone 47 Function-Port Closure

Milestone 47 localized semantic source-state gaps by FastChem function. Decision: FastChem semantic source-state functions must be ported before production.

- Four default-off Python function-port prototypes were implemented and compared against M45 trace references without using traced values as constructor inputs.
- Three M47 source-state candidates were gated through M41 over current-five plus 45:-5.
- Remaining blockers are exact lifecycle fields: molecule order/accumulators, backup/intertSol branch state, condensation-stage inputs, and correctValues reduced result/clipping state.
## Milestone 48 Lifecycle Emitters

Milestone 48 emitted default-off KL lifecycle state records and reran five M48 candidates through M41. Decision: M48 improves but molecule order state remains missing.

- Implemented lifecycle emitters: condensation_stage_state, correctValues_reduced_result_state, gas_solver_branch_state, minor_major_accumulator_state.
- Unavailable lifecycle emitters: molecule_order_state.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem reference-source transplant was changed.
## Milestone 49 Molecule Order Closure

Milestone 49 emitted KL molecule-order state proxies and reran minor/major accumulator replay. Decision: exact molecule_order_state requires FastChem-specific ordering not present in KL.

- KL-native molecule-order emitters: A_KL_gas_species_order, B_FastChem_label_aligned_KL_density, D_best_non_hidden_molecule_order.
- Exact element-specific FastChem minor/major molecule order remains unavailable from KL artifacts and would need a diagnostic trace for validation.
- No production solver behavior, presets, defaults, tolerance, row/species/element/case coverage, or FastChem source-vector transplant was changed.
## Milestone 50 Molecule Order Trace

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
