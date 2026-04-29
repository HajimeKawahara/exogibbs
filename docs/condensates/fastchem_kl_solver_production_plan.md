# FastChem/KL Solver Production Plan

The current work remains audit-only. Production ExoGibbs solvers and presets are unchanged.

## Latest Broad Projection Residual Decomposition

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_broad_projection_residual_decomposition_compact.md`

Result summary:

- The focused one-sided frontier is diagnostically closed with `12 -> 0` unresolved rows: CH4 data-validity mask `5`, MgCO3/SiC FC donor snapshot `2`, and Group-B intentional reduced-solve exclusion `5`.
- The broad projection is not promoted into a focused regression. It remains the `embedded_broad_10row_projection` outside-selected/full-vector probe.
- Broad projection rows originally failing are `40`. Focused-frontier levers annotate `12` rows, but only `2` numeric broad failures are explained; `38` broad rows still fail and are preserved in the compact.
- Broad full-vector term differences are now emitted for all five broad cases, and all five classify as neutral molecule full-vector provenance dominated.
- A diagnostic-only broad replay protocol now preserves `embedded_broad_10row_projection`. Combined available-term alignment reduces the broad residual in all five cases. Beta/electron-only and row-scaling/Jacobian-only broad replays are still unavailable and recorded with exact missing fields.

Production implication:

- No production electron rule was added.
- No guarded mode was added.
- No solver, preset, FastChem, KL, RHS, molecule, inventory, removed-tail, selected-row, row-scaling, lifecycle, labelled-system, density-gauge, donor, thermochemistry, result-index, or broad-projection behavior changed.
- The CH4 data-validity mask, FC donor snapshot, Group-B exclusion, and repaired alpha/beta remain diagnostic-only and non-promotable.

Decision: broad projection residual is dominated by neutral molecule full-vector source state.

## Latest Neutral Molecule Full-Vector Provenance

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_neutral_molecule_full_vector_provenance_compact.md`

Result summary:

- All `40` broad failing rows are retained under `embedded_broad_10row_projection`; the broad projection is not used as focused regression.
- Molecule RHS alignment alone worsens all `40` rows, and inventory/atom alignment alone worsens all `40` rows. Combined available-term alignment improves all `40` rows, so the result is a coherent bundle requirement rather than a molecule-only rule.
- The current broad replay does not emit top neutral molecule species by projected row or source-stage lineage for those species. The audit records exact missing fields and patch sites instead of inferring species contributors.

Production implication:

- No production molecule transplant rule is justified.
- No donor transplant, electron, guarded, selected-row, result-index, or broad-projection rule is promotable.

Decision: neutral molecule residual requires coherent molecule+inventory+removed/tau bundle.

## Latest Fixed-Element Source Decomposition Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_source_decomposition_compact.md`

Result summary:

- Molecule cache is still treated as a downstream symptom. The new boundary is the FastChem cached full-element vector just before molecule refresh.
- The exact source split proven at that stage is `reduced overwrite from correctValues/correctValuesFull` versus `carry-forward full-element value`. That explicit source decomposition self-closes to the final cached value.
- The requested additive physical components are still not explicit in production source at that stage. The audit records the missing fields exactly and does not infer them:
  `free_atomic_gas_component`
  `condensed_or_fixed_correction_component`
  `total_inventory_component`
  `electron_specific_component`
  all from `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`.
- On the focused layers, replaying FastChem fixed-element overwrite values into the KL input vector remains the only single-component replay that materially improves molecule reconstruction:
  `45:-10`: `147.92838359730413 -> 15.226673958408794`
  `60:-5`: `140.48218322609824 -> 19.407748535067963`
  Electron-only replay stays near baseline, and overwrite + electron still does not produce a coherent selected-row RHS replay.
- The earliest fixed-subset divergence remains before `post_selectActiveCondensates_reset` on both focused layers.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No fixed-element overwrite, membership, `phi`, degree, electron, molecule, or RHS rule is promotable from this audit.

Decision: mismatch is dominated by fixed-element overwrite values.

## Latest Fixed-Element Overwrite Provenance Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_overwrite_provenance_compact.md`

Result summary:

- The FastChem cached fixed-element overwrite is source-proven as `elem_densities_new[i]` written through `elements_cond[i]->number_density = elem_densities_new[i]` inside `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`.
- In the focused entrance smoke the overwrite path is the reduced condensed-phase solver path, so the overwrite values come from `CondensedPhase::correctValues`. The audit does not infer any additive physical component split at this boundary.
- The overwrite is a full replacement, not a partial modification. The exact numeric prior full-element carry-forward value at the write point is still not present in the compact artifact, so the audit records the missing local variable explicitly: `full_element_densities_before_write[i]` in `CondensedPhase::calculate`.
- KL still does not expose an overwrite-like operation, overwrite mask consumer, or a source-clean carry-forward vs overwrite distinction at the corresponding stages.
- Overwrite-only replay remains the decisive single-component result:
  `45:-10`: `147.92838359730413 -> 15.226673958408794`
  `60:-5`: `140.48218322609824 -> 19.407748535067963`
  while full FC cached vector still gives the known selected-row boundary:
  `45:-10`: `0.08475871276711094`
  `60:-5`: `1.3724132941206335e-11`

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No overwrite-value, overwrite-mask, overwrite-timing, electron, molecule, or RHS rule is promotable from this audit.

Decision: mismatch is dominated by overwrite values themselves.

## Latest Fixed-Element Materialization Boundary Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_element_materialization_boundary_compact.md`

Result summary:

- The KL gas-recoupling output exists and is adopted in the diagnostic gas replay path, but the later molecule-input boundary is still rebuilt inline before iter1 RHS/Jacobian assembly.
- The exact pre-gas vector is not emitted as a standalone compact boundary object. The audit records the exact missing field rather than inferring it:
  `gas_trace.post_condensed_phase_fixed_atomic_element_species_state`
  from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::actual_fastchem_like_coupled_loop`.
- The first later consumer is `_second_post_seed_update_actualization_solve`, which reads `gas_only["ln_nk"][:n_elem]` and forwards that `u` into `_assemble_fastchem_reduced_update`.
- `_assemble_fastchem_reduced_update` reconstructs the atom/full-element vector inline from current `u`, and no explicit fixed-element overwrite consumer is exposed before that reconstruction. The audit records the exact missing object:
  `iter1_molecule_input.fixed_element_bookkeeping_consumer`
  from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- On the focused layers, gas-recoupling output does not improve the molecule replay, while the known overwrite boundary remains unchanged:
  `45:-10`: current `147.92838359730413`, gas recoupling `149.20929529868224`, current + FC fixed overwrite `15.226673958408794`, overwrite + electron `0.4347374576125725`, full FC cached input `0.08475871276711094`.
  `60:-5`: current `140.48218322609824`, gas recoupling `140.07824601545863`, current + FC fixed overwrite `19.407748535067963`, overwrite + electron `0.565493860944129`, full FC cached input `1.3724132941206335e-11`.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No gas-recoupling carry rule, fixed-overwrite materialization rule, fixed-mask rule, molecule rule, RHS rule, or consumer rule is promotable from this audit.

Decision: KL explicit fixed-overwrite consumer is missing before molecule reconstruction.

## Latest Synthetic Fixed-Overwrite Consumer Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_synthetic_fixed_overwrite_consumer_compact.md`

Result summary:

- Gas-recoupling output adoption is unchanged from the materialization-boundary audit: the diagnostic gas replay path adopts `gas_result.ln_nk`, but the later molecule boundary is still the inline-recomputed `u` consumed by `_assemble_fastchem_reduced_update`.
- The explicit fixed-overwrite consumer remains missing before molecule reconstruction:
  `iter1_molecule_input.fixed_element_bookkeeping_consumer`
  from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- Diagnostic synthetic consumers that splice gas-recoupling fixed-element values into the later KL molecule-input vector do not improve the selected-row replay on either focused layer. The current KL vector `A` remains the best KL-side rung, while full FC cached input `H` remains the only coherent upper bound:
  `45:-10`: `A=36.90705832854512`, `C=36.90706738203272`, `D=36.90706738203608`, `H=0.08475871276711094`
  `60:-5`: `A=50.41942341750188`, `C=50.41942442202005`, `D=50.41942442400552`, `H=1.3724132941206335e-11`
- The fixed-subset adoption test classifies gas-recoupling fixed values as insufficient to explain the FastChem overwrite boundary on both focused layers, and the added electron term is negligible in this synthetic ladder.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No gas-recoupling fixed-subset consumer rule, electron-enhanced synthetic consumer rule, or mixed carry/overwrite materialization rule is promotable from this audit.

Decision: gas-recoupling fixed values are insufficient even with a synthetic consumer.

## Latest Elem Densities New Source Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_elem_densities_new_source_compact.md`

Result summary:

- The synthetic fixed-overwrite consumer result remains unchanged upstream: gas-recoupling fixed values are insufficient even with a synthetic consumer, so the next diagnostic boundary is the exact FastChem local construction of `elem_densities_new[i]`.
- In the focused reduced branch, `CondensedPhase::correctValues` now exposes the exact local chain used for every fixed row later written back through `elem_densities_new[i]`:
  `elem_number_dens_old[i] -> result(i + nb_cond_jac) -> delta_n_elem -> update_factor -> elem_number_dens_new[i]`
- Every emitted fixed row on both focused layers classifies as `pure overwrite`. No fixed row emits a dedicated additive free-gas carry-forward, condensed/fixed correction, or electron-specific component local.
- Overwrite-term-only replay remains the only source-clean component candidate and reproduces the established molecule-only improvement without coherent selected-row closure:
  `45:-10`: `147.92838359730413 -> 15.226673958408794`
  `60:-5`: `140.48218322609824 -> 19.407748535067963`
  while full FC cached input still gives the known selected-row upper bound:
  `0.08475871276711094`
  `1.3724132941206335e-11`
- The dominant source decision is therefore: mismatch is dominated by the overwrite term itself.
- The earliest source divergence for that dominant component remains `before post_selectActiveCondensates_reset` on both focused layers.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No overwrite-term rule, carry-forward rule, condensed/fixed correction rule, electron-specific rule, or mixed source-construction rule is promotable from this audit.

Decision: mismatch is dominated by the overwrite term itself.

## Latest Reduced-Newton Result Slot Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_reduced_newton_result_slot_compact.md`

Result summary:

- The write-site provenance result remains unchanged upstream: the overwrite term itself still dominates `elem_densities_new[i]`, so this step resolves the fixed-row reduced-Newton result slot that later appears at `result(i + nb_cond_jac)`.
- For each fixed row, the audit now emits the exact same-iteration reduced slot index, scaled RHS entry, solver result entry before caller-side/global rescaling, row scaling factor, reduced Jacobian diagonal, and the later `correctValues` local chain.
- Exact same-iteration row and column labels still are not emitted in the reduced-newton anatomy trace. The audit records those as exact missing fields and only uses the nearest labelled iter1 full reduced-system record for structural classification.
- No explicit retained-condensate, removed-condensate, fixed/condensed correction, or electron-coupling local exists at the fixed-row write site. Those remain exact missing locals rather than inferred numeric subcomponents.
- The new decisive solver-side distinction is the explicit caller bridge in `CondensedPhase::calculate`: `global_scaling_factor`. Replaying the same-iteration solver result slot with that bridge reproduces the raw-result overwrite rung on both focused layers, while the unbridged slot is materially worse:
  `45:-10`: bridged `15.226673958408796`, unbridged `22.443213083775767`
  `60:-5`: bridged `19.40774853506796`, unbridged `26.494334739330682`
- Full FC cached input is still required for coherent selected-row closure, so the narrowed solver-side result does not create a promotable rule.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No reduced-result-slot rule, global-scaling transplant rule, retained-condensate coupling rule, removed-condensate coupling rule, fixed/condensed correction rule, or electron-coupling rule is promotable from this audit.

Decision: mismatch is dominated by missing global-scaling bridge after solve.

## Latest Fixed-Row Solve-Space Audit

The latest compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_fixed_row_solve_space_compact.md`

Result summary:

- The reduced-Newton bridge result is now fixed as provenance, not as a rule: `global_scaling_factor * z_i` reproduces the fixed-row write-site result exactly, so the next frontier is the solve-space origin of `z_i`.
- For each fixed row, the audit emits the exact same-iteration slot index, scaled RHS entry `b_i`, solve result `z_i`, caller/global scaling factor `g`, bridged value `g*z_i`, row scaling factor, and solve-space diagonal `J_ii`.
- Exact same-iteration labelled matrix rows are still not emitted from `CondPhaseSolver::newtonStep`. The audit records the exact missing locals `jacobian` and `rhs` for that same-iteration matrix-row boundary, then uses the emitted labelled iter1 full reduced-system analogue for structural decomposition.
- On that emitted labelled analogue, the fixed-row equations close to roundoff on a relative basis and the Schur-style attribution can be isolated into RHS-row-only, condensate-coupling, and other-element-coupling pieces.
- No single solve-space component explains the fixed-row slot. The bridged full slot remains better than any isolated solve-space component on both focused layers, even though RHS-row-only is the best of the isolated additive candidates:
  `45:-10`: bridged slot `15.226673958408796`, RHS-only `15.977102788049098`, condensate `16.19561867754053`, other-element `15.924659915242676`
  `60:-5`: bridged slot `19.40774853506796`, RHS-only `20.027064857591796`, condensate `20.28568059122682`, other-element `20.357508000322714`
- The dominant solve-space decision is therefore mixed, not a clean RHS or coupling-only result.
- Earliest divergence for that dominant object is `inside reduced Newton solve-space assembly`.

Production implication:

- No production solver fix was added.
- No guarded mode was added.
- No fixed-row RHS-only rule, condensate-coupling rule, other-element-coupling rule, mixed coupling rule, or solve-space replay rule is promotable from this audit.

Decision: mismatch is dominated by mixed solve-space coupling in fixed rows.

## Latest Coherent Bundle Audit

The latest compact coherent gas-state bundle audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_coherent_gas_state_bundle_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_coherent_gas_state_bundle_compact.md`

Result summary:

- The selected-row mapping metric is preserved end to end. The ladder reproduces the established `baseline_ABC` residuals exactly on the selected-row mean metric and explicitly labels the selected-row max and full-vector norms separately.
- Inventory gauge normalization remains source-proven but diagnostic-only, molecule/inventory cancellation remains recovered after gauge normalization, and the layer-45 removed tail remains source-proven and separate.
- The coherent closure ladder on the mapped selected rows now shows the current bundle boundary directly:
  KL current RHS under coherent FC Jacobian is still large (`22.53965045170672` / `258.2372817218515`),
  activity + burden reduces to `3.0603000036721433` / `2.5285823085810524`,
  full complementarity tightens to `0.5620237706782278` / `0.9142498895939201`,
  FC molecule alone is destructive,
  FC molecule + inventory/atom in the best gauge-normalized basis closes layer `60:-5`,
  and exact removed closes layer `45:-10` to `9.230739629452324e-12`.
- Bundle availability is asymmetric at iter1 RHS entry. FastChem exposes the full element vector used for molecule reconstruction; KL does not expose a symmetric full element vector or a per-molecule mass-action constant ledger in the entrance-smoke trace.
- Therefore KL-native molecule and full-bundle reconstruction are still not source-proven. The audit keeps the production boundary unchanged: full FastChem gas-state bundle remains diagnostic-only; no smaller KL production rule is promotable.

## Latest KL-Native Molecule Reconstruction Audit

Molecule refresh-timing / FC mass-action ledger audit:

- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_entrance_smoke.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_entrance_smoke_traces.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_molecule_refresh_timing_compact.md`

Result summary:

- FastChem per-molecule mass-action fields are now available at iter1 RHS/Jacobian entry and in refresh-stage cached-density records. FastChem formula self-closure from FC full element vector plus FC mass action closes to max absolute error `5.551115123125783e-17` on both focused layers.
- The earlier FC-mass-action unavailable variants are now runnable. They do not recover a KL-native molecule/inventory cancellation. KL vector + FC mass action behaves like the density-bridge variant and remains non-closing; FC vector + KL mass action is destructive.
- A distinct KL cached-refresh molecule snapshot remains unavailable. The KL audit path still reports `no_discrete_refresh_stage`: molecule cache is computed inline at RHS/Jacobian assembly from `u`, `A_mol`, and `hmol`.
- Direct FC cache remains the only robust molecule replay reaching the known cancellation boundary; layer `45:-10` still needs the separate exact removed correction and layer `60:-5` closes after gauge-normalized inventory/atom.

Production implication:

- No production solver fix, guarded mode, molecule-cache refresh rule, mass-action/hvector rule, density-gauge rule, or cached-snapshot transplant is promotable.
- The next source-clean requirement is a stage-labelled KL post-refresh full element vector and molecule cache, if one exists; otherwise the timing provenance remains diagnostic-only.

Decision: KL-native molecule timing provenance remains mixed or inconclusive.

Fresh field-completion audit:

- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_entrance_smoke.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_entrance_smoke_traces.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_fields_fresh_compact.md`

Fresh result summary:

- The KL-side missing-field blocker is resolved for the focused entrance smoke. `_assemble_fastchem_reduced_update` now emits the RHS-entry full element vector `u`, physical atom basis `exp(clip(u))`, per-molecule `mass_action_constant=-hmol`, molecule cache `exp(clip(A_mol.T @ u - hmol))`, and an explicit inline refresh-equivalent ledger.
- KL molecule formula self-closure from the emitted fields closes to roundoff on both focused layers, with max absolute closure error `2.2737367544323206e-13`.
- Source-clean KL molecule reconstruction still does not recover the FastChem molecule/inventory cancellation. KL full vector + KL mass-action leaves paired residuals `36.907067382036104` / `50.419424423672645`; the density-gauge bridge variant leaves `36.90705832854512` / `50.41942341750188`.
- Direct FC molecule cache remains the only available molecule vector that reaches the established cancellation boundary when paired with gauge-normalized inventory/atom. Layer `45:-10` still needs the separate exact removed correction; layer `60:-5` closes at roundoff.
- Variants requiring FC per-molecule mass-action constants are explicitly unavailable because the current FastChem trace exposes the RHS-entry molecule cache and full element vector, but not `FastChem.iter1_RHS_assembly_entry.mass_action_constants_by_molecule`.

Production implication:

- No production solver fix, guarded mode, molecule-cache transplant, hvector convention change, density-gauge rule, row-scaling rule, or lifecycle change is promotable.
- The completed KL molecule-source fields are provenance instrumentation only.

Decision: molecule mismatch is dominated by molecule-cache refresh timing.

The latest compact KL-native molecule reconstruction audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_reconstruction_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_kl_native_molecule_reconstruction_compact.md`

Result summary:

- FastChem molecule-cache provenance remains internally coherent in the focused entrance smoke. The cached vector is stable from `after_iter0_calcNumberDensity_refresh` to iter1 RHS entry, and the direct FC cache continues to recover the known post-complementarity closure boundary when paired with gauge-normalized inventory/atom.
- KL still does not emit the source-clean fields required to test a KL-native replay at the same stage. The blocking fields remain the symmetric full element vector and per-molecule mass-action ledger at molecule-cache refresh / iter1 RHS entry, all missing from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- Earlier molecule audits already proved that FastChem and KL each self-close their own molecule-density formulas to roundoff. The remaining blocker is therefore not local formula algebra but the missing cross-state reconstruction inputs needed to rebuild the FastChem cache from KL-native variables.
- No KL-native reconstruction branch is available from the emitted fields. The only molecule replay that still reaches the coherent closure boundary uses the direct FC molecule cache, with layer `45:-10` then requiring the already-proven removed tail and layer `60:-5` already closed after gauge-normalized inventory/atom.
- Production implication is unchanged and stricter at the molecule level:
  no production solver fix,
  no guarded mode,
  no KL-native molecule reconstruction rule,
  and no molecule-cache transplant are promotable.

Decision: molecule state requires FastChem hidden/coupled snapshot.

## Latest Baseline / Gauge Reconciliation

The latest diagnostic compact audit is:

- `results/actual_fastchem_gas_phase_transplant_phase2_inventory_gauge_baseline_reconciliation_compact.json`
- `results/actual_fastchem_gas_phase_transplant_phase2_inventory_gauge_baseline_reconciliation_compact.md`

Result summary:

- Fresh exact total-inventory rows remain available on both sides and both focused layers, so the compact replay is no longer blocked by missing exact inventory rows.
- The fresh baseline_ABC mismatch was not a new source-state or solver mismatch. A common-code reconstruction from the fresh entrance-smoke artifact reproduces the old post-complementarity baseline exactly on the prior selected-row metric, while the larger fresh compact values are reproduced by a full-vector infinity norm. The mismatch is therefore an extractor bug / metric mismatch, not a changed physical baseline.
- KL budget-vector inventory and FastChem exact physical inventory are now reconciled by gauge:
  KL budget equals FastChem normalized epsilon rows to roundoff.
  KL budget multiplied by FastChem total-element density equals FastChem physical total-inventory rows to roundoff.
- In solve space, the scaled inventory discrepancy is explained by gauge normalization rather than a newly proven total-density, epsilon, sign, or row-scaling defect.
- After that reconciliation, the post-complementarity replay returns to the established pattern:
  layer `60:-5` closes with molecule + inventory + atom,
  layer `45:-10` reduces to `0.08475871276904615` with molecule + inventory + atom and closes to `9.736646057313515e-12` after exact removed replay.

Production implication:

- No production solver fix, guarded mode, or smaller replay rule is promotable.
- Gauge conversion is source-proven but diagnostic-only.
- Full coherent gas-state bundle remains diagnostic-only; no smaller production rule is promotable.

## Latest Exact Source Audit

The latest compact exact total-inventory / removed-source audit is now running on a truly fresh entrance-smoke artifact:

- `fastchem/fastchem_src/diagnostic_trace.h` now emits exact per-element total-inventory rows and per-removed per-element correction rows from `CondPhaseSolver::assembleRightHandSide`.
- `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` now emits the KL exact analogue for the RHS-consumed total-inventory vector entry and per-removed correction rows.
- Fresh artifact paths:
  `results/actual_fastchem_gas_phase_transplant_phase2_total_inventory_removed_source_fresh_entrance_smoke.json`
  `results/actual_fastchem_gas_phase_transplant_phase2_total_inventory_removed_source_fresh_entrance_smoke_traces.json`
  `results/actual_fastchem_gas_phase_transplant_phase2_total_inventory_removed_source_fresh_compact.json`

Fresh field-presence gate result:

- FastChem exact total-inventory rows are present on both layers: `22` rows at `45:-10`, `23` rows at `60:-5`.
- KL exact total-inventory rows are present on both layers: `28` rows at `45:-10` and `60:-5`.
- Layer `45:-10` FastChem removed trace now emits `22` per-removed per-element rows and self-closes to roundoff.
- KL removed trace is explicitly empty when `condensates_rem` is empty.

As a result:

- exact total-inventory source closure is replay-ready on both sides and closes to roundoff from the emitted RHS-consumed fields,
- FastChem/KL exact total-inventory comparison remains provenance-mixed because KL exposes `budget[element]` directly rather than separate `total_element_density` and `epsilon`,
- molecule plus exact total-inventory materially improves both layers, but closure still requires the full coherent gas-state bundle,
- layer-45 removed-tail provenance is now source-proven and separate,
- no solver fix, guarded mode, or smaller production rule is promotable.

## Frozen Inputs

- Preserve total-inventory `maxDensity`.
- Preserve full prescan row materialization.
- Preserve the pressure/density gauge bridge.
- Preserve exact physical atomic donor scalar conversion from `gas_only`.
- Preserve immediate entry seed semantics: `n=maxDensity`, `lambda=1`.
- Preserve first post-seed `correctValues` semantics.
- Preserve the currently audited second `correctValues` algebra.

## Not Production Candidates Yet

- Do not transplant guessed donor, lnK, or preset changes.
- Do not replay FastChem `delta_n_cond` in production.
- Do not change later Newton steps, lifecycle schedules, final-tail handling, or gas refresh from this audit.

## Current Blocker

The second `correctValues` formula is not the blocker. The four-way replay validates the algebra for shared rows:

- `FC_old + FC_delta` through the FastChem trace closes exactly.
- `FC_old + FC_delta` through the KL `correctValues` closure closes to machine precision.
- Exact iter-1 `delta_n_cond` replay remains a negative diagnostic result: the focused mean log-density mismatch worsened from `0.485277` to `0.732594`, while the lambda mismatch stayed at `0`.

The current frontier is the iter-1 reduced Newton row universe and assembly:

- retained/removed partition,
- KL-only row participation,
- reduced RHS assembly,
- reduced Jacobian assembly,
- row scaling,
- solver result vector,
- result-to-condensate mapping,
- old `n`, `lambda`, log activity, or element densities.
- maxDensity cap status inherited from iter 0.

The focused mapped old-state audit classified rows as `log_activity_old_mismatch=12`, `inherited_iter0_n_old_mismatch=5`, and `row_mapping_or_alignment_mismatch=3`, but component swaps showed old log activity alone is not the controlling lever. FastChem now emits iter1 old element densities in full global element order, with 22/28 mapped elements at `45:-10` and 23/28 at `60:-5`; FastChem old log activity reconstructs exactly for 13 focused rows.

The reduced-system row-universe audit shows the stronger current blocker:

- Across the focused smoke, FastChem has 219 iter1 active rows and KL has 234.
- All 15 KL-only rows participate in RHS/Jacobian/row scaling.
- Removing KL-only rows improves `45:-10` focused mean log-n mismatch from `1.031812` to `0.451817`; using the FC partition on the shared rows improves it further to `0.354056`.
- The same shared-row restriction worsens `60:-5`, so this is not yet a production rule.
- Full FC RHS/Jacobian/scaling/result matrices are now emitted for the focused smoke, but exact component replay is still unavailable because the labelled reduced systems are not isomorphic: KL has extra reduced rows/columns after alignment.

The guarded diagnostic mode `actual_true_kl_atomic_branch_exact_iter1_row_universe_replay` was added only to replay the FastChem iter1 active row set inside the KL audit path. It preserves the frozen upstream states and only restricts iter1 reduced Newton rows. In the focused entrance smoke it improves aggregate mean log-n mismatch from `0.6477089522` to `0.4598570508`, keeps the lambda mismatch at `0`, and confirms all 15 KL-only rows participate in RHS/Jacobian/row scaling before replay. The per-layer behavior remains mixed: `45:-10` improves, while `60:-5` worsens. Therefore a row drop is not production-ready.

FastChem exclusion provenance from the current raw sequence classifies all KL-only rows as `never_candidate_in_fastchem` at the prescan candidate trace. KL persistence is classified as `missing_fastchem_eviction_rule` for all 15 rows because KL keeps them active into iter1 and all enter the reduced RHS/Jacobian/scaling path. This is sufficient for a diagnostic replay, but not sufficient for a production row-exclusion rule because full RHS/Jacobian/scaling/result replay still shows large projected differences.

No reduced-system replay mode is promoted.

## Full Reduced-System Trace Status

The latest audit adds machine-readable iter1 full reduced-system traces on both sides:

- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep` record `condensed_phase_iter1_full_reduced_system`.
- KL source: `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_second_post_seed_update_actualization_solve` field `full_reduced_system_trace`.

For `45:-10`, FastChem emits `124` reduced rows/columns and KL emits `137`; all FastChem rows/columns match labels, but KL has `13` unmatched reduced rows/columns. For `60:-5`, FastChem emits `139` and KL emits `153`; KL has `14` unmatched reduced rows/columns. Because the systems are not label-isomorphic, RHS/Jacobian/scaling/solver-result component replay would require a source-proven projection rule. No such rule is proven in this smoke.

Matched-entry differences already show the remaining reduced-system mismatch is material. At `45:-10`, RHS-after-scaling mean absolute residual is `1.384750`, row-scaling mean absolute residual is `7.3375e16`, solver-result mean absolute residual is `1807.030749`, and matched `delta_n_cond` mean absolute residual is `2392.648097`. These are diagnostics, not a production rule.

## Projection and Scaling Audit Status

The latest projection audit tested shared labelled subspaces rather than positional arrays:

- `45:-10`: shared rows/cols are `102` retained condensates plus `22` free elements; KL extras are `6` KL-only condensates, `1` additional retained condensate, and `6` free elements.
- `60:-5`: shared rows/cols are `116` retained condensates plus `23` free elements; KL extras are `9` KL-only condensates and `5` free elements.

The drop-only, freeze, and source-labelled KL-only-condensate projections are numerically solvable but do not close the mismatch. Focused log-n mismatch becomes `1.5367` at `45:-10` and `2.5021` at `60:-5`. Schur complement projection is also solvable but worse or still high: `2.4337` and `2.1889`, with KL-extra block condition numbers `1.5121e6` and `4.4298e6`.

Scaling is now a material candidate blocker. On the projected shared subspace, FastChem row scaling and common scaling computed from the FastChem projected matrix reproduce the FastChem scaled solve to near roundoff (`~1e-11` solver-result residual). KL row scaling gives mean solver-result residuals of `776.7` at `45:-10` and `1168.2` at `60:-5`. No production change follows from this; a source-level FastChem-equivalent scaling rule must be proven before any guarded actualization.

## Row-Scaling and Solver-Conditioning Audit Status

The latest audit records the source-level scaling formulas and solver backends:

- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian` computes `scaling_factors = jacobian.rowwise().maxCoeff()` and divides Jacobian rows by those factors.
- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleRightHandSide` divides the RHS by the same row factors.
- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::solveSystem` uses `Eigen::PartialPivLU` by default, optionally `FullPivLU`, with SVD or perturbed normal-equation fallback only when configured.
- KL audit source: `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_safe_row_scaling` uses the row maximum with an absolute-row fallback for near-zero signed row maxima, then divides both the matrix rows and RHS by that factor.
- KL audit solve helpers use `numpy.linalg.solve` with `numpy.linalg.lstsq` fallback.

On the projected shared systems, the source-level scaling audit shows that scaling/conditioning remains the controlling reduced-system frontier, but not a production lever yet:

- FastChem scaling reproduces the FastChem projected solver result to `5.51e-12` at `45:-10` and `6.55e-12` at `60:-5`.
- Common scaling from the FastChem projected matrix gives the same near-roundoff reproduction.
- KL scaling no longer changes the exact solution when solving the same projected FastChem system in the local replay, but the earlier FastChem-vs-KL scaled-system comparison still exposes large solver-result residuals because the KL projected Jacobian/RHS rows differ materially.
- No-scaling is very ill-conditioned (`1.4484e18` and `1.8776e20`) but still reproduces the FastChem projected solve in this diagnostic.
- Fixed-scaling swaps under FastChem scaling show RHS swaps produce solver-result mean residuals `51.2` and `170.8`, while KL Jacobian swaps produce `779.2` and `1170.1`. This points to Jacobian/assembly under a fixed good scaling as the next detailed source audit, while the broader decision remains row scaling and conditioning because the residual is only visible after the labelled projection/scaling context is fixed.

No `actual_true_kl_atomic_branch_exact_iter1_row_scaling_proven` mode was added. The audit did not prove a single FastChem-equivalent scaling rule that closes the focused density mismatch without also resolving the projected assembly/Jacobian residual.

## Reduced Jacobian/RHS Block Assembly Audit Status

The latest audit keeps FastChem row scaling fixed and decomposes the projected shared iter1 reduced system by labelled row/column block. This confirms that row scaling itself is not the main fix: the next source-level blocker is reduced Jacobian assembly, with RHS differences secondary.

Block decomposition under the projected shared basis:

- Condensate-row / condensate-column: exact match in both layers.
- Condensate-row / element-column: exact match in both layers.
- Element-row / condensate-column: secondary mismatch. Frobenius difference is `2.9746e14` at `45:-10` and `2.6857e13` at `60:-5`.
- Element-row / element-column: dominant mismatch. Frobenius difference is `9.0946e18` at `45:-10` and `1.2651e18` at `60:-5`.
- Removed-condensate analytic branch rows are absent in this projected shared iter1 block, so no removed-row block fix is proven here.

RHS differences are real but smaller under the same fixed-scaling interpretation:

- Retained condensate RHS mean absolute difference is `3.6489` at `45:-10` and `15.6021` at `60:-5`.
- Free-element RHS mean absolute difference is `2.2779e14` at `45:-10` and `3.3561e13` at `60:-5`.

Fixed FastChem-scaling block replay:

- `FC J + FC RHS` closes the projected solve to `5.51e-12` and `6.55e-12`.
- `FC J + KL RHS` gives moderate solver-result residuals `51.20` and `170.80`.
- `KL J + FC RHS` gives large residuals `779.20` and `1170.12`.
- Replacing only the KL element-row / element-column block with the FC block gives the largest improvement: residuals fall to `500.24` and `625.56`.
- Replacing only that block in the opposite direction is the largest degradation.

No `actual_true_kl_atomic_branch_exact_iter1_jacobian_block_proven` mode was added. The element-row / element-column block is the dominant labelled assembly term, but block replay does not close the focused density mismatch by itself. The next audit should isolate the source of that block difference: atom diagonal, molecular stoichiometric outer products, removed-condensate fold-in, element basis/order, and old element-density values.

## Element-Element Jacobian Subterm Audit Status

The latest audit splits the dominant element-row / element-column block into source-labelled subterms:

- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`.
- KL audit source: `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- Atom diagonal term: `J_ii += n_i`.
- Molecule outer-product term: `J_ij += sum_m nu_mi * nu_mj * n_m`.
- Removed-condensate fold-in term: `J_ij += sum_r nu_ri * nu_rj * n_r / lambda_r`.

The molecule outer-product term is the dominant current source-level mismatch:

- Layer `45:-10`: total element-element difference Frobenius norm is `9.0946e18`; molecule outer-product difference is `9.0946e18`, atom diagonal difference is `9.1798e11`, and removed-condensate fold-in difference is `7.6350e13`.
- Layer `60:-5`: total element-element difference Frobenius norm is `1.2651e18`; molecule outer-product difference is `1.2651e18`, atom diagonal difference is `7.3695e10`, and removed-condensate fold-in difference is `0`.
- Focused high-residual entries `H/H`, `C/H`, `H/C`, `H/O`, and `O/H` are all dominated by the molecule outer-product term in both layers.

Physical-density and variable-basis diagnostics are now printed. The physical scalar is `4.9159e18` at `45:-10` and `6.8382e17` at `60:-5`, but a simple `J_phys = J_raw * S` global scaling identity does not close the block (`relative_difference` about `0.815` and `0.9998`). The current classification is therefore not a production scalar patch; it is a molecule outer-product physical-density/basis mismatch that still needs an exact source-level rule.

Subterm replay under fixed FastChem scaling is diagnostic only:

- Inserting the FastChem molecule outer-product term into the KL element-element block improves solver-result residual from `779.20` to `499.65` at `45:-10`, and from `1170.12` to `625.70` at `60:-5`.
- Replacing the full element-element block gives similar residuals (`500.24`, `625.56`), confirming the molecule term controls the block.
- Focused density residuals do not close, so no guarded `actual_true_kl_atomic_branch_exact_iter1_element_element_jacobian_subterm_proven` mode was added.

## Molecule Outer-Product Density Provenance Status

The latest audit decomposes the molecule outer-product term molecule by molecule:

```text
J_mol[j,k] = sum_m nu_mj * nu_mk * n_m
```

FastChem and KL now both emit the molecule density used in this term:

- FastChem source: `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`, traced through `diagnostic_trace::append_iter1_full_reduced_system`.
- KL source: `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.

The top residual entries are explained by a small set of gas molecules:

- `H/H`: dominated by `H2` in both layers.
- `C/H` and `H/C`: dominated by `C1H4`.
- `H/O` and `O/H`: dominated by `H2O1`, then `Fe1H2O2` and `H2Mg1O2`.

Both sides close their own molecule-density formulas to roundoff:

- FastChem closure max abs log error: `2.13e-14` at `45:-10`, `5.68e-14` at `60:-5`.
- KL closure max abs log error: `5.68e-14` at both focused layers.

The audit therefore proves that the mismatch is not a missing density field. It is a provenance mismatch between the FastChem molecule-density state and the KL molecule-density state entering the same source-level formula.

Candidate transforms were tested without changing production behavior:

- Current KL molecule density remains the baseline residual.
- Raw species amount to physical number density worsens the molecule block.
- Physical atomic old densities plus KL hvector worsens the molecule block.
- The gas molecule density-gauge bridge as tested does not change the residual materially.
- FastChem old atomic densities plus KL hvector worsens the block.
- KL old atomic densities plus FastChem-implied hvector worsens the block.
- Direct FastChem traced molecule-density replay closes the molecule outer-product residual by construction, but only improves the fixed-scaling solver-result residual to the known subterm-replay level and does not close focused density.

Per-molecule classification remains mixed in both layers:

- `molecule_density_gauge_mismatch=25`
- `molecule_lnk_hvector_mismatch=18`
- `molecule_atomic_donor_old_state_mismatch=6`

No `actual_true_kl_atomic_branch_exact_iter1_molecule_outer_product_density_proven` mode was added. The next exact lever is not yet proven; it requires isolating the gas molecule-density state handoff, hvector convention, or old atomic donor state that FastChem uses before iter1 condensed reduced Newton assembly.

## Molecule Factorization and Post-Jmol Replay Status

The follow-up factorization audit keeps the same frozen seed/update semantics and uses the labelled projected shared systems under fixed FastChem scaling. It adds molecule-by-molecule factorization, source provenance, stage-lag diagnostics, top-k replay, and post-`J_mol` residual attribution.

Top molecule factorization:

- `H/H` is controlled by `H2`: residual fraction `0.99598` at `45:-10` and `0.99601` at `60:-5`.
- `C/H` and `H/C` are controlled by `C1H4`: residual fraction is effectively `1.0` in both focused layers.
- `H/O` and `O/H` are controlled mostly by `H2O1`, with `Fe1H2O2` and `H2Mg1O2` secondary.
- The per-molecule decomposition remains mixed: top rows include large `lnK/hvector`, old atomic donor, and unexplained/gauge residual components, with no single transform generalizing across both layers.

Source provenance for the top gas `logK.dat` records is now resolved by the later mass-action/hvector audit. FastChem traces `Molecule::number_density`, `Molecule::stoichometric_vector`, `Molecule::mass_action_constant`, `Molecule::mass_action_coeff`, and the originating gas `logK.dat` species/coefficient lines into the iter1 `J_mol` record. The KL audit emits the matching source record and the hvector entry used by `_assemble_fastchem_reduced_update`. Gas molecule records in this path have no temperature segment.

Stage-lag attribution is also trace-limited. The current FastChem molecule-density trace is available at iter1 `J_mol` assembly, but not at gas solve exit, post initial gas coherent state, iter0 pre-`correctValues`, or iter0 post-`correctValues`. The KL molecule-density trace is available at iter1 reduced assembly, but not at gas-only final, physical donor converted state, or iter0 post update in the same molecule-density record.

Candidate molecule-density replays remain negative:

- Direct FastChem traced molecule replay closes `J_mol` residual to `0`, but solver-result residual remains `499.65` / `625.70` and focused iter1 log-density mismatch remains `1.5367` / `2.2604`.
- Top-molecule-only replay for `H2`, `C1H4`, or `H2O1` does not generalize and often worsens the solver-result residual.
- Cumulative top-k replays through the first `10` molecule contributions keep about `86.4%` of the `J_mol` residual and worsen the solver-result residual in both layers.
- Raw amount to physical species density, physical old atoms plus KL hvector, gas density-gauge bridge, FastChem old atoms plus KL hvector, and KL old atoms plus FastChem-implied hvector remain non-closing diagnostics.

After `J_mol` replay, the molecule outer-product residual is closed by construction, but the residual does not disappear. The remaining attribution is mixed across broader reduced-system blocks:

- RHS maximum absolute row-type differences remain large (`3.7478e15` at `45:-10`, `5.3635e14` at `60:-5`).
- Element-row / condensate-column Frobenius differences remain (`2.9746e14`, `2.6857e13`).
- Projection/row-universe, retained/removed mapping, and conditioning remain unchanged by the molecule-only replay.
- Full FastChem Jacobian replay still closes the projected linear solve to roundoff, so `J_mol` is only one source-level blocker inside a larger reduced-assembly mismatch.

No `actual_true_kl_atomic_branch_exact_iter1_molecule_density_rule_proven` mode was added. A direct traced-density replay is diagnostic only; it is not a source-proven reconstruction rule and does not close the focused iter1 density residual.

## Cached Molecule-Vector Audit Status

The cached molecule-vector audit tests the source-level distinction between FastChem and KL:

- FastChem `CondPhaseSolver::assembleJacobian` consumes cached `molecules[m].number_density` directly in `J_mol[j,k] += nu_mj * nu_mk * molecules[m].number_density`.
- KL `_assemble_fastchem_reduced_update` recomputes `mol = exp(A_mol.T @ u - hmol)` during reduced-system assembly.

FastChem cached-stage trace is now available for the top molecule contributors. Using raw record sequence around the same iter1 `assembleJacobian` block, the cached vector is stable from the iter0 molecule refresh through iter1 Jacobian assembly:

- `H2`, `C1H4`, `H2O1`, `H3N1`, `H4Si1`, `Fe1H2O2`, and the other top contributors have matching `ln(number_density)` at `after_iter0_calcNumberDensity_refresh`, `immediately_before_iter1_newtonStep`, and `inside_iter1_assembleJacobian_J_mol_cached_vector`.
- The initial apparent stage mismatch was an aggregate-stage artifact; the corrected audit uses raw sequence proximity and no longer supports a stale cached-vector conclusion.

`Molecule::calcNumberDensity` is located at `fastchem/fastchem_src/gas_phase/molecule_struct.cpp::Molecule<double_type>::calcNumberDensity`:

```text
number_density = exp(mass_action_constant + sum_i stoichiometric_vector[i] * log(elements[i].number_density))
```

It uses the full `elements` vector supplied by `CondensedPhase::calculate`, uses `mass_action_constant` directly, and does not apply `checkN`, floor, or cap inside `calcNumberDensity`; `checkN` is a separate method in the same source file.

Top-molecule reconstruction results:

- FastChem cached molecule density closes to FastChem full element logs plus FastChem mass-action constant at roundoff: representative closure errors are `0` to `~4e-14`.
- KL current molecule density remains far from the FastChem cached value: examples include `H2` errors `116.719` and `143.521`, `C1H4` errors `104.138` and `121.990`, and `H2O1` errors `44.597` and `32.119` in log units.
- FastChem full element logs plus KL hvector does not close; KL element logs plus FastChem mass-action constant also does not close. This separates the cached-vector issue from a simple hvector-only or cache-staleness explanation.
- Classification across the top molecule rows is dominated by `full_element_vector_mismatch` (`47` rows in each focused layer, with `2` mixed/unresolved).

Cached-vs-recomputed replay remains diagnostic only:

- Direct FastChem cached/traced molecule replay closes `J_mol` residual to `0`, but leaves solver-result residuals `499.65` / `625.70` and focused log-density mismatch `1.5367` / `2.2604`.
- FastChem full-element reconstruction plus FastChem hvector reduces the `J_mol` residual only to about `86.4%` of baseline and worsens the solver result.
- FastChem full-element reconstruction plus KL hvector and KL atoms plus FastChem hvector both worsen badly.
- Top-molecule-only and cumulative top-k replays do not generalize across both layers.

No `actual_true_kl_atomic_branch_exact_iter1_cached_molecule_vector_proven` mode was added. The cached vector is now source-traced, but a production or guarded rule is not proven because direct replay still leaves the focused density residual. The later source audit shows full element-vector mismatch alone is not sufficient; the top-molecule frontier is the mass-action/hvector gauge convention plus the reduced-system state used with it.

## Full Element-Vector / Mass-Action Audit Status

The follow-up full-vector audit records the exact element logs used by FastChem at the iter0 molecule refresh and compares them with the KL iter1 element log vector used by `J_mol` reconstruction. The focused elements `H`, `C`, `O`, `Mg`, `Fe`, `Si`, `Cl`, `F`, `K`, and `Al` are present on both sides as reduced/free element rows, but their log-density offsets are large: about `42-44` log units at `45:-10` and `40-42` at `60:-5`.

Top molecule log-error factorization does not close from the full-element offsets alone:

- `H2`: `delta_ln_n_m` is `116.719` / `143.521`; `delta_k + 2*delta_u_H` predicts only `93.077` / `111.231`.
- `C1H4`: `delta_ln_n_m` is `104.138` / `121.990`; `delta_k + delta_u_C + 4*delta_u_H` predicts only `33.247` / `20.946`.
- `H2O1`: `delta_ln_n_m` is `44.597` / `32.119`; `delta_k + 2*delta_u_H + delta_u_O` predicts `-105.336` / `-192.886`.

The mass-action/hvector audit classifies all `49` top molecule rows in each focused layer as `mass_action_hvector_gauge_mismatch`. Representative FastChem-minus-KL-equivalent mass-action residuals are:

- layer `45:-10`: `H2 = 7.418`, `C1H4 = -180.119`, `H2O1 = -234.733`
- layer `60:-5`: `H2 = 29.160`, `C1H4 = -183.553`, `H2O1 = -316.531`

Full-vector reconstruction candidates remain diagnostic-only:

- Current KL `J_mol` residual is `9.0946e18` / `1.2651e18`.
- FastChem traced cached molecule replay closes `J_mol` to `0`, but leaves solver-result mismatch `499.65` / `625.70` and focused log-density mismatch `1.5367` / `2.2604`.
- FastChem full element vector plus KL hvector explodes the molecule residual to `~2e46`.
- FastChem full element vector plus traced FastChem mass-action constants does not become a KL solver rule because the available KL state reconstruction still fails candidate-wide, even though the top-molecule source-record/no-segment mapping is now proven.

No `actual_true_kl_atomic_branch_exact_iter1_full_element_vector_for_jmol_proven` mode was added. The exact FastChem full element vector is now traced, but a KL reconstruction rule is not proven; full element-vector mismatch alone does not close the top molecule errors.

## Mass-Action / Hvector Source Provenance Status

The latest entrance-smoke audit adds source-record provenance on both sides without changing production solvers or presets:

- FastChem parser provenance: `fastchem/fastchem_src/gas_phase/init_read_files.cpp::GasPhase::readSpeciesData` records the `logK.dat` species line and coefficient line for each gas molecule.
- FastChem evaluator provenance: `fastchem/fastchem_src/gas_phase/molecule_struct.cpp::Molecule<double_type>::calcMassActionConstant` records the temperature, raw evaluated natural-log `logK`, pressure/density correction, and final `Molecule::mass_action_constant`.
- KL parser provenance: `src/exogibbs/presets/fastchem.py::_parse_fastchem_coeffs_with_metadata` records the same gas `logK.dat` species line, coefficient line, and coefficients, while preserving the existing parser behavior.
- KL evaluator provenance: `src/exogibbs/presets/fastchem.py::logk`, `hvector_func`, and the audit `_assemble_fastchem_reduced_update` now expose the source-record raw `logK(T)`, `hvector`, `-hvector`, and the reduced-update hvector entry mismatch.

Gas `logK.dat` records in this path have no temperature segments; the selected "segment" is the single 5-coefficient gas species record. For top molecules `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2`, FastChem and KL source records match in both focused layers.

The source-proven relationship is:

```text
k_FC = raw_logK_source(T) + (sum_nu - 1) * ln(1e-6 * k_B * T)
```

With KL source convention `h_source = -raw_logK_source(T)`, this is equivalently:

```text
k_FC = -h_source - (sum_nu - 1) * ln(1e6 / (k_B * T))
```

This source-derived formula has residuals at roundoff for the top molecules across `45:-10` and `60:-5` (`<=6e-14`). The earlier current-hvector candidates remain non-closing because they use the KL reduced-update hvector entry, not the source-record convention:

- Candidate A, `k_FC = -h_KL`, leaves residuals such as `H2=7.418/29.160`, `C1H4=-180.119/-183.553`, and `H2O1=-234.733/-316.531`.
- Candidate B and C density-bridge signs do not generalize across the same molecule set.
- Candidate D, treating the source value as base-10 and multiplying by `ln(10)`, is not supported by the FastChem source path.
- Candidate E, the source-derived FastChem formula above, closes the common-convention comparison.

Top molecule factorization with the source-proven `k` term proves that the current `mass_action_hvector_gauge_mismatch` classification is real, but it is not a standalone solver fix. Reconstructing `J_mol` with the source-proven conversion and current KL/full-vector candidates worsens the molecule residual and focused iter1 density in the focused smoke. Direct FastChem cached molecule replay still closes `J_mol` by construction, but leaves the projected solver residual at `499.65` / `625.70` and focused density residuals high.

No `actual_true_kl_atomic_branch_exact_iter1_mass_action_hvector_proven` mode was added. The source formula is proven, but it does not materially close `J_mol` under the available KL state reconstruction or improve focused iter1 density across both layers. Production implication: keep production solvers and presets unchanged; the proven rule should only be used as audit provenance until the hvector entry, full element vector, and reduced-system state are source-aligned together.

## KL Hvector Plumbing Audit Status

The earlier hvector-plumbing smoke was superseded by the corrected molecule-column attribution audit. The physical source facts remain:

- `src/exogibbs/presets/fastchem.py::logk` evaluates the source gas `logK.dat` record as `L_m(T)`.
- `src/exogibbs/presets/fastchem.py::hvector_func` defines the source convention `h_source = -L_m(T)`.
- `src/exogibbs/optimize/pipm_rgie_cond.py::gas_molecule_density_gauge_bridge` can compute `(sigma_m - 1) * ln(1e6/(k_B*T))`, but this bridge is not applied to `h_current` in the iter1 audit path.
- `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update` receives the copied `gas_setup.hvector_func(state.temperature)` vector and uses `h_current = hgas_np[n_elem:]` in `mol = exp(A_mol.T @ u - hmol)`.

The source-proven density-basis hvector remains:

```text
h_needed = h_source + (sigma_m - 1) * ln(1e6/(k_B*T))
```

The corrected-column audit shows the prior `current_is_unknown_convention` classification was not physical. It was caused by attributing molecule columns with `gas_setup.species[29:]`, while molecule columns begin at `n_elem=28`.

After using `gas_molecule_species_by_col = gas_setup.species[n_elem:]`, `H2`, `C1H4`, `H2O1`, `Fe1H2O2`, and `H2Mg1O2` all classify as `current_is_source_hvector` with `h_current-h_source=0` in both `45:-10` and `60:-5`. The source-proven `h_needed` remains diagnostic-only:

- Current/source `J_mol` residuals remain `9.0946e18` and `1.2651e18`.
- `h_needed` does not reduce `J_mol` or improve focused iter1 density.
- Wrong-sign bridge worsens.
- FastChem cached molecule replay closes `J_mol` to `0` as an upper-bound diagnostic, but leaves iter1 density residuals `1.5367` and `2.2604`.

No `actual_true_kl_atomic_branch_exact_iter1_hvector_plumbing_proven` mode was added. There is no safe hvector conversion: it neither reduces `J_mol` across both layers nor improves focused iter1 density across both layers. Production implication remains unchanged: do not alter production hvector semantics or solver assembly from this audit.

## Molecule-Column Label Alignment Audit Status

The follow-up entrance-smoke audit proves that the prior `h_current` unknown convention for the required top molecules was caused by molecule-column label misalignment in the audit attribution path.

The audit now keeps two species lists separate:

- `gas_species_compare = gas_setup.species[29:]` remains reserved for the pyfastchem VMR comparison path.
- `gas_molecule_species_by_col = gas_setup.species[n_elem:]` is used only for `_assemble_fastchem_reduced_update` molecule records, `J_mol` provenance, hvector plumbing, and molecule-density factorization.

Focused setup facts:

- `len(gas_setup.elements)=28`
- `n_elem=formula_matrix_gas.shape[0]=28`
- `len(gas_setup.species)=523`
- `gas_setup.species[28:38]` starts with `Al1Cl1`, while `gas_setup.species[29:39]` starts with `Al1Cl1F1`, proving the old comparison list is shifted by one for molecule-column attribution.

Required top molecule invariant checks pass in both focused layers:

- `C1H4`: column `94`, old shifted label `C1H4O2`, corrected `h_current-h_source=0`.
- `Fe1H2O2`: column `309`, old shifted label `Fe1O1`, corrected `h_current-h_source=0`.
- `H2`: column `331`, old shifted label `H2K2O2`, corrected `h_current-h_source=0`.
- `H2Mg1O2`: column `333`, old shifted label `H2N1`, corrected `h_current-h_source=0`.
- `H2O1`: column `337`, old shifted label `H2O2`, corrected `h_current-h_source=0`.

After corrected labels, all five top molecules in both layers classify as `current_is_source_hvector`. The source-proven density-basis candidate remains `h_needed = h_source + (sigma_m-1)*ln(1e6/(k_B*T))`, but testing it does not reduce `J_mol` or improve focused iter1 density. Current aligned `h_current` and `h_source` have the same `J_mol` residuals (`9.0946e18`, `1.2651e18`), while FastChem cached molecule replay closes `J_mol` by construction but leaves focused density mismatch (`1.5367`, `2.2604`).

The audit cannot claim strict all-species column/name stoichiometry proof: 18 alias/suffix gas species are not parseable by the local formula parser, and `F2Si1` has a formula-matrix/name discrepancy. This blocks a blanket all-column proof, but it does not block top-molecule hvector attribution.

No guarded `actual_true_kl_atomic_branch_exact_iter1_hvector_plumbing_proven` mode was added. Production implication remains unchanged: do not alter production hvectors, presets, solvers, gauge bridges, row scaling, row universe, or `correctValues`. The only established change is audit attribution: `J_mol` molecule names must come from `gas_setup.species[n_elem:]`.

## Post-Jmol Residual Block Attribution

The corrected-label post-`J_mol` entrance smoke compares the current KL reduced system, FastChem cached `J_mol` replay, full `J_ee`, full element-row/element-column block, RHS-only, element-row/condensate-column-only, partition/mapping availability, and full FastChem Jacobian replay under fixed FastChem scaling.

Layer `45:-10`:

- Current KL solver-result mismatch: `779.20`, `J_mol` residual `9.0946e18`.
- FastChem cached `J_mol` replay: solver-result mismatch `499.65`, `J_mol` residual `0`, focused log-n mismatch `1.5367`.
- Full FastChem `J_ee` and full element-row/element-column replay: `500.24`, effectively unchanged from cached `J_mol`.
- RHS-only: `779.20`, no improvement.
- Element-row/condensate-column block only: `593.19`, partial movement.
- Full FastChem Jacobian upper bound: `5.51e-12`.
- FastChem Jacobian with KL RHS diagnostic: `51.20`.

Layer `60:-5`:

- Current KL solver-result mismatch: `1170.12`, `J_mol` residual `1.2651e18`.
- FastChem cached `J_mol` replay: solver-result mismatch `625.70`, `J_mol` residual `0`, focused log-n mismatch `2.2604`.
- Full FastChem `J_ee` and full element-row/element-column replay: `625.56`, effectively unchanged from cached `J_mol`.
- RHS-only: `1170.12`, no improvement.
- Element-row/condensate-column block only: `951.83`, partial movement.
- Full FastChem Jacobian upper bound: `6.55e-12`.
- FastChem Jacobian with KL RHS diagnostic: `170.80`.

Classification: after `J_mol` is closed, the next dominant attributable block is the element-row / condensate-column Jacobian. This is an audit attribution, not a production rule. No `actual_true_kl_atomic_branch_exact_iter1_post_jmol_next_block_proven` mode was added because no exact source rule for that block was proven to improve focused iter1 density across both layers.

## Element-Row / Condensate-Column J_ec Provenance

The `J_ec` follow-up keeps the corrected molecule labels and fixed FastChem scaling, and adds a source-level decomposition for every matched projected element-row / condensate-column entry.

Source formulas:

- FastChem trace: `J_ec_FC[j,c] = stoich_cj * number_densities[condensate]` from `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::assembleJacobian`.
- KL audit: `J_ec_KL[j,c] = A_jac[j,c] * n_cond[active_local]` from `examples/comparisons/audit_actual_fastchem_gas_phase_transplant_phase2.py::_assemble_fastchem_reduced_update`.
- The traced sign convention is positive `stoich*n_old` in the reduced Jacobian.

Entry-level decomposition:

- Layer `45:-10`: `2161` matched entries classify as `old_condensate_density_mismatch`; `83` classify as `cap_or_maxDensity_state_mismatch`.
- Layer `60:-5`: `2552` matched entries classify as `old_condensate_density_mismatch`; `116` classify as `cap_or_maxDensity_state_mismatch`.
- The largest entries are explained by `stoich_shared*(n_old_FC-n_old_KL)` with negligible unexplained residual. Examples include `O/SiO2(s,l)` and `O/MgCO3(s)` at layer 45, and `O/Fe(OH)3(s)`, `O/Fe(OH)2(s)`, and `O/SiO2(s,l)` at layer 60.
- Aggregates are dominated by volatile element rows, especially `O` and `H`. Focused condensate rows such as `MgCO3(s)` and `SiC(s)` show the same density-driven entry closure when present in the projected shared system.

Old-density provenance:

- Layer `45:-10`: retained-condensate old-density rows split into `67` `inherited_from_iter0_update` and `35` `cap_or_maxDensity_mismatch`.
- Layer `60:-5`: retained-condensate old-density rows split into `69` `inherited_from_iter0_update` and `47` `cap_or_maxDensity_mismatch`.
- Required FastChem stage fields are still missing from the `condensed_phase_iter1_full_reduced_system` trace: post-seed condensate `n`, post-iter0 condensate `n` separated from the old density consumed by `assembleJacobian`, and FastChem-side `max_number_density`/cap/tau for the same retained columns.

Fixed-scaling replay diagnostics:

- Layer `45:-10`: current KL solver-result mismatch is `779.20`; FastChem cached `J_mol` only gives `499.65`; FastChem `J_ec` only gives `595.18` but worsens focused log-n mismatch from `1.5367` to `2.1344`; FastChem `J_mol+J_ec` gives solver mismatch `4.35` but leaves focused log-n mismatch at `1.5367`; full FastChem Jacobian upper bound closes the projected linear solve to `5.51e-12`.
- Layer `60:-5`: current KL solver-result mismatch is `1170.12`; FastChem cached `J_mol` only gives `625.70`; FastChem `J_ec` only gives `954.24` and improves focused log-n mismatch from `2.5021` to `1.7909`; FastChem `J_mol+J_ec` gives solver mismatch `0.77` but leaves focused log-n mismatch at `2.2604`; full FastChem Jacobian upper bound closes the projected linear solve to `6.55e-12`.
- FastChem old `n_c` in `J_ec` with KL stoichiometry/mapping reproduces the full `J_ec` block replay, while FastChem stoichiometry/mapping with KL `n_old` does not move the residual. That makes old condensate density the entry-level formula driver, but not a proven production handoff rule.

No `actual_true_kl_atomic_branch_exact_iter1_element_condensate_jacobian_proven` mode was added. The exact entry formula is source-attributed, but the old-density handoff remains mixed between inherited iter0 update and cap/maxDensity state, with missing FastChem stage fields. The replay also does not improve focused iter1 density across both layers.

## Old Condensate State Handoff

The old-state handoff smoke widens FastChem `correctValues_rule` tracing from focused condensates to all retained/Jac shared condensates and adds full-reduced arrays for FastChem `max_number_density`, `tau`, and `log_tau`. The audit then builds a matched ladder:

- immediate seed `n` and lambda
- post-iter0 `correctValues` `n` and lambda
- iter1 old `n` consumed by `assembleJacobian`
- iter1 old `n`, lambda, and tau consumed by `correctValues`
- iter0 delta before/after clipping, uncapped `n`, capped `n`, maxDensity, and cap status

No requested retained-column FastChem old-state field is missing in the focused smoke after this trace update.

Ladder classification:

- Layer `45:-10`: all `102` shared retained columns classify as `inherited_from_seed_or_maxDensity_mismatch`.
- Layer `60:-5`: all `116` shared retained columns classify as `inherited_from_seed_or_maxDensity_mismatch`.
- The `J_ec` entry mismatch remains old-density driven: `2161`/`2552` entries classify as `old_condensate_density_mismatch`, with `83`/`116` cap/maxDensity-state entries.
- Old-density provenance now collapses to `cap_or_maxDensity_mismatch` for `102`/`116` retained columns.

Coherent old-state replay diagnostics under fixed FastChem scaling:

- Layer `45:-10`: current focused log-n mismatch is `1.7620`; `J_mol` only remains `1.7620`; `J_ec` only worsens to `2.2103`; FC old `n_c` in both `J_ec` and `correctValues` worsens to `2.2474`; adding FC lambda and tau is unchanged; FC old state + FC `J_ec` + FC `J_mol` is `1.9536`.
- Layer `60:-5`: current focused log-n mismatch is `1.9461`; `J_mol` only improves to `1.7581`; `J_ec` only improves to `1.3929`; FC old `n_c` in both `J_ec` and `correctValues` improves to `1.2958`; adding FC lambda and tau is unchanged; FC old state + FC `J_ec` + FC `J_mol` is `1.7079`.
- Solver-result mismatch can become small when `J_mol+J_ec` are replayed (`4.35`/`0.77`), but the focused density closure still does not generalize across the two layers.

Minimal rule decision: old condensate state handoff is dominated by iter0 cap/maxDensity mismatch, but no transplantable rule is proven because the coherent old-state replay helps only layer 60 and worsens layer 45. No `actual_true_kl_atomic_branch_exact_iter1_old_condensate_state_handoff_proven` mode was added.

## Iter0 Cap/MaxDensity Provenance

The follow-up iter0 cap audit decomposes every shared retained/Jac column with the branch equations:

- both capped: `log n_old` residual equals the `log maxDensity` residual
- neither capped: residual equals `log n_seed` residual plus clipped-delta residual
- FastChem-only capped: residual equals `log maxDensity_FC - (log n_seed_KL + delta_clip_KL)`
- KL-only capped: residual equals `(log n_seed_FC + delta_clip_FC) - log maxDensity_KL`

The prior Eq.13 total-inventory maxDensity source propagates into the iter0 cap state for all shared retained rows in the focused smoke:

- Layer `45:-10`: `102/102` shared retained rows have the proven total-inventory maxDensity in the cap state.
- Layer `60:-5`: `116/116` shared retained rows have the proven total-inventory maxDensity in the cap state.

The remaining branch provenance is mixed rather than a single cap rule:

- Layer `45:-10`: `70` rows classify as `retained_carryover_state_mismatch`, `27` as `both_capped_maxDensity_value_mismatch`, and `5` as `kl_only_capped`.
- Layer `60:-5`: `72` rows classify as `retained_carryover_state_mismatch`, `42` as `both_capped_maxDensity_value_mismatch`, and `2` as `kl_only_capped`.
- MaxDensity source/stage rows split between limiting-element and maxDensity-value classifications: `80/22` at layer 45 and `86/30` at layer 60.

Cap/maxDensity replay diagnostics remain non-promotable:

- Layer `45:-10`: current focused log-n mismatch is `1.7620`; FC maxDensity-only, FC seed, FC delta, FC seed+delta+cap, and FC post-iter0 old-state replays all worsen the focused mean (`2.0987` to `2.9489` depending on candidate).
- Layer `60:-5`: FC delta and FC seed+delta+cap improve focused log-n (`1.9461 -> 1.2958` best), but the improvement does not generalize to layer 45.

No `actual_true_kl_atomic_branch_exact_iter0_cap_maxdensity_handoff_proven` mode was added. The minimal rule decision is `no promotable rule`: the branch table is mixed across retained/carryover, capped maxDensity value, and one-sided cap rows.

## Coherent Iter0-to-Iter1 Active-State Transition

The coherent-transition follow-up moves beyond per-row cap truth tables. FastChem globally scales the reduced result before `correctValues`, updates retained and removed rows, refreshes element densities, activity, and molecule densities, and then carries the refreshed state into the next reduced Newton step. FastChem RHS element rows also include all active condensates, not just retained/Jac columns, plus removed-condensate fold-in terms.

The new audit records full active-state rows at iter0 pre reduced Newton, iter0 post `correctValues`, nearest post-refresh iter1 pre state, iter1 pre partition, and iter1 pre reduced Newton. It also computes all-active burden `B_cond[j]=sum_active nu_cj*n_c`, removed fold-in `F_rem[j,k]=sum_removed nu_rj*nu_rk*n_r/lambda_r`, and global result-scaling factors.

Focused smoke result:

- All-active burden residuals are large: `5.22e15` at `45:-10` and `7.37e14` at `60:-5`.
- The largest all-active burden contributor in both layers is KL-only `CH4(s,l)` (`5.40e15` and `7.51e14` norms).
- Removed fold-in is not a two-layer explanation: layer `45:-10` has one FastChem-only removed row and fold-in residual `7.63e13`, while layer `60:-5` has no removed rows and zero fold-in residual.
- Global scaling differs at the raw-result level. Iter1 FastChem factors are `4.54e-4` and `1.50e-4`; KL iter1 factors are `2.25e-4` and `1.37e-4`.
- Coherent full-state proxies remain layer-mixed: layer `45:-10` focused log-n worsens from `1.7620` to `2.2474`, while layer `60:-5` improves from `1.9461` to `1.2958`.
- `J_mol+J_ec` replay nearly closes the projected linear solve but still does not close focused density. Full FastChem Jacobian replay closes the projected solve, but focused `correctValues` density remains controlled by the coherent old state and transition path.
- A distinct FastChem post-`correctValues` refreshed all-active snapshot is still missing from `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate`; the audit uses the iter1 pre reduced state as the nearest labelled snapshot.

No `actual_true_kl_atomic_branch_exact_iter0_to_iter1_coherent_transition_proven` mode was added. The minimal rule decision is `coherent transition remains mixed or inconclusive`.

## CH4 Burden Lifecycle Isolation

The CH4 follow-up traces `CH4(s,l)` and phase aliases through both state machines. In FastChem, `CH4(s,l)` appears as a candidate row in the initial activity/maxDensity scan, but it is not selected into the active set, is not seeded, and does not participate in iter1 all-active burden, RHS, retained/Jac columns, removed rows, or `correctValues`. In KL, the same species persists as a retained/Jac active row through iter0 and iter1.

Classification:

- FastChem exclusion: `candidate_but_not_selected_active`
- KL persistence: `missing_fastchem_eviction_rule`

Burden isolation:

- Layer `45:-10`: current all-active burden residual norm `5.2216e15`; without CH4 `6.7658e14`; without all KL-only rows `6.7947e14`. CH4 removes `87.0%` of the norm.
- Layer `60:-5`: current all-active burden residual norm `7.3689e14`; without CH4 `4.7811e13`; without all KL-only rows `4.8557e13`. CH4 removes `93.5%` of the norm.
- H and C residuals are mostly CH4 (`95.7%`/`98.4%` at layer 45, `97.7%`/`98.6%` at layer 60). O residual is unchanged by CH4 removal.

FastChem iter0 raw scaling is now exposed by the Python extractor from `post_reduced_newton_step_result`:

- Layer `45:-10`: FastChem iter0 raw max `217.4001`, factor `2.2999e-2`; KL iter0 raw max `119.6398`, factor `4.1792e-2`.
- Layer `60:-5`: FastChem iter0 raw max `303.9086`, factor `1.6452e-2`; KL iter0 raw max `198.6454`, factor `2.5170e-2`.

Replay result:

- Removing CH4 from the projected all-active RHS burden does not change focused log-n in either layer (`1.7620` and `1.9461`).
- Removing all KL-only active rows from the burden also does not change focused log-n.
- The `J_mol+J_ec` upper-bound with CH4 removal improves only layer `60:-5` (`1.7581`) and leaves layer `45:-10` unchanged (`1.7620`).

No `actual_true_kl_atomic_branch_exact_iter1_ch4_or_burden_lifecycle_proven` mode was added. CH4 dominates the burden magnitude, but burden replay does not move the focused density target across both layers.

## Exact Post-CorrectValues Refreshed Snapshot

The exact FastChem refreshed all-active snapshot is now instrumented in `fastchem/fastchem_src/condensed_phase/calculate.cpp::CondensedPhase::calculate` at the boundary after:

- `correctValues` updates `cond_densities_new`, `elem_densities_new`, and `activity_corr_new`.
- `elements_cond[i]->number_density` is refreshed from `elem_densities_new`.
- Active condensates run `calcActivity`.
- Molecules run `calcNumberDensity(elements)`.

The trace is emitted before objective evaluation, old-state assignment, and the next reduced Newton setup. It records all active condensate rows, refreshed element densities, refreshed molecule densities, and exact all-active burden.

Focused smoke result:

- The exact snapshot is available for both focused layers: `103` FastChem active rows at `45:-10` and `116` at `60:-5`.
- It has the same FastChem active row universe as the previous iter1 pre-reduced proxy, and `CH4(s,l)` is absent from both FastChem snapshots while present in the KL closest proxy.
- The exact snapshot is not numerically equivalent to the old proxy. Exact-vs-proxy mean absolute residuals are:
  - Layer `45:-10`: log-n `29.6791`, log-lambda `10.5754`, log-activity `123.1413`.
  - Layer `60:-5`: log-n `29.8670`, log-lambda `10.6496`, log-activity `176.5168`.
- The exact all-active burden residual versus KL is larger than the old proxy:
  - Layer `45:-10`: exact `6.1122e15`, old proxy `5.2216e15`.
  - Layer `60:-5`: exact `9.0194e14`, old proxy `7.3689e14`.
- CH4 remains a real lifecycle mismatch but no longer explains the exact refreshed burden by itself:
  - Layer `45:-10`: removing CH4 removes `62.1%` of the exact burden residual norm.
  - Layer `60:-5`: removing CH4 removes `56.9%` of the exact burden residual norm.
- Exact refreshed-state replay remains layer-mixed:
  - Layer `45:-10`: current focused log-n `1.7620`; exact full condensate state `2.2474`; exact full state plus `J_mol+J_ec` `1.9536`.
  - Layer `60:-5`: current focused log-n `1.9461`; exact full condensate state `1.2958`; exact full state plus `J_mol+J_ec` `1.7079`.
- Full FastChem Jacobian replay still closes the projected linear solve (`5.51e-12` and `6.55e-12` solver mismatch), but focused density remains controlled by state/update semantics rather than a single refreshed-state handoff.

No `actual_true_kl_atomic_branch_exact_iter0_post_refresh_state_proven` mode was added. The minimal rule decision remains `still mixed/inconclusive`, with final decision `coherent transition remains mixed after exact refreshed-state trace`.

## Promotion Criteria

Current repaired alpha/beta broad replay status:

- Direct numeric broad replay is implemented diagnostically for `30:-10`,
  `45:-10`, `60:-5`, `75:-5`, and `90:-5`.
- Canonical FastChem e-first vector construction passes all broad cases.
- The evaluator uses the convention-safe molecule RHS builder and explicitly
  records that no legacy KL-reference burden-ratio conversion is used.
- Focused reference values remain validated in the focused tail compact, but
  the direct broad regression gate fails under the embedded broad selected-row
  mapping because that mapping is a different 10-row diagnostic projection.
- Full-vector outside-selected molecule RHS residuals remain material.
- No production electron rule, guarded mode, selected-row rule, molecule RHS
  rule, inventory rule, removed-tail rule, row-scaling rule, or solver behavior
  is promotable from this result.

The current non-promotion decision is: repaired candidate remains mixed or
inconclusive.

Selected-row reconciliation update:

- The direct broad evaluator now distinguishes metric families explicitly.
  The focused regression uses `focused_raw_result_provenance_metric`; the
  embedded broad 10-row projection is reported as a separate diagnostic
  scorecard and is not used as the focused regression.
- Focused 45/60 references pass under the focused metric, but
  focused-compatible mappings are unavailable for `30:-10`, `75:-5`, and
  `90:-5` because the raw-result provenance compact is focused-only.
- The broad 10-row diagnostic projection remains non-closing and links to the
  material outside-selected/full-vector residual. This blocks promotion.

Updated non-promotion decision: focused-compatible broad mapping cannot be
constructed; broad generalization remains metric-inconclusive.

Current focused-compatible broad replay update:

- The direct broad evaluator now consumes the accepted
  `raw_result_provenance_broad_compact` mapping for all five broad cases and
  records `focused_raw_result_provenance_metric` against the fresh focused
  reference. The old focused artifact is stale historical context only.
- Direct projected focused metric lineage is available for all broad cases,
  but every current selected row is one-sided or mapping/index-only. Shared
  projected numeric row count is `0` for each case, so current fresh
  focused-compatible closure is not numerically meaningful for the full
  selected-row set.
- The stale `45:-10` and `60:-5` historical values are reported for audit
  continuity, but current fresh D/E/F values are undefined because the selected
  rows are one-sided. The old closure claim was metric-dependent and must not
  be promoted or transferred.
- The embedded broad 10-row projection remains separate as an
  outside-selected/full-vector residual probe and still fails closure. It is
  not a focused regression substitute.

Updated non-promotion decision: current focused-compatible metric is available
but not numerically projectable for one-sided rows; broad generalization
remains partially metric-inconclusive.

One-sided attribution update:

- Current focused-compatible selected rows are row-universe/mapping dominated:
  `12` selected rows, `0` shared numeric rows, `12` one-sided rows.
- Focused-layer `CH4(s,l)` rows trace to candidate/activity-threshold mismatch
  where source-stage artifacts are available. Broad-only layer rows require a
  regenerated broad row-universe/mapping lineage trace before their earliest
  stage can be source-proven.
- Repaired alpha/beta is therefore not the current production blocker. It is
  diagnostic replay state only; the next diagnostic frontier is
  row-universe/mapping provenance, especially candidate/activity threshold,
  active selection, partition/reset/update, and labelled reduced-system index
  lineage.

Updated non-promotion decision: repaired alpha/beta is irrelevant to the
current focused blocker; next blocker is row-universe/mapping provenance.

Deep one-sided lineage update:

- Label normalization was tested for label/mapping rows and does not remove the
  blocker; zero rows are explained as label-normalization artifacts.
- Current earliest divergence counts are mixed: `partition_split_mismatch=5`,
  `activity_threshold_crossing_mismatch=2`, `missing_trace=4`, and
  `mixed_or_unresolved=1`.
- Broad source-stage coverage is incomplete for candidate/active/reset tables.
  The next patch site is broad case-keyed row-universe lineage emission, not a
  production active-selection rule or electron/beta rule.

Updated non-promotion decision: next blocker is mixed row-universe/mapping
provenance.

Broad case-keyed lineage update:

- The one-sided attribution compact now emits broad case-keyed selected-row
  lineage tables for all five cases. Available selected-row evidence covers
  full-catalog identity, activity/maxDensity values from broad delta
  provenance, post-correctValues snapshot presence, partition labels,
  iter1 RHS/result indices, and labelled reduced-system row presence.
- The requested broad `candidate_set`, `active_set`,
  `post_selectActiveCondensates_reset`, and `partition_split_before` tables are
  still unavailable as full case-keyed row tables. The compact records exact
  missing source functions, artifacts, and patch sites instead of inferring the
  stage.
- Current earliest divergence counts remain `partition_split_mismatch=5`,
  `activity_threshold_crossing_mismatch=2`, `missing_trace=4`, and
  `mixed_or_unresolved=1`. The broad-only `CH4(s,l)` rows are unresolved
  because FastChem activity/candidate evidence and broad pre-partition active
  rows are missing.
- Reduced-system materialization mismatch is not conclusively proven, and
  repaired alpha/beta still cannot affect absent or unprojectable selected
  rows.

Updated non-promotion decision: one-sided row attribution remains unresolved
due to missing traces.

Broad full-row lineage update:

- Broad case-keyed full row tables are now emitted from existing diagnostic
  compacts for candidate set, active set, post-select reset, pre-partition,
  post-partition, and labelled reduced-system materialization. No C++ trace
  patch or pyfastchem rebuild was required.
- The former missing-trace rows are resolved. Current earliest divergence
  counts are `activity_threshold_crossing_mismatch=7` and
  `result_index_mapping_mismatch=5`.
- Broad-only `CH4(s,l)` rows now classify as value-driven threshold crossings:
  FastChem log activity is `-10` with threshold fail, while KL log activity is
  positive with threshold pass.
- The former partition rows are no longer earliest partition blockers; the full
  tables prove their split was already determined by earlier
  candidate/activity-threshold membership.
- Full FastChem labelled reduced-system arrays are present. KL rows/columns can
  be materialized from split indices, but exact KL labelled RHS/Jacobian
  row/column arrays, row scaling, and solver result vectors by label are still
  not emitted. That prevents a materialization-mismatch promotion.

Updated non-promotion decision: next blocker is mixed row-universe/mapping
provenance.

A production change is eligible only after a smoke proves:

1. FastChem and KL old states entering iter 1 match.
2. FastChem and KL retained/removed partitions match.
3. Reduced RHS and compact Jacobian checksums/row norms match.
4. Row scaling and solver result vector match.
5. Four-way closure proves whether `FC_old + FC_delta`, `KL_old + KL_delta`, `KL_old + FC_delta`, or `FC_old + KL_delta` accounts for the residual.
6. Old log activity decomposition identifies whether the residual is donor refresh, lnK, floor/alignment, or stale activity.
7. Any old-state handoff replay closes the focused density residual before a guarded audit mode is added.
8. The same rule survives at least one additional cheap trace beyond the entrance smoke.
9. KL-only candidate rows are either source-proven to enter FastChem or excluded by a source-proven FastChem-equivalent row-universe rule.
10. Any projection/isomorphism rule must be paired with a source-proven row-scaling rule and must close focused iter1 density residuals before promotion.
11. Any row-scaling rule must be tested under fixed RHS/Jacobian swaps so that scaling is not conflated with reduced assembly differences or solver backend details.
12. Any Jacobian block transplant must first decompose the element-row / element-column block into atom, molecule, and removed-condensate fold-in terms and close the focused iter1 residual in audit mode.
13. Any element-element subterm transplant must prove the exact molecule-density gauge or variable-basis rule before changing the KL audit path, and must close focused iter1 density residuals without altering frozen seed/update semantics.
14. Any molecule outer-product density actualization must prove a single source-level reconstruction rule for `n_m`; direct traced density replay is not enough because it closes the subterm by construction but leaves focused iter1 density residuals.
15. Any mass-action/hvector conversion must use source-record provenance and must improve `J_mol` plus focused iter1 density across both focused layers before a guarded audit mode is added.
16. Any KL hvector plumbing conversion must prove whether `h_current` is source, density-needed, wrong-sign, double-bridged, or another source convention before changing audit reconstruction; an unknown convention is not promotable.
17. Any molecule-column hvector attribution must use `gas_setup.species[n_elem:]` for reduced-update molecule columns; VMR comparison species slices are not valid molecule-column labels.
18. Any post-`J_mol` block actualization must prove the exact source rule for the next residual block and improve focused iter1 density across both focused layers; labelled block attribution alone is not promotable.
19. Any `J_ec` actualization must source-prove the old-condensate-density handoff, retained column mapping, sign/basis, and cap/maxDensity state consumed by the element-row / condensate-column block; entry-level `stoich*n_old` closure alone is not promotable.
20. Any old-condensate-state handoff actualization must improve focused iter1 density across both focused layers after replaying the same old `n_c`, lambda, tau, maxDensity, and cap state in both `J_ec` assembly and `correctValues`; layer-specific improvement is not promotable.
21. Any iter0 cap/maxDensity handoff actualization must prove one branch rule and improve focused iter1 density across both focused layers. The current truth table is mixed and the replay is layer-specific, so no cap/maxDensity mode is promotable.
22. Any coherent iter0-to-iter1 transition actualization must prove full active row universe, all-active RHS burden, removed fold-in, global result scaling, refreshed activity/molecule state, and partition mapping together, and must improve focused iter1 density across both focused layers.
23. Any CH4 or KL-only active-row lifecycle actualization must prove the FastChem exclusion source, reproduce all-active burden and RHS movement, and improve focused iter1 density across both focused layers; burden magnitude alone is not promotable.
24. Any exact post-`correctValues` refreshed-state actualization must improve focused iter1 density across both focused layers from the exact refreshed trace, not from the previous iter1 pre-reduced proxy. The current exact snapshot proves the proxy was stale, but the replay remains layer-mixed and is not promotable.
25. Any one-step delta actualization must first pass a compact delta-provenance audit that separates raw solver result, global result scaling, local clipping, solver-result-to-row mapping, removed-row analytic delta, and projected/focused row coverage. The current compact audit classifies the available selected rows as raw-result dominated (`89`) with mapping/index secondary (`3`), but standalone global-scaling and max-raw fields remain absent from the compact source rows, so the next production-relevant target is source-level raw-result/global-scaling trace completion, not a solver fix.
26. Any raw-solver-result actualization must first prove an exact labelled FC/KL reduced-system projection and isolate RHS assembly, Jacobian assembly, row universe/partition, mapping/vector index, or solver backend with a two-layer replay. The current compact raw-result audit has labelled J/RHS/result fields for both focused layers and rules out solver backend as primary, but row-level provenance remains mixed (`raw_result_Jacobian_dominated=45`, `raw_result_mixed_or_unresolved=41`, `raw_result_mapping_or_index_dominated=3`, `raw_result_requires_RHS_and_Jacobian=3`). Per-term RHS vectors and an isomorphic `J_mol` subterm projection are still missing, so no production solver, guarded mode, row-universe, RHS, Jacobian, or block-level rule is promotable.
27. Any Jacobian block or subterm actualization must improve the raw direction and focused one-step metric across both focused layers after projecting the subterm into the exact shared labelled row/column basis. The current compact subterm audit projects `J_atom`, `J_molecule`, `J_removed`, and `J_other` exactly enough for replay, with additivity errors at roundoff, but no single subblock is promotable: `J_ee`/molecule-only swaps reduce raw residual only slightly, `J_ec` worsens, and row-level attribution remains mostly `raw_result_not_explained_by_J_blocks=80`. RHS term vectors are still missing, so the residual after FC `J` + KL RHS is not RHS-term attributable yet.
28. Any coherent element-row or Schur-complement actualization must prove both the Jacobian-side coherent block and the remaining RHS/effective-RHS source terms. The current coherent audit shows that replacing `[J_ec,J_ee]` together exactly reproduces the full-FC-J + KL-RHS replay because `J_cc` and `J_ce` already match on the shared projection, but it still leaves raw residuals `22.539650451705914` and `258.23728172185525`. The Schur `C+D` replacement recovers the FC-like Schur matrix, while effective RHS still differs and separated RHS term vectors are missing. This is not promotable as a production block transplant.
29. Any RHS-term actualization after coherent Jacobian alignment must use labelled RHS term vectors that add back to `rhs_vector_before_scaling` and `rhs_vector_after_scaling` in both FastChem and KL. Diagnostic-only trace fields have now been added for FastChem and KL RHS term vectors, but the current entrance-smoke artifact predates them and reports the term containers as missing. Full-vector replay confirms the remaining residual is RHS-side, but termwise attribution is still unavailable; no RHS term, coherent block, Schur, or production rule is promotable until a regenerated entrance smoke closes RHS additivity and term sensitivity.
30. A regenerated RHS-term entrance smoke must close both unscaled and scaled RHS term additivity before any term sensitivity result is interpreted. The fresh artifact now exposes FastChem and KL condensate/element RHS term containers for both focused layers, and scaled reconstruction closes to roundoff (`1.42e-14` max error), but unscaled reconstruction fails (`295.0`/`1.0` at `45:-10`, `82.15625`/`0.03125` at `60:-5` for FastChem/KL). The full RHS vector swap still closes the residual after coherent Jacobian alignment, but term-level attribution remains blocked by diagnostic additivity. No RHS term, coherent block, Schur, production solver, guarded mode, row-universe, lifecycle, schedule, `J_mol`, or `J_ec` rule is promotable.
31. Scaled RHS term replay may be interpreted in solve space because `CondPhaseSolver::newtonStep` passes `scaling_factors` into `assembleRightHandSide` and `solveSystem` consumes the resulting scaled RHS. The scaled compact audit confirms scaled term additivity to roundoff and treats unscaled additivity failure as a bookkeeping/convention diagnostic. The scaled term replay still does not produce a promotable single RHS rule: molecule burden helps most at `45:-10`, log activity helps most at `60:-5`, all-active burden has the largest Schur effective norm but worsens direct replay, and cumulative closure requires multiple terms. Therefore no condensate-row RHS, total-inventory, atom, molecule-burden, all-active-burden, removed-correction, other-RHS, Schur-effective, production solver, guarded mode, row-universe, lifecycle, schedule, `J_mol`, or `J_ec` actualization is promotable.
32. Any RHS term interaction actualization must prove a smaller source-coherent rule, not just a numerical subset. The compact interaction audit confirms contribution-vector additivity to roundoff (`6.05e-12` and `1.93e-11` max raw-result-space errors), and it explains the Schur/direct replay split: all-active burden is anti-aligned with the full residual even though it has the largest Schur effective norm. The minimal common improving pair is `log_activity + all_active_condensate_burden`, but it leaves raw residuals `3.0603` and `2.5286`. The minimal common closing subset is the full nonzero RHS state, closing to `8.55e-12` and `1.31e-11`. No single RHS term, pair, condensate-row family, element-row family, burden family, removed/burden family, guarded mode, or production solver rule is promotable; full coherent RHS state remains diagnostic-only.
33. Any RHS source-state actualization must prove a smaller physically meaningful source-state group, not only full-state replay. The compact source-state provenance audit reconstructs scaled RHS terms from emitted/inferred source variables to roundoff (`1.42e-14`/`1.42e-14` at `45:-10`, `2.84e-14`/`1.42e-14` at `60:-5` for FastChem/KL), and classifies the RHS formulas as equivalent given the same state for terms with complete source variables. The remaining blocker is source-state handoff, not formula mismatch. `molecule_state` is the minimal source group that improves both layers but it does not close; `condensate_activity + active_condensate_burden_state` leaves residuals `3.0603` and `2.5286`; only the full coherent RHS source state closes both layers. No condensate activity, complementarity, inventory/atom, molecule, active-burden, removed-correction, mixed pair, guarded mode, or production solver rule is promotable.
34. Any RHS source-state handoff actualization must prove one earliest divergence, a physically meaningful KL production rule, and two-layer replay improvement. The compact lineage audit shows RHS formulas are equivalent and the source-state mismatch is distributed across upstream stages: condensate activity, complementarity, and active-row universe/burden state already diverge before candidate selection; removed-correction becomes material at RHS assembly / retained-removed split; inventory/atom and molecule-state lineage are unresolved before RHS assembly because symmetric total-density/epsilon, atom-density, full-element-vector, mass-action, and molecule-cache handoff fields are missing. Current KL RHS assembly state leaves residuals `22.53965045170672` and `258.2372817218515`; FC full RHS assembly state closes to `8.556197725637401e-12` and `1.3137289236262653e-11`. No active-selection, post-refresh, partition, molecule-cache, hvector, RHS assembly, guarded mode, or production solver rule is promotable; the next frontier is symmetric KL post-refresh/pre-RHS lineage and expanded element source vectors.
35. Any preselection or missing-field closure actualization must prove that one decomposed source component both explains the distributed lineage and improves both focused layers. The compact preselection closure audit covers all `186` condensates per layer and shows full-catalog row materialization matches, while value divergence dominates: layer `45:-10` has `139` log-activity value mismatches, `34` donor mismatches, `7` lnK/hvector mismatches, and `6` threshold mismatches; layer `60:-5` has `90`, `70`, `17`, and `9`. Pre-seed complementarity is not meaningful and is reclassified to entry seed. Active-burden universe counts differ at candidate/selected/RHS-active stages, but n_c value dominance is blocked by missing per-condensate burden expansion. Inventory/atom and molecule upstream splits remain unresolved before RHS assembly despite RHS-term availability because symmetric per-stage total-density/epsilon, atom density, full element vectors, molecule-cache refresh vectors, and mass-action constants are missing. Removed correction remains a later partition/RHS-assembly issue. No preselection row-universe, donor, lnK/hvector, threshold, active-selection, inventory/atom, molecule-cache, removed-partition, guarded mode, or production solver rule is promotable.
36. Any preselection activity-value or active-burden actualization must prove a single source-level rule that improves both focused layers, not only a counterfactual reconstruction. The compact activity-value audit closes `log_activity = lnK_final_density_basis + donor_sum` for all `186` paired condensates on both sides (`1.14e-13`/`4.55e-13` FastChem max residuals and `0.0` KL max residuals), so activity formula mismatch is ruled out. Single-side counterfactuals do not improve candidate agreement: FC lnK with KL donor gives Jaccard `0.617`/`0.689`, while KL lnK with FC donor gives `0.565`/`0.635`; only full FC lnK plus full FC donor recovers the FastChem candidate set by construction. Active-burden exact set decomposition is also mixed: selected/RHS-active shared-value norms are `4.064e15`/`5.580e14`, while KL-only universe norms are `5.400e15`/`7.522e14`, and all active-burden replay variants worsen the raw residuals. No donor, lnK/hvector, density-gauge, threshold/candidate, active-burden universe, shared-row value, guarded mode, or production solver rule is promotable.
37. Any candidate-to-active selection actualization must prove an independent source rule beyond the activity threshold and must improve both focused layers. The compact selection audit shows FastChem `selectActiveCondensates` selects rows with `log_activity >= 0` and `is_calculated == false`; no phase-rank, linear-dependence, replacement, or eviction trace is available. KL likewise exposes candidate and selected-active rows as a `log_activity >= 0` candidate mask with no separate compact eviction rule. In both focused layers candidate rows equal selected-active rows on each side: `103`/`103` FastChem and `109`/`109` KL at `45:-10`, `116`/`116` FastChem and `125`/`125` KL at `60:-5`. All selected-active mismatches are KL-only candidate-threshold mismatches (`6` and `9` rows), and there are zero selection-only mismatches after candidate agreement. Replacing KL selected-active rows with FastChem selected-active rows improves set Jaccard to `1.0` but worsens raw residuals to `95.8163` and `362.5233`. No candidate threshold, selected-row transplant, missing-eviction, phase/rank, lifecycle, guarded mode, or production solver rule is promotable.
38. Any coherent activity-state plus active-burden actualization must improve both focused layers and leave only a source-proven smaller residual class before promotion. The compact pair audit proves that the pair is real but still not promotable. `FC log_activity` alone gives raw residuals `74.1879` and `101.0823`; `FC all_active_condensate_burden` alone gives `94.2256` and `359.5332`; KL-only-row-only edits are neutral or worse. The coherent pair `FC log_activity + FC all_active_condensate_burden` is the known useful mixed replay and reduces the raw residuals to `3.0603` and `2.5286`, but only the full coherent RHS source state closes (`8.48e-12`, `1.28e-11`). The remaining residual therefore moves to other RHS source groups, including molecule, inventory/atom, removed-correction, and condensate complementarity. No activity-only, burden-only, KL-only-row, selected-row, CH4-only, coherent-pair, guarded mode, or production solver rule is promotable.
39. Any post-activity-burden actualization must prove a smaller remaining source-state rule after taking `baseline_AB = KL RHS + FC log_activity + FC all_active_condensate_burden` as fixed. The compact post-activity-burden audit verifies the baseline residuals (`3.0603000036721433`, `2.5285823085810524`) and then replays the remaining source groups. Condensate complementarity is the smallest group that improves both layers, reducing the residual to `0.5620` and `0.9142`, but it does not close. Molecule-only, inventory/atom-only, and removed-only do not improve both layers, and molecule plus inventory/atom still leaves `2.9376` and `1.9877`. `complementarity + molecule + inventory_atom` closes `60:-5` but leaves `45:-10` at `0.08476`. The minimal common closing subset is the full remaining quartet `complementarity + molecule + inventory_atom + removed`, which closes to `8.20e-12` and `1.30e-11`. Condensate complementarity is strongly anti-aligned with the residual after `baseline_AB`, while molecule and inventory/atom are almost perfectly cancelling against each other, so no smaller production rule survives both layers. Therefore no post-activity-burden complementarity, molecule, inventory/atom, removed-correction, mixed subset, guarded mode, or production solver actualization is promotable; the residual requires the full remaining RHS source state.
40. Any complementarity-specific actualization after `baseline_AB` must prove one source-proven subcomponent, formula self-closure on comparable rows, and two-layer closure without borrowing the later molecule/inventory cancellation. The compact complementarity-provenance audit shows `activity_correction` is neutral, `log_tau` is the strongest single improving subcomponent (`2.0848`, `0.9991`), `-log_n` is weaker, and `-log_lambda` is neutral/worse. Full emitted complementarity is the smallest common improving group (`0.5620`, `0.9142`), but source reconstruction does not self-close on comparable rows and cross-state comparison is blocked by row mapping / missing iter1 RHS-entry rows (`Al(s)`, `CH4(s,l)`) plus missing KL exact post-`correctValues` condensate fields. Molecule-only and inventory-only replays each worsen sharply after full complementarity, while the paired `molecule + inventory_atom` replay is still required for closure. Therefore no complementarity subcomponent, no standalone `log_tau` rule, no full complementarity transplant, no guarded mode, and no production solver actualization is promotable.
41. Any complementarity source-closure or tau-lineage actualization must use exact iter1 RHS-entry source variables and identify the earliest common exact divergence before promotion. The compact source-closure audit closes the FastChem and KL complementarity formulas to roundoff from exact RHS-entry `n_old`, `activity_correction_old`, `tau`, and `log_tau` values, so formula mismatch is ruled out. Source-clean replay confirms `log_tau` is the strongest single complementarity subcomponent (`0.86793` and `0.99908` residuals), but it still does not close both layers. The dominant exact provenance class is `tau_seed_rule_mismatch` (`102/110` comparable rows at `45:-10`, `116/125` at `60:-5`), while the residual after clean complementarity still requires the later `molecule + inventory_atom` cancellation and a layer-45 removed-correction tail. Therefore no tau-seed rule, `log_tau` rule, complementarity rule, guarded mode, or production solver actualization is promotable; the source-clean result is diagnostic provenance only.
42. Any tau-seed actualization must prove the exact seed formula components and show that one physically meaningful subset closes both layers. The compact tau-seed audit now proves the upstream formulas directly: FastChem uses `cond_tau * total_element_density * epsilon(reference_element)` with `cond_tau = 1e-15`, while KL uses `tau_scale * exp(epsilon)` with `tau_scale = 1.0`. FastChem seed closure is exact to roundoff after reconstructing the layer-specific total element density from the traced tau and the source formula. The dominant mapped-row class is support/reference-element driven (`102` rows at `45:-10`, `116` at `60:-5`), but scalar-only, total-density-only, and reference-element-only replays all fail to close. Full FC tau helps (`0.86793`, `0.99908`) yet still does not close without later complementarity completion, molecule/inventory cancellation, and the layer-45 removed tail. Therefore the tau source rule is proven but not promotable: no tau-seed rule, no `cond_tau` transplant, no total-density transplant, no epsilon-element rule, no guarded mode, and no production solver actualization is justified.
43. Any post-complementarity tail actualization must prove a smaller physically meaningful rule after fixing `baseline_ABC = KL RHS + FC log_activity + FC all_active_condensate_burden + FC full complementarity`. The compact tail audit verifies `baseline_ABC` at `0.5620237706783301` and `0.914249889596486`, then shows that molecule-only and inventory/atom-only replays are each destructive while the paired `molecule + inventory_atom` replay closes `60:-5` and reduces `45:-10` to `0.08475871276818528`. Adding removed correction closes `45:-10` to `8.199347373900107e-12` and leaves `60:-5` at `1.2971603589242228e-11`. Molecule and inventory/atom are an inseparable cancellation pair after complementarity (cosines `-0.9997438633784276` / `-0.9997471855058516`, cancellation indices `0.9886099574272349` / `0.9870268652530958`), and removed correction is only a layer-45 tail. Molecule-burden aggregate self-closure succeeds from `molecule_density_provenance`, atom-density closure is exact, total-inventory remains coupled to shared gas-diagnostic source fields, and removed per-condensate contribution vectors are still not emitted. Therefore the post-complementarity tail is source-proven but diagnostic-only: no molecule, inventory/atom, removed-correction, guarded-mode, or production solver rule is promotable.
44. Any molecule/inventory or removed-tail promotion after the post-complementarity audit must prove source closure from independently emitted RHS-entry source variables, not just from term replay. The compact molecule/inventory/removed source audit shows that scaled molecule burden closes exactly from molecule caches and atom density closes exactly from the iter1 element state, but independent total-inventory source closure fails: `total_inventory_by_element = total_element_density * epsilon` from the emitted gas diagnostics does not reconstruct the RHS total-inventory term, and side-specific RHS-entry total-density / epsilon sources are not emitted. As a result, strict source-clean inventory replays blow up (`6.40e4` / `8.65e4` raw residual scales), and the molecule/inventory pair must be treated as requiring the full coherent gas-state bundle rather than a promotable source rule. The layer-45 removed tail also remains unresolved because the exact FastChem per-removed RHS contribution rule is not emitted separately enough to self-close. Therefore no molecule-cache, full-element-vector, mass-action, total-inventory, atom, removed-tail, guarded-mode, or production solver rule is promotable from this audit.
45. Any KL-native molecule timing actualization must first pass a FastChem reconstructed-cache replay gate and then close from an exposed KL stage snapshot. The compact timing-resolution audit passes the FastChem gate: reconstructed FastChem molecule cache equals direct cached density to roundoff at both the post-iter0 refresh and iter1 RHS/Jacobian entry, and `D_clean` now equals `K_direct`. Exposed KL cached-stage reconstructions from `gas_only_final`, `post_initial_activity_maxdensity_scan`, and `post_selectActiveCondensates_reset` remain destructive after gauge-normalized inventory/atom (`36.907067382036104` at `45:-10`, about `50.4194244237` at `60:-5`), while later KL post-correctValues / pre-partition stage vectors are not emitted and KL computes molecule cache inline at RHS/Jacobian assembly. Therefore the FastChem cached molecule snapshot remains diagnostic-only, and no KL molecule-cache timing, hvector/mass-action, density-gauge, hidden-snapshot, guarded-mode, or production solver rule is promotable.
46. The later KL molecule snapshot audit removes the missing-field blocker for `post_correctValues_update`, exact post-refresh, `iter1_pre_partition`, and `iter1_RHS_assembly_entry`, but it does not produce a production rule. The field gate passes on both focused layers and the FastChem reconstructed-cache gate remains reconciled (`D_clean == K_direct`). However all later KL snapshots replay at the KL inline RHS residuals (`36.907067382036104` at `45:-10`, `50.419424423672645` at `60:-5`) and fail to recover the molecule/inventory cancellation. Clean/direct FC molecule remains the closing diagnostic path (`0.08475871276711094` plus exact removed to `9.230739629452324e-12` at `45:-10`, `1.3724132941206335e-11` at `60:-5`). Therefore KL inline RHS recomputation is the best KL-native exposed stage but still insufficient; no molecule timing, hidden snapshot, inventory, removed-tail, guarded-mode, or production solver rule is promotable.
47. Any KL later-stage molecule cache carry must first prove that the carried cache is numerically distinct from inline RHS recomputation and moves toward the FastChem cached molecule snapshot. The compact distinctness audit shows the opposite: `post_correctValues_update`, exact post-refresh, `iter1_pre_partition`, and `iter1_RHS_assembly_entry` are identical to tolerance on both focused layers, with zero pairwise difference in `u`, atom/full-element vector, and molecule cache. Frozen-cache carry variants exactly reproduce the inline residuals (`36.907067382036104` at `45:-10`, `50.419424423672645` at `60:-5`), remain far from FastChem cache (max log residuals `66.90247419848673` and `210.94452675909793`), and do not recover cancellation. Therefore exposed KL later stages collapse to one effective molecule snapshot; no freeze-and-carry, timing, hvector, density-gauge, inventory, removed-tail, guarded-mode, or production solver rule is promotable.
48. Any KL gas-refresh molecule snapshot actualization must prove that a refreshed KL atomic state both approaches the FastChem cached molecule state and improves selected-row replay after gauge-normalized inventory/atom. The compact gas-refresh audit finds distinct diagnostic snapshots (`post_gas_recoupling_atomic_element_species_state` and the proxy `gas_replay_final_atomic_element_species_state`), but they still fail. At `45:-10`, gas-refresh carries remain at `36.907067382036104`; at `60:-5`, they are `50.419424423951064`, slightly worse than inline. They also remain far from FastChem RHS-entry cache (`72.80738647765477` max log residual at `45:-10`, `156.7200546573116` at `60:-5`). The requested iter1-pre-partition gas-recompute proxy cannot be constructed because `fastchem_target_donor_replay_from_gas_replay_final` contains a non-finite `e-` entry. Therefore KL gas-refresh snapshots are materially distinct but still insufficient; no gas-refresh, coupled-loop, hidden-snapshot, molecule-cache, inventory, removed-tail, guarded-mode, or production solver rule is promotable.
49. Any molecule-cache input-vector actualization must show that a KL atomic/full-element input vector can reproduce the FastChem cached molecule state when the source-clean FastChem mass-action constants are reused. The compact input-vector audit shows it cannot. KL gas-only, post-selection, later collapsed, gas-recoupling, and RHS-entry input vectors remain tens of log units away from the FastChem molecule-cache input vector, and hybrid replay with direct FastChem mass-action constants still leaves the same destructive residuals (`36.90706` scale at `45:-10`, `50.41942` scale at `60:-5`). FC input-vector reconstruction reproduces the direct FC molecule boundary (`0.08475871276711094` plus removed closure at `45:-10`, `1.3724132941206335e-11` at `60:-5`). Top input-vector residuals are dominated by FastChem fixed-by-condensation elements (`75/90` top rows at `45:-10`, `73/90` at `60:-5`), with `e-` as a large non-fixed residual. Therefore the molecule cache is a downstream symptom of fixed-element/input-vector handoff mismatch; no molecule timing, gas-refresh, fixed-element, inventory, removed-tail, guarded-mode, or production solver rule is promotable.
50. Any fixed-element handoff actualization must isolate one handoff component and improve both focused layers without borrowing an unproven KL solver rule. The compact fixed-element handoff audit decomposes the FastChem cached full-element vector by source, emits KL analogue vectors for all exposed candidate stages, and splits residuals by the FastChem fixed mask. The earliest exposed fixed-subset divergence is already present before `post_selectActiveCondensates_reset` on both layers. Fixed-element-only value substitution strongly improves molecule reconstruction (`147.92838359730413 -> 15.226673958408794` at `45:-10`, `140.48218322609824 -> 19.407748535067963` at `60:-5`), while non-fixed-only and electron-only substitutions do not. Adding FC electron on top of FC fixed values improves molecule residual further (`0.4347374576125725`, `0.565493860944129`) but still does not yield a coherent closing RHS replay, and no KL mask-consumption or separable `phi` / `degree_of_condensation` correction field is emitted at the molecule input-vector stage. Full FC input vector / direct FC molecule replay remains the only molecule-side upper bound (`0.08475871276711094` and `1.3724132941206335e-11` selected-row residuals before any removed-tail addition). Therefore the source-proven result is narrow: molecule cache is downstream symptom only, the dominant unresolved object is fixed-element values themselves, and the handoff remains diagnostic-only. No fixed-element value rule, fixed-mask rule, `phi` / degree rule, electron rule, guarded mode, or production solver rule is promotable.
51. Any overwrite-source actualization at the cached fixed-element boundary must prove additive source components before promotion. The latest compact overwrite-source audit keeps the same FastChem boundary, but resolves the overwrite provenance one step further: fixed-by-condensation rows are full overwrites from local `elem_densities_new[i]` inside `CondensedPhase::calculate`, with the focused reduced branch pointing to `CondensedPhase::correctValues`. Overwrite-only replay is still the decisive single-component improvement (`147.92838359730413 -> 15.226673958408794` and `140.48218322609824 -> 19.407748535067963`), while electron-only replay remains near baseline (`147.5263302999384`, `138.4526607344315`). However additive source components are still not emitted cleanly: `free_atomic_gas_component`, `condensed_or_fixed_correction_component`, `total_inventory_component`, and `electron_specific_component` remain unavailable at that boundary, and the exact prior carry-forward value local `full_element_densities_before_write[i]` is not yet surfaced in the compact artifact. Overwrite plus electron still fails the selected-row replay (`64072.83656065602`, `86511.80029512591`), and only full FC cached input / direct FC replay closes. Therefore overwrite source is further resolved but remains diagnostic-only; no overwrite-source, condensed/fixed correction, total-inventory, electron, guarded-mode, or production solver rule is promotable.
52. Any KL materialization-boundary actualization must prove that the gas-recoupling output is the later molecule-input vector before any fixed-element handoff rule is promoted. The latest compact materialization-boundary audit shows the opposite. The diagnostic gas replay emits a true post-recoupling atomic state from `actual_fastchem_like_coupled_loop`, but the later iter1 molecule input is rebuilt inline inside `_assemble_fastchem_reduced_update`. On both focused layers the gas-recoupling output and iter1 molecule input differ materially (`u` max differences `14.87958673430007` and `8.370219832642732`), while the pre-gas current-state proxy `post_correctValues_update` matches the iter1 molecule input exactly. The iter1 cache is source-marked `cache_is_computed_inline = true` and `cache_is_carried_from_earlier_stage = false`, so the best current classification is that the molecule input is recomputed from a different current-`u` state. Direct gas-recoupling adoption remains diagnostic-only and does not close (`149.20929529868224` and `140.07824601545863` molecule mean residuals), and the fixed-mask consumer at the molecule-input side is still missing. Therefore the missing object is now explicit: a KL fixed-element materialization boundary exists and remains unresolved for promotion. No gas-recoupling adoption, materialization, fixed-mask, guarded-mode, or production solver rule is promotable.
53. Any exact same-iteration fixed-row subspace actualization must first emit labelled matrix rows from the exact `CondPhaseSolver::newtonStep` iteration that feeds `correctValues`. The current compact exact-subspace audit adds diagnostic-only trace emission for `exact_same_iteration_fixed_row_reduced_system`, but the rebuilt entrance-smoke trace still contains no `condensed_phase_exact_fixed_row_reduced_system` records on layers `45:-10` or `60:-5`. The missing local object is `jacobian` in `fastchem/fastchem_src/condensed_phase/solver.cpp::CondPhaseSolver::newtonStep`, with exact same-iteration row and column labels. Because the matrix trace gate fails, fixed-row closure, coherent subspace condition/coupling/Schur diagnostics, coherent subspace replay, and FastChem-vs-KL subspace comparison are not interpretable. No fallback to the older labelled analogue is promotable. Therefore no fixed-row RHS, fixed-row Jacobian, outside-coupling, Schur-complement, full-system, guarded-mode, or production solver rule is promotable; exact fixed-row solve-space provenance remains diagnostic-only and inconclusive.
54. The fixed-row trace repair must pass the requested phase2 entrance-smoke raw field-presence gate before any coherent subspace replay is interpreted. A diagnostic-only v2 probe and compact exact fixed-row matrix emitter were added at the post-`solveSystem` point in `CondPhaseSolver::newtonStep`, where local `jacobian`, local `rhs`, output `result`, and output `scaling_factors` are all available. A clean rebuild/import identity check proves that the repository extension can emit `condensed_phase_exact_fixed_row_reduced_system_probe` and `condensed_phase_exact_fixed_row_reduced_system` with marker `exact_fixed_row_subspace_trace_v2` for the focused layer inputs. However the requested phase2 entrance smoke still emits only the older `condensed_phase_iter1_full_reduced_system` records in its raw traces and does not emit the v2 probe or exact matrix records. This is classified as a build/import path mismatch for the requested smoke, not as a physical solver result. No subspace replay was run, no fallback labelled analogue is allowed, and no fixed-row RHS, Jacobian, outside-coupling, Schur-complement, full-system, guarded-mode, or production solver rule is promotable.
55. The fixed-row import repair resolves that build/import-path blocker for the requested phase2 smoke, but it still does not create a production rule. Both candidate shared objects were cleaned, the extension was rebuilt, and the rebuilt root object was copied into `fastchem/python`; both paths now share checksum `7493746969d5ce442beaa2339e8dc4e35d7d7446ef3f006038b2012623d126ee`. The phase2 audit process loaded `/home/kawahara/exogibbs/fastchem/python/pyfastchem.cpython-310-x86_64-linux-gnu.so`, and the fresh raw traces contain the legacy iter1 full-system marker plus v2 probe and exact fixed-row reduced-system records for both focused layers. The gate passes with non-empty fixed rows-by-all-columns, all-rows-by-fixed-columns, fixed RHS, solver result, and row/column labels. The subsequent compact replay proves fixed-row equation closure to roundoff (`7.28e-16` and `1.26e-15` max relative error), but no coherent subspace S0-S5 closes or improves enough to promote a two-layer rule. Exact fixed-row matrix availability is now a diagnostic prerequisite, not a solver change. No fixed-row RHS, fixed-row Jacobian, outside-coupling, Schur-complement, full-system coherence, guarded-mode, or production solver actualization is promotable.
56. Any fixed-row subspace molecule-source actualization must be tested inside the already-proven post-complementarity molecule/inventory tail context before promotion. The tail-context reconciliation audit reproduces the known ladder in one selected-row code path: `baseline_ABC` is `0.5620237706782278` and `0.9142498895939201`; direct FC molecule alone is destructive; direct FC inventory/atom alone is destructive; direct FC molecule plus gauge-normalized inventory/atom closes layer `60:-5` and nearly closes `45:-10`; adding exact removed closes layer `45:-10`. The catastrophic exact-subspace `I/J` result is explained as a source-artifact mismatch: that path reconstructs molecule burden from the FC cached input vector, while the closing tail context uses the exact emitted FC molecule RHS term. Inside the proven tail context, S0-S5 fixed-row subspace molecule sources remain at the KL-current residual scale (`36.907...` and `50.419...`) and do not recover the molecule/inventory cancellation. Therefore no fixed-row subspace molecule source, cached-input reconstruction, molecule/inventory pair, removed-tail, row-scaling, guarded-mode, or production solver actualization is promotable.
57. Any molecule RHS artifact actualization must distinguish the emitted solve-space RHS term from diagnostic source-space burden reconstruction. The molecule RHS artifact reconciliation audit shows that the exact emitted FastChem `element_rhs_terms.molecule_burden` is source-proven and reconstructable to roundoff from the enriched FastChem molecule cache and cached full-element vectors when the reconstruction is converted directly into the FastChem scaled RHS convention. Direct reconstructions from the cached full-element vector, post-`correctValues` vector, and iter1 RHS-entry molecule cache match the emitted term with max element-RHS residuals near `1e-14`/`1e-15`. The catastrophic cached-input replay is reproduced only by the legacy KL-reference burden-ratio conversion, which is a source-space versus solve-space convention mismatch and not a molecule-density, stoichiometry, sign, row-scaling, or row-mapping failure. This reconciles the source artifact but still does not promote a KL rule: the closing object is the exact emitted FC solve-space molecule RHS term plus gauge-normalized inventory/atom and the layer-45 removed tail. No molecule RHS transplant, cached-input reconstruction, hvector/density-gauge rule, fixed-row subspace rule, guarded mode, or production solver change is promotable.
58. Any convention-safe fixed-row subspace molecule replay must use the FastChem scaled solve-space molecule RHS builder, not the legacy KL-reference burden-ratio conversion. The convention-safe compact validates the builder `-sum_m stoich[j,m] * n_molecule[m] / row_scaling[j]` against the exact emitted FastChem molecule RHS term to roundoff on both layers. The negative-control legacy conversion remains catastrophic, proving the prior source-space versus solve-space mismatch. With the convention fixed, full FC cached-input reconstruction and direct FC molecule replay now close in the known molecule+inventory context, but S0-S5 remain at the KL-current residual scale (`36.907...` and `50.419...`) and do not recover cancellation. Therefore the convention bug is resolved diagnostically, but no fixed-row subspace molecule source, molecule RHS transplant, cached-input rule, row-scaling rule, guarded mode, or production solver actualization is promotable.
59. Any full-element cached-vector actualization for the molecule RHS must prove a small source-coherent subset, not just replay the full FC vector. The subset compact shows the smallest audited useful subset is broad: all FastChem fixed-by-condensation elements plus the FastChem electron. Fixed-only with KL electron is catastrophic, electron-only is insufficient, and non-fixed-only is indistinguishable from KL current. Fixed+electron recovers the full FC cached-input molecule RHS behavior (`0.0847587` at `45:-10` before the known removed tail and `3.15e-10` at `60:-5`), while top-k RHS/log residual subsets up to `k=15` do not recover the cancellation. This makes electron an essential secondary term, but not a standalone or sufficient production rule. No fixed-only, electron-only, fixed+electron, non-fixed, top-k, cached-vector, molecule RHS, guarded-mode, or production solver actualization is promotable; layer 45 still requires exact removed correction after molecule+inventory.
60. Any fixed+electron molecule-group actualization must separate charged electron sensitivity from neutral molecule recovery before promotion. The charged/neutral compact keeps the convention-safe solve-space builder and shows that the selected-row recovery is carried by neutral molecules: neutral fixed+electron gives `0.08475871276713746` at `45:-10` before the exact removed tail and `3.14866355211052e-10` at `60:-5`, while charged-only fixed+electron remains at the KL-current scale. Charged positive molecules are still the source of the catastrophic fixed-only plus KL-electron failure, so electron is a required secondary state to suppress the charged-ion term, but no exposed KL electron source or small molecule-support subset materializes the FC fixed+electron bundle. Layer `45:-10` removed replay pairs with the neutral molecule term; charged-only plus removed does not close. Therefore no charged-molecule, neutral-molecule, electron, fixed+electron, fixed-support subset, removed-tail, guarded-mode, or production solver actualization is promotable.
61. Any KL-native fixed+electron materialization actualization must expose a same-boundary KL fixed bundle and an electron source that jointly reproduce the FastChem fixed+electron molecule RHS bundle. The compact materialization audit shows no such boundary. Current/post-`correctValues`/RHS-entry KL fixed values with KL electron remain at the KL-current residual scale, gas-recouping fixed plus gas-recouping electron remains non-closing, and gas-recouping fixed plus the FC electron is still non-closing. FC fixed values plus the best KL-native electron candidate are catastrophic (`2.143e10` and `1.222e9`), so electron materialization is also unresolved. The reduced-Newton overwrite-derived fixed candidates are unavailable because `KL.reduced_Newton_overwrite_derived_fixed_full_element_vector` is not emitted from `_assemble_fastchem_reduced_update / reduced_update_diagnostics`. Only the FC fixed+electron bundle reaches the known upper bound, with layer 45 still needing exact removed. Therefore no KL-native fixed materialization, electron materialization, gas-recouping adoption, overwrite-derived fixed vector, fixed+electron bundle, removed-tail, guarded-mode, or production solver actualization is promotable.
62. The reduced-Newton overwrite-derived fixed-vector compact resolves the missing fixed-side candidate without changing production behavior. The vector is materialized diagnostically from `CondensedPhase::correctValues` write-site fields for all fixed elements (`22` / `23` rows). Paired with FC electron, it reproduces the FC fixed+electron upper bound (`0.08475871276713746` before removed at `45:-10`, `3.14866355211052e-10` at `60:-5`), and exact removed closes layer 45. Paired with KL current/post/RHS-entry electron it remains catastrophic, and paired with gas-recouping electron it improves but remains non-closing (`2.143e10` / `1.222e9`). This proves the fixed side can be materialized from the reduced-Newton write-site chain, while the KL-native electron source remains the blocker. No electron materialization, same-boundary fixed+electron bundle, gas-recouping adoption, removed-tail, guarded-mode, or production solver actualization is promotable.
63. Any electron materialization actualization must source-prove the electron value at the same boundary as the reduced-Newton fixed vector, including the FastChem gas electron equation branch locals if a charge-neutrality reconstruction is claimed. The compact electron-provenance audit shows that the FC cached electron is consumed by molecule reconstruction, but the focused trace does not emit the prior FC gas-only/activity/reset/gas-solver electron stages or the `alpha`/`beta` and `positive_ion_density`/`negative_ion_density` locals from `calc_electron_densities.cpp`. The best KL-native electron remains gas-recouping / post-adoption gas recompute, with log residuals `59.59` / `74.88` from the FC electron and selected-row residuals `2.143e10` / `1.222e9` when paired with KL reduced fixed. Only the FC electron reaches the diagnostic upper bound. No KL electron, scalar gauge offset, charge-neutrality reconstruction, gas-solver path, floor/cap/clipping, guarded-mode, or production solver actualization is promotable.
64. The FastChem electron-solver trace resolves the FC source equation for the cached electron without changing gas-solver behavior. The matching FC cached electron calls are `GasPhase::calculateSinglyIonElectrons` with branch `singly_ion_analytic`: layer `45:-10` uses `alpha = 5.442750603286076e-15` and `beta = 6314520127.835389`; layer `60:-5` uses `alpha = 3.2674752491051913e-40` and `beta = 2.8163690441222026e17`. The analytic reconstruction `sqrt(alpha/(1+beta))` exactly reproduces the cached electron that is then carried through `iter0_post_correctValues_full_element_vector_before_molecule_refresh` and consumed by molecule reconstruction. The KL same-boundary electron still does not carry this value, and the best mixed-boundary KL gas-recouping/post-adoption electron remains far from FC. Therefore the remaining blocker is electron boundary/carry mismatch. No electron carry rule, same-boundary materialization rule, gas-recouping adoption, guarded mode, or production solver actualization is promotable.
65. Any KL-side FastChem-style electron reconstruction must be treated as diagnostic until it closes with KL-native source states, not with FC non-fixed upper-bound inputs. The compact reconstruction audit uses the emitted FastChem ion list and singly-ion alpha/beta definitions and proves the FC formula identity exactly, but KL current/post-`correctValues` and gas-recouping/post-adoption reconstructions remain far from the FC electron and do not improve the selected-row replay when paired with the KL reduced fixed vector. The dominant log-electron mismatch is the beta / ion-correction contribution on both focused layers. No same-boundary KL-native reconstructed electron reaches within `10x` of the FC fixed+electron residual; only FC electron or FC non-fixed diagnostic input reaches the known upper bound. Therefore no beta correction, alpha/beta replay, electron materialization, fixed+electron bundle, guarded mode, or production solver rule is promotable.
66. Any beta / negative-ion actualization must close from a KL-native same-boundary non-fixed/electron source state, not from FC beta or FC non-fixed borrowing. The beta attribution compact proves the FastChem beta identity for all `50` beta-side negative-ion entries on both focused layers, and the alpha-beta swap replay shows `KL alpha + FC beta` reaches the known diagnostic FC fixed+electron residual scale while `FC alpha + KL beta` remains catastrophic. Per-ion residuals are broad with large leading contributors (`F6S1-`, `Al1F4-`, `F5S1-`, and layer-dependent metal/carbon ions), so no small negative-ion subset is source-proven. Same-boundary reduced-fixed alpha/beta reconstruction still lacks non-fixed support logs for global element indices `2`, `13`, and `19` (`Ar`, `He`, `Ne`) from the diagnostic KL-native fixed+electron bundle overlay. FC non-fixed and full FC cached-input support hybrids are diagnostic upper bounds only. Therefore no beta correction, support-element hybrid, alpha/beta swap, electron rule, guarded mode, or production solver actualization is promotable; FC non-fixed source state is required and no KL-native beta candidate closes.
67. Any follow-up beta actualization must use canonical FastChem global element indices when evaluating FastChem-style alpha/beta stoichiometry. The same-boundary vector repair compact shows the previous `Ar`/`He`/`Ne` missing fields were a diagnostic construction bug: KL vectors were complete but in KL element order with `e-` last, while the alpha/beta stoichiometry expects FastChem global indices with `e-` at index `0`. Canonicalizing by element label repairs all `28` logs for candidates `A-G`, and `Ar`, `He`, and `Ne` are finite on both focused layers. The repaired KL reduced-fixed plus KL current/gas/post-adoption non-fixed candidates no longer have catastrophic beta; they match the prior KL reduced-fixed plus FC non-fixed diagnostic residuals (`0.6006` / `0.8754` beta log1p residual, `0.2025` / `0.3300` electron log residual). Full FC cached input still closes beta, while requested non-fixed ablations, including `Ar+He+Ne` and all non-fixed elements, do not move the repaired residual. Therefore the missing-field blocker is repaired, but no source rule is proven: no beta correction, non-fixed support hybrid, alpha/beta swap, electron rule, guarded mode, or production solver actualization is promotable. The beta attribution remains mixed or inconclusive.
68. Any repaired alpha/beta electron actualization must prove a production source boundary, not only a diagnostic tail replay. The repaired tail compact shows the canonical-vector gate passes and the repaired same-boundary KL reduced-fixed plus FastChem-style alpha/beta electron recovers the known molecule/inventory cancellation in the common post-complementarity context. Candidate molecule RHS plus gauge-normalized inventory/atom gives `0.08475871276713746` at `45:-10` and `1.3947630928999967e-11` at `60:-5`; adding exact removed closes layer `45:-10` to `9.001471173380398e-12`. In the coherent FC-Jacobian raw-result context, molecule-only remains destructive, but candidate molecule plus inventory/atom recovers the same selected-row cancellation. The exact beta residual is nonblocking for selected-row closure but remains a provenance residual. Therefore no production electron rule, beta rule, guarded mode, molecule RHS rule, inventory rule, or removed rule is promotable; layer 45 still requires the separate exact removed tail.
69. Any integrated alpha/beta ladder actualization must survive both the full semantic RHS ladder and the labelled reduced-system solve context without relying on a full-vector metric fallback. The integrated repaired-alpha/beta compact keeps selected-row mean/max residuals as the primary metric and preserves the canonical FastChem order gate with `e-` at index `0`. The repaired candidate recovers the diagnostic tail bundle: `baseline_ABC` is `0.5620237706782278` / `0.9142498895939201`, repaired molecule plus gauge-normalized inventory/atom gives `0.08475871276713746` / `1.3947630928999967e-11`, and layer `45:-10` closes only after exact removed (`9.001471173380398e-12`). In the labelled reduced-system solve context, molecule plus inventory alone does not close (`2.9375647869000923` / `1.9877020617705434`); closure requires the full tail bundle with tau/complementarity. The exact beta mismatch is selected-row nonblocking but remains full-vector relevant through element-RHS l2 differences `1.6836286868830854` and `2.2365923396683853`. Positional-boundary regression has no unknown alpha/beta or molecule-RHS occurrence. Therefore no production electron rule, beta rule, molecule/inventory pair, removed-tail rule, tau/complementarity rule, guarded mode, or solver actualization is promotable.
70. Any diagnostic matrix diagonal read must prove its basis before using a positional index. The positional-boundary follow-up resolves the four remaining unknowns in the fixed-row result-entry and reduced-Newton result-slot compacts. Those unknowns were false positives caused by ambiguous variable naming: the old helper argument was named `element_index`, but it was actually `element_labels.index(element)` inside `element_element_jacobian_subterms` local element-label order. The helper now uses `subterm_element_pos`, and the diagnostics emit basis, expected element, label position, label at position, global element index when available, matrix dimensions, and per-subterm `basis_guard.safe` flags. The regenerated positional audit has `0` unknown occurrences. This is a diagnostic safety fix only; no solver rule, guarded mode, Jacobian rule, or production behavior is promotable.
71. Any repaired alpha/beta molecule RHS actualization must close as a coherent source-state bundle and remain nonblocking in labelled raw solve context before it is considered for promotion. The coherent bundle compact confirms the repaired same-boundary molecule RHS is destructive in isolation (`36.6124` / `49.7254`) and inventory/atom is also destructive in isolation (`36.9071` / `50.4194`), but the coherent molecule plus inventory/atom pair recovers the selected-row tail (`0.0847587` / `1.39e-11`) and exact removed closes layer `45:-10`. In raw solve context, the full tail bundle with removed/tau/complementarity is nonblocking relative to full FC (`1.13e-10` / `2.05e-10` l2 differences). Exact beta mismatch is still full-vector relevant outside selected rows (`1.6836` / `2.2366` element-RHS l2, `22` outside-selected rows each), and broader smoke is blocked by focused-only repaired-alpha/beta artifacts. Therefore no production electron rule, beta rule, molecule RHS rule, inventory rule, removed-tail rule, tau/complementarity rule, guarded mode, or solver actualization is promotable.
72. Any repaired alpha/beta broad-smoke promotion requires regenerated broad alpha/beta source-state artifacts, not extrapolation from the focused layers. The previous phase2 broad-smoke blocker was an over-strict diagnostic assertion: it compared raw element labels (`Al`, ..., `e-`) to formula-like element species labels (`Al1`, ..., `e1-`). The assertion now checks formula-equivalent one-atom species in ExoGibbs e-last order and verifies `formula_matrix_gas[:, :n_elem]` is the identity block. The attempted compact case set `30:-10 45:-10 60:-5 75:-5 90:-5` now completes phase2 artifact generation, but the repaired alpha/beta dependency graph remains incomplete: raw-result provenance depends on focused selected-row delta input, reduced-Newton result-slot provenance has `LAYERS=(45, 60)` and only `--trace-45` / `--trace-60`, and molecule-vector/coherent-bundle/electron-reconstruction compacts still emit focused layers only. Therefore no broad repaired-alpha/beta rule is promotable; no electron, beta, molecule/inventory, removed-tail, guarded-mode, species-boundary, or production solver rule is promotable.
72a. Direct broad extraction from `/tmp/exogibbs_phase2_broad_smoke.json` does not remove the blocker. The broad artifact exposes full reduced-system records, row/column labels, RHS/Jacobian/solver vectors, row scaling, FastChem alpha/beta electron traces, molecule provenance, and exact inventory/removed traces for all five requested cases. The phase2 driver now has a diagnostic-only `repaired_alpha_beta_source_state_snapshot` hook for future broad runs, but the existing broad artifact and derived snapshot remain incomplete: they do not expose the repaired same-boundary KL non-fixed vector at `iter1_RHS_assembly_entry`, and they do not expose the selected-row delta/raw-result mapping used as the primary metric. Therefore the direct evaluator records `repaired source-state snapshot emission incomplete`; no canonical repaired candidate, molecule+inventory replay, raw-solve replay, outside-selected attribution, or production rule is promotable.
72b. A fresh broad rerun with the embedded diagnostic source-state snapshot narrows the broad repaired-alpha/beta blocker but does not clear it. The refreshed artifacts under `results/actual_fastchem_gas_phase_transplant_phase2_repaired_alpha_beta_broad_snapshot_entrance_smoke*` emit `same_boundary_KL_non_fixed_values` for all five broad cases from `iter1_RHS_assembly_entry`, including ExoGibbs e-last labels and FastChem canonical e-first converted entries. Canonical mapping inputs, reduced fixed values, molecule RHS inputs, inventory/atom inputs, exact removed inputs, and raw labelled reduced-system fields are present. The remaining missing field is `selected_row_mapping` with row position, result index, row label, and delta classification for the selected-row metric. Therefore the broad direct evaluator still records `repaired source-state snapshot emission incomplete`; no broad alpha/beta replay, selected-row closure, raw-solve conclusion, outside-selected attribution, electron rule, beta rule, molecule/inventory rule, selected-row rule, guarded mode, or production solver behavior is promotable.
72c. The broad source-state snapshot now emits selected-row mapping for all five requested cases. The mapping is diagnostic-only and uses the existing focused row source `PRESELECTION_ACTIVITY_FOCUSED_NAMES`, projected through labelled FastChem reduced-system columns and FastChem/KL `solver_result_to_delta_n_cond_mapping`. Focused validation against raw-result provenance passes for `45:-10` and `60:-5` on row count, row labels, and FC/KL result indices. The source snapshot is now complete for same-boundary KL non-fixed values and selected-row mapping, but broad numeric repaired-alpha/beta replay is still not complete: the direct broad compact has not implemented the convention-safe alpha/beta molecule-density reconstruction and molecule/inventory tail replay without the focused compact stack. Therefore no broad selected-row closure, raw-solve nonblocking result, outside-selected attribution, electron rule, beta rule, molecule/inventory rule, selected-row rule, guarded mode, or production solver behavior is promotable.
73. Any diagnostic that uses contracted formula rows must prove the row count is not being reused as a species boundary. The positional-boundary compact now audits `contract_formula_matrix`, `formula_matrix_gas.shape[0]`, `species[n_elem:]`, `hvector[n_elem:]`, `n_elem + mol_i`, `element_vector_full[element_mask]`, and `element_names` from `element_mask`. The phase2 full element-count guard covers all species/hvector boundary uses, and contracted-basis uses propagate `element_mask` and `element_names`; the regenerated compact reports `26` contracted-boundary occurrences with `0` unsafe and `0` unknown. This is a diagnostic guardrail only. No production solver, preset, RHS, molecule, electron, inventory, row-scaling, hvector, lifecycle, guarded-mode, formula-matrix, or species-boundary rule is promotable.
74. Any broad repaired alpha/beta generalization must use the same selected-row metric family as the focused raw-result provenance audit, or explicitly report that the metric is unavailable. The broad raw-result provenance compact now reuses the focused construction: selected delta-provenance rows are those classified as `delta_raw_result_dominated` or `delta_mapping_or_index_dominated`, with condensate row identity, mapping status, and labelled FastChem/KL raw-result indices. The generated broad mapping matches the old focused mapping for `45:-10` and `60:-5`, but it cannot be emitted for `30:-10`, `75:-5`, or `90:-5` because the delta-provenance compact has no selected rows for those cases. The embedded broad 10-row projection remains a separate outside-selected diagnostic and is not a focused-regression substitute. Therefore broad repaired alpha/beta remains metric-inconclusive; no electron, beta, molecule/inventory, selected-row, removed-tail, guarded-mode, or solver behavior is promotable.
75. Any broad delta-provenance reconstruction from solver-result mappings alone must validate against the existing focused delta compact before it can feed the focused-compatible selected-row metric. The broad delta candidate now emits rows for `30:-10`, `45:-10`, `60:-5`, `75:-5`, and `90:-5`, but it fails the focused `45:-10`/`60:-5` validation: selected row labels/counts, classifications, result indices, and mapping statuses do not match the old focused delta provenance. The missing source is not the broad solver-result mapping itself; it is the `one_step_compact_extract`-style per-row old/new `correctValues` state, focused-source row membership, dominant-term selection, and top-residual row context. The candidate broad delta rows are therefore not accepted as focused-compatible, and broad repaired alpha/beta remains metric-inconclusive. No selected-row, electron, beta, molecule/inventory, removed-tail, guarded-mode, or solver rule is promotable.
76. Any broad one-step compact extract must pass the focused delta-provenance validation before broad selected-row mappings are generalized. The broad one-step compact extract now preserves the focused schema and covers all five broad cases, but it reports missing row-level fields on some rows, especially FastChem raw-result fields outside the emitted FastChem one-step trace subset. Rebuilding broad delta provenance from this upstream extract still fails focused `45:-10`/`60:-5` validation on selected row count, row labels, classifications, result indices, and mapping status. The stop gate is therefore correct: broad delta provenance is diagnostic-only and rejected as focused-compatible. No selected-row metric, electron, beta, molecule/inventory, removed-tail, guarded-mode, or solver behavior is promotable.
77. Any broad delta-provenance validation must first decide whether the focused reference is current. A fresh focused one-step extract generated through the same code path as the broad extract proves the historical focused one-step and delta artifacts are stale relative to the current diagnostic extractor. The repair is diagnostic-only: `_fastchem_iter1_one_step_records` now prefers retained FastChem iter1 `correctValues_rule` records carrying `raw_solver_result_value` over later eliminated records for the same condensate. Broad one-step extraction then has no unexpected missing shared-row fields, and broad delta provenance validates against the fresh focused delta reference for `45:-10` and `60:-5`. This does not promote the repaired alpha/beta candidate: the historical focused closure values must be rebaselined under the current focused metric before any broad generalization or RGIE/PIPM transfer. No production electron, beta, molecule/inventory, selected-row, removed-tail, guarded-mode, or solver rule is promotable.
78. Any one-sided broad selected-row actualization must now handle two separate diagnostic tracks rather than a single mixed provenance bucket. Track A contains `7` activity-threshold crossing rows (`30:-10:MgCO3(s)`, `30:-10:SiC(s)`, and `CH4(s,l)` at all five broad cases), with available decomposition pointing to CH4 sentinel/thermo source-state mismatch and atomic gas element density snapshot mismatch for `MgCO3(s)` / `SiC(s)`. Track B contains `5` result-index mapping rows (`Al(s)` at `30:-10` and `45:-10`, plus `K3AlF6(s)`, `Na3AlF6(s,l)`, and `Na5Al3F14(s,l)` at `30:-10`), all classified as row present but no result index after label normalization recheck. Exact KL labelled RHS/Jacobian row/column arrays, row scaling, and solver result vectors by label are still missing, so labelled reduced-system materialization mismatch is not promoted. No repaired alpha/beta, electron, molecule RHS, inventory, removed-tail, row-selection, row-scaling, lifecycle, labelled-system, guarded-mode, or production solver behavior is promotable. Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.
79. Any CH4 activity-threshold actualization must respect the FastChem
data-validity sentinel path before interpreting thermo or atomic
counterfactuals. The compact CH4 audit now consumes the diagnostic-only
whitebox trace from `Condensate<double_type>::calcActivity`: all five broad
CH4 rows emit raw mass-action `raw_log_activity_before_floor_clip` before the
data-validity floor, while the stored `displayed_log_activity_after_floor_clip`
remains `-10` with `clipped_or_floored = true` and
`data_validity_floor = true` from the `condensate_struct.cpp:77-92` branch.
The raw pre-floor values are `58.169686102954905`, `-4.530329499225015`,
`-15.485824491588767`, `-19.05959499690444`, and `-21.452325720828508`.
CH4 stored `-10` is therefore not a true computed activity.
KL CH4 activity is positive in all five selected cases. The decomposition
records the logK source record, phase segment `l`, selected temperature
interval, standard-state/density-gauge terms, formula row `C + 4H`, atomic
contributions, and source-state counterfactuals. This is an activity
source-state diagnostic, not an electron/beta rule.
80. Any MgCO3/SiC threshold actualization must first source-prove the atomic
element-density snapshot consumed by activity. For `30:-10:MgCO3(s)` and
`30:-10:SiC(s)`, the compact reports FastChem fixed post-condensation atomic
donor terms versus KL gas-only `ln_nk` donor terms, plus hvector/lnK,
density-gauge, formula-row, conserved-inventory, and source-state swap
provenance. The per-element post-`correctValues` contribution before
`calcActivity` is still not separately emitted. This remains an activity
source-state diagnostic, not a production fixed-element overwrite or
density-gauge rule.
81. Any result-index mapping actualization must first emit exact KL labelled
reduced-system arrays. The current compact reconstructs KL labels, row/column
positions, and result indices from
`current_best_upstream_kl_branch.split_history[0].condensates_jac`, but exact
KL labelled RHS rows, Jacobian rows, Jacobian columns, row scaling by label,
and solver result vector by label remain missing. The five Track B rows are
classified as row present but no result index. This is a labelled
reduced-system / KL exact row-materialization diagnostic, not a production
solver, row-scaling, lifecycle, or selected-row rule.

## Split Frontier Display-Floor and Materialization Audit

The split frontier remains diagnostic-only under `focused_raw_result_provenance_metric`: `12` selected rows, `0` shared numeric rows, and `12` one-sided rows. Track A keeps the `7` activity-threshold rows, and Track B keeps the `5` result-index rows. No one-sided row is reinterpreted as shared numeric, and the embedded broad 10-row projection remains a separate non-closing diagnostic rather than a focused regression.

For `CH4(s,l)`, the diagnostic-only FastChem trace from `Condensate<double_type>::calcActivity` now emits the exact display-floor condition flags. All five broad cases have `data_validity_floor=true`, finite raw/stored values, valid species/phase and density/maxDensity flags, and `candidate_absence_display_flag=false`. The threshold input used by `selectActiveCondensates` is the stored `log_activity`, not the raw pre-floor value. At `30:-10`, raw pre-floor `58.169686102954905` would pass, but the data-validity floor stores `-10`, so the stored threshold fails and the display-floor path affects candidate selection. At `45:-10`, `60:-5`, `75:-5`, and `90:-5`, both raw pre-floor and stored threshold fail on the FastChem side. KL CH4 remains threshold-positive in all five cases, and the thermo/lnK side-by-side record remains emitted for source comparison.

For `30:-10:MgCO3(s)` and `30:-10:SiC(s)`, the compact now emits a per-element C/Mg/O/Si stage table. The available earliest divergent stage is the fixed-element/full-element donor vector consumed by FastChem `calcActivity`: FastChem uses post-condensation fixed full-element values, while KL uses gas-only `ln_nk` values. Per-element post-`correctValues` and per-element density-gauge transformed atomic values are still reported as missing diagnostic fields. Counterfactuals show `FC/KL thermo + FC full-element vector` pass and `FC/KL thermo + KL gas-only vector` fail, so the atomic vector is the threshold-crossing component.

For Track B, exact KL labelled RHS row labels, Jacobian row labels, Jacobian column labels, row scaling by label, and solver result vector by label remain unavailable. The compact reports the exact missing Python locals / trace records and patch site, while split-history materialization still reconstructs labels, positions, and result indices. In the reconstructed materialization all five Group B rows are absent before reduced-system assembly and have no result slot, but exact KL arrays are still required before this can be promoted to a final basis claim.

Repaired alpha/beta remains irrelevant unless shared numeric selected rows appear. No production electron rule, guarded mode, solver behavior, preset, RHS, molecule, inventory, removed-tail, selected-row, row-scaling, active-selection, lifecycle, labelled-system, maxDensity, or density-gauge bridge behavior is promotable.

Decision: next blocker is split: CH4 data-validity floor plus MgCO3/SiC donor snapshot plus result-index mapping.

## Focused Frontier Closure and Broad Projection Pivot

The focused one-sided blocker is closed diagnostically, not promoted. The
integrated compact records `12` original one-sided rows, with `5` CH4 rows
removed by the diagnostic FastChem data-validity mask, `2` MgCO3/SiC rows
explained by the diagnostic FastChem full-element donor snapshot, and `5`
Group-B rows classified as `intentionally_excluded_from_reduced_solve` after
exact KL labelled arrays were consumed. The final focused unresolved count is
`0`.

The production plan remains unchanged: no production electron rule, guarded
mode, thermochemistry default change, donor rule, selected-row rule,
result-index rule, row scaling change, lifecycle change, labelled-system
change, density-gauge bridge change, FastChem behavior change, or KL solver
behavior change is authorized by this closure.

The next admissible diagnostic target is the broad projection residual. The
embedded broad 10-row projection remains a separate
`outside-selected/full-vector residual probe`; it must not be used as focused
regression and does not close under the focused-frontier diagnostic levers.
The current broad residual is still dominated by outside-selected/full-vector
source state.

Decision: broad projection residual remains dominated by outside-selected/full-vector source state.

## Integrated Diagnostic Counterfactual

The latest split-frontier counterfactual is not a production plan item. It
combines two already source-proven diagnostic transforms: a FastChem CH4
data-validity candidate mask on KL CH4 rows, and the FastChem fixed/full-element
donor snapshot on the `30:-10:MgCO3(s)` and `30:-10:SiC(s)` KL activity rows.
The CH4 mask removes all five CH4 one-sided rows, and the donor snapshot
explains both atomic-source rows. No production electron rule, guarded mode,
thermochemistry default, density-gauge bridge, maxDensity behavior, row
selection, active selection, row scaling, lifecycle, labelled-system, RHS,
molecule, inventory, removed-tail, FastChem, KL, preset, or solver behavior is
changed or promotable from this audit.

After the diagnostic transforms, the only remaining blocker is the five-row
Group-B result-index mapping set. Exact KL labelled RHS/Jacobian row/column
arrays, row scaling by label, and solver result vector by label remain the
required audit materialization before result-index provenance can be closed.
Repaired alpha/beta remains irrelevant unless shared numeric selected rows
appear, and the embedded broad 10-row projection remains separate and
non-closing.

Decision: integrated counterfactual reduces blocker to Group-B result-index mapping.

## Group-B Result-Index Materialization Gate

The remaining Group-B rows cannot be closed by a production rule. FastChem
exact labelled materialization is available for the relevant `30:-10` and
`45:-10` cases, but KL exact labelled RHS/Jacobian row and column arrays, row
scaling by label, and solver result vector by label are still absent. Current
KL split-history index materialization is useful provenance, but it is not the
exact labelled array set required to prove intentional exclusion or a compact
mapping artifact.

The production plan therefore remains unchanged: do not modify result-index
semantics, selected-row semantics, labelled-system behavior, row scaling, active
selection, lifecycle, solver behavior, or presets. The next admissible step is
a Python audit materialization patch that emits exact KL labelled arrays for
diagnostics only.

Decision: Group-B result-index blocker remains blocked by missing KL exact labelled arrays.

## Exact KL Array Wiring Outcome

The exact KL labelled reduced-system arrays were found in the existing broad
snapshot under the diagnostic `actual_true_kl_atomic_branch_exact_second_post_seed_update_proven`
mode and have been wired into the Group-B compact. The audit now consumes exact
KL RHS/Jacobian labels, row and column positions, row scaling, solver result
vector, and result-index mapping without changing production result-index
semantics or solver behavior.

All five Group-B rows classify as `absent_before_candidate_selection`. This
closes the missing-array materialization blocker diagnostically, but it does
not create a production selected-row rule or result-index rule. CH4
data-validity masking and MgCO3/SiC donor snapshot remain diagnostic-only, and
repaired alpha/beta remains irrelevant unless shared numeric selected rows
appear.

Decision: Group-B result-index blocker reduces after exact labelled materialization.

Under the final Group-B taxonomy, exact lifecycle absence before candidate
selection is recorded as `intentionally_excluded_from_reduced_solve`. This is a
diagnostic classification only; it does not alter result-index semantics,
selected-row rules, active-selection rules, row scaling, labelled-system
behavior, or solver behavior.

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

## Production-Readiness Package Plan

The production-readiness compact is:

- `results/fastchem_cond_kl_production_readiness_compact.json`
- `results/fastchem_cond_kl_production_readiness_compact.md`

The semantic design note is:

- `docs/condensates/fastchem_parity_kl_semantic_design_note.md`

Plan status:

- No production solver behavior, preset, default, row selection, active
  selection, row scaling, lifecycle, molecule, inventory, removed-tail,
  FastChem, or KL behavior changes are authorized.
- The only production-design candidates are semantic interfaces: normalized
  donor versus physical donor, coherent molecule + inventory state, and
  removed-tail provenance reporting.
- A guarded KL option may be designed as a default-off diagnostic/prototype
  state-interface mode. It is not approved for implementation or promotion by
  this package.
- Production candidacy requires the invariant checklist in the compact:
  metric-lineage preservation, explicit state basis, coherent-bundle handling,
  no legacy burden-ratio conversion, no infinity-norm fallback, no row/species
  or case dropping, supersession ledger preservation, and explicit treatment of
  `45:-10` removed-tail locality.
- Extra broad generalization was not run. Cases beyond the current five require
  broad phase2 diagnostic regeneration and downstream compact regeneration
  before they can support promotion.

Decision: semantic levers ready for production design note but not promotable.
