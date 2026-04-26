# ExoGibbs/FastChem Experimental Branch Map

Generated: 2026-04-12T23:11:31Z

## Branches

### current_true_kl_branch
- Modes: current_true_kl_branch
- Kind: actual experimental branch
- Changes: KL/FastChem-like atomic branch; RGIE remains control and production path is untouched.
- Files: examples/comparisons/audit_true_kl_atomic_branch.py, src/exogibbs/optimize/pipm_rgie_cond.py
- Best metric: not available
- Bottleneck: condensate lifecycle, update-map, and downstream semantics.
- Next: compare against exact FastChem lifecycle traces.

### exact_newly_active_lifecycle_only
- Modes: exact_newly_active_lifecycle_only
- Kind: actual diagnostic branch
- Changes: FastChem-like newly-active lifecycle and maxDensity seeding.
- Files: examples/comparisons/audit_kl_exact_maxdensity_lifecycle.py
- Best metric: not available
- Bottleneck: actual reduced/full update map.
- Next: transplant update-map ordering.

### actual_fastchem_update_map_reduced
- Modes: actual_fastchem_update_map_reduced, current_actual_fastchem_update_map_reduced
- Kind: actual experimental branch
- Changes: FastChem-like reduced update-map equations for current KL branch.
- Files: examples/comparisons/audit_actual_fastchem_update_map_kl.py
- Best metric: {"best_mode": "fastchem_update_map_reduced_replay_reference", "best_value": 0.1653250773993808, "decision": "the transplant target is still incomplete; inspect reduced reconstruction details next", "metric": "mean_final_stable_jaccard", "source": "/home/kawahara/exogibbs/results/actual_fastchem_update_map_kl_audit.json"}
- Bottleneck: reduced reconstruction and downstream semantics.
- Next: audit exact reduced reconstruction.

### update_map_alignment_phase2
- Modes: update_map_alignment_phase2_reduced, update_map_alignment_phase2_full, update_map_alignment_phase2_reduced_plus_row_scaling, update_map_alignment_phase2_reduced_plus_update_clipping
- Kind: actual experimental branch
- Changes: Reduced/full assembly, multiplicative n/lambda updates, maxDensity cap timing, atomic-density update order.
- Files: examples/comparisons/audit_fastchem_update_map_alignment_phase2.py
- Best metric: {"best_mode": "update_map_alignment_phase2_reduced", "best_value": 0.12886002886002884, "decision": "the remaining gap is still in hidden solver details or incomplete reduced/full reconstruction", "metric": "mean_final_stable_jaccard", "source": "/home/kawahara/exogibbs/results/fastchem_update_map_alignment_phase2_audit.json"}
- Bottleneck: stable set barely moves despite reduced update mismatch.
- Next: audit exact reduced reconstruction.

### exact_reduced_reconstruction
- Modes: exact_reduced_reconstruction_replay, exact_reduced_reconstruction_actual, exact_reduced_reconstruction_actual_plus_row_scaling, exact_reduced_reconstruction_actual_plus_update_clipping
- Kind: replay plus actual diagnostic branch
- Changes: Exact condensates_jac/rem split, folded RHS terms, eliminated-condensate reconstruction, FastChem reference replay.
- Files: examples/comparisons/audit_fastchem_exact_reduced_reconstruction.py
- Best metric: {"best_mode": "phase2_reduced_current", "best_value": 0.12886002886002884, "decision": "the next dominant mismatch likely moves downstream to fixed_by_condensation / phi semantics", "metric": "mean_final_stable_jaccard", "source": "/home/kawahara/exogibbs/results/fastchem_exact_reduced_reconstruction_audit.json"}
- Bottleneck: stable set still does not materially improve.
- Next: audit downstream final removal, fixed_by_condensation, and phi recoupling.

### downstream_staged_transplant
- Modes: current_best_upstream_kl_branch, downstream_removal_only, downstream_fixed_elements_only, downstream_phi_only, downstream_full_transplant
- Kind: diagnostic replay/transplant branch
- Changes: Stages final removal, condensed-element fixing, phi/b_eff gas recoupling, and full downstream package.
- Files: examples/comparisons/audit_fastchem_downstream_staged_transplant.py, examples/comparisons/audit_fastchem_downstream_semantics_parity.py
- Best metric: {"dominant_downstream_lever": "final_removal", "gains_vs_current_best_upstream": {"fixed_by_condensation": 0.0, "full_downstream": -133.5881651827046, "phi_gas_recoupling": -133.58816550263117, "removal": 0.24090949298379646}, "gas_rms_improves_only_after_phi_recoupling": false, "mean_condensed_element_jaccard": {"current_best_upstream_kl_branch": 0.2863157894736842, "downstream_fixed_elements_only": 0.2863157894736842, "downstream_full_transplant": 0.3566666666666666, "downstream_phi_only": 0.2863157894736842, "downstream_removal_only": 0.3566666666666666}, "mean_final_stable_jaccard": {"current_best_upstream_kl_branch": 0.1244155844155844, "downstream_fixed_elements_only": 0.1244155844155844, "downstream_full_transplant": 0.36532507739938086, "downstream_phi_only": 0.1244155844155844, "downstream_removal_only": 0.36532507739938086}, "mean_major_species_rms": {"current_best_upstream_kl_branch": 14.455518770593397, "downstream_fixed_elements_only": 148.04368395329828, "downstream_full_transplant": 148.043683953298, "downstream_phi_only": 148.04368427322456, "downstream_removal_only": 14.455518770593397}, "message": "the next actual transplant should implement FastChem final removal semantics"}
- Bottleneck: final_removal
- Next: the next actual transplant should implement FastChem final removal semantics

## FastChem To ExoGibbs Source Map

### selectActiveCondensates
- FastChem: fastchem/fastchem_src/condensed_phase/calculate.cpp, fastchem/fastchem_src/condensed_phase/condensed_phase.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_kl_exact_maxdensity_lifecycle.py, examples/comparisons/audit_fastchem_reduced_reconstruction_parity.py

### calculate_entry_seeding
- FastChem: fastchem/fastchem_src/condensed_phase/calculate.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_kl_exact_maxdensity_lifecycle.py

### correctValues
- FastChem: fastchem/fastchem_src/condensed_phase/calculate.cpp, fastchem/fastchem_src/condensed_phase/solver.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_fastchem_reduced_reconstruction_parity.py, examples/comparisons/audit_fastchem_exact_reduced_reconstruction.py

### correctValuesFull
- FastChem: fastchem/fastchem_src/condensed_phase/calculate.cpp, fastchem/fastchem_src/condensed_phase/solver.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_fastchem_update_map_alignment_phase2.py

### final_removal
- FastChem: fastchem/fastchem_src/calc_densities.cpp, fastchem/fastchem_src/diagnostic_trace.h
- ExoGibbs diagnostics: examples/comparisons/audit_actual_fastchem_removal_semantics.py, examples/comparisons/audit_fastchem_downstream_staged_transplant.py

### fixed_by_condensation
- FastChem: fastchem/fastchem_src/condensed_phase/condensed_phase.cpp, fastchem/fastchem_src/calc_densities.cpp, fastchem/fastchem_src/gas_phase/solver_newtsol_mult.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_fastchem_downstream_semantics_parity.py

### phi_gas_recoupling
- FastChem: fastchem/fastchem_src/elements/element_struct.cpp, fastchem/fastchem_src/calc_densities.cpp, fastchem/fastchem_src/gas_phase/solver_coeff.cpp
- ExoGibbs diagnostics: examples/comparisons/audit_fastchem_downstream_semantics_parity.py, examples/comparisons/audit_fastchem_downstream_staged_transplant.py

## Post-Update Ordering Hypotheses

- `exact_postupdate_cap_after_n_then_lambda_then_u`
- `exact_postupdate_cap_after_n_then_u_then_lambda`
- `exact_postupdate_cap_after_lambda_then_n_then_u`
- `exact_postupdate_cap_after_n_only_before_lambda`
- `exact_postupdate_cap_after_full_stage_order_match`
- `full-reference-ordering_debug`
- Best first-update ordering: `full-reference-ordering_debug`
- Decision: `the next dominant mismatch likely moves downstream to fixed_by_condensation / phi semantics`
