N2H4: bounded candidates and fixed-network substitution
=======================================================

Prefer the retained-thermal ATcT energy update; do not adopt the substantially different NASA thermal function until torsional and inversion treatments are reconciled. No curated data or runtime function changes in this review.
Source selection and physical energy
------------------------------------

The energy-aligned JANAF/NASA ratio is about 1.499 to 2.263, far larger than their roughly 85 J/mol native energy difference. Large-amplitude torsion and inversion therefore matter more than adjusting the NASA native energy zero. ATcT gas Hf298=97.69 +/- 0.29 kJ/mol and the inspected 2017 theory result 97 +/- 3 kJ/mol support a modern energy reference, but their quoted widths are not a reaction covariance. The 2001 thermodynamic survey is a concrete follow-up for the thermal model; its publisher abstract was inspected, not the complete paper. Missing that paper does not block the separately bounded energy candidate.

Physical JANAF D0=1696.355000 kJ/mol changes to the balanced ATcT 1.228 D0=1693.616000 kJ/mol. The retained-thermal variant changes formation G and H by +2739.000 J/mol, leaving S and Cp unchanged.

All atom and molecular ATcT values use version 1.228. The balance is the sum of neutral ground-atom Hf(0) minus molecular Hf(0), not just the molecular formation-enthalpy difference. This physical D0 includes the zero-point convention already; no extra ZPE is added. The formal empirical R*a is not D0. Retained-thermal variants preserve the original literal fit and its source-table residual, with the background atomic potentials fixed. The NASA alternatives instead replace the source-specific reaction thermal function and inherited fit residual; they remove their physical header-derived D0/R using native R=8.314510, then insert the selected D0 using SI R=8.31446261815324 exactly once. They are not called pure molecular-thermal changes with fixed atomic thermals. No polynomial or fitted expression is evaluated at zero temperature.

The original source packet and source-review history remain unchanged. See the `original source review <n2h4_source_review.html>`_ for exact primary records, native and energy-aligned source comparisons, access status and DOI links. Marginal ATcT widths are retained in candidate.json; without reaction covariance they are not combined into a claimed confidence interval.
Calculation scope
-----------------

The actual named ``curated_ch2`` factory (dataset ``curated-ch2-v1``) and its hashed 150 archived baseline endpoints are verified before reuse. The two fixed networks have 537 species including ions and 421 neutral species. Each uses 75 conditions: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. Standard-state pressure is one bar. There are 300 fresh main endpoints, 18 fresh tighter cold controls, 10 primary-candidate interpolation controls and 0 selected alternative controls. The common 150 baseline endpoints are reused once; they are not additional fresh calculations.

The numerical source checks cover 2752 temperatures, including 251 off-grid points. Independent source G/H/S/Cp, scalar/JAX derivatives, source-fit residual, domain, target-only substitution, positive finite amounts, all-element conservation, nitrogen and other participating elements, charge, mass action and cold-start convergence are checked. The 6 standalone FastChem executions compare every main variant and the current model at two distinct ion-inclusive conditions; the largest mole-fraction difference is 5.89402e-09 dex. Numerical agreement is not physical source validation.
Finite response
---------------

The following rows report the ion-inclusive network. The neutral network is separately calculated and archived; no equilibrium-constant ratio is presented as a solved abundance ratio. Element shares are percentages of all conserved nitrogen atoms. Maximum before and after shares may occur at different conditions.

.. list-table:: Actual target abundance and nitrogen allocation
   :header-rows: 1

   * - Variant
     - Species
     - Mole-fraction ratio range
     - Maximum N share before / after (%)
   * - retained
     - H4N2
     - 0.80282596 to 0.92094359
     - 5.8085427e-08 / 5.3493402e-08
   * - nasa
     - H4N2
     - 0.40413808 to 0.52273262
     - 5.8085427e-08 / 2.3474533e-08

Same-condition changes and all leading receiving/depleting nitrogen carriers are retained in summary.json. Trace responses are reported without concentration floors.

Only 40/300 main comparisons have both feasibility-corrected same-model Gibbs exchange signs above the displayed floating-point scale. The remaining 260 are explicitly unresolved. This scale is diagnostic, not a rigorous error bound. Small total-G signals can remain unresolved while target abundances and mass action pass tighter controls. Cross-model absolute Gibbs values do not rank the physical sources.

Decision and remaining limits
-----------------------------

Prefer the retained-thermal ATcT energy update; do not adopt the substantially different NASA thermal function until torsional and inversion treatments are reconciled.

A NASA replacement would combine an energy change with an unresolved factor-of-order-two thermal change. Reconcile internal rotation, inversion, symmetry and entropy conventions before preferring it; keep the energy correction independent.

These results apply to ideal-gas chemical equilibrium in a hot hydrogen-rich fixed network. They do not validate cooler oxidizing atmospheres, condensation, photochemistry, non-LTE populations or kinetics. Small abundance here is not a reason to declare the molecule unimportant elsewhere. Full correlated physical uncertainty remains unknown; missing ancestor papers are not an indefinite merge condition for this bounded review.

Replay and evidence
-------------------

The candidate and impact execution receipts preserve their actual commands, input bytes and numerical outputs. Tests and this document were created afterward and are pinned separately by the MCEG contract, not retroactively inserted into those receipts. All original source packets, production data and earlier receipts remain unchanged.

Run the source-specific replay with::

    pytest tests/model_evidence/n2h4_candidate_test.py

The candidate packet is ``tests/data/validation/n2h4_candidate/20261106`` and the fixed-network packet is ``tests/data/validation/n2h4_equilibrium_impact/20261106``. The MCEG contract is ``mceg/contracts/n2h4_candidate.yaml``.
