N2O: bounded candidates and fixed-network substitution
======================================================

Prefer the retained-thermal ATcT energy candidate for a bounded update; keep NASA plus ATcT as a tested thermal sensitivity, not an established improvement. No curated data or runtime function changes in this review.
Source selection and physical energy
------------------------------------

The energy-aligned JANAF/NASA ratio decreases from about 0.983 to 0.912. The inspected 2025 principal-isotopologue TIPS values, divided by nuclear spin weight 9, give Q/NASA about 1.00023 to 0.97053 at the source knots. NASA is closer to that particular high-temperature partition diagnostic, but this does not isolate the absolute thermal reference or provide complete molecular/atomic caloric derivatives. Sparse Q knots cannot be promoted to a derivative-complete replacement. The approximately 4520 K state-sum crossover lies above this calculation interval. Retaining the existing thermal function makes the well-defined energy correction separately reviewable without declaring the two thermal functions equally accurate.

Physical JANAF D0=1102.951000 kJ/mol changes to the balanced ATcT 1.228 D0=1102.122000 kJ/mol. The retained-thermal variant changes formation G and H by +829.000 J/mol, leaving S and Cp unchanged.

All atom and molecular ATcT values use version 1.228. The balance is the sum of neutral ground-atom Hf(0) minus molecular Hf(0), not just the molecular formation-enthalpy difference. This physical D0 includes the zero-point convention already; no extra ZPE is added. The formal empirical R*a is not D0. Retained-thermal variants preserve the original literal fit and its source-table residual, with the background atomic potentials fixed. The NASA alternatives instead replace the source-specific reaction thermal function and inherited fit residual; they remove their physical header-derived D0/R using native R=8.314510, then insert the selected D0 using SI R=8.31446261815324 exactly once. They are not called pure molecular-thermal changes with fixed atomic thermals. No polynomial or fitted expression is evaluated at zero temperature.

The original source packet and source-review history remain unchanged. See the `original source review <n2o_source_review.html>`_ for exact primary records, native and energy-aligned source comparisons, access status and DOI links. Marginal ATcT widths are retained in candidate.json; without reaction covariance they are not combined into a claimed confidence interval.
Calculation scope
-----------------

The actual named ``curated_ch2`` factory (dataset ``curated-ch2-v1``) and its hashed 150 archived baseline endpoints are verified before reuse. The two fixed networks have 537 species including ions and 421 neutral species. Each uses 75 conditions: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. Standard-state pressure is one bar. There are 300 fresh main endpoints, 18 fresh tighter cold controls, 10 primary-candidate interpolation controls and 0 selected alternative controls. The common 150 baseline endpoints are reused once; they are not additional fresh calculations.

The numerical source checks cover 2752 temperatures, including 251 off-grid points. Independent source G/H/S/Cp, scalar/JAX derivatives, source-fit residual, domain, target-only substitution, positive finite amounts, all-element conservation, nitrogen and other participating elements, charge, mass action and cold-start convergence are checked. The 6 standalone FastChem executions compare every main variant and the current model at two distinct ion-inclusive conditions; the largest mole-fraction difference is 1.04597e-08 dex. Numerical agreement is not physical source validation.
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
     - N2O1
     - 0.9356905 to 0.97538165
     - 3.5643469e-07 / 3.4765986e-07
   * - nasa
     - N2O1
     - 0.94211018 to 1.0652742
     - 3.5643469e-07 / 3.797007e-07

Same-condition changes and all leading receiving/depleting nitrogen carriers are retained in summary.json. Trace responses are reported without concentration floors.

Only 10/300 main comparisons have both feasibility-corrected same-model Gibbs exchange signs above the displayed floating-point scale. The remaining 290 are explicitly unresolved. This scale is diagnostic, not a rigorous error bound. Small total-G signals can remain unresolved while target abundances and mass action pass tighter controls. Cross-model absolute Gibbs values do not rank the physical sources.

Decision and remaining limits
-----------------------------

Prefer the retained-thermal ATcT energy candidate for a bounded update; keep NASA plus ATcT as a tested thermal sensitivity, not an established improvement.

Independent derivative-consistent modern Q/moments and a common absolute entropy/isotope reference could justify a later thermal choice. The finite TIPS comparison does not quantify complete physical uncertainty.

These results apply to ideal-gas chemical equilibrium in a hot hydrogen-rich fixed network. They do not validate cooler oxidizing atmospheres, condensation, photochemistry, non-LTE populations or kinetics. Small abundance here is not a reason to declare the molecule unimportant elsewhere. Full correlated physical uncertainty remains unknown; missing ancestor papers are not an indefinite merge condition for this bounded review.

Replay and evidence
-------------------

The candidate and impact execution receipts preserve their actual commands, input bytes and numerical outputs. Tests and this document were created afterward and are pinned separately by the MCEG contract, not retroactively inserted into those receipts. All original source packets, production data and earlier receipts remain unchanged.

Run the source-specific replay with::

    pytest tests/model_evidence/n2o_candidate_test.py

The candidate packet is ``tests/data/validation/n2o_candidate/20261106`` and the fixed-network packet is ``tests/data/validation/n2o_equilibrium_impact/20261106``. The MCEG contract is ``mceg/contracts/n2o_candidate.yaml``.
