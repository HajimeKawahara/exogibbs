HNO: bounded candidates and fixed-network substitution
======================================================

Recommend the retained-thermal ATcT energy correction; the NASA thermal alternative is a quantified sensitivity without demonstrated physical superiority. No curated data or runtime function changes in this review.
Source selection and physical energy
------------------------------------

The 7.613 kJ/mol decrease in balanced atomization energy is appreciably larger than the other small nitrogen corrections. The inspected HNO -> H + NO threshold near 16450 cm-1 is a single fragmentation threshold, not the total H + N + O atomization energy. Same-version ATcT atom and molecule values are balanced explicitly. A complete modern hot partition function was not available; the NASA comparison tests the thermal consequence without treating numerical agreement with an independent solver as source validation.

Physical JANAF D0=831.145000 kJ/mol changes to the balanced ATcT 1.228 D0=823.532000 kJ/mol. The retained-thermal variant changes formation G and H by +7613.000 J/mol, leaving S and Cp unchanged.

All atom and molecular ATcT values use version 1.228. The balance is the sum of neutral ground-atom Hf(0) minus molecular Hf(0), not just the molecular formation-enthalpy difference. This physical D0 includes the zero-point convention already; no extra ZPE is added. The formal empirical R*a is not D0. Retained-thermal variants preserve the original literal fit and its source-table residual, with the background atomic potentials fixed. The NASA alternatives instead replace the source-specific reaction thermal function and inherited fit residual; they remove their physical header-derived D0/R using native R=8.314510, then insert the selected D0 using SI R=8.31446261815324 exactly once. They are not called pure molecular-thermal changes with fixed atomic thermals. No polynomial or fitted expression is evaluated at zero temperature.

The original source packet and source-review history remain unchanged. See the `original source review <hno_source_review.html>`_ for exact primary records, native and energy-aligned source comparisons, access status and DOI links. Marginal ATcT widths are retained in candidate.json; without reaction covariance they are not combined into a claimed confidence interval.
Calculation scope
-----------------

The actual named ``curated_ch2`` factory (dataset ``curated-ch2-v1``) and its hashed 150 archived baseline endpoints are verified before reuse. The two fixed networks have 537 species including ions and 421 neutral species. Each uses 75 conditions: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. Standard-state pressure is one bar. There are 300 fresh main endpoints, 18 fresh tighter cold controls, 10 primary-candidate interpolation controls and 0 selected alternative controls. The common 150 baseline endpoints are reused once; they are not additional fresh calculations.

The numerical source checks cover 2752 temperatures, including 251 off-grid points. Independent source G/H/S/Cp, scalar/JAX derivatives, source-fit residual, domain, target-only substitution, positive finite amounts, all-element conservation, nitrogen and other participating elements, charge, mass action and cold-start convergence are checked. The 6 standalone FastChem executions compare every main variant and the current model at two distinct ion-inclusive conditions; the largest mole-fraction difference is 3.38969e-07 dex. Numerical agreement is not physical source validation.
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
     - H1N1O1
     - 0.54312146 to 0.79540167
     - 0.0001752821 / 0.00013941967
   * - nasa
     - H1N1O1
     - 0.50053834 to 1.0091288
     - 0.0001752821 / 0.00017688221

Same-condition changes and all leading receiving/depleting nitrogen carriers are retained in summary.json. Trace responses are reported without concentration floors.

Only 109/300 main comparisons have both feasibility-corrected same-model Gibbs exchange signs above the displayed floating-point scale. The remaining 191 are explicitly unresolved. This scale is diagnostic, not a rigorous error bound. Small total-G signals can remain unresolved while target abundances and mass action pass tighter controls. Cross-model absolute Gibbs values do not rank the physical sources.

Decision and remaining limits
-----------------------------

Recommend the retained-thermal ATcT energy correction; the NASA thermal alternative is a quantified sensitivity without demonstrated physical superiority.

The source threshold, total atomization energy and source-specific thermal function remain distinct. Neither hot hydrogen-rich trace abundance nor solver precision validates atmospheric kinetics or complete physical uncertainty.

These results apply to ideal-gas chemical equilibrium in a hot hydrogen-rich fixed network. They do not validate cooler oxidizing atmospheres, condensation, photochemistry, non-LTE populations or kinetics. Small abundance here is not a reason to declare the molecule unimportant elsewhere. Full correlated physical uncertainty remains unknown; missing ancestor papers are not an indefinite merge condition for this bounded review.

Replay and evidence
-------------------

The candidate and impact execution receipts preserve their actual commands, input bytes and numerical outputs. Tests and this document were created afterward and are pinned separately by the MCEG contract, not retroactively inserted into those receipts. All original source packets, production data and earlier receipts remain unchanged.

Run the source-specific replay with::

    pytest tests/model_evidence/hno_candidate_test.py

The candidate packet is ``tests/data/validation/hno_candidate/20261106`` and the fixed-network packet is ``tests/data/validation/hno_equilibrium_impact/20261106``. The MCEG contract is ``mceg/contracts/hno_candidate.yaml``.
