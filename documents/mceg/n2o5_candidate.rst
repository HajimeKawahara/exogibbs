N2O5: bounded candidates and fixed-network substitution
=======================================================

Recommend the retained-thermal ATcT H298 energy correction; keep NASA plus ATcT as a tested thermal sensitivity, with thermal superiority unresolved. No curated data or runtime function changes in this review.
Source selection and physical energy
------------------------------------

The selected species is the neutral covalent gas O2N-O-NO2, not solid nitronium nitrate or an independent NO2+/NO3- pair. The original JANAF 0 K cells are blank. Consequently no retained-thermal D0 update is invented from the empirical 1/T coefficient or extrapolated table. A separate, available H298 anchor supplies the energy-only correction below. The independently available NASA header gives D0=2152.710569 kJ/mol and the same-version ATcT cycle gives 2151.095 kJ/mol. Removing the former in native NASA units and inserting the latter defines a valid bounded NASA-thermal candidate. However, it also replaces the thermal shape: the old source comparison at fixed 298.15 K enthalpy differs appreciably. The inspected cold NO2+NO3 equilibrium provenance does not discriminate the competing 1500-4000 K thermal functions. The NASA candidate is complete as a numerical sensitivity, while the H298 alternative below supplies a separately supported energy-only choice. No unbounded literature requirement is imposed on merging this review.

The original physical JANAF D0 is unavailable. The first packet evaluates NASA thermals with its own available D0; the supplementary packet instead retains the existing thermals and uses the independently available 298.15 K reaction enthalpy.

All atom and molecular ATcT values use version 1.228. For the NASA/ATcT D0 comparison, the balance is the sum of neutral ground-atom Hf(0) minus molecular Hf(0), not just the molecular formation-enthalpy difference. This physical D0 includes the zero-point convention already; no extra ZPE is added. The formal empirical R*a is not D0. Retained-thermal variants preserve the original literal fit and its source-table residual, with the background atomic potentials fixed. The NASA alternatives instead replace the source-specific reaction thermal function and inherited fit residual; they remove their physical header-derived D0/R using native R=8.314510, then insert the selected D0 using SI R=8.31446261815324 exactly once. They are not called pure molecular-thermal changes with fixed atomic thermals. No polynomial or fitted expression is evaluated at zero temperature.

The original source packet and source-review history remain unchanged. See the `original source review <n2o5_source_review.html>`_ for exact primary records, native and energy-aligned source comparisons, access status and DOI links. Marginal ATcT widths are retained in candidate.json; without reaction covariance they are not combined into a claimed confidence interval.
Retained-thermal H298 alternative
--------------------------------

JANAF lacks a 0 K N2O5 row, but both evaluations supply 298.15 K formation enthalpies. Balanced atomization H298 is 2179.934 kJ/mol for JANAF and 2176.321 kJ/mol for ATcT 1.228. Therefore formation G/H can be raised by 3613 J/mol while retaining the existing thermal function and its source-fit residual. This is an H298 anchor correction, not a D0 update or a fitted-intercept interpretation. It removes the need to replace an unresolved thermal function merely to adopt the modern energy reference.

At 298.15 K the original Hf values in J/mol are N=472683, O=249173,
N2O5=11297; same-version ATcT gives N=472518, O=249229, N2O5=14860.
The balance is 2 N + 5 O - N2O5. Using only the molecular increase
3563 J/mol would omit the -50 J/mol atomic contribution; the formation
reaction correction is +3613 J/mol. S and Cp remain unchanged. No D0,
ZPE, 0 K caloric entry, extra torsional factor or extrapolated empirical
potential is manufactured. The original source-table residual at the four
1500-4000 K source knots remains unchanged.

The NASA candidate and its original execution receipts are kept intact.
The H298 alternative is a separate definition and actual run using the same
verified named factory and common baseline. Its correction is fixed by the
source reaction H298, although the empirical runtime function itself is
only evaluated within 1500-4000 K. This does not assert that the empirical
fit reproduces its source exactly at 298.15 K.

Calculation scope
-----------------

The actual named ``curated_ch2`` factory (dataset ``curated-ch2-v1``) and its hashed 150 archived baseline endpoints are verified before reuse. The two fixed networks have 537 species including ions and 421 neutral species. Each uses 75 conditions: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. Standard-state pressure is one bar. Each of the two separately executed definitions contributes 150 fresh main endpoints, 12 fresh tighter cold controls and 10 interpolation controls: 300 fresh endpoints and 44 selected controls in total. There are no source-width edge controls. The common 150 distinct baseline endpoints are counted once; they are not additional fresh calculations.

The numerical source checks cover 2752 temperatures, including 251 off-grid points. Independent source G/H/S/Cp, scalar/JAX derivatives, source-fit residual, domain, target-only substitution, positive finite amounts, all-element conservation, nitrogen and other participating elements, charge, mass action and cold-start convergence are checked. Each definition has four standalone FastChem executions comparing its candidate and the current model at two distinct ion-inclusive conditions: eight executions in total. The largest mole-fraction difference across both packets is 1.04991e-08 dex. Numerical agreement is not physical source validation.
Finite response
---------------

The following rows report the ion-inclusive network. The neutral network is separately calculated and archived; no equilibrium-constant ratio is presented as a solved abundance ratio. Element shares are percentages of all conserved nitrogen atoms. Maximum before and after shares may occur at different conditions.

.. list-table:: Actual target abundance and nitrogen allocation
   :header-rows: 1

   * - Variant
     - Species
     - Mole-fraction ratio range
     - Maximum N share before / after (%)
   * - retained_h298
     - N2O5
     - 0.74849106 to 0.89705688
     - 1.0776917e-34 / 9.6675074e-35
   * - nasa
     - N2O5
     - 1.1742079 to 1.3860544
     - 1.0776917e-34 / 1.265434e-34

Same-condition changes and all leading receiving/depleting nitrogen carriers are retained in summary.json. Trace responses are reported without concentration floors.

Only 0/300 main comparisons have both feasibility-corrected same-model Gibbs exchange signs above the displayed floating-point scale. The remaining 300 are explicitly unresolved. This scale is diagnostic, not a rigorous error bound. Small total-G signals can remain unresolved while target abundances and mass action pass tighter controls. Cross-model absolute Gibbs values do not rank the physical sources.

Decision and remaining limits
-----------------------------

Recommend the retained-thermal ATcT H298 energy correction; keep NASA plus ATcT as a tested thermal sensitivity, with thermal superiority unresolved.

A source-specific hot caloric or partition-function comparison would decide a later thermal adoption. The available H298 energy-only correction is already separately evaluated; an original 0 K value is not required for it. Cold kinetics/aerosol equilibrium and full physical accuracy remain outside this grid.

These results apply to ideal-gas chemical equilibrium in a hot hydrogen-rich fixed network. They do not validate cooler oxidizing atmospheres, condensation, photochemistry, non-LTE populations or kinetics. Small abundance here is not a reason to declare the molecule unimportant elsewhere. Full correlated physical uncertainty remains unknown; missing ancestor papers are not an indefinite merge condition for this bounded review.

Replay and evidence
-------------------

The candidate and impact execution receipts preserve their actual commands, input bytes and numerical outputs. Tests and this document were created afterward and are pinned separately by the MCEG contract, not retroactively inserted into those receipts. All original source packets, production data and earlier receipts remain unchanged.

Run the source-specific replay with::

    pytest tests/model_evidence/n2o5_candidate_test.py

The candidate packet is ``tests/data/validation/n2o5_candidate/20261106`` and the fixed-network packet is ``tests/data/validation/n2o5_equilibrium_impact/20261106``. The MCEG contract is ``mceg/contracts/n2o5_candidate.yaml``.

Supplementary evidence is archived in
``tests/data/validation/n2o5_h298_candidate/20261106`` and
``tests/data/validation/n2o5_h298_equilibrium_impact/20261106``.
The local ``n2o5_h298_candidate.py`` helper is pinned independently; no
shared executed candidate helper or previous packet was rewritten.
