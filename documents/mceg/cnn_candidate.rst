CNN bounded candidates and fixed-network substitution
=====================================================

A provisional retained-thermal ATcT energy update is scientifically supported as one bounded improvement. Do not describe the retained thermal model as validated: a source-resolved vibration/electronic update remains necessary for a stronger full-function claim. NASA is a sensitivity, not an automatic replacement. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear triplet C-N-N diazomethylene, C1N2_cnn, symmetry 1; distinct from NCN.

ATcT 1.228 gives D0=1074.780 kJ/mol, +5.807 relative to JANAF. Its current provenance is largely theoretical, not a new direct calorimetric measurement. NASA has a much less binding native anchor, so energy must be aligned before assessing thermals.

Original physical D0=1068.973000000 and selected D0=1074.780000000 kJ/mol. The retained G/H shift is -5.807000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

The old 2847 cm^-1 stretching estimate conflicts with a later gas assignment near 1425+/-120 cm^-1; low singlet terms also matter. A single replacement frequency is not a complete hot thermal model. The NASA alternative exposes a large remaining thermal/reference difference without claiming to resolve these deficiencies.

The original :doc:`cnn_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

The NASA alternative uses the exact molecule-minus-neutral-atoms NASA9 reaction. It removes source D0 with native R=8.314510 and inserts the selected physical energy with SI R=8.31446261815324, once. Its molecular/atomic thermal gauge and removed fit residual make it a full reference-model sensitivity, not a pure molecular correction or uncertainty interval.

Calculation scope and counts
----------------------------

All variants independently start from named ``curated-ch2-v1`` after HCP/C2H2/CH2 adoption. Each uses 75 conditions in both 537-species ion and 421-species neutral networks. The 150 common current endpoints are reused once with file hashes, named factory potentials, species, formula matrix and elemental budgets verified.

Temperatures are 1500/2000/2500/3000/4000 K,  pressures 1e-6/1e-4/0.01/1/100 bar,  and C/O=0.5888436553555908/1/1.5. Only the explicitly declared target species change; the whole fixed gas network re-equilibrates.

There are **300 fresh main endpoints**, 18 tighter cold solves, 10 path solves, and 0 selected-alternative solves. FastChem executes 6 times at two ion-network conditions, including current and every main variant. These are actual executions, not that many distinct conditions. No neutral FastChem run is implied.

Finite composition response
---------------------------

.. list-table:: Solved target response; shares are maxima on each model grid
   :header-rows: 1

   * - Network/model
     - Minimum/maximum x over current
     - Maximum N share (%)
   * - ions/current
     - 1 / 1
     - 1.00129205e-06
   * - ions/retained
     - 1.19077623 / 1.59299259
     - 1.19231478e-06
   * - ions/nasa
     - 2.05117202 / 2.13836332
     - 2.05382224e-06
   * - neutral/current
     - 1 / 1
     - 1.00144132e-06
   * - neutral/retained
     - 1.19077623 / 1.59299259
     - 1.19249252e-06
   * - neutral/nasa
     - 2.05117202 / 2.13836332
     - 2.05412841e-06

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 1.00129205e-06 / 1.19231478e-06 / 2.05382224e-06 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 1.00144132e-06 / 1.19249252e-06 / 2.05412841e-06 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C1N2_cnn 8.78949e-09 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 5.76697e-09 percentage points at 4000 K/100 bar/C/O=1.5, C1 2.59546e-09 percentage points at 4000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C1N2_cnn 1.91023e-07 percentage points at 4000 K/100 bar/C/O=1.5, N2 1.02522e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 6.2667e-08 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C1N2_cnn 4.84299e-08 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 3.17757e-08 percentage points at 4000 K/100 bar/C/O=1.5, C1 1.43006e-08 percentage points at 4000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute N-allocation changes are C1N2_cnn 1.05253e-06 percentage points at 4000 K/100 bar/C/O=1.5, N2 5.64895e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 3.45292e-07 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.59e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.31e-13, and formation residual 1.53e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.18e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 9/75, ions nasa 13/75, neutral retained 9/75, neutral nasa 13/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

A provisional retained-thermal ATcT energy update is scientifically supported as one bounded improvement. Do not describe the retained thermal model as validated: a source-resolved vibration/electronic update remains necessary for a stronger full-function claim. NASA is a sensitivity, not an automatic replacement.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1021/jp9802735 <https://doi.org/10.1021/jp9802735>`_.
* `DOI 10.1063/1.1288267 <https://doi.org/10.1063/1.1288267>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/cnn_candidate_test.py

The definition is ``tests/data/validation/cnn_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.
