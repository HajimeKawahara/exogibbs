C2O bounded candidates and fixed-network substitution
=====================================================

Recommend the evaluated energy as a bounded improvement while retaining the literal thermal function. The old electronic model remains a material inherited limitation; NASA sensitivity does not resolve its physical completeness. A complete thermal update is deferred, not silently invented. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear triplet C-C-O, X3Sigma, C2O; not singlet or cyclic CCO.

The original 1966 JANAF energy is an estimate with quoted 63 kJ/mol uncertainty. ATcT 1.228 gives D0=1292.503 kJ/mol, a -94.082 kJ/mol change; agreement of older JANAF/NASA anchors does not independently validate the obsolete energy.

Original physical D0=1386.585000000 and selected D0=1292.503000000 kJ/mol. The retained G/H shift is +94.082000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

JANAF includes estimated 4050/6450 cm^-1 electronic terms, whereas later state evidence places low singlets near 5276/8205 cm^-1. Those terms alone do not define complete state-specific rovibrational thermals. NASA is evaluated as an explicit alternative, not as a newly validated electronic-state sum.

The original :doc:`c2o_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - Maximum C share (%)
   * - ions/current
     - 1 / 1
     - 0.00087230905
   * - ions/retained
     - 0.000529465542 / 0.0590806784
     - 5.15366104e-05
   * - ions/nasa
     - 0.000556502376 / 0.0603493084
     - 5.26432479e-05
   * - neutral/current
     - 1 / 1
     - 0.000872322987
   * - neutral/retained
     - 0.000529465542 / 0.0590806784
     - 5.15374338e-05
   * - neutral/nasa
     - 0.000556502376 / 0.0603493084
     - 5.2644089e-05

At the ions current maximum-C-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 0.00087230905 / 5.15366104e-05 / 5.26432479e-05 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-C-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 0.000872322987 / 5.15374338e-05 / 5.2644089e-05 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C2O1 0.000820772 percentage points at 4000 K/100 bar/C/O=1.5, C1O1 0.000412332 percentage points at 4000 K/100 bar/C/O=1.5, C2H2 0.000260246 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute O-allocation changes are C1O1 0.000618498 percentage points at 4000 K/100 bar/C/O=1.5, C2O1 0.000615579 percentage points at 4000 K/100 bar/C/O=1.5, H2O1 3.97277e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C2O1 0.000819666 percentage points at 4000 K/100 bar/C/O=1.5, C1O1 0.000411776 percentage points at 4000 K/100 bar/C/O=1.5, C2H2 0.000259895 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute O-allocation changes are C1O1 0.000617664 percentage points at 4000 K/100 bar/C/O=1.5, C2O1 0.000614749 percentage points at 4000 K/100 bar/C/O=1.5, H2O1 3.96741e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.05e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.4e-13, and formation residual 1.71e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.31e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 55/75, ions nasa 55/75, neutral retained 55/75, neutral nasa 55/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the evaluated energy as a bounded improvement while retaining the literal thermal function. The old electronic model remains a material inherited limitation; NASA sensitivity does not resolve its physical completeness. A complete thermal update is deferred, not silently invented.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1063/1.475839 <https://doi.org/10.1063/1.475839>`_.
* `DOI 10.1063/1.473000 <https://doi.org/10.1063/1.473000>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/c2o_candidate_test.py

The definition is ``tests/data/validation/c2o_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.
