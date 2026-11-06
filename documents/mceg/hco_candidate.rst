HCO bounded candidates and fixed-network substitution
=====================================================

Recommend the retained-thermal ATcT energy update. Retain NASA as a distinct thermal/reference sensitivity, without declaring it the superior full-range thermal function. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral bent doublet formyl H-C=O, C1H1O1; distinct from HOC and formaldehyde.

The selected D0=1132.863 kJ/mol uses same-version ATcT C/H/O atoms and HCO. The +2.004 kJ/mol change relative to JANAF is linked to the evaluated formaldehyde/formyl threshold network. NASA already cites Terentis 1996, so its evidence is not wholly independent of the modern network.

Original physical D0=1130.859000000 and selected D0=1132.863000000 kJ/mol. The retained G/H shift is -2.004000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

After energy alignment the source thermal contrast is about 4.2%. NASA molecular and atomic thermals replace the inherited fit residual in the second candidate. Neither a close K ratio nor a newer source establishes complete high-temperature bound-state accuracy.

The original :doc:`hco_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 0.0475701582
   * - ions/retained
     - 1.06207752 / 1.17431367
     - 0.0505231951
   * - ions/nasa
     - 1.01527548 / 1.11209343
     - 0.048296815
   * - neutral/current
     - 1 / 1
     - 0.0475702387
   * - neutral/retained
     - 1.06207752 / 1.17431367
     - 0.0505232806
   * - neutral/nasa
     - 1.01527548 / 1.11209343
     - 0.0482968967

At the ions current maximum-C-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 0.0475701582 / 0.0505231951 / 0.048296815 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-C-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 0.0475702387 / 0.0505232806 / 0.0482968967 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C1H1O1 0.00295304 percentage points at 4000 K/100 bar/C/O=0.588844, C1O1 0.00291874 percentage points at 4000 K/100 bar/C/O=0.588844, C2H2 2.74086e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/retained, the largest absolute O-allocation changes are C1H1O1 0.00293888 percentage points at 4000 K/100 bar/C/O=1.5, C1O1 0.00289228 percentage points at 4000 K/100 bar/C/O=1.5, H2O1 8.76005e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C1H1O1 0.000726657 percentage points at 4000 K/100 bar/C/O=0.588844, C1O1 0.000718216 percentage points at 4000 K/100 bar/C/O=0.588844, C1H4 1.20785e-05 percentage points at 2500 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute O-allocation changes are C1H1O1 0.000723173 percentage points at 4000 K/100 bar/C/O=1.5, C1O1 0.000711706 percentage points at 4000 K/100 bar/C/O=1.5, H2O1 2.15558e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.43e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.23e-13, and formation residual 1.48e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.05e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 66/75, ions nasa 60/75, neutral retained 66/75, neutral nasa 60/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the retained-thermal ATcT energy update. Retain NASA as a distinct thermal/reference sensitivity, without declaring it the superior full-range thermal function.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1016/0009-2614(96)00705-1 <https://doi.org/10.1016/0009-2614(96)00705-1>`_.
* `DOI 10.1021/acs.jpca.5b01346 <https://doi.org/10.1021/acs.jpca.5b01346>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/hco_candidate_test.py

The definition is ``tests/data/validation/hco_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.
