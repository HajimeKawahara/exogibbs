PS source, energy and thermal-function review
=============================================

Retain Lodders2004 pending a scoped thermal candidate. POPS shows a material high-temperature heat-capacity difference despite a small Q difference.

Neutral 31P32S is heteronuclear X2Pi (symmetry number 1). The current FastChem row cites `Lodders2004 <https://doi.org/10.1063/1.1611178>`_, Tables 8/9 and section 2.4: D0=438 +/-10 kJ/mol from Drowart1973, Hf298=152.44 +/-10 kJ/mol, with Gurvich1989 thermal functions. This corrects inconsistent JANAF historical dissociation/formation data. NASA9 uses the same thermal ancestry and D0 approximately 440 kJ/mol. Current is already an evaluated correction; closeness to raw JANAF is not a selection criterion.

`Prajapat et al. (2017), POPS <https://doi.org/10.1093/mnras/stx2229>`_ provides 225,997 distributed levels across X2Pi, a4Pi and B2Pi, J through 320.5. The full state file includes blank lines, which are ignored rather than counted as levels. The nuclear-spin factor is 2; spin-orbit components, parity and electronic multiplicities are already resolved in the states. The chosen potential De=37,004 cm-1 is not D0 and is not a new independent absolute-energy measurement.

At 4000 K POPS/NASA Q is only 1.01450, but H-H0 is higher by 2.92685 kJ/mol, S by 0.85143 and Cp by 4.21479 J/mol/K. Excited a4Pi and B2Pi together contribute 1.626 percent of the finite Q. Thus a small partition-function difference does not imply negligible caloric differences. The top fifth of the distributed energy range carries 0.188 percent of Q; this is a sensitivity diagnostic, not an upper bound on missing states.

Next: prioritize a POPS thermal candidate with the retained Lodders D0, explicitly checking electronic-state limits and the temperature derivatives, then perform a fixed-network replacement calculation. Compare energy-only and thermal-only variants before any adoption. The present review does not silently replace the experimental-energy evaluation with a fitted potential De.

Normalized comparison
---------------------

All reactions form the neutral molecule from neutral gas atoms at 1 bar. NASA uses its native R=8.314510 for dimensional H/S/Cp; reaction logarithms use R=8.31446261815324. Official molecular intervals start at 300 K, so H0 is reconstructed from header H298 and H298-H0 rather than extrapolating a polynomial to zero or 298.15 K. JANAF uses exact H/S nodes and its own atomic tables. Energy H0, thermal H-H0 and -T S terms are reported separately; changing the phosphorus elemental reference consistently cancels in a balanced reaction. Isotopologue state sums use natural-mass translation only as an explicitly labelled elemental comparison proxy, with the isotope translation offset reported separately.

.. list-table:: Reaction and thermal-source diagnostics
   :header-rows: 1

   * - T (K)
     - Current/NASA K
     - Current/JANAF K
     - H0-aligned JANAF/NASA K
   * - 1500
     - 0.839053
     - 0.325506
     - 1.023223
   * - 2000
     - 0.874150
     - 0.427091
     - 1.023575
   * - 3000
     - 0.907148
     - 0.558478
     - 1.023392
   * - 4000
     - 0.926338
     - 0.641315
     - 1.021470

The extra evaluated-source comparison combines the explicitly transcribed Lodders molecular nodes with NASA neutral atoms. It is a labelled source diagnostic, not an exact reconstruction of every historical FastChem atomic fit. The remaining percent-level fit/source difference is retained in the replay.

Archived importance and limits
------------------------------

The existing curated-cos-v1 75-case ion-enabled archive contains 537 species and reaches a phosphorus allocation of 21.02 percent in PS. All endpoints converged and their target potential matches the named current factory. This is gas-only allocation over 1500-4000 K, 1e-6-100 bar and three C/O values, not a new substitution response or a universal planetary abundance claim.

Numerical reproduction, source attribution and scoped retention are supported. Complete physical uncertainty, state completeness and atmospheric accuracy remain unknown. Missing ancestral books alone do not block merging this evidence packet. Curated and runtime files are unchanged.

Replay and MCEG
---------------

Run ``python mceg/reviews/phosphorus_diatomic_source_review.py ps`` and ``pytest tests/model_evidence/ps_source_review_test.py``. Inputs, full compact state arrays where available, source hashes, diagnostics and receipt live in ``tests/data/validation/ps_source_review/20261106``; the MCEG record is ``mceg/contracts/ps_source_review.yaml``. Full papers are not committed.
