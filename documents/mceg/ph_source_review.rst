PH source, energy and thermal-function review
=============================================

Retain the already corrected Lodders1999 PH source. LaTY supports a useful thermal cross-check, but its upper-temperature completeness needs care.

Neutral 31PH is heteronuclear (symmetry number 1), with X3Sigma- ground and a1Delta excited states. The current FastChem row cites `Lodders1999 <https://doi.org/10.1063/1.556046>`_. Tables 2/3 and section 2.1 give Hf298=236.09 +/-33.5 kJ/mol on the white-P reference, versus the uncorrected JANAF web value 253.55. Current already uses the corrected evaluation; applying another -17.46 kJ/mol would double-correct it. NASA9 instead uses Gurvich1989 Hf298=230.752104 kJ/mol. The comparison to uncorrected JANAF is retained only as a diagnostic of the misleading screening difference.

`Langleben et al. (2019), LaTY <https://doi.org/10.1093/mnras/stz1856>`_ supplies 2528 levels of X3Sigma- and a1Delta through J=80. The nuclear-spin product is 4, removed once. No additional triplet factor is applied. The physical lowest level is J=1, v=0, N=0; the paper gives ZPE=1170.47 cm-1 above the potential minimum. Its adopted De=25,700 cm-1 uses an older D0 around 293.2 kJ/mol; this is not a new dissociation measurement from the line fit.

At 4000 K the distributed-state Q/NASA ratio is 0.97270, with H-H0 lower by 0.98330 kJ/mol, S lower by 0.47600 and Cp lower by 1.46165 J/mol/K. The a1Delta contribution is 4.046 percent of this finite Q. The downloaded 4000 K PF is 8106.7600, whereas paper Table 10 prints 8112.16: that 0.0666 percent version discrepancy is recorded, not silently forced to equality. Direct summation reproduces the downloaded table to approximately 7e-7 relative. Some distributed a-state levels exceed the paper's simple ground D0 cutoff wording; all are retained with their actual energies.

Next: inspect omitted higher electronic states and thermal moments before proposing a LaTY candidate, then pair any justified thermal function with a separately evaluated energy. The current corrected enthalpy has a broad quoted uncertainty; an old potential asymptote is insufficient reason to replace it. The apparent factor-of-four JANAF discrepancy at 1500 K is principally an already-corrected reference problem, not an unresolved curated bug.

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
     - 0.645293
     - 4.066537
     - 0.995723
   * - 2000
     - 0.718407
     - 2.860835
     - 0.995595
   * - 3000
     - 0.796310
     - 2.008756
     - 0.993020
   * - 4000
     - 0.837598
     - 1.690315
     - 0.986666

The extra evaluated-source comparison combines the explicitly transcribed Lodders molecular nodes with NASA neutral atoms. It is a labelled source diagnostic, not an exact reconstruction of every historical FastChem atomic fit. The remaining percent-level fit/source difference is retained in the replay.

Archived importance and limits
------------------------------

The existing curated-cos-v1 75-case ion-enabled archive contains 537 species and reaches a phosphorus allocation of 37.14 percent in PH. All endpoints converged and their target potential matches the named current factory. This is gas-only allocation over 1500-4000 K, 1e-6-100 bar and three C/O values, not a new substitution response or a universal planetary abundance claim.

Numerical reproduction, source attribution and scoped retention are supported. Complete physical uncertainty, state completeness and atmospheric accuracy remain unknown. Missing ancestral books alone do not block merging this evidence packet. Curated and runtime files are unchanged.

Replay and MCEG
---------------

Run ``python mceg/reviews/phosphorus_diatomic_source_review.py ph`` and ``pytest tests/model_evidence/ph_source_review_test.py``. Inputs, full compact state arrays where available, source hashes, diagnostics and receipt live in ``tests/data/validation/ph_source_review/20261106``; the MCEG record is ``mceg/contracts/ph_source_review.yaml``. Full papers are not committed.
