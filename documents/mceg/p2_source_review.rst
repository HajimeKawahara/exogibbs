P2 source, energy and thermal-function review
=============================================

Retain the current JANAF fit. A 2026 theoretical energy comparison is now quantitative, but does not establish a compelling wholesale replacement.

Neutral 31P2 has an X1Sigma_g+ ground state, rotational symmetry number 2 and nuclear-spin product 4. Homonuclear exchange weights must be treated consistently; no new P2 state sum or extra spin factor is introduced here. The current FastChem row cites Chase1998. Selected official NASA9 records cite Gurvich1989, part 1 p.398 and part 2 p.255.

`Barman et al. (2026) <https://doi.org/10.1021/acs.jpca.6c00467>`_ Table 4 gives physical TAE0=116.38 and 116.23 kcal/mol for W5prelim1 and W5prelim2, respectively, versus W4-17 116.22. The 116.05 +/-0.09 comparison entry is explicitly Gurvich (footnote a), despite the broad ATcT column heading. The inspected downloaded PDF is the published article. Clamped-nuclei energy and a potential-well De must not be substituted for this zero-point-inclusive TAE0.

The JANAF neutral-atom D0 reconstructed here is 485.637 kJ/mol. W5prelim2 exceeds it by 0.66932 kJ/mol: a deliberately conditional energy-only multiplier of 1.0551 at 1500 K and 1.0203 at 4000 K with thermal functions held fixed. The 0.15 kcal/mol difference between W5prelim approximations is not a statistical uncertainty, but counsels against declaring a decisive new energy standard. There is no new thermal model or equilibrium substitution.

Next: an optional energy-only W5prelim2 sensitivity candidate can reuse the retained thermal functions, with the theoretical-method spread explicit. A full replacement needs a state inventory, homonuclear exchange convention and high-temperature caloric check. Missing Gurvich full text alone does not block merging this source packet.

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
     - 1.016212
     - 1.004978
     - 1.000874
   * - 2000
     - 1.009721
     - 1.001457
     - 1.000535
   * - 3000
     - 0.999692
     - 0.996141
     - 0.998438
   * - 4000
     - 0.996502
     - 1.000246
     - 0.992438

Archived importance and limits
------------------------------

The existing curated-cos-v1 75-case ion-enabled archive contains 537 species and reaches a phosphorus allocation of 74.11 percent in P2. All endpoints converged and their target potential matches the named current factory. This is gas-only allocation over 1500-4000 K, 1e-6-100 bar and three C/O values, not a new substitution response or a universal planetary abundance claim.

Numerical reproduction, source attribution and scoped retention are supported. Complete physical uncertainty, state completeness and atmospheric accuracy remain unknown. Missing ancestral books alone do not block merging this evidence packet. Curated and runtime files are unchanged.

Replay and MCEG
---------------

Run ``python mceg/reviews/phosphorus_diatomic_source_review.py p2`` and ``pytest tests/model_evidence/p2_source_review_test.py``. Inputs, full compact state arrays where available, source hashes, diagnostics and receipt live in ``tests/data/validation/p2_source_review/20261106``; the MCEG record is ``mceg/contracts/p2_source_review.yaml``. Full papers are not committed.
