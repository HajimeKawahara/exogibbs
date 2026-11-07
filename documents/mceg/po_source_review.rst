PO source, energy and thermal-function review
=============================================

Retain current PO. POPS independently checks the thermal functions closely; it supplies no new independently determined formation-energy anchor.

Neutral 31P16O is a heteronuclear X2Pi radical (symmetry number 1). The current FastChem row cites Goos/Burcat2023, whose selected official PO record explicitly fits Gurvich1989 with Hf298=-27.858 +/-3 kJ/mol. NASA9 cites that same evaluation (part 1 p.404, part 2 p.258). Agreement of these fits is shared ancestry, not independent validation. JANAF web gives -23.548 kJ/mol; its sign is negative, regardless of the positive sign in the Burcat parenthetical commentary.

`Prajapat et al. (2017), POPS <https://doi.org/10.1093/mnras/stx2229>`_ supplies 43,148 distributed X2Pi levels, J through 234.5. We use every level, remove the nuclear-spin factor 2, and do not add another electronic/spin-orbit degeneracy. The two spin-orbit components and parity levels are already present. The selected potential De=48,980 cm-1 is an adopted asymptote; refined low-lying transition energies do not determine a new physical D0.

At 4000 K the normalized POPS/NASA Q ratio is 1.00426; H-H0, S and Cp differ by -0.09146 kJ/mol, +0.01244 and -0.21263 J/mol/K. This is too small to justify replacing the existing functions with a ground-electronic-state-only model. The current/NASA reaction K ratio is 1.00027 there. The larger current/JANAF ratio mainly reflects the energy anchor.

Next: retain production PO, and revisit only if an independently improved dissociation energy or material excited-state contribution becomes available. The published 5000 K line-list range is not itself a completeness guarantee for thermodynamic moments.

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
     - 0.991840
     - 1.408146
     - 1.003619
   * - 2000
     - 0.991882
     - 1.288934
     - 1.003600
   * - 3000
     - 0.995421
     - 1.185163
     - 1.002573
   * - 4000
     - 1.000271
     - 1.141404
     - 1.000792

Archived importance and limits
------------------------------

The existing curated-cos-v1 75-case ion-enabled archive contains 537 species and reaches a phosphorus allocation of 68.67 percent in PO. All endpoints converged and their target potential matches the named current factory. This is gas-only allocation over 1500-4000 K, 1e-6-100 bar and three C/O values, not a new substitution response or a universal planetary abundance claim.

Numerical reproduction, source attribution and scoped retention are supported. Complete physical uncertainty, state completeness and atmospheric accuracy remain unknown. Missing ancestral books alone do not block merging this evidence packet. Curated and runtime files are unchanged.

Replay and MCEG
---------------

Run ``python mceg/reviews/phosphorus_diatomic_source_review.py po`` and ``pytest tests/model_evidence/po_source_review_test.py``. Inputs, full compact state arrays where available, source hashes, diagnostics and receipt live in ``tests/data/validation/po_source_review/20261106``; the MCEG record is ``mceg/contracts/po_source_review.yaml``. Full papers are not committed.
