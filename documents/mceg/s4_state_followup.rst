S4 recovered state and energy data
==================================

This follow-up to :doc:`s4_source_review` recovers a physical W4 atomization
energy and two numerical Jackson structures. It narrows two earlier evidence
gaps but does not establish a complete high-temperature S4 model. Curated
remains unchanged.

Physical energy, with reference temperatures kept distinct
----------------------------------------------------------

`Karton et al. (2008), DOI 10.1021/jp801805p
<https://doi.org/10.1021/jp801805p>`_, Table 2 (printed page 12872), explicitly
lists the following W4-08 values for C2v S4. The row and footnotes were visually
inspected in the full primary article.

.. list-table:: S4 atomization energies
   :header-rows: 1

   * - Quantity
     - kcal/mol
   * - Nonrelativistic clamped-nuclei TAEe, no ZPE
     - 234.35
   * - Physical TAEe, relativistic/DBOC, no ZPE
     - 231.44
   * - Physical TAE0, at 0 K
     - 228.15

The physical TAE0 is **954.5796 kJ/mol**. Its zero-point and physical atomic
reference corrections must not be subtracted again. This is the inspected
2008 W4-08 table, not a claim that the 2011 W4-11 supporting file was obtained.

The inspected `JANAF S-017 table <https://janaf.nist.gov/tables/S-017.txt>`_
has no numerical 0/100/200/250 K entries or molecular H298-H0. Its balanced
298.15 K dissociation enthalpy is
4 x 276.980 - 145.771 = **962.149 kJ/mol**. This has a different reference
temperature from W4 TAE0, so their direct difference is not an energy error.
An intrinsic JANAF D0 cannot be derived from that table alone.

One explicitly hybrid sensitivity uses the recovered Jackson molecular
H298-E0 = 17.104917 kJ/mol and JANAF atomic H298-H0 = 6.657 kJ/mol:
D0 = 962.149 - 4 x 6.657 + 17.104917 = **952.625917 kJ/mol**.
W4 is 1.953683 kJ/mol above this hybrid value. This depends on the Jackson
thermal assumptions; it is neither a recovered legacy JANAF D0 nor a proposed
energy correction. The energy datum alone does not select a thermal model.

What the two-structure thermal calculation resolves
---------------------------------------------------

The archive contains ``S4_eclipsed`` (C2v, symmetry number 2) and
``S4_buckled`` (D2d, symmetry number 4). Including their already-scaled ZPE,
the second lies 43.174073 kJ/mol above the first. Adding this ring to the
lowest archived structure gives:

.. list-table:: Conditional contribution of the second structure
   :header-rows: 1

   * - T (K)
     - Q(sum) / Q(lowest)
     - Ring fraction in this two-entry set
   * - 1500
     - 1.000976
     - 0.09750%
   * - 4000
     - 1.007700
     - 0.76408%

The corresponding summed molecular S/Cp are 456.003/82.574 J/mol/K at
1500 K, compared with JANAF 437.266/84.271 and NASA 418.108/82.184.
At 4000 K they are 537.376/83.147, versus JANAF 523.581/92.911 and
NASA 499.243/83.008. These differences do not make any source correct merely
by agreement. This particular second structure has a small partition effect,
but that result places no bound on an omitted triplet.

`Matus et al. (2007) <https://doi.org/10.1063/1.2774973>`_ still has only
abstract-level state evidence in this review. Its 10.8 kcal/mol triplet gap
and the earlier illustrative 3 exp(-gap/RT) weights cannot supply triplet
rovibrational entropy, low-barrier corrections or complete state counting.
A production candidate still needs those inputs and a consistent handling of
symmetry, possible equivalent wells and anharmonic motion. No unavailable
state was invented and no partition weight was treated as a physical error
bound.

Numerical reconstruction
------------------------

The author archive associated with `Jackson et al. (2016)
<https://doi.org/10.1039/C5SC03088A>`_ was recovered from the
`v1.0 GitHub tag <https://github.com/WMD-Bath/sulfur-model/tree/v1.0>`_
linked to `Zenodo 28536 <https://doi.org/10.5281/zenodo.28536>`_. This is the
initial submission archive; it is not asserted to reproduce every result in
the final paper. The selected PBE0 numerical records include coordinates,
frequencies, electronic energies and rotational symmetry numbers. The 0.96
frequency factor is already present. The original GPL-3.0 code is neither
copied nor executed; the new helper independently evaluates standard RRHO
relations from selected numerical facts. The Figshare raw-calculation archive
metadata was inspected, but its full 74.8 MB archive was not downloaded.

The reconstruction retains the largest 3N-6 positive frequencies and checks
that the six discarded translation/rotation residues are below 2 cm\ :sup:`-1`.
It computes inertias from the coordinates with declared sulfur mass 32.065 u,
uses 1 bar, and adds the retained-mode ZPE to relative electronic energies.
These energies are relative within one molecular size; the author's absolute
S8 calibration is not adopted. Each stored geometry has one electronic
component, with its supplied rotational symmetry number and no added spin or
enantiomer factor. The source code's spin=0 setting is an archive convention,
not evidence that all relevant physical states are singlets.

For state offsets E0_i and excitation functions q_i, the internal sum is
sum_i q_i exp(-E0_i/RT). Enthalpy includes the population-weighted excitation
energy; Cp also includes its variance divided by RT squared. Finite-difference
checks verify dH/dT=Cp and -dG/dT=S, including changing conditional weights.
Reported state fractions are conditional on this small archived set, not
atmospheric abundances or a complete physical isomer population.
The 250-4000 K computation is a formal RRHO diagnostic; in particular,
1500-4000 K is an extrapolation stress test rather than a validity claim.

Reproducibility and decision
----------------------------

The new ``tests/data/validation/s4_state_followup/20261105`` packet
records selected numerical inputs, access history, hashes and the complete
diagnostic. Replay with
``python mceg/reviews/sulfur_state_followup.py s4``.
The original source packet, document and receipt remain unchanged as the
historical review; this follow-up supersedes its data-access status where
explicitly stated. MCEG records this reconstruction separately from adoption
and physical-accuracy assessments. No production thermochemistry, curated
record or equilibrium solve changes. The bounded evidence update can merge;
missing production-model ingredients remain an adoption follow-up.
