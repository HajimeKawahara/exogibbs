S6 recovered isomer data and thermal sensitivity
=================================================

This follow-up to :doc:`s6_source_review` recovers four numerical Jackson
structures and reconstructs their limited RRHO sum. It shows why a complete
state inventory matters before replacing S6 thermochemistry. Curated remains
unchanged.

Recovered structures and missing ground-state coverage
------------------------------------------------------

The initial author archive contains exactly these S6 records. Relative E0
includes the already-scaled harmonic ZPE and uses the lowest archived entry
as zero; it does not prove that entry is the physical ground state.

.. list-table:: Archived S6 subset
   :header-rows: 1

   * - Record
     - Point group / rotational symmetry number
     - E0 offset (kJ/mol)
   * - S6_buckled
     - C2v / 2
     - 0
   * - S6_stack_S3
     - D3h / 6
     - 28.290813
   * - S6_branched
     - Cs / 1
     - 40.918803
   * - S6_chain_63
     - C1 / 1
     - 57.612947

No entry is labelled armchair, and the stored cyclic ``S6_buckled`` entry is
labelled C2v. The inspected full article by
`Fedyaeva et al. (2023) <https://doi.org/10.1039/D2CP05498A>`_ instead reports
an armchair minimum below a boat by 0.53 eV at B3LYP/6-311+G(d,p).
Its supplemental coordinates and frequencies were not recovered. This
archive therefore does not establish a complete ground-state/isomer inventory.
That separate DFT gap cannot be inserted into the Jackson PBE0 dataset
without identifying matching geometries and consistent energy/ZPE conventions.

What the restricted sum demonstrates
-----------------------------------

.. list-table:: Formal RRHO sensitivity to the archived isomer set
   :header-rows: 1

   * - T (K)
     - Q(sum) / Q(lowest)
     - Chain fraction conditional on four entries
   * - 1500
     - 3.614349
     - 67.3106%
   * - 4000
     - 51.491049
     - 96.0720%

The chain has low modes at 21.7445, 44.6957 and 70.0172 cm\ :sup:`-1`.
Their harmonic entropy makes the formal high-temperature sum sensitive to
state selection; it also requires checking anharmonic or hindered motion
before assigning physical populations. The summed molecular S/Cp at 1500 K
are 620.213/170.270 J/mol/K, versus JANAF 558.207/132.066 and
NASA 563.446/136.706. At 4000 K they are 758.876/133.494, versus
JANAF 688.708/134.024 and NASA 700.682/138.850.

This calculation resolves the earlier lack of any reproducible individual
Jackson input and demonstrates a large possible thermal response. It does
not justify adopting that response: the missing armchair data, incomplete
state count and uncertain soft-mode treatment directly affect the model.
Next work is to obtain the missing individual geometries/frequencies,
reconcile isomers and energy references, and assess thermal motion before
building a candidate or running fixed-network substitution. Further papers
are not required to merge this explicitly bounded evidence review.

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

The new ``tests/data/validation/s6_state_followup/20261105`` packet
records selected numerical inputs, access history, hashes and the complete
diagnostic. Replay with
``python mceg/reviews/sulfur_state_followup.py s6``.
The original source packet, document and receipt remain unchanged as the
historical review; this follow-up supersedes its data-access status where
explicitly stated. MCEG records this reconstruction separately from adoption
and physical-accuracy assessments. No production thermochemistry, curated
record or equilibrium solve changes. The bounded evidence update can merge;
missing production-model ingredients remain an adoption follow-up.
