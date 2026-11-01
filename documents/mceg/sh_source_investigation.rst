SH source investigation (2026-11-02)
====================================

The SH review retains the existing **NASA CEA thermal functions plus balanced
ATcT 1.172 atomization energy**, with its conditional status. The large internal
JANAF formation-column discrepancy in issue #308 is a documented publication
error. It does not justify a new correction to the current curated preset.
This closes the scoped L0--L3 source investigation; high-temperature physical
accuracy and abundance effects remain unassessed. Production data are unchanged.

Identity and inspected sources
------------------------------

``H1S1`` denotes neutral gas-phase sulfanyl (SH, HS, mercapto), CAS 13940-21-1,
with ground electronic state X :sup:`2` Pi. It is distinct from SH-minus and
H2S. The `JANAF H-046 table <https://janaf.nist.gov/tables/H-046.html>`_
is dated June 1977 (1 bar). Its original description on printed page 1306 of
the `fourth-edition hydrogen volume
<https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Hydrogen.pdf>`_
includes the 0/377 cm\ :sup:`-1` spin-orbit components and higher electronic
states. The present record does not replace natural-element thermal functions
with a single-isotopologue partition function.

The SH ``g10/01`` record in Appendix D of
`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_ has the same
18 coefficients and 200--1000 / 1000--6000 K bounds as the frozen baseline.
The existing official-derived major-molecule fixture starts its low interval
at 300 K; its coefficients are identical. Its annotations identify Continetti
1991 for D0 and Gurvich 1989, volume 1 page 296 / volume 2 page 179 for thermal
functions. This confirms the selected SH record, not the complete local NASA
file's acquisition history. The original Gurvich pages have not been inspected.

`Continetti, Balko and Lee (1991)
<https://doi.org/10.1016/0009-2614(91)90097-S>`_, Chemical Physics Letters 182,
400--405, measured SH photofragment translational energies. The
`institutional preprint <https://escholarship.org/uc/item/25w9z95s>`_
(pages 8--9) assigns D0 = 3.62 +/- 0.03 eV to SH X :sup:`2` Pi\ :sub:`3/2`.
Its analysis assumed equal populations for the sulfur :sup:`3` P spin-orbit
products. The preprint was inspected through the web reader; direct downloads
did not return PDF bytes, so no original-file hash is claimed.

`Lodders (2004) <https://doi.org/10.1063/1.1611178>`_, pages 359--361,
identifies erroneous formation-H, formation-G and log-K columns introduced
between JANAF editions; Hf(298.15 K) is the exception. The fourth edition
reprinted the error despite announcing a correction. The exact transfer
arithmetic was not established. Her Table 3 uses Hf298 = 142.920 kJ/mol and
Gurvich thermal functions, with direct rovibrational summation preferred over
the older approximation. Selected Table 3 nodes are archived separately from
the old JANAF values. These are related evaluations, not independent experiments.

The inspected `ATcT 1.172 SH record
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=344>`_
matches the already adopted Hf0 = 142.81 and Hf298 = 143.26 kJ/mol,
with reported 95% confidence limits of +/- 0.18 kJ/mol. Its provenance includes
`Zhou et al. (2005) <https://doi.org/10.1063/1.1961565>`_ (JCP 123, 054330),
whose publisher-indexed abstract reports D0 = 29245 +/- 25 cm\ :sup:`-1`.
The latter paper's full text remains uninspected. ATcT reports strong
correlation with H2S and nonzero correlation with atomic S; database values and
their inputs must not be counted as independent evidence or combined as
uncorrelated uncertainties. The current choice remains an energy anchor,
not a validation of heat capacities or high-temperature partition completeness.

Numerical diagnosis
-------------------

At 298.15 K the molecular properties are as follows. Hf is relative to each
evaluation's elemental standard states, not the neutral-atom reaction energy.
The NASA Cp and S values use its native R; ATcT supplies no replacement thermal
functions in this review.

.. list-table:: Molecular anchors and thermal functions
   :header-rows: 1

   * - Source
     - Hf (kJ/mol)
     - Cp (J/mol/K)
     - S (J/mol/K, 1 bar)
   * - JANAF
     - 139.327
     - 32.447
     - 195.628
   * - NASA
     - 142.135309
     - 32.479391
     - 195.553975
   * - Lodders 2004
     - 142.920
     - 32.479
     - 195.552
   * - ATcT 1.172
     - 143.26
     - Not supplied
     - Not supplied

All ratios below use H(g) + S(g) -> SH(g), dimensionless Kp and p0 = 1 bar.
Fresh official H-046, H-001 and S-006 text tables match all 12 molecular/atomic
nodes in baseline-001. The baseline is preserved. The raw-column difference
is log10(K from formation-G columns / K from reconstructed H/S).

.. list-table:: Source comparisons
   :header-rows: 1

   * - T (K)
     - JANAF H/S / NASA
     - Raw-column difference (dex)
     - Curated / NASA
   * - 500
     - 1.895192
     - 0.950495
     - Outside curated range
   * - 1000
     - 1.378821
     - 1.529923
     - Outside curated range
   * - 1500
     - 1.238634
     - 2.234328
     - 1.064472
   * - 3000
     - 1.105946
     - 3.086537
     - 1.031732

The 1500 K raw-column error is separate from the 24% reconstructed-source
difference. With delta denoting JANAF minus NASA reaction quantities,
delta ln K = -[delta H298/R + delta thermal-H/R]/T + delta S/R.
At 1500 K these terms are -315.172945 K, -9.361724 K and -0.00234750,
giving delta ln K = 0.214008946. The anchor dominates; the residual thermal
terms remain explicit. These diagnostics use the baseline's common R scale;
NASA's native coefficient constant is 8.314510 J/mol/K.

The adopted balanced ATcT energy is
D0 = Hf0(H) + Hf0(S) - Hf0(SH) = 350.054 kJ/mol.
Using the existing source conventions gives an energy shift of
93.7180738 K, hence delta ln K = 93.7180738/T relative to raw NASA.
This reproduces direct curated evaluation at 1500 and 3000 K. Updating only
SH's formation enthalpy would even give the wrong sign: atomic S must be
updated consistently. No additional ATcT correction is applied.

Replay, disposition and remaining evidence
------------------------------------------

Run offline from the repository root::

   python mceg/reviews/sh_source_investigation.py --check
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     pytest -q tests/model_evidence/sh_source_investigation_test.py

The dated archive under
``tests/data/validation/sh_source_investigation/20261102/`` contains official
tables, acquisition hashes, source search/inspection notes and the numerical
comparison. ``mceg/contracts/sh_source_investigation.yaml`` separates inspected
source support and numerical replay from unknown physical/application support.
No MCEG runtime dependency is introduced.

Retain the current conditional source choice. Before claiming physical accuracy,
inspect the Gurvich state sums and Zhou full text, quantify isotope/fine-structure
and excited-state completeness over 1500--4000 K, and obtain correlated reaction
uncertainties. A future candidate change requires a separately recorded fixed
network abundance comparison and numerical diagnostics; source ratios here are
not abundance errors. The exact historic column-generation bug remains unknown.
