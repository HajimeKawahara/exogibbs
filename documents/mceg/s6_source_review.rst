S6: source energy, thermal functions and next calculation
=========================================================

Retain the current JANAF fit provisionally; the remaining factor-of-two thermal discrepancy is not resolved by a newer DFT isomer calculation alone.

This 2026-11-05 review follows issues #308 and #309. It changes no runtime
function, packaged thermochemical record or default preset. It compares
6 S(g) -> S6(g), audits the official source transcription, and reads the
latest archived ``curated-cos-v1`` grid. It does not run a replacement solve.

Source identity and conventions
-------------------------------

The current S6 record remains the inherited FastChem/JANAF fit. Official
`JANAF S-019 <https://janaf.nist.gov/tables/S-019.html>`_
labels its evaluation September 1977 and conversion from 1 atm to 1 bar.
Eight positive-temperature rows were inspected in the official HTML and
transcribed; the four original screening nodes and Hf298 agree exactly.
The original printed evaluation notes were not read. NASA Glenn report
`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_, printed
page 184, attributes its S6 record to Gurvich 1989, volume 1,
part 1 page 282 and part 2 page 170. The printed record
was visually inspected and all 18 coefficients match the frozen screening.
The original Gurvich pages remain uninspected; no DOI has been verified.

Both source reactions are balanced against their own neutral S atomic
functions. The decomposition uses a common R = 8.31446261815324 J/mol/K;
a separate printed NASA header check uses its historical R = 8.31451.
Pressure is 1 bar throughout; there is no extra 1 atm conversion. H298-H0 in
a NASA record is a thermal increment, not a formation enthalpy at 0 K.
The H298 alignment removes the *whole balanced reaction* energy difference.
No extra spin, symmetry, isotope, mass or statistical-weight correction is
imposed on either published function. Their original detailed state inventory
and mass assumptions have not been independently reconstructed. A future
partition candidate must state them explicitly and avoid counting a state or
isomer both inside a table and as an additional factor.

Numerical comparison
--------------------

Ratios below use dimensionless equilibrium constants for the same balanced
neutral-atom reaction. ``Aligned`` removes the balanced H298 difference from
JANAF only; it is a diagnostic, not a candidate model.

.. list-table:: Source ratios at exact table nodes
   :header-rows: 1

   * - T (K)
     - JANAF / NASA
     - Aligned JANAF / NASA
     - Current / JANAF
   * - 1500
     - 0.528613
     - 0.607654
     - 1.01906
   * - 2000
     - 0.516647
     - 0.573565
     - 0.993483
   * - 3000
     - 0.454738
     - 0.487552
     - 0.982698
   * - 4000
     - 0.395916
     - 0.417155
     - 1.00598

The inherited formation K differs from that reconstructed from sampled JANAF
H/S by less than 3%, so fitting error does not explain the source disagreement.
Energy alignment still leaves a substantial thermal difference. The archived JSON separates
H298, enthalpy increments, S298 and entropy increments for the molecule and
its atomic references. Printed formation-G values independently close the
H/S reconstruction within the propagation of their printed rounding. Neither closer
NASA agreement nor a newer publication alone establishes physical accuracy.

The official HTML's unused 298.15 K free-energy-function cell differs from
S - (H-H298)/T by 168.803 J/mol/K. The review retains and flags
this inconsistency; it uses H and S, never that cell, to reconstruct K. Other
inspected positive-temperature cells agree within their printed rounding (at
most 0.00134 J/mol/K). This does not
identify the historical origin of the cell or establish that the printed
edition contains the same inconsistency. It is not a reason to edit curated.

Modern evidence and its limits
------------------------------

Fedyaeva et al. (2023), DOI `10.1039/D2CP05498A <https://doi.org/10.1039/D2CP05498A>`_,
provide a global structure search followed by B3LYP/6-311+G(d,p) refinement of
the ten lowest structures. The inspected full article reports an S6 armchair
structure 0.53 eV lower than the boat structure. This is a difference between
isomers in that computational model, not a ground-state atomization energy,
uncertainty estimate or a correction to the current fit. Their definition of
atomization energy is negative per atom; it must not be confused with positive
D0 per molecule. Their finite-temperature comparisons extend to 1500 K,
which does not validate the intended upper limit of 4000 K.

Jackson et al. (2016) inspect several cyclic, branched and prism structures.
Their calibrated DFT/RRHO sulfur-vapour model offers a useful alternative
thermal construction, but its species inventory, symmetry factors and energy
reference must be reproduced explicitly. The sulfur-mixture chemical potential
cannot replace the S6 standard Gibbs function. The numerical supporting
coordinates/frequencies of either model were not reproduced in this review.
A new date or a better fit to NASA alone is not sufficient for adoption.

Common alternatives and source access
-------------------------------------

`Jackson et al. 2016 <https://doi.org/10.1039/C5SC03088A>`_ was inspected as a
full author manuscript. Its DFT/RRHO construction uses calibrated vibrational
frequencies and an experimental S8 energy reference. Its individual-species
functions are a potential candidate source, while its mixture function is
not interchangeable with any one species. The code DOI is
``10.5281/zenodo.28536`` and the paper's DFT-data DOI is
``10.6084/m9.figshare.1513736``; neither dataset was reproduced here.
`Fedyaeva et al. 2023 <https://doi.org/10.1039/D2CP05498A>`_ was inspected as a
full article on the author host; its supporting data were not recovered.
`Rau et al. 1973 <https://doi.org/10.1016/S0021-9614(73)80045-X>`_ was inspected
only through the publisher abstract and metadata. Its 823--1273 K mixture
density evidence does not directly establish a single-species error bound.

The source manifest records actual access separately from citation identity,
plus hashes of inspected public PDFs without committing the articles. Missing
original pages do not block merging this bounded source review; unresolved
claims remain unknown in MCEG.

Archived atmospheric footprint
-------------------------------

The existing ``curated_cos_adoption/20261105/ions.npz`` archive contains
75 cases and 537 species, 1500--4000 K, 1e-6--100 bar, and reference C/O,
1 and 1.5. All archived cases report convergence. The maximum sulfur fraction
in S6 is 1.21992857e-30 at 1500 K,
0.0001 bar and C/O = 0.58884366; its mole fraction there
is 4.59697172e-36. The sulfur share counts all 6 atoms. The target
thermodynamic column matches the named current factory, and the maximum
sulfur conservation residual is 2.936e-12.

These trace values describe this archive only. Numerical convergence does not
guarantee relative accuracy of an extraordinarily small component, nor bound
physical abundances. Cooler sulfur-rich atmospheres, photochemistry and
condensation are outside this archive. No new solve or abundance sensitivity
to a replacement was performed.

Follow-up calculations and references
-------------------------------------

* Recover S6 individual coordinates, frequencies, isomer energies and symmetry factors from Fedyaeva 2023 supporting information (10.1039/D2CP05498A) and Jackson 2016 data (10.1039/C5SC03088A).
* Construct a state-counted RRHO candidate with a declared atomic energy reference; compare S298, H increments and Cp, including isomer contributions and temperature coverage.
* Inspect the original Gurvich 1989 S6 evaluation and Rau 1973 full text (10.1016/S0021-9614(73)80045-X) for historical energy/thermal assumptions; record uncertainty gaps without requiring complete physical validation to merge the review.

Reproduction and MCEG
---------------------

Run ``python mceg/reviews/s6_source_review.py`` and
``pytest tests/model_evidence/s6_source_review_test.py`` offline. The
fixture in ``tests/data/validation/s6_source_review/20261105`` pins
source facts, consumed implementation/data files and numerical outputs.
The named ``curated_cos`` factory prevents a future default switch from
silently changing this review.

``mceg/contracts/s6_source_review.yaml`` marks scoped source attribution,
modern-source screening and numerical replay supported. Complete physical
accuracy and adoption readiness remain unknown. The latter can be assessed
for a limited candidate without requiring complete physical validation of all
legacy literature. No uncertainty band is inferred from source spread.
