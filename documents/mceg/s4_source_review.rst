S4: source energy, thermal functions and next calculation
=========================================================

Keep the inherited JANAF fit provisionally; the large entropy/thermal difference and thermally accessible electronic states need a declared state sum before replacement.

This 2026-11-05 review follows issues #308 and #309. It changes no runtime
function, packaged thermochemical record or default preset. It compares
4 S(g) -> S4(g), audits the official source transcription, and reads the
latest archived ``curated-cos-v1`` grid. It does not run a replacement solve.

Source identity and conventions
-------------------------------

The current S4 record remains the inherited FastChem/JANAF fit. Official
`JANAF S-017 <https://janaf.nist.gov/tables/S-017.html>`_
labels its evaluation September 1977 and conversion from 1 atm to 1 bar.
Eight positive-temperature rows were inspected in the official HTML and
transcribed; the four original screening nodes and Hf298 agree exactly.
The original printed evaluation notes were not read. NASA Glenn report
`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_, printed
page 184, attributes its S4 record to Gurvich 1989, volume 1,
part 1 page 280 and part 2 page 168. The printed record
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
     - 3.71094
     - 8.88808
     - 1.01273
   * - 2000
     - 4.81469
     - 9.26963
     - 0.99601
   * - 3000
     - 6.60098
     - 10.2158
     - 0.988863
   * - 4000
     - 8.26117
     - 11.4627
     - 1.00419

The inherited fit differs from the sampled JANAF H/S values by less than 3%,
so fitting error does not explain the source disagreement. Energy alignment
still leaves a substantial thermal difference. The archived JSON separates
H298, enthalpy increments, S298 and entropy increments for the molecule and
its atomic references. Printed formation-G values independently close the
H/S reconstruction within the propagation of their printed rounding. Neither closer
NASA agreement nor a newer publication alone establishes physical accuracy.

The official HTML's unused 298.15 K free-energy-function cell differs from
S - (H-H298)/T by 77.695 J/mol/K. The review retains and flags
this inconsistency; it uses H and S, never that cell, to reconstruct K. Other
inspected positive-temperature cells agree within their printed rounding (at
most 0.00134 J/mol/K). This does not
identify the historical origin of the cell or establish that the printed
edition contains the same inconsistency. It is not a reason to edit curated.

Modern evidence and its limits
------------------------------

Matus et al. (2007), DOI `10.1063/1.2774973 <https://doi.org/10.1063/1.2774973>`_,
report a singlet C2v ground state, a low 1.6 kcal/mol rearrangement barrier and a
triplet state 10.8 kcal/mol above the singlet. Only the primary abstract was
inspected. With equal rovibrational functions, the electronic triplet/singlet
factor alone is 0.0801, 0.4902 and 0.7710 at 1500, 3000 and 4000 K. These are
illustrative weights, not populations, a complete partition sum or an adopted
correction. The low barrier also warrants checking the rigid-isomer counting.
The reported fragmentation energies are not a molecular atomization energy.

W4-11, DOI `10.1016/j.cplett.2011.05.007 <https://doi.org/10.1016/j.cplett.2011.05.007>`_,
includes S4. The author manuscript's methods distinguish the clamped-nuclei,
nonrelativistic, zero-point-exclusive benchmark from physical, relativistic,
zero-point-inclusive atomization energies. The needed S4 numerical supporting
table was not recovered. A benchmark electronic atomization number must not be
silently adopted as D0.

Jackson et al. (2016) and Fedyaeva et al. (2023) also identify low-energy chain
structures. Neither inspected article supplies a validated 1500--4000 K S4
state sum. An individual S4 function and its explicit state inventory are
needed; the sulfur-mixture chemical potential is not a one-species function.

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
in S4 is 8.77024189e-17 at 1500 K,
0.0001 bar and C/O = 0.58884366; its mole fraction there
is 4.95724359e-22. The sulfur share counts all 4 atoms. The target
thermodynamic column matches the named current factory, and the maximum
sulfur conservation residual is 2.936e-12.

These trace values describe this archive only. Numerical convergence does not
guarantee relative accuracy of an extraordinarily small component, nor bound
physical abundances. Cooler sulfur-rich atmospheres, photochemistry and
condensation are outside this archive. No new solve or abundance sensitivity
to a replacement was performed.

Follow-up calculations and references
-------------------------------------

* Recover the S4 physical TAE0 entry and its convention from W4-11 supporting information (10.1016/j.cplett.2011.05.007); compare against a consistently balanced legacy energy.
* Obtain Matus 2007 state energies, rovibrational constants and zero-point conventions (10.1063/1.2774973). Inspect singlet/triplet, symmetry, low barriers and possible double-counting before a partition calculation.
* Reconstruct a declared S4 thermal candidate using individual species data from Jackson 2016 or Fedyaeva 2023; compare H, S, Cp and ln K before any fixed-network substitution.

Reproduction and MCEG
---------------------

Run ``python mceg/reviews/s4_source_review.py`` and
``pytest tests/model_evidence/s4_source_review_test.py`` offline. The
fixture in ``tests/data/validation/s4_source_review/20261105`` pins
source facts, consumed implementation/data files and numerical outputs.
The named ``curated_cos`` factory prevents a future default switch from
silently changing this review.

``mceg/contracts/s4_source_review.yaml`` marks scoped source attribution,
modern-source screening and numerical replay supported. Complete physical
accuracy and adoption readiness remain unknown. The latter can be assessed
for a limited candidate without requiring complete physical validation of all
legacy literature. No uncertainty band is inferred from source spread.
