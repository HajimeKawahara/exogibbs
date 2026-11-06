S8: source energy, thermal functions and next calculation
=========================================================

Prioritize a direct-W4L atomization-energy candidate next. Keep curated unchanged in this source-review PR until that candidate and substitution are explicitly evaluated.

This 2026-11-05 review follows issues #308 and #309. It changes no runtime
function, packaged thermochemical record or default preset. It compares
8 S(g) -> S8(g), audits the official source transcription, and reads the
latest archived ``curated-cos-v1`` grid. It does not run a replacement solve.

Source identity and conventions
-------------------------------

The current S8 record remains the inherited FastChem/JANAF fit. Official
`JANAF S-021 <https://janaf.nist.gov/tables/S-021.html>`_
labels its evaluation September 1977 and conversion from 1 atm to 1 bar.
Eight positive-temperature rows were inspected in the official HTML and
transcribed; the four original screening nodes and Hf298 agree exactly.
The original printed evaluation notes were not read. NASA Glenn report
`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_, printed
page 185, attributes its S8 record to Gurvich 1989, volume 1,
part 1 page 284 and part 2 page 172. The printed record
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
     - 0.491905
     - 0.51809
     - 1.02598
   * - 2000
     - 0.374824
     - 0.38969
     - 0.991183
   * - 3000
     - 0.251357
     - 0.25796
     - 0.976323
   * - 4000
     - 0.196858
     - 0.200724
     - 1.00798

The inherited formation K differs from that reconstructed from sampled JANAF
H/S by less than 3%, so fitting error does not explain the source disagreement.
Energy alignment still leaves a substantial thermal difference. The archived JSON separates
H298, enthalpy increments, S298 and entropy increments for the molecule and
its atomic references. Printed formation-G values independently close the
H/S reconstruction within the propagation of their printed rounding. Neither closer
NASA agreement nor a newer publication alone establishes physical accuracy.

The selected positive-temperature free-energy-function cells close to
0.001 J/mol/K. The official zero-temperature row is retained for its Hf0,
while the undefined free-energy-function expression at T=0 is not evaluated.

Modern evidence and its limits
------------------------------

Karton (2021), DOI `10.1016/j.chphi.2021.100047 <https://doi.org/10.1016/j.chphi.2021.100047>`_,
provides an accessible full article and a high-level W4L evaluation of the gas
S8 crown ring. Table 2 directly gives TAE0 = 505.67 kcal/mol, TAEe = 513.45
kcal/mol and ZPVE = 7.77 kcal/mol. The direct TAE0 includes zero-point,
relativistic, atomic spin-orbit and diagonal Born--Oppenheimer contributions.
Use the direct TAE0, not subtraction of rounded TAEe and ZPVE, or a molecular
formation enthalpy without its atomic references.

The frozen JANAF S and S8 rows give D0 = 8(274.735) - 104.388 = 2093.492
kJ/mol. The W4L direct value is 2115.72328 kJ/mol, a **+22.23128 kJ/mol** change
in the balanced atomization energy. By contrast, comparing only rounded S8
formation enthalpies gives -2.13104 kJ/mol: the paper's rounded atomic reference
adds 19.97472 kJ/mol across eight S atoms. The rounded enthalpy balance misses
the direct TAE0 by -0.12552 kJ/mol; this is recorded as rounding, not fitted
away or promoted to an uncertainty estimate.

Holding thermal functions fixed, the energy-only K ratio relative to JANAF
would be 5.9449, 2.4382 and 1.9512 at 1500, 3000 and 4000 K. This is an analytic
sensitivity, not a new fitted model or equilibrium substitution. The paper
supports prioritizing a direct-TAE0 energy candidate without waiting for
another article. It does not supply a complete 4000 K isomer/electronic state
sum or a quantified atmospheric error bound.

Jackson et al. (2016) use the experimental 100.416 kJ/mol S8 sublimation/formation
anchor and a D4d ring; this does not independently validate that legacy energy.
They did not perform a global S8 search. Fedyaeva et al. (2023) provide a wider
structure search and identify the crown ground state, with finite-temperature
comparisons only through 1500 K. Their results can inform a separate thermal
review; they are not added to the W4L ground-state energy without a consistent
partition and reference convention.

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
in S8 is 3.52398745e-43 at 3000 K,
100 bar and C/O = 0.58884366; its mole fraction there
is 9.89532668e-49. The sulfur share counts all 8 atoms. The target
thermodynamic column matches the named current factory, and the maximum
sulfur conservation residual is 2.936e-12.

These trace values describe this archive only. Numerical convergence does not
guarantee relative accuracy of an extraordinarily small component, nor bound
physical abundances. Cooler sulfur-rich atmospheres, photochemistry and
condensation are outside this archive. No new solve or abundance sensitivity
to a replacement was performed.

Follow-up calculations and references
-------------------------------------

* Build an energy-only S8 candidate from the already inspected direct W4L TAE0 (10.1016/j.chphi.2021.100047), with a single balanced delta(D0)/(RT) correction and retained thermal terms. No additional full text is required for this scoped calculation.
* Compare the candidate with the actual inherited fit, document its 298 K energy anchor rather than assuming an exact JANAF D0, and run the fixed-network substitution with conservation, mass-action and sulfur allocation checks.
* Review S8 excited/isomer thermal contributions separately using Jackson 2016 (10.1039/C5SC03088A) and Fedyaeva 2023 (10.1039/D2CP05498A); include cooler and sulfur-rich regimes if they become intended applications.

Reproduction and MCEG
---------------------

Run ``python mceg/reviews/s8_source_review.py`` and
``pytest tests/model_evidence/s8_source_review_test.py`` offline. The
fixture in ``tests/data/validation/s8_source_review/20261105`` pins
source facts, consumed implementation/data files and numerical outputs.
The named ``curated_cos`` factory prevents a future default switch from
silently changing this review.

``mceg/contracts/s8_source_review.yaml`` marks scoped source attribution,
modern-source screening and numerical replay supported. Complete physical
accuracy and adoption readiness remain unknown. The latter can be assessed
for a limited candidate without requiring complete physical validation of all
legacy literature. No uncertainty band is inferred from source spread.
