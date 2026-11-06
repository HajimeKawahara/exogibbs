S5 energy and thermal-source investigation
==========================================

This 2026-11-03 follow-up to #308 and #309 retains the existing S5 fit
provisionally. The large JANAF/NASA difference is reproduced and separated
into energy and thermal terms, but the available evidence does not select a
replacement over 1500-4000 K. This completes an initial source investigation;
curated coefficients and runtime behavior are unchanged. Physical accuracy
remains unknown in the MCEG record.

Identity and inspected sources
------------------------------

The reaction is 5 S(g) -> S5(g), for neutral gas at standard pressure 1 bar
(CAS 12597-10-3). The formula identifies five sulfur atoms; it does not establish
that two compilations include the same geometry, electronic states or isomers.

* Current curated S5 inherits the FastChem4 five-coefficient fit labelled
  Chase/JANAF 1998. It is still an inherited, incompletely reviewed entry.
* `Official JANAF table S-018 <https://janaf.nist.gov/tables/S-018.html>`_
  carries the September 1977 evaluation converted to 1 bar. Eight selected
  HTML rows were transcribed; the four frozen comparison nodes and the
  298.15 K formation enthalpy match. Direct text/PDF retrieval failed, so this
  is an HTML inspection, not a newly downloaded raw table or an inspection of
  the printed evaluation notes. The existing pinned S(g) table is reused.
* `NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_,
  Appendix D printed page 184 (PDF page 192), was downloaded and visually
  checked. All 18 S5 coefficients match the frozen NASA record. Its source is
  Gurvich (1989), volume 1, part 1 page 281 and part 2 page 169. Those original
  pages were not obtained. The NASA record covers 200-6000 K; this is a fit
  interval, not a demonstrated accuracy interval.

The NASA header gives Hf(298.15)=132.993354 kJ/mol and
H(298.15)-H(0)=19.053354 kJ/mol. The header's thermal increment is distinct
from a formation enthalpy at 0 K. JANAF supplies no S5 nodes below 298.15 K,
so this review uses a 298 K decomposition without inventing a 0 K anchor.

What causes the difference
--------------------------

At 298.15 K, JANAF has Hf=109.370 kJ/mol and S=308.638 J/mol/K;
NASA has Hf=132.993354 kJ/mol and S=354.086233 J/mol/K. The latter
header check uses NASA's historical R=8.31451 J/mol/K. The comparison itself
uses the baseline R=8.31446261815324 J/mol/K throughout, including both
sources' native atomic backgrounds.

.. list-table:: Dimensionless atom-formation equilibrium constants
   :header-rows: 1

   * - T, K
     - JANAF / NASA
     - H298-aligned JANAF / NASA
     - Current curated / NASA
   * - 500
     - 1.02377
     - 0.00437356
     - Outside curated domain
   * - 1000
     - 0.0688568
     - 0.00450052
     - Outside curated domain
   * - 1500
     - 0.0260947
     - 0.00423411
     - 0.0265097
   * - 3000
     - 0.00968812
     - 0.00390252
     - 0.00954754
   * - 4000
     - 0.00792233
     - 0.00400573
     - Not sampled in frozen baseline

The current fit is within +1.59% and -1.45% of JANAF at 1500 and 3000 K,
while its constants are approximately 38 and 105 times smaller than NASA's.
Thus the inherited fit residual does not explain the source discrepancy at
these nodes. This is not a full continuous fit-error audit.

For JANAF minus NASA, the balanced reaction difference in G is
+45.4721 kJ/mol at 1500 K. Its components are -22.6805 kJ/mol from the
reaction H298 anchor, -2.33335 from sensible enthalpy, +68.1468 from the
298 K entropy difference and +2.33917 from subsequent entropy increments.
At 3000 K the total is +115.6589 kJ/mol. The five atomic references contribute
only +0.9182 and +0.8911 kJ/mol, respectively; the difference is principally
molecular. Nevertheless, using molecular formation enthalpies alone would
miss those atomic terms.

Aligning the balanced H298 anchor leaves JANAF constants roughly 236-256 times
smaller at these two temperatures. This is a diagnostic, not an adopted
correction. The large entropy difference survives; a single energy correction
does not reconcile the thermal models. The apparent agreement near 500 K is
largely cancellation between energy and entropy differences.

A separate JANAF column inconsistency
-------------------------------------

The official HTML reports -[G-H(298.15)]/T=477.413 J/mol/K at 298.15 K,
but the same row gives S=308.638 and H-H(298.15)=0. The defining identity
therefore gives 308.638, a discrepancy of 168.775 J/mol/K. The other seven
inspected rows close within 0.000667 J/mol/K. The archived transcription
preserves the reported value and records this as a column inconsistency;
its historical origin and whether it occurs in the printed edition remain
unverified.

This free-energy-function column is not used by the frozen H/S reconstruction.
At the six source-comparison temperatures, balanced JANAF formation-G and
H/S constructions agree to less than 0.000482 in ln K (consistent with printed
rounding). Consequently, the anomalous 298 K cell does not explain the
JANAF/NASA disagreement or justify modifying curated. No source entry is
silently repaired.

Literature and scientific limits
--------------------------------

`Rau, Kutty and Guedes de Carvalho (1973)
<https://doi.org/10.1016/S0021-9614(73)80045-X>`_ reports sulfur-vapor density
measurements at 823-1273 K and a mixture reduction for S2-S8 with real-gas
corrections. Only its publisher abstract and metadata were inspected. The
full text is needed to examine the S5 assumptions and correlated uncertainties;
the experimental temperature range alone does not establish high-temperature
S5 accuracy.

The full author manuscript of `Jackson, Tiana and Walsh (2016)
<https://doi.org/10.1039/C5SC03088A>`_ was inspected. Its introduction describes
the JANAF S3-S7 assignments as derived from empirical interpolation using Rau's
work; this is the authors' account of the genealogy, pending the original
evaluation notes. Section 3.1.4 identifies a five-membered Cs ring as the main
S5 candidate. Their thermodynamics uses DFT energies, rigid-rotor/harmonic
functions, calibrated frequencies and an experimental S8 sublimation reference.
This supplies a candidate route, with model-dependent energies and a different
state inventory, rather than a completed accuracy assessment through 4000 K.

The paper's **mixture** chemical potential includes several sulfur species and
isomers. Substituting it for a single S5 standard chemical potential would
double-count the mixture calculation. Any future candidate must extract the
individual S5 functions, define isomer counting and align molecular and atomic
reference energies before evaluating the existing network. The linked
`code archive <https://doi.org/10.5281/zenodo.28536>`_ and
`DFT data <https://doi.org/10.6084/m9.figshare.1513736>`_ were identified but
not independently reproduced in this review.

Application priority and follow-ups
-----------------------------------

Reusing the archived 75-case, 537-species ion-inclusive current-curated grid
(1500-4000 K, 1e-6-100 bar, reference C/O, 1 and 1.5) gives a maximum S5
sulfur fraction of 1.97012e-24, with mole fraction 8.90862e-30 at 1500 K,
1e-4 bar and reference C/O=0.588844. All original cases report convergence.
These are existing model results, not new solves, observed abundances, a
replacement sensitivity study or a physical error bound. They lower the
priority of a new S5 substitution run for this sampled regime. They do not
exclude importance in cooler or sulfur-rich mixtures outside that grid.

MCEG retains two actionable follow-ups:

1. Obtain the Gurvich S5 pages above (no verified DOI) and Rau's full paper,
   DOI 10.1016/S0021-9614(73)80045-X. Review geometry, symmetry, electronic
   weights, vibrational frequencies, energy reductions, correlations and the
   original JANAF notes. Separate the entropy offset from energy uncertainty.
2. If a replacement is scientifically preferred, reconstruct individual S5
   standard functions with explicit reference/state conventions. Test it in a
   fixed network relevant to its domain, preserving conservation, residuals,
   absolute abundances and sulfur allocation. Do not extrapolate the existing
   trace footprint into a universal guarantee.

These open scientific tasks do not prevent merging the completed source review.
The canonical record is ``mceg/contracts/s5_source_review.yaml``; the compact
packet contains factual excerpts, acquisition limits, hashes and an offline
receipt. Full downloaded publications are not committed. Replay with::

   python mceg/reviews/s5_source_review.py
   pytest -q tests/model_evidence/s5_source_review_test.py
