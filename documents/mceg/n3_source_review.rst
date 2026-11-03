N3 energy and thermal-source investigation
==========================================

This 2026-11-03 investigation follows #308 and #309 for neutral gaseous N3.
The current JANAF-derived curated fit is retained provisionally. Modern
energy evidence favors **weaker binding than both the inherited JANAF and
the historical NASA values**. A separate thermal-model assessment and a
controlled substitution remain before adoption. Merging this source review
does not establish that the current coefficients need no future correction.

Identity and historical construction
-------------------------------------

The reaction compared is 3 N(g) -> N3(g), at standard pressure 1 bar.
The inspected `JANAF printed page 1632
<https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Nitrogen.pdf>`_ specifies a
linear D-infinity-h neutral radical with X2Pi ground state, rotational
symmetry number 2, and electronic levels at 0, 71.3 and 36811 cm^-1 with
weights 2 each. It is neither the closed-shell azide anion nor cyclic N3.
The `JANAF index <https://janaf.nist.gov/tables/N-index.html>`_ associates
14343-69-2 with this neutral table; the modern ATcT anion uses that number,
while its neutral radical uses 12596-60-0. The printed state, charge and
formula resolve this metadata conflict; CAS matching alone would fail.

Although republished in 1998, the N3 evaluation is **December 1970**.
The `official N-034 table <https://janaf.nist.gov/tables/N-034.html>`_
gives Hf(298.15)=414.216 kJ/mol and S(298.15)=226.469 J/mol/K. The printed
uncertainties are approximately 20.9 kJ/mol and 2.1 J/mol/K. The energy
cycle uses cyanogen-azide photolysis to CN + N3, with the precursor's solid
formation enthalpy and an estimated 14.6 kcal/mol sublimation enthalpy.
This inherited estimate is not a modern direct N3 atomization measurement.
The original experiments cited by JANAF were not inspected.

JANAF estimated its vibrational frequencies from related isoelectronic
molecules: 1800, 500 (doubly degenerate) and 1200 cm^-1. The later
`NIST/Jacox compilation
<https://webbook.nist.gov/cgi/cbook.cgi?ID=C12596600&Mask=800>`_ lists gas-phase
fundamentals near 1320, 457 and 1644.68 cm^-1; the first two are explicitly
tentative/approximate. These are fundamentals, not interchangeable harmonic
frequencies. Matrix bands at other frequencies must also remain separate.
Updating three numbers alone would not establish an accurate hot partition
function, including spin-orbit, vibration-rotation and excited-state effects.

`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_ cites
Gurvich (1989), volume 1 part 1 page 325 and part 2 page 202, for neutral N3.
The record gives Hf(298.15)=436.000 kJ/mol and H(298.15)-H(0)=9.570906 kJ/mol,
with intervals 200-1000 and 1000-6000 K. All N3 and N coefficients in the
inspected report excerpt reproduce the pinned baseline. Gurvich's original
state inventory and energy evaluation were not obtained.

Separating energy from thermal functions
----------------------------------------

Native JANAF and NASA atomic references are subtracted before comparison.
The balanced JANAF-minus-NASA reaction H298 difference is -21.7986 kJ/mol
using the common R of the frozen comparison. Atomic contributions to the
full Gibbs difference are only -0.019 to -0.030 kJ/mol over the six nodes.
Thus changing the nitrogen atom convention does not explain the discrepancy.

.. list-table:: Formation equilibrium-constant ratios
   :header-rows: 1

   * - T, K
     - Raw JANAF / NASA
     - H298-aligned JANAF / NASA
     - Current curated / NASA
   * - 1500
     - 12.3303
     - 2.14730
     - 12.4369
   * - 2000
     - 8.53504
     - 2.30089
     - 8.51395
   * - 3000
     - 5.97532
     - 2.49357
     - 5.94208
   * - 4000
     - 5.01978
     - 2.60633
     - 5.03630

An energy shift removes much of the discrepancy at 1500 K, but leaves a
factor 2.15-2.61 thermal difference across 1500-4000 K. The source difference
is therefore not just an enthalpy constant. At 3000 K, the four contributions
(H298, enthalpy increment, -T*S298, entropy increment) are -21.7986,
+4.2629, -10.2077 and -16.8464 kJ/mol. The inherited fit differs from the
JANAF H/S reconstruction by at most 0.008607 in ln K (about 0.86%) over
26 nodes from 1500 to 4000 K. That is numerical consistency, not a recovery
of the original fitting procedure. H298 alignment is diagnostic only.

The modern energy direction
----------------------------

The pinned `ATcT version 1.172 NNN evaluation
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=304>`_
gives Hf(0)=452.38 and Hf(298.15)=449.79 +/- 0.59 kJ/mol; the quoted
uncertainty is an estimated 95% confidence limit. Using the
`same-version N atom
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=24>`_
with Hf(0)=470.583 kJ/mol gives total atomization
D0 = 3*470.583 - 452.38 = **959.369 kJ/mol**.
The dataset DOI is `10.17038/CSE/2440254
<https://doi.org/10.17038/CSE/2440254>`_. Later ATcT versions are not mixed
into this comparison.

For JANAF and NASA, H(0)=Hf(298.15)-[H(298.15)-H(0)] is first formed in
each source's 298 K gauge, then the source's atoms are balanced. The resulting
D0 values are **995.495 kJ/mol (JANAF)** and **973.018622 kJ/mol (NASA headers)**.
The modern value is lower by 36.126 and 13.649622 kJ/mol respectively.
Molecular Hf differences alone are not these reaction corrections. NASA's
printed headers use its R=8.31451 convention, kept separate from the
common-R polynomial decomposition above. No combined uncertainty is inferred
without the missing covariance information.

The ATcT evaluation, rather than proximity to NASA, supports prioritizing
a weaker modern energy anchor. Its displayed top 20 inputs cover only 68.9%
of NNN provenance. The largest contribution is the H-N3 cleavage measurement
of `Cook et al. (1999), 10.1039/A805936E
<https://doi.org/10.1039/A805936E>`_; that bond energy is not N3 total D0.
The publisher abstract and ATcT provenance were inspected, not the full paper.
`Yuan et al. (2008), 10.1021/jp800062t
<https://doi.org/10.1021/jp800062t>`_ supplies another ATcT input.
`Dixon et al. (2004), 10.1021/ja0303182
<https://doi.org/10.1021/ja0303182>`_ supplies theoretical constraints; its
author abstract reports Hf(0)=109.2 +/- 1 kcal/mol for neutral 2Pi N3.
Those original full numerical tables remain uninspected. Their ATcT
participation prevents treating them as independent validation of ATcT.

The author-hosted full text of `Jamieson and Kaiser (2007)
<https://doi.org/10.1016/j.cplett.2007.04.042>`_ was inspected. Their 10 K
irradiated nitrogen-ice experiment identifies linear N3 and finds no evidence
of cyclic formation or access. Its discussion quotes an earlier potential
surface with cyclic X2B1 about 127 kJ/mol above linear. This supports keeping
the isomers distinct; it supplies neither hot gas equilibrium populations nor
a complete cyclic partition function. Driven photochemistry, matrix trapping
and radical lifetimes cannot be used as equilibrium abundance validation.

Practical footprint and MCEG follow-ups
---------------------------------------

The saved 75-case, 537-species ion-inclusive ``curated-c2h-v1`` grid has a
maximum N3 nitrogen-budget fraction of **6.69843e-10** (6.69843e-8%), at
4000 K, 100 bar and reference C/O=0.5888436553555908. The mole fraction is
2.41431e-14. This reuses converged endpoints, verifies nitrogen conservation
and checks all archived N3 potentials against the current fit. It is a
baseline footprint, not a new substitution, observation or universal error
bound. The sampled grid covers 1500-4000 K, 1e-6-100 bar and three C/O values.

``mceg/contracts/n3_source_review.yaml`` records supported source decomposition,
the conditional modern-energy direction and numerical replay. Complete
thermal-model and atmospheric accuracy remain unknown. Two follow-ups are:

* Construct a named candidate with a same-version, atom-balanced energy anchor
  and a documented linear-radical thermal model. Inspect the relevant
  spectroscopy as needed: `Beaman et al. (1987), 10.1021/j100308a006
  <https://doi.org/10.1021/j100308a006>`_ and
  `Brazier et al. (1988), 10.1063/1.455122
  <https://doi.org/10.1063/1.455122>`_. Gurvich's cited pages help resolve the
  legacy construction, but obtaining every historical original is not a
  prerequisite to evaluating a modern candidate.
* Assess that explicit candidate by finite substitution on a fixed network,
  checking elemental conservation, mass-action residuals, nitrogen allocation
  and absolute abundances. Keep thermal/state uncertainties and gas-equilibrium
  applicability separate from the very small baseline footprint.

The compact packet is ``tests/data/validation/n3_source_review/20261103``.
Full research papers are not redistributed. Reproduction is offline::

   PYTHONPATH=src python mceg/reviews/n3_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/n3_source_review_test.py

No production coefficients, preset identifier, temperature bounds or archived
equilibrium endpoints are changed.
