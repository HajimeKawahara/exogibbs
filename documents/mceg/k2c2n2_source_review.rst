K2C2N2 neutral dimer identity and source review
===============================================

Resolve the identity as neutral (KCN)2 gas and retain its current function.
The six-atom dimer is not the three-atom monomer, a four-atom hypervalent
K2CN molecule, a cyanogen complex without metals, or condensed cyanide.
Source inspection identifies concrete thermal/energy follow-ups rather than
an immediately justified coefficient replacement. No new equilibrium solve
or curated change is made.

Printed identity and model
--------------------------

Official JANAF C-130 and printed page 678 explicitly identify
``C2K2N2`` as ideal gas. The March 1966 evaluation assumes a
planar D2h structure, sigma=4, ground weight 1, and 12 vibrational degrees.
The source uses estimated geometry and frequencies; a crystal-based analogy
is not an experimental gas dimer structure. Hf298=-8.368
+/-42 kJ/mol is kept distinct from dimer dissociation and
from total atomization into six neutral atoms.

JANAF estimates the dimer-to-two-monomers H298 as 40 kcal/mol by analogy with
the sodium dimer. NASA's record has the same old JANAF ancestry. The balanced
physical atomization difference is only -12.740 J/mol; native agreement does
not supply an independent modern validation.

A printed-list inconsistency is exposed by reconstruction: the page lists
four 2158 cm-1 modes, whereas its explanation assigns the last six modes from
two historical linear KCN monomers. The literal printed list misses the
1500 K table by -12.206 kJ/mol in H-H298, -36.322 J/mol/K in S and
-4.797 J/mol/K in Cp. A diagnostic interpretation with four 207 cm-1 bends
and two 2158 cm-1 stretches reproduces the table within 5 J/mol in H and
0.005 J/mol/K in S across the reviewed nodes. Both lists are preserved.
This supports a typographical-error hypothesis, not a newly measured model;
the numerical JANAF table and runtime fit are unchanged.

.. list-table:: Formation K at 1 bar
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - D0-aligned JANAF / NASA
     - 1977 abstract energy-only factor
   * - 1500
     - 1.01229190
     - 1.00112554
     - 0.04883455
   * - 2000
     - 1.00453929
     - 1.00093371
     - 0.10388322
   * - 3000
     - 0.99995104
     - 1.00153454
     - 0.22098541
   * - 4000
     - 1.00379263
     - 1.00219439
     - 0.32230920

The packet includes source molecular and atom-balanced formation G/H/S/Cp,
physical total D0, dimer-to-monomers D0/H298/H1000, analytic retained-fit
calorics and source closure. SI R is used for JANAF; any NASA comparison
retains its native R=8.314510. Each monomer reference is copied into this
packet, so the review is independent of the separate monomer-review PR.

Primary energy evidence and actionable limit
--------------------------------------------

The publisher record and indexed primary first-page transcription of
`Simmons et al. (1977) <https://doi.org/10.1021/j100523a006>`_ were inspected.
The abstract gives dimer Hf298=7.0 +/-3 kcal/mol and reports a large
second/third-law disagreement for the dimer. The full body was not obtained;
a public library full-issue download timed out. Its Hf298 corresponds to
29.288 +/-12.552 kJ/mol, or +37.656 kJ/mol relative to the selected JANAF value.
The last table column is exp(-37656/(RT)), keeping current thermals fixed.
It is an explicitly bounded abstract-based sensitivity, not an adopted
formation-energy recommendation or an abundance response. The JANAF energy's
own quoted 42 kJ/mol uncertainty and unresolved third-law thermal model prevent
an automatic replacement based only on the central-value difference.

Obtain the 1977 body, resolve the printed frequency list against the original
thermal construction, and jointly review monomer/dimer energies. A finite
candidate may then separate a justified thermal change from the energy shift.
The old table's apparent print typo alone does not justify changing the
runtime function that already follows the numerical table.

Importance and reproducibility
------------------------------

Across 75 archived curated-tio2-v1 endpoints, the maximum K share is
9.22702594e-11% at 1500 K,
100 bar, C/O=1.5; maximum mole fraction is
9.32025642e-20. All endpoints converge with 537 species.
This hot H-rich trace abundance does not validate cold salt-rich vapor or
condensation. Missing full source bodies are focused follow-ups, not a demand
for every historical ancestor before merging the documented source review.

Run ``python mceg/reviews/k2c2n2_source_review.py`` and
``pytest tests/model_evidence/k2c2n2_source_review_test.py``. The packet is
``tests/data/validation/k2c2n2_source_review/20261106`` and contract is
``mceg/contracts/k2c2n2_source_review.yaml``. Inputs use historical hash
resolution; outputs are directly pinned. Complete physical accuracy remains
unknown, distinct from the supported identity/source/retention assessment.
