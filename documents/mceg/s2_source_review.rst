S2: retain the balanced energy and review thermal completeness
==============================================================

The current S2 construction is retained: NASA thermal functions with the
same-version ATcT 1.172 atomization energy. The 46.18% difference from unmodified
NASA at 1500 K arises from an intentional energy correction. The newer
ATcT 1.228 tables give essentially the same balanced energy. The Gomez UV line
list is useful spectroscopic evidence but does not supply a complete 1500--4000 K
thermal replacement. No production coefficient or default preset changes here.

This is a source review for Issue #308, not a finite substitution or an
observational validation. Its evidence and follow-ups can be merged without
waiting for a new production model or every historical original paper.

Energy balance and the dissociation limit
-----------------------------------------

All energies refer to ``S2(g) -> 2S(g)``. Forming S2 uses the reverse reaction
and dimensionless equilibrium constants at one bar.

.. list-table:: Balanced ground-relative atomization energies
   :header-rows: 1

   * - Source
     - D0 (kJ/mol)
     - Interpretation
   * - NASA / Gurvich
     - 421.557260
     - Original NASA reaction anchor
   * - JANAF
     - 421.170000
     - Native JANAF atom and molecule values
   * - ATcT 1.172, current
     - 426.290000
     - ``2 * 276.830 - 127.370``
   * - ATcT 1.228
     - 426.280000
     - ``2 * 276.85 - 127.42``; rounded displayed values

The latest sulfur atom and S2 values move together. Their reported correlation
is 99.2%, so independently propagating their quoted uncertainties is unsuitable.
The 10 J/mol difference between displayed balanced energies is comparable to
rounding. Holding thermal functions fixed would multiply the current formation
constant by 0.999198 at 1500 K and 0.999699 at 4000 K. This is not evidence for
an urgent production update. Official records are
`ATcT S <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=138>`_
and `ATcT S2 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=273>`_.

Frederix et al. report ``35636.9 +/- 2.5 cm^-1`` relative to the lowest
rovibrational level, with two ground ``S(3P2)`` products. The current balanced
energy corresponds to 35635.061 cm^-1. Their product-state imaging reassigned
the older excited-atom dissociation asymptote; thus reverting to the smaller
NASA energy because the source ratio is large is unwarranted. The publisher
abstract was inspected, not the full article:
`10.1021/jp905104u <https://doi.org/10.1021/jp905104u>`_.
The 2026 sulfur evaluation
`10.1021/acs.jpca.6c05060 <https://doi.org/10.1021/acs.jpca.6c05060>`_
was checked through official evaluated tables and publisher abstract/metadata;
the full paper was unavailable. Its large revision relative to CODATA is
already largely reflected in the current ATcT 1.172 anchor.

NASA dimensionless functions retain native ``R = 8.31451 J/mol/K``. The exact
current-to-NASA logarithmic shift is ``569.506718981 / T``, rather than simply
the energy difference divided by modern R and T. Molecular caloric comparisons
below consistently convert NASA dimensionless functions to modern R.

What the UV state list can establish
------------------------------------

The full author manuscript of Gomez et al.
(`10.1093/mnras/stae246 <https://doi.org/10.1093/mnras/stae246>`_)
and the official 2026-05-04 ExoMol distribution were inspected. All 17,766
distributed states were summed. Nuclear spin product is one; the allowed
N parities and triplet components already implement permutation statistics.
No additional factor of three or division by two is appropriate. The occupied
ground level has ``J=0, N=1`` and energy zero. The paper's PGOPHER zero shift
has already been applied; adding it again would double count it.

.. list-table:: Common-ground, common-natural-mass thermal diagnostics
   :header-rows: 1

   * - T (K)
     - Gomez Q / NASA Q
     - Gomez Cp (J/mol/K)
     - NASA Cp (J/mol/K)
   * - 1500
     - 0.983437
     - 36.550
     - 38.985
   * - 3000
     - 0.848610
     - 30.603
     - 41.690
   * - 4000
     - 0.717657
     - 27.591
     - 42.603

The molecule has low singlet ``a1Delta_g`` and ``b1Sigma+_g`` manifolds absent
from this X/B/B-double-prime UV model. Its X-state coverage ends at v=10 and
J=150; the authors explicitly caution above 1500 K. At 4000 K the last
v=10 shell carries 2.012% of the finite Q and J>=145 carries 2.125%.
The two shells overlap and cannot be added into a missing-population estimate.
The declining Cp and boundary populations diagnose limited thermal coverage;
they do not show that NASA is exact or justify applying a 28.2% correction.
The tiny populated B/B-double-prime contribution does not constrain omitted
low singlets.

The ExoMol and HITRAN q152 tables are identical at every printed temperature.
The independent moment replay agrees with those tables within 4.5 ppm at the
selected nodes. These are one source lineage, not two independent validations;
tabulation to 5000 K does not establish completeness there. Some distributed
excited-state g/u and Sigma-sign labels disagree with the paper. The packet
preserves the raw labels, energies and weights and records the discrepancy;
this label issue is separate from the intended UV coverage.

At 4000 K, JANAF with the same energy and atomic thermal reference gives
``Q/Q_NASA = 0.986131``; the frozen Barklem--Collet 2016 table gives 0.866307.
This unresolved thermal spread is retained. The 32S2 internal spectrum with
natural molecular translation is an elemental proxy; a separate isotope-mass
ratio is reported, not an isotope-resolved reaction. Predissociative states
were preserved for source replay, not accepted as an equilibrium bound-state
prescription. No correction for missing singlets is invented from their Te
values, which are not lowest-level T0 values.

Atmospheric scope and next work
-------------------------------

In the existing 75-case ``curated-h2o-v1`` archive, S2 contains at most 1.882%
of sulfur and has maximum mole fraction ``2.112e-7`` at 1500 K, ``1e-6`` bar,
and baseline C/O. All 75 endpoints converge; the complete sulfur budget closes
within ``3e-12``. These are current abundances, not responses to a replacement.
Gas equilibrium alone does not address UV-driven sulfur chemistry.

The next calculation must converge X and low excited electronic manifolds,
J/v tails and first/second energy moments through 4000 K, on a consistent
ground zero. It must distinguish bound levels, resonances and continuum.
Only then should a derivative-consistent candidate and finite sulfur-budget
substitution be assessed. Xing et al.
`10.1080/00268976.2012.741721 <https://doi.org/10.1080/00268976.2012.741721>`_
is a useful follow-up for electronic potentials; it was cited by the inspected
Gomez paper but not itself acquired. Full Frederix/Thorpe texts and original
Gurvich pages are provenance follow-ups, not gates on this scoped review.

Reproduction and evidence
-------------------------

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/s2_source_review.py
   pytest tests/model_evidence/s2_source_review_test.py

The compact states, original-file hashes, source inspection status, comparison
and receipt are under ``tests/data/validation/s2_source_review/20261105``.
The optional ``--extract`` path verifies official raw hashes before rebuilding
the numeric packet. CI uses only the frozen packet. MCEG separates supported
source reconstruction, numerical diagnostics and provisional retention from
unknown full physical accuracy in ``mceg/contracts/s2_source_review.yaml``.
