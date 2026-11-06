NH3: inversion, energy and high-temperature partition sources
=============================================================

Keep the current NASA thermal functions plus ATcT 1.172 energy provisionally.
This review does **not** establish that a future thermal update is unnecessary.
The high-temperature differences deserve further investigation, but the available
sources do not yet support a replacement across the complete 1500--4000 K domain.
No production coefficients, default preset or equilibrium results are changed.

NH3 is an important nitrogen-bearing atmospheric molecule. Its strongest share
in the existing 75-case ``curated-h2o-v1`` grid is 26.887% of nitrogen at 1500 K,
100 bar and C/O=0.58884, with mole fraction 3.1233e-5. This is a saved gas-only
equilibrium footprint, not a new substitution or observational validation.
Hydrogen/nitrogen budgets close below 7.1e-13 relative and all saved cases converged.

Current construction and energy separation
------------------------------------------

The unchanged ``H3N1`` column uses CEA NASA9 thermal functions, whose source
header cites Gurvich (1989, part 1 p354, part 2 p219) and Haar (1968), with an
ATcT 1.172 atomization anchor. NASA supplies 200--1000 and 1000--6000 K intervals;
the production domain remains 1500--4000 K. The historical original books were
not newly inspected, which does not prevent merging this source-only record.

For the balanced reaction N(g) + 3 H(g) -> NH3(g), D0 is 1157.249 kJ/mol from
same-version ATcT values, 1157.869893 kJ/mol from NASA and 1157.835 kJ/mol from
JANAF. Individual NASA Hf(298)-[H(298)-H(0)] values are assigned energies in a
reference gauge; they are not compared directly to ATcT formation enthalpies.
Native NASA R=8.314510 J/mol/K is retained before converting to dimensionless
functions. The resulting adopted-minus-NASA D0/k shift is -73.882665 K, making
curated/raw-NASA formation K equal to 0.951938 at 1500 K and 0.981699 at 4000 K.
This energy shift is separate from every thermal-only comparison below.

Partition functions, inversion and isotope conventions
------------------------------------------------------

The `CoYuTe paper <https://doi.org/10.1093/mnras/stz2778>`_, author manuscript and
official `20200730 distribution <https://zenodo.org/records/5716876>`_ were
inspected. Its D3h(M) weights are 0,12,6,0,12,6 for A1',A2',E',A1'',A2'',E''.
The chemical convention divides the reported total Q by 24; another symmetry,
inversion or electronic factor must not be applied. This assumes equilibrium
between nuclear-spin species, rather than a prescribed frozen ortho/para ratio.

The formal vibrational J=0 zero has forbidden symmetry. Table 1 gives the
allowed J=0 inversion partner at 0.793374 cm-1 with weight 12. Its single-state
sum at 1 K is 3.832121, matching the distributed Q=3.8321 to printed precision.
Shifting Q to this allowed-state zero changes the energy reference by
9.490861 J/mol; the D0 convention must shift consistently. The packet reports
this reference sensitivity without asserting that ATcT needs a correction.

NASA uses natural-element mass 17.03052 u while CoYuTe uses 14NH3 mass
17.026549 u. The separate translation factor is approximately 0.999650.
Common-mass comparisons below use the dominant isotopologue internal spectrum
as an elemental proxy. Neither that proxy nor the isotope-mass sensitivity is
a fully isotope-balanced equilibrium reaction.

Thermal comparisons and their limits
------------------------------------

All ratios hold D0 and atomic thermal functions fixed, use one bar, and remove
ideal-gas translation from NASA before comparing internal Q. They are proxies
for a molecular thermal replacement, not computed abundance responses.

.. list-table:: Thermal-only formation-K proxy relative to current NASA thermal functions
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - T (K)
     - JANAF
     - TIPS 2021 / 24
     - TIPS 2025 / 24
     - CoYuTe / 24
   * - 1500
     - 1.009746
     - 0.992108
     - 0.994369
     - 0.994480
   * - 2000
     - 1.018851
     - 0.977818
     - 0.988577
     - 0.991332 (range caveat)
   * - 3000
     - 1.038240
     - 0.883740
     - 0.936692
     - unavailable
   * - 4000
     - 1.049250
     - 0.694573
     - 0.915291
     - unavailable

The `2025 TIPS evaluation <https://doi.org/10.1016/j.jqsrt.2025.109568>`_
uses inherited direct sums below 692 K for 14NH3, then analytical rotation
times harmonic-oscillator vibration through 5000 K. Thus the entire production
range uses the product approximation. Exact official HAPI arrays are frozen
from commit ``5353f4021d4b344eb9dd3484feeab9d483f30faa``. A newer publication
does not make this a more complete variational calculation. The 8.47% deficit
against NASA at 4000 K is a source discrepancy, not a measured error bar.

The `2021 TIPS paper <https://doi.org/10.1016/j.jqsrt.2021.107713>`_ describes
its NH3 direct sums as converged to 6000 K. Its archived 4000 K result is 30.54%
below NASA, and the 2025 value is substantially larger. These are different
published constructions; the 2021 label does not independently establish
high-temperature energy-moment convergence. Shared ExoMol ancestry also limits
independence between these evaluations.

CoYuTe's published opacity target is 1500 K. The definition file says 2000 K
and its PF table reaches 2000 K; these facts do not extend the paper's
completeness claim to 4000 K. Its lower-state opacity completeness and a sum
over all distributed states are different quantities. This review reads the
published PF rather than summing all 5,095,730 distributed states.

JANAF/NASA thermal differences are evaluated independently from exact tables:
the common-R Cp difference is +0.609 J/mol/K at 1500 K and -1.513 J/mol/K at
4000 K. H increments and entropy differences reproduce the thermal-K ratios.
They show a shape difference, but do not by themselves identify the better model.
No caloric derivatives are inferred from rounded spectroscopic PF tables.

Dedicated 2014 evaluation and unresolved printed fit
----------------------------------------------------

`Sousa-Silva et al. (2014) <https://doi.org/10.1016/j.jqsrt.2014.03.012>`_
extend the NH3 variational sum to about 7.5 million levels, J=45 and a source
cutoff of 41051 cm-1, using symmetry reconstruction at high J. This cutoff is
not the full atomization energy N+3H and must not replace the ATcT D0 above.
The paper reports Q convergence below approximately 3000 K and convergence of
the more demanding thermodynamic quantities to at least 2000 K. Approximate
higher-temperature usefulness is not equivalent to verified 4000 K convergence.

The visually checked Table 1 coefficients, evaluated with Eq. 18 in base ten,
give Q(296)=1779.076858. The text explicitly reports direct-sum Q(296)=1725.24:
the 3.1205% discrepancy exceeds the stated 0.4% fit accuracy and the conservative
0.2083% bound from rounding every printed coefficient by half its last digit.
This is recorded as an **unresolved representation discrepancy**, not a claimed
erratum. A targeted DOI/title search found no correction; absence from this
search does not prove none exists. The printed fit is archived only as a
diagnostic and is not treated as the unavailable direct Q/moment table.

The immediate next input is the paper's electronic supplement containing
1 K Q, first and second moments, or author-confirmed corrected data. The full
article is already available; a publisher supplement attempt returned HTTP 403.
Even after reconciling the representation, a full-domain candidate needs an
explicit upper-temperature tail assessment and consistent zero/mass handling,
then fixed-network finite substitutions. Current retention is provisional.

.. list-table:: Literature and calculation follow-ups
   :header-rows: 1
   :widths: 32 35 33

   * - Item / DOI
     - Already inspected
     - Needed for a candidate
   * - 10.1016/j.jqsrt.2014.03.012
     - Full text, methods and printed coefficients
     - Electronic Q/moment supplement or corrected data; reconcile the fit
   * - 10.1016/j.jqsrt.2025.109568
     - NH3 methods and official TIPS arrays
     - Explain high-T differences against converged direct moments
   * - 10.1093/mnras/stz2778
     - Author text, state example, official PF/definition
     - Extend or bound state completeness if used above its published target
   * - Historical NASA and ATcT provenance
     - Pinned source records; same-version balanced reaction
     - No blanket demand for every historical original before this review merges

Reproduction and MCEG scope
---------------------------

Run the offline review and tests from the repository root::

   python mceg/reviews/nh3_source_review.py
   pytest tests/model_evidence/nh3_source_review_test.py

The packet is ``tests/data/validation/nh3_source_review/20261104``. Acquisition
URLs, original full-download hashes, factual extracts, inspection limits and
source perturbation rejection are recorded. The existing 75-case grid is
reused explicitly; every NH3 potential is reproduced within 7.2e-15.

``mceg/contracts/nh3_source_review.yaml`` supports source reconstruction,
scoped comparisons and the provisional-retention decision. Complete physical
accuracy remains unknown, including high-temperature moment completeness,
natural-isotope mixtures, the exact energy-zero mapping and correlated ATcT
uncertainty. These concrete limitations permit a source-only merge while
keeping a scientific replacement decision open.
