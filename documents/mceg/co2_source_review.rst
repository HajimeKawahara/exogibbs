CO2 energy and thermal-source investigation
===========================================

This 2026-11-03 investigation follows #308 and #309. **Keep the current
NASA thermal functions plus ATcT 1.172 energy in production for this review;
prefer the Tashkun--Harvey thermodynamic evaluation as the basis of a
bounded thermal candidate.** Its modern effective Hamiltonian and uncertainty
analysis give a scientific reason to improve the molecular thermal model,
even though the differences are small. A representative-isotope diagnostic
is not yet a consistently normalized natural-composition replacement.
No curated coefficients or equilibrium solutions are changed here.

Why CO2 deserves attention
--------------------------

The JWST Transiting Exoplanet Community Early Release Science Team identified
CO2 absorption near 4.3 micrometres in WASP-39b: the inspected primary article
and abstract, `10.1038/s41586-022-05269-w
<https://doi.org/10.1038/s41586-022-05269-w>`_, appeared online in 2022 and in
Nature 614 (2023). This motivates priority independently of a large
NASA discrepancy. The approximately 1000 K planet is below our 1500 K
runtime lower bound; its absorption does not validate this equilibrium
model or the 1500-4000 K interval.

The inherited energy choice
---------------------------

The neutral reaction is C(g) + 2 O(g) -> CO2(g), at 1 bar. The inspected
`ATcT 1.172 index
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
reproduces the frozen adoption values Hf(0): C = 711.381, O = 246.844 and
CO2 = -393.111 kJ/mol. Thus the total ground-state atomization energy is

.. math::

   D_0 = 711.381 + 2(246.844) - (-393.111)
       = 1598.180\ \mathrm{kJ\,mol^{-1}}.

The inspected `CO2 provenance page
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=25>`_
lists eight principal contributors accounting for 91.1% of provenance,
largely graphite-combustion measurements. The O provenance page was also
read; the C-specific page was inaccessible. Original combustion papers
were not individually inspected. Their absence is recorded, without
requiring all of them before merging this source assessment. Dataset DOI:
`10.17038/CSE/2440254 <https://doi.org/10.17038/CSE/2440254>`_. The displayed
species uncertainties are correlated network results; adding independent
error bars would not give a justified reaction uncertainty.

The molecular zero is the ground rovibrational level, not the bottom of an
electronic potential well. The inspected NIST atomic tables give
`C <https://physics.nist.gov/PhysRefData/Handbook/Tables/carbontable5.htm>`__
3P0,1,2 at 0, 16.41671, 43.41350 cm^-1 and
`O <https://physics.nist.gov/PhysRefData/Handbook/Tables/oxygentable5.htm>`__
3P2,1,0 at 0, 158.265, 226.977 cm^-1. Existing NASA atomic thermal functions
are retained; a term centre or a constant factor of nine is not substituted.

The frozen NASA record cites Gurvich (1991), volume 2, part 1 p27 / part 2
p24. The `NASA report metadata
<https://ntrs.nasa.gov/citations/20020085330>`_ were inspected; the original
Gurvich pages were not obtained. Native, atom-balanced D0 is 1597.918774
kJ/mol for NASA and 1597.916 kJ/mol for JANAF. NASA's historical
R = 8.314510 J/mol/K is kept in its energy convention. The existing hybrid
therefore differs from native NASA by D0/k = 32.513472 K, giving the exact
K ratio exp(32.513472/T): 1.021912 at 1500 K and 1.008161 at 4000 K.
This energy adjustment is distinct from the molecular thermal comparison.
At the four available 1500, 2000, 3000, 4000 K JANAF nodes, matching both
D0 and atomic functions leaves JANAF/NASA K = 1.000568-1.001491.
No interpolation of missing JANAF nodes is introduced.

A direct modern thermodynamic source
-------------------------------------

The full NIST author manuscript of Tashkun and Harvey, *Partition Functions
and Ideal-Gas Thermodynamics for Carbon Dioxide*, JPCRD 54, **023102** (2025),
`10.1063/5.0276615 <https://doi.org/10.1063/5.0276615>`_, and all 13 official
tables were inspected. The dataset `10.18434/mds2-2364
<https://doi.org/10.18434/mds2-2364>`_ was published in 2021 from work completed
in 2020; the 2025 paper is not a new numerical release of that dataset.

The effective Hamiltonian uses extensive measured transition frequencies
and propagates its fitted-parameter covariance to thermodynamic functions.
It includes all 12 stable isotopologues. The dominant 12C16O2 (626) model
contains 11,590,044 levels, J up to 342 and a nominal 44,000 cm^-1 dissociation
cutoff. This cutoff concerns molecular dissociation to CO + O, and is not
the total C + 2 O atomization energy above. It must not replace the adopted
ATcT anchor. The ground electronic state is used; excited-state, continuum
and model-extrapolation errors are not certified by parameter covariance.

The author's PDF prints a plus sign for the squared mean in its heat-capacity
equation (5). The statistical-mechanical identity requires a minus:

.. math::

   C_p/R = 5/2 + \langle (E/kT)^2\rangle - \langle E/kT\rangle^2.

This is a formula-level problem, not evidence that the numerical tables
used that sign. Five-point derivatives of the separately tabulated H, S
and Q agree with published Cp/R within 1.7e-6 and with H/(RT) within 9e-8
at eight 500-4000 K nodes. Using the printed plus instead would add
20.50-36.55 to Cp/R at 1500-4000 K. Equation (18)'s parameter derivative is
also consistent with the variance sign. The evidence therefore supports
treating equation (5) as a typographical error while testing the data
independently; no table correction is applied.

The tables retain CODATA 2014 constants, including c2 = 1.43877736 K cm
and R = 8.3144598 J/mol/K. This review preserves their dimensionless values
and multiplies both NASA and NIST Cp/R by the same modern R when comparing
joules. NASA's separate historical-R contribution is recorded. A first-order
rescaling of NIST Q to the modern c2 would change ln Q by less than 1.5e-6
in the production interval, much smaller than the source differences here.
That diagnostic is not presented as an exact regeneration of the Hamiltonian.

Normalization and thermal comparison
--------------------------------------

For 626, both ExoMol state files have zero at the lowest rovibrational state
and every retained weight is 2J+1. Nuclear spins are zero; exchange-symmetry
selection is already implemented in the allowed levels. No extra factor
of two, symmetry division, spin multiplier or ZPE shift is applied.
The comparison removes translation explicitly:

.. math::

   Q_{\rm NASA}=\exp[-g_{\rm thermal}]/
   \left[(2\pi m k_B T/h^2)^{3/2} k_B T/p^\circ\right],
   \qquad g_{\rm thermal}=(G-H(0))/(RT).

The NASA record has natural-composition mass 44.0095 u; 626 has
43.989829239132 u. Changing only translation multiplies Q times the
translational factor by 1.000670824. In the table below, all internal-Q
ratios keep the NASA mass, atomic functions and ATcT anchor fixed. They
are representative-isotope thermal proxies, not final natural-mixture
formation constants or abundance ratios.

.. list-table:: Independent thermal diagnostics
   :header-rows: 1

   * - T, K
     - NIST 626 Q / NASA internal Q
     - Dozen states Q / NIST 626 Q
     - Natural NIST minus NASA Cp, J/mol/K
     - NIST natural standard u(Cp), J/mol/K
   * - 1500
     - 1.000651
     - 1.000027
     - +0.06966
     - 0.00235
   * - 2000
     - 1.001629
     - 1.000022
     - +0.10806
     - 0.00923
   * - 3000
     - 1.004358
     - 0.999779
     - +0.24402
     - 0.06394
   * - 4000
     - 1.007913
     - 0.995641
     - -0.05455
     - 0.27105

Six production-domain nodes include 2500 and 3500 K; two lower nodes are
diagnostics only. The maximum sampled natural Cp difference is 0.24782
J/mol/K at 3500 K (0.395%). At 4000 K the difference is smaller than NIST's
own standard uncertainty. Neither uniform superiority at every temperature
nor an atmospheric error bound follows from these numbers.

NIST's natural-composition H, S and Cp reproduce abundance-weighted means
of its 12 isotope tables, within 3e-7 in dimensionless H, S and Cp. They contain nuclear-spin
conventions and no additional ideal-mixture entropy term. The mean logarithm
of the isotope nuclear-spin product is 0.0087784. Thus natural S cannot simply
be put into the NASA atomic reference convention: consistent isotope
fractions, spin removal, entropy convention and isotope energy zero must be
specified together. Natural versus 626 Cp differs by at most 0.000257 R at
these production nodes. This is a small, measurable approximation that can
be bounded in a candidate; it is not a reason to demand every historical
paper before further work.

ExoMol versions and completeness
--------------------------------

Full published texts and current official files were inspected for
`UCL-4000 (2020), 10.1093/mnras/staa1874
<https://doi.org/10.1093/mnras/staa1874>`_ and
`Dozen (2026), 10.1093/mnras/staf2135
<https://doi.org/10.1093/mnras/staf2135>`_. Definition versions are 20200630
and 20260115, respectively. Dozen improves spectral intensities and empirical
energy coverage. This improves opacity modeling, but is not itself a new
thermochemical measurement. Both use Ames-2 potential-energy surfaces and
TROVE; Dozen's empirical CDSD/HITRAN inputs also overlap NIST's spectroscopic
ancestry. Agreement is not three independent experimental confirmations.

Every state was summed: current UCL has 3,562,798 levels, J <= 230, and
Dozen has 3,646,814 levels, J <= 250; both extend to just below 36,000 cm^-1.
The original UCL paper's 3,480,477 states and J <= 202 refer to its line-list
range, while its partition calculation included J through 230. Full current
UCL sums give Q = 114728.7675 and 376550.9118 at 3000 and 4000 K, reproducing
the integer UCL values in NIST's Table 6 to about one unit.

However, current UCL ``.pf`` values are 114428.1111 and 367222.6396 at those
temperatures. At 4000 K the full state sum exceeds that PF by 2.5402%.
Restricting current states to E <= 26,000 cm^-1 reproduces the distributed
PF within 1.4e-6 throughout 1500-4000 K and its distributed Cp within
0.0036 J/mol/K. Restricting J to 202 does not explain the discrepancy.
This identifies a numerical cutoff explanation, without asserting an
unverified release history or intent. The PF/Cp pair must not be described
as the complete 36,000 cm^-1 state sum. It is unsuitable as a direct
replacement without specifying its cutoff.

Dozen full-state sums reproduce its distributed PF within 1.5e-6.
Nevertheless, at 4000 K Dozen Q is 0.436% below NIST and Cp/R is 7.28630,
versus NIST 7.59508. Its thermal moments become more sensitive to missing
high states than Q alone suggests. Dozen states above 30,000 cm^-1 already
supply 0.745% of Q at that temperature. NIST's more complete high-energy
construction is preferred for the thermal candidate, with Dozen as a
cutoff-sensitive comparison, rather than selecting whichever agrees best
with NASA. A PF file tabulated to 5000 K is not completeness validation;
Dozen's main line list is intended for at least 3000 K, with about 2000 K
for the minor isotopologues and distinct transition-intensity limits.

The natural NIST table explicitly warns that non-dissociating CO2 becomes
unphysical above about 3500 K. The paper explains the artificial frozen-species
Cp maximum near 4000 K: an actual reacting gas dissociates. ExoGibbs includes
CO, O and other products, so a bound-species thermal function is not the
same as the bulk heat capacity of a constrained pure-CO2 gas. This warning
does not by itself invalidate every bound-species Q at 3500-4000 K; it does
require a bounded assessment of high-state completeness, the dissociation
cutoff and continuum treatment, rather than claiming physical validation
from the tabulation's 6000 K endpoint.

Application footprint and next action
--------------------------------------

The existing ``curated-tio-v1`` archive contains 75 converged gas cases and
537 species. All saved CO2 potentials equal the retained polynomial; C and
O conservation residuals are below 2.3e-13 relative. The largest CO2 shares
occur at 1500 K, 1 bar and C/O = 0.58884: 0.01316% of carbon, 0.01550% of
oxygen and mole fraction 6.5142e-8. This hydrogen-rich, hot grid is not a
survey of cooler, metal-enriched or CO2-dominated planets. A small saved
footprint does not reduce CO2's spectral importance. No finite replacement,
new atmospheric model or observational validation was performed.

The next useful step is a derivative-consistent NIST-based candidate with
explicit isotope/spin/entropy normalization, unchanged balanced ATcT energy
and existing NASA atoms, followed by a finite substitution over the current
domain. Compare representative-626 and natural-composition approximations
quantitatively, and retain high-temperature limitations. Resolving the UCL
file-pair documentation is useful provenance work but does not block that
NIST-based candidate or this review. Reconstructing the complete ATcT
covariance or obtaining every Gurvich/combustion original is likewise not
a prerequisite for recording the present scientifically motivated direction.

Reproducibility and MCEG
-------------------------

``mceg/reviews/co2_source_review.py`` replays the compact, frozen packet in
``tests/data/validation/co2_source_review/20261103`` without network access.
Its ``--extract DIRECTORY`` option reads all official NIST tables and both
complete compressed ExoMol state files, computes energy moments and records
original SHA-256 values. Full papers and large state files are not vendored.
The packet distinguishes downloaded originals, inspected pages and unavailable
original papers. The shared full unit suite applies because runtime and
unit-test inputs are unchanged; targeted evidence tests additionally check
balanced energy, independent thermodynamic identities, isotope aggregation,
cutoff diagnostics and conserved archived element budgets.

``mceg/contracts/co2_source_review.yaml`` records supported numerical/source
claims and a scoped scientific candidate direction. Complete physical
accuracy remains **unknown**. Supported source attribution, internally
consistent thermodynamics and a sensible next candidate do not establish
planetary abundance accuracy.
