H2O energy and thermal-source investigation
===========================================

This 2026-11-03 review follows #308 and #309. **Prioritize the 2025 evaluated
water thermal functions as a candidate for a later curated update.** The
current NASA thermal functions differ from that evaluation by up to about
1.8% in the molecular partition function over the sampled 1500-4000 K
domain. This difference survives alignment of energy, atomic reference and
translation. The balanced ATcT 1.172 atomization anchor remains appropriate
for this comparison. Production is unchanged in this source investigation;
a consistent continuous candidate and finite equilibrium substitution are
the next steps, rather than a claim that the existing thermal choice has
already been validated.

Priority and current construction
---------------------------------

Water's importance in hot planetary atmospheres motivates checking even
modest thermochemical differences. The full author manuscript of
`Polyansky et al. (2018), POKAZATEL <https://arxiv.org/html/1807.04529v1>`_,
DOI `10.1093/mnras/sty1877 <https://doi.org/10.1093/mnras/sty1877>`_,
discusses high-temperature stellar and planetary spectra. Opacity coverage
and spectral detections do not independently validate an equilibrium
constant. Our current ``curated-tio-v1`` archive provides a separate
numerical priority diagnostic: H2O contains **90.6115895% of the oxygen
budget** at 1500 K, 100 bar and C/O=0.5888436553555908, with mole fraction
0.0007625443. Oxygen conservation and the actual H2O chemical potential
are checked in all 75 saved cases, with 537 species. This is an existing
hydrogen-rich gas grid, not a source substitution or an abundance error
bound.

The inherited reaction is 2 H(g) + O(g) -> H2O(g), at 1 bar. The original
major-molecule choice separates NASA thermal shape from the same-version
ATcT energy. Fresh official ATcT 1.172
`H <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=19>`__,
`O <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=21>`__
and `H2O <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=27>`__
pages reproduce Hf(0)=216.034, 246.844 and -238.907 kJ/mol:

.. math::

   D_0 = 2H_f(\mathrm{H},0)+H_f(\mathrm{O},0)-H_f(\mathrm{H_2O},0)
       = 917.819\ \mathrm{kJ\,mol^{-1}}.

Dataset DOI: `10.17038/CSE/2440254 <https://doi.org/10.17038/CSE/2440254>`_.
This is total atomization to ground-state atoms, not the first H + OH
threshold near 41145.92 cm^-1 quoted in the water provenance. Neither an
electronic well depth nor that first threshold replaces the total D0.
The water page's 20 displayed contributors account for 65% of its network
provenance; 305 contributors are needed for 90%. The original combustion
and spectroscopic measurements were not all independently reinspected.
The reaction page did not expose a usable covariance result. Displayed
species uncertainties are correlated and are not combined in independent
quadrature; a rounded zero atomic uncertainty is not an exact zero.

The freshly retrieved official NASA coefficient file, pinned in the
source packet, matches the frozen H2O/H/O adoption records. Its water
header cites Cox (1989), Woolley (1987) and TRC (10/88) tuv25; the original
Woolley/TRC pages were not obtained. Native NASA and JANAF atomization
energies are 917.782492 and 917.782 kJ/mol. NASA's original R=8.314510 is
used for its energy conversion. The adopted minus native NASA D0/k is
5.019946969 K, giving current/raw-NASA K ratios of 1.00335224 at 1500 K
and 1.00125577 at 4000 K. This small anchor correction cannot explain the
modern thermal residual below. Dimensionless NASA caloric functions are
multiplied by modern R for the thermal comparisons, matching the runtime
gauge. Legacy JANAF comparisons remain diagnostics, not selection rules.

Modern primary sources and conventions
---------------------------------------

The inspected sources include the POKAZATEL author manuscript and current
`official ExoMol distribution
<https://www.exomol.com/data/molecules/H2O/1H2-16O/POKAZATEL/>`_, the full
author manuscript of `Furtenbacher et al. (2016)
<https://arxiv.org/html/1611.00324v1>`_, DOI
`10.1063/1.4967723 <https://doi.org/10.1063/1.4967723>`_, and the full
`2025 accepted manuscript at NIST
<https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959474>`_ by
Furtenbacher, Harvey and Csaszar, DOI
`10.1063/5.0273661 <https://doi.org/10.1063/5.0273661>`_. The latter updates
constants and empirical energies and provides explicit uncertainty
estimates. Its official supplementary
`PW2024 levels <https://doi.org/10.60893/figshare.jpr.29087330.v1>`_,
`partition functions <https://doi.org/10.60893/figshare.jpr.29087339.v1>`_
and `thermochemical functions <https://doi.org/10.60893/figshare.jpr.29087342.v1>`_
were retrieved in full. These direct thermochemical evaluations are a
stronger candidate starting point than an opacity temperature limit.
Their lineage shares potential-energy surfaces and empirical spectroscopy
with POKAZATEL; agreement is not independent physical validation.

The raw ExoMol and PW2024 partition sums use nuclear-spin weights 3 for
ortho and 1 for para, multiplied by 2J+1. Thermochemical normalization
removes the overall proton spin product four: Q_internal=Q_raw/4.
State-dependent ortho/para weights remain, and their thermally summed
ratio approaches 3:1 in this domain; a fixed ratio is not imposed on each
level. The allowed permutation-symmetry levels already encode exchange
symmetry, so no second division by two is made. The para 0_00 ground state
is zero, while the ortho ground state is about 23.79436 cm^-1 above it.
There is no separate ortho energy reset or added zero-point energy. NASA
H/O atomic thermal functions are retained; the H electronic doublet is
distinct from proton nuclear spin.

The 2025 entropy table **already removes the spin product four**. This is
confirmed numerically from Q and its first moment, and by the 2016 entropy
convention. Removing it a second time would introduce a factor-four error.
Another source detail needs explicit correction: the supplementary CSV
labels enthalpy and its uncertainty as kJ/mol, but their numerical values
are **J/mol**. At 298.15 K, the listed value 9904.040932 equals the
independent reconstruction R*T*(5/2+Q'/Q), or 9.904040932 kJ/mol; even its
1 K value verifies the scale. The frozen packet preserves original headers
and tests this interpretation. Printed high-temperature moments and
caloric values are rounded separately, so their closure tolerances retain
that precision limit.

All molecular thermal comparisons use 1 bar. The 2025 thermochemistry
represents H2(16O), with mass 18.01056468403 u. NASA uses 18.01528 u.
Replacing only the translational mass changes entropy by
1.5*R*ln(18.01528/18.01056468403)=0.003264766 J/mol/K. We apply that shift
before comparing G or remove translation before comparing internal Q.
This represents the dominant isotopologue's internal states at the
element-aggregate mass. It is **not** an exact natural-isotope mixture;
minor-isotope internal functions, mixing entropy and isotope-dependent
reaction energies have not been constructed.

Normalized thermal results
----------------------------

The following K ratios hold the current total D0 and NASA atomic thermal
functions fixed, changing only the water molecular thermal function. They
are formation-constant ratios, not abundance ratios. Q ratios and H/S
ratios agree to the separately printed source precision (up to about
1.5e-5 in K at these nodes).

.. list-table:: 2025 evaluation relative to the current NASA thermal model
   :header-rows: 1

   * - T, K
     - Internal Q ratio
     - K ratio from H and S
     - Cp difference, J/mol/K
     - Published expanded Cp uncertainty, J/mol/K
   * - 1500
     - 0.999926
     - 0.999926
     - -0.17380
     - 0.00035
   * - 2000
     - 0.998001
     - 0.998001
     - -0.37657
     - 0.0024
   * - 2500
     - 0.994653
     - 0.994653
     - -0.50534
     - 0.0076
   * - 3000
     - 0.990464
     - 0.990478
     - -0.48617
     - 0.017
   * - 3500
     - 0.986121
     - 0.986136
     - -0.37073
     - 0.029
   * - 4000
     - 0.982063
     - 0.982070
     - -0.34550
     - 0.042

The 4000 K partition discrepancy is about 1.79%, compared with the 2025
reported expanded relative Q uncertainty of 6.29e-5. The latter is a
conservative upper/lower-bound estimate with coverage factor two; it is
not an uncertainty for natural-isotope chemistry, the total ATcT energy,
or a complete planetary abundance calculation. The modern evaluation is
preferred as a candidate on its evaluated spectroscopy, thermodynamic
scope and uncertainty treatment, not because it is closer to a legacy
source.

Dataset boundaries and upper-state sensitivity
-----------------------------------------------

The current ExoMol definition identifies version 20230621 and an updated
MARVELized distribution. Direct parsing finds 810269 states, J<=69 and
maximum energy 45000.633993 cm^-1. PW2024 has 810270 states with the same
limits. These actual file limits differ from the 2018 paper's stated
J<=72 and 41000 cm^-1 coverage; the current downloaded file is not
silently relabeled as the original 2018 release. All supplied states are
included in the direct sums. The actual temperature columns are read from
the separate PF and Cp files; tabulated range is not a completeness claim.
Current ExoMol sums match its published Q within 1.2e-6 relative and Cp
within 0.0035 J/mol/K at the diagnostic nodes. PW2024 direct Q lies within
the 2025 published expanded Q uncertainty at every sampled node.

A diagnostic cut at the first H + OH threshold removes 22 current ExoMol
states or 23 PW2024 states. At 4000 K they contribute less than 2.5e-8 of Q
and 3e-5 J/mol/K to Cp. This is sensitivity to represented levels, not a
bound on missing states. Exceeding the first threshold alone is not used
to reclassify every supplied level as an unbound continuum state.

The full PW2024 discrete sum also does **not** reproduce every central
2025 tabulated value exactly. At 4000 K, evaluated Q=39760 is larger than
the direct sum by about 1.3444 (direct/evaluated=0.999966188), and evaluated
Cp exceeds direct Cp by 0.067981 J/mol/K. The sign is consistent with a
positive extra contribution, but this Cp difference exceeds the published
expanded Cp uncertainty of 0.042 J/mol/K and is not explained by removing
the few above-threshold discrete levels. The 2016 work, Sec. 2.5 and Eqs. 12-16,
treats unbound OH + H scattering approximately; the 2025 Sec. 3.3 explicitly
retains its contribution in the uncertainty treatment. This lineage and
the numerical excess support keeping the evaluated-table scope distinct
from the discrete-state sum. We have not rerun the scattering model or
established that it uniquely accounts for the entire excess. The
approximately 3.4e-5 Q difference between these modern calculations is
much smaller than their roughly 0.018 residual relative to NASA.

Decision, MCEG and concrete follow-ups
---------------------------------------

``mceg/contracts/h2o_source_review.yaml`` supports source construction,
normalized thermal comparison and numerical replay, while leaving
complete physical accuracy unknown. No production coefficients, default
identifier, runtime bounds or equilibrium endpoints change here.

* Build a small continuous candidate from the 2025 evaluation, retaining
  the current ATcT total atomization energy and NASA atomic gauge. Verify
  G/H/S/Cp derivative closure, interval joins and independent off-node
  values over 1500-4000 K. Resolve source units before fitting.
* State whether that candidate includes the evaluated table's additional
  thermodynamic contributions or only the PW2024 discrete sum. Assess
  consistency of an unbound/scattering contribution with independently
  represented OH and H chemical species. Compare those two scopes as a
  sensitivity, rather than calling them the same model. Preserve the
  explicit dominant-isotope approximation or implement a justified
  isotope treatment.
* Run a conserved finite substitution on a fixed species network and
  budget before adopting a replacement. The saved 90.61% oxygen footprint
  establishes priority but does not predict that substitution's effect.
* Record the same-version ATcT reaction covariance when accessible, and
  inspect original energy or NASA references if a concrete discrepancy
  requires them. Obtaining every historical paper is not a merge condition
  for this source review. The modern full texts and numerical supplements
  needed to begin the candidate are already available.

The compact packet is ``tests/data/validation/h2o_source_review/20261103``.
Full papers and large state lists are not redistributed. Download hashes,
source status, selected original rows and derived full-state moments are
frozen. Offline replay and optional extraction from the pinned originals::

   PYTHONPATH=src python mceg/reviews/h2o_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/h2o_source_review_test.py
   PYTHONPATH=src python mceg/reviews/h2o_source_review.py --extract /path/to/originals --output /tmp/h2o-extracted
