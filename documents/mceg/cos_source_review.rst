COS / OCS: energy revision and thermal completeness
===================================================

Decision
--------

Retain the current JANAF thermal model provisionally and prioritize an
ATcT 1.228 energy-only candidate. This review changes no curated coefficients.
``COS`` and spectroscopic ``OCS`` identify the same neutral ``C1O1S1`` molecule.
The current column uses JANAF molecular thermal functions and JANAF balanced
D0, with NASA atomic thermal functions. Its piecewise quintic representation
is restricted to 1500--4000 K. NASA cites Gurvich (1991); original volumes
were not independently reaudited. JANAF C-094 carries a March 1961 evaluation.

The `ATcT 1.228 evaluated network
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
provides the previously missing OCS energy anchor. Same-version Hf(0), in
kJ/mol, is C=711.370, O=246.843, S=276.85, OCS=-141.79. Balanced
D0=1376.853 kJ/mol exceeds the current JANAF 1371.241 by 5.612 kJ/mol.
Current NASA D0=1374.747877 kJ/mol uses its native gas constant. Updating
energy alone would multiply current K by 1.56828, 1.25231 and 1.18381 at
1500, 3000 and 4000 K. This is a concrete next candidate; no atmospheric
response has yet been calculated. Network covariance is unavailable.

Thermal evidence
----------------

Current/NASA K is 0.754787/0.884794/0.928503 at those three temperatures.
With D0 and atom functions fixed, JANAF/NASA thermal ratios are
0.999239/1.018040/1.031514. Their Cp differences are
+0.36064/+0.67064/+0.83819 J/mol/K after conversion to common R. Energy
and thermal differences therefore require separate judgments.

`OYT8 <https://doi.org/10.1093/mnras/stae1110>`_ has a recommended maximum
of 2000 K. The distributed PF goes to 5000 K, but missing J>223 contributions
already prevent convergence at 3000 K. The principal isotope has nuclear
spin factor 1. Its ground-relative Q/NASA is
1.002328/0.963262/0.847090 at 1500/3000/4000 K. The latter two values are
finite-list diagnostics. Changing translation from natural mass 60.0751 u
to isotope mass 59.96698562 u gives corresponding fixed-atom K proxies
0.999624/0.960663/0.844805. No isotopic mixture is constructed.

`TIPS2025 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_ uses a more extended
Ames direct sum and changes to a product approximation at 3380 K for the
principal isotope. The pinned HAPI v1.3.0.0 arrays cite the TIPS v1.2 data.
Q/NASA is 1.002674/0.998207/0.986709 at 1500/3000/4000 K. The grid join
at 3380 K lies 1.0073 percent above continuation of the preceding local
log-Q slope. Consequently it cannot be used unchanged as a differentiable
free-energy model. The July 2026 v3 release note corrects the label of a
different molecule; this review does not claim a byte-for-byte latest-data
comparison.

Five-point derivatives at 10 K and 20 K spacing separate caloric behavior
from Q agreement. TIPS Cp at 3000 K is 56.79/57.19 J/mol/K for those two
steps, compared with NASA 62.79. Table rounding contributes a conservative
1.92 J/mol/K bound at the 10 K step. OYT8 instead gives 51.579 J/mol/K,
well outside its recommended range. Small Q differences alone do not
establish derivative accuracy. A later thermal candidate must resolve the
join, high-temperature completeness, isotope and caloric conventions.

Footprint and scope
-------------------

Replaying the existing 75-case, 537-species ``curated-oh-v1`` grid reproduces
all COS potentials. The maximum sulfur fraction in COS is 1.91890e-5
(case 39). This is an archived footprint, not a new substitution or a claim
that COS is unimportant in other atmospheres. The compact packet includes
PF data, primary-source access records, independent calculations and tests.
MCEG supports source attribution, normalized comparisons and energy-candidate
priority; full physical accuracy and historical source completeness remain
unknown. No additional inaccessible legacy paper is required to merge this
scoped review.

Completed energy-candidate follow-up (2026-11-05)
-------------------------------------------------

The separately recorded :doc:`cos_candidate` now implements the same-version
ATcT 1.228 energy-only candidate and completes 150 new endpoint solves,
conservation, strict cold controls, finite paths and independent FastChem
checks. This dated source-review comparison is preserved; its original
no-substitution footprint is not relabeled as a replacement calculation.
Production curated is unchanged, and the thermal/physical limitations remain.
