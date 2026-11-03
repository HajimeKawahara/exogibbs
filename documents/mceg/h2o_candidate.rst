H2O evaluated thermal candidate and finite substitution
=======================================================

This 2026-11-04 study follows :doc:`h2o_source_review` and updates PR #330.
It constructs continuous thermal candidates from the 2025 water evaluation
and the PW2024 discrete levels, then changes H2O alone in the existing gas
equilibrium network. The dated source investigation and its uncertainties
remain intact. This study evaluates a possible update; production curated
data are unchanged.

Shared energy and molecular conventions
----------------------------------------

Both candidates retain the current ATcT 1.172 total atomization energy
917.819 kJ/mol for 2 H(g) + O(g) -> H2O(g), the NASA atomic thermal
functions, and standard pressure 1 bar. They use H2(16O) internal states
at the aggregate molecular mass 18.01528 u. This is an explicit dominant
isotope approximation, rather than a complete natural-isotope reaction.
Changing molecular translation from 18.01056468403 u adds
0.003264766 J/mol/K to entropy. The molecular zero remains the para
rovibrational ground level; the ortho ground remains above it. There is
no additional ZPE or symmetry divisor.

The source is the full inspected 2025 evaluation by Furtenbacher, Harvey
and Csaszar, `10.1063/5.0273661 <https://doi.org/10.1063/5.0273661>`_,
with its official thermochemical and partition CSVs and PW2024 levels.
Their hashes, versioned supplement DOIs and access scope are preserved in
the source-review packet. Source entropy already removes the proton
nuclear-spin product four. The raw discrete partition sum retains ortho
and para weights 3 and 1 before division by four. The published enthalpy
numbers are J/mol despite the CSV's kJ/mol labels; the previous independent
partition-moment check establishes that interpretation.

One continuous Gibbs function for each candidate
-------------------------------------------------

The primary ``evaluated`` candidate fits Cp/R with a degree-8 polynomial in
u=(T-2750)/1250. It uses 251 published Cp rows, at 10 K intervals between
1500 and 4000 K, weighted by inverse published expanded Cp uncertainty.
These are numerical approximation weights, not independent measurement
errors or a new fitted covariance. Analytical integrals give H and S,
anchored at the source's 1500 K values, and G=H-TS. The entropy integral
includes its logarithmic term. All four thermal functions therefore obey
the same derivatives throughout the interval.

Directly interpolating the rounded source H and S as Gibbs endpoints is
less suitable here: six nodes at 500 K spacing amplify that rounding
to Cp errors up to 0.04624 J/mol/K and S errors up to 0.001912 J/mol/K.
The integrated-Cp construction avoids fitting those rounding steps. Its
approximation errors against all 2501 published integer-temperature rows
are:

.. list-table:: Evaluated candidate approximation checks
   :header-rows: 1

   * - Quantity
     - Maximum absolute difference
     - Maximum difference / published expanded uncertainty
   * - H
     - 0.5587 J/mol
     - 0.484
   * - S
     - 0.00002298 J/mol/K
     - 0.238
   * - Cp
     - 0.001426 J/mol/K
     - 0.0545

The 2250 Cp values withheld from fitting have maximum difference
0.001159 J/mol/K. H/S columns other than the two anchor values, and every
Q value, are independent checks on that fit. The largest ln K difference
from the independently rounded Q column is 1.551e-5; its high-temperature
printed precision must not be confused with a fitted physical uncertainty.
The fit's continuous minimum Cp is 47.1444 J/mol/K, checked at polynomial
stationary points and endpoints.

The alternative ``discrete`` candidate sums all 810270 supplied PW2024
levels and fits molecular (G-H0)/R with the existing C2 quintic Hermite
construction at six 500 K-spaced nodes. The same Gibbs function determines
H, S and Cp. Forty-five withheld direct state sums give maximum absolute
error in ln K of 8.64e-7 and Cp error 0.002098 J/mol/K. Common ground zero, thermal spin
weights, interval joins, scalar/vector/JIT evaluation and automatic first
and second derivatives are checked. Both candidates are confined to
1500-4000 K; the scalar interface rejects and the traced interface returns
NaN outside that domain.

.. list-table:: Thermal-only formation-constant ratios at fixed D0 and atoms
   :header-rows: 1

   * - T, K
     - Evaluated / current
     - Discrete / current
     - Evaluated / discrete
   * - 1500
     - 0.99992593
     - 0.99992594
     - 0.999999991
   * - 2000
     - 0.99800109
     - 0.99800108
     - 1.000000002
   * - 2500
     - 0.99465279
     - 0.99465277
     - 1.000000025
   * - 3000
     - 0.99046435
     - 0.99046389
     - 1.000000468
   * - 3500
     - 0.98612095
     - 0.98611535
     - 1.000005678
   * - 4000
     - 0.98207135
     - 0.98202981
     - 1.000042302

The major source effect is the approximately 1.79% thermal change relative
to NASA at 4000 K. The two modern candidate scopes differ by only about
4.23e-5 in ln K there. This latter number includes their representation and
source-table rounding effects; it is not a direct measurement of a missing
physical term. Candidate/current K ratios alone do not give abundance ratios.

Evaluated versus discrete thermodynamic scope
---------------------------------------------

The evaluated candidate retains the table's additional thermodynamic
contributions through Cp and the H/S anchors. The discrete alternative
adds no separately reconstructed continuum or scattering contribution.
Their 4000 K Cp difference is 0.06778 J/mol/K, still larger than the
published expanded Cp uncertainty of 0.042 J/mol/K. A few supplied levels
above the first OH+H threshold do not account for that difference; the
previous source study quantified their negligible sensitivity.

The full inspected `2016 treatment <https://arxiv.org/html/1611.00324v1>`_,
DOI `10.1063/1.4967723 <https://doi.org/10.1063/1.4967723>`_, Sec. 2.5 and
Eqs. 12-16, uses an OH+H scattering density proportional to the energy
derivative of the scattering phase shift. Such a contribution is not
simply another copy of a free OH and H population. Including OH and H as
independent species therefore does not, by itself, prove double counting.
Conversely, assigning that interaction contribution to a molecular
partition function in this ideal reacting mixture requires a consistent
chemical or cluster decomposition that has not been derived here.

The 2025 text explicitly retains unbound-state effects in its uncertainty
treatment. The additional central-table contribution remains a scoped
interpretation from its numerical difference and inherited method, not a
unique reconstruction of the scattering calculation. Comparing both
continuous candidates measures the consequence of this scope choice
without asserting that the two models have identical physical definitions.

Conserved finite substitutions
-------------------------------

Each candidate was solved from a cold initialization at the 75 archived
``curated-tio-v1`` conditions, giving 150 new endpoints on the same
537-species ion-inclusive network. Only H2O's formation potential changes;
elemental budgets, charge handling and every other species potential are
identical. The grid covers 1500, 2000, 2500, 3000 and 4000 K, pressures
1e-6 to 100 bar, and three C/O ratios. It is a hot hydrogen-rich ideal-gas
comparison, not a validation of transport, condensation or observed spectra.

Water's evaluated/current mole-fraction ratio ranges from 0.98207135 to
0.99999970. The largest relative change occurs where water is scarce;
the largest absolute mole-fraction and oxygen-budget response occurs at
4000 K, 100 bar and C/O=0.5888436553555908:

.. list-table:: Case 24, maximum absolute water response
   :header-rows: 1

   * - Quantity
     - Current
     - Evaluated candidate
     - Discrete candidate
   * - H2O mole fraction
     - 0.0002718623980
     - 0.0002710809908
     - 0.0002710791574
   * - Oxygen budget in H2O
     - 34.70623956%
     - 34.60649998%
     - 34.60626597%

The primary candidate lowers water by 0.28743% in this case; its share of
the oxygen inventory falls by 0.0997396 percentage points. The liberated
oxygen redistributes among other species: OH's mole fraction rises from
2.9656982e-5 to 3.0111594e-5 at this same point. CO, CH4 and HCN also
respond elsewhere in the grid even though their thermochemical inputs
are unchanged. The ratios in the thermal table therefore cannot simply
be assigned to every abundance.

Changing from evaluated to discrete thermodynamics changes water by at
most about 0.00423% relatively; discrete/evaluated ratios range from
0.999957700 to 1.000000009. The largest absolute difference in any species
mole fraction is 1.8334e-9. Among species with mole fraction at least 1e-12,
the largest absolute change in ln x is 1.054e-4; this all-species metric is distinct
from the water-only response. These are controlled source-scope
sensitivities, not confidence intervals for the physical atmosphere.

All endpoints converge. Mass-action residuals are below 1.54e-12 in log
units and relative charge residuals below 4.57e-14. H/C/O inventory errors
are tightly checked separately from rare-element errors. The largest
relative element residual, 1.35e-6, belongs to the trace fluorine
inventory and does not describe the precision of the oxygen response.
Nine tighter cold solves cover all three models at three selected cases.
Water ln x changes by at most 1.21e-13, and the evaluated/discrete water
response changes by at most 1.57e-13. The largest absolute mole-fraction
change under tightening is 3.13e-14. The more sensitive F component at
x=2.32e-8 changes by 1.35e-6 in ln x, so a 2e-6 visible-species tolerance
is reported alongside the separate water and absolute-fraction checks.

A five-point potential interpolation verifies conservation and monotonic
water amount at the highlighted case. Each endpoint also minimizes its
own Gibbs function within the recorded feasibility/roundoff bounds;
first-order inventory-residual terms are removed in the tighter convexity
check. Independent FastChem 4.0.3 calculations at two cases, for all three
models, agree within 5.89e-9 dex. These checks establish numerical response
to the specified change, not new experimental thermochemical validation.

Scientific decision and remaining scope
----------------------------------------

**Recommend the evaluated 2025 candidate for a scoped production adoption
on 1500-4000 K.** Its modern spectroscopic evaluation, caloric
consistency and quantified approximation errors improve the evidence
beyond the inherited NASA thermal model. Both modern scopes produce the
same direction of change, and their finite-equilibrium difference is
small compared with the NASA-to-modern change in this tested domain.
The tests now resolve the source review's missing continuous candidate
and fixed-network finite substitution.

A production adoption would still name the chosen evaluated or discrete
thermodynamic scope and its dominant-isotope approximation explicitly,
and preserve these numerical and domain checks in the runtime package.
The scattering interaction's chemical assignment remains a physical
limitation; small scope sensitivity does not prove that assignment. There
is no requirement to obtain every historical energy paper before making
that scoped adoption decision. The ATcT anchor was retained, so its
unavailable reaction covariance remains the dated source-audit follow-up.
No new complete physical-accuracy claim is made and no production
adoption is performed by this study.

Reproduction and evidence
-------------------------

The candidate packet is ``tests/data/validation/h2o_candidate/20261104``.
The source extractor verifies the original files against their previously
recorded SHA256 values. Published table rows, direct-state moments,
coefficients and withheld checks are sufficient for offline evaluation;
the complete papers and large level file are not redistributed.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/h2o_candidate.py
   PYTHONPATH=src pytest tests/model_evidence/h2o_candidate_test.py tests/model_evidence/h2o_equilibrium_impact_test.py

The independent finite-substitution packet is
``tests/data/validation/h2o_equilibrium_impact/20261104``. It retains all
endpoint potentials and log amounts, tighter controls, the finite path,
cross-solver diagnostics and the execution receipt. Its README gives the
optional full-rerun command; ordinary evidence tests are offline.

``mceg/contracts/h2o_candidate.yaml`` records this new study separately
from the dated source audit. Numerical candidate construction and finite
substitution do not establish complete planetary physical accuracy.
