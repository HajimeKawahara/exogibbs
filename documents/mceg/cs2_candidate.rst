CS2: retained NASA thermals with a balanced ATcT energy update
==============================================================

This calculation follows :doc:`cs2_source_review` and Issues #308/#309.
**The ATcT 1.228 energy-only candidate is scientifically preferable for a
scoped energy refresh, with the current NASA thermal function retained.**
The finite substitution is now evaluated. This evidence PR does not alter
curated data, choose a new thermal model, or merge an adoption.

Balanced construction
---------------------

The existing hybrid uses native NASA dimensionless thermal functions and
ATcT 1.172 for the atomization energy. For the reaction C(g) + 2 S(g)
-> CS2(g), use gas-phase ground-state atoms and the same ATcT version for
all participants. The two balanced 0 K energies, in kJ/mol, are::

    ATcT 1.172: 711.381 + 2(276.830) - 115.880 = 1149.161
    ATcT 1.228: 711.370 + 2(276.850) - 115.170 = 1149.900

The resulting **delta D0 = +739 J/mol** is applied once to the literal
current reaction coefficients from ``gas_v1.json``:

.. math::

   h_{new}(T)=h_{current}(T)-\frac{739}{RT},
   \quad \Delta G=\Delta H=-739\ {\rm J/mol},
   \quad \Delta S=\Delta C_p=0.

The ``curated_cos`` baseline identifies ``curated-cos-v1``. Its entire
2501-temperature column is reproduced before applying the shift; the new
potential is also independently reconstructed from the original molecular
and atomic NASA records at the new balanced D0. Native NASA R=8.314510
remains in those dimensionless thermal terms and native energy alignment;
R=8.31446261815324 converts the ATcT energy to the common potential.
No molecular-only formation-enthalpy shift or second NASA alignment is added.
The standard pressure remains 1 bar; masses, electronic degeneracies,
isotope conventions and symmetry factors are unchanged. No partition sum
or additional spin division enters this energy-only update.

The official `ATcT 1.228 index
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
and displayed C/S/CS2 values were rechecked against the pinned source
review on 2026-11-05. ``sources.json`` distinguishes that browser inspection
from a raw-byte archive and retains the prior primary-source access limits.
ATcT version differences are correlated evaluated central values, not
independent measurements. The molecular marginal uncertainty is not a
balanced reaction uncertainty; full network covariance is not available.
The small version change does not establish a statistically significant
error in the older model. Preference for the newer energy is based on
its evaluated thermochemical provenance and consistent balanced use,
not on the magnitude of the simulated response.

The prior source review found a caloric discrepancy between TIPS2025 and
NASA at high temperatures despite small partition-function differences.
The 2098 K crossover and high-temperature Cp difference remain reasons to
retain the current thermal function pending a complete replacement assessment.
TIPS2025, DOI 10.1016/j.jqsrt.2025.109568, is not adopted here.

Numerical substitution
----------------------

The fixed 537-species ion-inclusive network spans 1500, 2000, 2500, 3000
and 4000 K; 1e-6, 1e-4, 1e-2, 1 and 100 bar; and C/O=0.5888436554,
1 and 1.5. Only the ``C1S2`` potential changes. The 75 baseline endpoints
are reused from ``curated_cos_adoption/20261105`` after checking species,
elements, formula matrix, budgets and every potential against the named
factory. **All 75 candidate endpoints are fresh cold solves.** Three
selected conditions have both endpoints freshly solved at tighter tolerance
(six solves); a five-node finite interpolation path and four standalone
FastChem comparisons are also new. No neutral network was evaluated here.

The formation-K multiplier is 1.06104490 at 1500 K and
1.02246903 at 4000 K. Actual mole-fraction ratios span
**1.02246897--1.06104490**, including elemental feedback.
The largest CS2 sulfur-inventory change occurs at 3000 K,
1 bar and C/O=1.5:

.. list-table:: Finite response at the highlighted condition
   :header-rows: 1

   * - Quantity
     - Current
     - Candidate
   * - CS2 mole fraction
     - 6.1636054451e-11
     - 6.3489448740e-11
   * - Fraction of total sulfur in CS2
     - 5.8452209365e-06
     - 6.0209865529e-06

CS2 remains a minor sulfur reservoir on this hot grid. At the highlighted
case its sulfur share changes by 1.75766e-7 of the total sulfur inventory
(0.0000175766 percentage points); both sulfur atoms are counted. The coupled
CS change is -1.14627e-12 in mole fraction. Smaller background changes near
the tighter-control variation should not be interpreted as resolved effects.

All endpoints and controls converged. Maximum relative H/C/O/S balance
residual is 2.964e-12; the maximum over all elements is
1.459e-06, dominated by a small inventory in the reused baseline.
Charge residual is below 2.300e-14, and formation mass-action residual is
below 1.65e-12 in natural-log units. Tighter cold solves change CS2 log
mole fractions by at most 2.203e-13; maximum absolute mole-fraction variation
across all species is 3.378e-14. All-trace log variation is 1.458e-5,
while species reaching x>=1e-12 vary by at most 1.458e-6 in log fraction.
Independent FastChem agreement is within 3.375e-08 dex across every species
at the two selected conditions, for both endpoint models.

Raw and element-feasibility-corrected same-model Gibbs exchanges are
archived. Both correct signs exceed a displayed floating-point comparison
scale in 34/75 conditions, including the highlighted case. The remaining
conditions are explicitly unresolved at that scale. This is a diagnostic,
not a rigorous error bound, and comparing the two models' absolute Gibbs
values would not establish which thermochemistry is physically more accurate.
The finite path and mass-action response provide separate numerical checks.

Evidence and decision scope
---------------------------

The continuous candidate supports scalar/JIT/vectorized evaluation only
within 1500--4000 K. Dense replay and caloric derivatives verify the literal
NASA representation and the single energy correction. The receipts pin the
actual numerical input closure, output hashes and FastChem executable hash.
The archived execution is ``results/cs2-candidate-001``; counts above refer
to that run. Tests replay the stored evidence offline without new solves.
Historical source-review receipts and conclusions remain as originally made.

The new ``mceg/contracts/cs2_candidate.yaml`` supports continuous construction,
finite fixed-network response and the scoped energy preference. Complete
physical accuracy and guaranteed Gibbs resolution at every condition remain
unknown. A separately authorized adoption can use this tested energy update
without waiting for every legacy paper. Full thermal replacement, complete
uncertainty propagation, condensation and planetary validation require their
own evidence; this ideal-gas calculation supplies none of those guarantees.
