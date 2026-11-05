KH spectroscopic-energy candidate and finite substitution
=========================================================

The spectroscopic-energy candidate is the preferred bounded next model for
KH: lower the balanced zero-temperature dissociation energy while retaining
the actual current thermal fit. The calculation changes no production data.
It completes the energy-candidate and finite-substitution follow-ups of
:doc:`kh_source_review`, including both ion-inclusive and neutral networks.
The separate question of complete physical accuracy remains unknown.

Scientific choice and continuous construction
----------------------------------------------

The inspected publisher abstract of
`Zemke et al. (1988) <https://doi.org/10.1016/0009-2614(88)87016-7>`_
gives D0=14282.6 cm-1, or 170857.838639 J/mol with exact SI conversion
constants. This is 9532.161361 J/mol below the balanced JANAF anchor
180390 J/mol. BC2016's 1.7708 eV agrees within 2 J/mol; Khan et al. (2026)
independently use the associated De in their potential model. De and D0
are different quantities: the abstract's +/-0.6 cm-1 belongs to De, and
is not a complete uncertainty for D0. Source agreement is supporting
consistency, not independent measurements with a combined error bar.

For H(g)+K(g)->KH(g), with the one-bar ideal-gas standard state, the exact
candidate is

.. math::

   \ln K_{\rm candidate}(T) = \ln K_{\rm current}(T)
     + \frac{170857.8386391254-180390}{RT}.

Only the existing five-coefficient fit's 1/T coefficient changes, by
-1146.455495520 K. No new fit or interpolation is introduced. The reaction
enthalpy increases by 9532.161361 J/mol; entropy and heat capacity are
unchanged. The original molecular/atomic thermal convention remains intact.
The candidate's domain is 1500-4000 K, with invalid scalar input rejected
and invalid traced JAX input returning NaN. Dense independent reconstruction
at 2501 temperatures and analytic/JAX H/S/Cp derivatives are checked.

This is a proportionate scientific improvement over the larger inherited
JANAF energy. The source review found no corresponding need to replace the
entire thermal representation: source thermal differences are smaller than
the energy effect over this domain, and a full Khan state-sum adoption
would introduce additional bound/quasi-bound and state-selection questions.
The candidate therefore makes the supported energy change directly. It
neither applies the raw BC partition function's factor two to JANAF nor
uses De in place of D0. The 1988 full text would refine a precision uncertainty
claim; acquiring it is not a new gate for this bounded candidate study.

.. list-table:: Formation-constant contrasts
   :header-rows: 1

   * - T (K)
     - Candidate/current K
     - NASA thermal-reference control/candidate K
   * - 1500
     - 0.465658
     - 1.008168
   * - 2000
     - 0.563703
     - 1.018741
   * - 2500
     - 0.632179
     - 1.031264
   * - 3000
     - 0.682391
     - 1.044373
   * - 4000
     - 0.750802
     - 1.064023

The NASA control uses KH-H-K coefficients from the single 1000-6000 K
interval and the identical spectroscopic D0. Its energy correction is
D0_target/R_SI minus D0_NASA/R_NASA, with D0_NASA=177999.972 J/mol,
R_NASA=8.314510 and R_SI=8.31446261815324. This preserves NASA's native
source convention before converting the reaction. Both the molecular and
atomic thermal functions differ from the primary model: the comparison
is a thermal-reference sensitivity, not a purely molecular correction or
physical uncertainty interval. The original source review retains the
separate molecular and atomic diagnostics.

Finite abundance and potassium redistribution
----------------------------------------------

The baseline is the actual current ``curated-oh-v1``. Before reusing its
archived endpoints, the run verifies all species, elements, formula entries,
75 full chemical-potential vectors and every elemental budget against that
current setup. Each network contains five temperatures
(1500/2000/2500/3000/4000 K), five pressures (1e-6 through 100 bar) and three
C/O ratios (0.5888436554, 1 and 1.5). The 537-species ion-inclusive and
421-species neutral networks each receive 75 fresh candidate solves. Only
the KH potential changes; inventories and all other species are fixed.

.. list-table:: Finite response over each 75-case network
   :header-rows: 1

   * - Network
     - KH mole-fraction ratio, candidate/current
     - Largest loss of K inventory from KH
     - Condition of largest K redistribution
   * - Ion-inclusive
     - 0.465658-0.753321
     - 5.10150% -> 3.28852% (-1.81298 percentage points)
     - 2500 K, 100 bar, C/O=1
   * - Neutral
     - 0.465659-0.768271
     - 7.15367% -> 4.99509% (-2.15857 percentage points)
     - 3000 K, 100 bar, C/O=1

At the ion-inclusive highlighted case, KH's mole fraction changes from
1.02757e-8 to 6.62393e-9. Atomic K's inventory share grows from 87.03169%
to 88.74425%, K+ from 7.69243% to 7.78942%, and KOH from 0.055236%
to 0.056323%. At the neutral highlighted case, atomic K grows from
92.77595% to 94.93289%. The maximum *initial* neutral KH allocation is
larger still, 9.12465% at 4000 K, 100 bar, C/O=1; it becomes 7.01021%.
That case differs from the maximum redistribution case because the energy
change has a stronger effect at lower temperature.

The numerical packet also records Na, Na+, NaH, NaOH, H, H2 and electrons.
Across ion-inclusive cases, the largest absolute Na and Na+ responses are
+2.50e-10 and -2.64e-10 in mole fraction; electrons change by at most
+5.52e-10. These small coupled changes follow from equilibrium and charge
balance, despite only the KH thermodynamic column changing. A species
abundance ratio need not equal its formation-constant ratio because atomic
K and the total gas abundance readjust.

Six selected fresh NASA-control solves (three per network, including each
highlighted case) increase KH by 0.817-6.402% relative to the primary
candidate. Their maximum absolute change in any mole fraction is
4.21e-10. This is a limited set of thermal-reference controls, not another
150-case grid or a bound on unexamined physical effects.

Numerical evidence and limits
------------------------------

The primary tolerance is 5e-14. All 150 fresh endpoints converge. The
largest relative K inventory residual is 4.23e-10; H/C/O residuals are
below 8.85e-13, charge closure below 4.96e-14 and mass-action residuals
below 1.85e-12 in natural logarithm. Across *all* elements, the largest
relative residual is 1.48e-6, reflecting trace inventories rather than
K/H/C/O accuracy; it is not reported as 1e-12 all-element accuracy.

Twelve fresh cold controls at tolerance 2e-14 reproduce KH log mole
fractions within 9.24e-14. The maximum difference across all trace species
is 1.47e-5 in ln(x), or 1.47e-6 for species with x>=1e-12; the maximum
absolute mole-fraction difference is 3.41e-14. Ten intermediate
lambda=0/0.25/0.5/0.75/1 solves confirm monotonic KH response at the two
highlighted cases. Same-model Gibbs exchange checks subtract the first-order
inventory-feasibility term before applying a 5e-14 ordering tolerance;
raw and corrected quantities are both archived. These checks resolve the
reported KH effect without claiming unlimited trace-species precision.

Four independent FastChem v4.0.3 comparisons (current and candidate at two
ion-inclusive conditions) agree with the archived compositions within
8.73e-9 dex across all species. The selected temperatures are supported
by the existing node-export helper; this does not certify the entire
continuous domain independently. FastChem validates the same supplied
thermodynamics, not their physical accuracy.

The retained fit's remaining source error, D0 uncertainty, isotope mixture,
high-state/continuum contributions and complete atmospheric accuracy are
not quantified here. The calculations assume fixed ideal-gas networks
without condensation, transport, photochemistry or opacity validation.
The MCEG candidate contract supports the construction, finite response and
scoped model preference while keeping the physical-accuracy claim unknown.
A curated change requires a separate adoption decision and runtime record.

Reproduction and MCEG records
------------------------------

The bounded candidate and finite-run packets are
``tests/data/validation/kh_candidate/20261105`` and
``tests/data/validation/kh_equilibrium_impact/20261105``. They contain the
source hashes, coefficients, arrays, solver diagnostics and actual execution
receipts. Build/replay the candidate with::

   python mceg/reviews/kh_candidate.py --build
   pytest tests/model_evidence/kh_candidate_test.py tests/model_evidence/kh_equilibrium_impact_test.py

Fresh solves require an unused output directory::

   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 4-7 python mceg/reviews/kh_equilibrium_impact.py \
     --output-dir results/kh-candidate-impact-002 \
     --fastchem-executable /tmp/mceg-fastchem-v403-bin

The frozen results replay offline without FastChem. The new contract
``mceg/contracts/kh_candidate.yaml`` completes source follow-ups 1 and 3
and leaves the Khan thermal-selection follow-up explicitly open. The dated
source packet and its assessments remain separate historical evidence.
