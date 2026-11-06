OH NASA/ATcT candidate and finite substitution
==============================================

This 2026-11-05 extension of :doc:`oh_source_review` constructs the proposed
NASA thermal + ATcT energy candidate and evaluates changing OH alone in
the current gas network. The source investigation remains a dated record.
The candidate uses an exact combination of existing NASA coefficients;
no additional fit or production curated change is needed for this study.

Why both the thermal functions and energy change
------------------------------------------------

The inherited ``H1O1`` column comes from the JANAF1998/FastChem fit.
The inspected `IUPAC evaluation (2005)
<https://doi.org/10.1063/1.1724828>`_, OH section pp. 633--638, identifies
incorrect electronic-degeneracy treatment and caloric increments in JANAF,
including the 1998 edition. Changing only the dissociation energy would
retain those defective thermal functions. The NASA/Gurvich thermal route
addresses that identified problem. IUPAC adopts Gurvich thermal functions,
so the agreement of these evaluations is not independent validation.

The reaction is H(g) + O(g) -> OH(g), at standard pressure 1 bar. The
balanced ATcT 1.172 ground-relative atomization energy is
``216.034 + 246.844 - 37.274 = 425.604 kJ/mol``. Atomic H and O come
from the same evaluation as OH. The NASA header reconstruction gives
425.7859 kJ/mol. Since NASA's native R is 8.31451 J/mol/K and the ATcT
anchor uses modern SI R, the change in D0/R is

.. math::

   \Delta_D = 425604/R_{\rm SI} - 425785.9/R_{\rm NASA}
            = -21.5857097163\ {\rm K},
   \qquad \ln K_{\rm candidate} = \ln K_{\rm NASA} + \Delta_D/T.

Using the raw energy difference divided by modern R would silently change
the NASA thermal convention. The molecular OH uncertainty is not the
uncertainty of this balanced reaction; no independent-error propagation
or correlated ATcT uncertainty band is claimed.

Continuous construction and thermal control
--------------------------------------------

All temperatures in 1500--4000 K lie in the same 1000--6000 K NASA segment.
Subtracting the H and O nine-coefficient arrays from the OH array and
applying the constant energy shift therefore gives an exact continuous
reaction function. The NASA atomic thermal functions, including O
fine-structure populations, are retained. Its ground energy refers to
the lowest fine-structure level, not a term barycentre. Replacing its
thermal function by a fixed degeneracy would define a different reaction.
The source dimensionless thermal functions remain unchanged. Their H0
offset is removed with native NASA R; reported ground-relative H, G, S
and Cp are expressed with common modern SI R.

The scalar interfaces reject nonfinite or out-of-domain temperatures.
The traced JAX interface accepts scalar or array temperatures, supports
JIT/vectorization/differentiation, and returns NaN outside 1500--4000 K.
H and Cp follow the first and second derivatives of the same Gibbs
function. This avoids independently fitted caloric curves.

At all 2501 integer temperatures from 1500 to 4000 K, the maximum
dimensionless formation-potential reconstruction error is 2.31e-14.
Independent molecular G/H/S/Cp expressions agree within 4.66e-10 J/mol,
8.74e-11 J/mol, 5.69e-14 J/mol/K and 1.43e-14 J/mol/K, respectively.
JAX first/second derivatives reproduce reaction H and Cp within
4.66e-10 J/mol and 4.45e-13 J/mol/K. These are floating-point reproduction
checks on an exact construction, not fitted or physical uncertainty bounds.

The control uses the archived 10606 MYTHOS states described by
`Mitev et al. (2025) <https://doi.org/10.1093/mnras/stae2803>`_. Both
routes retain the same balanced energy, NASA atoms and 1 bar convention.
The control sums the supplied states directly, with no interpolation fit.
For the state sum, divide the tabulated ``2(2J+1)`` weights by the nuclear
spin factor two once; keep the electronic doublet, fine structure and
parity levels. The internal states represent :sup:`16` O :sup:`1` H,
with aggregate translation mass 17.00734 u. This is an explicit
representative-isotope approximation, not a complete isotope reaction.

.. list-table:: Pointwise formation-constant ratio to current curated
   :header-rows: 1

   * - T, K
     - NASA thermal + ATcT
     - MYTHOS thermal + ATcT
   * - 1500
     - 1.12914110
     - 1.12922346
   * - 3000
     - 1.06582346
     - 1.06537479
   * - 4000
     - 1.05448246
     - 1.05345412

These are equilibrium-constant ratios; composition constraints can change
the corresponding abundance response. MYTHOS/NASA Q differs by less than
0.1% at all 2501 comparison temperatures: its ratio ranges from
0.9990247892 at 4000 K to 1.0000729348 at 1500 K. This supports using the simpler continuous
NASA route as the primary candidate, without claiming that it is more
physically accurate than MYTHOS.

Finite substitution
--------------------

The finite calculation retains the ``curated-h2o-v1`` temperature,
pressure and C/O grid, elemental inventories and every species except OH.
There are 75 new cold endpoints on each of the 537-species ion-inclusive
and 421-species neutral networks. The 150 baseline endpoints are reused
only after checking every potential, species and inventory against the
current model. The grid has five temperatures (1500, 2000, 2500, 3000 and
4000 K), five pressures from 1e-6 to 100 bar, and three C/O ratios.

OH's candidate/current mole-fraction ratio ranges from 1.038811 to
1.129141 across either network. Its largest absolute response occurs at
4000 K, 1 bar and C/O=0.5888436553555908 (case 23). In the ion-inclusive
calculation, x(OH) rises from 6.0911629e-5 to 6.3275666e-5: **+3.8811%**.
Oxygen in OH rises from 11.2390648% to 11.6752507% of the inventory,
or **+0.4361859 percentage points**. The neutral calculation gives almost
the same response: x(OH) rises from 6.0904914e-5 to 6.3268737e-5.
The formation-K increase at 4000 K is 5.4482%; conservation and competition
for oxygen explain why the actual OH increase is smaller.

.. list-table:: Largest absolute response for each oxygen species, ion-inclusive grid
   :header-rows: 1

   * - Species
     - T, K
     - P, bar
     - Current mole fraction
     - Candidate mole fraction
     - Relative change
   * - OH
     - 4000
     - 1
     - 6.091163e-5
     - 6.327567e-5
     - +3.8811%
   * - H2O
     - 4000
     - 100
     - 2.710809e-4
     - 2.697096e-4
     - -0.505855%
   * - O
     - 4000
     - 1
     - 1.220448e-4
     - 1.202310e-4
     - -1.48616%
   * - O2
     - 2000
     - 1e-6
     - 1.360813e-8
     - 1.320803e-8
     - -2.94015%

All rows have C/O=0.5888436553555908. Each row selects that species'
largest absolute change, so they are not all the same equilibrium case.
Only OH's potential changes; the other species respond through the shared
elemental budgets. The O2 percentage change describes a trace abundance.

Six tighter MYTHOS controls, at three selected cases per network, retain
the same ATcT energy and atomic reference. Their OH ratios to NASA/ATcT
range from 0.99902479 to 1.00007293. The largest absolute difference in
any mole fraction is 4.3973e-8, in OH at case 23; there MYTHOS lowers OH
by 0.069495% relative to NASA/ATcT. This is a smaller thermal-source
response than the JANAF-to-candidate change, measured on the selected
controls, not a full-grid MYTHOS replacement or a physical uncertainty band.

Numerical checks and tolerances
--------------------------------

All 150 new endpoints converge. Mass-action residuals are below 1.85e-12
in log units, ion-network charge closure is below 4.96e-14, and H/C/O
inventory residuals are below 9.04e-13 in both networks. The largest
relative elemental residual, 1.465e-6, belongs to trace fluorine and
does not set the precision of the oxygen response.

Twelve tighter cold solves cover both models at three cases in each
network. The executed primary tolerance is 5e-14 and the tighter tolerance
2e-14. OH log fractions remain stable within 2.85e-13 and all absolute
mole fractions within 3.40e-14. Among species with x>=1e-12 the maximum
log-fraction change is 1.463e-6, for F at x=2.318e-8. The unrestricted
maximum is 1.463e-5 in extremely scarce F10S2. These separate metrics
avoid assigning a trace-species relative residual to the OH response.
A preliminary run at tolerance 1e-13 exceeded the 2e-6 visible-species
stability criterion (2.5713e-6); it was rejected and the endpoints rerun
at 5e-14. The archived execution receipt describes only the completed run.

Five potential-interpolation points at case 23 in each network conserve
the inventories and increase OH monotonically. Same-model Gibbs exchange
checks subtract first-order inventory-residual terms before testing the
minima; the residual ordering error is below 2e-18. Absolute Gibbs values
of different thermochemical models are not used to rank their physical
accuracy. Independent FastChem 4.0.3 runs for both models at two ion-network
cases agree within 1.19e-8 dex across all species. These tests establish
the numerical response to the specified change, not atmospheric accuracy.

Scientific scope and next decision
-----------------------------------

**Recommend NASA thermal functions plus balanced ATcT 1.172 for a scoped
OH production update on 1500--4000 K.** The known JANAF thermal defect
gives a scientific reason for replacement. The exact continuous model
and conserved finite response now complete the source review's
``candidate_and_substitution`` follow-up. MYTHOS provides a smaller
controlled thermal alternative. This supports a bounded adoption decision
without claiming NASA is intrinsically more accurate than MYTHOS or that
the comparison establishes full thermochemical accuracy.

The source study records quasi-bound levels and excluded continuum in
MYTHOS. At 4000 K, removing every supplied level above the ATcT ground-atom
threshold reduces Q by 33.56 ppm and Cp by 0.03815 J/mol/K. That defined
sensitivity is neither a complete continuum treatment nor an omitted-state
error bound. The isotope approximation, ATcT reaction covariance and
historical Gurvich source access remain explicit limitations. They do not
require collecting every historical paper before reviewing this bounded
candidate. An application demanding greater precision must justify the
corresponding physical model and uncertainty treatment.

Reproduction and MCEG
----------------------

The candidate packet is ``tests/data/validation/oh_candidate/20261105``.
It reuses frozen NASA records, same-version ATcT values and the compact
MYTHOS state archive from the dated source review. Offline evaluation
requires no paper download or external MCEG installation.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/oh_candidate.py
   PYTHONPATH=src pytest tests/model_evidence/oh_candidate_test.py tests/model_evidence/oh_equilibrium_impact_test.py

The separate finite-substitution README gives the optional full-rerun
command. ``mceg/contracts/oh_candidate.yaml`` separates continuous
thermodynamics, finite numerical response and scoped scientific preference
from complete physical accuracy, which remains unknown. Production
adoption is a subsequent decision; this study does not change curated.

MCEG structure and selected-file/artifact hashes pass. The three scoped
claims have recorded support; physical accuracy is recorded as unknown.
The finite numerical assessment's source audit passes. Source-audit status
remains unknown for assessments using the partially inspected historical
source scope or the secondary scientific reasoning; that source-trace
classification is distinct from both the numerical checks and physical
accuracy, and does not impose a new paper-acquisition gate.
