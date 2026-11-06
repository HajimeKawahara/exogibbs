Acetylene energy and thermal candidate evaluation
===================================================

Prefer the retained-thermal ATcT 1.228 energy candidate for a separately
reviewed adoption on 1500--4000 K. The evaluated atom-balanced energy has a
clear source basis, and its effect is resolved by both equilibrium solvers.
The NASA-thermal alternative produces an additional response, but this
comparison does not establish superior high-temperature thermal accuracy.
Keep it as a documented alternative. Production remains ``curated-tio2-v1``;
this packet extends the source review without adopting either candidate.

Physical construction and conventions
---------------------------------------

The molecule is neutral singlet acetylene HCCH, not vinylidene. The reaction
is ``2 C(g) + 2 H(g) -> C2H2(g)`` at a 1-bar standard pressure. The original
source evidence is retained in :doc:`c2h2_source_review`.

The first candidate preserves every coefficient of the inherited FastChem
JANAF-labelled formation fit. Its physical source atomization energy is
1627.153 kJ/mol, while the same-version ATcT 1.228 cycle gives 1626.018 kJ/mol.
Only their difference, -1.135 kJ/mol, is added to D0. Thus G and H increase
by 1135 J/mol, S and Cp remain unchanged, and ln K decreases by 1135/(RT).
The fit/source residual is preserved. R times the fitted 1/T coefficient is
not interpreted as physical D0, and the fit is never extrapolated to zero K.
Printed millikJ energies are balanced as integral J/mol before division by R.
No extra isotope, ZPE or electronic-weight correction is introduced.

The second candidate forms the exact NASA9 coefficient combination
``acetylene - 2 C - 2 H`` within its 1000--6000 K source interval, restricted
here to 1500--4000 K. Header Hf298 and H298-H0 values give the physical NASA
D0 of 1625.696849 kJ/mol. The single b1 correction is
``-(D0_ATcT/R_SI - D0_NASA/R_NASA)`` with
R_NASA=8.314510 and R_SI=8.31446261815324 J/mol/K. Its numerical magnitude
is 39.73983211696 K; replacing it with the raw energy difference divided
by SI R would lose the source gas-constant normalization. This removes the
native NASA energy once and inserts the ATcT energy once.

NASA dimensionless molecular and atomic thermal functions are retained;
candidate G/H/S/Cp use SI R. The second-versus-first comparison therefore
includes the atomic thermal reference and removal of the inherited fit
residual, as well as molecular thermal differences. It is not a purely
molecular correction or a physical confidence interval.

The aCeTY four-node partition diagnostic remains in the source packet only.
Its normalized Q/NASA falls to about 0.441 at 4000 K. The lower-state line
completeness criterion does not certify a complete hot thermodynamic sum;
no finite-PF extrapolation or additional electronic manifold is adopted.

Continuous and source validation
----------------------------------

The analytic candidates are checked at 2501 integer temperatures and 251
off-grid temperatures. Independent source expansion and JAX derivatives
reproduce G/H within 1.7e-9 J/mol and S/Cp within 1e-12 J/mol/K. The literal
current fit matches the named current factory within 2.85e-14 in G/(RT).
Tests separately guard the native-R energy replacement, physical D0 balance,
unchanged S/Cp, preserved fit residual, scalar/JIT/vmap agreement, and invalid
inputs. Caloric replay uses dimensional absolute allowances rather than a
near-zero relative comparison; a changed Gibbs result is explicitly rejected.

Numerical differentiability does not remove inherited caloric fit error.
For example, the current fit minus the original JANAF Cp is
-1.373, -0.060, +2.418 and +0.027 J/mol/K at 1500, 2000, 3000 and 4000 K.
The retained-energy candidate preserves those residuals. JANAF already
includes anharmonic and rotation-vibration corrections; calling it a bare
harmonic model would misstate the source.

.. list-table:: Formation K ratios, not abundance ratios
   :header-rows: 1

   * - T (K)
     - Retained ATcT / current
     - NASA ATcT / current
     - NASA ATcT / retained ATcT
   * - 1500
     - 0.91301216
     - 0.87674910
     - 0.96028195
   * - 2000
     - 0.93402268
     - 0.90221430
     - 0.96594475
   * - 3000
     - 0.95551670
     - 0.91543852
     - 0.95805602
   * - 4000
     - 0.96644849
     - 0.89591277
     - 0.92701554

Fixed-network responses and controls
--------------------------------------

Each candidate receives 75 fresh cold starts on each of two fixed networks:
537 species with ions and 421 neutral species. The 150 current endpoints are
reused from the final TiO2 adoption archive after checking its hashes, species,
elements, formula matrix, every potential and every budget against the named
factory. All 300 fresh candidate endpoints converge. Only C2H2's potential
changes; no formula, abundance budget or other species function changes.
The grid spans 1500--4000 K, 1e-6--100 bar and C/O=0.5888436554, 1 and 1.5.

.. list-table:: C2H2 mole-fraction response on the full grids
   :header-rows: 1

   * - Candidate
     - Ion network x/new over x/current
     - Neutral network x/new over x/current
     - Largest absolute target delta x (ions)
   * - Retained thermals + ATcT
     - 0.913012--0.995680
     - 0.913012--0.995680
     - 2.24707e-6
   * - NASA thermals + ATcT
     - 0.876749--0.993457
     - 0.876749--0.993457
     - 3.81757e-6

At the original maximum carbon allocation (1500 K, 1e-4 bar, C/O=1.5), the
ion-network C2H2 share of carbon changes from 30.122595% to 29.953620%
(retained) or 29.877444% (NASA). The largest *change* in carbon allocation
occurs elsewhere: 2000 K, 1e-6 bar, C/O=1.5, where it changes from 11.578609%
to 11.050650% or 10.787325%. These are respectively -0.527959 and -0.791284
percentage points, not relative percentages.

The largest absolute change among all mole fractions is CH4 at 2000 K,
1 bar, C/O=1.5: +3.11130e-6 or +4.71015e-6. The packet records every
carbon-bearing species. Major redistributed carbon also enters C3H, HCN,
atomic C and CH3. At the largest-allocation-change condition, C3H's carbon
share rises from 11.666934% to 12.050825% or 12.242688%. This finite-network
response is conditional on the unchanged C3H and other background sources;
it does not independently validate their thermochemistry. Atomic carbon and
hydrogen redistribution buffer the C2H2 response, so its K multiplier is not
its mixture abundance multiplier.

There are 18 additional tighter cold solves (three conditions, three models,
two networks) and ten five-point interpolation-path solves (one path per
network for the retained-energy candidate). Tighter cases include maximum
current C2H2 carbon allocation, maximum relative response and a 4000 K
condition. The largest target change with tighter tolerance is 3.98e-13
in ln x; the largest absolute change across all species is 3.49e-14.
A neutral trace-species ln x change reaches 8.83e-6, so numerical claims
are not extended to arbitrary tiny relative effects.

Standalone FastChem 4.0.3 independently evaluates two ion-network conditions
for current and both candidates: six actual comparisons, not six distinct
grid points. The same h, formula matrix and budget are exported. Its maximum
mole-fraction difference is 1.39e-7 dex. No neutral FastChem comparison is
claimed. Primary and tighter tolerances are 5e-14 and 2e-14, respectively.

All-element residual reaches 1.40e-6 in a trace inventory; H/C/O close within
2.64e-13, charge within 1.77e-14 and formation equations within 1.37e-12 in
log units. Raw and feasibility-corrected Gibbs differences are archived.
Corrected same-model ordering meets the existing 5e-14 dimensionless
absolute test; individual differences near 1e-18 are below that resolution
and do not establish universal strict Gibbs ordering.

Decision and reproducibility
------------------------------

The evaluated energy update is scientifically preferable to retaining the
older JANAF anchor, within the declared domain and unchanged thermal model.
Its abundance effect is resolved, including redistribution among important
carbon carriers. Neither being closer to NASA nor a larger response is a
reason to prefer the NASA thermal alternative. Its differing calorics and
atomic reference are useful controls, but the current evidence does not
establish more accurate full-range molecular thermals. A later thermal
revision should compare complete G/H/S/Cp, source corrections and state
coverage independently of the now-balanced energy.

Complete physical uncertainty, correlated ATcT reaction uncertainty,
application-specific planetary accuracy and tiny Gibbs-difference resolution
remain unknown. These limits do not prevent recording the completed finite
candidate calculation. No adoption or merge is performed by this packet.

Run ``python mceg/reviews/c2h2_candidate.py --build`` for the analytic checks.
Fresh substitution is separately invoked through
``python mceg/reviews/c2h2_equilibrium_impact.py --output-dir <new-directory>``
with ``--fastchem-executable`` for the independent comparisons. Offline tests
are ``tests/model_evidence/c2h2_candidate_test.py`` and
``tests/model_evidence/c2h2_equilibrium_impact_test.py``. Original source
packet bytes remain unchanged. Input hashes use the existing historical-source
resolver; candidate and execution outputs are pinned directly.
