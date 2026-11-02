Curated gas with the C2H correction
===================================

``exogibbs.presets.curated.chemsetup()`` now selects **curated-c2h-v1**:
NASA CEA thermal functions for neutral ethynyl with balanced ATcT 1.172
atomization energy. ``curated_c2h.chemsetup`` remains an explicit name for the
same factory. Only C2H changes from the original **curated-gas-v1**, retained
byte-for-byte as ``curated_v1.chemsetup`` for historical reproduction.

Use the corrected default or explicitly reproduce the original::

   import jax
   jax.config.update("jax_enable_x64", True)
   from exogibbs.api.gas import EquilibriumOptions, solve
   from exogibbs.presets import curated, curated_v1

   setup = curated.chemsetup()       # Reviewed C2H correction
   legacy = curated_v1.chemsetup()   # Original constants and dataset identity
   result = solve(setup, T=2000., P=0.01,
                  b=setup.element_vector_reference,
                  options=EquilibriumOptions(epsilon_crit=1e-13))

The evaluation interval remains 1500--4000 K, with pressure in bar and
dimensionless formation Kp at 1 bar. Scalar/array temperatures, JAX derivatives,
537 species with ions and ``ions=False`` (421 neutral species) are supported.
Values outside the interval are NaN. MCEG and FastChem are optional tools.

Why the correction is now the default
--------------------------------------

The scientific source comparison favors NASA/ATcT over the inherited fit.
The old JANAF C2H evaluation dates to March 1967 and uses a formation enthalpy
at 298.15 K of 476.976 kJ/mol, versus ATcT 1.172's 567.85 kJ/mol: a difference
of about **90.9 kJ/mol**. In addition, aligning reaction energies does not
remove the thermal-function discrepancy. The inspected spectroscopy identifies
outdated estimates for the bending mode and low electronic state in the old
evaluation. Adopting NASA thermal functions and a balanced ATcT reaction energy
therefore addresses both identified problems.

This is a **conditional scientific preference**, not a claim that all NASA
thermal inputs have been reconstructed or that high-temperature uncertainty is
quantified. Those limits remain explicit below. They do not make the superseded
JANAF estimate the preferable default.

The earlier opt-in policy prioritized compatibility while the substitution was
investigated. It is superseded by this explicit default migration. The numerical
correction and its ``curated-c2h-v1`` identity are unchanged; the old
``curated-gas-v1`` identity is never rebound to new constants. Users reproducing
old calculations must import ``curated_v1``. Historical comparison scripts now
do so explicitly, and their original source bytes and receipts remain available.

Differences between the presets
-------------------------------

.. list-table:: Implementation and evidence differences
   :header-rows: 1

   * - Item
     - ``curated_v1.chemsetup()``
     - ``curated.chemsetup()`` (also ``curated_c2h``)
   * - Dataset identity
     - ``curated-gas-v1``
     - ``curated-c2h-v1``, with a pinned reference to its base
   * - C2H thermal functions
     - Inherited FastChem fit to JANAF, whose C2H evaluation dates to 1967
     - NASA2002 CCH/C/H reaction coefficients
   * - C2H energy anchor
     - Inherited JANAF-based fit
     - Balanced ATcT 1.172 atomization energy, 1075.066 kJ/mol
   * - C2H review status
     - ``inherited_unreviewed``
     - ``prefer_conditionally``
   * - Other species' formation functions
     - Original curated choices
     - Identical to the original curated choices
   * - Species, solver and reference element budget
     - 537 species with ions; 421 without ions
     - Identical network, solver and budget for the same ``ions`` choice
   * - Temperature and standard pressure
     - 1500--4000 K; 1 bar
     - 1500--4000 K; 1 bar
   * - Abundance evidence
     - Historical curated and HCN comparison records
     - 75-pair substitution study and 75 new default solves per network; old HCN results are not rebound

Relative to the old curated fit, **both C2H thermal functions and its energy
anchor change**. The energy-only shift described below is the second step,
from raw NASA coefficients to the NASA/ATcT candidate. All other entries of
the formation vector h(T) are unchanged; their equilibrium abundances can
still change because the molecules share conserved elements.

For C2H, K(corrected)/K(legacy) is 0.001898625 at 1500 K and 0.08910022 at
3000 K (about 1/527 and 1/11.2). These ratios compare formation constants,
not mole fractions. The abundance table below shows the result of solving
the coupled equilibrium equations.

Source choice
-------------

The :doc:`source investigation <../mceg/c2h_source_review>` identified the old
JANAF energy estimate and a separate thermal-function discrepancy. This update
uses the inspected NASA2002 CCH, C and H records. Their 1000--6000 K intervals
cover the narrower preset domain. NASA includes later spectroscopic/vibronic
evaluations; no new partition function is inferred from a few frequencies.

The adopted zero-temperature atomization energy is

.. math::

   D_0 = 2\Delta_f H^0_0(\mathrm{C}) + \Delta_f H^0_0(\mathrm{H})
         - \Delta_f H^0_0(\mathrm{CCH}) = 1075.066\ \mathrm{kJ\,mol^{-1}}.

All three energies use ATcT 1.172. NASA's assigned H(0) is Hf(298.15 K) minus
H(298.15)-H(0), not the enthalpy coefficient alone. The NASA energy uses
R=8.314510 J/mol/K; modern energy uses R=8.31446261815324 J/mol/K, as in existing
curated adoption. Only the reaction enthalpy integration constant is shifted,
by +152.225866962 K; entropy and Cp retain NASA's model. No second correction
is applied to HCN or another species.

This is a **conditional source choice**. High-temperature thermal uncertainty
is not quantified; original-paper access and vibronic-state completeness gaps
remain. ATcT energy uncertainties are correlated, not independent errors.
The default migration retains this candidate's data identity and source limits.

Equilibrium impact
------------------

The fixed-network study covers 1500, 2000, 2500, 3000, 4000 K; pressures
10^-6, 10^-4, 0.01, 1, 100 bar; and C/O=0.5888436553555908 (Asplund 2021),
1.0, 1.5: **75 paired conditions**. Only C/H changes between budgets; all other
element/H ratios are preserved before normalization. Within each pair, h(C2H)
alone changes and every species is re-equilibrated.

The strongest HCN response occurs at C/O=1.5, 2000 K and 10^-6 bar:

.. list-table:: Mole fractions at the highlighted gas-only condition
   :header-rows: 1

   * - Species
     - Original
     - C2H correction
     - Ratio
   * - C2H
     - 8.18082e-5
     - 2.67443e-6
     - 0.03269
   * - C2H2
     - 1.82916e-5
     - 4.67992e-5
     - 2.5585
   * - HCN
     - 1.61823e-5
     - 2.38136e-5
     - 1.4716
   * - CH4
     - 4.66658e-12
     - 7.46456e-12
     - 1.5996

C2H's share of total carbon falls from **20.213% to 0.661%**. Carbon moves
to other carriers. CH4's relative change accompanies a tiny mole fraction;
HCN and acetylene have larger absolute changes. CO remains nearly unchanged.
The source K ratio is therefore not an abundance ratio.

At the Asplund C/O ratio, all tracked molecules other than C2H change by less
than 0.026% across this grid. The carbon-rich result cannot be generalized
to solar composition or arbitrary atmospheric conditions.

Validation and scope
--------------------

All 150 endpoint solves and 150 tighter cold solves pass convergence,
element/charge conservation and all-species mass-action checks. Tightening
1e-13 to 5e-14 changes any mole fraction by at most 9.2e-14; the largest
logarithmic change over all species is 2.4e-5. Finite log amounts are retained
for trace species. A 2e-15 tolerance trial failed convergence and is recorded
separately, not counted as a pass.

Five finite substitution fractions confirm the nonlinear response at the
highlighted point. Standalone FastChem with identical constants and budget
agrees within 3.7e-7 dex for every species in both endpoint models. This is
numerical evidence, not a physical uncertainty estimate.

These are ideal, gas-only calculations. Condensation, rainout, high-pressure
nonideality, photochemistry, quenching and spectra are outside the evidence.
The original paired substitution study covers the 537-species network. The
additional default-adoption run below covers both networks. Whole-atmosphere
physical accuracy remains unknown.

Archive: ``tests/data/validation/c2h_equilibrium_impact/20261102``. Replay::

   python mceg/reviews/build_c2h_correction.py --output results/c2h_v1.json
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/c2h_equilibrium_impact.py \
     --output-dir results/c2h-impact-replay

Optional records are in ``mceg/contracts/curated_c2h.yaml`` and the packaged
catalog returned by ``exogibbs.provenance.describe(setup)``.

Default-adoption verification
------------------------------

A separate execution constructs the public ``curated`` default and cold-solves
75 conditions for **each** supported network: the same five temperatures, five
pressures and three C/O settings, with tolerance 5e-14 in float64. All 150 solves
pass. Maximum relative elemental residuals are 1.38e-6 with ions and 1.41e-6
without ions; mass-action residuals are below 1.2e-12 in natural log units.
The maximum relative charge residual is 1.64e-14. All 75 ion-inclusive log mole
fractions exactly reproduce the archived corrected candidate in this environment.
No new independent-solver comparison is claimed for neutral mode.

Both modes preserve the original formula matrix, reference budget and all
non-C2H formation functions. The archive retains full log amounts, log mole
fractions, thermochemistry, budgets, convergence flags and source/output hashes:
``tests/data/validation/curated_default_adoption/20261102``. Replay into a new
directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_default_adoption.py \
     --output-dir results/curated-default-replay

MCEG's ``default_adoption_review`` supports this numerical claim for both modes.
``conditional_selection_review`` and ``whole_mixture_accuracy_review`` remain
``unknown`` for complete primary/thermal support and whole-mixture physical
accuracy, respectively. The default factory's ion-inclusive catalog entry is
``exogibbs.curated_c2h.v1``; the earlier HCN study remains attached exclusively
to ``exogibbs.curated_gas.v1``. Neutral-mode evidence is in the contract directly;
the packaged lookup still restricts its entry to ``ions=True``.
