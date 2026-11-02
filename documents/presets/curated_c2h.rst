Curated gas with the C2H correction
===================================

``exogibbs.presets.curated_c2h.chemsetup()`` selects **curated-c2h-v1**:
NASA CEA thermal functions for neutral ethynyl with balanced ATcT 1.172
atomization energy. Only C2H changes from :doc:`curated`. The frozen
``curated.chemsetup()`` retains its original behavior and evidence snapshots.

Use the corrected preset explicitly::

   import jax
   jax.config.update("jax_enable_x64", True)
   from exogibbs.api.gas import EquilibriumOptions, solve
   from exogibbs.presets import curated_c2h

   setup = curated_c2h.chemsetup()
   result = solve(setup, T=2000., P=0.01,
                  b=setup.element_vector_reference,
                  options=EquilibriumOptions(epsilon_crit=1e-13))

The evaluation interval remains 1500--4000 K, with pressure in bar and
dimensionless formation Kp at 1 bar. Scalar/array temperatures, JAX derivatives,
537 species with ions and ``ions=False`` (421 neutral species) are supported.
Values outside the interval are NaN. MCEG and FastChem are optional tools.

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
A future default-dataset migration is separate from this opt-in correction.

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
The 421-species option shares corrected constants but was not used for the
archived abundance comparison. Whole-atmosphere physical accuracy remains unknown.

Archive: ``tests/data/validation/c2h_equilibrium_impact/20261102``. Replay::

   python mceg/reviews/build_c2h_correction.py --output results/c2h_v1.json
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/c2h_equilibrium_impact.py \
     --output-dir results/c2h-impact-replay

Optional records are in ``mceg/contracts/curated_c2h.yaml`` and the packaged
catalog returned by ``exogibbs.provenance.describe(setup)``.
