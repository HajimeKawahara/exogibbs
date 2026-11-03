Curated gas with evaluated H2O and CO2 thermochemistry
======================================================

``exogibbs.presets.curated.chemsetup()`` selects **curated-h2o-v1**, also
available as ``curated_h2o.chemsetup``. It adopts the evaluated 2025 H2O
thermal model on **curated-co2-v1**, inheriting the adopted CO2, TiO and C2H
updates. Only H2O's formation function changes relative to that immediate
CO2 base. Species sets, elemental budgets, charge handling and atomic
thermal references remain fixed. The domain is **1500--4000 K** at
standard pressure **1 bar**.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   from exogibbs.api.gas import EquilibriumOptions, solve
   from exogibbs.presets import curated, curated_co2, curated_tio

   setup = curated.chemsetup()       # Evaluated H2O and CO2, plus TiO and C2H
   before_water = curated_co2.chemsetup()
   before_both = curated_tio.chemsetup()
   result = solve(setup, T=2000., P=0.01,
                  b=setup.element_vector_reference,
                  options=EquilibriumOptions(epsilon_crit=5e-14))

All factories support ``ions=False``. The retained networks have 537 species
with ions and 421 without ions. Scalar and array JAX evaluation, JIT and
temperature gradients are supported. Nonfinite and out-of-domain inputs
return NaN. Runtime evaluation uses packaged numerical data and has no
dependency on MCEG, review scripts or original spectroscopic downloads.
Earlier explicit curated modules and their numerical payloads remain
available under their original dataset identities.

Why the water model is adopted
------------------------------

Furtenbacher, Harvey and Csaszar's 2025 thermodynamic evaluation,
`DOI 10.1063/5.0273661 <https://doi.org/10.1063/5.0273661>`_, updates the
spectroscopic inputs and uncertainty treatment for H2(16O). Its official
`thermochemical table <https://doi.org/10.60893/figshare.jpr.29087342.v1>`_
provides evaluated H, S and Cp. The inspected full text and supplements
support replacing the inherited NASA molecular thermal model while
retaining the balanced ATcT 1.172 total atomization energy of
**917.819 kJ/mol** for 2 H(g) + O(g) -> H2O(g). This is total atomization,
not the first dissociation threshold to H + OH.

After matching energy, atomic reference, pressure and translation, the
adopted thermal model gives a formation-constant ratio of **0.98207135**
relative to the previous water model at 4000 K. Its independent discrete
PW2024 alternative gives 0.98202981. The agreement in direction, evaluated
source treatment and verified continuous thermodynamics support the
scoped scientific choice. Source agreement alone is not a physical
uncertainty estimate, and the old energy anchor was not changed merely to
bring it closer to another source.

The preceding :doc:`candidate study <../mceg/h2o_candidate>` executed
150 cold finite substitutions on a fixed 537-species network. Evaluated
water mole fractions were 0.98207135--0.99999970 times the previous model,
and changing between evaluated and discrete scope altered water by at
most about 0.00423%. Tight controls and separate FastChem calculations
resolved those responses. That experiment changed H2O alone on
``curated-tio-v1``; the final production model also includes the CO2
adoption, so those saved endpoints are not relabeled as the combined
production calculation.

The exact adopted representation
--------------------------------

The packaged payload copies the previously reviewed degree-8 Cp/R
polynomial and its analytical H/R and S/R integrals **without refitting**.
Its scaled temperature is (T-2750)/1250, and H/S are anchored at 1500 K.
G=H-TS and its derivatives define all caloric quantities consistently.
This construction avoids amplifying the separately rounded source H/S
values into artificial Cp curvature.

The source fit uses 251 Cp rows at 10 K spacing. Checks against all 2501
integer-temperature rows, including 2250 withheld Cp values and independent
H/S/Q columns, give maximum H, S and Cp differences of 0.5587 J/mol,
0.00002298 J/mol/K and 0.001426 J/mol/K. The continuous minimum Cp is
47.1444 J/mol/K. These are approximation errors relative to the selected
evaluation, not total thermochemical or atmospheric error bars.

The adopted conventions are explicit:

* H2(16O) internal thermodynamics represents element-aggregate H2O, with
  translational mass 18.01528 u and existing NASA H/O thermal functions.
  This is not a complete natural-isotope or isotope-fractionation model.
* The published entropy already removes the proton nuclear-spin product
  four. State-dependent ortho/para weights and their shared para ground
  energy are retained; no second spin or symmetry division or ZPE is added.
* The numerical source enthalpies are J/mol despite their CSV's kJ/mol
  labels. Independent partition-moment closure verifies the interpretation.
* The evaluated model retains the table's thermodynamic scope. Its
  4000 K Cp difference from the discrete sum is about 0.06778 J/mol/K.
  The inherited scattering treatment involves an energy derivative of the
  OH+H phase shift; separately representing OH and H does not alone imply
  a duplicate free-particle contribution. The interaction's chemical or
  cluster assignment in the ideal reacting mixture remains unverified.

These limits are preserved in the packaged source metadata and MCEG.
The small measured evaluated/discrete scope response supports a bounded
adoption without proving the interaction assignment. Complete physical
accuracy remains unknown; retrieving every historical energy paper is
not a condition for this specified thermal improvement.

Combined production verification
---------------------------------

The actual public default was cold-solved at **75 conditions per network,
150 production cases in total**, with convergence required in every case.
The production tolerance is 5e-14. Separately applying the two frozen
candidate evaluators to ``curated_tio`` constructs an independent reference
for the combined update. Its formation potentials match the packaged
model over 2501 temperatures with maximum absolute difference 4.27e-14.
All other potentials, species, formula matrices and budgets are unchanged.

Six additional cold controls use that independently assembled reference
with tolerance 2e-14. CO2/H2O log mole fractions differ by at most 5.33e-14;
the largest absolute mole-fraction difference is 3.11e-15. The maximum
all-species log difference is 1.73e-8, including trace species. An earlier
diagnostic request of 1e-14 at 1500 K and 1e-6 bar reached a float64
stopping-residual floor of 1.54e-14 without declaring convergence; the
recorded controls require and achieve convergence at 2e-14. That numerical
limit is documented rather than counted as a successful 1e-14 solve.

Maximum relative element residuals are below 1.45e-6, governed by trace
elements; H/C/O conservation is checked separately. Relative charge
residuals are below 3.63e-14, and mass-action residuals below 1.37e-12 in
natural log units. Two new ion-inclusive FastChem comparisons agree with
the combined production model within 1.61e-8 dex. These are numerical
implementation checks, not atmospheric observations.

For the ion-inclusive production grid, water mole fractions are
0.98207135--0.99999955 times the ``curated-tio-v1`` values, while CO2 mole
fractions are 1.00065147--1.02354756 times those values. These are the
jointly equilibrated effects of both updates, not the earlier separate
one-species experiments. At 4000 K, 100 bar and reference C/O:

.. list-table:: Joint production response at case 24
   :header-rows: 1

   * - Mole fraction
     - Previous curated-tio-v1
     - Current curated-h2o-v1
   * - H2O
     - 0.0002718623980
     - 0.0002710808608
   * - CO2
     - 0.0000000190192668
     - 0.0000000194671240

The CO2 response includes redistribution of the shared oxygen inventory
after the water update. Reusing the H2O-only archived endpoints as joint
production evidence would miss that coupling. Neutral-mode results are
recorded and verified separately in the same production packet.

Reproduction and model records
-------------------------------

``src/exogibbs/data/curated/h2o_v1.json`` pins the reviewed evaluated model
and its ``curated-co2-v1`` base by SHA256. Setup verifies both identities.
The source-only, candidate and earlier production archives retain their
original bytes and scientific scopes. Rebuild this small payload offline::

   python mceg/reviews/build_h2o_adoption.py --output /tmp/h2o_v1.json

The joint production archive is
``tests/data/validation/curated_h2o_adoption/20261104``. It retains both
networks' complete amounts and potentials, tighter independently assembled
controls, diagnostics, and input/output hashes. Reproduce into a new
directory with an installed FastChem executable::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_h2o_adoption.py \
     --output-dir results/curated-h2o-adoption-replay \
     --fastchem-executable /path/to/fastchem

``mceg/contracts/curated_h2o.yaml`` records the production selection and
numerical validation separately from the source review and candidate
study. The model omits condensation, rainout, transport, irradiation,
nonideality and radiative transfer. Its numerical support and source
selection do not establish a complete planetary abundance error bound.
