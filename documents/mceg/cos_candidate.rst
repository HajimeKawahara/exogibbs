COS JANAF thermal / ATcT energy candidate
=========================================

The energy-only candidate is numerically ready for a separately authorized,
bounded adoption decision. It preserves the actual current JANAF molecular
thermal fit and NASA atomic background, and changes only the balanced
ATcT 1.228 energy. All 150 new equilibrium endpoints converge. COS increases
by 18.38--56.83% in the sampled hot hydrogen-rich gas, with a maximum mole
fraction increase of 1.61912e-10. Production ``curated`` is unchanged.

Construction and scientific choice
-----------------------------------

The reaction is C(g) + O(g) + S(g) -> OCS(g), at a one-bar standard pressure.
The inspected ATcT 1.228 Hf(0) values for C/O/S/OCS are respectively
711.370/246.843/276.85/-141.79 kJ/mol. Using all four values from that same
version gives D0=1376.853 kJ/mol, compared with the current balanced JANAF
1371.241 kJ/mol. The candidate changes h=-ln K by exactly
-5612/(R T), with R=8.31446261815324 J/mol/K. Thus reaction G and H both
decrease by 5612 J/mol; S and Cp are unchanged. The existing background
already contains the old energy anchor, so only this difference is inserted.

Prefer this evaluated, consistently balanced central energy for a scoped
energy update. This preference does not imply that a version change is an
independent measurement or a significance test: correlated ATcT reaction
uncertainty is not available here, and the molecular uncertainty alone would
not be a valid uncertainty on C+O+S->OCS. The detailed access records and
source comparison remain in :doc:`cos_source_review`.

The molecular piecewise quintic coefficients, knots at
1500/2000/3000/4000 K, and NASA atomic thermal coefficients are copied from
the actual current record without fitting. There is no TIPS switch at
3380 K and no OYT8 thermal substitution above its nominal 2000 K scope.
The source review's partition-function, caloric, isotope and high-temperature
limitations remain a separate thermal-model issue. They do not require an
additional inaccessible paper before completing this bounded energy study.

On all 2501 integer temperatures from 1500 to 4000 K, the current-model
replay and exact energy shift agree within 4.8e-14 in h. Scalar/JAX
thermodynamic identities agree within 1.9e-9 J/mol in H and 9.1e-13 J/mol/K
in S/Cp. The internal knots preserve G/S/Cp continuity to rounding error.
Tests include scalar/array/JIT/vmap evaluation, derivatives, both internal
knots, the unrelated TIPS join, and rejection of out-of-domain temperatures.
The equilibrium-constant multiplier decreases from 1.568279 at 1500 K to
1.183815 at 4000 K.

Finite substitution
-------------------

The run compares the actual ``curated-oh-v1`` provider with the COS candidate,
on unchanged 537-species ion-inclusive and 421-species neutral networks.
Before reuse, the OH-adoption baseline was checked against every actual
potential, species, element, formula-matrix entry and C/O-adjusted inventory.
Each network has 75 conditions: T=1500/2000/2500/3000/4000 K,
P=1e-6/1e-4/0.01/1/100 bar, and C/O=0.5888436553555908/1/1.5.
All 150 candidate endpoints are fresh cold solves; only COS changes.

At the largest sulfur-allocation response (1500 K, 1 bar, C/O=0.58884),
ion-inclusive COS x changes from 2.849208e-10 to 4.468320e-10.
The following are fractions of each total elemental inventory, not mole
fractions or percentage points:

.. list-table:: COS allocation at the highlighted condition
   :header-rows: 1

   * - Element
     - Current allocation
     - Candidate allocation
   * - S
     - 1.259153e-5
     - 1.974688e-5
   * - C
     - 5.755439e-7
     - 9.026066e-7
   * - O
     - 3.389054e-7
     - 5.314942e-7

The sulfur shift is 7.15535e-6 of the inventory, or 0.000715535 percentage
points. The neutral-network result agrees at the displayed precision.
The largest absolute response for each species on the ion-inclusive grid is:

.. list-table:: Species response, each at its own largest absolute change
   :header-rows: 1

   * - Species
     - T / K
     - P / bar
     - C/O
     - Delta mole fraction
     - Candidate / current
   * - COS
     - 1500
     - 1
     - 0.58884
     - +1.61911e-10
     - 1.568267
   * - CS
     - 3000
     - 1
     - 1.5
     - -4.34392e-12
     - 0.999999319
   * - CS2
     - 3000
     - 100
     - 1.5
     - -1.78857e-16
     - 0.999990819
   * - CO
     - 1500
     - 1
     - 0.58884
     - -1.60891e-10
     - 0.999999672
   * - H2S
     - 1500
     - 1
     - 0.58884
     - -1.32208e-10
     - 0.999992712
   * - SO2
     - 1500
     - 1e-6
     - 0.58884
     - -5.98870e-17
     - 0.999999452

The archived tables contain the complete ion/neutral extrema and H2O, CO2,
SO, S, S2 and electron responses. Trace COS in this particular hot grid
cannot establish low-temperature atmospheric unimportance. No condensation,
photochemistry, transport or opacity calculation is included.

Numerical checks and evidence scope
------------------------------------

Primary tolerance is 5e-14. Twelve selected cold reruns at 2e-14 cover the
largest COS sulfur-allocation change, largest COS ratio and largest elemental
residual in each network. Ten additional cold solves trace five equally
spaced finite substitutions at the highlighted condition. All converge.
The COS logarithmic mole fraction changes by less than 2.92e-13 under the
tighter solve; absolute changes across all species are below 3.40e-14.
The maximum visible-species logarithmic change is 1.463e-6, while the maximum
across trace species is 1.463e-5; these are separately reported.

Across both endpoints and networks, H/C/O/S relative inventory residuals are
respectively below 3.06e-14 / 2.37e-13 / 8.82e-13 / 2.97e-12.
The all-element maximum is 1.465e-6, driven by a much smaller trace inventory;
this does not replace the explicit C/O/S checks. Formation-equation residuals
are below 1.87e-12 in ln K, and relative charge residual is below 4.96e-14.

The same-model Gibbs exchange compares both compositions using each model's
own potentials, with the first-order inventory residual correction recorded
alongside the raw value. At the highlighted ion condition the corrected
(candidate-composition minus current-composition) differences in G/RT per
normalized inventory are +2.10582e-11 under the current model and
-1.81293e-11 under the candidate. Thus each model prefers its own equilibrium;
absolute Gibbs values under different energy models are not compared as a
selection criterion. Elsewhere the numerical ordering floor is below 9e-19.
The finite paths are conserved, stationary and monotone in COS amount.

Four FastChem 4.0.3 comparisons at two ion-inclusive conditions reproduce
mole fractions within 8.73e-9 dex, with element closure below 4.26e-11,
charge residual below 4.43e-12 and formation residual below 3.92e-10.
This is independent solver verification using the same thermodynamic model,
not physical validation. Neutral-network FastChem checks were not run.

The compact packets under ``tests/data/validation/cos_candidate/20261105``
and ``cos_equilibrium_impact/20261105`` retain source and output hashes,
actual execution commands, diagnostics and amounts. Reproduction uses::

   PYTHONPATH=src python mceg/reviews/cos_candidate.py --build
   PYTHONPATH=src python mceg/reviews/cos_equilibrium_impact.py \
       --output-dir results/cos-candidate-new \
       --fastchem-executable /path/to/fastchem

MCEG supports continuous construction, finite substitution and the scoped
energy preference. Full physical uncertainty and atmospheric abundance
accuracy remain unknown. This completes the source review's energy-candidate
follow-up; production adoption is a separate decision.
