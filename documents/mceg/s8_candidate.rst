S8 retained-thermal W4L energy candidate
========================================

A continuous S8 energy-only candidate has been constructed and evaluated in
75 new ion-inclusive equilibrium solves. Its S8 abundance multiplier is
1.9512--5.9449 on this grid, but the largest candidate mole fraction is only
2.41270e-48. The S8 logarithmic response is numerically stable; changes in
other species and the full-system Gibbs ordering are below the numerical
resolution established by the controls. Production curated is unchanged.

Energy anchor and retained model
--------------------------------

The reaction is 8 S(g) -> S8(g), with standard pressure 1 bar and
R=8.31446261815324 J/mol/K. The inspected full
`Karton 2021 article <https://doi.org/10.1016/j.chphi.2021.100047>`_ gives the
direct W4L TAE0=505.67 kcal/mol, or 2115723.28 J/mol. This includes the
zero-point and relativistic corrections and uses ground-state dissociation
products. It is not TAEe, a fragment dissociation energy or a solid-phase
formation energy. The detailed inspection and rounding diagnostics remain
in :doc:`s8_source_review`.

The same-version JANAF S/S8 zero-temperature rows give
D0=8(274735)-104388=2093492 J/mol. Thus the source-energy correction is
+22231.28 J/mol. Comparing molecular formation enthalpies alone would miss
the eight atomic reference changes. The existing formation fit already
contains its original atomic reference, so this difference is applied once::

   ln K_candidate(T) = ln K_current(T) + 22231.28 / (R T)

The actual FastChem4 S8 five-coefficient formation fit is retained, with no
new fitting, symmetry multiplier, isotope change, isomer addition or thermal
switch. Reaction G and H decrease by 22231.28 J/mol; S and Cp are unchanged.
The model is evaluated only at 1500--4000 K and is smooth throughout that
interval. No production fit is extrapolated to zero temperature.

This is a **source-anchor correction**, not a claim that the finite-range
inherited fit has a uniquely recoverable D0 equal to the original table.
The original fit's caloric residuals remain explicit:

.. list-table:: Current fit minus the JANAF molecule-minus-eight-atoms reaction
   :header-rows: 1

   * - T / K
     - H / kJ/mol
     - S / J/mol/K
     - Cp / J/mol/K
   * - 1500
     - -1.37774
     - -0.70523
     - -2.32813
   * - 2000
     - -1.82027
     - -0.98377
     - +0.67695
   * - 3000
     - +1.40252
     - +0.26828
     - +4.70748
   * - 4000
     - +4.33141
     - +1.14896
     - -0.80283

The high-level direct energy supports preferring this scoped energy update;
the residual table prevents calling it an exact reconstruction of W4L plus
all original JANAF thermal functions. No quantified complete reaction
uncertainty is assigned. Missing high-temperature electronic/isomer states,
the inherited fit and thermal evaluation remain separate limitations.

Across all 2501 integer temperatures, the baseline factory replay differs
by at most 2.85e-14 in h=-ln K; the single energy shift closes within
5.27e-14. Scalar and JAX derivatives agree within 1.40e-9 J/mol in H and
1.04e-12 J/mol/K in Cp. Entropy and heat capacity are exactly unchanged in
the analytic representation. Scalar, array, JIT and vmap checks reject
out-of-domain inputs rather than silently extrapolating them.

Fixed-network replacement calculation
-------------------------------------

The baseline is the named ``curated_cos`` factory, ``curated-cos-v1``. Its
75 existing endpoints from ``curated_cos_adoption/20261105/ions.npz`` are
reused only after checking every species, element, formula-matrix entry,
C/O-adjusted inventory and potential against that factory. The network has
537 species and the same charge treatment in both models. Only S8 changes.

The archived execution includes 75 new cold candidate solves, six new
selected tighter endpoint solves (three baseline and three candidate),
five new finite-path solves, and four standalone FastChem comparisons.
It covers T=1500/2000/2500/3000/4000 K, P=1e-6/1e-4/0.01/1/100 bar, and
C/O=0.5888436553555908/1/1.5. All converge. A neutral-network replacement
was not run.

The largest S8 absolute response and sulfur-allocation change occur at
3000 K, 100 bar and reference C/O. S8 mole fraction changes from
9.89533e-49 to 2.41270e-48, a factor 2.438223 and an absolute increase
1.42317e-48. Counting all eight sulfur atoms, its fraction of the sulfur
inventory changes from 3.52399e-43 to 8.59227e-43. No concentration floor
is introduced; amounts and logarithms are retained in the packet.

The finite path at this condition uses five equally spaced substitution
fractions. S8 increases monotonically. Its logarithmic response also
satisfies the independently checked formation relation: the change in
ln x(S8) equals eight times the change in ln x(S), plus the imposed change
in ln K. This verifies the tiny component within the chosen numerical
model; it does not establish a physical abundance error bound.

Numerical limits and controls
-----------------------------

Primary and strict solver tolerances are 5e-14 and 2e-14. Under tighter cold
reruns, S8 ln x changes by at most 5.58e-12. The all-species maximum is
1.46e-5, while the maximum for species with x>=1e-12 is 1.46e-6. Absolute
mole-fraction changes under tighter settings are below 3.38e-14. These
checks separate a stable S8 ratio from the numerical background.

Across both endpoints, the combined H/C/O/S relative inventory residual is
below 2.95e-12, charge residual below 1.96e-14 and formation-equation
residual below 1.65e-12 in ln K. The all-element relative maximum is
1.46e-6 for a much smaller trace inventory; it is reported separately.
FastChem 4.0.3 uses exactly the same thermodynamic functions, catalog and
inventory at two conditions and both endpoints. Its all-species logarithmic
agreement is within 5.06e-7 dex, with element closure below 5.27e-11,
charge residual below 1.29e-11 and formation residual below 3.22e-10.
This is independent numerical verification, not physical validation.

The largest apparent background response is an H atom mole-fraction change
of 3.30e-15, smaller than the tighter-control variation. It is not assigned
to a physical redistribution caused by S8. Similarly, the largest isolated
S8 contribution to dimensionless total G is about 1.17e-48, whereas the
displayed floating-point comparison scale reaches 1.94e-16. This scale is
a numerical diagnostic, not a rigorous uncertainty interval. All 75
full-system Gibbs-ordering comparisons are marked **unresolved**. Raw and
first-order feasibility-corrected differences are archived, and their
small signs are not used to claim that either energy model is physically
preferred. The energetic source assessment is independent of those signs.

This hot hydrogen-rich equilibrium grid does not assess cooler sulfur-rich
chemistry, condensation, photochemistry or atmospheric transport. Trace
numbers from it do not establish atmospheric irrelevance or abundance
bounds in another regime.

Reproduction and MCEG
---------------------

The candidate and execution packets are
``tests/data/validation/s8_candidate/20261105`` and
``tests/data/validation/s8_equilibrium_impact/20261105``. Their receipts
identify the actual commands, source/implementation bytes, output hashes
and FastChem binary hash. The latest archived run is ``s8-candidate-002``;
a prior successful run of the same 75 conditions preceded expansion of
the automatic dependency manifest. The counts above describe the archived
execution, not an additional set of distinct conditions.

Reproduce with::

   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/s8_candidate.py --build
   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/s8_equilibrium_impact.py \
       --output-dir results/s8-candidate-new \
       --fastchem-executable /path/to/fastchem

MCEG supports continuous energy construction, the scoped S8 logarithmic
response and the source-based energy preference. Resolved total-system
Gibbs ordering and complete physical accuracy remain unknown. This completes
the requested candidate and substitution evaluation; curated adoption is
not performed by this PR update.
