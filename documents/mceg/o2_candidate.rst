O2 evaluated natural-isotope thermal candidate
==============================================

The 2025 evaluated natural-isotope thermals with same-version ATcT energy are
a scientifically supported candidate for a separately scoped O2 update.
The fresh fixed-network response is small: O2 increases by 0.281-0.823%.
This review implements and tests the candidate; production is unchanged.
The energy-only alternative is an explicit selected control.

Continuous evaluated thermals and physical energy
-------------------------------------------------

The inspected primary evaluation is Furtenbacher, Harvey and Csaszar (2025),
`DOI 10.1063/5.0273661 <https://doi.org/10.1063/5.0273661>`_. Its official
natural-O2, pure-16O2 and partition-function supplements are SHA-pinned.
All 2501 published natural rows from 1500 to 4000 K are retained numerically.
A degree-8 Cp/R polynomial is fit to 251 ten-kelvin rows, using inverse
published expanded Cp uncertainty as a numerical weight. Analytic integration
with the published 1500 K H/S anchors gives one smooth Gibbs function.
The remaining 2250 Cp rows and all non-anchor H/S rows are held out of fitting.
This weighting is not a covariance model or a new physical uncertainty estimate.

Maximum differences from the rounded source are 3.895 J/mol in G,
1.520 J/mol in H, 0.001208 J/mol/K in S and 0.005648 J/mol/K in Cp.
They are below 7.9% of the corresponding published expanded H/S/Cp
uncertainties. Independent Simpson integration of the published Cp rows agrees
with source H/S within 0.936 J/mol and 0.001017 J/mol/K, respectively.
The positive molecular Cp exceeds 36 J/mol/K on the candidate domain.
The full published tables extend to 5000 K; no out-of-domain extrapolation is
needed for the present 1500-4000 K calculation.

The natural-isotope table already includes the authors' heat-capacity correction,
its integrated H/S effects and mass-translation entropy. Its rotational,
electronic and nuclear statistical conventions are retained. No extra triplet
factor, homonuclear symmetry division, isotope mixing or mass correction is
applied. The pure-isotope table and Q moments remain independent source controls;
they are not relabeled as natural-isotope thermodynamics.

For 2 O(g) -> O2(g), ATcT 1.228 gives 2(246.843)-0=493.686 kJ/mol.
This is a physical D0, distinct from the formal empirical-fit intercept.
Relative to physical JANAF D0=493.579 kJ/mol, delta D0=+107 J/mol lowers
formation G/H by 107 J/mol in the energy-only control and leaves S/Cp unchanged.
O2's zero elemental formation enthalpy does not mean zero atomization energy.
No additional zero-point correction is applied.

The old molecular thermal proxy is NASA9 G-H0, after removing its printed H0
with native R=8.314510 J/mol/K and normalizing the dimensionless thermals with
SI R=8.31446261815324 J/mol/K. The primary candidate is

.. math::

   G_{\rm candidate}=G_{\rm current}-107\ {\rm J/mol}
     +\phi_{\rm natural,2025}-\phi_{\rm NASA,O_2}.

Thus it preserves the complete literal-fit residual relative to the NASA
molecular/atomic thermal gauge and physical JANAF D0. Atomic thermals are not
changed. This is a proxy-relative update, not a direct replacement of a literal
JANAF table. The source review's pointwise 2025+ATcT/NASA-atom diagnostic omitted
this empirical residual; its quoted ratios are therefore not the new candidate
or its solved abundance response. Both constructions remain documented.

Fresh substitutions and finite numerical evidence
-------------------------------------------------

The primary receives 150 fresh cold endpoints on the named
``curated-ch2-v1`` baseline: 75 ion-inclusive with 537 species and 75 neutral
with 421 species. Five temperatures (1500, 2000, 2500, 3000, 4000 K), five
pressures (1e-6, 1e-4, 0.01, 1, 100 bar) and three C/O ratios are used.
The unique 150 baseline cases are reused after exact input hashes, species,
formula matrices, conditions, elemental budgets and factory potentials match.
Only the O2 potential is substituted; the current HCP/C2H2/CH2 adoptions are
part of the fixed common background.

There are 12 fresh tighter baseline/candidate controls, 10 finite-path nodes
and 4 energy-only selected controls. Four independent FastChem executions
(two selected ion-inclusive conditions, baseline and primary at each) agree
within 1.38e-8 dex. All solves converge. Scalar/JAX G/H/S/Cp are independently
checked at 2752 temperatures, including 251 noninteger points, for both main
and selected potentials.

The O2 mole-fraction ratio to baseline is 1.00280614-1.00823092 in both networks.
The largest absolute response is at 2000 K, 1e-6 bar and reference
C/O=0.5888436554: ion-inclusive x(O2) changes from 1.32080945e-8 to
1.32877249e-8. The oxygen fraction in O2 increases from 0.00489407903%
to 0.00492358498%. These are coupled solved responses, not equilibrium-constant
sensitivities. The small footprint belongs to this hot H-rich inventory;
it does not constrain oxidizing atmospheres or photochemical oxygen.

Major H/C/O closure is within 2.30e-13, all-element trace-relative closure
within 1.41e-6, charge residual within 4.01e-14 and formation residual within
1.26e-12 in ln units. Tighter target ln(x) differences are below 8.76e-12.
Only 14/75 same-model Gibbs exchanges per network are resolved above the
floating-point diagnostic scale; 61 remain unresolved. This is expected to
limit the precision of a total-system diagnostic for such a small perturbation.
It is not silently promoted to a rigorous numerical bound or a physical
source-selection test.

Decision and limitations
------------------------

Prefer the evaluated natural-isotope thermal construction plus balanced ATcT
central energy for a separately authorized, bounded O2 adoption. The source
includes explicit thermodynamic conventions and an uncertainty evaluation;
closeness to NASA or the small response alone is not the reason for preference.
The current fit residual is deliberately retained, so this is not an independent
refit of all molecular and atomic thermochemistry.

The 2025 evaluation retains the 15946-level O2 manifold from 2019 and updates
constants and uncertainty treatment. Unbound-state uncertainty is not quantified,
and the natural-mixture Cp correction carries the authors' expanded uncertainty.
Complete reaction covariance, scattering/continuum assignment, isotope-network
accuracy and atmospheric validation remain unknown. The ideal-gas calculation
does not validate condensation, transport or photochemical kinetics.

The MCEG record separates supported continuous construction, finite substitution
and scoped preference from universal Gibbs resolution, complete physical accuracy
and production adoption. Existing source-review packets and runtime bytes remain
unchanged. Offline replay is ``pytest tests/model_evidence/o2_candidate_test.py``.
