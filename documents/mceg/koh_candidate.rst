KOH thermal-reference candidate and finite substitution
=======================================================

Retain current KOH production coefficients pending a resolved thermal choice.
A continuous NASA/Gurvich comparison candidate is implemented and its finite
response is numerically resolved, but the calculation does not establish it
as scientifically preferable to the current thermal function. The modern OYT4
state sum is not yet supported as a complete 1500--4000 K replacement.

The full Gurvich 1997 evaluation, DOI
`KOH evaluation 10.1063/1.555996 <https://doi.org/10.1063/1.555996>`_,
page 1050 and Table 23, provides a defined linear-singlet RRHO model with
symmetry number one, electronic statistical weight one and low-temperature
quantum corrections, tabulated to 6000 K. The paper explicitly estimates
increasing high-temperature uncertainty in this approximation. NASA reproduces
selected printed caloric nodes. This makes a controlled full-range numerical
comparison possible; it does not make the source's age or polynomial range
an accuracy guarantee.

The primary comparison replaces the molecular/atomic thermal functions by
NASA/Gurvich while retaining the balanced JANAF source D0=781106 J/mol.
The selected secondary control instead uses Gurvich's evaluated energy,
D0=780727.572 J/mol. Its -378.428 J/mol change is isolated from the thermal
change. A retained-thermal energy-only diagnostic is also evaluated in the
candidate packet, but it is not the 75-case primary substitution.

Source energy and normalization
--------------------------------

This continues :doc:`koh_source_review` for the neutral monomer
``H1K1O1``; the hydroxide dimer is a separate chemical species.
The original source packet and its historical MCEG assessments are unchanged.

For every molecule and neutral atom, the zero-energy gauge is
``Hf(298.15) - [H(298.15)-H(0)]``. A balanced atomization energy is the
stoichiometric sum of atomic gauges minus the molecular gauge. The reference
element heat contents cancel only in that balanced reaction. No polynomial is
evaluated at zero temperature, and no unique D0 is inferred by extrapolating
the finite-range FastChem fit.

The original JANAF source anchor is 781.106000 kJ/mol.
The evaluated target combines the printed Gurvich molecular
Hf298=-232.000 kJ/mol and
H298-H0=11.674 kJ/mol with the
explicit NASA neutral-atom energy headers. It is
780.727572 kJ/mol. Relative to JANAF, the atomic gauge sum
changes by 0.572 J/mol and the molecular gauge by
379.000 J/mol. Their difference is
-378.428 J/mol. The individual input gauges and source hashes
are archived in ``candidate.json``.

NASA's own molecular heat-content header gives native D0=
780.728222 kJ/mol; its small difference from the target
is retained explicitly. The target uses the printed evaluated heat content,
not an unexplained extra correction. The original NASA dimensionless reaction
is converted to the selected target once::

   h_control(T) = h_NASA(T) - [D0_target/R_SI - D0_NASA/R_NASA]/T

Here h=Gformation/(RT), R_SI=8.31446261815324 and R_NASA=8.314510 J/mol/K.
The single NASA 1000--6000 K interval covers the entire candidate domain, so
no internal interval join or new polynomial fit is introduced. Both candidate
paths are restricted to 1500--4000 K and one-bar standard pressure. Analytic
H/S/Cp and JAX derivatives agree; an independent dense reconstruction uses
2501 temperatures. Native-R energy conversion is separate from thermal terms.

The published molecular formation-enthalpy margin is +/-3 kJ/mol.
It is not propagated as an independent balanced-D0 uncertainty: atomic
covariance, the thermal approximation and source ancestry are not supplied.
Natural-mass source conventions remain in the inherited/NASA representations.
No additional nuclear-spin, symmetry, isotope or dissociation-zero factor is
applied to those published thermal functions.

.. list-table:: Formation-K diagnostics; these are not abundance ratios
   :header-rows: 1

   * - T (K)
     - Energy-only/current
     - Same-energy NASA/current thermal reference
     - NASA thermal + Gurvich energy/current
   * - 1500
     - 0.97011278
     - 1.26075776
     - 1.22307721
   * - 2000
     - 0.97749978
     - 1.26911292
     - 1.24055759
   * - 2500
     - 0.98195895
     - 1.27382523
     - 1.25084408
   * - 3000
     - 0.98494303
     - 1.27681671
     - 1.25759172
   * - 4000
     - 0.98868588
     - 1.27962932
     - 1.26515145

Finite response on the current network
--------------------------------------

Before baseline reuse, all 537 species, elements, stoichiometric entries,
75 full chemical-potential vectors, C/O settings and elemental budgets are
verified against the named ``curated_cos`` factory. The 75 archived current
endpoints come from ``curated_cos_adoption/20261105/ions.npz``. The new work
performs 75 cold primary-candidate solves on exactly that network and those
inventories. Only the KOH potential changes. No source-specific species
is added, and the combined influence of multiple candidate substitutions is
not assessed.

The grid has 1500/2000/2500/3000/4000 K, five pressures from 1e-6 to 100 bar,
and C/O=0.5888436554, 1 and 1.5. Across the grid, the KOH mole-fraction
ratio is 1.24971395--1.27962932.
The largest K redistribution occurs at 1500 K, 100 bar and reference
C/O: the KOH allocation changes from
3.345401% to 4.180795%
(+0.835393 percentage points).
Its mole fraction changes from 6.75351025e-09 to 8.43995598e-09.
These are newly solved candidate responses against verified archived current
endpoints, not ratios inferred directly from K.

Adding Gurvich's small energy change to the same NASA thermal function gives
KOH abundance ratios 0.971342--0.988686 relative to the primary comparison at
the two selected conditions. The largest absolute change in any mole fraction
is 2.42e-10. At the primary highlighted condition, atomic K allocation changes
from 80.06862% to 79.36740%; KH, K+, Na species and electrons adjust through
equilibrium. Thus the thermal-reference effect is larger than this energy
control, but its numerical size does not decide which thermal source is correct.

Numerical checks and scope
---------------------------

All 75 new endpoints converge at tolerance 5e-14. There are also
11 fresh selected cold controls at 2e-14:
4 baseline/candidate tighter solves, five points
on a finite lambda path at the highlighted condition, and
2 secondary controls. Baseline endpoints are
therefore reused only after full input matching and selected fresh checks.
No new neutral-network grid is claimed.

The largest relative K inventory residual in the endpoint comparison
is 4.23e-10; H/C/O closure
is below 2.27e-13,
charge closure below 1.95e-14,
and mass-action residual below 1.65e-12
in natural logarithm. Across all trace-element inventories the maximum relative
residual is 1.46e-06; it
must not be described as 1e-12 accuracy for every element.

The tighter controls differ by at most 9.95e-14
in KOH ln(x). The all-species maximum is
1.46e-05, or
1.46e-06 for x>=1e-12; maximum
absolute mole-fraction change is 3.38e-14.
The finite path has monotonic target response. Same-model Gibbs exchange
checks retain raw values and remove first-order inventory-feasibility shifts
before applying the 5e-14 ordering tolerance. All selected controls are checked
for conservation, charge and formation equations.

Four standalone FastChem v4.0.3 comparisons, current and primary candidate
at two exported temperatures/pressures, agree within
6.34e-06 dex
across all species. This independently checks numerical equilibrium for the
same supplied thermodynamics. It neither verifies physical source accuracy
nor independently covers every temperature or trace component.

Remaining thermal decision
---------------------------

A NASA/Gurvich production preference remains unsupported. Its bending
fundamental is estimated as 300 cm-1, whereas the inspected modern OYT4 work
predicts 343 cm-1 with different ab initio thermal behavior. At 4000 K the
available OYT4 sum gives Cp=43.854 versus NASA 61.201 J/mol/K; upper-energy
and energy-moment completeness have not been established. A smooth fit to
those numbers would not resolve the missing-state problem. The comparison
candidate is therefore a bounded sensitivity study, not an OYT4 substitute
or a recommendation to adopt NASA merely because it is a later evaluation.

The source review retains the OYT definition-file nuclear-weight conflict
(6 versus paper/state weight 8), explicit isotope translation and the distinction
between an opacity lower-state cutoff and total partition/moment completeness.
Those issues are not bypassed by this continuous NASA comparison. The relevant
modern full article is `Owens 2021 KOH thermal source
<https://doi.org/10.1093/mnras/staa4041>`_. No new OYT extrapolation is performed.

The fixed ideal-gas calculations exclude condensation, transport, photochemistry,
isotope-mixture validation and complete atmospheric error bounds. Numerical
convergence and small absolute abundance do not establish physical accuracy.

Reproduction and MCEG
----------------------

The candidate packet is ``tests/data/validation/koh_candidate/20261105``;
the finite-run packet is ``tests/data/validation/koh_equilibrium_impact/20261105``.
Both are offline and pin inputs, outputs and source conventions. Run::

   python mceg/reviews/koh_candidate.py --build
   pytest tests/model_evidence/koh_candidate_test.py tests/model_evidence/koh_equilibrium_impact_test.py

Fresh finite solves require an unused output directory::

   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 10-11 python mceg/reviews/koh_equilibrium_impact.py \
     --output-dir results/koh-candidate-impact-new \
     --fastchem-executable /tmp/mceg-fastchem-v403-bin

The new contract ``mceg/contracts/koh_candidate.yaml`` records continuous
construction, finite response and the scoped decision separately from unknown
complete physical accuracy. The historical source contract and receipts are
preserved. Its energy/thermal-comparison and finite-substitution work is now
completed to the scope above; the full OYT thermal-selection follow-up remains
open. No curated adoption or merge is performed by these calculations.
