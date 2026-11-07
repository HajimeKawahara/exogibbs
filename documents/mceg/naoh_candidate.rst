NaOH evaluated-energy candidate and finite substitution
=======================================================

The evaluated-energy candidate is ready for a separate scoped production decision:
use Gurvich's reassessed central formation energy while retaining the current
thermal fit. Its change is numerically resolved and larger than the selected
same-energy NASA thermal control. This is a central-value preference, not proof
that the legacy energy lies outside experimental uncertainty. No production
coefficient or default changes in this investigation.

The inspected full Gurvich 1996 evaluation, DOI
`NaOH evaluation 10.1063/1.555982 <https://doi.org/10.1063/1.555982>`_,
page 1266, selects Hf298 = -191 +/- 8 kJ/mol from a slightly rounded weighted
average of reassessed flame-photometry studies. The quoted margin is a total
error estimate; it is not an independent Gaussian standard deviation. The
6.545428 kJ/mol balanced energy change is smaller than that molecular margin.
The underlying original flame studies were not independently reanalysed here.
This evaluation provides a defensible central energy without waiting for every
historical paper, while its uncertainty remains explicit.

The primary candidate changes only the actual inherited fit's 1/T coefficient::

   ln K_energy(T) = ln K_current(T) + (757949.572 - 764495.000)/(R T)

Thus reaction H and G increase by 6545.428 J/mol, and S/Cp remain unchanged.
The selected NASA/Gurvich control uses the same target D0 with different
molecular and atomic thermal functions. It is a reference-model sensitivity,
not a pure molecular correction or an uncertainty interval.

Source energy and normalization
--------------------------------

This continues :doc:`naoh_source_review` for the neutral monomer
``H1Na1O1``; the hydroxide dimer is a separate chemical species.
The original source packet and its historical MCEG assessments are unchanged.

For every molecule and neutral atom, the zero-energy gauge is
``Hf(298.15) - [H(298.15)-H(0)]``. A balanced atomization energy is the
stoichiometric sum of atomic gauges minus the molecular gauge. The reference
element heat contents cancel only in that balanced reaction. No polynomial is
evaluated at zero temperature, and no unique D0 is inferred by extrapolating
the finite-range FastChem fit.

The original JANAF source anchor is 764.495000 kJ/mol.
The evaluated target combines the printed Gurvich molecular
Hf298=-191.000 kJ/mol and
H298-H0=11.396 kJ/mol with the
explicit NASA neutral-atom energy headers. It is
757.949572 kJ/mol. Relative to JANAF, the atomic gauge sum
changes by 200.572 J/mol and the molecular gauge by
6746.000 J/mol. Their difference is
-6545.428 J/mol. The individual input gauges and source hashes
are archived in ``candidate.json``.

NASA's own molecular heat-content header gives native D0=
757.951243 kJ/mol; its small difference from the target
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

The published molecular formation-enthalpy margin is +/-8 kJ/mol.
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
     - 0.59166026
     - 1.03034870
     - 0.60961638
   * - 2000
     - 0.67461237
     - 1.02715785
     - 0.69293339
   * - 2500
     - 0.72986652
     - 1.02465486
     - 0.74786128
   * - 3000
     - 0.76919455
     - 1.02251267
     - 0.78651118
   * - 4000
     - 0.82134790
     - 1.01882769
     - 0.83681198

Finite response on the current network
--------------------------------------

Before baseline reuse, all 537 species, elements, stoichiometric entries,
75 full chemical-potential vectors, C/O settings and elemental budgets are
verified against the named ``curated_cos`` factory. The 75 archived current
endpoints come from ``curated_cos_adoption/20261105/ions.npz``. The new work
performs 75 cold primary-candidate solves on exactly that network and those
inventories. Only the NaOH potential changes. No source-specific species
is added, and the combined influence of multiple candidate substitutions is
not assessed.

The grid has 1500/2000/2500/3000/4000 K, five pressures from 1e-6 to 100 bar,
and C/O=0.5888436554, 1 and 1.5. Across the grid, the NaOH mole-fraction
ratio is 0.59166026--0.82137517.
The largest Na redistribution occurs at 1500 K, 100 bar and reference
C/O: the NaOH allocation changes from
0.884386% to 0.525180%
(-0.359205 percentage points).
Its mole fraction changes from 2.52187150e-08 to 1.49757938e-08.
These are newly solved candidate responses against verified archived current
endpoints, not ratios inferred directly from K.

The NASA same-energy control gives NaOH abundance ratios of
1.018828--1.030349 relative to the primary candidate at the three selected
conditions. Its largest absolute difference in any mole fraction is
4.52e-10. At the primary highlighted condition, atomic Na allocation changes
from 93.81487% to 94.15873%; NaH, Na+, water and electrons respond through
shared equilibrium constraints. The detailed packet records each response.

Numerical checks and scope
---------------------------

All 75 new endpoints converge at tolerance 5e-14. There are also
14 fresh selected cold controls at 2e-14:
6 baseline/candidate tighter solves, five points
on a finite lambda path at the highlighted condition, and
3 secondary controls. Baseline endpoints are
therefore reused only after full input matching and selected fresh checks.
No new neutral-network grid is claimed.

The largest relative Na inventory residual in the endpoint comparison
is 2.67e-11; H/C/O closure
is below 2.27e-13,
charge closure below 1.95e-14,
and mass-action residual below 1.65e-12
in natural logarithm. Across all trace-element inventories the maximum relative
residual is 1.46e-06; it
must not be described as 1e-12 accuracy for every element.

The tighter controls differ by at most 1.36e-12
in NaOH ln(x). The all-species maximum is
1.46e-05, or
1.46e-06 for x>=1e-12; maximum
absolute mole-fraction change is 3.93e-14.
The finite path has monotonic target response. Same-model Gibbs exchange
checks retain raw values and remove first-order inventory-feasibility shifts
before applying the 5e-14 ordering tolerance. All selected controls are checked
for conservation, charge and formation equations.

Four standalone FastChem v4.0.3 comparisons, current and primary candidate
at two exported temperatures/pressures, agree within
4.6e-06 dex
across all species. This independently checks numerical equilibrium for the
same supplied thermodynamics. It neither verifies physical source accuracy
nor independently covers every temperature or trace component.

Remaining thermal decision
---------------------------

The OYT5 thermal candidate remains open. Its predicted bending fundamental
228 cm-1 differs from Gurvich's estimated 300 cm-1, so the NASA energy rationale
does not establish NASA thermal superiority. The downloaded 4000 K partition
sum gives Cp=48.427 versus NASA 61.190 J/mol/K. A smooth interpolation of that
finite sum would not establish missing-state or moment convergence. Neither
OYT5 nor an extrapolation is used in these finite solves. The independent
energy-only step is deliberately separated from that unresolved thermal choice.

The source review retains the OYT definition-file nuclear-weight conflict
(6 versus paper/state weight 8), explicit isotope translation and the distinction
between an opacity lower-state cutoff and total partition/moment completeness.
Those issues are not bypassed by this continuous NASA comparison. The relevant
modern full article is `Owens 2021 NaOH thermal source
<https://doi.org/10.1093/mnras/staa4041>`_. No new OYT extrapolation is performed.

The fixed ideal-gas calculations exclude condensation, transport, photochemistry,
isotope-mixture validation and complete atmospheric error bounds. Numerical
convergence and small absolute abundance do not establish physical accuracy.

Reproduction and MCEG
----------------------

The candidate packet is ``tests/data/validation/naoh_candidate/20261105``;
the finite-run packet is ``tests/data/validation/naoh_equilibrium_impact/20261105``.
Both are offline and pin inputs, outputs and source conventions. Run::

   python mceg/reviews/naoh_candidate.py --build
   pytest tests/model_evidence/naoh_candidate_test.py tests/model_evidence/naoh_equilibrium_impact_test.py

Fresh finite solves require an unused output directory::

   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 10-11 python mceg/reviews/naoh_equilibrium_impact.py \
     --output-dir results/naoh-candidate-impact-new \
     --fastchem-executable /tmp/mceg-fastchem-v403-bin

The new contract ``mceg/contracts/naoh_candidate.yaml`` records continuous
construction, finite response and the scoped decision separately from unknown
complete physical accuracy. The historical source contract and receipts are
preserved. Its energy/thermal-comparison and finite-substitution work is now
completed to the scope above; the full OYT thermal-selection follow-up remains
open. No curated adoption or merge is performed by these calculations.
