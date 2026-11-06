TiO2 measured-thermal candidate and finite substitution
=======================================================

Decision
--------

Within the explicitly stated rigid-rotor/harmonic-oscillator (RRHO)
approximation, prefer measured ground-state rotation and bending over the
1973 estimated geometry and bend for the next scoped TiO2 thermal update.
The candidate is numerically verified over 1500--4000 K and changes only the
TiO2 potential on the fixed gas network. Its roughly 9--11% formation-K change
is larger than the selected bend and electronic sensitivities evaluated here.
Those sensitivities are not a complete uncertainty interval, and the numerical
result does not establish universal physical accuracy.

This review is ready to merge as candidate evidence. A separately authorized
curated adoption can use this bounded thermal direction with its stated limits;
acquisition of every historical energy paper is not a prerequisite. Production
``curated-cos-v1`` and its energy reference remain unchanged in this PR. A new
atomization energy would require a separate source-cycle decision.

Continuous construction and reference convention
-------------------------------------------------

The candidate retains the literal current FastChem fit and adds the molecular
thermal difference only:

.. math::

   G_{\mathrm{candidate}}(T) = G_{\mathrm{current}}(T)
       + [G_{\mathrm{RRHO}}(T)-H_{0,\mathrm{RRHO}}]
       - [G_{\mathrm{NASA}}(T)-H_{0,\mathrm{NASA}}].

The NASA TiO2 function is a continuous proxy for the inherited JANAF thermal
function. All existing fit residuals relative to that proxy are retained,
including their temperature dependence; the candidate does not reconstruct the
whole reaction from a new atomic background. All atomic and other molecular
potentials remain unchanged. Thus the documented JANAF source D0=1267.776 kJ/mol
is retained, without inventing a unique zero-temperature limit of a finite-range
fit. This is distinct from replacing the entire fitted reaction by RRHO plus
fresh atomic functions.

The official NASA header gives Hf298=-305430 J/mol and H298-H0=11351.690 J/mol
using its native R=8.31451 J/mol/K. Their difference divided by native R is removed
from the dimensionless NASA coefficients before applying the common
R=8.31446261815324. No zero-point energy is added to D0 a second time. The
formation reaction is ``Ti(g) + 2 O(g) -> TiO2(g)``, with one-bar standard state.
The NASA 1000--6000 K interval covers the entire candidate domain; it is not
extended outside its interval to evaluate a zero-temperature limit.

RRHO uses the measured ``48Ti16O2`` rotational constants
30520.54148, 8471.75838 and 6613.58731 MHz from
`Brunken et al. 2008 <https://doi.org/10.1086/528934>`_, and ground-state gas
fundamentals 959, 322 and 944 cm-1, treated explicitly as harmonic frequencies.
The ground bend and its 6 cm-1 experimental interval are from
`Wang et al. 2009 <https://doi.org/10.1039/B821849H>`_; the stretches are gas
values in the `NIST compilation
<https://webbook.nist.gov/cgi/cbook.cgi?ID=C13463677&Mask=3EE9&Units=SI>`_.
The ground state is a singlet, the symmetry number is two, and nuclear-spin
weight is one. No additional spin/parity multiplicity is introduced. The JANAF
mean mass 79.8788 u is retained to isolate thermal structure, despite using
principal-isotopologue rotational constants; this is not an isotope mixture.

The separate ``h298`` control subtracts the new-minus-NASA 298 K molecular heat
content, -169.818432 J/mol, from the primary candidate. Thus it holds the old
molecular H298 reference instead of D0, adding +169.818432 J/mol to reaction
G and H with unchanged S and Cp. The earlier source-only RRHO/JANAF diagnostic
held JANAF H298 and used the printed JANAF heat content, 11357 J/mol, so it must
not be presented as the same candidate or the same K ratio.

.. list-table:: Continuous candidate K divided by current K
   :header-rows: 1

   * - T (K)
     - Retained D0, primary
     - H298-fixed control
   * - 1500
     - 0.908104
     - 0.895823
   * - 2000
     - 0.903388
     - 0.894209
   * - 2500
     - 0.900476
     - 0.893149
   * - 3000
     - 0.898314
     - 0.892219
   * - 4000
     - 0.893213
     - 0.888663

Physical scope and selected sensitivities
------------------------------------------

This is a ground-state RRHO candidate, not a complete modern rovibronic state
sum. It omits anharmonicity, centrifugal distortion and a full electronic-state
manifold. The measured excited-state geometry is not used as ground-state
geometry. The 316--328 cm-1 bend sensitivity gives K multipliers
1.01791--0.98275 at 4000 K relative to the primary candidate. This varies one
input only; it does not propagate all physical uncertainties.

The primary abstract of `Zhuang et al. 2010
<https://doi.org/10.1039/C0CP00861C>`_ supplies a singlet A-state origin of
17593 cm-1 and harmonic frequencies 876, 184 and 316 cm-1. Under the expressly
assumed equal rotational partition functions, adding this state multiplies K
by 1.008819 at 4000 K (1.001021 at 3000 K). Its softer vibrational modes matter:
an equal-rovibrational Boltzmann factor alone understates this diagnostic.
The legacy estimated triplet at 18880 cm-1 contributes a relative weight
0.003371 at 4000 K under equal rovibrational functions. Neither approximation
bounds all missing states; the NIST lower-case state label alone is not used to
infer a triplet multiplicity. No excited-state correction is silently included
in the ground-state primary candidate.

The observed central thermal shift is therefore worth updating within the
same RRHO approximation, while the full high-temperature error remains unknown.
Printed JANAF Hf298 uncertainty 12.6 kJ/mol and S298 uncertainty 4.2 J/mol/K are
inherited evaluation scales, not reduced or replaced by these sensitivity tests.
Hildenbrand's separate D0 evaluation is not adopted on the basis of an abstract.
Missing historical energy-cycle papers do not block this thermal evidence PR.

Finite network response and numerical checks
---------------------------------------------

All 75 candidate endpoints were solved afresh in float64. The archived 75 current
endpoints were reused only after checking all 537 species, elements, formula
matrix, elemental budgets and baseline potentials against named
``curated-cos-v1``. The grid has five temperatures (1500--4000 K), five pressures
(1e-6--100 bar) and three C/O ratios (reference, 1, 1.5). Six tighter cold endpoint
solves, five interpolation-path nodes and three H298-control solves also
converged. Four standalone FastChem comparisons at two selected conditions
agreed within 6.76e-9 in log10 mole fraction.

TiO2 mole-fraction ratios span 0.893213--0.951930. The largest Ti allocation
response is at 1500 K, 100 bar and reference C/O: TiO2 falls from 50.0872% to
47.6795% of Ti, while TiO rises from 48.9555% to 51.3179%. TiO2 mole fraction
there changes from 8.03171e-8 to 7.64562e-8. The largest absolute mole-fraction
change anywhere is 3.86096e-9. At the three selected conditions the H298-control
TiO2 abundance is 0.99288--0.99491 times the primary candidate abundance.
These are gas-network allocations, not condensation, cloud or opacity predictions.

The maximum target tight-control difference is 7.82e-14 in ln mole fraction;
Ti conservation is within 5.43e-10 relative, H/C/O within 2.28e-13, charge within
1.96e-14 and mass action within 1.65e-12 in log units. Trace-element residuals
reach 1.46e-6 and all-species tight differences 1.46e-5 in ln mole fraction;
they are reported separately rather than interpreted as TiO2 accuracy.
Only 36 of 75 same-model, feasibility-corrected total-G comparisons have both
exchange signs above the displayed floating-point scale. The rest remain
numerically unresolved. Absolute G differences between different models cannot
choose a physically superior source, and small background responses near the
control variation must not be overinterpreted.

Reproduction and MCEG
----------------------

Run ``python mceg/reviews/tio2_candidate.py --build`` to reproduce the continuous
candidate, then ``python mceg/reviews/tio2_equilibrium_impact.py --output-dir
results/tio2-impact-new --fastchem-executable /path/to/fastchem`` for fresh
substitutions. The frozen packets under
``tests/data/validation/tio2_candidate/20261106`` and
``tests/data/validation/tio2_equilibrium_impact/20261106`` replay offline with
``pytest tests/model_evidence/tio2_candidate_test.py
 tests/model_evidence/tio2_equilibrium_impact_test.py``.

The 2501-temperature independent reconstruction agrees within 4.3e-14 in
G/(RT); dense scalar/JAX comparison and independent analytic/automatic G/H/S/Cp
derivatives are checked. MCEG ``tio2_candidate.yaml`` separately records supported
continuous thermodynamics, finite substitution and scoped thermal direction;
universal numerical Gibbs resolution and complete physical accuracy remain
unknown. This completes the earlier ``thermal_candidate`` follow-up within its
bounded numerical scope, preserving the source-review contract and receipts as
historical records.
