PS curated adoption with retained energy
========================================

``curated_ps.chemsetup`` now applies the scoped POPS thermal update on
``curated-cn-v1``. Only P1S1 changes. The adopted model preserves the Lodders
D0, atomic functions and inherited fit/source residual, while including explicit
ground and excited rovibronic levels. This is a defensible conditional scientific
choice; the unrefined ab initio a4Pi manifold prevents a claim of decisively
improved physical accuracy. The completed candidate reasoning remains in
:doc:`ps_candidate`, with its source packet unchanged.

The named ``curated-ps-v1`` factory supports both ion-inclusive and neutral
networks on 1500--4000 K. The public default switch is coordinated with the
subsequent TiO2 adoption. Production loads a small packaged coefficient record;
it does not load the 223645 retained state rows or require MCEG.

Compact thermodynamics
----------------------

The construction retains the candidate definition exactly at the source level:
add the POPS molecular G-H0 minus the NASA/Gurvich molecular G-H0 to the inherited
reaction fit. Delta D0 is zero. The retained energy is 438 +/-10 kJ/mol, with a
fixed central cutoff at 36613.940864 cm-1. Nuclear-spin factor 2 is removed once;
31P32S internal levels retain the existing natural-mass translation proxy.
The native NASA gas constant and header-defined H0 are unchanged. Neither the
finite-range inherited fit nor the compact representation is extrapolated to
zero temperature or interpreted as a new energy measurement.

To avoid a large state array at runtime, package 19 coefficients for

.. math::

   \ln Q(T) = 2\ln T + \sum_{k=0}^{18} c_k T_k(x),
   \qquad
   x = \frac{\ln T-\tfrac12(\ln1500+\ln4000)}
                  {\tfrac12(\ln4000-\ln1500)}.

The polynomial is projected at 80 Chebyshev nodes. It is continuous with smooth
thermodynamic derivatives throughout the restricted interval. Independent
state-energy moments check lnQ, G, H, S and Cp at all 2501 integer temperatures
and 257 fixed off-grid temperatures. Degree convergence is measured from 10
through 24; degree 18 reaches the numerical roundoff regime, and increasing the
degree further does not reliably improve derivative errors.

.. list-table:: Maximum absolute errors of the packaged JAX reaction against direct candidate moments
   :header-rows: 1

   * - Quantity
     - Measured maximum
     - Acceptance limit
   * - G (J/mol)
     - 8.74e-10
     - 5e-8
   * - H (J/mol)
     - 6.85e-8
     - 1e-6
   * - S (J/mol/K)
     - 4.56e-11
     - 1e-9
   * - Cp (J/mol/K)
     - 8.53e-9
     - 1e-7
   * - G/(RT)
     - 2.85e-14
     - 2e-12

These are sampled numerical approximation errors, not physical uncertainty or
a rigorous uniform error bound. They are negligible relative to the reviewed
4000 K thermal differences: K/current = 1.014493 and delta Cp = +4.20505 J/mol/K.
No extra energy shift is used to force a fit. The source-defined energy and
finite-temperature fit residual are retained within the displayed approximation
accuracy. Scalar, array, JIT and differentiation paths share one function;
invalid temperatures return NaN and the full state sum remains an offline check.

Production verification and abundance response
----------------------------------------------

Both 537-species ion-inclusive and 421-species neutral networks receive 75 fresh
cold production solves: five temperatures from 1500 to 4000 K, five pressures
from 1e-6 to 100 bar and C/O = 0.5888436554, 1 and 1.5. Species, elements, formula
matrix, budgets and all baseline potentials are verified against the preceding
CN archive. Every non-PS potential remains identical. The target reproduces the
independent direct-state candidate to below 3e-14 in G/(RT) across 2501 nodes.

Six tighter cold solves use the original direct-state candidate assembled on
the CN base, independently of the compact runtime representation. Two standalone
FastChem comparisons additionally verify the supplied production potentials.
All 150 production endpoints and all six tighter controls converge.

.. list-table:: PS response relative to the immediate CN base
   :header-rows: 1

   * - Quantity
     - Ion-inclusive
     - Neutral
   * - Minimum x(new)/x(base)
     - 0.999314096
     - 0.999314096
   * - Maximum x(new)/x(base)
     - 1.014492964
     - 1.014492964
   * - Largest phosphorus-share response, before (%)
     - 21.0165515
     - 21.0165527
   * - Same condition, after (%)
     - 21.0054430
     - 21.0054442
   * - Maximum PS ln(x) tighter-control difference
     - 9.60e-14
     - 4.98e-14

The largest phosphorus redistribution is at 1500 K, 1e-4 bar and reference C/O.
In the ion-inclusive network, x(PS) changes from 9.26510054e-8 to 9.26020337e-8,
a decrease of 4.89718e-11 and 0.01111 percentage points of phosphorus allocation.
The high-temperature 1.45 percent relative increase concerns smaller PS amounts.

Across both networks the maximum selected absolute mole-fraction control
variation is 3.25e-14. Element closure is below 1.38e-6 across all trace elements
and below 1.42e-10 for H/C/O/P/S; logarithmic mass-action residual is below
1.17e-12. Relative charge residual is below 1.26e-14 where charge is present.
The two FastChem comparisons differ by at most 5.95e-6 dex across all species.
Trace responses near the control scale are not interpreted as meaningful.

The new CN background must be separated from compact-fit reproduction. Relative
to the old PS-only ion-inclusive candidate on ``curated-cos-v1``, PS abundance
ratios span 0.997633456--1.000000035. The largest whole-network absolute change,
1.17123e-6, is due to the CN background update; it is not the PS thermal response
or a fitting error. The previous PS candidate did not contain a neutral solve,
so no such neutral comparison is invented.

Evidence and limits
-------------------

The contract ``mceg/contracts/curated_ps.yaml`` separates supported scientific
selection and production adoption from unknown complete physical accuracy.
The numerical record includes the fit construction, direct state moments,
continuous derivatives, preserved non-PS potentials, finite responses,
conservation, charge and independent solver checks. The source/candidate packets
and their historical MCEG assessments remain unchanged.

At 4000 K the a4Pi and B2Pi manifolds provide 1.57709 and 0.04834 percent of the
retained Q. The a4Pi potential is unrefined and omitted couplings remain a
limitation. Fixed-cutoff sensitivity, incomplete higher-state knowledge,
representative-isotope translation and the broad evaluated energy margin are
not converted into complete uncertainty bounds. The result concerns ideal-gas
equilibrium, not condensation, disequilibrium chemistry, transport or observed
planetary atmospheres. No missing ancestral full text is made a prerequisite
for this scoped adoption.

Reproduce the compact fit and archived tests with::

   python mceg/reviews/build_ps_adoption.py --validate
   pytest tests/unittests/presets/curated_ps_test.py
   pytest tests/model_evidence/ps_adoption_fit_test.py tests/model_evidence/curated_ps_adoption_test.py

For fresh production endpoints, run
``python mceg/reviews/curated_ps_adoption.py --output-dir results/ps-production-new --fastchem-executable /path/to/fastchem``.
The production and fit receipts are separately stored under
``tests/data/validation/curated_ps_adoption/20261106``. Runtime coefficients live
in the explicitly packaged ``src/exogibbs/data/curated/ps_v1.json``.
