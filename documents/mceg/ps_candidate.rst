PS retained-energy POPS candidate and finite substitution
========================================================

A POPS thermal update is a scientifically defensible **optional, scoped adoption**
for 1500--4000 K. It includes resolved ground and excited levels and produces a
measurable caloric change, with a small abundance response in the sampled gas
network. It has lower priority than the CN energy update. The main high-temperature
change depends on the unrefined ab initio a4Pi manifold, so this review does not
establish a decisive gain in physical accuracy. Curated remains unchanged because
this is a candidate evaluation; missing ancestral literature is not a merge gate.

Construction and energy convention
----------------------------------

The neutral reaction is P(g) + S(g) -> PS(g), at a one-bar standard pressure.
Retain the literal current FastChem fit and the
`Lodders (2004) <https://doi.org/10.1063/1.1611178>`_ source energy,
D0 = 438 +/-10 kJ/mol. The continuous candidate is

.. math::

   G_{\mathrm{candidate}}^{\mathrm{formation}}(T)
   = G_{\mathrm{current}}^{\mathrm{formation}}(T)
   + [G_{\mathrm{POPS}}(T)-H_{0,\mathrm{POPS}}]
   - [G_{\mathrm{NASA/Gurvich}}(T)-H_{0,\mathrm{NASA/Gurvich}}].

Only the molecular thermal difference is added. Atomic functions and the current
fit/source residual remain intact. Lodders and the selected NASA record share
Gurvich thermal ancestry; this is an explicit difference reference, not a claim
that NASA exactly reproduces every original fit. The NASA molecular H0 gauge is
140815.2 J/mol, reconstructed from its H298 and H298-H0 headers. The molecular
polynomial starts at 300 K and is never extrapolated to zero. NASA dimensional
values use native R = 8.314510; the state sum and reaction logarithms use
R = 8.31446261815324 J/mol/K. This distinction is retained in the formula.

There is no energy-only change: delta D0 = 0. Current and POPS candidates isolate
the thermal change. D0 remains a documented physical source anchor, not an asserted
zero-temperature limit of the finite-range FastChem fit. The POPS potential
De = 37004 cm-1 is not substituted for D0, and no extra zero-point correction is
added to the already ground-relative state energies.

State inventory and sensitivities
---------------------------------

`Prajapat et al. (2017) <https://doi.org/10.1093/mnras/stx2229>`_ distributes
225997 levels of 31P32S with X2Pi, a4Pi and B2Pi labels, through J = 320.5.
The nuclear-spin factor 2 is removed once. Resolved spin-orbit, parity and
electronic multiplicities receive no extra weights; symmetry number is 1.
The paper and definition file give a 5000 K line-list/PF limit. Its 25000 cm-1
lower-state transition threshold does not truncate the partition sum. X and B
curves are empirically refined; a4Pi is unrefined because experimental data were
unavailable, and its coupling to X was omitted. These are physical limitations,
not numerical convergence failures.

The central retained D0 corresponds to 36613.940864 cm-1. The candidate uses the
223645 distributed levels below that fixed threshold. All 225997 levels remain
in the immutable source packet and a separate full-distribution diagnostic.
The 2352 above-threshold levels are not identified here as bound levels or
quasibound resonances. Their inclusion/exclusion is an explicit modeling choice.
At 4000 K the full sum raises Q by 9.733 ppm and Cp by 0.009735 J/mol/K relative
to the candidate; at 1500 K its Q effect is below 2e-14 relative. Lowering only
the cutoff by the published 10 kJ/mol evaluation margin lowers Q by 57.03 ppm
and Cp by 0.05507 J/mol/K at 4000 K. Raising it by that margin retains the full
distribution. These are cutoff sensitivities, not energy shifts or uncertainty
bounds for missing levels.

Truncating instead at 30000 cm-1 or J = 240.5 would lower Cp at 4000 K by 1.1153
or 1.1456 J/mol/K, respectively. The high levels must therefore be retained for
caloric comparisons even where their Q fractions look small. The candidate uses
all levels below its stated central threshold. Finite-list convergence cannot
bound unlisted manifolds or errors in the a4Pi potential.

Internal levels are 31P32S, while translation retains the current natural-mass
proxy 63.038761 u. Using the principal isotopologue mass 62.94583263 u would
multiply the molecular formation constant by 0.9977896. This is a separately
reported mass-convention sensitivity, not a natural-isotope mixture calculation.

.. list-table:: Candidate divided by current formation constant and caloric differences
   :header-rows: 1

   * - T (K)
     - K ratio
     - delta H (J/mol)
     - delta S (J/mol/K)
     - delta Cp (J/mol/K)
   * - 1500
     - 0.99936124
     - -5.1371
     - -0.0087373
     - 0.0051620
   * - 2000
     - 0.99931410
     - 5.3097
     - -0.0030500
     - 0.0563830
   * - 2500
     - 0.99970795
     - 89.2081
     - 0.0332547
     - 0.3377936
   * - 3000
     - 1.00142881
     - 421.7170
     - 0.1524437
     - 1.0870041
   * - 4000
     - 1.01449296
     - 2923.2556
     - 0.8504502
     - 4.2050547

At 4000 K, a4Pi supplies 1.57709 percent and B2Pi 0.04834 percent of the retained
Q. A ground-electronic-only control gives K/current = 0.9980031 and Cp lower by
5.02965 J/mol/K than the full candidate. Thus the high-temperature change mainly
comes from the excited manifolds, especially the unrefined a4Pi contribution.
Its physical inclusion is useful, but reproducing the finite sum cannot establish
the accuracy of its electronic excitation energy. The original paper PF and the
full distributed sum agree within 1 ppm at the archived temperatures.

Fixed-network response
----------------------

The substitution changes only P1S1 on the named ``curated-cos-v1`` network.
Species, elements, formula matrix, all baseline potentials and budgets match
before reusing 75 archived baselines. There are 75 fresh ion-inclusive candidate
endpoints with 537 species, five temperatures from 1500 to 4000 K, five pressures
from 1e-6 to 100 bar and C/O = 0.5888436554, 1.0 and 1.5. Six tighter cold endpoints,
five interpolation-path solves and four standalone FastChem comparisons complete
the numerical controls.

PS mole-fraction ratios range from 0.99931410 to 1.01449296. The largest absolute
change is at 1500 K, 1e-4 bar and reference C/O: x(PS) changes from
9.26510054e-8 to 9.26020337e-8, a difference of -4.89718e-11. Its phosphorus share
changes from 21.01655 to 21.00544 percent, or -0.01111 percentage points. P2, atomic
P and PO take most of the released phosphorus. The 1.45 percent relative increase
at 4000 K concerns much smaller PS amounts; it is not a 1.45 percent shift of the
bulk phosphorus reservoir.

Maximum selected cold-control variability is 9.24e-14 in PS ln(x) and 3.39e-14 in
absolute x across the whole network. The largest PS response exceeds that global
absolute variability by over three orders of magnitude. Trace background changes
below the control scale are not interpreted as meaningful. The maximum residuals
are 1.46e-6 for all elements, 1.41e-10 for H/C/O/P/S, 1.95e-14 for relative charge,
and 1.65e-12 for logarithmic mass action. The independent solver differs by at
most 5.68e-6 in log10(x) and satisfies its recorded conservation checks.

Feasibility-corrected, same-model total-G ordering is numerically resolved in
19 of 75 conditions, including the largest PS response. The remaining signs are
below the displayed floating-point scale. Comparing absolute G values between
different thermodynamic models does not establish scientific preference.

Decision, limits and replay
---------------------------

The evidence supports a scoped optional POPS adoption with the retained energy:
the continuous model includes explicit excited-state thermodynamics and its
small abundance response is numerically resolved where PS matters most in this
grid. The limited impact reduces urgency; it does not prove the old heat capacity
is correct. The a4Pi uncertainty prevents a claim of decisively improved accuracy,
so the value of adopting it is more transparent physical modeling, not a new
precision standard. A future a-state comparison could strengthen this choice;
it is not a prerequisite for merging this completed candidate evaluation.

The scope is gas-only equilibrium, not condensation, photochemistry, transport,
opacity or observational validation. The published D0 margin is retained without
being treated as an independent Gaussian covariance. Complete physical uncertainty,
state completeness and accuracy for arbitrary atmospheres remain unknown.

Run ``python mceg/reviews/ps_candidate.py --build`` for the 2501-node independent
thermal replay and scalar/JAX/derivative checks. Fresh substitutions use
``python mceg/reviews/ps_equilibrium_impact.py --output-dir results/ps-candidate-new --fastchem-executable /path/to/fastchem``.
The archived endpoints replay offline with
``pytest tests/model_evidence/ps_candidate_test.py tests/model_evidence/ps_equilibrium_impact_test.py``.
The packets are under ``tests/data/validation/ps_candidate/20261106`` and
``tests/data/validation/ps_equilibrium_impact/20261106``; the separate MCEG record
is ``mceg/contracts/ps_candidate.yaml``. The source review and production data
remain byte-for-byte intact.
