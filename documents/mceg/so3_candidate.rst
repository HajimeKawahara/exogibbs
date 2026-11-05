SO3 retained-thermal energy candidate
=====================================

The optional ATcT energy refresh is now evaluated as an actual finite
substitution. Its effect is small: SO3 mole fractions decrease by
0.159--0.424 percent on the declared hot ideal-gas grid. A consistent
ATcT 1.228 central-energy refresh is scientifically reasonable, but this
correlated version difference does not establish that the current value is
erroneous or demand an urgent production correction. Curated remains unchanged.
The current NASA thermal functions are retained exactly; UYT2 is not
extrapolated above its distributed partition-function range.

Balanced source energy
----------------------

This follows :doc:`so3_source_review`; its original source packet and
historical MCEG assessments remain intact. The existing production reaction
already uses NASA thermals and ATcT 1.172 energy. Ground-state gas-atom and
molecular Hf(0), in kJ/mol, are:

.. list-table:: Both sides of the same atomization reaction
   :header-rows: 1

   * - ATcT version
     - O(g)
     - S(g)
     - SO3(g)
     - 3 O + S - SO3
   * - 1.172
     - 246.844
     - 276.830
     - -389.750
     - 1407.112
   * - 1.228
     - 246.843
     - 276.850
     - -389.680
     - 1407.059

The balanced D0 change is -53 J/mol: the atom contribution changes by +17,
while the molecular Hf(0) changes by +70 J/mol. Applying a molecular-only
+70 J/mol shift would therefore be inconsistent. Published millikJ values
are balanced as integral J/mol to avoid Python summation-version differences.

The original adopted NASA9 reaction is preserved. The only modification is
Gformation_new = Gformation_current +53 J/mol, so H shifts by the same
amount while S and Cp are identical. The ATcT energy correction is applied
once, using the SI gas constant 8.31446261815324 J/mol/K. Independent
reconstruction from original NASA molecular/atomic records removes the native
NASA energy using its own R=8.314510 before inserting the new balanced energy.
No finite-temperature fit is extrapolated to zero K. The candidate is
continuous and restricted to 1500--4000 K and one-bar standard pressure.
Existing isotope, electronic, nuclear-spin and symmetry conventions remain.

The displayed O/S energies and the gaseous SO3 row in the
`official ATcT 1.228 table
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
were rechecked. The new access record distinguishes that inspection from
recovering every historical network input. The two versions are correlated
network evaluations; molecular uncertainty is not an independent reaction
uncertainty, and no significance claim follows from this small shift.

Finite response and numerical limits
------------------------------------

The network has 537 ion-inclusive species and fixed element inventories.
Before reuse, all 75 archived ``curated-cos-v1`` baselines are checked against
the named factory for species, elements, stoichiometry, budgets and every
chemical potential. Only SO3 changes. The run solves 75 new cold candidate
endpoints on five temperatures (1500/2000/2500/3000/4000 K), five pressures
(1e-6 to 100 bar) and C/O=0.5888436554, 1 and 1.5.

The candidate/current mole-fraction ratio is 0.995759394--0.998407660.
At the largest target/inventory response (2500 K, 0.01 bar, reference C/O),
x(SO3) changes from 3.81086511e-20 to 3.80116065e-20. Its share of the total
sulfur inventory changes from 1.87724797e-15 to 1.87246751e-15. These are
fractions, not percentages. The largest raw background change is 6.11e-15
in x, below the tighter-control variation of 3.38e-14, so no resolved
bulk-gas redistribution is claimed.

All endpoints converge at tolerance 5e-14. Six fresh baseline/candidate cold
controls use 2e-14, and a five-point finite path reaches the candidate smoothly.
The largest target ln(x) tighter-control difference is 1.72e-12. H/C/O/S
relative closure is below 2.95e-12, charge closure below 1.96e-14 and
mass-action residual below 1.65e-12 in natural logarithm. The maximum
relative residual over all trace-element inventories is 1.46e-6, not a
universal 1e-12 conservation result.

All 75 same-model total-Gibbs comparisons remain unresolved at the recorded
floating-point scale. The maximum isolated target Gibbs contribution is
5.82e-23 versus a displayed total comparison scale as large as 1.94e-16.
The stable target logarithmic response must not be interpreted as a resolved
physical total-Gibbs improvement. Four standalone FastChem comparisons at two
exportable conditions agree within 8.57e-9 dex across all species. They
independently check the supplied thermodynamics and numerical solver, without
establishing physical source accuracy.

Decision and reproduction
-------------------------

The previous optional same-thermal calculation is now complete. The new
central value is a suitable source refresh for a separately scoped adoption;
retaining the present coefficient pending such a coordinated update is also
proportionate. The very small response is not a reason to promote a less
supported thermal replacement. UYT2 high-temperature completeness, full
reaction covariance, and planetary disequilibrium/condensation remain outside
this result. No new legacy full text is required merely to merge this evidence.

``mceg/reviews/so3_candidate.py --build`` reproduces 2501-temperature checks.
``so3_equilibrium_impact.py --output-dir <new-directory>`` with
``--fastchem-executable`` replays the actual substitutions. Candidate and
impact packets have separate execution receipts; the old source packet is
unchanged. ``mceg/contracts/so3_candidate.yaml`` separates numerical construction,
finite response and scoped preference from unresolved universal Gibbs
resolution and complete physical accuracy. No neutral-network solve is claimed.
