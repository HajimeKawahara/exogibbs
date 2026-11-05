N2O3 candidate: thermal change reverses the energy-only response
================================================================

The consistent NASA-thermal/ATcT-1.228 candidate is numerically ready for an
adoption decision. Scientifically, it is the preferred provisional replacement
for the inherited JANAF-based fit: it combines an explicitly identified newer
thermal evaluation with balanced modern evaluated energy. The factor-of-three
abundance correction is a model comparison, not an experimentally established
accuracy improvement. Original Gurvich thermal-input recovery and the known
low-temperature dissociation discrepancy remain limitations.
No production coefficients are changed by this PR.

Construction and scientific boundaries
--------------------------------------

The primary candidate uses the single NASA 1000--6000 K interval, restricted to
1500--4000 K. Molecular minus neutral-atom coefficients form one continuous NASA9
reaction, with no internal join or new fit. Native NASA D0=1590.804204 kJ/mol is
removed using R=8.314510 J/mol/K. ATcT 1.228 N=470.655, O=246.843 and asymmetric
ONN(O)O=90.85 kJ/mol Hf(0) give D0=1590.989 kJ/mol, inserted once using SI R.
Dimensionless NASA thermal functions remain intact; caloric properties use SI R.
No ONONO conformers, isotope mixture or extra nuclear-spin factor are added.

An energy-only control keeps the actual FastChem4 five-coefficient reaction.
Its offset is -1.404 kJ/mol relative to the documented JANAF molecule-plus-atoms
anchor of 1592.393 kJ/mol. This reference-based offset does not infer a unique
zero-temperature D0 from the finite-range fit. The original fit's atomic basis
has not been fully recovered. At 1500/3000/4000 K, primary formation-K/current
is 2.158415/2.828838/2.966144, while the control gives
0.893530/0.945267/0.958663. The thermal replacement reverses the energy-only sign.

NASA cites Gurvich (1989), while the current JANAF evaluation is dated 1964.
The source review identifies a large thermal entropy difference. Its separate
300 K N2O3 -> NO + NO2 check remains -2.020986 kJ/mol from NASA versus the
ATcT network input -1.84 +/- 0.07; the difference is outside that quoted input
margin. Neither extending the production domain below 1500 K nor using that
ATcT-linked input as independent validation is justified. These qualifications
remain recorded rather than being removed because numerical checks pass.
See :doc:`n2o3_source_review` for source-access and historical details.

Actual finite substitutions
---------------------------

The same 75 temperatures/pressures/C/O cases are evaluated on the 537-species
ion-inclusive and 421-species neutral networks. Species, atomic references and
all element inventories are fixed. The 150 saved curated-cos-v1 baseline
endpoints are reused; 300 candidate/control endpoints, 18 tighter cold controls
and 10 finite-path points are newly solved. The primary tolerance is 5e-14 and
the tighter tolerance 2e-14. The complete run took about 26 seconds in this
execution environment; this is not a performance benchmark.

The primary candidate/current mole-fraction ratio is 2.1584--2.9661 in both
networks; the energy-only control is 0.89353--0.95866. The maximum ion-inclusive
N2O3 mole fraction changes from 1.04124e-26 to 3.08846e-26 at 4000 K, 100 bar,
baseline C/O=0.588844. Nitrogen inventory allocation, including two N atoms,
changes from 1.92592e-22 to 5.71257e-22. This species is a trace constituent
throughout this hot grid; cooler oxidizing atmospheres are not assessed.

The largest raw changes in other species are around 1e-14--3e-14 in mole
fraction, comparable to the tighter-control differences (up to 3.38e-14).
They are unresolved numerical differences, not established redistribution of
bulk gas. The target's logarithmic abundance response is nevertheless stable
to 8.7e-13 in the selected tighter controls and follows the finite mass-action
identity to 8.1e-13. A five-point finite path reaches the endpoint smoothly;
this is an actual substitution rather than a linearized extrapolation.

All new endpoints converge, with N/O conservation residual below 7.1e-13 and
mass-action residual below 1.7e-12. The largest relative residual among all
trace-element inventories is about 1.46e-6, within the explicitly used 1e-5
criterion; it is not reported as a universal 1e-12 conservation result.
Feasibility-corrected Gibbs ordering passes the 5e-14 numerical tolerance.
Because the N2O3 contribution is much smaller, these values do not resolve a
physical Gibbs improvement. Standalone FastChem checks all three models at two
selected cases; maximum absolute log10(mole-fraction) difference is 4.43e-9.

Recommendation
--------------

The primary NASA/ATcT construction is the coherent candidate to carry forward;
the energy-only control is not the preferred correction. Numerical readiness
and the recorded source comparison support provisional adoption, with the hot
thermal accuracy and historical inputs explicitly unresolved. No additional
paper is made a prerequisite for this bounded decision. A broader physical
claim should address the original Gurvich assumptions, conformer scope and
low-temperature dissociation data identified in the source review.

Reproducibility
---------------

``mceg/reviews/n2o3_candidate.py --build`` regenerates the exact candidate
and dense 2501-temperature validation. ``n2o3_equilibrium_impact.py`` runs
new fixed-network endpoints and controls with ``--output-dir`` and optional
``--fastchem-executable``. All archived production data remain unchanged.
The new candidate/impact packets retain their own receipts; the prior source
review and its hashes are preserved. MCEG records supported numerical claims
and a bounded recommendation, with full physical accuracy explicitly unknown.
