CN retained-thermal ATcT energy candidate
=========================================

The same-thermal calculation recommended in :doc:`cn_source_review` is now
complete. The tested energy update is scientifically preferable to the older
JANAF central energy for a scoped curated adoption. On the fixed hot gas grid,
CN mole fractions decrease by 12.79--31.95 percent. The largest inventory
response reduces the fraction of nitrogen held in CN from 8.6825 to 6.9829
percent. This is a material redistribution, and supports prioritizing CN.
This evidence PR leaves curated unchanged; adoption remains a separate step.

The recommendation concerns the energy correction and the tested domain.
It does not require replacing the supported thermal model, obtaining every
historical source paper, or claiming universal physical accuracy. The previous
source packet and its historical MCEG assessments remain unchanged.

Energy correction and retained thermal model
--------------------------------------------

The current CN entry is an inherited five-coefficient JANAF-derived FastChem
fit. The candidate applies the difference between two **physical atom-balanced
source energies**, while retaining every literal fit coefficient:

.. math::

   \ln K_{\mathrm{new}}(T) = \ln K_{\mathrm{current}}(T)
   + \frac{D_{0,\mathrm{ATcT}}-D_{0,\mathrm{JANAF}}}{RT}.

Official JANAF molecular and atomic tables give D0 = 750.156 kJ/mol.
ATcT 1.228 gas-phase Hf(0) values are 711.370 kJ/mol for C, 470.655 for N,
and 436.670 for CN. Their same-version balance is
711.370 + 470.655 - 436.670 = 745.355 kJ/mol. Thus delta D0 = -4801 J/mol,
and the reaction G and H both increase by 4801 J/mol. S and Cp are unchanged.
The ground-level atomization energy already includes ZPE; adding another
zero-point term would double count it. One-bar normalization and all existing
isotope, electronic, nuclear-spin and symmetry conventions are retained.

The physical source energy is not confused with a fitted intercept. The
current fit's formal R*a is 750234.141 J/mol, 78.141 J/mol above JANAF's
physical D0. The fit is not evaluated at zero temperature, and its original
weights and exact construction remain incompletely recovered. This candidate
preserves that residual instead of forcing its intercept to 745355 J/mol.
At source knots within 1500--4000 K, the inherited current/JANAF ln K residual
is at most 0.002286 and is preserved after the energy shift to below 1e-14.
A different direct intercept reset would silently add another -78.141 J/mol
correction and would not be this isolated source-energy experiment.

The continuous candidate uses R = 8.31446261815324 J/mol/K and is bounded to
1500--4000 K. Its K/current ratios at 1500, 3000 and 4000 K are 0.680483,
0.824914 and 0.865579. The earlier independent Trihybrid state-sum review
supports the inherited thermal shape in this interval: molecular Q, H-H0,
S and Cp agree closely with NASA, and the JANAF/NASA energy-aligned reaction
also agrees closely. It does not establish missing-state or isotope-mixture
uncertainty bounds. No new thermal fit or unbounded partition extrapolation
is introduced here.

The official ATcT pages for
`CN <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=79>`_,
`C <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=21>`_
and `N <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=25>`_
were rechecked on 2026-11-06 against the frozen source packet. The evaluated
network provides the basis for preferring the modern central energy; numerical
agreement alone does not do so. Marginal ATcT uncertainties are correlated and
are not combined as independent errors. Complete reaction covariance is
unavailable, so this update is not a statistical significance claim.

Finite substitution and numerical checks
----------------------------------------

Only C1N1 changes on the named ``curated-cos-v1`` network: the same 537
ion-inclusive species, elements, formula matrix, budgets and every baseline
potential are verified before reusing the 75 archived baseline endpoints.
There are 75 fresh cold candidate solves: five temperatures
(1500/2000/2500/3000/4000 K), five pressures (1e-6--100 bar), and three C/O
ratios (0.5888436554, 1, 1.5). No neutral-network solve is claimed.

The candidate/current CN mole-fraction ratio spans 0.680483491--0.872135149.
At 2500 K, 1e-4 bar and C/O = 1.5, x(CN) falls from 5.98336882e-6 to
4.81214066e-6, a decrease of 19.5747 percent and 1.17122816e-6 in x.
The nitrogen share changes from 0.08682482 to 0.06982909 (fractions),
a decrease of 1.69957 percentage points. N2 gains nitrogen at that same
condition: its mole fraction rises 2.03259 percent, and its nitrogen share
increases from 69.3583 to 70.7680 percent. The full packet records CN,
HCN/HNC, N2, atomic N/C and other background responses separately.

The abundance ratio differs from the isolated K ratio because atoms and
competing molecules re-equilibrate. Independent mass-action replay verifies
that delta ln x(CN) = delta ln x(C) + delta ln x(N) - delta h(CN), without
trace concentration floors. The largest absolute network change is CN
itself. Small background shifts near numerical control variation are not
interpreted as resolved physical effects.

All endpoints converge at tolerance 5e-14. Six fresh baseline/candidate cold
controls use 2e-14, and five finite interpolation points connect the baseline
to the candidate at the largest inventory response. The largest CN ln(x)
control difference is 1.50e-13, and the largest absolute x control difference
is 3.38e-14. Across the stored endpoints, H/C/N/O/S relative inventory closure
is below 2.95e-12, charge closure below 1.96e-14, and the natural-log
mass-action residual below 1.65e-12. The maximum relative residual over all
trace-element inventories is 1.46e-6; it is not a universal 1e-12 result.

Both feasibility-corrected same-model total-Gibbs exchange signs exceed the
displayed floating-point scale in 67 of 75 conditions. Eight conditions remain
unresolved at that scale and are retained explicitly. This diagnostic is not
a rigorous error bound; comparing absolute G across the two different models
would not establish physical preference. Four standalone FastChem comparisons
at two exportable conditions agree within 1.05e-7 dex across all species.
They independently check the supplied potentials and numerical response,
not the physical accuracy of the energy source.

Reproduction and limits
-----------------------

Replay the continuous candidate and offline archived checks with::

   python mceg/reviews/cn_candidate.py --build
   pytest tests/model_evidence/cn_candidate_test.py tests/model_evidence/cn_equilibrium_impact_test.py

The candidate replay checks 2501 temperatures, independently reconstructs the
literal current fit, differentiates the same potential with JAX, and verifies
the exact G/H shift and unchanged S/Cp. Candidate/source residuals and physical
versus formal-fit energy anchors are explicit in the frozen JSON.

``cn_equilibrium_impact.py --output-dir <new-directory>`` with
``--fastchem-executable`` reruns the finite substitutions. The candidate and
impact packets have separate execution receipts. The MCEG contract
``mceg/contracts/cn_candidate.yaml`` separates continuous thermodynamics,
finite substitution and the scoped adoption recommendation from unknown
universal Gibbs resolution and complete physical accuracy.

The result is a fixed element-aggregated ideal-gas experiment, not a
condensation-inclusive or disequilibrium planetary prediction. Complete
thermal-state, isotope-mixture and correlated energy uncertainty remain
unquantified. These limits are recorded as MCEG follow-ups and do not require
an indefinite source-recovery campaign before merging this evidence review.
