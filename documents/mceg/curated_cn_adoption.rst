CN curated energy adoption
==========================

``curated-cn-v1`` adopts the evaluated ATcT 1.228 energy balance recommended
in :doc:`cn_candidate`, on the unchanged ``curated-cos-v1`` base. Its named
factory is ``exogibbs.presets.curated_cn.chemsetup``. The cumulative default
switch follows with the separately verified PS and TiO2 adoptions.

Why this update is preferred
----------------------------

Same-version gas-atom and molecular Hf(0) values give
711.370 + 470.655 - 436.670 = 745.355 kJ/mol for C + N - CN,
compared with the inherited physical JANAF D0 of 750.156 kJ/mol.
The modern evaluated network supports replacing that central energy.
The independent Trihybrid thermal review supports retaining the existing
thermal shape over 1500--4000 K. These are separate scientific decisions:
solver agreement verifies implementation, not source preference.

Only delta D0 = -4801 J/mol is applied. Reaction G and H increase by
4801 J/mol; S and Cp are unchanged. The literal fit and its formal
78.141 J/mol intercept residual are retained. Resetting the formal intercept,
adding ZPE again, or changing isotope/spin conventions would introduce an
additional unsupported change. The small packaged JSON pins both the reviewed
candidate and immediate base; the builder rejects changed source bytes.

Production verification
-----------------------

The named production factory was freshly solved at 150 conditions: 75 each
for 537-species ion-inclusive and 421-species neutral networks, spanning
1500--4000 K, 1e-6--100 bar and three C/O ratios. Six tighter independent
candidate controls and two standalone FastChem comparisons pass. Dense
production/candidate h agreement is within 2.14e-14; CN control ln(x) agreement
is within 1.07e-13, and FastChem agreement within 1.05e-7 dex.
All non-CN potentials equal the immediate COS base exactly.

CN mole-fraction ratios span 0.680483--0.872136. At 2500 K, 1e-4 bar and
C/O = 1.5, the ion-inclusive CN abundance decreases from 5.98337e-6 to
4.81214e-6, and its nitrogen inventory share from 8.68248 to 6.98291 percent.
N2 and HCN receive redistributed nitrogen. Both networks give the same
interpretation. The ion-inclusive production reproduces the earlier CN-only
candidate because its background is unchanged; no earlier neutral candidate
execution is claimed.

H/C/N/O/S relative inventory closure is below 2.95e-12, charge closure below
1.36e-14 and mass-action residual below 1.60e-12. The maximum relative
residual including trace-element inventories is 1.46e-6 and is recorded
separately. The strict controls preserve these scoped numerical limits.

Reproduce the compact model and archived checks with::

   python mceg/reviews/build_cn_adoption.py
   pytest tests/unittests/presets/curated_cn_test.py tests/model_evidence/curated_cn_adoption_test.py

The actual 150-condition execution command and input/output hashes are in
``tests/data/validation/curated_cn_adoption/20261106/receipt.json``.
``mceg/contracts/curated_cn_adoption.yaml`` records supported scoped selection
and production adoption separately from unknown full physical accuracy.
Correlated reaction uncertainty, thermal-state and isotope completeness,
condensation and planetary observational validation remain follow-ups.
A molecular marginal uncertainty is not the complete reaction uncertainty.
These limits do not invalidate the authorized bounded energy improvement.
