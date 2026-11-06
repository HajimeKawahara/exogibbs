P4S3 gas identity and modern energy-source review
=================================================

Retain production for this source review, then evaluate a 2019 cycle-based
H298 energy candidate. JANAF explicitly estimates all P4S3 gas data, so its
agreement with a fitted implementation is not scientific confirmation.
A modern energy sensitivity is available independently of constructing a
new cage thermal function. No curated change or fresh equilibrium solve
is made here.

Exact source coverage
---------------------

The official JANAF P-017 table and printed page 1833 identify neutral P4S3
ideal gas, distinct from P-014/P-015/P-016 condensed phases. The December 1960
record supplies neither a structural-isomer model nor a molecular zero-K
anchor. Both its energy and thermal sections state that the data are estimated.
The later isolated-cage calculation therefore needs its explicit molecular
model, rather than an assertion that the old table already resolved all cages.

The complete official NASA file contains no matching neutral gas record.
NASA comparisons stay null and the frozen 93-species numerical screen remains
unchanged. No solid, different P4Sn stoichiometry or arbitrary isomer is used
as a substitute. This resolves the appendix phase/formula coverage while
preserving its thermal-model limitation.

The old Hf298=-151.042 kJ/mol, S298=319.265 J/mol/K and constant
Cp=154.808 J/mol/K reconstruct the full table above 298.15 K. Its H increment
is Cp*(T-298.15) and entropy increment is Cp*ln(T/298.15). This constant Cp
is 3.16679 J/mol/K below the nonlinear seven-atom harmonic high-temperature
limit 19R; that diagnostic is not a physical error estimate. Extrapolating
the constant Cp to zero would give an unphysical entropy divergence and is
not used to invent Hf0 or physical D0. Source decomposition is explicitly
H298-based, with balanced neutral P/S atomic references.

Modern primary evidence and its distinct alternatives
-----------------------------------------------------

The complete author manuscript of `Kroeger and Karton (2019)
<https://doi.org/10.1007/s11224-019-01352-7>`_ favors Hf298=-94.5 kJ/mol
from P4S3 + 0.5 S2 -> 2 P2S2, combining W4 fragment anchors and a W3lite-F12
reaction energy. The three method values for that cycle are -93.8, -95.3 and
-94.5 kJ/mol. Cycles 3-9 span -97.2 to -92.4 kJ/mol; this spread is not a
confidence interval. The paper's direct atomization route gives
Hf298=-103.9 kJ/mol and TAE0=2182.4 +/-6.4 kJ/mol, but is explicitly less
converged. That TAE uncertainty must not be attached to the preferred cycle.
Its B3LYP-D3/Def2-TZVPP thermal calculation uses harmonic frequencies and
0.99-scaled zero-point energies; the detailed frequency/geometry supplement
was not acquired. A new thermal candidate remains separate.

The paper uses CODATA elemental heat contents. The official `reference table
<https://cccbdb.nist.gov/refstatex.asp>`_ specifies white crystalline P
(H298-H0=5.360 kJ/mol) and rhombic S (4.412 kJ/mol). The selected JANAF atomic
tables independently imply those same increments: atom heat content minus
its Hf298-Hf0 difference. Its atomic formation energies and the theory's
atomic anchors are not identical; each ledger is kept separate. No red/white
phosphorus phase correction or condensed-to-gas conversion is added to the
reported isolated-gas formation anchor.

.. list-table:: Source and bounded H298-energy sensitivities
   :header-rows: 1

   * - T (K)
     - Current / JANAF
     - Preferred cycle energy factor
     - Atomization-route energy factor
   * - 1500
     - 1.01426587
     - 0.01074165
     - 0.02282453
   * - 2000
     - 1.00422976
     - 0.03336594
     - 0.05872207
   * - 3000
     - 0.98917731
     - 0.10364194
     - 0.15107788
   * - 4000
     - 0.99893712
     - 0.18266345
     - 0.24232637

The preferred shift is +56.542 kJ/mol; the atomization-route control is
+47.142 kJ/mol. These factors are exp(-delta-H298/(RT)) with the current
thermal shape, atomic gauge and fit residual held fixed. G and H shift
equally; S and Cp are unchanged. The common-JANAF atomization enthalpy at
298 K changes from 2247.542 to 2191.000 kJ/mol for the preferred alternative.
This is **not** a zero-K D0 or a finite abundance response. No extrapolated
modern thermal function is packaged.

The packet supplies molecular/atomic/formation G/H/S/Cp, H298 decomposition
closure, constant-Cp reconstruction, both source energy sensitivities and
reference-state checks. SI R and a 1-bar gas standard state are explicit.
A finite-network energy candidate can proceed from this evidence; obtaining
every historical combustion paper is not a prerequisite. A production
recommendation should evaluate that response and document the selected cycle
and its remaining uncertainty, with thermal refinement reviewed separately.

Current importance and replay
-----------------------------

All 75 curated-tio2-v1 archived endpoints converge with 537 species. The
maximum phosphorus share is 4.32940916e-23% at
1500 K, 1 bar and
C/O=0.588844; maximum mole fraction is 4.77545111e-32.
Thus this candidate has lower immediate hot H-rich importance than HCP,
without establishing a general bound for other compositions or condensation.
Complete physical uncertainty and observational abundance accuracy are unknown.

Run ``python mceg/reviews/p4s3_source_review.py`` and
``pytest tests/model_evidence/p4s3_source_review_test.py``. The packet is
``tests/data/validation/p4s3_source_review/20261106`` and contract is
``mceg/contracts/p4s3_source_review.yaml``. Inputs use historical hash resolution;
outputs are directly pinned. The source review is independently mergeable,
with candidate construction/adoption remaining separate decisions.
