Curated gas v2 production evidence
==================================

This packet validates the actual named ``curated_v2`` factory against the
fixed ``curated_ch2`` baseline: 27 selected reviews update 28 unique molecular
species. ``selection.json`` pins the selected frozen candidates and exact
variants, including coordinated HNCO/NCO, NO2/N2O4 and cis/trans-HNO2.

``ions.npz`` and ``neutral.npz`` contain 75 fresh cold production endpoints
each, six tighter independently assembled references in total, and ten
additional joint-path endpoints. ``summary.json`` contains all 28 cumulative
response ranges, every-element closure and redistribution, dense identity,
Gibbs diagnostics, and two independent FastChem comparisons. ``diagnostics.json``
and ``controls.json`` retain actual convergence checks. The 150 baseline
endpoints are reused once, not counted as fresh.

``receipt.json`` freezes the actual execution inputs before endpoint solves
and rejects changes before completion. It also pins the final output bytes.
Its numerical outputs and old source/candidate packets remain immutable.
``default_validation.json`` separately checks exact public-alias binding,
without repeating equilibrium solves. ``execution_note.json`` explains the
separate initial passed execution and final rerun after an import-time wording
ambiguity; its counts must not be added to final-evidence counts.

The standard state is one bar and the reviewed interval is 1500-4000 K.
Numerical agreement establishes execution consistency, not complete physical
uncertainty or atmospheric accuracy. The scientific choices and limitations
are documented in ``documents/mceg/curated_gas_v2_adoption.rst``.
