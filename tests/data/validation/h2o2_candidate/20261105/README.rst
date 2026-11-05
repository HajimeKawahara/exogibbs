H2O2 candidate, 2026-11-05
==========================

This is a new candidate calculation packet, not a production adoption.
``receipt.json`` records the actual executed code, source inputs and outputs.
Source-only historical receipts remain unchanged. Numerical diagnostics are
offline; no external MCEG installation is needed for the regression tests.

For equilibrium impact, the named curated-cos-v1 archive supplies 150 reused
baseline endpoints, while the diagnostics explicitly count all newly solved
candidate, tighter-cold and finite-path cases. Independent FastChem checks
share h, stoichiometry and budgets; they check solvers rather than physical
source accuracy. Tiny non-target responses may be numerical noise.

See ``documents/mceg/h2o2_candidate.rst`` for interpretation and limits.
