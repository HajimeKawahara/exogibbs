N2O3 candidate, 2026-11-05
==========================

This is a new candidate calculation packet, not a production adoption.
``receipt.json`` records the actual executed code, source inputs and outputs.
Source-only historical receipts remain unchanged. Numerical diagnostics are
offline; no external MCEG installation is needed for the regression tests.

``dissociation_inputs.json`` pins a separate 300 K source-cycle diagnostic:
all three molecules use ATcT 1.228 energies and original NASA low-temperature
thermal intervals. Its agreement with an ATcT network input is internal
consistency, not independent validation or an extension of the candidate's
1500--4000 K domain. Original source-review artifacts remain unchanged.

For equilibrium impact, the named curated-cos-v1 archive supplies 150 reused
baseline endpoints, while the diagnostics explicitly count all newly solved
candidate, tighter-cold and finite-path cases. Independent FastChem checks
share h, stoichiometry and budgets; they check solvers rather than physical
source accuracy. Tiny non-target responses may be numerical noise.

See ``documents/mceg/n2o3_candidate.rst`` for interpretation and limits.
