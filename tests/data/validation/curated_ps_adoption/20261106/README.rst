PS production adoption on the CN base
=====================================

This packet records the authorized retained-energy POPS thermal update as
``curated-ps-v1`` on the immutable ``curated-cn-v1`` base. The public default
switch is coordinated with the subsequent TiO2 adoption; this packet verifies
the named PS factory independently.

``fit_validation.json`` and ``fit_receipt.json`` cover the 19-coefficient
representation, direct state moments, degree convergence, 2501 integer and
257 off-grid temperatures, and scalar/JAX thermodynamic derivatives.

``ions.npz`` and ``neutral.npz`` contain 150 newly solved production endpoints
and six tighter direct-candidate controls. Species, budgets, formula matrix
and all baseline potentials match the preceding CN archive. The production
receipt also records two standalone FastChem comparisons. Differences from
the older PS-only candidate include the CN background change; they are not
claimed as errors of the compact mathematical approximation.

The separate receipts pin actual inputs and outputs. No previous candidate,
source-review packet or historical production data are overwritten. Complete
excited-state, isotope, energy-covariance and atmospheric accuracy remain
unknown; see ``documents/mceg/ps_curated_adoption.rst``.
