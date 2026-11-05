COS energy candidate, 2026-11-05
================================

``candidate.json`` freezes the actual curated JANAF piecewise quintic molecular
thermal fit and NASA atomic background. The only change is the balanced
ATcT 1.228 C/O/S/COS dissociation energy, from 1371241 to 1376853 J/mol.
The existing reaction h is shifted by -5612/(R T), once. No partition-table
join, thermal refit, isotope change or production adoption occurs.

``validation.json`` retains the 2501-point production replay, scalar/JAX
thermodynamic identities and internal-knot continuity. ``receipt.json`` records
the actual execution and source/output hashes. Physical thermal uncertainty,
ATcT covariance and atmospheric accuracy are outside this numerical claim.
