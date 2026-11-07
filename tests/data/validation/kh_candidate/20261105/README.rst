KH spectroscopic-energy candidate (2026-11-05)
=============================================

This offline packet preserves the current five-coefficient KH formation fit
and shifts only D0 from 180390 to 170857.8386391254 J/mol. Entropy and heat
capacity are unchanged. NASA molecular/atomic thermal functions normalized
to the same spectroscopic D0 are a separate bounded reference control.
Neither model is adopted in production.

Rebuild with ``python mceg/reviews/kh_candidate.py --build``. The actual
invocation, input hashes and output hashes are retained in ``receipt.json``.
``validation.json`` records dense source reconstruction and float64 analytic/
JAX derivatives. The inspected 1988 abstract gives D0=14282.6 cm-1, but its
+/-0.6 cm-1 error belongs to De. No complete D0 uncertainty is inferred.

The source packet, documents/mceg/kh_candidate.rst and the candidate MCEG
contract record full source and domain limits. Run
``pytest tests/model_evidence/kh_candidate_test.py`` offline.
