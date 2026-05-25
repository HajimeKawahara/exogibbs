Guarded Retained-Floor Diagnostic Policy
========================================

Scope
-----

The guarded retained-floor policy is an explicit opt-in diagnostic policy for
condensate real-callsite experiments. It is not a default production
initialization rule and it is not an equilibrium acceptance gate.

The policy compares two retained-support candidates for the same case:

``no_floor``
    The retained condensate amounts are updated without applying an absolute
    retained amount floor.

``floor_1e-14``
    The retained condensate amounts are updated with an absolute retained
    amount floor of ``1.0e-14``.

The floor candidate is selected only when it is budget-safe and has a lower
KKT diagnostic residual than the no-floor candidate. Otherwise, the no-floor
candidate is retained.

Public Diagnostic Helpers
-------------------------

The policy surface is intentionally narrow:

``exogibbs.diagnostics.condensate_retained_floor_selector``
    Provides candidate dataclasses and the guarded selector.

``exogibbs.diagnostics.condensate_guarded_retained_floor_policy``
    Provides explicit opt-in policy configuration and policy-gate validation.

Both helpers are explicit-import only. They are not imported by normal
``exogibbs`` or ``exogibbs.presets`` imports.

Allowed Inputs
--------------

The policy may consume only ExoGibbs-native diagnostic metrics:

* selected candidate budget residuals
* selected candidate KKT diagnostic residuals
* selected candidate finite-input flags
* selected candidate labels
* explicit native policy configuration

The policy may not consume FastChem4 public, runtime, or trace values as
constructor inputs.

Guardrails
----------

The policy requires:

* explicit opt-in
* default-off execution
* no production solver behavior change
* no production return signature change
* no presets or defaults wiring
* no FastChem4 import
* no pyfastchem import
* no FastChem4 trace, public, or runtime constructor inputs
* KKT residuals treated as diagnostics, not acceptance gates
* budget non-worsening as a guard

Observed FC4-M1981 to M2020 Status
----------------------------------

The real-callsite policy review and production-adjacent gate recorded:

* case count: ``4``
* budget-safe selected cases: ``4``
* KKT-improved selected cases: ``4``
* finite candidate pairs: ``4``
* floor-selected cases: ``3``
* no-floor-selected cases: ``1``

This supports the guarded policy as a release-candidate experimental
diagnostic surface. It does not authorize default-on production wiring.

Next Use
--------

The next appropriate use is an explicit opt-in production-adjacent experiment
that records the selected candidate and solver-stage diagnostics. The normal
production path must remain unchanged.
