# Retained parcel initialization retry

`parcel_initialization_retry.py` provides an optional wrapper around the
existing condensate solver. Its default `enabled=False` delegates exactly one
unchanged call and returns the original result. It changes no physical module,
catalog, standard, composition bound, solver option, or acceptance tolerance.

With `enabled=True`, a returned cold `converged=False` can trigger one retry
using the most recent solver-converged species amounts. The setup object,
caller budget bytes, and pressure standard must match. Donor amounts must be
finite, nonnegative, independently conserved, and have positive gas amounts;
present condensates must remain temperature eligible. Explicit initialization
or support arguments and rainout are excluded. Fugacity callbacks remain
excluded by default.

The donor already uses the local solver's caller gauge. The wrapper passes
`log(gas_n)`, `sum(gas_n)`, and the full condensate amounts without a second
normalization. It supplies no element potential, barrier, fixed support, or
final active-set constraint. The existing solver searches the catalog and
retains all its physical acceptance gates. A failed retry stays failed.

```python
retry = make_previous_parcel_retry(
    original_solve_condensate, enabled=True, record=save_receipt,
)
```

For a fixed nonideal EOS/column context, `allow_nonideal=True` additionally
requires an explicit `nonideal_identity` object. Construct a new wrapper for
each context and bind the physical EOS recipe in the consumer receipt. The
context identity, setup, budget, pressure standard, full solver options and
presence of a fugacity callback must match the donor. An EOS fixed-point loop
may supply a fresh fugacity callback for each iteration. The warm call uses
the current callback and options unchanged; it does not reuse fugacity values,
EOS states or chemical potentials. Only the primitive gas and condensate
amounts become an initializer. The provider's full support search and the
consumer's independent EOS fixed-point, convexity and KKT audits still decide
acceptance.

```python
retry = make_previous_parcel_retry(
    original_solve_condensate, enabled=True, record=save_receipt,
    allow_nonideal=True, nonideal_identity=original_column_parcel_callback,
)
```

`save_receipt` receives a strict-JSON-compatible record of the cold failure,
its raw result and diagnostics, the donor and initial amounts when eligible,
and the warm result or exception. Nonfinite diagnostic floats are explicit
`{"nonfinite_float": "nan"}` (or `inf`/`-inf`) records. A solver exception is
recorded and propagated without a retry. Consumers must persist the receipts
and retain any original failure. Recording a converged donor does not assert
that a consumer's independent parcel or global closure audit passed.

For a column-only recovery, construct a new wrapper per pressure evaluation
and inject it only while calling the returned column parcel callback. Restore
the original solver in `finally`; do not replace source equilibrium or its
insertion/scalar calls. The original atmosphere audit and consumer element,
mass, pressure, and phase proofs remain mandatory. Do not share this mutable
history between concurrent solves.

## Motivation and limited validation

The saved Mg/Si case uses **1.2 times the BSE Mg/Si atomic ratio**, while
holding the combined MgO and SiO2 mass fixed. At source pressure
287.8726991192998 bar, source equilibrium and the basal contact passed, but
column layer 170 failed at 1094.5481443370368 K and 9.33095948907405 bar.
The physical provider was `a5895d081df646e0529d27e46cd2835188d0a925`.

A standalone cold replay reproduced `NORMAL_MAX_ITER` and a failed exact
support polish; the latter exhausted 400 evaluations. A one-point comparison
using only layer 169's accepted species amounts converged with unchanged
options. The independent audit found a maximum element residual of
7.55e-15, gas stationarity 3.41e-13 RT, present-condensate residual 2.28e-13 RT,
and zero absent-condensate violation. These diagnostics establish a useful
initialization for this point, not a recovered pressure root or a completed
composition/grid validation.

The wrapper itself was also exercised with the saved layer-169 donor and
recorded cold failure, followed by exactly one fresh warm solve. That call
converged in 53.41 seconds (peak RSS 1.35 GiB); its unchanged independent audit
passed with element residual 7.99e-15, gas stationarity 7.96e-13 RT,
present-condensate residual 4.55e-13 RT, and zero absent-condensate violation.
The cold failure and successful warm result were both retained in the helper
receipt. The two initial calls in this helper diagnostic replayed saved data;
they were not additional chemistry solves.

The focused tests cover default delegation, unchanged options and caller
gauge across amount scales, exact budget/setup matching, ineligible donors,
bounded failure, exception propagation, and complete diagnostic retention.
