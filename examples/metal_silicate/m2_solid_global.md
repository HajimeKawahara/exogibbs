# Global insertion bounds for declared mineral models

`m2_solid_global.py` consumes an ExoEOS site expression, fresh native pure
standards, and the actual native or published liquid host potential. It
subtracts the host tangent plane, including the declared dissolved-H2
dilution, and encloses the result over the full physical site domain with
outward-rounded Decimal interval arithmetic. The provider owns all physical
expressions, stoichiometry, ordering coordinates, and domain restrictions.

All admissible ordering states are covered, including those missed by a local
native ordering solver. Empty-site entropy uses its continuous energy limit.
Positive singular barriers retain their divergent endpoint limit. Pure
ordering reference energies are subtracted as enclosing intervals. A finite
node budget that leaves any negative box unresolved cannot certify stability.
The output preserves every accepted, excluded, and unresolved box.

The canonical oxide-to-host reaction is solved with exact rational
elimination. An absent host potential is never filled with zero. Restrictions
such as an Fe-only native alloy or the Fe/Mg/Ca olivine face require absence
of the excluded elements in the host budget. Signed endmember coordinates
are not clipped to a nonnegative simplex.

Run from the repository root:

```sh
python examples/metal_silicate/run_m2_solid_global.py \
  --saved-physical-audit /path/to/physical_audit.json \
  --exoeos-checkout /path/to/exoeos --runtime /path/to/alphamelts/runtime \
  --python /path/to/native/worker/python --output-directory /path/to/new/results
```

The runner reevaluates the supplied host with its original native/published
liquid model, obtains native standards at the same T/P, and bounds every
declared provider model. It preserves raw input, native receipts, code hashes,
parameter declarations, and the complete proof partition. It does not solve
a new global pressure root. Formal expression stability, finite native
compatibility, native binary error bounds, empirical property validity, and
the Fe/Si/O/H source-alloy stability are distinct assertions.
