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
`formal_global_insertion_bound_accepted` requires a nonnegative verified
lower bound. `bound_within_requested_tolerance` separately records closure
within the requested numerical tolerance; a small negative bound never
receives the formal flag.

Linear physical-site constraints also tighten boxes and support an affine
lower bound. A numerical LP proposes nonnegative dual multipliers; its
Lagrangian is reevaluated with intervals and minimized on the box. LP solver
tolerances are never a certificate. Stronger bounds replace positive site
entropy with a global quadratic minorant and negative site entropy with its
lower endpoint chord. Interval Hessians and an interval LDL decomposition
verify a convex relaxation before its supporting plane is passed to that
same checked affine bound. The basic interval enclosure remains available
when the stronger relaxation cannot be verified.

For provider-declared equilibrium pure references, the full [0,1] ordering
curve is partitioned into 256 closed subintervals. Each has a verified lower
bound, and an evaluated point supplies an upper witness. Subtracting the
enclosed global minimum retains uncertainty in the reference rather than
silently treating an optimizer result as exact. These pure-reference
partitions are preserved alongside the mixture proof.

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

The runner reevaluates the supplied host with its original native, published
or reconstructed-water liquid model, obtains native standards at the same T/P, and bounds every
declared provider model. It preserves raw input, native receipts, code hashes,
parameter declarations, and the complete proof partition. It does not solve
a new global pressure root. Formal expression stability, finite native
compatibility, native binary error bounds, empirical property validity, and
the Fe/Si/O/H source-alloy stability are distinct assertions.

Reconstructed water requires the complete dry declaration and saved gas
standard. The fresh model expression, standards, component amounts, chemical
potentials, Gibbs energy and basis must match the supplied audit. The core
reference is always the bare selected provider `mu_RT` plus
`log(N_native/(N_native+nH2))`. A source's separate He host correction is
explicitly excluded from this reference and recorded as such. For a He-bearing
source, these bounds must be rebased from that same bare reference to the
common elemental plane; adding the He correction only during rebasing would
be inconsistent. They are not independently labeled as the actual He-bearing
host's self-tangent bounds. The common-plane converter verifies the recorded
reference and keeps signed reaction coefficients throughout.
