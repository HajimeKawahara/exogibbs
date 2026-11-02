C3 source investigation, 2026-11-02
=================================

This packet reviews neutral linear gaseous C3 through the balanced formation
reaction 3 C(g) -> C3(g), standard pressure 1 bar. It is separate from cyclic
C3 isomers and the fragmentation reaction C3 -> C2 + C. No production
coefficients or dataset identities change.

``literature.json`` records original material, quoted ancestry and access gaps;
``sources.json`` identifies archived source bytes. The packet contains official
JANAF tables and selected printed pages, official NASA records, an explicit
ATcT 1.172 numerical transcription and selected ExoMol AtLast inputs.

``comparison.json`` reproduces the frozen four-node source comparison and
separates balanced energy, atomic references and thermal contributions. It
checks printed/online agreement, graphite-reference closure, current-fit and
caloric residuals, NASA interval continuity and original-R header consistency.
The ATcT comparison uses both C3 and C from the same version; quoted 95%
confidence limits are not treated as independent one-sigma errors.

JANAF already includes the low-frequency bend, but its high bending levels
use an empirical stiffening assumption. Modern rovibrational diagnostics
therefore matter independently of the energy preference. Quoted historical
entropy measurements remain secondary evidence, with an explicit conditional
1 atm to 1 bar conversion and no invented measurement uncertainty.

The downloaded AtLast definition declares spin weights of 6, inconsistent
with the original 12C3 paper, low-temperature normalization and inspected
state weights. No division by 6 is applied. The partition file also differs
from the direct state sum; neither a version-mix explanation nor the cause
of this mismatch is asserted. Ground-state coverage and stable derivatives
do not certify complete high-temperature caloric functions.

``state_sum_checks.json`` records a separate fresh execution over all
2,166,503 states, the external compressed-file hash, command, environment,
implementation hash, energy moments and included population near finite bounds.
The large external state file is not vendored. Recompute this optional
calculation only when those exact bytes are already available locally::

   PYTHONPATH=src python mceg/reviews/c3_source_review.py \
     --states-file /path/to/12C3__AtLast.states.bz2 \
     --states-output results/c3-state-replay/state_sum_checks.json

Ordinary source replay uses the compact archived diagnostics and performs
no downloads or full external-state re-sum. Write new output and a new receipt
rather than overwriting the archive::

   PYTHONPATH=src python mceg/reviews/c3_source_review.py \
     --output results/c3-source-replay/comparison.json \
     --receipt results/c3-source-replay/receipt.json

``receipt.json`` identifies the selected source/implementation bytes, output
and actual execution environment. New timestamps and paths are expected in
replayed receipts; numerical results are tested independently. Offline checks::

   PYTHONPATH=src pytest tests/model_evidence/c3_source_review_test.py

The separate ``c3_equilibrium_impact/20261102`` packet contains the raw-NASA
C3-only sensitivity, not a prospective ATcT/modern-thermal adoption. Ordinary
checks require neither MCEG nor an independent FastChem executable.
