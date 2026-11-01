Gas-source comparison: baseline-001
==================================

This immutable screening snapshot preserves the 1500 K table discussed in
Issue #308 before further source investigations. It does not change the
production preset or establish physical source accuracy.

* ``inputs.json`` freezes the selected JANAF and NASA molecular/atomic records,
  source annotations, curated identity, mapping status, exclusions, major-list
  membership, sampled temperatures and expected historical values.
* ``comparison.json`` records the methods and sampled results. Its 1500 K
  projection produces ``table.csv``, ``table.rst`` and ``table.md``.
* ``receipt.json`` identifies the fresh execution, its command, environment,
  selected code/input bytes, output hashes and replay results. It is not a
  reconstructed receipt for the historical scratch probe.

The comparison covers 93 included molecules and 14 excluded candidates at
500, 1000, 1500 and 3000 K. Seventy mappings remain provisional. Curated values
outside its fitted 1500--4000 K range are recorded as missing. The local NASA
file's original acquisition is unverified; matching 94 coefficient intervals
to the existing official-derived reference fixture does not authenticate the
remaining file. Replay requires no sibling checkout or network download.

The ratios use dimensionless formation Kp from neutral atoms at 1 bar. Colors
use the unrounded symmetric factor ``max(r, 1/r)``: green below 2, yellow from
2 to below 10, and red from 10. For frozen atmospheric major molecules, 🔷 marks
``r <= 0.9`` or ``r >= 1.1`` separately in each ratio column. ``*`` marks a
provisional molecular mapping; ``†`` identifies the historical JANAF
formation-column inconsistency. The final column records the pinned curated
basis; ``[U]`` means ``inherited_unreviewed``, not a verified absence of ATcT
ancestry. Missing values and agreement colors do not imply physical support.

Run from the repository root with an unused output directory::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python mceg/reviews/gas_source_comparison.py \
     --output-dir results/gas_source_comparison/replay-001
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python mceg/reviews/gas_source_comparison.py --check

The comparator refuses overwrites. New exploratory runs belong under
``results/``; reviewed snapshots receive new archive identifiers and new MCEG
subjects/evidence/assessments. Retain this snapshot and its judgments. See
``documents/mceg/gas_source_comparison.rst`` and
``mceg/contracts/gas_source_comparison.yaml`` for conventions and follow-ups.
