Offline validation fixtures
===========================

These files are the saved numerical references and execution records used by
the regression tests, MCEG contracts, and documentation. They were moved
byte-for-byte from ``results/`` at revision ``be2bf7113``. Historical commands,
paths, timestamps, and hashes inside the records remain unchanged. Resolve a
recorded ``results/<path>`` here as ``tests/data/validation/<path>``; the move
does not represent a new numerical execution. The additional FastChem and
sensitivity records come from ``cc0dbb5`` (merged PRs #298 and #299).

``new_fastchem_logk/source/`` preserves the two scripts pinned by the FastChem
receipts. Current scripts differ only in their fixture input paths, as checked
against these snapshots in the tests. The MCEG comparison snapshot refers to
the archived sources; its historical hashes and scientific judgments are retained.

Examples still write new runs to the ignored ``results/`` directory. Review
new numerical results before replacing fixtures; keep earlier execution
records attached to their original inputs and revisions. These fixtures ship
in the source distribution for offline tests, but are not runtime package data.

The large historical phase-search archives remain in earlier Git revisions
and local backups. Documentation-only sensitivity figures live under
``documents/_static/equilibrium_sensitivity/``.
