S2 source review, 2026-11-05
===========================

This packet preserves all 17,766 Gomez state energies and statistical weights
in compact arrays, plus the printed ExoMol/HITRAN partition table. No full
paper or publisher page is redistributed. ``sources.json`` records original
URLs, SHA256 hashes, download failures and the actual inspection scope.
``literature.json`` separates primary evidence from uninspected follow-ups.

Run offline from the repository root::

   PYTHONPATH=src python mceg/reviews/s2_source_review.py

To reconstruct ``states.npz`` and ``state_inventory.json``, place the original
``Gomez.states.bz2``, ``Gomez.pf`` and ``q152.txt`` files in one directory::

   PYTHONPATH=src python mceg/reviews/s2_source_review.py --extract RAW_DIRECTORY

Their hashes must match the acquisition record. The extraction retains energy,
gtot, J, v, N and a raw electronic-label index for every state. The excited-state
labels are not silently corrected. No additional nuclear spin or symmetry
factor is applied. Original 1--5000 K tables are retained for source replay,
not an assertion of completeness at 5000 K.

``comparison.json`` separates balanced ATcT/NASA energies, same-energy thermal
ratios, moments and archived current-model sulfur allocation. No production
data, immutable prior receipt or equilibrium solution is changed.
``receipt.json`` pins selected existing inputs and new evidence; optional MCEG
checks the matching scoped contract independently of ordinary package tests.
