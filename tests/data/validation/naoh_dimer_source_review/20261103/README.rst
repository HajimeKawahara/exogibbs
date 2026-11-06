Neutral sodium hydroxide dimer source investigation
==================================================

This packet freezes the 2026-11-03 source review without changing production.
Replay from the repository root::

   python mceg/reviews/naoh_dimer_source_review.py
   pytest tests/model_evidence/naoh_dimer_source_review_test.py

``H-062.txt`` is the exact official NIST-JANAF 1-bar dimer table. Atomic and
monomer functions and NASA9 coefficients reuse the pinned ``baseline-001``
inputs. ``inputs.json`` contains factual numerical excerpts from JANAF1974
p.449 and Gurvich1996 Tables3-23,3-26,3-27. The comparisons use absolute
functions; they retain the inconsistent Table3-24 entropy-difference cell as
a separate fact, not a silently corrected numerical input.

``sources.json`` records original URLs, inspected pages, full-PDF hashes and
access limits. Only compact numerical facts and publisher-deposited bibliographic
metadata are committed. Full articles, screenshots and extracted text are not
vendored. ``literature.json`` records the current-source genealogy, scientific
preference, shared ancestral experiments and unresolved original-paper checks.

``comparison.json`` separates molecular H298, thermal enthalpy, entropy and
atomic references, checks the published thermal construction, and computes a
separate all-gas association reaction using both source-specific monomers.
The latter is not a dimer-only replacement response. ``receipt.json`` pins
selected inputs and outputs, including the existing 75-case ``ions.npz``.
The archived sodium budget is counted with two Na atoms per dimer. No new
equilibrium solve or atmospheric accuracy claim is made.

The canonical record is ``mceg/contracts/naoh_dimer_source_review.yaml``.
MCEG source-confirmation follow-ups do not prevent merging this completed,
scoped investigation. The preferred adoption direction is the inspected
Gurvich1996/NASA representation, not an assertion of perfect thermal accuracy.
