S2O source investigation snapshot
================================

Separate follow-up to the immutable gas_source_comparison/baseline-001.
``sources.json`` records the fresh 2026-11-02 JST acquisition hashes and inspected
locations. The three JANAF text tables are complete official downloads; the
NASA text is the S2O record extracted from the official report's Appendix D.
The selected Lodders table entries are transcribed numerical facts, checked
against Table 5. Neither full publications nor HTML pages are required for replay.

``comparison.json`` contains four gas-atom formation-K comparisons, a molecular
anchor/thermal diagnostic, and the balanced rounded ATcT atomization value.
Curated values are reused from baseline-001, not freshly solved abundances.
``receipt.json`` records the fresh offline execution and selected-file hashes.
Run ``python mceg/reviews/s2o_source_investigation.py --check`` from the root.
The report and MCEG contract preserve unresolved physical support and follow-ups.

The second supplied literature batch adds local PDF identities and inspected
facts for Berkowitz (1977) and Norwood (1989). Full papers and extracted text
remain outside git. The replay checks the wavelength/thermal correction and
the SO ionization subtraction in the fragment cycle. Abstract/body error bars
are preserved separately; no new uncertainty or production energy is adopted.
The current receipt records a fresh replay after integration with develop;
earlier receipt bytes remain in git history.
