NH3 source review packet
========================

Replay from the repository root, without network access::

   python mceg/reviews/nh3_source_review.py
   pytest tests/model_evidence/nh3_source_review_test.py

``comparison.json`` separates balanced ATcT/NASA/JANAF energy, normalized
partition comparisons, JANAF caloric differences, the unreconciled printed
2014 fit, and the existing curated-h2o-v1 75-case footprint. No new equilibrium
substitution or production change is represented by this packet.

``sources.json`` pins original acquisition URLs and hashes separately from
the compact archived factual data. ``literature.json`` records inspection limits,
state conventions and the concrete supplementary-data follow-up. Papers and
the large CoYuTe state distribution are not vendored. The 2014 electronic
Q/moment supplement was not acquired; printed-fit values are diagnostic only.

The HAPI extraction is reproducible from its pinned original source::

   python mceg/reviews/nh3_source_review.py --extract-hapi /path/to/hapi.py --packet /tmp/nh3-arrays

Extraction retains exact temperature/Q arrays for M=11, I=1 with no interpolation.
``receipt.json`` and the MCEG subject pin the replay implementation and inputs.
