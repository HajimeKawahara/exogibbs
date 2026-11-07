HO2: evaluated energy update candidate
==========================================

The inherited JANAF fit should be replaced by a modern evaluated-energy
candidate after continuous-model and finite-substitution checks. This review
supports that next step and leaves production unchanged. Numerical replay
alone does not establish physical accuracy.

The current ``H1O2`` record cites Chase/JANAF 1998 and has
Hf(298.15 K) = 2.092 kJ/mol. NASA's header uses 12.020 kJ/mol.
Burgess 2016, section 2.5/Table 5, selects 12.30 +/- 0.35 kJ/mol and records that
the old JANAF evaluation dates to 1978 with much larger uncertainty.
The inspected ATcT 1.172 index instead gives HO2 Hf(0) = 15.06 and
Hf(298.15 K) = 12.15 +/- 0.14 kJ/mol. These are version-specific evaluations,
not independent observations to average. H/O/HO2 from that same ATcT version
give D0 = 694.662 kJ/mol. Species covariance is unavailable; no reaction
uncertainty is synthesized from marginal uncertainties.

The old fit reproduces JANAF closely. The 1500 K JANAF/NASA lnK difference
separates into energy 0.796219, molecular thermal 0.002273 and atomic
thermal 0.000215. After common ATcT D0 and NASA atomic thermal functions,
JANAF/NASA molecular thermal ratios are 1.00228 at 1500 K, 0.99415 at 3000 K
and 0.97992 at 4000 K. The dominant known problem is the old formation
energy, while residual caloric/state completeness still needs review.

A pointwise NASA-thermal/ATcT diagnostic gives K ratios to curated of
0.44639 at 1500 K, 0.67809 at 3000 K and 0.75516 at 4000 K. This is an
explicitly normalized source diagnostic, not an adopted continuous model
or an equilibrium abundance prediction.

Archived current-model footprint and reproducibility
--------------------------------------------------------

The reused ``curated_oh_adoption/20261105/ions.npz`` archive contains 75
converged cases and 537 species: 1500-4000 K, 1e-6-100 bar, reference C/O,
C/O = 1 and 1.5. The largest O budget share carried by this molecule is
3.2885192e-08, at 4000 K / 1 bar /
C/O = 0.58884366; its mole fraction there is 8.9112967e-12.
The archived chemical-potential column agrees with the current fit and
its element budget closes. This is an existing-endpoint footprint, not
candidate substitution or a bound on changes after replacement. Trace
abundance in this restricted grid does not imply atmospheric irrelevance.

The offline packet separates exact source nodes, polynomial-fit residuals,
balanced zero-K energy and molecular/atomic thermal contributions. It
uses 1 bar, modern R = 8.31446261815324 and NASA's native R = 8.314510 J/mol/K.
NASA headers supply H(0) anchors; polynomials are never extrapolated to 0 K.
The pinned current CEA records begin at 300 K where the frozen comparison
used 200 K; all source comparisons here lie at 500-4000 K.

``sources.json`` records inspected/uninspected material, download hashes
where bytes were obtained, and selected numerical transcriptions.
``comparison.json`` and ``receipt.json`` replay offline. The MCEG contract
supports scoped source attribution and numerical replay; physical accuracy
remains **unknown**. Sources are not silently promoted to measurements.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/ho2_source_review.py
   pytest tests/model_evidence/ho2_source_review_test.py

Next calculation and source work
------------------------------------

* Build a continuous NASA-thermal plus ATcT 1.172 energy candidate with
  explicit same-version atoms, native-R normalization and boundary checks.
* Check thermal derivatives and excited-state/partition-function completeness;
  compare with JANAF after matching energy and atomic conventions.
* Run paired finite substitutions on the current 75-case grid; report
  element-budget changes, conserved totals and solver residuals.
* Extend to lower temperatures and a photochemical network before making
  atmospheric radical-abundance claims. The archived hot equilibrium grid
  does not model photochemical formation or destruction.
* Existing accessible evidence: Burgess 2016
  `10.6028/jres.121.005 <https://doi.org/10.6028/jres.121.005>`_ and
  `ATcT 1.172 <https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_.
  Historical network ancestry follow-up:
  `10.1021/jp056311j <https://doi.org/10.1021/jp056311j>`_. Its original full
  text was not inspected; obtaining every historical paper is not a merge
  requirement for this scoped source-review record.
