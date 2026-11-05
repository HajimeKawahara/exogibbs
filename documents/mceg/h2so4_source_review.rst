H2SO4: retain the revised thermal source
============================================

Retain the current Dorofeeva 2003 source. The disagreement with the older
NASA/Gurvich functions is not a reason to replace it. Direct table replay
supports the source attribution, while polynomial caloric residuals and
physical accuracy remain separate concerns. No production change is made.

The ``H2O4S1`` FastChem record explicitly cites Dorofeeva et al. 2003,
not the older JANAF sulfuric-acid table. We inspected the official full
paper, section 3 and Table 3. Its revised C2 geometry includes two optical
isomers and hindered OH internal rotations. Older Gurvich functions used
C2v geometry without this rotation treatment. The evaluation retains
Hf(298.15 K) = -732.7 +/- 2.0 kJ/mol and gives S(298.15 K) = 311.333 J/mol/K.
Its source uncertainties and approximate rotor model remain limitations.

At 1500 K, the revised-table/NASA lnK difference is 1.355541: energy 0.133877,
molecular thermal 1.220883, atomic thermal 0.000782. The thermal difference
survives energy alignment: revised-table/NASA ratios are 3.39283 at 1500 K
and 2.57877 at 3000 K after matching D0. Consequently, an energy-only patch
or a return to legacy NASA would not preserve the evaluated thermal model.

The current fit/revised-table ratio is 1.04635 at 1500 K, 0.97207 at 3000 K
and 1.00950 at 4000 K. Across six sampled production-domain knots the
largest absolute lnK residual is 0.045313. The largest reaction Cp residual
is 5.47825 J/mol/K at 3000 K, larger than the paper's stated 4 J/mol/K
Cp uncertainty there. Faithful source attribution is therefore not a
claim of a precision caloric fit. A constrained refit is a useful follow-up
if caloric derivatives matter. Neither these six knots nor the thermal
source's nominal 6000 K table limit validates an entire temperature interval.

Archived current-model footprint and reproducibility
--------------------------------------------------------

The reused ``curated_oh_adoption/20261105/ions.npz`` archive contains 75
converged cases and 537 species: 1500-4000 K, 1e-6-100 bar, reference C/O,
C/O = 1 and 1.5. The largest S budget share carried by this molecule is
6.9512397e-22, at 4000 K / 100 bar /
C/O = 0.58884366; its mole fraction there is 1.4655591e-26.
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

   PYTHONPATH=src python mceg/reviews/h2so4_source_review.py
   pytest tests/model_evidence/h2so4_source_review_test.py

Next calculation and source work
------------------------------------

* Fit a thermodynamically consistent continuous representation to the inspected
  Dorofeeva table, checking lnK and H/S/Cp together throughout 1500-4000 K.
  Preserve the revised rotor/isomer convention and avoid overfitting rounded nodes.
* If a better fit is proposed, run paired finite substitutions and check
  element conservation before adoption; no such substitution was run here.
* Treat lower-temperature vapor, aqueous acid, hydrates, condensation and
  aerosol nucleation as separate model domains. The hot gas grid cannot
  establish their accuracy or importance.
* Main evaluation is already available:
  `10.1063/1.1547435 <https://doi.org/10.1063/1.1547435>`_. Its underlying
  original experiments and Gurvich 1989 part 1 p.300 and part 2 p.182 remain ancestry
  follow-ups, not merge gates. No new paper is required to retain this source.
