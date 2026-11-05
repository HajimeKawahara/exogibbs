C5: obsolete state assumptions and thermal cancellation
===========================================================

Retain the present fit provisionally; a state-consistent replacement is
unresolved. The historical model has identified scientific weaknesses.
Do not substitute a modern singlet energy into its estimated triplet
thermal functions. This review makes no production change.

The ``C5`` FastChem record reproduces JANAF 1998. The original JANAF
C5 notes, printed p. 696, explicitly assume an estimated linear triplet
and estimated molecular constants. Three doubly degenerate bending modes
were raised from 130/330/520 to 550/550/550 cm^-1 to agree with vapor-pressure
observations. The notes restrict their useful agreement to 2000-3000 K
and caution against treating other temperatures as representative.
Thus the old fit's numerical reproduction does not validate its state inventory.

Karton et al.'s inspected W4 author manuscript, abstract/Table V, identifies
linear singlet 1Sigma_g+ C5 and TAE0 = 596.64 kcal/mol = 2496.34176 kJ/mol.
This includes zero-point energy. Native balanced JANAF and NASA anchors
give 2583.528 and 2515.988024 kJ/mol, respectively. W4 is a state-specific
energy reference, not a complete hot partition function or a cyclic-isomer
model. Gurvich 1991 part 1 p.20 and part 2 p.18, cited by NASA, was not obtained;
its detailed state inventory remains a recorded ancestry gap.

The current/NASA K ratio reverses from 4.30140 at 1500 K to 0.177999 at 3000 K
and 0.082686 at 4000 K. At 1500 K, JANAF-minus-NASA lnK has energy +5.416611,
molecular thermal -3.986131 and atomic thermal +0.000490. At 3000 K these
become +2.708305, -4.416908, +0.000829. After energy alignment, ratios remain
only 0.01858 and 0.01208. Both energy and thermal-state construction matter;
changing spin degeneracy alone cannot repair the remaining discrepancy.

Archived current-model footprint and reproducibility
--------------------------------------------------------

The reused ``curated_oh_adoption/20261105/ions.npz`` archive contains 75
converged cases and 537 species: 1500-4000 K, 1e-6-100 bar, reference C/O,
C/O = 1 and 1.5. The largest C budget share carried by this molecule is
1.2002277e-06, at 2000 K / 1e-06 bar /
C/O = 1.5; its mole fraction there is 1.9431177e-10.
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

   PYTHONPATH=src python mceg/reviews/c5_source_review.py
   pytest tests/model_evidence/c5_source_review_test.py

Next calculation and source work
------------------------------------

* Reconstruct a linear-singlet thermal model with defensible bending
  frequencies, rotational symmetry and electronic-state inventory. Keep any
  cyclic isomers or excited states explicit before deciding whether a single
  composition column represents one state or an equilibrium sum.
* Align W4 TAE0 and neutral-atom anchors, check H/S/Cp and state completeness,
  then construct a continuous candidate and run finite substitutions.
* Revisit the experimental third-law reductions using the same thermal model.
  The relevant papers are
  `10.1063/1.461173 <https://doi.org/10.1063/1.461173>`_ and
  `10.1021/ja00088a025 <https://doi.org/10.1021/ja00088a025>`_. Their full texts
  were not inspected here; obtaining them is useful for replacement design,
  not required to merge this explicit unresolved-state record.
* The inspected W4 energy reference is
  `10.1080/00268970802708959 <https://doi.org/10.1080/00268970802708959>`_.
  NASA's original Gurvich 1991 pages are useful for its state genealogy;
  no DOI is asserted for those book pages.
