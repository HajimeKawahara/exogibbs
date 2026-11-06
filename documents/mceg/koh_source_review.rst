KOH source review
=================

Retain the current KOH coefficients for this source-review PR. Evaluate an OYT4
thermal candidate with a consistent energy and atomic reference before choosing
a replacement; replacing KOH by NASA solely because it is newer is unsupported.

Source and thermal findings
---------------------------

Neutral linear ``H1K1O1`` is the monomer, distinct from ``H2K2O2``.
The current JANAF-labelled fit/NASA formation-K ratios are 0.817859 and 0.795292 at
1500/3000 K. Official JANAF Hf298=-232.630 kJ/mol is close to the
-232 +/- 3 kJ/mol selected in `Gurvich et al. (1997)
<https://doi.org/10.1063/1.555996>`_. That evaluation reanalyses monomer/dimer
vapor measurements and warns about quantitative limitations in older studies.
At1500 K the molecular source difference contains -0.631 kJ/mol in the H298
anchor, -0.092 in its enthalpy increment and +3.266 from entropy; atomic
references are accounted for separately. Thus an energy-only correction cannot
explain the full source difference. Seven NASA thermal nodes reproduce Table 23
within 1.59 J/mol in H,0.00075 J/mol/K in S and 0.0064 J/mol/K in Cp.

`Owens et al. (2021) <https://doi.org/10.1093/mnras/staa4041>`_ calculate a
343 cm-1 bending fundamental, versus Gurvich's estimated300 cm-1.
OYT4 Q/8/NASA is 0.869552,0.843848 and 0.769147 at 1500/3000/4000 K;
including isotope translation gives0.866318/0.840709/0.766287. The isotope mass
55.96644633 u is kept separate from NASA's mean mass 56.105640 u. At4000 K
partition-derived Cp is 43.854 versus NASA 61.201 J/mol/K. These differences are
model diagnostics; no full-range uncertainty or substitution response is claimed.

The downloaded OYT4 ``.def`` lists nuclear weights 6, while the paper and ten
J=0 state records give 8; the ground record has energy 0 and weight 8. The calculation
uses 8. The paper's Qred/Q approximately 0.90 at 3500 K diagnoses opacity's lower-state
cutoff, not a 10% missing total Q estimate. Rotational convergence alone does not
establish completeness of the upper-energy manifold or caloric moments.

Decision and follow-up
-----------------------

Compare current, NASA/Gurvich and OYT4 at aligned energy and atomic references,
then construct a continuous candidate only where H/S/Cp and state convergence
are supported. Include a one-species fixed-network substitution before adoption.
Original vapor experiments can extend provenance later; their complete recovery
is not a prerequisite to merging this bounded review. All inspected evaluations
and the outstanding thermal-model work are recorded in MCEG.

Archived importance and reproducibility
---------------------------------------

The saved ``curated-cos-v1`` grid covers 75 cases each for 537 ion-inclusive and 421
neutral species, 1500–4000 K, 1e-6-100 bar and three C/O values. Its maximum
K inventory fraction in KOH is 3.345401% in the ion-inclusive network
(6.75351025e-09 mole fraction), at 1500 K, 100 bar and baseline C/O.
Both network target potentials reproduce the named production model exactly.
This is an archived ideal-gas footprint, not a new solve, substitution result,
observation, or a conclusion about all planetary atmospheres.

``mceg/reviews/koh_source_review.py`` replays the numerical packet offline.
Source-native NASA R=8.314510 and current SI R=8.31446261815324 are explicit;
G comparisons use common-R normalization. Exact JANAF nodes, source coefficients,
partition values and selected published-table values are pinned. Rounded PF
calorics use two stencil sizes and a rounding bound; these are diagnostic only.
No production potential, species set, abundance, domain or default is changed.
The MCEG contract is ``mceg/contracts/koh_source_review.yaml``.

.. code-block:: bash

   JAX_PLATFORMS=cpu python mceg/reviews/koh_source_review.py --output results/koh_review.json
   JAX_PLATFORMS=cpu pytest tests/model_evidence/koh_source_review_test.py
