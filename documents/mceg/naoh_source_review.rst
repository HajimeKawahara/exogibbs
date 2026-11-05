NaOH source review
==================

Retain the current NaOH coefficients for this source-review PR. The next useful
calculation is a controlled comparison of a Gurvich/NASA energy correction and an
OYT5 thermal candidate using the same energy and atomic references. Current
agreement with older JANAF does not validate its formation energy.

Source and thermal findings
---------------------------

The neutral linear monomer is ``H1Na1O1``; it is distinct from the hydroxide dimer.
The inherited FastChem fit is labelled JANAF. At 1500/3000 K its formation-K ratio
to NASA is 1.640724/1.271573. Official JANAF Hf298 is -197.757 kJ/mol.
`Gurvich et al. (1996) <https://doi.org/10.1063/1.555982>`_ evaluate
-191 +/- 8 kJ/mol, using a reassessment of flame studies; their uncertainty is an
evaluation estimate, not an independent Gaussian error bar. The seven selected
NASA thermal nodes reproduce their Table 3-22 within 1.91 J/mol in H, 0.0036 J/mol/K
in S and 0.0063 J/mol/K in Cp. This supports attribution to that evaluated source.

`Owens et al. (2021) <https://doi.org/10.1093/mnras/staa4041>`_ calculate a
228 cm-1 bending fundamental, compared with the estimated 300 cm-1 in Gurvich.
After division by eight, OYT5 internal Q/NASA is 1.228573 at 1500 K,
1.049118 at 3000 K and 0.944133 at 4000 K. The corresponding same-energy,
same-atom proxy including isotope translation is 1.228361/1.048937/0.943971.
These are thermal diagnostics, not abundance ratios. They do not establish that
NASA is the preferred thermal function simply because its energy evaluation is
better documented. At4000 K the downloaded sum gives Cp=48.427 versus NASA 61.190
J/mol/K; completion and smooth caloric consistency must precede full-range use.

The downloaded OYT5 ``.def`` gives nuclear weights 6, whereas the paper and
first ten J=0 ``.states`` records give 8. The ground record has energy 0 and weight 8.
The review uses 8 and preserves the inconsistent definition as evidence. The
3500 K soft line-list limit and Qred/Q approximately 0.90 refer to opacity's
16000 cm-1 lower-state cutoff; that ratio is not a 10% bound on missing total Q.

Decision and follow-up
-----------------------

Build two separately named candidates: NASA/Gurvich thermochemistry, then OYT5
thermal functions at the same balanced energy and atomic references. Check the
complete1500–4000 K thermal domain, nuclear-spin/ground-energy normalization,
J and upper-energy convergence, and H/S/Cp before fixed-network substitution.
No mandatory unavailable paper blocks this completed source review. Detailed
original flame/vapor experiments remain an MCEG provenance follow-up.

Archived importance and reproducibility
---------------------------------------

The saved ``curated-cos-v1`` grid covers 75 cases each for 537 ion-inclusive and 421
neutral species, 1500–4000 K, 1e-6-100 bar and three C/O values. Its maximum
Na inventory fraction in NaOH is 0.884386% in the ion-inclusive network
(2.5218715e-08 mole fraction), at 1500 K, 100 bar and baseline C/O.
Both network target potentials reproduce the named production model exactly.
This is an archived ideal-gas footprint, not a new solve, substitution result,
observation, or a conclusion about all planetary atmospheres.

``mceg/reviews/naoh_source_review.py`` replays the numerical packet offline.
Source-native NASA R=8.314510 and current SI R=8.31446261815324 are explicit;
G comparisons use common-R normalization. Exact JANAF nodes, source coefficients,
partition values and selected published-table values are pinned. Rounded PF
calorics use two stencil sizes and a rounding bound; these are diagnostic only.
No production potential, species set, abundance, domain or default is changed.
The MCEG contract is ``mceg/contracts/naoh_source_review.yaml``.

.. code-block:: bash

   JAX_PLATFORMS=cpu python mceg/reviews/naoh_source_review.py --output results/naoh_review.json
   JAX_PLATFORMS=cpu pytest tests/model_evidence/naoh_source_review_test.py
