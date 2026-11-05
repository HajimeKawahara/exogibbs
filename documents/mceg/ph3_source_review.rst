PH3 source review
=================

Retain the current Lodders-labelled PH3 fit. Its phosphorus-reference correction
is already present. The next scientific improvement is a checked high-temperature
thermal model, using the 2014 thermodynamic supplement rather than the incomplete
SAlTY state list or an unverified printed polynomial.

Energy reference and existing model
-----------------------------------

Neutral pyramidal singlet PH3 is ``H3P1``. `Lodders (1999)
<https://doi.org/10.1063/1.556046>`_, Tables 2/4 and section 2.2, selects
Hf298=5.440 +/-1.7 kJ/mol on the white-P reference; NASA uses 5.439.
The currently distributed JANAF H-088 table and frozen screening instead carry
22.886 kJ/mol. The approximately 17.45 kJ/mol separation is associated with the
red-V/white reference issue, and must not be applied again to curated. Lodders
also identifies errors in the printed fourth-edition formation functions beyond
the 298.15 K entry. This review keeps the online-table discrepancy distinct from
those historical printed-table errors. Elemental reference changes cancel only
when every molecular and atomic contribution is shifted consistently.

Current/NASA formation K is 1.008915 at 1500 K and 0.982915 at 3000 K. Six selected
NASA thermal nodes agree with Lodders Table 4 within 1.72 J/mol in H,
0.0043 J/mol/K in S and 0.0041 J/mol/K in Cp. The close fit agreement is a source
attribution check, not independent physical validation or exact fit recovery.

Thermal evidence and a reproducible obstacle
---------------------------------------------

`Sousa-Silva et al. (2014) <https://doi.org/10.1016/j.jqsrt.2014.03.012>`_
construct a separate high-temperature thermodynamic state set with rotational
extensions, reaching PH3 J=100 and the dissociation cutoff. It is distinct from
`SAlTY (2015) <https://doi.org/10.1093/mnras/stu2246>`_, whose line-list
recommendation is 1500 K. Nuclear-spin normalization is 16 for PH3, not the
per-symmetry weight 8. The distributed SAlTY table ends at 3000 K; normalized
Q/NASA is 1.056991 at 1500 K and 1.014697 at 3000 K, but its 3000 K Cp diagnostic is
53.020 versus NASA 80.159 J/mol/K. A near-unity Q ratio does not establish correct
high-temperature derivatives or convergence.

Eq. 18/Table 1 of the 2014 paper yields Q(296)=3216.473994 from the printed PH3
coefficients, versus the stated direct sum 3249.46: a -1.0151% mismatch. The
rounding envelope from the printed coefficient digits is 0.01035%, far smaller
than the mismatch, and the advertised fit accuracy is 0.4%. These coefficients
are diagnostic only. Their high-temperature Q or derivatives are not accepted
as production thermal data. The full electronic tables and first/second moments
were not obtained; they are the highest-priority input for the next candidate.

Decision and follow-up
-----------------------

Obtain the electronic PH3 thermodynamic tables/moments for
DOI 10.1016/j.jqsrt.2014.03.012; reconstruct H/S/Cp with the same ground zero,
spin convention, one-bar translation and declared isotope convention. Retain
or independently reassess a consistently balanced energy; a phase-reference
conversion is not a new energy measurement. Test a continuous thermal candidate
and one-species fixed-network substitution. Full physical accuracy, older
calorimetry recovery and disequilibrium atmospheric behavior remain unknown.
This source-review PR can be assessed independently of later candidate adoption.

Archived importance and reproducibility
---------------------------------------

The saved ``curated-cos-v1`` grid covers 75 cases each for 537 ion-inclusive and 421
neutral species, 1500–4000 K, 1e-6-100 bar and three C/O values. Its maximum
P inventory fraction in PH3 is 85.599192% in the ion-inclusive network
(3.78051535e-07 mole fraction), at 1500 K, 100 bar and baseline C/O.
Both network target potentials reproduce the named production model exactly.
This is an archived ideal-gas footprint, not a new solve, substitution result,
observation, or a conclusion about all planetary atmospheres.

``mceg/reviews/ph3_source_review.py`` replays the numerical packet offline.
Source-native NASA R=8.314510 and current SI R=8.31446261815324 are explicit;
G comparisons use common-R normalization. Exact JANAF nodes, source coefficients,
partition values and selected published-table values are pinned. Rounded PF
calorics use two stencil sizes and a rounding bound; these are diagnostic only.
No production potential, species set, abundance, domain or default is changed.
The MCEG contract is ``mceg/contracts/ph3_source_review.yaml``.

.. code-block:: bash

   JAX_PLATFORMS=cpu python mceg/reviews/ph3_source_review.py --output results/ph3_review.json
   JAX_PLATFORMS=cpu pytest tests/model_evidence/ph3_source_review_test.py
