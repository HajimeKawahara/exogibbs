HCP identity and priority thermochemical source review
======================================================

HCP is a high-priority follow-up: the current archive assigns up to 99.963786%
of phosphorus to it, yet the inherited JANAF formation energy is an old estimate
with a 63 kJ/mol uncertainty. The official CCCBDB correction and independent
high-level theory both favor a substantially less stable molecule. Retain
production for this source-only PR, then evaluate energy alternatives and a
separate modern thermal candidate. No abundance response is claimed yet.

Exact identity and actual source coverage
-----------------------------------------

``C1H1P1`` is neutral linear H-C-P gas, phosphaethyne/methinophosphide,
CAS 6829-52-3, ground X 1Sigma+, symmetry number 1. It is not H-P-C, a cation,
polymer or condensed phase. The official JANAF C-056 table and printed page 604
explicitly identify this gas. Its evaluation dates to December 1969, with a
later 1-bar conversion. Thus the Issue #308 appendix identity can be resolved
without assigning a guessed NASA analogue.

The complete pinned official NASA ``thermo.inp`` contains no neutral HCP/CHP
record. The original screening appendix likewise records no unique candidate.
NASA ratios and NASA D0 remain null; the frozen 93-species comparison is not
silently expanded. The source test instead verifies the literal retained
FastChem fit against official JANAF molecular and neutral C/H/P atomic tables.
P-008 is the phosphorus **gas atom**; P-001 is an elemental condensed reference
and is not used as a gaseous reactant.

The printed JANAF discussion estimates the C-H bond from HCN and combines it
with historical CP thermochemistry. The selected table has Hf298=149.900 and
Hf0=151.192 kJ/mol, with a quoted +/-63 kJ/mol uncertainty. Its thermal model uses
B0=0.6676545 cm-1 and fundamentals 3216.9, 674.7 (twice), 1278.4 cm-1, including
HCN-estimated anharmonic corrections. Eight electronic origins are listed; the
source applies ground-state molecular constants to their thermal contribution.
The high-temperature source is therefore more than a harmonic-only model,
but its anharmonic and excited-state assumptions remain approximate.

Modern energy evidence and balanced interpretation
--------------------------------------------------

The full official `CCCBDB HCP entry
<https://cccbdb.nist.gov/exp2x.asp?casno=6829523>`_ reports Hf298=212.20 +/-20.00
and Hf0=213.50 +/-20.00 kJ/mol, explicitly adjusted from JANAF using modern CP
formation energy. Its entropy/heat-content values retain JANAF ancestry.
This is an evaluated correction, not a new independent HCP calorimetric result.
The official `CCCBDB elemental reference table
<https://cccbdb.nist.gov/refstatex.asp>`_ specifies white crystalline phosphorus,
with H298-H0=5.360 +/-0.002 kJ/mol. The pinned JANAF P-008 atom independently
implies the same elemental heat: 6.197-(316.390-315.553)=5.360 kJ/mol.
Thus the CCCBDB/selected-JANAF formation comparison uses compatible white-P
references; no extra red/white phase shift is applied. The old JANAF narrative's
40 kcal/mol differs from its selected 149.900 kJ/mol table by 17.460 kJ/mol,
consistent with the historical phosphorus reference conversion. The selected
numerical table, rather than that unrevised narrative number, is used here.

The complete author-hosted 2001 paper,
`DOI 10.1039/B106927F <https://doi.org/10.1039/B106927F>`_, independently obtains
Hf0(HCP)=225 +/-8 kJ/mol from CCSD(T)/aug-cc-pVxZ atomization calculations.
The paper's adopted atomic Hf0 values are C=711, P=316 and H=216 kJ/mol:
their sum minus 225 gives a total atomization energy of 1018 kJ/mol. Its
separately labeled H-CP ``De`` must not be mistaken for total atomization D0.
The two modern formation anchors overlap within their reported uncertainties,
but do not establish a sub-kJ/mol recommendation. No HCP entry was found in the
inspected full ATcT 1.228 index, so no ATcT value is inferred.

For transparent sensitivity, keep the JANAF elemental reference and thermal
functions fixed and change only the molecular formation-energy anchor.
Physical balanced JANAF atomization D0 is 1091.582 kJ/mol. The CCCBDB anchor
shifts Hf0 by +62.308 kJ/mol, lowering common-reference D0 to 1029.274 kJ/mol.
The 2001 Hf0 anchor shifts it by +73.808 kJ/mol, giving 1017.774 kJ/mol on that
same reference. The 0.226 kJ/mol difference from the paper's own 1018 kJ/mol
atomization energy is its different rounded atomic reference, not an additional
measured correction. A future adopted energy must select and document one
balanced convention, rather than mix these numbers silently.

.. list-table:: Formation K and energy-only sensitivities
   :header-rows: 1

   * - T (K)
     - Current / JANAF
     - CCCBDB energy factor
     - 2001 Hf0 energy factor
   * - 1500
     - 1.01250132
     - 0.00676527
     - 0.00269046
   * - 2000
     - 1.00072524
     - 0.02358923
     - 0.01181328
   * - 3000
     - 0.99141995
     - 0.08225125
     - 0.05186969
   * - 4000
     - 1.00276435
     - 0.15358786
     - 0.10868891

The last columns are exp(-delta-Hf0/(RT)) applied to the current function while
preserving its thermal shape and fit residual. They are not finite abundance
ratios. G and H shift equally, while S and Cp remain unchanged. The reported
modern energy uncertainties alone correspond at 1500 K to multiplicative K
factors 4.97 (CCCBDB) and 1.90 (theory), before inherited thermal uncertainty.
The packet includes source molecular/atomic/reaction G/H/S/Cp, physical D0,
energy/thermal closure, raw formation-column closure and literal fit derivatives.
All comparisons use SI R and 1 bar.

Modern thermal route and its precise data gap
---------------------------------------------

The 2024 primary abstract,
`DOI 10.1021/acs.jpca.4c01823 <https://doi.org/10.1021/acs.jpca.4c01823>`_,
reports a ground-state TROVE line list with about 1.21 million levels, J<=200,
and an intended temperature limit of 3000 K. Its public ACS/figshare supplement
was actually downloaded and inspected: it contains three Fortran PES/DMS
routines, **not** the states file or numerical Q table. The main full text,
state list and Q moments were not obtained. A PES routine alone is not a
finished thermodynamic model, and 4000 K lies outside the advertised limit.

Obtain the states or Q/moment data and normalization details, remove the global
H/P nuclear-spin factor only if present, preserve bend degeneracy and separate
isotopologues. Calculate derivative-consistent thermal moments within the
supported range; test a justified extension or retain the current thermal
shape above it. The old electronic-origin-only factor is just 1.000628 at
4000 K with the old same-geometry approximation; it is a bounded sensitivity,
not an independent complete excited-electronic error bar. Energy-only candidate
calculations can proceed before resolving this separate line-list data gap.

Importance, decision and replay
-------------------------------

All 75 ``curated-tio2-v1`` ion-inclusive endpoints converge with 537 species.
HCP's maximum P share is 99.96378572% at
1500 K, 0.01 bar and C/O=1.5;
maximum mole fraction is 4.40908719e-07.
This inventory makes the old energy issue urgent, but does not imply that P
share scales linearly with K or that this hot H-rich result applies universally.

Prioritize finite-network energy-alternative calculations, with the 2024 thermal
candidate as a separate follow-up. The source review is mergeable without
pretending unavailable full texts or full physical uncertainty are resolved.
Run ``python mceg/reviews/hcp_source_review.py`` and
``pytest tests/model_evidence/hcp_source_review_test.py``. The packet is
``tests/data/validation/hcp_source_review/20261106`` and contract is
``mceg/contracts/hcp_source_review.yaml``. Inputs use historical hash resolution;
outputs are checked directly. No full manuscript, PES routine or large state
list is committed, and no curated change or new equilibrium solve is made.
