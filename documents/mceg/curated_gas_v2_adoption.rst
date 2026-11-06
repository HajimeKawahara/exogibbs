Curated gas v2: combined molecular adoption
===========================================

The public ``curated.chemsetup()`` now delegates to ``curated_v2.chemsetup()``
and the packaged ``curated-gas-v2`` dataset on the fixed ``curated-ch2-v1`` base.
Twenty-seven source decisions update 28 neutral molecular species: the cis-HNO2
recommendation includes the corresponding trans-HNO2 energy. Every approved
energy is inserted once; only O2 also changes its molecular thermal function.
The 537-species ion-inclusive and 421-species neutral networks, elemental
inventories, atomic gauge, charges and one-bar standard state are unchanged.
Earlier named presets and all original source/candidate records remain available.

The selection is scientifically supported within its stated scope. Complete
physical uncertainty and planetary atmospheric accuracy remain unquantified.
Finite source differences, isolated sensitivities, and the cumulative solved
abundance changes below are distinct quantities.

Energy conventions and coordinated selections
---------------------------------------------

For a retained thermal function, the correction is
``h_new(T) = h_base(T) + delta_GH/(R*T)``, with
``R = 8.31446261815324 J/(mol K)``. A positive ``delta_GH`` increases reaction
G and H; for a physical ground-state atomization anchor it equals
``D0_old - D0_selected``. S and Cp are unchanged by that constant shift.
The fitted ``R*a`` intercept is not a measured D0. Its residual relative to
the physical source anchor is retained, along with the original atomic thermal
gauge. No fit extrapolation to zero temperature, second zero-point correction,
extra spin/symmetry weight, or second native-to-SI gas-constant adjustment is made.

ATcT choices use atom-balanced same-version 1.228 energies. These are not
molecular formation-enthalpy changes alone. The coupled choices are:

* HNCO/NCO: the joint NCO candidate supplies both energy corrections. They are
  deduplicated against the HNCO selection. The HNCO -> H + NCO reaction shift
  is -14.724 kJ/mol.
* NO2/N2O4: the joint N2O4 candidate and the single NO2 choice agree exactly;
  each species changes once. The 2 NO2 -> N2O4 reaction shift is -0.425 kJ/mol.
* cis/trans-HNO2: select both ATcT energies together. The ground-energy gap
  changes from 2.165 to 1.610 kJ/mol; the trans -> cis shift is -0.555 kJ/mol.
  Neither a cis-only change nor the separate measured-gap hybrid is adopted.

These coordinated shifts preserve a single consistent selected reaction cycle.
They do not imply a complete modern caloric network: retained source thermals
and atom functions still limit agreement with full ATcT values at other
reference temperatures.

Source choices and explicit qualifications
------------------------------------------

Most rows use the retained thermal fit and the selected physical D0. Exact
species identities are preserved, including cis-HNNH, CNN versus NCN, and CNC
versus CCN. NASA thermal alternatives and finite partition-function or RRHO
sensitivity models are not adopted by this batch. NH2 in particular retains
its old thermals: the 2026 RRHO electronic-weight convention remains unresolved.
The source/candidate documents linked in the energy table retain the primary
references, inspected access status and each thermal limitation.

CNN is a provisional energy adoption: the ATcT central value improves the
explicit source selection but has substantial theoretical provenance, while
old vibrational/electronic thermals remain uncertain. This is not a newly
validated high-temperature thermal model.

C4N2 uses the NASA-header-balanced physical D0 based on Hf298=529.2 kJ/mol.
The NIST corrected combustion value is 529.3 kJ/mol, a distinct number; the
selected correction is neither an exact 529.3 construction nor ATcT. The
1963 correction (DOI 10.1021/j100806a601, following 10.1021/j100840a514)
supports revising the old energy while retaining its thermal shape.

P4S3 adopts the 2019 preferred reaction-cycle Hf298=-94.5 kJ/mol instead of
-151.042 kJ/mol (DOI 10.1007/s11224-019-01352-7), giving G/H+56.542 kJ/mol
at the retained white-phosphorus/rhombic-sulfur reference. It has no supported
JANAF molecular H0/D0. The alternate direct-atomization value -103.9 kJ/mol
and its route-specific 6.4 kJ/mol width are not the preferred-cycle result or
its uncertainty. The old constant-Cp thermal estimate remains a limitation.

N2O5 likewise uses a 298.15 K anchor, not an invented D0. Balanced JANAF and
ATcT atomization enthalpies are 2179.934 and 2176.321 kJ/mol, giving
G/H+3.613 kJ/mol. A molecular-only Hf298 difference would give 3.563 kJ/mol
and omit the atomic contribution. The ``retained_h298`` candidate is selected;
the earlier NASA thermal sensitivity and formal fit intercept do not supply
a missing physical 0 K value.

VN adopts neutral ground-state D0=4.9968 +/- 0.0020 eV
(DOI 10.1063/1.4953782), not a cation energy or De. Na2 and K2 use measured
v=0,J=0 ground-atom dissociation values, respectively 5942.688 +/- 0.0049
and 4404.808 +/- 0.004 cm^-1 (DOIs 10.1103/PhysRevA.54.R1006 and
10.1063/1.2375120). Na2's inspected primary publisher abstract supplies the
numerical result; its full original article was not obtained. The K2 article
was inspected in the author's thesis. These marginal source widths do not
quantify all thermal, isotopic or reaction-covariance errors.

O2 adopts the 2025 evaluated natural-isotope molecular thermals
(DOI 10.1063/5.0273661) and the ATcT energy. Its compact degree-8 Cp/R
polynomial is integrated analytically to anchored H/R and S/R. The runtime
adds ``evaluated molecular thermal - NASA molecular thermal proxy`` to the
inherited formation function, retaining its full fit/proxy residual. The
NASA native-R source energy was already removed during candidate construction;
the common -107 J/mol energy shift is inserted exactly once. Source isotope,
translation and statistical conventions are already included, so no extra
mass, nuclear-spin, symmetry or electronic multiplier is applied. The adopted
interval remains 1500-4000 K; the evaluated source's wider range is not claimed.

NaCN and KCN remain unchanged because their large-amplitude thermal physics
makes the explored RRHO changes sensitivity models rather than supported
replacements. C2N2 also retains its current model: the candidate review does
not establish a meaningful improvement over that source. Their completed
candidate reviews remain useful records and do not require adoption to merge.

All adopted energy corrections
------------------------------

The physical anchor column is in kJ/mol, old -> selected. D0 rows denote
atomization from the specified ground reference. N2O5 instead lists balanced
atomization H298; P4S3 lists molecular Hf298. The signed correction column is
G/H in kJ/mol, not a signed D0 change. O2 additionally receives the thermal
correction described above. Packaged JSON retains unrounded numerical values.

.. list-table:: Selected anchors and formation-energy shifts
   :header-rows: 1
   :widths: 15 20 30 15 20

   * - Molecule
     - Anchor basis
     - Physical anchor: old -> selected
     - Delta G/H
     - Candidate record
   * - HNCO
     - ATcT-balanced D0
     - 1743.502 -> 1760.762
     - -17.26
     - :doc:`hnco_candidate`
   * - HCO
     - ATcT-balanced D0
     - 1130.859 -> 1132.863
     - -2.004
     - :doc:`hco_candidate`
   * - NCO
     - ATcT-balanced D0
     - 1269.854 -> 1301.838
     - -31.984
     - :doc:`nco_candidate`
   * - CNN
     - ATcT-balanced D0
     - 1068.973 -> 1074.78
     - -5.807
     - :doc:`cnn_candidate`
   * - NCN
     - ATcT-balanced D0
     - 1180.558 -> 1201.35
     - -20.792
     - :doc:`ncn_candidate`
   * - C2
     - ATcT-balanced D0
     - 593.109 -> 602.8
     - -9.691
     - :doc:`c2_candidate`
   * - CNC
     - ATcT-balanced D0
     - 1341.362 -> 1221.195
     - +120.167
     - :doc:`cnc_candidate`
   * - C2O
     - ATcT-balanced D0
     - 1386.585 -> 1292.503
     - +94.082
     - :doc:`c2o_candidate`
   * - C3O2
     - ATcT-balanced D0
     - 2723.954 -> 2722.256
     - +1.698
     - :doc:`c3o2_candidate`
   * - C4N2
     - NASA-balanced D0
     - 3257.782 -> 3262.14097
     - -4.358965
     - :doc:`c4n2_candidate`
   * - HNO
     - ATcT-balanced D0
     - 831.145 -> 823.532
     - +7.613
     - :doc:`hno_candidate`
   * - cis-HNO2
     - ATcT-balanced D0
     - 1250.945 -> 1251.685
     - -0.74
     - :doc:`cis_hno2_candidate`
   * - trans-HNO2
     - ATcT-balanced D0
     - 1253.11 -> 1253.295
     - -0.185
     - :doc:`cis_hno2_candidate`
   * - NH2
     - ATcT-balanced D0
     - 709.647 -> 713.736
     - -4.089
     - :doc:`nh2_candidate`
   * - cis-H2N2
     - ATcT-balanced D0
     - 1153.717 -> 1144.458
     - +9.259
     - :doc:`cis_h2n2_candidate`
   * - N2H4
     - ATcT-balanced D0
     - 1696.355 -> 1693.616
     - +2.739
     - :doc:`n2h4_candidate`
   * - K2
     - Measured D0
     - 52.661 -> 52.6932053
     - -0.0322053338
     - :doc:`k2_candidate`
   * - NO
     - ATcT-balanced D0
     - 627.835 -> 626.811
     - +1.024
     - :doc:`no_candidate`
   * - NO2
     - ATcT-balanced D0
     - 928.473 -> 927.415
     - +1.058
     - :doc:`no2_candidate`
   * - NO3
     - ATcT-balanced D0
     - 1133.661 -> 1131.744
     - +1.917
     - :doc:`no3_candidate`
   * - VN
     - Measured D0
     - 459.848 -> 482.117908
     - -22.2699076
     - :doc:`vn_candidate`
   * - N2O
     - ATcT-balanced D0
     - 1102.951 -> 1102.122
     - +0.829
     - :doc:`n2o_candidate`
   * - N2O4
     - ATcT-balanced D0
     - 1910.083 -> 1908.392
     - +1.691
     - :doc:`n2o4_candidate`
   * - N2O5
     - Atomization H298
     - 2179.934 -> 2176.321
     - +3.613
     - :doc:`n2o5_candidate`
   * - Na2
     - Measured D0
     - 70.539 -> 71.0903356
     - -0.55133561
     - :doc:`na2_candidate`
   * - O2
     - ATcT-balanced D0
     - 493.579 -> 493.686
     - -0.107
     - :doc:`o2_candidate`
   * - O3
     - ATcT-balanced D0
     - 595.021 -> 596.12
     - -1.099
     - :doc:`o3_candidate`
   * - P4S3
     - Hf298 (preferred cycle)
     - -151.042 -> -94.5
     - +56.542
     - :doc:`p4s3_candidate`

Combined production verification
--------------------------------

The actual packaged factory passed 150 fresh cold endpoints: 75 for each
network, using 1500, 2000, 2500, 3000 and 4000 K; 1e-6, 1e-4, 0.01, 1 and
100 bar; and C/O=0.5888436553555908, 1 and 1.5. The 150 archived named-CH2
baseline cases are reused once after checking their hashes, potentials,
species/element matrices, conditions and budgets. They are not fresh solves.

An independent reference is assembled from the 27 selected frozen candidate
JSON files, including O2's callback, rather than from the new packaged energy
map. At 2752 temperatures (2501 integer nodes plus 251 off-grid nodes), all 28
selected columns agree with the actual factory within 8.53e-14 in G/(RT).
All other columns are unchanged. Independent scalar G/H/S/Cp and JAX
potential derivatives agree within 1.87e-9 J/mol for G/H and 1.95e-12 J/(mol K)
for S/Cp. Six tighter cold reference solves agree within 6.26e-13 in target
ln(x), 1.09e-12 over all species and 5.0e-15 in absolute mole fraction.

Ten additional cold solves trace five fractions of the joint replacement at
one selected condition in each network. The conjugate coordinate is
nonincreasing within the recorded numerical tolerance; individual molecular
abundances need not be monotonic. The finite Gibbs ordering diagnostic is
resolved at all 150 endpoints. These are numerical diagnostics, not rigorous
bounds or evidence that one source is physically exact.

Two independent standalone FastChem comparisons use the same potentials,
network and budgets at selected ion-inclusive conditions. Maximum disagreement
is 1.44e-7 dex. All final solves converge. Across both baseline/updated networks,
maximum relative H/C/O closure is 2.37e-13, nitrogen closure 7.23e-13, charge
closure 3.73e-14 and formation-equation residual 1.43e-12 in ln units. The
all-element relative maximum is 1.40e-6, driven by trace fluorine; it is
reported separately from major-element closure.

An initial passed execution was repeated because a docstring-only change
occurred during model initialization and its import timing was ambiguous.
The final process used the frozen final source bytes throughout. The initial
150 endpoints, six tighter references, ten path solves and two FastChem runs
are recorded separately in ``execution_note.json``; none is added to the
final-evidence counts above. Both networks' saved arrays are bitwise identical
between executions. Old candidate/production receipts are not rewritten.

Cumulative abundance response
-----------------------------

All ratios below are solved mole-fraction ratios, new ``curated-gas-v2`` over
``curated-ch2-v1``, including all 28 corrections simultaneously. They are not
analytic K ratios or the previous isolated 7350 candidate endpoints. Those
prior evaluations had 830 finite controls and 160 FastChem runs and remain
separate evidence. Their common 150-case baseline is not duplicated here.

.. list-table:: Range across the 75 conditions in each fixed network
   :header-rows: 1
   :widths: 30 35 35

   * - Molecule
     - Ion-inclusive minimum / maximum
     - Neutral minimum / maximum
   * - HNCO
     - 1.67887376 / 3.99056995
     - 1.67887376 / 3.99056995
   * - HCO
     - 1.06208179 / 1.17431367
     - 1.06208179 / 1.17431367
   * - NCO
     - 2.61390635 / 12.994517
     - 2.61390635 / 12.994517
   * - CNN
     - 1.1886307 / 1.59299323
     - 1.1886306 / 1.59299323
   * - NCN
     - 1.86520745 / 5.29697344
     - 1.8652073 / 5.29697344
   * - C2
     - 1.33130566 / 2.17502408
     - 1.33130552 / 2.17502408
   * - CNC
     - 6.5387525e-05 / 0.0269656551
     - 6.5387525e-05 / 0.0269656551
   * - C2O
     - 0.000529465524 / 0.0590853438
     - 0.000529465524 / 0.0590853428
   * - C3O2
     - 0.872713144 / 0.95029733
     - 0.872713144 / 0.950297313
   * - C4N2
     - 1.12909654 / 1.41837135
     - 1.12909627 / 1.41837135
   * - HNO
     - 0.543121281 / 0.797805688
     - 0.543121281 / 0.797805718
   * - cis-HNO2
     - 1.02138615 / 1.06113001
     - 1.02138619 / 1.06113001
   * - trans-HNO2
     - 1.00448291 / 1.03124773
     - 1.00448294 / 1.03124776
   * - NH2
     - 1.12988851 / 1.38799826
     - 1.12988851 / 1.38799826
   * - cis-H2N2
     - 0.475969357 / 0.757608512
     - 0.475969357 / 0.757608491
   * - N2H4
     - 0.802825446 / 0.921692105
     - 0.802825446 / 0.92169208
   * - K2
     - 1.00096874 / 1.00258561
     - 1.00096882 / 1.00258561
   * - NO
     - 0.921174064 / 0.972610263
     - 0.921174064 / 0.972610301
   * - NO2
     - 0.91866619 / 0.975374104
     - 0.91866619 / 0.97537422
   * - NO3
     - 0.857522006 / 0.951801985
     - 0.857522006 / 0.951802121
   * - VN
     - 1.95167653 / 5.96337307
     - 1.95167426 / 5.96337307
   * - N2O
     - 0.93568988 / 0.979065197
     - 0.93568988 / 0.97906524
   * - N2O4
     - 0.873202544 / 0.967703261
     - 0.873202544 / 0.967703492
   * - N2O5
     - 0.748490525 / 0.909587876
     - 0.748490525 / 0.909588088
   * - Na2
     - 1.0167157 / 1.04519864
     - 1.01671578 / 1.04519864
   * - O2
     - 1.00254547 / 1.02471047
     - 1.0025455 / 1.0247105
   * - O3
     - 1.03319337 / 1.09211869
     - 1.03319342 / 1.09211869
   * - P4S3
     - 0.0107416512 / 0.182759184
     - 0.0107416512 / 0.182759187

The largest absolute mole-fraction change is C2 at 2500 K, 1e-4 bar and
C/O=1.5: 2.893402e-6 -> 4.522034e-6 in the ion-inclusive network
(ratio 1.562878). Carbon allocated to C2 changes from 0.7727624% to
1.2077333%, an increase of 0.4349709 percentage points. This highlighted case
is distinct from the maximum C2 ratio in the table. The neutral result is
nearly identical. Allocation changes propagate to unmodified species: maximum
absolute changes include N2 at 0.3080184 percentage points of nitrogen and
CS at 0.1934355 percentage points of sulfur. The summary reports conditions,
all 28 target responses and the largest five allocation changes for every
element, without concentration floors. Tiny changes at numerical resolution
are not asserted to be physical redistribution.

Records and remaining physical limits
-------------------------------------

``curated_gas_v2_adoption.yaml`` records exactly three assessments:
``scientific_selection_review`` and ``production_adoption_review`` are
supported; ``physical_accuracy_review`` remains unknown. Source choices,
actual execution inputs/outputs, independent reference construction, packaged
runtime and direct public-alias binding are separately pinned. The direct
alias is the same callable, so its identity check adds no equilibrium solves.

Focused runtime tests passed (6 tests), as did offline production-evidence
replay (6 tests). Full unit, combined documentation and CI results are
recorded at publication; no pending result is represented as passed here.

The limits include correlated energy covariance, incomplete excited-state
and large-amplitude thermals, isotope conventions, and the retained empirical
fit residuals. Source-reported uncertainty widths are not a complete reaction
or atmospheric uncertainty model. Ideal-gas equilibrium also excludes
condensation, transport and photochemical kinetics. These explicit limits
permit the scoped adoption while leaving complete physical accuracy unknown.
