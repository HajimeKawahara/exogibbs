Na2 dissociation and finite-state thermal source review
=======================================================

Retain production in this source review. Measured D0 and modern partition
information support a follow-up candidate, but neither the NASA fit nor the
Barklem-Collet compilation should be substituted wholesale. The high-temperature
thermal disagreement requires explicit finite-level and electronic-state checks.
No curated coefficient or new equilibrium endpoint changes here.

Identity and primary sources
----------------------------

``Na2`` denotes neutral homonuclear Na2 gas, ground X 1Sigma_g+,
not atomic Na, a cation or a condensed phase. The source-native NASA record is
``Gurvich,1982 pt1 p314 pt2 p357.``. Official JANAF molecule/atom tables and
the exact NASA coefficients are pinned. Their native R values remain distinct:
8.314510 (NASA) versus SI 8.31446261815324 J/mol/K, both at 1 bar.

Direct spectroscopy, `DOI 10.1103/PhysRevA.54.R1006 <https://doi.org/10.1103/PhysRevA.54.R1006>`_, reports
D0=5942.6880000 +/- 0.0049 cm-1,
or 71.090335610 kJ/mol. This is the v=0,J=0-to-atom
energy, not the potential-well depth De. Primary publisher abstract inspected; full original paper not obtained.
The 1996 direct experiment combines vibrational, electronic and photoassociation
intervals to obtain D0 of 23Na2. Its publisher abstract was inspected, but the
complete original uncertainty analysis was not obtained. Natural sodium is
monoisotopic for this thermodynamic purpose; the global nuclear-spin factor is
still excluded under the thermochemical convention.

Barklem and Collet, `DOI 10.1051/0004-6361/201526961
<https://doi.org/10.1051/0004-6361/201526961>`_, were inspected in full together
with their author data at revision ``cd2f54ed46687659017b343763e95478ed93653e``.
The 2022 atomic and equilibrium-table corrections are included; molecular Q is
unchanged. The pinned molecule state file has eight listed states and no shallow
a 3Sigma_u+ state. Its older spectroscopic constants and asymptotic level
truncation do not establish complete modern high-temperature accuracy. Nuclear
weights are already divided by the product of nuclear-spin degeneracies: do not
divide Q by another factor of 16 or apply symmetry number 2 a second time.

The 2025 electronic-potential study,
`DOI 10.1063/5.0239463 <https://doi.org/10.1063/5.0239463>`_, was inspected at
primary indexed-abstract level; the advertised author manuscript returned 404.
It treats several dissociation channels and distinguishes De from D0. It is not
a reason to replace direct measured D0 by a less accurate theoretical well depth.

The full author preprint of the 2025 thermal study,
`DOI 10.1007/s10765-025-03628-z <https://doi.org/10.1007/s10765-025-03628-z>`_,
and the final publisher abstract were inspected. It uses 1 atm and excludes
shallow triplet and quasibound/continuum contributions in its quantum model.
The publisher links a correction,
`DOI 10.1007/s10765-025-03639-w <https://doi.org/10.1007/s10765-025-03639-w>`_.
The corrected final tables and correction text were not obtained. Its 1-atm
entropy requires +R ln(101325/100000) at 1 bar, and its preprint discussion of
JANAF nuclear weights cannot justify adding an extra spin entropy. No numerical
candidate is adopted from the preprint. These are finite checks for a follow-up,
not a demand for every ancestral paper before merging the source review.

Replayed energy, thermal and caloric diagnostics
------------------------------------------------

Physical JANAF D0 is 70.539000 kJ/mol;
NASA's balanced physical D0 is 70.669144 kJ/mol.
The measured-minus-JANAF offset is 0.551336
kJ/mol, producing an energy-only K factor
1.04519864 at 1500 K and
1.01671578 at 4000 K.
This factor preserves current thermals and fitted-proxy residuals; it is not an
implemented candidate or a finite abundance response.

.. list-table:: Formation K ratios at 1 bar
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - D0-aligned JANAF / NASA
     - BC / current
     - D0-aligned BC / JANAF
   * - 1500
     - 1.00706746
     - 1.01273944
     - 1.066632
     - 1.025398
   * - 2000
     - 0.99501144
     - 1.01289881
     - 1.175122
     - 1.125467
   * - 3000
     - 1.01695990
     - 1.02456669
     - 1.506643
     - 1.470390
   * - 4000
     - 1.05855190
     - 1.05429688
     - 1.828534
     - 1.812785

BC publishes log10 of dissociation pressure K in Pa. Conversion to the present
dimensionless formation K is ln(K)=ln(100000)-ln(10)*log10(pK). Common-D0 alignment
then removes (D0(BC)-D0(JANAF))/(RT). The BC input D0 is
71.090393 kJ/mol; it is not silently replaced by the direct
measurement. This balanced K diagnostic includes molecular and atomic thermals.

The packet additionally reconstructs an effective JANAF internal Q from its
H-H0 and S with the source molecular mass, and compares the published BC Q.
It keeps source mass fixed to expose thermal shape; it is not a natural-isotope
sum or a newly differentiated partition model. Four Q nodes alone do not define
trustworthy H/S/Cp derivatives. Native JANAF/NASA molecular and reaction G/H/S/Cp,
literal current derivatives and energy/molecular/atomic decomposition are all
recorded independently in ``comparison.json``.

Importance and next calculation
-------------------------------

The final ``curated-tio2-v1`` archive has 75 converged, 537-species ion endpoints.
Maximum Na share in Na2 is 0.001172457559%, at
1500 K, 100 bar and C/O=1.5.
This small hot H-rich inventory is not a general constraint on alkali-rich,
cooler or non-LTE atmospheres. It supports lower priority than the larger CN,
PS, TiO2 and cyanide issues while retaining a concrete scientific follow-up.

Construct a derivative-consistent finite-state candidate using measured D0,
relevant electronic surfaces and an explicit bound/quasibound convention. Check
the corrected 2025 tables and isotope/spin normalization before finite-network
substitution. Complete physical uncertainty and planetary abundance accuracy
remain unknown. Run ``python mceg/reviews/na2_source_review.py`` and
``pytest tests/model_evidence/na2_source_review_test.py``. The packet and
contract are ``tests/data/validation/na2_source_review/20261106`` and
``mceg/contracts/na2_source_review.yaml``. All replay inputs are pinned via
the historical resolver and outputs directly. No full paper is committed.
