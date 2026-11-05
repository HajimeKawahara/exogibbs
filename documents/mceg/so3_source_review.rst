SO3: retain the adopted thermal model
=====================================

The current ``O3S1`` reaction is reproduced by NASA thermal functions with
the balanced ATcT 1.172 atomization energy. Retain it in this source review.
The newer evaluated energy makes a small change, and the available ExoMol
partition function does not cover the production 1500--4000 K interval.

Energy and source attribution
-----------------------------

The inherited balanced D0 is 1407.112 kJ/mol. The inspected
`ATcT 1.228 network <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
has Hf(0) values S=276.85, O=246.843 and SO3=-389.68 kJ/mol,
giving D0=1407.059 kJ/mol. Its -0.053 kJ/mol shift is smaller in magnitude
than the reported SO3 formation-energy margin of 0.22 kJ/mol; this is not
an independent statistical significance test. It alone changes K by less
than 0.43 percent in the reviewed production interval.

.. list-table:: Dimensionless formation-K diagnostics
   :header-rows: 1

   * - Temperature (K)
     - Current/native NASA
     - ATcT 1.228 energy only/current
     - JANAF/native NASA
   * - 1500
     - 1.143295
     - 0.995759
     - 1.003206
   * - 3000
     - 1.069250
     - 0.997877
     - 1.016794
   * - 4000
     - 1.051500
     - 0.998408
     - 1.020123

The NASA header explicitly cites Gurvich (1989), part 1 page 292 and
part 2 page 177; its Hf(298) is -395.900 kJ/mol. Original book inputs were
not independently recovered. `JANAF O-058 <https://janaf.nist.gov/tables/O-058.txt>`_
is a separate September 1965 evaluation, not the adopted thermal model.
Its balanced D0 is 1405.129 kJ/mol, versus native NASA 1405.449874.
After removing the energy term, the JANAF/NASA combined molecular and atomic
thermal factors are 1.028691, 1.029628 and 1.029764 at the three temperatures.
The corresponding source decomposition is archived; a difference between
current and JANAF is not automatically a fitting error.

Partition function and scope
----------------------------

The full `UYT2 paper <https://doi.org/10.1093/mnras/stw1828>`_ and its
1--1000 K partition table were inspected. Its line-list completeness claim
is for spectra up to 800 K. For principal 32S16O3, nuclear spins are zero;
no extra spin or symmetry divisor is applied. Ground-relative Q/NASA at
1000 K is 1.003805. Accounting for translation with isotope mass
79.95681584 u rather than NASA's natural mass 80.0632 u gives a fixed-atom
K proxy of 1.001805. This is an elemental proxy, not an isotope mixture.
No Q beyond the distribution's 1000 K limit is extrapolated. Agreement here
does not validate a replacement thermal function at 1500--4000 K.

The archived ``curated-cos-v1`` ion-inclusive gas grid contains 75 converged
cases and 537 species. SO3's maximum sulfur inventory fraction is
2.64225e-15, at 4000 K, 100 bar and C/O=0.588844; its maximum mole fraction
is 5.57078e-20. This hot-grid footprint is not an abundance response to a
replacement, and does not cover cooler, oxidizing or photochemical
atmospheres.

Follow-up
---------

An energy-only candidate can be calculated immediately from the pinned
ATcT values if desired; its small size does not justify changing curated
without that separate evaluation. A thermal replacement needs complete
high-temperature state sums and caloric checks. No missing paper is a
merge blocker. The UYT2 DOI above is available in full text; the old Gurvich
volumes are an optional provenance follow-up.

Reproducibility and evidence boundary
-------------------------------------

All source comparisons use one bar and separate the balanced atomization
energy from molecular and atomic thermal functions. NASA polynomials retain
R=8.314510 J/mol/K; tabulated SI functions and the production reaction use
R=8.31446261815324. The zero-temperature gauge is obtained from each source's
fixed Hf(298.15) and H(298.15)-H(0), with identical treatment of all atoms.
ATcT energies use molecule and atoms from version 1.228 together. Reported
network uncertainties are not independent errors; no covariance propagation
or complete physical uncertainty is claimed.

The offline script reproduces literal source nodes and coefficients, with
no fitted replacement or new equilibrium solve. Tests independently compare
the inherited function with the named ``curated_cos`` factory, verify
thermodynamic identities, energy/thermal closure and archived stoichiometry.
MCEG supports attribution, normalized comparison and the scoped decision;
complete physical accuracy and recovery of every historical input remain
unknown. Historical recovery is a follow-up, not a prerequisite to merge
this bounded review. Production curated coefficients are unchanged.

Run ``python mceg/reviews/so3_source_review.py`` from the repository root.
Pinned inputs, source-access records, numerical output and a hash receipt are
in ``tests/data/validation/so3_source_review/20261105``.
