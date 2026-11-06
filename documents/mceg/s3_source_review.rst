S3 source review and bounded sulfur-vapor interpretation
========================================================

Retain the current S3 function in this source-only review. Its JANAF origin
is an estimated effective component of sulfur vapor, while modern theory
provides a route to explicit isomer thermodynamics. That route requires an
individual-species energy and thermal construction before adoption. The
75-case hot H-rich archive contains at most 5.81431019e-08%
of sulfur in S3; this is a priority indicator, not an accuracy bound.

Identity and the actual inherited model
---------------------------------------

``S3`` is neutral gas with 3 sulfur atoms. It is not a solid allotrope,
an ion or the chemical potential of a mixed sulfur vapor. Official JANAF
S-016 and printed page 1874 use the September 1977 evaluation.
The selected Hf298 is 141.461 +/-8 kJ/mol and S298 is 269.517 +/-4 J/mol/K.
The source estimates vibrational heat capacity per vibrational degree of
freedom using S/S2/S8 and fits third-law energies to vapor measurements. It
does not resolve every structural isomer. The NASA record cites Gurvich 1989;
shared source ancestry is not independent physical confirmation.

The printed 298.15 K free-energy-function cell is 304.899 J/mol/K,
although H298-H298=0 requires it to equal S298=269.517. Both the printed
page and official machine table contain this inconsistency. Preserve the
raw cell and reconstruct G from H and S, avoiding this isolated cell. No
blank 0 K molecular value is fabricated: the source decomposition aligns
**H298**, not physical D0. Formation-column closure at the four hot nodes is
checked separately, including the precision of the printed numbers.

Native and aligned comparison
-----------------------------

.. list-table:: Formation K at 1 bar
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - JANAF / NASA
     - H298-aligned JANAF / NASA
   * - 1500
     - 0.52962035
     - 0.52491988
     - 0.42235831
   * - 2000
     - 0.50659104
     - 0.50781273
     - 0.43141436
   * - 3000
     - 0.52268517
     - 0.52667579
     - 0.47242989
   * - 4000
     - 0.58649858
     - 0.58483494
     - 0.53904972

NASA uses its native R=8.314510 J/mol/K; JANAF reconstruction uses the SI
value. The packet records molecular and balanced atomic-formation G/H/S/Cp,
energy/thermal closure and analytic derivatives of the literal FastChem fit.
The isolated H298 +/-8 kJ/mol and S298 +/-4 J/mol/K sensitivities are recorded
separately; they are not independent Gaussian errors or a complete covariance.

Modern evidence and next calculation
------------------------------------

The complete author manuscript of `Jackson et al. (2016)
<https://doi.org/10.1039/C5SC03088A>`_ provides a global structural search and
ideal-gas RRHO thermodynamics calibrated to S2/S8 vibrational information.
Its species calculations and experimental S8 energy reference must be
reconstructed separately from its final mixed-vapor chemical potential.
The latter cannot replace one species function in the current network.
The complete author manuscript of `Fedyaeva et al. (2023)
<https://doi.org/10.1039/D2CP05498A>`_ adds USPEX/PBE search with
B3LYP/6-311+G(d,p) refinement. Its finite-temperature calculations extend only
to 1500 K; they do not establish a 4000 K validity range.

The 2016 search identifies a bent C2v ground structure and a triangular D3h
alternative about 0.5 eV higher. The packet evaluates exp(-0.5 eV/kT) only
as an equal-thermal illustrative weight: it is about 0.0209 at 1500 K and
0.2344 at 4000 K. Different symmetries, rotational/vibrational functions and
uncertain relative energies prevent treating it as an isomer population or
formation-K correction. The 2026 S3/S4 oxidation paper, DOI
10.1080/17415993.2026.2659134, was found only as publisher/Crossref metadata;
its full text supplies no inspected numerical input here.

The original vapor-pressure paper, `DOI 10.1016/S0021-9614(73)80045-X
<https://doi.org/10.1016/S0021-9614(73)80045-X>`_, is represented by its primary
abstract and explicit JANAF attribution; its complete text was not obtained.
This access gap is documented without making every ancestral paper a merge
gate. A useful next candidate would collect each low-lying isomer's geometry,
frequencies, symmetry, energy and degeneracy; align one energy reference;
then compare consistent thermal derivatives and high-temperature truncation.
No new numerical modern candidate or finite-network substitution is claimed.

Replay and scope
----------------

The maximum sulfur share occurs at 1500 K,
0.0001 bar and C/O=0.588844; maximum mole fraction
is 4.38193189e-15. All 75 archived endpoints converge and the
actual named ``curated_tio2`` function agrees with the literal fit. This
finite hot H-rich result does not validate cold, sulfur-rich, oxidizing or
photochemical conditions. Complete physical uncertainty remains unknown.

Run ``python mceg/reviews/s3_source_review.py`` and
``pytest tests/model_evidence/s3_source_review_test.py``. Evidence is in
``tests/data/validation/s3_source_review/20261106``; the MCEG contract is
``mceg/contracts/s3_source_review.yaml``. Inputs are hash-pinned with
historical resolution, outputs directly pinned. No curated coefficients,
new equilibrium solve or full manuscript are included.
