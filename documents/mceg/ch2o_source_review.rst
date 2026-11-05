Formaldehyde source review
==========================

Decision and identity
---------------------

Keep the production coefficients unchanged in this source review. Prioritize an
ATcT energy-only candidate, with a separate NASA-thermal control, before a finite
replacement calculation. This is neutral formaldehyde (methanal, C=O, ground
singlet; CAS 50-00-0), not a hydroxymethylene isomer. The runtime species is
``C1H2O1`` and the original CEA record is ``HCHO,formaldehy``.

The inherited FastChem five-coefficient fit cites JANAF. Its formation ln K and
analytic H/S/Cp are checked against the named ``curated_cos`` factory. Current
curated/NASA ratios are reproduced; no production potential, network, budget,
standard pressure or domain is changed.

Energy and thermal diagnosis
----------------------------

The official JANAF table uses Hf(298) = -115.897 kJ/mol. NIST WebBook dates that
assessment to March 1961. The CEA header instead uses -108.580 kJ/mol and cites
TRC (1978) and Gurvich (1991). The modern ATcT 1.228 value is -109.226 kJ/mol.
The da Silva et al. 2006 author abstract independently describes the historical
-26.0 versus -27.7 kcal/mol dispute and favors the less negative branch; its
full article was not obtained. It is corroborating context, not a second
independent uncertainty estimate for ATcT.

Balance C(g) + 2 H(g) + O(g) -> CH2O(g), including atoms in each source version.
This gives D0 = 1502.104 kJ/mol (JANAF), 1494.796222 (CEA), and 1495.667
(ATcT 1.228). The latter uses Hf(0) = 711.370, 216.034, 246.843 and -105.386
kJ/mol for C/H/O/CH2O. The proposed energy change is therefore -6.437 kJ/mol
relative to JANAF. It weakens formation, increasing reaction H/G by 6.437 kJ/mol
if applied alone, with S/Cp retained. The printed molecular uncertainty
0.078 kJ/mol is not a balanced-reaction uncertainty: covariance is not supplied.

Here JANAF D0 is reconstructed from Hf(298) and H(0)-H(298) for all species.
Balancing its separately rounded zero-K formation-enthalpy column instead gives
1502.102 kJ/mol, a 2 J/mol rounding difference (and a -6.435 kJ/mol correction).
The replay keeps the former path consistently; printed rounding is not a new
physical uncertainty estimate.

NASA's native R = 8.314510 is distinguished from SI R = 8.31446261815324.
No NASA polynomial is extrapolated to zero kelvin: H(0) uses the printed
Hf(298) and H(298)-H(0). The original molecular low interval begins at 300 K;
the older screening JSON has the same coefficients but declares 200 K. This
review preserves the original 300 K bound and samples NASA only at or above it.
JANAF comparisons use H/S columns and an explicit energy/thermal decomposition.
The original historical fits are not claimed to be fully reconstructed.

.. list-table:: Formation constants; these are not abundance response factors
   :header-rows: 1

   * - T (K)
     - Current / CEA
     - ATcT energy-only / current
     - CEA thermal + ATcT / current
   * - 1500
     - 1.88688625
     - 0.59682654
     - 0.56868726
   * - 3000
     - 1.35702802
     - 0.77254549
     - 0.76334488
   * - 4000
     - 1.29152337
     - 0.82403005
     - 0.79502345

At 1500 K, JANAF minus NASA ln K separates into +0.58663213 (energy),
+0.02531694 (molecular thermal) and +0.00032438 (atomic thermal). Thus energy is
the main source difference, with a smaller thermal contribution. The inherited
fit itself has a sampled reaction-Cp residual as large as 3.545607 J/mol/K
relative to the original JANAF nodes. A NASA-thermal control therefore has a
purpose; reproducing a formation constant alone does not verify its derivatives.

AYTY partition diagnostics
--------------------------

The 2015 AYTY author manuscript states a 1500 K spectral applicability limit,
uses states up to 18000 cm-1 and discusses missing high-energy contributions.
Its polynomial/table extends to 3000 K; extension does not establish completeness.
The downloaded definition (version 20230430) labels 3000 K. These different
limits are retained explicitly. The 2021 MARVEL update improves empirical line
positions; it does not establish complete thermodynamic sums through 4000 K.

The downloaded PF is pinned rather than identified with rounded 2015 table
values: at 1500 K it is 130188.594 versus the paper's 128635.40. The distribution
page identifies MARVELized states but does not independently trace every PF
revision. The current PF has no points above 3000 K. It is a source diagnostic,
not a replacement thermal model.

Divide nuclear-spin-weighted Q by four (two proton spins; symmetry weights
1,1,3,3). At 1500/3000 K its spin-removed Q/NASA is 1.02838396/0.93034265.
Isotope mass 30.01056468 u versus natural-mass 30.02598 u gives the additional
translation factor (m_iso/m_nat)**1.5; the resulting ratios are
1.02759210/0.92962629. This remains a principal-isotope proxy, not a complete
natural-isotope mixture. PF Cp diagnostics use two finite-difference steps and
explicit tabulation-rounding bounds; truncation and missing states are separate.
No extrapolation or endpoint finite-difference stencil is used.

Archived atmospheric footprint
-------------------------------

The named curated-cos-v1 archive contains 75 ion-inclusive (537 species) and
75 neutral (421 species) endpoints at 1500–4000 K, 1e-6–100 bar and three C/O
ratios. All converged; the target potential matches the current factory and
independent fit. At 2500 K, 100 bar and reference C/O, the maximum carbon share
in formaldehyde is 4.64270e-5 (0.00464270 percent) in both networks, with mole
fraction about 2.29582e-8. These are existing endpoints, not new solves or
energy-substitution responses. The hot gas grid does not determine low-temperature
photochemical relevance, transport chemistry, or non-solar atmospheric behavior.

Reproduction and follow-ups
------------------------------

Run offline with float64 enabled by the script::

   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/ch2o_source_review.py
   PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/model_evidence/ch2o_source_review_test.py

``tests/data/validation/ch2o_source_review/20261105`` records source access,
selected data, the actual-input receipt and comparison. The MCEG contract is
``mceg/contracts/ch2o_source_review.yaml``. Numerical/source support and the
limited candidate recommendation are separate from unknown full physical accuracy.

Next, construct the energy-only continuous candidate on the current background,
compare the NASA-thermal control, and perform fixed-network substitutions with
conservation and convergence checks. All essential energy data are available;
no missing article is a mandatory gate for that bounded calculation. A broader
AYTY thermal adoption requires an independently converged partition sum or other
validated high-temperature thermal assessment, including derivatives and state
completeness. Recover original JANAF/TRC derivations only if a broader historical
provenance claim is needed. Those follow-ups do not block merging this review.

Sources
-------

* `NIST JANAF C-061 <https://janaf.nist.gov/tables/C-061.html>`_ and
  `NIST WebBook <https://webbook.nist.gov/cgi/cbook.cgi?ID=C50000&Units=SI&Mask=1>`_.
* `ATcT 1.228 formaldehyde <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=39>`_
  and the same-version neutral atomic entries.
* da Silva et al. (2006), DOI `10.1002/cphc.200500667 <https://doi.org/10.1002/cphc.200500667>`_
  (author abstract inspected; full text optional for expanded provenance).
* Al-Refaie et al. (2015), DOI `10.1093/mnras/stv091 <https://doi.org/10.1093/mnras/stv091>`_
  (`2015 author manuscript <https://arxiv.org/html/1506.00172>`_, full text inspected).
* Al-Derzi et al. (2021), DOI `10.1016/j.jqsrt.2021.107563 <https://doi.org/10.1016/j.jqsrt.2021.107563>`_
  (`2021 author manuscript <https://arxiv.org/html/2102.07423>`_, full text inspected).
* `ExoMol AYTY data <https://www.exomol.com/data/molecules/H2CO/1H2-12C-16O/AYTY/>`_,
  numeric distribution retrieved 2026-11-05; hashes identify the inspected bytes.
