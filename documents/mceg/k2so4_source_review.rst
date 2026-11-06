K2SO4 gas identity and source review
====================================

Retain the current K2SO4 function. Official JANAF and the inspected 2011 mass
spectrometry support the neutral gas identity, but do not establish a complete
modern replacement. The old model is reproducible and its uncertainties are
explicit. The current 75-case hot H-rich archive holds at most
9.67493933e-22% of K in this molecule.

Actual gas and inherited thermal construction
---------------------------------------------

``K2O4S1`` is neutral K2SO4(g), not condensed sulfate or free
SO4(2-). The printed JANAF page 1485, evaluated June 1978, uses D2d symmetry,
sigma=4 and ground weight 1. Its tetrahedral sulfate unit bridges two metal
atoms. The 15 vibrational degrees include measured spectral constraints and
estimated low-frequency modes. Matrix spectra are not direct unperturbed gas
fundamentals. The packet pins all printed modes, mass and inertia product and
reconstructs the nonlinear ground-state RRHO thermal functions; this verifies
the source construction, not independent physical accuracy.

The selected Hf298 is -1094.116 +/-16.7 kJ/mol and the quoted S298
uncertainty is +/-16.7 J/mol/K. JANAF evaluates molecular vaporization separately
from decomposition and corrects experimental total pressures. The NASA header
cites Gurvich 1982. On balanced neutral-atom references JANAF D0 differs from
NASA by -1.751119 kJ/mol. That physical atomization difference is kept
separate from the molecular/atomic thermal terms and the fitted intercept.

.. list-table:: Formation K at 1 bar
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - JANAF / NASA
     - D0-aligned JANAF / NASA
   * - 1500
     - 1.04560082
     - 1.02436272
     - 1.17741910
   * - 2000
     - 1.06087588
     - 1.07008530
     - 1.18789042
   * - 3000
     - 1.10114826
     - 1.11771510
     - 1.19831163
   * - 4000
     - 1.15207896
     - 1.14324875
     - 1.20453584

The packet contains molecular and formation G/H/S/Cp and exact decomposition
closure, retaining NASA's native R=8.314510 and the SI R for JANAF. Analytic
formation-fit derivatives are checked against the actual named production
factory. Isolated H298 and S298 uncertainty factors are sensitivities, without
an invented covariance or a claim of quantified planetary abundance error.

What the newer experiment establishes
-------------------------------------

The complete primary article of `Armatys et al. (2011)
<https://doi.org/10.2478/s13536-011-0016-6>`_ identifies molecular sulfate
parents using both parent and oxide fragment ions. Over the pure sulfate at
1350 K it reports p(K2SO4)=0.12 Pa, with mean oxide/parent ion ratio
1.13 +/-0.14. It explicitly describes its pressures as approximate:
70 eV ionization permits neglected fragmentation channels and molecular cross
sections are estimated. Thus these pressures cannot be treated as pure
neutral-atom formation constants or used directly as atmospheric abundance
validation. The experiment also includes condensed-solution activities and
mixed NaKSO4, outside this single-gas source comparison.

The identified next energy sources are the original `1979 K2SO4 study
<https://doi.org/10.1149/1.2129068>`_ and `1985 alkali-sulfate study
<https://doi.org/10.1149/1.2113719>`_. Their bibliographic/abstract scope and the
2011 citations were inspected; their complete texts were not obtained. A future
energy update should reconcile molecular sublimation with a consistent
condensed-phase anchor, dissociation and fragmentation treatment. A separate
thermal update should resolve the estimated soft modes and high-temperature
anharmonic contributions. These focused follow-ups are not an unconditional
requirement to obtain all historical ancestors before merging this review.

Footprint and verification
--------------------------

The maximum K share occurs at 1500 K,
100 bar, C/O=0.588844. Maximum mole fraction is
9.76561489e-31. All 75 archived curated-tio2-v1 endpoints
converge with 537 species. This trace hot H-rich abundance does not establish
behavior in oxidizing, cold, sulfate-rich or condensed-phase environments.
No finite-network substitution or curated change is made.

Run ``python mceg/reviews/k2so4_source_review.py`` and
``pytest tests/model_evidence/k2so4_source_review_test.py``. The packet is
``tests/data/validation/k2so4_source_review/20261106``; the MCEG contract is
``mceg/contracts/k2so4_source_review.yaml``. Inputs use historical hash
resolution and outputs use direct hashes. Full physical accuracy remains
unknown, separately from the supported scoped source/retention assessments.
