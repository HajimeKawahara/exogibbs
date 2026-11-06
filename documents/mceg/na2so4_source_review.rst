Na2SO4 gas identity and source review
=====================================

Retain the current Na2SO4 function. Official JANAF and the inspected 2011 mass
spectrometry support the neutral gas identity, but do not establish a complete
modern replacement. The old model is reproducible and its uncertainties are
explicit. The current 75-case hot H-rich archive holds at most
1.17806265e-21% of Na in this molecule.

Actual gas and inherited thermal construction
---------------------------------------------

``Na2O4S1`` is neutral Na2SO4(g), not condensed sulfate or free
SO4(2-). The printed JANAF page 1669, evaluated June 1978, uses D2d symmetry,
sigma=4 and ground weight 1. Its tetrahedral sulfate unit bridges two metal
atoms. The 15 vibrational degrees include measured spectral constraints and
estimated low-frequency modes. Matrix spectra are not direct unperturbed gas
fundamentals. The packet pins all printed modes, mass and inertia product and
reconstructs the nonlinear ground-state RRHO thermal functions; this verifies
the source construction, not independent physical accuracy.

The selected Hf298 is -1033.615 +/-25.1 kJ/mol and the quoted S298
uncertainty is +/-16.7 J/mol/K. JANAF evaluates molecular vaporization separately
from decomposition and corrects experimental total pressures. The NASA header
cites Gurvich 1982. On balanced neutral-atom references JANAF D0 differs from
NASA by -6.921119 kJ/mol. That physical atomization difference is kept
separate from the molecular/atomic thermal terms and the fitted intercept.

.. list-table:: Formation K at 1 bar
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - JANAF / NASA
     - D0-aligned JANAF / NASA
   * - 1500
     - 0.71605132
     - 0.70014575
     - 1.21815238
   * - 2000
     - 0.80160242
     - 0.80988442
     - 1.22689647
   * - 3000
     - 0.92106830
     - 0.93677719
     - 1.23564265
   * - 4000
     - 1.01694065
     - 1.00763091
     - 1.24020626

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
1350 K it reports p(Na2SO4)=0.0019 Pa, with mean oxide/parent ion ratio
1.3 +/-0.12. It explicitly describes its pressures as approximate:
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

The maximum Na share occurs at 1500 K,
100 bar, C/O=0.588844. Maximum mole fraction is
1.67965361e-29. All 75 archived curated-tio2-v1 endpoints
converge with 537 species. This trace hot H-rich abundance does not establish
behavior in oxidizing, cold, sulfate-rich or condensed-phase environments.
No finite-network substitution or curated change is made.

Run ``python mceg/reviews/na2so4_source_review.py`` and
``pytest tests/model_evidence/na2so4_source_review_test.py``. The packet is
``tests/data/validation/na2so4_source_review/20261106``; the MCEG contract is
``mceg/contracts/na2so4_source_review.yaml``. Inputs use historical hash
resolution and outputs use direct hashes. Full physical accuracy remains
unknown, separately from the supported scoped source/retention assessments.
