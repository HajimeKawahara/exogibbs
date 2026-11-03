TiO: modern ground energy and partition functions
=================================================

.. note::

   This is the historical source investigation. Its follow-ups were completed
   in :doc:`tio_candidate`, and the resulting model is now adopted as
   :doc:`../presets/curated_tio`. The source-only assessments and numerical
   archive remain attached to the pre-adoption model.

The pre-adoption curated TiO model combines JANAF thermal functions with the
Barklem--Collet (BC) threshold, 6.869 +/- 0.061 eV. It was already a conditional
source choice in ``major_molecule_adoption.yaml``, with normalized ExoMol
partitions explicitly left for follow-up. This investigation advances that
follow-up. **Modern evidence favors developing a replacement; retaining the
present functions is provisional, not a finding that an update is unnecessary.**
No runtime coefficients or curated selection change in this review. The new
``tio_source_review.yaml`` contract supplements the historical adoption record.

Why TiO has priority
--------------------

TiO has strong optical opacity relevant to irradiated planetary atmospheres.
Piette et al. (2020), `DOI 10.1093/mnras/staa1592
<https://doi.org/10.1093/mnras/staa1592>`_, investigate its effect on hot-Jupiter
spectra and inversions. Prinoth et al. (2022), `DOI
10.1038/s41550-021-01581-z <https://doi.org/10.1038/s41550-021-01581-z>`_, report
TiO in WASP-189 b. These motivate accurate thermochemistry; agreement of an
opacity line list with spectra does not validate an equilibrium constant.
Condensation, rainout, cold trapping, irradiation and transport can limit
available gaseous TiO. None is added to the present gas-only model.

The existing 75-case, 537-species ion-inclusive archive assigns up to **92.01%
of titanium** to TiO, at 2000 K, 1e-6 bar and reference C/O = 0.58884; its mole
fraction is 9.4629e-8. Thus a trace total abundance can still represent most of
an element. The archive has no condensation or cold trap. This is a finite
baseline footprint, not an observed abundance or a bound on replacement error.
All archived TiO potentials agree with the actual current quintic hybrid to
7.2e-15 in dimensionless chemical potential, and titanium conservation is checked.

What the sources actually assume
--------------------------------

The printed JANAF entry, pages 1738 and 1813 of the `Oxygen volume
<https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Oxygen.pdf>`_, is evaluated in
December 1973. Its notes put the low singlet a1Delta electronic origin at
**1000 +/- 1000 cm-1** and move the other singlets consistently with that choice.
The current TOTO states place the lowest a1Delta rovibronic level at
**3446.481388 cm-1** above the molecular ground level. The former is an estimated
term parameter, the latter a rovibronic minimum: ZPE and lowest-J corrections
prevent treating their subtraction as an exact transition energy. They do
identify an obsolete low-state assumption requiring a modern thermal treatment.

The source-native atom-balanced D0 values inferred from the pinned tables and
NASA headers are 6.87955 eV (JANAF) and 6.92329 eV (NASA). Header H(298)-H(0) is
used; NASA polynomials are not extrapolated to zero temperature. The current
6.869 eV value is selected from **Luo in BC Table 1**, not a new 2016 measurement.
The Luo original entry and NASA's Gurvich 1982, part 1 p.99/part 2 p.101
ancestry remain uninspected; the header supplies no verified volume number.

Jain et al. (2023), `DOI 10.1063/5.0146365
<https://doi.org/10.1063/5.0146365>`_, obtain **D0(TiO) = 6.824 +/- 0.010 eV** from
cryogenic electron recombination and the molecular ionization threshold.
The public primary full text was inspected, including state and uncertainty
analysis. Its atomic zero is Ti(a3F2) + O(3P2), not a term centroid. Atomic and
ionic fine structure and rotational cooling enter the analysis. The measured
48Ti16O+ beam's TiOH+ contamination is addressed separately. The quoted
uncertainty includes model systematics and is described as effectively one sigma.
This is D0, so neither a second ZPE subtraction nor the paper's separate
-25.1 meV theoretical spin-orbit correction belongs on the experimental value.
The result is compatible with the older uncertainty and provides a more precise
energy candidate; it does not alone validate a hot partition function.

Hildenbrand (1976), `DOI 10.1016/0009-2614(76)80510-6
<https://doi.org/10.1016/0009-2614(76)80510-6>`_, gives 662.7 +/- 6.3 kJ/mol from
exchange equilibria in the inspected publisher abstract. Jain cites this older
measurement; it has not been proven to be the direct ancestor of Luo's entry.
The ionization threshold source is Huang et al. (2013), `DOI
10.1063/1.4803161 <https://doi.org/10.1063/1.4803161>`_; its abstract and citation
were checked, with the numerical threshold taken from Jain's inspected text.

Partition normalization and completeness
----------------------------------------

The official TOTO downloads were acquired on 2026-11-03. Definition files carry
version **20240509**. The original model is McKemmish et al. (2019), `DOI
10.1093/mnras/stz1818 <https://doi.org/10.1093/mnras/stz1818>`_; its modern hybrid
update is McKemmish et al. (2024), `DOI 10.1093/rasti/rzae037
<https://doi.org/10.1093/rasti/rzae037>`_. Published author-hosted full texts,
current definitions and numerical files were inspected.

* Nuclear spin factors for 46,47,48,49,50Ti16O are 1,6,1,8,1. Divide Q by these
  factors before comparison. Raw minor-isotope Q values must not be averaged
  into a chemical partition.
* The 48Ti16O current states have two zero-energy X3Delta, v=0, J=1 parity
  components, each with weight 3. No negative energy is present among 301245
  states. This is a lowest-rovibronic-level zero, not a potential minimum.
* JANAF molecular mass is 63.8794 u; the NASA header uses 63.8664 u; ExoMol's
  48Ti16O definition uses 63.94286092 u. Source molecular Q is recovered from
  its own thermal G-H(0) using the respective translational mass and p0=1 bar.
* The direct current-state sum differs from the distributed PF by less than
  4e-5 over 1500--4000 K. The PF and revised states are not silently treated as
  identical versions. Only compact acquisition-time sums are frozen; replay
  does not regenerate the full line list.
* The 2019 recommended PF includes approximate higher electronic states.
  Their reported contribution is about 0.16% at 3000 K and 1.79% at 5000 K.
  Better line positions and partition completeness have different implications.
  A PF tabulated to 8000 K is not validated there merely by being tabulated;
  line-list metadata specifies 5000 K. Current runtime remains 1500--4000 K.

At fixed molecular mass, the normalized monoisotope Q isolates the internal
thermal change. Replacing JANAF's translational mass by the 48Ti16O mass alone
would multiply the molecular partition contribution by **1.00149054**. This
small factor cannot explain the 12--13% partition difference. It is shown
separately: a physical isotope-specific reaction also needs matching atomic
masses, isotope energy zeros and budgets. A natural-isotope mixture has not
been constructed by either calculation.

.. math::

   \ln Q_{\rm int} = -\frac{G^\circ-H(0)}{RT}
       -\frac32\ln\left(\frac{2\pi m k_B T}{h^2}\right)
       -\ln\left(\frac{k_B T}{p^\circ}\right).

The following replay holds the current NASA atomic thermal functions and JANAF
molecular translational mass fixed. Its energy-only factor is
``exp[(6.824 - 6.869) eV / (k_B T)]``; combined factors multiply this by the
normalized ExoMol/JANAF Q ratio. These are **diagnostics**, not a fitted natural
isotope production model or a finite equilibrium response.

.. list-table:: Independent source and fixed-mass diagnostic ratios
   :header-rows: 1

   * - T (K)
     - Current K / NASA K
     - ExoMol Q / JANAF Q
     - ExoMol Q / NASA Q
     - New D0 only K factor
     - Combined K factor vs current
   * - 1500
     - 0.74440
     - 0.88452
     - 1.00184
     - 0.70600
     - 0.62448
   * - 2000
     - 0.84178
     - 0.86926
     - 1.00237
     - 0.77020
     - 0.66951
   * - 3000
     - 0.94039
     - 0.86600
     - 1.00439
     - 0.84024
     - 0.72765
   * - 4000
     - 0.98060
     - 0.87949
     - 1.00924
     - 0.87761
     - 0.77185

The modern Q resembles NASA's molecular thermal functions, but the new D0 is
weaker than NASA's source energy. The combined diagnostic therefore moves
**farther from the historical NASA equilibrium constant** (ratios 0.46486 and
0.68427 at 1500 and 3000 K). NASA agreement is not the scientific selection rule.
The 0.010 eV energy uncertainty alone gives multiplicative factors 1.0804 and
1.0394 at those temperatures, not total thermal or abundance uncertainty.

Decision and bounded next step
------------------------------

The evidence supports a preferred direction: modern normalized TiO thermal
functions and the Jain ground energy, with an explicit isotope convention.
Curated remains unchanged because this review has not supplied and tested that
complete numerical candidate. It is not held for every historical paper.

1. Specify a natural-isotope approximation or an explicitly scoped 48Ti16O
   candidate; reconcile distributed PF and current states, preserving their
   zeros, degeneracies and completeness limits. A NASA-thermal proxy can also
   be compared against the full modern Q without adopting NASA's old energy.
2. Build a minimal derivative-consistent G/H/S/Cp representation on
   1500--4000 K; check interpolation and uncertainties. Apply the 6.824 eV D0
   in the existing common neutral-atom gauge without reapplying ZPE/spin shifts.
3. Run a controlled finite substitution on the fixed network, checking
   conserved Ti/O budgets, mass action, abundance response and failures.
   Then record the adoption assessment. Condensation and radiative transfer
   accuracy remain separate questions.

No additional document is a prerequisite for merging this source investigation.
For the candidate, the 2019/2024 ExoMol and 2023 Jain primary texts are already
inspected and available publicly. Huang's full text can tighten the threshold
provenance; Luo and Gurvich originals are historical traceability follow-ups,
not reasons to postpone a modern candidate indefinitely.

Offline replay
--------------

The compact packet in ``tests/data/validation/tio_source_review/20261103`` pins
source excerpts, downloaded whole-file hashes, source-inspection levels, output
and selected implementation files. It reuses the existing major-molecule
reference and baseline rather than duplicating numerical databases::

   PYTHONPATH=src python mceg/reviews/tio_source_review.py
   pytest tests/model_evidence/tio_source_review_test.py

MCEG separates supported source attribution, supported candidate direction and
numerical replay from **UNKNOWN physical accuracy/adoption readiness**. Passing
file integrity and replay checks does not convert that unknown into validation.
