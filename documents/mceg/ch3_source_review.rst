CH3 source review (2026-11-06)
==============================

Decision and molecular evidence
-------------------------------

The recommended next calculation is a **bounded energy-only ATcT candidate**,
followed by fixed-network substitutions. The inherited JANAF atomization energy
is 1210.260 kJ/mol; same-version ATcT 1.228 C + 3 H - CH3 gives
1209.580 kJ/mol, lower by 0.680 kJ/mol. The same-thermal K factors are
0.94694, 0.97311 and 0.97976 at 1500, 3000 and 4000 K. This modest correction
can be calculated without waiting for a new high-temperature spectral model.
Curated remains unchanged in this source PR.

NASA identifies its energy input as Ruscic 1999 D0(H3C-H), with Jacox 1998
spectroscopy. The H3C-H threshold concerns methane dissociation; it is not the
total methyl-to-atoms energy. Here all comparisons use C + 3 H - CH3 and the
same atomic reference version. CH3 is the neutral planar X2A2-double-prime
methyl radical, not CH3+, methane, or a singlet state. Atomization is a
rovibronic ground-level D0, not De; ZPE must not be added again.

`Adam et al. 2019 <https://doi.org/10.1021/acs.jpca.9b02919>`_ gives the AYYJ
ground-electronic-state line list with 9,127,123 rovibrational states and
J <= 40. Total degeneracy includes the doublet factor 2 and D3h(M)
nuclear-spin weights 4,0,2,4,0,2. Divide the aggregate PF by the total proton
nuclear-spin multiplicity 8 to compare with thermochemical conventions;
retain electron spin, and do not divide again by a classical rotational
symmetry number. This assumes equilibrium spin populations, not separately
conserved nuclear-spin species.

The actual PF file runs from 1 to **2000 K**, and is not extrapolated.
At 1500/2000 K its spin-normalized Q is 1.565%/1.864% below NASA; the
principal-isotope translation proxy changes these comparisons by a further
approximately 0.110%. The source's 1500 K recommendation is based on the
95% population coverage of lower states below 9000 cm-1 for intensities.
That is distinct from convergence of the larger PF sum: neither the line-list
recommendation nor a close Q value proves accurate H, S or Cp through 4000 K.
The 9-million-state file was not ingested in this review. Derivatives of the
rounded PF table are not presented as physical thermal functions.

After energy alignment, JANAF/NASA K still differs by about 3.60-4.36% over
1500-4000 K. The current fit also departs from the source reaction by up to
2.320 kJ/mol in H at the sampled 4000 K endpoint. A NASA-thermal/ATcT diagnostic
would reduce K by 10.05% at 1500 K and 6.41% at 4000 K relative to current,
larger than the energy-only change; choosing that route requires a caloric
comparison. A future AYYJ thermal replacement specifically needs converged
high-temperature state moments, sensitivity to energy/J cutoffs and omitted
electronic states, and a controlled fit. These bounded checks do not require
recovering every historical NASA antecedent before merging the source record.

Source comparison and normalization
-----------------------------------

The current ``curated-cos-v1`` entry is the inherited FastChem JANAF 1998 fit,
marked unreviewed in the source screening. Its literal coefficients and exact
molecular identity are checked against the named production preset. Official
JANAF molecular/atomic tables and selected NASA CEA records are pinned. NASA
uses native R = 8.314510 J/mol/K; JANAF and current fits use
R = 8.31446261815324. Both use one bar. The official C, CN and CH NASA lower
bound is 300 K; this review does not copy the modified screening file's
200 K extension. Selected polynomial coefficients do agree exactly.

NASA H(298)-H(0) and JANAF H(T)-H(298) produce compatible fixed-reference
zero-energy offsets. Their individual offsets are not relabelled as physical
formation enthalpies at 0 K; only the **atom-balanced difference** is a D0.
ATcT molecule and atoms all come from version 1.228. Marginal network
uncertainties are not combined as independent errors because covariance is
unavailable. Agreement with ATcT inputs is not independent physical validation.

At exact 300-4000 K source knots, the replay decomposes JANAF minus NASA ln K
into balanced energy, molecular thermal and atomic thermal contributions.
Current fits are evaluated only at 1500-4000 K. The following dimensionless
factors are source diagnostics, not abundance responses:

.. list-table::
   :header-rows: 1

   * - T (K)
     - Current / NASA K
     - Energy-aligned JANAF / NASA K
     - ATcT energy-only / current K
   * - 1500
     - 1.141578
     - 1.036042
     - 0.946936
   * - 3000
     - 1.072891
     - 1.042492
     - 0.973107
   * - 4000
     - 1.079192
     - 1.043563
     - 0.979761

The last column means an additive D0 shift applied to the inherited current
thermal function; it is an algebraic diagnostic, not a saved candidate model.
The fit/source residual is retained separately, rather than interpreting it
as an atomic thermochemical measurement. Principal-isotope Q is compared with
NASA at a common translation mass, with the isotope translation effect also
reported separately. Neither comparison is a natural-isotope mixture model.

Importance, reproducibility and limits
--------------------------------------

The frozen 75-case, 537-species, ion-inclusive ``curated-cos-v1`` grid places
at most 5.9418% of C in CH3, at
4000 K, 100 bar and C/O = 1.5.
Its maximum mole fraction is 6.97962e-05. Archived target
potentials match the current literal fit. These endpoints establish investigation
priority; they are not a new substitution calculation, universal importance
bound, or validation of photochemical/cool atmospheres.

Replay offline with::

   python mceg/reviews/ch3_source_review.py
   pytest tests/model_evidence/carbon_radical_source_review_test.py

The packet is ``tests/data/validation/ch3_source_review/20261106``.
It contains source URLs/hashes, exact selected NASA records, JANAF and PF tables,
selected state fields where used, input facts, numerical output and dependency
receipt. PDFs are not committed. MCEG records source attribution, normalized
comparisons and the scoped next-step decision as supported; full physical
accuracy and complete historical source recovery remain unknown. The existing
source's literature remains an MCEG provenance matter. Recovering every original
Gurvich/JANAF antecedent or FastChem fit weight is not a gate for this evidence PR.
No runtime model, curated payload or equilibrium result is changed.

Primary evaluated sources include the `JANAF database
<https://janaf.nist.gov/>`_, `pinned official NASA CEA coefficients
<https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_
and `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_.
Individual access limits and the inspected modern-paper DOI are in ``sources.json``.
