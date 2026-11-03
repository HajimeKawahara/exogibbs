SiO energy and thermal-source investigation
===========================================

This 2026-11-03 review follows #308 and #309. **Retain the current NASA
thermal functions plus ATcT 1.172 balanced atomization energy.** Modern
SiOUVenIR level sums closely reproduce the inherited molecular thermal
functions over the sampled 1500-4000 K domain. No newly demonstrated error
requires replacing the curated coefficients. This is a scoped retention
decision, not a claim that every temperature or atmospheric abundance is
physically validated.

Why this molecule matters
-------------------------

Small source differences do not make SiO a low-priority molecule.
The inspected author manuscript of `Ito et al. (2015)
<https://arxiv.org/abs/1501.05393>`_, DOI
`10.1088/0004-637X/801/2/144 <https://doi.org/10.1088/0004-637X/801/2/144>`_,
models mineral atmospheres above molten rocky surfaces. SiO is an important
gas constituent, UV absorber and infrared emitter in those calculations.
Their gas/melt boundary uses MELTS, while the gas calculation uses NASA
CEA. That shared thermochemical lineage is not independent validation of
our NASA thermal functions. Opacity and predicted emission features motivate
priority; they do not measure an equilibrium constant.

The inherited choice and energy zero
-------------------------------------

The reaction is Si(g) + O(g) -> SiO(g), at standard pressure 1 bar, for
neutral X1Sigma+ silicon monoxide, not solid SiO or an ion. The existing
major-molecule adoption already separates energy from thermal shape. For
ATcT 1.172, the inspected `versioned index
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
gives Hf(0) = 450.34, 246.844 and -98.37 kJ/mol for Si, O and SiO:

.. math::

   D_0 = H_f(\mathrm{Si},0)+H_f(\mathrm{O},0)-H_f(\mathrm{SiO},0)
       = 795.554\ \mathrm{kJ\,mol^{-1}}.

The `Si <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=54>`_
and `O <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=21>`_
provenance pages were inspected. Si's displayed 20 contributors cover 85.6%
of its provenance and include SiO atomization calculations from Feller (2008)
and Karton (2011). Feller's bibliography resolves to
`10.1063/1.3008061 <https://doi.org/10.1063/1.3008061>`_; its original numerical
tables were not inspected. The SiO-specific provenance page was inaccessible;
the index value is independently present in the original frozen adoption
input. These access limits are recorded in MCEG, not disguised as a full
audit. Dataset DOI: `10.17038/CSE/2440254 <https://doi.org/10.17038/CSE/2440254>`_.
No reaction uncertainty is formed by treating correlated species errors as
independent. Later ATcT versions are not mixed into this decision.

D0 starts at the molecular rovibrational ground level and includes the
zero-point correction relative to an electronic well depth De. Atomic
ground levels are Si 3P0 and O 3P2. The inspected NIST
`Si <https://physics.nist.gov/PhysRefData/Handbook/Tables/silicontable5_a.htm>`_
and `O <https://physics.nist.gov/PhysRefData/Handbook/Tables/oxygentable5.htm>`_
tables show their respective fine-structure levels at (0, 77.112, 223.157)
and (0, 158.265, 226.977) cm^-1. Thermal populations use 2J+1 weights, not
a temperature-independent factor of nine. The current NASA atomic thermal
functions remain in the balanced reaction; an atomic term average is not
silently substituted for the ground-state zero.

The inspected `NASA coefficient report
<https://ntrs.nasa.gov/citations/20020085330>`_ cites Gurvich (1991), volume 2
part 1 page 247 / part 2 page 227 for SiO. Headers and coefficients match
the frozen adoption inputs. The original report starts at 200 K; the
existing Si/SiO review fixture conservatively starts at 300 K. Neither
extends the curated runtime domain below 1500 K. Gurvich's original pages
were not obtained. The official `JANAF SiO table
<https://janaf.nist.gov/tables/O-012.html>`_ and printed page 1728 are a
September 1967 evaluation, despite the 1998 compilation date. Its printed
formation enthalpy is -100.42 +/- 8.4 kJ/mol. Original thermochemical-cycle
measurements were not independently reaudited.

For each legacy source, H(0) is obtained from Hf(298.15) minus the thermal
enthalpy increment, then its own Si/O atoms are balanced. Native D0 is
792.456865 kJ/mol for NASA and 794.029 kJ/mol for JANAF. A molecular-only
enthalpy difference would be the wrong correction. NASA's native
R=8.314510 convention is preserved separately from modern R. The adopted
minus native NASA D0/k is 373.042867 K, so the curated/NASA K ratio is
exactly exp(373.042867/T), with unchanged NASA thermal shape.

.. list-table:: Formation constant and molecular partition comparisons
   :header-rows: 1

   * - T, K
     - Curated / raw NASA
     - Raw JANAF / NASA
     - JANAF / NASA, same D0 and atomic functions
     - SiOUVenIR / NASA internal Q
   * - 1500
     - 1.282351
     - 1.131828
     - 0.997139
     - 0.998247
   * - 2000
     - 1.205050
     - 1.096206
     - 0.996739
     - 0.998154
   * - 3000
     - 1.132409
     - 1.061081
     - 0.995802
     - 0.998018
   * - 4000
     - 1.097748
     - 1.043144
     - 0.994527
     - 0.997971

The aligned JANAF column is diagnostic, not an adopted replacement. Its
remaining molecular thermal difference is 0.29-0.55% at these four nodes.
Agreement with either historical source alone is not the selection rule.

Modern thermal comparison and its limits
-----------------------------------------

The full author manuscript of `Yurchenko et al. (2022), SiOUVenIR
<https://arxiv.org/html/2111.04859v1>`_, DOI
`10.1093/mnras/stab3267 <https://doi.org/10.1093/mnras/stab3267>`_, and the
official `28Si16O dataset
<https://www.exomol.com/data/molecules/SiO/28Si-16O/SiOUVenIR/>`_ were inspected.
Its nine electronic states extend an infrared line list into the UV. The
174250 states, definition, partition and heat-capacity files were read;
the large transition file is unnecessary for this calculation. The paper
fixes De=8.26 eV from Huber and Herzberg (1979), rather than providing a
new atomization threshold. This old electronic-well parameter must not
replace the adopted D0. The stated 10000 K line-list range does not certify
our thermochemistry or extend its 4000 K upper bound.

The state sum uses Q=sum[g*exp(-c2*E/T)], with the tabulated X, v=0, J=0
level at zero. Every tabulated degeneracy equals 2J+1: the nuclear-spin
factor for 28Si16O is one, and the heteronuclear symmetry number is one.
No second ZPE or electronic multiplicity factor is added. Full-state sums
agree with the separately published partition values within 8e-7 relative
at six 500-4000 K nodes. Heat capacity from the energy variance,
Cp/R=5/2+Var(E/k)/T^2, agrees with the separate published Cp values within
0.0025 J/mol/K. These are numerical checks on the supplied level model.

NASA internal Q is recovered by removing its ideal-gas translational
factor from (G-H0)/(RT), using its natural-mass convention 44.0849 u.
ExoMol supplies the internal Q of 28Si16O (mass 43.97184115 u). This removes
the translational normalization difference; it does **not** convert the
internal rovibrational isotope dependence into an exact natural-isotope
mixture. The table's residual 0.18-0.20% therefore is not an uncertainty
estimate. NASA and level-sum Cp differ by at most 0.0632 J/mol/K at the four
production-domain nodes. Their close thermal agreement supports retaining
the current model, while shared older spectroscopic inputs and the fixed
dissociation asymptote limit independence. A complete hot-state or energy
uncertainty audit is not claimed.

Application footprint and MCEG
-------------------------------

The existing 75-case, 537-species ion-inclusive ``curated-c2h-v1`` archive
contains a maximum SiO silicon-budget fraction of **99.9923328%**, at
2000 K, 1e-6 bar and reference C/O=0.5888436553555908; its mole fraction is
3.56587e-5. All archived SiO formation potentials match the actual current
hybrid, and silicon conservation is checked across every case. This is a
hydrogen-rich gas grid spanning 1500-4000 K, 1e-6-100 bar and three C/O
values. It is neither a lava-atmosphere calculation nor a finite source
substitution. No conclusion about magma composition, condensation, rainout
or observed spectra follows from this footprint.

``mceg/contracts/sio_source_review.yaml`` separates supported source
construction, normalized thermal comparison and numerical replay from
unknown complete physical accuracy. Follow-ups are proportionate:

* Complete the same-version SiO provenance/covariance trail when accessible;
  inspect the relevant original energy or Gurvich pages if their definitions
  could change the chosen balanced anchor. Obtaining every original paper
  is not a condition for merging this source review.
* For a specific lava or irradiated-atmosphere application, define the
  composition, melt/condensate boundary and thermal domain. Assess a named
  alternative with isotope conventions, caloric derivatives and conserved
  finite substitutions only when a material improvement is proposed. The
  present close thermal comparison does not require a production replacement.

The compact packet is ``tests/data/validation/sio_source_review/20261103``.
Full research papers and the full level file are not redistributed; their
hashes, selected data and state-sum regeneration method are recorded.
Reproduction is offline::

   PYTHONPATH=src python mceg/reviews/sio_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/sio_source_review_test.py

No production coefficients, preset identifier, runtime bounds or archived
equilibrium endpoints change.
