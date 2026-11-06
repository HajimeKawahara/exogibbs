Curated gas thermochemistry datasets
====================================

The current default is ``curated-gas-v2``: 27 reviewed recommendations update
28 species on ``curated-ch2-v1``, including coordinated cis/trans-HNO2 energies.
The existing thermal functions are retained except for evaluated 2025 O2
thermals. HNCO/NCO and NO2/N2O4 energies are coordinated, while NaCN, KCN and
C2N2 remain unchanged. It retains the preceding HCP, C2H2, CH2, TiO2, CN, PS,
COS, KH/HO2/OH, evaluated 2025 H2O, NIST 626 CO2, TiO and C2H updates. Earlier
JSON records and their named presets retain preceding models for explicit
historical reproduction. All datasets use 1500--4000 K
and a standard pressure of 1 bar.

Original curated-gas-v1
-----------------------

``gas_v1.json`` is the pinned numerical dataset for
``exogibbs.presets.curated_v1.chemsetup``. Its evaluation domain is 1500--4000 K
at a standard pressure of 1 bar. The native runtime needs neither MCEG nor
a FastChem installation. The packaged FastChem4 gas and Asplund 2021 files
remain pinned inputs for inherited species and reference abundances.

Source decisions
----------------

* 43 atomic ions use NASA CEA thermal functions with spectroscopic
  ionization/attachment energy updates. Four use adopted partition models.
  N- and Zn- are excluded because the adopted evaluation has no stable bound state.
* 35 major molecules retain the approved preferred, conditional or provisional
  choices. Twenty-three change numerical constants; twelve retain existing fits.
* 427 further species retain their FastChem4 constants and are explicitly marked
  ``inherited_unreviewed``. The 27 neutral atoms and electron define the zero
  formation-energy gauge. Including them gives 537 species, or 421 neutral
  species with the electron and all ions removed.

The ``decisions`` entries preserve the adoption studies' source-specific
review scopes and follow-up descriptions. The dataset-level
``temperature_range_K`` defines this runtime version's narrower domain.
Continuous representation and solver checks are recorded separately in
``tests/data/validation/curated_gas_validation`` in the source repository. The complete
mixture's physical accuracy is not established by these checks.

Continuous representations
--------------------------

NASA9 reaction coefficients combine neutral-atom/electron references before
shifting the enthalpy constant. Original NASA energy conventions use
R = 8.314510 J mol^-1 K^-1; modern energy thresholds use E/kB. This distinction
is inherited from the approved audits.

CH4, COS, TiO and VO use quintic Hermite representations of (G-H0)/R through
1500, 2000, 3000 and 4000 K. Values, first derivatives and second derivatives
are fixed by JANAF H, -S and -Cp/T, respectively. Intermediate JANAF rows are
withheld for verification. A single C2 function therefore defines H, S and Cp.

Five neutral/cation partition functions are represented as positive Boltzmann
sums fitted to Barklem & Collet Table 8 over 500--10000 K. This supports smooth
evaluation and nonnegative internal heat capacity; the runtime domain remains
1500--4000 K. Fitted excitation terms are numerical basis functions, not new
measured levels. The three anions use explicit evaluated bound-level sums.
Electron spin is two and heavy-particle translational mass ratios are one,
as in the approved reference calculation.

Attribution and reproduction
----------------------------

Full source URLs, original hashes, per-species choices and the numerical
input hashes are stored in the JSON. Principal sources are NASA CEA
(NASA/TP-2002-211556), NIST-JANAF, NIST ionization energies, evaluated electron
affinities, Barklem & Collet (2016, doi:10.1051/0004-6361/201526961), ATcT 1.172,
and the species-specific spectroscopy identified in the adoption records.
NASA license and notice files are packaged under ``data/nasa9/``; existing
FastChem attribution remains under ``data/FastChem4/`` and ``data/fastchem/``.

Rebuild offline from the repository root::

    python examples/comparisons/build_curated_preset.py --output /tmp/gas_v1.json

The builder uses SciPy only at development time. The runtime evaluates the
stored coefficients and partition sums with JAX. Changing numerical sources
requires an explicit dataset-version and pinned-hash review.

C2H correction
---------------

``c2h_v1.json`` pins the separate ``curated-c2h-v1`` dataset, selected by
``exogibbs.presets.curated_c2h.chemsetup()``. Only C2H1 changes: NASA CCH
thermal functions and the balanced ATcT 1.172 atomization energy replace the
inherited constants. All other species, both network options and the
1500--4000 K interval are preserved. ``gas_v1.json`` remains frozen.

Rebuild from inspected offline records with
``python mceg/reviews/build_c2h_correction.py --output /tmp/c2h_v1.json``.
The source choice is conditional; high-temperature thermal uncertainty and
whole-mixture physical accuracy remain unquantified. The 75 paired
ion-inclusive equilibrium conditions are archived under
``tests/data/validation/c2h_equilibrium_impact/20261102``; neutral-mode
substitution changes were not evaluated there. An additional 75 cold solves
per network verify the adopted default in
``tests/data/validation/curated_default_adoption/20261102``. The original
opt-in wording inside the frozen correction JSON records its creation policy;
runtime metadata records that adoption. See ``documents/presets/curated_c2h.rst``.

Versioned TiO adoption
----------------------

``tio_v1.json`` pins ``curated-tio-v1``, selected by
``exogibbs.presets.curated_tio.chemsetup()``. Only O1Ti1 changes relative to ``curated-c2h-v1``.
The production payload preserves the reviewed candidate's six-knot C2 Gibbs
representation without refitting, using current TOTO 48Ti16O internal levels,
NASA elemental molecular mass 63.8664 u and neutral-atom references, and
Jain D0 = 6.824 +/- 0.010 eV. It is an explicit representative-isotope
approximation, not an isotope-resolved network.

Modern state thermodynamics replace an identified obsolete JANAF singlet
assumption. The D0 shift from the preceding 6.869 +/- 0.061 eV is within the
old quoted uncertainty; adoption improves the measurement and conventions
without claiming that the old central value was statistically excluded.
High-state, isotope and complete physical uncertainty remain unquantified.

Rebuild offline with
``python mceg/reviews/build_tio_adoption.py --output /tmp/tio_v1.json``.
Source and candidate evidence remain under ``tests/data/validation/tio_*``;
the separate production execution is in
``tests/data/validation/curated_tio_adoption/20261103``. The production MCEG
contract is ``mceg/contracts/curated_tio.yaml``. See
``documents/presets/curated_tio.rst`` for the decision, numerical scope and
atmospheric limitations. Optional MCEG records are not runtime dependencies.

CO2 adoption
-------------

``co2_v1.json`` pins ``curated-co2-v1``, available explicitly through
``exogibbs.presets.curated_co2.chemsetup()`` and selected by the public default
at adoption. Only C1O2 changes relative to ``curated-tio-v1``. The exact
reviewed NIST 626 eleven-knot C2 Gibbs coefficients retain the ATcT 1.172
atomization energy, NASA atomic reference and molecular mass 44.0095 u.
Source internal c2, spin and ground-state conventions are explicit; no
additional ZPE, exchange-symmetry or isotope mixing correction is applied.

Rebuild with ``python mceg/reviews/build_co2_adoption.py --output /tmp/co2_v1.json``.
The two-network production packet contains 150 cold solves in
``tests/data/validation/curated_co2_adoption/20261104``. The scientific rationale,
finite response, high-temperature and isotope limits are in
``documents/presets/curated_co2.rst``; the production contract is
``mceg/contracts/curated_co2.yaml``. Physical accuracy remains unquantified.

H2O adoption
-------------

``h2o_v1.json`` pins the former ``curated-h2o-v1`` default, selected
explicitly through ``exogibbs.presets.curated_h2o.chemsetup()``. Only H2O1
changes relative to ``curated-co2-v1``. The exact reviewed evaluated 2025
degree-8 Cp/R polynomial and analytical H/S integrals retain the ATcT 1.172
total atomization energy, NASA atomic references and mass 18.01528 u.
H2(16O) internal thermodynamics represents element-aggregate water. Published
entropy already removes the proton nuclear-spin product; it is not divided
again. The source's numerical enthalpy values are J/mol despite kJ/mol labels.

Rebuild with ``python mceg/reviews/build_h2o_adoption.py --output /tmp/h2o_v1.json``.
The combined CO2/H2O production packet contains 150 cold solves, six tighter
independently assembled controls and two FastChem comparisons in
``tests/data/validation/curated_h2o_adoption/20261104``. Earlier source and
candidate evidence retains its original scope. The CO2 adoption's default
alias, production script and test are preserved under this packet's
``source/`` for historical replay, without rewriting its original receipt.
See ``documents/presets/curated_h2o.rst`` and
``mceg/contracts/curated_h2o.yaml`` for the scientific choice, joint response,
scattering-assignment and isotope limitations. Complete physical accuracy
remains unquantified.

OH adoption
------------

``oh_v1.json`` pins the former ``curated-oh-v1`` default, also selected
explicitly through ``exogibbs.presets.curated_oh.chemsetup()``. Only H1O1
changes relative to ``curated-h2o-v1``. The exact reviewed NASA9 OH-H-O
reaction receives balanced ATcT 1.172 D0=425.604 kJ/mol once, replacing
the inherited JANAF electronic/caloric defect. No new fit or additional
spin, symmetry, ZPE or atomic correction is applied. MYTHOS remains a
thermal sensitivity control, with less than 0.098 percent difference in K.

Rebuild with ``python mceg/reviews/build_oh_adoption.py --output /tmp/oh_v1.json``.
The production packet contains 150 new cold solves, six tighter candidate
controls and two FastChem comparisons in
``tests/data/validation/curated_oh_adoption/20261105``. Earlier default-dependent
sources are preserved in its ``source/`` directory and historical receipts
are unchanged. See ``documents/presets/curated_oh.rst`` and
``mceg/contracts/curated_oh.yaml`` for the scientific rationale, finite response,
isotope/continuum and energy-covariance limitations. Full physical accuracy
remains unknown.


KH, HO2 and COS adoption sequence
---------------------------------

``kh_v1.json`` preserves the inherited KH thermals and adopts the reviewed
spectroscopic D0 on ``curated-oh-v1``. ``ho2_v1.json`` adds exact NASA thermal
functions and balanced ATcT 1.172 energy for HO2 on that KH base. The versioned
``cos_v1.json`` then increases COS D0 by 5612 J/mol using the same-version
ATcT 1.228 atoms and molecule, preserving the inherited thermal functions.
Each layer changes only its named molecule and pins its immediate base.

The former default remains ``curated_cos.chemsetup``; ``curated_kh`` and
``curated_ho2`` also remain explicit historical versions. Each production packet
under ``tests/data/validation/curated_<molecule>_adoption/20261105`` records
150 fresh cold solves, six tighter candidate controls and two FastChem
comparisons. Earlier OH-only candidates are not relabeled as cumulative
composition results. See the corresponding preset guides and production
MCEG contracts for source choices, scope and physical uncertainty limits.


CN, PS and TiO2 adoption sequence
----------------------------------

``cn_v1.json`` applies the reviewed ATcT 1.228 difference of -4801 J/mol to
CN D0 on ``curated-cos-v1``, retaining the literal thermal fit and its residual.
``ps_v1.json`` adds the resolved POPS molecular thermal difference on that CN
base, retaining the Lodders energy and source-fit residual. Its degree-18
Chebyshev log-partition representation needs no runtime state file.
``tio2_v1.json`` then adds measured rigid-rotor/harmonic-oscillator thermals
on the PS base, retaining JANAF energy and the inherited fit residual.
The H298-matched and excited-state variants remain sensitivity controls.

The public default is ``curated_tio2.chemsetup``; ``curated_cn`` and
``curated_ps`` preserve the intermediate versions. Each layer changes only
its target molecule and pins its immediate base. Separate 20261106 production
packets under ``tests/data/validation/curated_<molecule>_adoption`` record
fresh solves on both networks and controls on each actual cumulative base.
Earlier source/candidate receipts retain their original models and hashes.
See ``documents/mceg/curated_cn_adoption.rst``,
``documents/mceg/ps_curated_adoption.rst`` and
``documents/mceg/curated_tio2_adoption.rst`` with their linked MCEG contracts
for numerical checks, source choices and physical limits.
