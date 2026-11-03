Curated gas thermochemistry datasets
====================================

The current default is ``curated-tio-v1``, with modern TOTO/Jain TiO on top of
the C2H-corrected model. ``gas_v1.json`` and ``c2h_v1.json`` retain the preceding
models for explicit historical reproduction. All datasets use 1500--4000 K
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

Default TiO adoption
---------------------

``tio_v1.json`` pins ``curated-tio-v1``, selected by
``exogibbs.presets.curated.chemsetup()`` and explicitly by
``curated_tio.chemsetup()``. Only O1Ti1 changes relative to ``curated-c2h-v1``.
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
