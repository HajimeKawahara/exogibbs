Curated gas thermochemistry, version 1
====================================

``gas_v1.json`` is the pinned numerical dataset for
``exogibbs.presets.curated.chemsetup``. Its evaluation domain is 1500--4000 K
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
