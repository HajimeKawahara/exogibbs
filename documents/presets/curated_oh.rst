Curated gas with NASA/ATcT OH thermochemistry
=============================================

``exogibbs.presets.curated.chemsetup()`` now selects **curated-oh-v1**,
also available as ``curated_oh.chemsetup``. It adopts the reviewed NASA
OH thermal functions with balanced ATcT 1.172 atomization energy on
**curated-h2o-v1**. Only the OH formation function changes. The adopted
H2O, CO2, TiO and C2H functions, all other species, elemental budgets,
atomic references and charge treatment are retained.

The domain remains **1500--4000 K**, with standard pressure **1 bar**.
The two supported networks retain 537 species with ions and 421 without
ions. Scalar and array JAX evaluation, JIT and temperature derivatives
are supported; invalid temperatures return NaN. Runtime requires only
packaged data. ``curated_h2o.chemsetup`` and the earlier versioned
factories retain their original datasets and evidence.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   from exogibbs.presets import curated, curated_h2o

   setup = curated.chemsetup()
   before_oh = curated_h2o.chemsetup()
   neutral = curated.chemsetup(ions=False)

Why this source is adopted
---------------------------

The inspected IUPAC evaluation,
`DOI 10.1063/1.1724828 <https://doi.org/10.1063/1.1724828>`_, identifies
incorrect ground-state electronic degeneracy and caloric treatment in
the inherited JANAF OH evaluation, including its 1998 edition. The
:doc:`source review <../mceg/oh_source_review>` separates that thermal
problem from the formation-energy choice. A known source defect is a
scientific reason to replace the old model even when its equilibrium
constant discrepancy is smaller than those of other surveyed molecules.

The :doc:`candidate study <../mceg/oh_candidate>` checks an exact NASA
CEA/Gurvich thermal reaction with ATcT 1.172 energy against independent
NASA expressions, caloric derivatives and the modern MYTHOS state sum
(`DOI 10.1093/mnras/stae2803 <https://doi.org/10.1093/mnras/stae2803>`_).
MYTHOS and NASA, with identical energy and atomic references, differ by
less than **0.098% in K** across the bounded domain. That is much smaller
than the **5.45--12.91%** NASA/ATcT-to-inherited change. This supports
using the exact NASA reaction as a small production change. MYTHOS
remains a thermal sensitivity control; source agreement is not an
independent experimental error bar or a guarantee of completeness.

The energy is balanced within one ATcT version:

.. math::

   D_0 = \Delta_f H_0({\rm H}) + \Delta_f H_0({\rm O})
       - \Delta_f H_0({\rm OH}) = 425.604\ {\rm kJ\,mol^{-1}}.

The NASA reaction carries 425.7859 kJ/mol. NASA's native gas constant
8.314510 J/mol/K must be used when removing its original energy;
ATcT uses the modern SI value. Thus the shift in D0/R is
**-21.585709716324345 K**, not simply the energy difference divided by
the modern gas constant. This shift enters the reaction energy once.

The exact adopted representation
--------------------------------

The packaged payload copies the nine reviewed coefficients for
OH-H-O on NASA's 1000--6000 K segment, restricted to 1500--4000 K.
Only the energy constant is shifted; no fit or interpolation is added.
NASA's dimensionless thermal convention and atomic functions are
preserved, with caloric quantities expressed using modern SI R.
There is no added spin factor, symmetry factor, ZPE or atomic correction.

Shared source ancestry, the element-aggregate isotope approximation,
high-state/continuum completeness and ATcT covariance remain limits.
The MYTHOS comparison removes nuclear spin two once and retains parity
and electronic/fine-structure weights. Those conventions are not a
complete isotope or continuum calculation. Full physical atmospheric
accuracy remains unknown; this does not prevent the justified bounded
source update.

Production verification and response
------------------------------------

The actual public factory passed **150 new cold solves**, 75 per
network: five temperatures (1500, 2000, 2500, 3000, 4000 K), five
pressures (1e-6, 1e-4, 0.01, 1, 100 bar) and three C/O ratios
(0.5888436553555908, 1, 1.5). Production thermodynamics agrees with the
independently evaluated frozen candidate on 2501 temperatures to
**1.07e-14 in G/(RT)**. Every non-OH potential is unchanged.

The full production grids reproduce the archived candidate responses
within 1.16e-11 in OH ln(x) and 7.67e-15 in any absolute mole fraction.
Six tighter candidate controls at tolerance 2e-14, compared with
production tolerance 5e-14, agree within 2.85e-13 in OH ln(x) and
3.40e-14 in any absolute mole fraction. The largest all-species log
response is 1.47e-5 for trace species; this is recorded separately
from target and absolute abundance errors.

H/C/O elemental relative residuals are below 8.82e-13, relative charge
residual below 4.96e-14 and formation-equation residual below 1.85e-12.
The all-element maximum is 1.47e-6, dominated by the trace-element
numerical floor. Two fresh FastChem comparisons agree within
**1.19e-8 dex**. Numerical closure and solver agreement establish
implementation consistency, not full physical accuracy.

.. list-table:: OH response on the retained composition grid
   :header-rows: 1

   * - Quantity
     - Result
   * - OH mole-fraction ratio, adopted / previous
     - 1.03881--1.12914 across both networks
   * - OH at 4000 K, 1 bar, reference C/O, with ions
     - 6.0911629e-5 -> 6.3275666e-5
   * - OH share of oxygen at that condition
     - 11.2391% -> 11.6753%

These finite responses include re-equilibration of the entire mixture.
They are source sensitivities, not atmospheric error bars. The preceding
candidate packet also retains path, tighter and MYTHOS controls.

Evidence and reproduction
--------------------------

``src/exogibbs/data/curated/oh_v1.json`` pins the exact candidate and
H2O base. ``mceg/reviews/build_oh_adoption.py`` reproduces that payload
offline and refuses changed inputs. The production execution is archived
in ``tests/data/validation/curated_oh_adoption/20261105`` with both
networks, diagnostics, summary and an actual input/output hash receipt.
The ``source`` directory preserves prior default-dependent bytes;
older execution receipts and numerical outputs are not rewritten.

The earlier H2O runner now explicitly selects ``curated_h2o``. Its
historical contract and the OH candidate contract resolve original
files to their byte-identical snapshots. They are not rebound to the
new default. Scoped source selection and production adoption are
supported in ``mceg/contracts/curated_oh.yaml`` and the packaged catalog;
the physical-accuracy assessment remains unknown.

Offline verification::

   PYTHONPATH=src pytest tests/unittests/presets/curated_oh_test.py
   PYTHONPATH=src pytest tests/model_evidence/curated_oh_adoption_test.py

Optional production replay into a new output directory::

   JAX_PLATFORMS=cpu PYTHONPATH=src \
     python mceg/reviews/curated_oh_adoption.py \
     --output-dir results/curated-oh-adoption-replay \
     --fastchem-executable /path/to/fastchem
