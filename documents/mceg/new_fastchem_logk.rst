Preliminary FastChem data: abundance evidence and retrieval follow-up
=====================================================================

This record accompanies `ExoGibbs issue #297
<https://github.com/HajimeKawahara/exogibbs/issues/297>`_. It compares Daniel's
preliminary 2026-10-23 tables with ``curated-gas-v1`` and the original FastChem 4
table. The :download:`v0.2 contract <../../mceg/contracts/new_fastchem_logk.yaml>`
separates numerical evidence, source choices and the untested downstream
metallicity-retrieval hypothesis. MCEG is an optional documentation tool.

Recorded judgments
------------------

.. list-table:: Claims and their scoped assessments
   :header-rows: 1
   :widths: 28 15 57

   * - Claim
     - Assessment
     - Evidence and scope
   * - Atomic-ion update scope
     - supported
     - The standard, extended and Hoeijmakers files change 47, 79 and 228
       atomic-ion records. Catalogs, stoichiometry and molecular fits agree
       with the respective original tables.
   * - Column numerical closure
     - supported
     - All 966 layer/model cases pass element, charge and mass-action gates.
       Every curve uses ExoGibbs; this is a data comparison, not independent
       FastChem solver validation.
   * - Molecular mass-action attribution
     - supported
     - Five recorded samples separate equilibrium-constant changes from the
       coupled constituent-atom response.
   * - Effect on retrieved metallicity
     - unknown
     - No spectrum or retrieval was calculated. The execution record is
       ``not_run`` and the observing configuration remains unspecified.
   * - Whole-mixture physical accuracy
     - unknown
     - Source agreement and numerical closure supply no independent
       whole-mixture observation or experimental validation.

The two 161-layer columns use the same 537 species and normalized Asplund 2021
budget, with pressure from 1e-6 to 100 bar. One is isothermal at 4000 K; the other
uses the prescribed 2500--4000 K inversion from PR #284. This does not establish
coverage of every T/P combination or another metallicity. The subject manifest
pins selected code, coefficients and archived arrays; its digest identifies the
reviewed snapshot without requiring the contract to contain its own future Git
commit. Earlier source-study contracts retain their historical scopes.

What explains the molecular differences?
----------------------------------------

For a neutral molecule at fixed temperature and pressure, mass action gives

.. math::

   \frac{x_{M,\mathrm{new}}}{x_{M,\mathrm{curated}}}
   = \frac{K_{M,\mathrm{new}}}{K_{M,\mathrm{curated}}}
     \prod_a \left(\frac{x_{a,\mathrm{new}}}
                         {x_{a,\mathrm{curated}}}\right)^{\nu_a}.

At 2500 K and 0.01 bar, the HCN ratio is
``0.75300568 * 0.98874048 = 0.74452720``. Its adopted atomization energy exceeds
the JANAF value by 0.06019568 eV. That energy difference alone predicts a
formation-constant ratio of 0.75622469; the remaining thermal/fit factor is
0.99574331. HCN's difference is therefore mostly attributable to its molecular
energy choice at this sample.

For SiS at 3500 K and 100 bar, the formation constants are identical, while the
abundance ratio is 1.10964313. The difference comes from the available S and Si
atoms. This atomic-product factor includes the response of the whole network;
it does not isolate an intervention on atomic-ion coefficients.

The :download:`attribution report
<../../results/mceg/new_fastchem_logk_20261024/attribution.json>` records all five
samples, absolute mole fractions, atom factors, residuals and selected-file
hashes. At the HCN inverted-column extremum the mole fractions are about 1e-10.
A 25.5 percent difference there is not a measured metallicity error, an opacity
change or a one-sigma uncertainty estimate.

Public-source inspection
------------------------

The :download:`2026-10-24 literature review
<../../mceg/reviews/new_fastchem_logk_public_sources_20261024.json>` records six
DOIs, public copies, inspected locations and access limits. No additional
user-supplied paper is required for this reading task.

* `Al-Refaie et al. (2022) <https://doi.org/10.3847/1538-4357/ac6dcd>`_ find
  consistent posteriors for matched assumptions in their 1500 K experiment.
  Their large mismatch biases cannot be assigned to our hotter table comparison.
* `ATcT 1.172 <https://doi.org/10.17038/CSE/2440254>`_ and
  `JANAF <https://doi.org/10.18434/T42S31>`_ confirm the selected HCN energy
  entries. The review reconciles the rounded JANAF routes with the archived
  atomization-energy calculation.
* `Ruscic (2014) <https://doi.org/10.1002/qua.24605>`_ explains uncertainty
  conventions and correlation. `Harding et al. (2008)
  <https://doi.org/10.1063/1.2835612>`_ is an input to the ATcT network, so its
  agreement is not independent validation of the adopted value.
* `Andersen et al. (1999) <https://doi.org/10.1063/1.556047>`_ distinguishes
  ground, excited and metastable anion states. The preliminary fits were not
  reconstructed from this reference.

The contract retains its earlier abstract-only evidence and assessments, then
adds inspected evidence and two ``*_public_sources`` assessments. Both retrieved
metallicity and whole-mixture accuracy remain ``unknown``. The completed
``inspect_public_references`` task records reading, not physical validation.
The earlier numerical validation receipt remains historical; checks for this
addition are under ``results/mceg/new_fastchem_logk_public_sources_20261024/``.

Deferred retrieval proposal
---------------------------

Spectral calculations and metallicity retrieval are outside the current task
at the user's request. The proposal below is retained for future use; executing
it is not required to complete this record.

The authored ``controlled_cross_retrieval`` follow-up requires a declared T/P
structure, composition, wavelength range, resolution, line opacities, clouds,
geometry, noise covariance, random seeds, priors and sampler checks. It calls
for both directions of table substitution and same-table injection/recovery
controls. A quantitative claim and reporting threshold must be defined before
assessing the result.

Completion requires spectra and posterior samples, convergence checks,
``Delta[M/H]`` in dex, C/O and temperature shifts, and their size relative to
posterior widths. A new context-bound assessment then records the measured
sensitivity. Synthetic recovery alone does not establish real-atmosphere
accuracy. No placeholder metallicity offset is assigned in the present record.

Offline reproduction
--------------------

The inputs, figures and primitive arrays are under ``results/new_fastchem_logk``.
The :download:`column receipt <../../results/new_fastchem_logk/receipt.json>`
records conditions, all six closure audits and input/output hashes.

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python examples/comparisons/compare_new_fastchem_logk.py \
     --output-dir /tmp/new-logk-columns

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python mceg/reviews/new_fastchem_logk_20261024.py --output /tmp/attribution.json

   PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
     tests/unittests/examples/new_fastchem_logk_test.py \
     tests/unittests/mceg_new_fastchem_logk_test.py

The attribution replay reads the checked-in column arrays. It creates new
evidence for the five decompositions, not a new atmosphere or retrieval solve.
With MCEG 0.2.0rc2 or later available separately:

.. code-block:: bash

   mceg check mceg/contracts/new_fastchem_logk.yaml \
     --assessment molecular_attribution_review --root .
   mceg explain mceg/contracts/new_fastchem_logk.yaml \
     --assessment retrieval_metallicity_effect_public_sources --lang ja
   mceg check mceg/contracts/new_fastchem_logk.yaml \
     --assessment whole_mixture_accuracy_public_sources --root .

Unknown support in the latter commands is intentional. Completing a follow-up
or passing structural checks does not automatically promote a scientific claim.
