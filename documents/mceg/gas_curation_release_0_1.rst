Gas curation v0.1
=================

**Gas curation v0.1**, canonically **0.1.0**, identifies the scientific
curation snapshot at commit
``f580e66157edbe6f604e2748d8e1a25d46d08b86``. It records the existing
``curated-gas-v2`` model and its source decisions, validation and remaining
limitations. Naming this release does not change any thermodynamic function,
coefficient, species network or applicability domain.

The public ``exogibbs.presets.curated`` factory continues to select
``curated_v2``. Its dataset remains ``curated-gas-v2`` and its catalog model
identifier remains ``exogibbs.curated_gas.v2``. The ion-inclusive and neutral
networks contain 537 and 421 species, respectively, on **1500--4000 K** at a
**1 bar** thermochemical standard pressure. This standard state is not a
claim that physical accuracy has been established at every gas pressure.

Release identity and records
----------------------------

The :download:`release manifest <../../mceg/releases/gas_curation/0.1.0.json>`
identifies the frozen scientific inputs and their hashes. The
:download:`aggregate MCEG contract <../../mceg/contracts/gas_curation_release_0_1.yaml>`
connects that manifest to the existing scoped evidence. It supplements the
gas-v2 catalog record without changing the model identifier or adding a
runtime API. Ordinary calculations still require no MCEG installation or
network access.

The gas-curation version is separate from the ExoGibbs software version,
the thermochemical dataset identifier and the catalog's record date. No Git
tag is created for this milestone, so it does not affect the software version
derived by setuptools-scm. To reproduce the model, select
``curated_v2.chemsetup`` and retain the manifest, source commit and applicable
execution receipt. A release label alone does not validate a modified callable
or an unrecorded calculation.

Scientific scope
----------------

The release freezes **108 molecule-review entries**: the original 93 NASA
comparison rows, 14 NASA identity/coverage exclusions and the separate C3H
investigation. Their source decisions and original identity qualifications
remain explicit. This review inventory is distinct from the 537-species
ion-inclusive and 421-species neutral equilibrium networks; it does not imply
that every network species has received an independent review.

The release includes the cumulative source choices described in
:doc:`../presets/curated` and :doc:`curated_gas_v2_adoption`. Gas-v2 combines
27 reviewed selections affecting 28 species, including the trans-HNO2
companion. It retains inherited thermal functions except for the scoped O2
thermal update; HNCO/NCO and NO2/N2O4 contributions are counted once. The
N2O5 and P4S3 energy corrections explicitly use 298.15 K anchors. These choices
retain their original state, isotope, atomic-reference and fitting limits.

The :doc:`VO2 <vo2_source_review>` and :doc:`NH3 <nh3_source_review>` records
are source investigations with open physical-model follow-ups. Their inclusion
does not adopt a new VO2 or NH3 thermal model. Historical sensitivity results
retain their named original datasets and must not be relabeled as gas-v2
abundance calculations.

The :doc:`official FastChem4 comparison <curated_fastchem4_comparison>` records
108 molecules at five temperatures, including the updated HCN entry in the
official v4.0.4 full gas table. Those 540 evaluations compare dimensionless
formation constants, not equilibrium abundances. The original NASA matching
exclusions remain explicit. The gas-v2 production calculations, earlier
candidate substitutions and constant comparisons remain distinct evidence.

Supported numerical checks and bounded source choices do not establish a
complete physical uncertainty distribution for the mixture. State completeness,
source ancestry, provisional choices and molecule-specific follow-ups remain
visible in MCEG. Condensation, rainout, nonideal gas effects and external
ionization are outside the gas-only model's validation scope. A green
comparison grade is not a source-adoption criterion.

Updating the gas-curation version
---------------------------------

* **0.1.1:** a record-only correction or clarification that changes neither
  numerical thermochemistry, source-adoption decisions nor the declared domain.
  State the amendment and its scope in a new release record.
* **0.2.0:** a change to numerical thermochemistry, an adopted source/model
  selection or the applicability domain. Document the scientific reason and
  appropriate validation for that change.

Previously published release records remain immutable. Create a new manifest
and retain the earlier record and referenced evidence; do not silently update
the hashes of an old execution or reassign it to a newer model. These rules
govern scientific curation records independently of software release numbering.
