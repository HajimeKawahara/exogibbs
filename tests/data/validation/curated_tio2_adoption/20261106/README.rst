TiO2 production adoption on the cumulative CN/PS base
=======================================================

150 fresh float64 production endpoints: 75 ion-inclusive (537 species) and 75
neutral (421 species), plus six tighter independently assembled candidate
controls and two standalone FastChem comparisons. All cases converged.

The immediate predecessor is the separately verified curated-ps-v1 production
archive. Every species, element, formula-matrix entry, elemental budget and
baseline potential is checked before reusing that archive. Only TiO2 changes.
Dense 2501-temperature comparison verifies the exact reviewed RRHO correction.

The summary distinguishes incremental TiO2 effects, cumulative changes from
curated-cos-v1, and CN/PS background effects relative to the earlier TiO2-only
ion-inclusive candidate. That earlier candidate has no neutral archive.
Complete physical accuracy, condensation and planetary validation are not
established by numerical solver agreement.

Reproduce with mceg/reviews/curated_tio2_adoption.py into a new results folder;
see documents/mceg/curated_tio2_adoption.rst for source conventions and commands.
The receipt records the actual completed execution inputs and outputs.
