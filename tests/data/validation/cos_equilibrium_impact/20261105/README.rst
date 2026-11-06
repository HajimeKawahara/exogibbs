COS finite substitution, 2026-11-05
==================================

The actual current ``curated-oh-v1`` provider was checked against the archived
OH-adoption baseline for every species, element, matrix entry, potential and
C/O-adjusted elemental inventory before reuse. The 150 candidate endpoints
are fresh cold solves, 75 each for the 537-species ion-inclusive and
421-species neutral networks. Only the COS potential changes.

The grid spans 1500/2000/2500/3000/4000 K, 1e-6/1e-4/0.01/1/100 bar and
C/O=0.5888436553555908/1/1.5. Primary tolerance is 5e-14; selected cold
controls and five-point finite paths use 2e-14. The NPZ files preserve both
endpoints, tighter controls and paths, without abundance floors. The JSON
files preserve all convergence diagnostics, source/output hashes, numerical
versions and four independent FastChem comparisons on the ion-inclusive
network. Neutral FastChem comparisons were not run.

The production grid is hot and hydrogen-rich. Trace COS in this grid does
not imply unimportance in cool or oxidizing planetary atmospheres. The
study excludes condensation, photochemistry, transport and opacity.
