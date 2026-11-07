CO source review, 2026-11-03
============================

states.npz retains all 11264 4thplus and 6383 Li2015 states using the columns
listed in state_inventory.json. No level, spin weight or energy-zero correction
is inserted during extraction. Published PF rows retain their actual printed
temperatures; 4thplus does not follow its metadata's claimed integer grid.

sources.json records original download hashes and access/inspection limits.
Metadata copies normalize CRLF to LF; original download and normalized archive
hashes are recorded separately. Only small metadata and derived state arrays are committed;
the original states/PF downloads and paper remain outside the repository.
The extraction command is documented in documents/mceg/co_source_review.rst.

comparison.json is regenerated offline from these data and the pinned existing
NASA/JANAF/ATcT reference. The current footprint uses curated-tio-v1's 75 saved
ion-inclusive endpoints. No production update or finite substitution is claimed.

Source agreement, caloric consistency and atmospheric physical accuracy are
separate claims. Natural-mass/representative-isotope and isotope-mass proxies
are both identified. The retained source choice is scoped to sampled 1500-4000 K.
