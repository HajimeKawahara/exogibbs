OH continuous NASA/ATcT candidate (2026-11-05)
=============================================

This packet completes the continuous candidate proposed in the dated
``oh_source_review/20261105`` packet, preserving that packet and its receipt.
The primary candidate replaces the inherited JANAF/FastChem thermal
construction with NASA OH/H/O functions and a balanced ATcT 1.172 energy.
It does not change production curated data.

The reaction is H(g) + O(g) -> OH(g), standard pressure 1 bar, restricted
to 1500--4000 K. All temperatures use the same 1000--6000 K NASA segment.
Subtracting atomic coefficients from the molecular coefficients is exact;
there is no approximation fit. The ground-relative ATcT atomization energy
is 425604 J/mol. The NASA header value is 425785.9 J/mol, and the correction
in D0/R is -21.5857097163 K, accounting for NASA's native R=8.31451 and
modern SI R. NASA atomic fine-structure thermal populations are retained.
The source dimensionless functions stay unchanged; ground-relative
calorics are expressed with common modern SI R.

The MYTHOS thermal control retains those same atoms, energy and pressure.
Its archived 10606 states retain fine structure, parity and electronic
levels; their weights are divided only by nuclear spin two. Internal
states represent 16O1H at aggregate translation mass 17.00734 u. This
declared approximation does not supply isotope-specific reaction energies
or elemental isotope balance.
All states are summed directly, including supplied quasi-bound levels;
no additional cutoff, symmetry divisor, ZPE or continuum term is applied.

``candidate.json`` freezes the continuous model and conventions.
``validation.json`` records reconstruction, caloric and alternative-source
checks. The associated offline tests check scalar/JAX evaluation and
thermodynamic derivatives. Numerical agreement does not bound ATcT
covariance, continuum contributions or the full physical uncertainty.

Offline reconstruction and checks::

   PYTHONPATH=src python mceg/reviews/oh_candidate.py
   PYTHONPATH=src pytest tests/model_evidence/oh_candidate_test.py

Rebuild the candidate and its validation report from the frozen sources::

   PYTHONPATH=src python mceg/reviews/oh_candidate.py --build \
       --output tests/data/validation/oh_candidate/20261105/validation.json

The original NASA records, official source identifiers and compact state
array are pinned in the historical source packet. No new download or full
paper redistribution is needed. The companion
``oh_equilibrium_impact/20261105`` packet measures actual conserved finite
substitution; a formation-K ratio alone is not an abundance ratio.
