Inspect a model's known uncertainties
=====================================

ExoGibbs calculations and provenance inspection work without MCEG installed.
The package includes a small, dated catalog of scientific records. Install
MCEG only when you want its detailed explanation of those records.

Start from your setup
---------------------

.. code-block:: python

   from exogibbs.presets import curated
   from exogibbs.provenance import describe

   setup = curated.chemsetup()
   info = describe(setup)
   print(info["notice"])
   print(info["record_version"])
   print(info["explain_command"])

The returned dictionary can be saved with ``json.dump``. It describes declared
setup metadata, not a saved execution or its temperature, pressure and elemental
composition. It does not verify a modified callable that retains the old
metadata. A different dataset hash or network choice gives
``no_matching_record``; that means no matching review, not physical reliability.

The first catalog binds the ion-inclusive curated dataset to its existing
model record and HCN follow-up. It lists the species with explicit source
decisions; it is not a complete uncertainty inventory. The neutral network
does not inherit the 537-species HCN comparison.

Inspect HCN with optional MCEG
------------------------------

With an MCEG version providing installed-package catalogs, run:

.. code-block:: console

   mceg explain --package exogibbs --model exogibbs.curated_gas.v1 --species HCN

For an editable checkout, use the ``--catalog`` command printed by
``describe(setup)`` and append ``--species HCN``. From the repository root this
is equivalent to:

.. code-block:: console

   mceg explain --catalog src/exogibbs/data/mceg/index.json \
     --model exogibbs.curated_gas.v1 --species HCN

``--json`` produces a machine-readable report. The package option identifies
the installed distribution; MCEG reads its files without importing ExoGibbs or
running a calculation. Older MCEG releases without these options can still
read the individual YAML contracts in ``data/mceg/contracts``.

The HCN report links the revised preliminary FastChem comparison, numerical
checks, unknown physical accuracy, and pending source-reconstruction work.
The largest recorded deviations from curated are approximately -4.38% and
-4.67% on the two prescribed profiles. These are dataset-choice sensitivities
under the recorded conditions, not an uncertainty distribution, an error bound,
or evidence that one table is more accurate. No spectrum or retrieval was run.
HCN and HNC have separate identifiers and are never matched by formula alone.

The report does not evaluate your conditions. In particular, temperature and
pressure within the sampled extrema do not establish coverage of another
profile or composition. Numerical verification, physical support, and local
file integrity remain separate. Contracts can reference archived files not
included in the wheel; a copied contract does not establish that those files
were checked on your machine. See :doc:`fastchem_hcn_followup` for the complete
comparison, source audit and replay procedure.

Records and their updates
-------------------------

The catalog has a record version separate from the thermochemical dataset
identity. It is a release-time snapshot; installing a newer MCEG reader alone
does not supply newer ExoGibbs scientific findings. A reviewed newer catalog
can be selected explicitly with ``--catalog``. The report identifies the
selected snapshot and hashes. Ordinary calculations perform no evidence lookup
or network access.

Maintainers edit the canonical records in ``mceg/contracts`` and the archived
study outputs, then regenerate the committed distribution files:

.. code-block:: console

   python mceg/build_catalog.py

The builder uses only the standard library. It copies the contracts and small
source/summary artifacts unchanged, and derives the numerical synopsis from
the archived summary. Normal package builds use these committed files and
do not require MCEG or rerun the study. A new scientific review should update
the record version; the original evidence and its assessment scope remain
historical records.
