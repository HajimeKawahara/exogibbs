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
   print(info["decisions_command"])

The returned dictionary can be saved with ``json.dump``. It describes declared
setup metadata, not a saved execution or its temperature, pressure and elemental
composition. It does not verify a modified callable that retains the old
metadata. A different dataset hash or network choice gives
``no_matching_record``; that means no matching review, not physical reliability.

The default ion-inclusive setup matches ``exogibbs.curated_gas.v2``. Its
record covers the cumulative gas-v2 adoption and the
:doc:`Gas curation v0.1 scientific snapshot <gas_curation_release_0_1>`.
The release version ``0.1.0`` is recorded separately from the unchanged model
identifier and software version. Explicit earlier named curated factories,
including ``curated_tio2``, retain their historical adoption records.
``curated_v1.chemsetup()`` matches ``exogibbs.curated_gas.v1`` and its historical
HCN comparison. These studies are not rebound across dataset identities.
The catalog is not a complete uncertainty inventory. Both-network production
verification is recorded in ``mceg/contracts/curated_gas_v2_adoption.yaml``;
the packaged lookup remains ion-inclusive.

Inspect the default with optional MCEG::

   mceg explain --catalog src/exogibbs/data/mceg/index.json \
     --model exogibbs.curated_gas.v2 --species O2

Inspect HCN with optional MCEG
------------------------------

The following commands inspect **the original curated-gas-v1 HCN study**.
They do not report evidence for the corrected default. With an MCEG version
providing installed-package catalogs, run:

.. code-block:: console

   mceg explain --package exogibbs --model exogibbs.curated_gas.v1 --species HCN

For an editable checkout, use the ``--catalog`` command printed by
``describe(curated_v1.chemsetup())`` and append ``--species HCN``. From the repository root this
is equivalent to:

.. code-block:: console

   mceg explain --catalog src/exogibbs/data/mceg/index.json \
     --model exogibbs.curated_gas.v1 --species HCN

``--json`` produces a machine-readable report. The package option identifies
the installed distribution; MCEG reads its files without importing ExoGibbs or
running a calculation. Older MCEG releases without these options can still
read compatible individual YAML contracts in ``data/mceg/contracts``. The
additional schema 0.4 records require MCEG's Decision support.

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

Inspect a source choice
-----------------------

Curated source choices live in native schema 0.4 Decision records in
``mceg/decisions``. Existing contracts retain their historical contents. Use an
MCEG checkout containing
`PR #24 <https://github.com/HajimeKawahara/mceg/pull/24>`_ to inspect a choice
and its recorded basis:

.. code-block:: console

   mceg explain mceg/decisions/c2h_v1.yaml \
     --decision source_choice_C2H1 --investigation
   mceg check mceg/decisions/c2h_v1.yaml --decision source_choice_C2H1

For an installed preset, ``describe(setup)["decisions_command"]`` returns the
equivalent ``mceg explain`` command for its bundled Decision records. Run it to
see the available IDs, then append ``--decision ID --investigation`` for one
choice. For example, ``curated_c2h.chemsetup()`` exposes
``source_choice_C2H1``. The field is ``None`` when the declared metadata has no
matching Decision record. Constructing the command does not require MCEG.

The records transcribe the archived choices without a new scientific review.
A selected option does not establish physical accuracy: unknown Assessment
outcomes and source limitations remain separate. Basis fingerprints identify
the recorded premises; a changed premise calls for review without replacing
the historical choice. ``--decision-revision 1`` selects an exact revision.

The pinned numerical JSON files retain their original ``decision`` or
``decisions`` fields, and presets still expose ``species_provenance`` metadata.
These are frozen compatibility snapshots, preserving dataset hashes and the
existing API. Author subsequent Decision revisions in ``mceg/decisions``,
retaining earlier revisions and their premises. Calculations continue to read
packaged numerical data without importing MCEG or evaluating Decision records.

Records and their updates
-------------------------

The catalog has a record version separate from the thermochemical dataset
identity. It is a release-time snapshot; installing a newer MCEG reader alone
does not supply newer ExoGibbs scientific findings. A reviewed newer catalog
can be selected explicitly with ``--catalog``. The report identifies the
selected snapshot and hashes. Ordinary calculations perform no evidence lookup
or network access.

Maintainers edit the canonical records in ``mceg/contracts`` and
``mceg/decisions`` and the archived study outputs, then regenerate the committed
distribution files:

.. code-block:: console

   python mceg/build_catalog.py

The builder uses only the standard library. It copies the contracts and small
source/summary artifacts unchanged, and derives the numerical synopsis from
the archived summary. Normal package builds use these committed files and
do not require MCEG or rerun the study. A new scientific review should update
the record version; the original evidence and its assessment scope remain
historical records.
