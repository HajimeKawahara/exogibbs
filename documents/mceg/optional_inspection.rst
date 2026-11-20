Inspect a model's known uncertainties
=====================================

ExoGibbs calculations work without MCEG installed. Scientific contracts and
Decision records are maintained in the repository's ``mceg/`` directory and
are not bundled with the installed package. Use optional MCEG from a checkout
to inspect these records; the commands below run from the repository root.

Inspect the recorded dataset
----------------------------

The current curated default uses ``curated-gas-v2``. Its contract covers the
cumulative gas-v2 adoption in both networks. The separate
:doc:`Gas curation v0.1 scientific snapshot <gas_curation_release_0_1>` freezes
the review at release version ``0.1.0``; that version is distinct from the
model identifier and software version.

.. code-block:: console

   mceg explain mceg/contracts/curated_gas_v2_adoption.yaml
   mceg explain mceg/contracts/gas_curation_release_0_1.yaml

Earlier named curated factories retain their historical contracts. These
records describe the stated datasets and study conditions, not a user's
modified setup or calculation. They are not a complete uncertainty inventory.

Inspect the historical HCN comparison
-------------------------------------

This command inspects **the original curated-gas-v1 HCN study**. It does not
report evidence for the corrected default:

.. code-block:: console

   mceg explain mceg/contracts/fastchem_hcn_followup.yaml

Append ``--json`` for a machine-readable report. MCEG reads the contract
without importing ExoGibbs or running a calculation.

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
file integrity remain separate. Reading a contract does not establish that
its referenced archived files were checked on your machine. See
:doc:`fastchem_hcn_followup` for the complete comparison, source audit and
replay procedure.

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

Run ``mceg explain mceg/decisions/gas_v2.yaml`` to list the source-choice IDs
for the current gas-v2 adoption.

The records transcribe the archived choices without a new scientific review.
A selected option does not establish physical accuracy: unknown Assessment
outcomes and source limitations remain separate. Basis fingerprints identify
the recorded premises; a changed premise calls for review without replacing
the historical choice. ``--decision-revision 1`` selects an exact revision.

The pinned numerical JSON files retain their original ``decision`` or
``decisions`` fields, and presets still expose ``species_provenance`` metadata.
These are frozen compatibility snapshots, preserving dataset hashes and the
existing preset metadata. Author subsequent Decision revisions in
``mceg/decisions``, retaining earlier revisions and their premises. Calculations
continue to read packaged numerical data without importing MCEG or evaluating
Decision records.

Records and their updates
-------------------------

The repository's records are versioned separately from the numerical datasets.
Installing a newer MCEG reader does not supply newer ExoGibbs scientific
findings. Select the repository revision containing the records you intend to
inspect. Published scientific snapshots and their assessment scopes remain
historical records; a new review should retain its predecessors.

Maintain records in ``mceg/contracts`` and ``mceg/decisions``. Documentation is
generated directly from these files using ``mceg/docs.yaml``; see
:doc:`build`. Package builds do not copy these records or require MCEG.
