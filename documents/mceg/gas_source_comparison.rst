Gas-source comparison baseline
==============================

``baseline-001`` preserves the JANAF/NASA and curated/NASA screening discussed
in `Issue #308 <https://github.com/HajimeKawahara/exogibbs/issues/308>`_. It fixes
the inputs, comparison rules and source annotations before individual molecular
investigations. This comparison does not change the production preset or select
a preferred database.

The versioned record is
``tests/data/validation/gas_source_comparison/baseline-001/``. Its
:download:`frozen inputs
<../../tests/data/validation/gas_source_comparison/baseline-001/inputs.json>`
contain molecular and atomic records, source identifiers, mappings, exclusions,
the major-molecule list and the expected historical table. The
:download:`comparison
<../../tests/data/validation/gas_source_comparison/baseline-001/comparison.json>`
contains sampled results and methods; the
:download:`execution receipt
<../../tests/data/validation/gas_source_comparison/baseline-001/receipt.json>`
records the command, environment and input/code/output hashes.

The historical Issue values are replay expectations. The archived receipt
belongs to a fresh execution of the portable comparator, not to the original
scratch probe. Neither an updated Issue nor a new run overwrites that history.

Reading the table
-----------------

Both ratios are dimensionless formation ``Kp`` ratios at 1500 K, relative to
NASA, for formation from neutral gas atoms at a 1 bar standard pressure. JANAF
uses ``G(T) = Hf(298.15) + H(T) - H(298.15) - T S(T)`` and its own atomic
references; NASA uses unmodified polynomial coefficients and its own atomic
references. Curated uses the actual preset pinned by the receipt and MCEG
subject. The broader report samples 500, 1000, 1500 and 3000 K. Curated samples
outside its 1500--4000 K fitted range are missing, rather than extrapolated.

For either ratio ``r``, the color uses ``F = max(r, 1/r)`` before display
rounding: green for ``F < 2``, yellow for ``2 <= F < 10``, and red for
``F >= 10``. Missing values receive no agreement grade. These colors measure
numerical disagreement; they do not establish which source is physically
accurate or express a calibrated uncertainty.

The diamond (🔷) is a separate flag for a molecule on the frozen atmospheric
major-molecule list when ``r <= 0.9`` or ``r >= 1.1`` in that comparison column.
It means at least a 10% difference relative to NASA. An asterisk (``*``) marks a
provisional molecular mapping; a dagger (``†``) marks the historical JANAF
formation-column inconsistency for SH, S2O or PH2. Freezing a mapping does not
confirm its molecular identity.

The rightmost column states the basis of the pinned curated entry, such as
``NASA CEA + ATcT 1.172``. It is not a Yes/No ATcT flag. ``[U]`` means that the
runtime entry is ``inherited_unreviewed``; its annotation does not establish
the exact upstream file, fitting procedure or absence of ATcT ancestry.
The word "current" in the historical table refers to the pinned baseline
implementation and data, not whichever preset a future checkout provides.

There are 93 included molecules, 14 explicitly excluded candidates and 70
provisional mappings. The separate count of 70 ``inherited_unreviewed`` curated
entries describes source-review status, not mapping status. The local NASA
copy has not been authenticated as an official acquisition, even though 94
coefficient intervals match the existing official-derived reference fixture.
All required records are frozen here; replay does not require a sibling VULCAN
checkout or network access.

.. include:: ../../tests/data/validation/gas_source_comparison/baseline-001/table.rst

Reproduce and inspect
---------------------

From the repository root, generate a new run in an unused directory and check
the archived baseline:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python mceg/reviews/gas_source_comparison.py \
     --output-dir results/gas_source_comparison/replay-001
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
     python mceg/reviews/gas_source_comparison.py --check
   python -m pytest -q tests/model_evidence/gas_source_comparison_test.py

The comparator refuses to overwrite an existing output directory. CSV,
reStructuredText and Markdown tables are generated from the same comparison
record. Markdown can be copied to the Issue; the Issue is a discussion surface,
while the archived inputs, report and receipt are the reproducible record.
The check verifies hashes and exact table/metadata agreement, with absolute
tolerance ``2e-11`` and relative tolerance ``2e-12`` for archived numerical
values. Historical ratios must replay within ``1e-10`` dex; their basis labels,
grades and major-molecule flags must match exactly.

With MCEG 0.2 installed separately:

.. code-block:: bash

   mceg check mceg/contracts/gas_source_comparison.yaml \
     --assessment baseline_replay_review --root .

The :download:`contract <../../mceg/contracts/gas_source_comparison.yaml>` binds
the selected code and inputs to a Subject, the sampled conventions to a
Context, the receipt and results to Evidence, and the scoped judgment to an
Assessment. The full execution receipt is an ExoGibbs artifact attached to
Evidence; it does not introduce an MCEG schema extension. MCEG verifies declared
file/artifact hashes and displays the stored assessment. The partial source
audit intentionally remains unknown. The comparator performs the numerical
replay. MCEG is not required to run that replay or use ExoGibbs.

Adding evidence
---------------

Generate exploratory outputs under ``results/gas_source_comparison/``. After
review, archive a new immutable snapshot under a new identifier, retain its
receipt, and add the corresponding MCEG Subject, Evidence and Assessment.
Preserve the earlier snapshot and assessment. A changed proposition or
tolerance also needs a new claim revision; unchanged criteria can be assessed
against a new subject without rewriting the old claim.

Keep source inspection, numerical agreement and physical validation separate.
The contract leaves physical accuracy unknown and records follow-ups with
required inputs and completion criteria for molecular identity, source
ancestry and physical constraints. A source adoption needs its own reviewed
evidence and preset change. This baseline does not start those investigations.
