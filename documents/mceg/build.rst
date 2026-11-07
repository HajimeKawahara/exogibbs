Build models and evidence
=========================

MCEG generates reStructuredText and CSS from the configuration in
``mceg/docs.yaml``. This repository owns the Sphinx build, theme and HTML output.
The generated pages need only Sphinx's standard Graphviz extension; they do not
import MCEG, MyST or scientific providers. The existing API documentation may
still import this package.

Install Graphviz (the ``dot`` executable), this package's documentation extras,
and a checkout of MCEG containing the ``docs`` command. With sibling checkouts
and Python 3.10 or newer:

.. code-block:: bash

   python -m pip install -e '.[docs]' -e '../mceg'
   ./update_doc.sh

The update script first generates RST, then builds the existing HTML docs.
To regenerate sources alone, without Sphinx or numerical execution:

.. code-block:: bash

   python -m mceg docs --config mceg/docs.yaml

Open ``documents/_build/html/mceg/index.html``. Generated sources and CSS live
in ``documents/mceg/_generated/`` and are ignored by Git. The generator's manifest
tracks its files, preserves neighboring authored files, and removes obsolete
owned pages. Edit the configuration or original records instead of generated RST.

Inputs and interpretation
-------------------------

Paths in ``mceg/docs.yaml`` resolve relative to that configuration file.
``contracts`` selects provider roots and explicit contract patterns; ``cases``
selects authored calculation connections. ``source_root`` and ``output`` identify
the Sphinx tree and its dedicated generated directory. The catalog includes all
``mceg/contracts/*.yaml`` contracts in this repository. Archived reviews and
unregistered models are separate; the catalog is not a complete model inventory.

Each model has a short overview and separate Applicability, Assessments,
Evidence, References and Implementation & records pages. Evidence and Assessment
records have individual pages. Technical metadata can be expanded; unknown
support and source-review gaps remain visible. No current assessment is selected
from a historical list.

Local source downloads contain build-checkout files. Recorded implementation
revisions and artifact hashes are displayed without rechecking those bytes.
External citations are links, not newly inspected papers. Generation never
imports scientific implementations, executes solvers or changes assessments.

The H2 case additionally reads ExoEOS's ``zhang_duan_h2.yaml`` from a sibling
``../exoeos`` checkout. Both selected contracts have SHA-256 pins; a changed pin
stops generation. The case retains unknown combined physical support and does
not inherit judgments from an older contract snapshot. To build only the local
catalog, use a configuration with ``cases: []`` and remove the case toctree entry.

Gallery execution is independent of MCEG generation. For a documentation build
without running numerical gallery examples:

.. code-block:: bash

   ./update_doc.sh -D plot_gallery=0

Refresh the package graph snapshot
----------------------------------

The :doc:`package_graph` embeds ``documents/_static/package-graph.html``,
a self-contained MCEG Viewer export. Sphinx copies this checked-in snapshot
with the other static assets, so ordinary documentation builds need neither
Node.js nor a running viewer. Rebuild the snapshot when the bundled catalog
or its contracts change; the offline evidence tests check their hashes.

The snapshot uses the ExoGibbs v0.8.0 wheel. Standalone HTML export requires
MCEG Viewer with **Save as HTML**, introduced in MCEG revision
``e997ace64439a3e111136a84bff520150cf9b371``.
To refresh it, use an environment containing the intended ExoGibbs release
and MCEG with its ``graph`` command, then export the complete package catalog::

   mkdir -p results
   python -m mceg graph --package exogibbs > results/package-graph.json

From the selected MCEG checkout, build and open its viewer::

   cd viewer
   npm ci
   npm run build
   npm run preview

Open ``package-graph.json`` in the viewer, select English, and choose
**Save as HTML**. Replace ``documents/_static/package-graph.html`` with the
downloaded file. Keep the Viewer license in
``documents/_static/package-graph-LICENSE.txt`` and retain the export's
embedded notices when updating the viewer.
The graph includes every record in the loaded export, regardless of the
current search or filters. It does not embed linked papers.

Run the offline snapshot checks and rebuild the manual::

   pytest tests/model_evidence/package_graph_test.py
   ./update_doc.sh -D plot_gallery=0 -W --keep-going
