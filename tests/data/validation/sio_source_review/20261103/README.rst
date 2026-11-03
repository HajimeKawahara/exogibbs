SiO source-review packet
=========================

This packet retains the current NASA thermal plus ATcT 1.172 energy choice.
Official JANAF SiO/Si/O tables and NASA coefficient excerpts are frozen;
existing adoption inputs and the 75-case grid remain shared, hash-pinned inputs.
``literature.json`` distinguishes original full-text, evaluated compilation
and unavailable-original inspection. ``sources.json`` hashes original files.

``partition_samples.json`` contains six direct samples from the official
SiOUVenIR PF/Cp files and full-state moment summaries. The latter use all
174250 levels with degeneracy 2J+1, ground rovibrational zero and no extra
nuclear-spin factor. The complete states file is not vendored. Given the
original compressed file matching the recorded hash, regenerate its summaries::

   PYTHONPATH=src python mceg/reviews/sio_source_review.py --states /path/to/28Si-16O__SiOUVenIR.states.bz2

The ordinary offline replay uses the frozen moments and independently
published PF/Cp samples. It reconstructs heat capacity from the variance,
removes NASA translation explicitly, and checks the archived silicon budget.
Internal isotope differences are not removed by translational normalization.
No new equilibrium calculation or production change is performed.

Full PDFs are not redistributed. The official NASA data license remains in
``src/exogibbs/data/nasa9/LICENSE.txt`` and ``NOTICE.txt``. The ExoMol dataset
is identified by DOI 10.5281/zenodo.5910818; only selected numerical facts and
derived moment summaries are archived here.
