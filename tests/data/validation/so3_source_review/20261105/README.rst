SO3 source review, 2026-11-05
=============================

``inputs.json`` pins molecular identity, same-version ATcT1.228 values,
atomic table paths, temperatures and partition normalization. ``sources.json``
records downloaded originals and access limits. ``nasa_records.json`` is
parsed from the literal selected NASA records; original-record coefficients
are independently checked against the existing frozen screening input.
Molecular and reused atomic source tables supply exact sampled knots.

``comparison.json`` is the actual offline output and ``receipt.json`` pins
its code, dependencies, numerical inputs and the existing curated-cos-v1
grid. No new solve, continuous candidate or production change is included.
The archive footprint is 75 cases and 537 species; it must not be reported
as a new equilibrium replacement or all-atmosphere physical validation.

Run ``PYTHONPATH=src python mceg/reviews/so3_source_review.py`` from the
repository root; ``--output PATH`` writes the same JSON. Tests additionally
check the named ``curated_cos`` model and source-specific physical limits.
Full copyrighted papers and web pages are not committed.
