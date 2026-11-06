Curated gas v2 / FastChem 4.0.4 formation constants
================================================

This offline packet compares dimensionless one-bar formation equilibrium
constants from neutral gas atoms to one neutral gas molecule. It contains
108 explicitly mapped molecules at 1500, 2000, 2500, 3000 and 4000 K, giving
540 constant comparisons and zero equilibrium abundance solves.

``source/fastchem-v4.0.4-logK.dat`` is the unmodified standard full gas table
from NewStrangeWorlds/FastChem commit
``0c4b448166e1d0a0f01e3416bf7da0082ef0546e`` (release v4.0.4).
Its SHA256 is
``3697abccf664aa5f531103b6c7aac6e77efcd4366f8b69d46b7e6ea29a10e0c5``.
It is byte-identical to the previously supplied HCN-update archive's
``logK.dat``. The official v4.0.4 ``wo_ions`` files still contain the older
HCN coefficients, so this comparison selects neutral records from the full
table. ``sources.json`` records the inspected release, source hashes,
HCN coefficients and the formation/density convention.

The FastChem data and two unmodified inspected C++ source files retain their
upstream attribution and GPLv3 licensing, compatible with the repository's
GPL-3.0-or-later license. Primary source URLs are in ``sources.json``.

The runtime comparison uses the named ``curated_v2`` factory. The historical
public alias is preserved in ``source/curated_at_execution.py``; replay
does not depend on a future default alias. Packaged presets and coefficients
are not changed by this comparison.

``targets.json`` preserves the 93 historical NASA mappings, 14 original NASA
exclusions and the additional C3H record. Identical source IDs and compositions
support record matching; they do not independently validate molecular
identities or physical accuracy. NASA-excluded records have no NASA ratio.
The 93 current-curated/NASA ratios use the unchanged NASA reference from
``gas_source_comparison/baseline-001``. The former Issue #308 body, including
its historical numerical screen, is preserved in ``source/issue308-before.md``.

``comparison.json`` and ``comparison.csv`` retain full-precision results;
``table1500.csv`` is the rendered projection. ``receipt.json`` records the
actual CPU float64 execution, selected input hashes and output hashes.
The scalar five-coefficient evaluation is checked against the existing
FastChem provider. Green/yellow/red mean factors below 2, from 2 to below 10,
and at least 10; they are screening labels, not accuracy assessments.

Replay from the repository root::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_fastchem4_comparison.py

Generated replay files go to the ignored ``results/`` directory. No network,
FastChem executable or MCEG installation is needed for numerical replay.
