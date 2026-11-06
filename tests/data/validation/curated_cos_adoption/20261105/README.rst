COS production adoption on the cumulative HO2 base
==================================================

The actual curated-cos-v1 factory was cold-solved at 150 conditions: 75 each
on the ion-inclusive and neutral networks. Six stricter independently
assembled COS-candidate-on-HO2 controls and two FastChem comparisons verify
numerical response. The earlier COS-only candidate used curated-oh-v1;
its target COS potential is retained, while KH/HO2 background changes are
reported separately from numerical reproduction errors.

The receipt pins the actual executed command, inputs, outputs and binary.
Historical source/candidate receipts remain unchanged; exact old source
bytes are retained under ``source/`` and resolved by their recorded hashes.
No full physical uncertainty, low-temperature or photochemical validation
is claimed. Reproduce with a new output directory::

   PYTHONPATH=src python mceg/reviews/curated_cos_adoption.py \
       --output-dir results/cos-adoption-new \
       --fastchem-executable /path/to/fastchem
