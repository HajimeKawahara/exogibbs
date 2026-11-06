CN production adoption on the unchanged COS base
=================================================

The versioned curated-cn-v1 factory passes 150 fresh cold solves, six tighter
independently assembled candidate controls and two independent FastChem
comparisons. Both ion-inclusive and neutral networks are included. The earlier
CN candidate used the same COS base and reproduces within numerical controls;
its archived ion-inclusive execution is not relabelled as a new calculation.

Only the physical source-energy difference is applied. G and H increase by
4801 J/mol; S, Cp and the literal thermal fit are unchanged. Complete physical
accuracy, reaction covariance and atmospheric validation remain unknown.
The receipt pins the actual command, source bytes, output bytes and binary.
The named factory is independently usable before the cumulative public-default
switch. Reproduce in a new output directory::

   PYTHONPATH=src python mceg/reviews/curated_cn_adoption.py \
       --output-dir results/cn-adoption-new \
       --fastchem-executable /path/to/fastchem
