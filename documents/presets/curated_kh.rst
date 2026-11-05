Curated KH spectroscopic-energy adoption
=========================================

``curated-kh-v1`` adopts the reviewed spectroscopic KH dissociation energy
on ``curated-oh-v1``. ``curated.chemsetup`` selects this version at adoption;
``curated_kh.chemsetup`` keeps it reproducible when later defaults are added.
The version preserves OH, H2O, CO2, TiO and C2H updates, the 537-species
ion-inclusive / 421-species neutral catalogs, element inventories and charge
convergence. Only ``H1K1`` thermodynamics changes over 1500--4000 K, at the
one-bar ideal-gas standard pressure. Invalid temperatures return NaN.

Why adopt the smaller energy
-----------------------------

The inspected publisher abstract of
`Zemke et al. (1988) <https://doi.org/10.1016/0009-2614(88)87016-7>`_
gives D0=14282.6 cm-1, or 170857.838639 J/mol using exact SI conversion
constants. The inherited balanced JANAF value is 180390 J/mol. The
9.532161 kJ/mol difference makes the old model bind KH too strongly relative
to this spectroscopic anchor. BC2016's rounded energy agrees within 2 J/mol;
related later potential-model use is supporting consistency, not another
independent measurement. The abstract's +/-0.6 cm-1 applies to De, not D0,
and is not turned into an unsupported reaction uncertainty.

The minimal supported correction is to retain the actual five-coefficient
JANAF/FastChem formation fit and add -delta(D0)/(RT) to formation G/(RT).
Reaction G and H increase by 9532.161361 J/mol; S and Cp are unchanged.
The packaged record pins its OH base and the reviewed energy once. No
thermal refit, factor-two adjustment, nuclear-spin, ZPE or atomic correction
is added. Scalar/array/JIT evaluations and thermodynamic derivatives retain
the original atomic convention.

A full Khan state-sum replacement is a separate thermal-selection question.
At identical spectroscopic D0, six NASA thermal-reference controls increase
KH by 0.817--6.402% relative to this candidate, smaller than the primary
energy response. Both molecular and atomic thermal functions change in
those controls; this is not a purely molecular uncertainty estimate. The
1988 full text would refine a precision claim and remains recorded as a
source-access limit, rather than an unconditional gate on bounded adoption.
See :doc:`../mceg/kh_candidate` for the source comparison and finite controls.

Production verification and finite response
--------------------------------------------

The actual public factory was tested on 2501 temperatures against the frozen
candidate. The maximum formation G/(RT) difference is 2.67e-15; every non-KH
potential is unchanged. It then ran 150 fresh cold equilibrium cases:
75 per network at five temperatures (1500/2000/2500/3000/4000 K), five
pressures (1e-6/1e-4/0.01/1/100 bar), and three C/O ratios
(0.5888436554/1/1.5). Species, matrix entries, inventories and source inputs
were verified before comparing the archived candidate.

.. list-table:: Production KH response relative to curated-oh-v1
   :header-rows: 1

   * - Network
     - KH mole-fraction ratio
     - Largest decrease of the K inventory held in KH
     - Condition
   * - Ion-inclusive
     - 0.465658--0.753321
     - 5.10150% -> 3.28852%
     - 2500 K, 100 bar, C/O=1
   * - Neutral
     - 0.465659--0.768271
     - 7.15367% -> 4.99509%
     - 3000 K, 100 bar, C/O=1

The production grids reproduce archived candidate KH ln(mole fraction)
within 3.20e-14 and all mole fractions within 3.22e-15. Six tighter,
independently assembled candidate controls at tolerance 2e-14 (primary
5e-14) agree in KH ln(mole fraction) within 2.85e-14 and in any absolute
mole fraction within 3.41e-14. Arbitrary trace-species ln(x) differences can
reach 1.47e-5, so this does not claim uniform relative accuracy for all traces.

K inventory relative residuals are below 4.23e-10, H/C/O below 8.85e-13,
charge below 4.60e-14, and formation-equation residuals below 1.31e-12 in
natural-log units. The all-element relative maximum is 1.48e-6 in a tiny
inventory and is reported separately. Two independent FastChem v4.0.3
comparisons reproduce the adopted compositions within 8.72e-9 dex using
the same supplied thermodynamics.

Runtime, evidence and limits
-----------------------------

The runtime uses only packaged ``curated/kh_v1.json`` and the retained base
records, with no MCEG import or source-tree dependency. ``build_kh_adoption.py``
rebuilds its exact record from pinned candidate bytes. The fresh packet is
``tests/data/validation/curated_kh_adoption/20261105``; its receipt records the
actual command, source hashes, numerical environment and completed outputs.
Earlier receipts remain unchanged; changed historical source files are
preserved byte-for-byte under the packet's ``source`` directory.

``mceg/contracts/curated_kh.yaml`` supports the scoped scientific choice and
production implementation. Full D0 uncertainty, retained-fit accuracy,
isotope/continuum treatment and atmospheric observations remain unknown.
This ideal-gas calculation does not validate condensation, photochemistry,
transport or opacity. Numerical agreement cannot eliminate shared source
errors. These limits do not invalidate the supported energy improvement.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/build_kh_adoption.py
   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 0-3 python mceg/reviews/curated_kh_adoption.py \
     --output-dir results/kh-production-new \
     --fastchem-executable /path/to/fastchem
