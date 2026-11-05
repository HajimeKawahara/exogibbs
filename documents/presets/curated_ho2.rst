Curated HO2 adoption
====================

``curated_ho2.chemsetup`` implements the reviewed NASA thermal functions
and balanced ATcT 1.172 energy for HO2 on the immutable ``curated-kh-v1``
base. It is selected as the public ``curated.chemsetup`` in this adoption.
Only HO2 changes. The KH, OH, H2O, CO2, TiO and C2H improvements, species,
element inventories, charge treatment and one-bar standard pressure remain.
Earlier versions remain explicitly available through their named presets.

Scientific reason and exact representation
------------------------------------------

The inherited JANAF HO2 formation energy, Hf(298.15 K)=2.092 kJ/mol, is
superseded by the inspected evaluations near 12 kJ/mol. The
`Burgess evaluation <https://doi.org/10.6028/jres.121.005>`_ independently
documents the old JANAF problem. The candidate review
(:doc:`../mceg/ho2_candidate`) separates energy and thermal effects, confirms
continuous calorics and quantifies conserved finite responses. A tiny bulk
abundance effect in a hot hydrogen-rich gas does not justify retaining the
known obsolete central energy.

Use the same ATcT 1.172 H/O/HO2 Hf(0) values, giving D0=694.662 kJ/mol
for H(g)+2O(g)->HO2(g). The exact HO2-H-2O NASA9 coefficients are copied
without a fit over 1500--4000 K, entirely within their 1000--6000 K segment.
NASA's ground anchors use its native R=8.314510; ATcT uses modern SI R.
The difference D0_ATcT/R_SI-D0_NASA/R_NASA=-2.02096231694 K is inserted
once in the enthalpy constant. Molecular Hf is not substituted without
balancing the atomic references; there is no extra spin factor or ZPE term.

At identical ATcT energy and NASA atomic thermal reference, JANAF original
nodes differ by +0.23% to -2.01% in K from the NASA candidate. That modest
thermal contrast supports the exact NASA representation while avoiding a
new JANAF interpolation. It is not a physical uncertainty interval.
ATcT reaction covariance, historical source ancestry, isotope/continuum
completeness and atmospheric accuracy remain explicitly unquantified.

Production verification and compositional response
--------------------------------------------------

The adoption uses a new KH background, so it verifies HO2 thermodynamics
against the frozen candidate and independently assembles the candidate on
KH for composition controls. It does not assert equality to the old
HO2-only abundances on ``curated-oh-v1``. The KH production baseline is
checked against its actual full potential vectors, species and inventories.

All 150 new production endpoints converge: 75 conditions each on the
537-species ion-inclusive and 421-species neutral networks, at five
temperatures 1500/2000/2500/3000/4000 K, five pressures 1e-6--100 bar and
C/O=0.5888436554/1/1.5. The dense 2501-temperature potential comparison
with the independently assembled candidate differs by at most 3.56e-14.
All non-HO2 potentials are exactly the KH base values.

HO2 mole fractions become 0.446393--0.755162 of the KH-base values in both
networks. At 4000 K, 1 bar and reference C/O, ion-inclusive x(HO2) changes
from 8.91130e-12 to 6.72947e-12, and its oxygen inventory fraction from
3.28852e-8 to 2.48336e-8. The largest absolute compositional response is
about 4.73e-12 in H2O. This result concerns the fixed hot gas equilibrium
grid; it says nothing about HO2's importance in cooler atmospheres or
photochemical and kinetic radical networks.

The primary tolerance is 5e-14. Six fresh candidate-on-KH cold controls at
2e-14 reproduce HO2 ln(x) within 3.27e-13 and any mole fraction within
3.39e-14. Maximum all-trace ln(x) disagreement is 1.47e-5; those trace
relative limits are distinct from the target response. H/C/O relative
inventory closure is below 8.95e-13, charge closure below 3.13e-14 and
formation-equation residual below 1.43e-12. The all-element relative
maximum is 1.47e-6, dominated by tiny inventories. Two fresh FastChem
comparisons agree within 1.20e-8 dex. These checks validate the numerical
model, not complete physical or atmospheric accuracy.

Records and reproduction
-------------------------

The packaged record is ``src/exogibbs/data/curated/ho2_v1.json``; runtime
validates its hash and the exact KH base. Runtime does not require MCEG.
Scalar/array/JIT/vmap evaluation, neutral projection and independent NASA
reaction G/H/S/Cp are tested; invalid temperatures return NaN. Float64
is required for the reviewed numerical precision.

The packet ``tests/data/validation/curated_ho2_adoption/20261105`` contains
the real execution receipt, conserved amounts, controls and summary.
The old source/candidate receipts remain unchanged, with byte-identical
snapshots resolving earlier default-dependent inputs. The MCEG contract
``mceg/contracts/curated_ho2.yaml`` supports the scoped scientific choice
and production verification; full physical accuracy stays unknown.

.. code-block:: console

   python mceg/reviews/build_ho2_adoption.py
   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 4-7 python mceg/reviews/curated_ho2_adoption.py \
     --output-dir results/ho2-adoption-new \
     --fastchem-executable /path/to/fastchem
   PYTHONPATH=src pytest tests/unittests/presets/curated_ho2_test.py tests/model_evidence/curated_ho2_adoption_test.py
