COS adoption: balanced energy on the cumulative HO2 base
========================================================

``curated-cos-v1`` adopts the reviewed COS energy update and becomes the
public ``curated.chemsetup`` default. It inherits KH, HO2, OH, H2O, CO2,
TiO and C2H updates through ``curated-ho2-v1``. Only COS thermodynamics
changes from that immediate base; networks, inventories, charge handling
and the one-bar ideal-gas convention are retained. Earlier versions remain
available through their named factories.

Scientific rationale and minimal construction
----------------------------------------------

The same ATcT 1.228 evaluation gives Hf(0) for C/O/S/OCS as
711.370/246.843/276.85/-141.79 kJ/mol. For C+O+S->OCS this gives
D0=1376.853 kJ/mol, 5.612 kJ/mol above the balanced inherited JANAF
1371.241 kJ/mol. The consistently balanced evaluated central energy is
preferred for this scoped update. This is not a significance test or an
independent measured error: reaction covariance is unavailable, and the
molecular marginal uncertainty is not a complete atomization uncertainty.

The implementation subtracts 5612/(R T) once from the actual inherited
COS formation G/(RT). Reaction G and H therefore decrease by 5612 J/mol;
S and Cp are unchanged. The JANAF molecular piecewise quintic and NASA
atomic thermal background are retained exactly. No thermal refit, isotope
or spin adjustment, TIPS crossover at 3380 K or OYT8 replacement is added.
The :doc:`source review </mceg/cos_source_review>` records the partition and
high-temperature limitations, and the :doc:`candidate study
</mceg/cos_candidate>` records the independent construction and finite
response that motivated adoption. These thermal limitations remain
separate from the supported energy update.

``cos_v1.json`` pins the reviewed candidate, the immediate HO2 base and the
energy delta. Runtime requires only packaged data and JAX, without MCEG.
Scalar/array/JIT evaluations and thermodynamic derivatives cover
1500--4000 K; invalid temperatures return NaN. Float64 is required for the
reviewed precision. There is no extrapolation claim outside that range.

Production verification and cumulative response
-----------------------------------------------

The actual public factory was cold-solved at 150 new conditions: 75 each
for the 537-species ion-inclusive and 421-species neutral networks.
Temperatures are 1500/2000/2500/3000/4000 K, pressures
1e-6/1e-4/0.01/1/100 bar and C/O=0.5888436554/1/1.5. The entire immediate
HO2 baseline is checked against its archived potentials, inventories,
species and formula matrix before reuse.

On 2501 temperatures, production h differs from an independently assembled
COS-candidate-on-HO2 model by at most 2.85e-14. Its COS column matches the
earlier COS-only candidate. Every non-COS column is identical to the
immediate HO2 base. The earlier candidate's whole network used
``curated-oh-v1``; replacing KH and HO2 changes that background. Comparing
whole compositions directly with the earlier COS-only run would mix the
background response with numerical reconstruction error.

.. list-table:: Incremental COS response relative to curated-ho2-v1
   :header-rows: 1

   * - Network
     - COS mole-fraction ratio, new/base
     - Largest sulfur-allocation change
   * - Ion-inclusive
     - 1.183812--1.568279
     - 1.259153e-5 -> 1.974688e-5
   * - Neutral
     - 1.183812--1.568279
     - 1.259153e-5 -> 1.974688e-5

The sulfur entries are fractions of the full S inventory. Their largest
change occurs at 1500 K, 1 bar and reference C/O, where ion-inclusive COS
x changes from 2.849208e-10 to 4.468320e-10. The change in S allocation is
0.000715535 percentage points. Relative to the *old COS-only candidate*,
adding KH/HO2 shifts COS by a factor 0.999999809--1.000000003. The largest
absolute difference across all species between those backgrounds is
4.321e-9, dominated by the KH change, not a COS numerical error.

Relative to the earlier ``curated-oh-v1`` production, all three adoptions
together produce KH mole-fraction ratios 0.465658--0.768271, HO2 ratios
0.446392--0.755162 and COS ratios 1.183812--1.568279 across both networks.
The archived summary retains each network and separates the incremental
COS effect from these cumulative changes. Small absolute COS/HO2 amounts
in this hot hydrogen-rich grid do not establish their unimportance for
cool atmospheres, photochemistry or other inventories.

Six fresh, tighter controls use the independently assembled candidate on
the same HO2 base at tolerance 2e-14, versus primary 5e-14. COS ln(x)
agrees within 1.64e-13; the largest absolute difference across all mole
fractions is 3.38e-14. The all-species trace logarithmic difference reaches
1.46e-5, reflecting trace numerical limits. H/C/O/S relative inventory
residuals remain below 2.96e-12, charge below 1.96e-14 and formation-equation
residuals below 1.65e-12 in natural-log units. All-element relative closure
reaches 1.46e-6 in much smaller elemental inventories and is reported
separately. Two independent FastChem 4.0.3 production comparisons agree
within 8.73e-9 dex over all species.

Evidence and remaining limits
------------------------------

MCEG supports the bounded scientific selection and verified production
adoption. Complete physical accuracy remains unknown: correlated energy
uncertainty, thermal-state completeness, isotopic mixture, condensation,
transport, photochemistry and application-specific atmospheric validation
are not supplied. These limits do not create a requirement to obtain every
historical source before this bounded energy improvement.

Original source/candidate receipts and numerical results are immutable.
Byte-identical historical source snapshots resolve files whose current
public default or replay entry point changed. The current production packet
pins the actual implementation and cumulative baseline used for this run.

Rebuild and reproduce with::

   python mceg/reviews/build_cos_adoption.py --output /tmp/cos_v1.json
   PYTHONPATH=src python mceg/reviews/curated_cos_adoption.py \
       --output-dir results/cos-production-new \
       --fastchem-executable /path/to/fastchem
   pytest tests/unittests/presets/curated_cos_test.py
   pytest tests/model_evidence/curated_cos_adoption_test.py

See ``mceg/contracts/curated_cos.yaml`` and
``tests/data/validation/curated_cos_adoption/20261105`` for the production
assessments, commands, hashes, compact amounts and diagnostics.
