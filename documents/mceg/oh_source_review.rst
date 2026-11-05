OH source review
================

OH should advance to a replacement candidate. Its inherited JANAF/FastChem
construction has a documented electronic-degeneracy and caloric problem;
agreement with that historical fit is not a reason to retain it indefinitely.
The smallest justified next candidate is NASA thermal functions with a balanced
ATcT 1.172 atomization energy. MYTHOS is a useful modern thermal alternative for
the candidate comparison. This review changes no production coefficients and
performs no equilibrium substitution.

The current source
------------------

Neutral ``H1O1`` is the OH radical, with an X :sup:`2` Pi ground electronic
state. It inherits the five-coefficient FastChem4 fit labelled JANAF 1998;
the CO2, H2O, TiO and C2H updates do not replace this column. The frozen
``curated-h2o-v1`` 75-case archive reproduces it to below 1e-15 in dimensionless
chemical potential. At 1500 K its formation equilibrium constant is
0.87297555 times the frozen NASA value; at 4000 K it is 0.94322869 times NASA.

Ruscic et al.'s `IUPAC evaluation (2005) <https://doi.org/10.1063/1.1724828>`_,
OH section pp. 633--638, explicitly identifies incorrect ground-electronic
degeneracy/multiplicity treatment in JANAF, and errors in caloric increments
and formation-enthalpy temperature corrections. Its discussion covers the
1998 edition too. The official OH table still gives H(298.15)-H(0)=9.172
kJ/mol; the IUPAC/Gurvich value is 8.813, and the MYTHOS state sum gives
8.813558 kJ/mol. Thus this is an identified source defect, beyond an
unexplained discrepancy between two compilations. IUPAC adopts Gurvich
thermal functions, so agreement with NASA/Gurvich is not independent evidence.

Energy and thermal separation
-----------------------------

The reaction is H(g)+O(g)->OH(g), with standard pressure 1 bar. The frozen
NASA headers, official JANAF H/S tables and same-version ATcT atom values give:

.. list-table:: Balanced ground-relative atomization energies
   :header-rows: 1

   * - Source
     - D0 (kJ/mol)
   * - NASA CEA headers
     - 425.785900
   * - JANAF H/S reconstruction
     - 424.435000
   * - ATcT 1.172: 216.034 + 246.844 - 37.274
     - 425.604000

The OH enthalpy alone cannot define this correction: H and O must come from
the same evaluation. Atomic O refers to its lowest fine-structure level, not
a term barycentre. The NASA atomic thermal functions retain the thermally
populated fine structure; replacing O by a constant degeneracy would change
the reaction. The native NASA gas constant is 8.31451 J/mol/K, while modern
SI R is used for ATcT energy and the state calculation. Consequently the
NASA-to-ATcT shift in D0/k is -21.5857097163 K, rather than the raw energy
difference divided by modern R. Correlated ATcT network uncertainty was not
propagated; the displayed OH uncertainty is not a reaction uncertainty.

At 1500 K, ln(K_JANAF/K_NASA) separates into -0.108123 from balanced energy,
-0.029530 from molecular thermal functions, and +0.000147 from atomic thermal
functions. The inherited fit is only 0.166% above the actual JANAF H/S
reconstruction there. A better fit to the same defective table would therefore
leave the main issue intact. The frozen historical comparison is preserved.

Modern states and conventions
-----------------------------

`MYTHOS (Mitev et al. 2025) <https://doi.org/10.1093/mnras/stae2803>`_ supplies
10606 distributed states from X, A, B and C electronic manifolds, up to J=49.5.
The downloaded ``MoLLIST-OH`` file has 1878 states, although its definition
metadata says 2640. The actual rows, not the metadata count, are summed.
`Brooke et al. (2016) <https://doi.org/10.1016/j.jqsrt.2015.07.021>`_ describes
the X-state sum and the unresolved hydrogen hyperfine factor of two.
The inspected Brooke copy is the author's arXiv preprint; MYTHOS was inspected
as the published author-hosted PDF.

Both distributions have g_tot=2(2J+1). Divide by the nuclear-spin factor two
once. Do not divide by the electronic doublet factor or remove Lambda doubling:
the fine-structure and parity levels are explicit. The ground level has
E=0, J=1.5, g_tot=8, giving spin-free Q->4 as T->0; the next parity level is
0.055403 cm-1 in MYTHOS. At 1 K both levels contribute substantially.
The comparison uses :sup:`16` O :sup:`1` H internal states and natural molecular
mass 17.00734 u, matching the elemental thermochemistry convention. Using
17.00273965 u changes molecular translation by a separate factor 0.9995943;
that diagnostic alone is not an isotope-resolved reaction calculation.

.. list-table:: Aligned source diagnostics (spin-free Q; common modern R for Cp)
   :header-rows: 1

   * - T (K)
     - MYTHOS Q / NASA Q
     - MYTHOS Cp
     - NASA Cp
     - JANAF Cp
   * - 1500
     - 1.00007293
     - 32.94853
     - 32.96395
     - 32.917
   * - 3000
     - 0.99957904
     - 36.99866
     - 37.03737
     - 36.736
   * - 4000
     - 0.99902479
     - 38.47025
     - 38.53611
     - 37.885

Cp units are J/mol/K. H, S and Cp are computed from exact Boltzmann moments,
not finite differences of a rounded Q table. MYTHOS Q reproduces its published
table within its printed rounding at the sampled nodes. MoLLIST's published
Q differs from the current state sum by 0.55 ppm at 4000 K, exceeding its
printed Q rounding; this small distribution residual is recorded without an
invented correction. MYTHOS and MoLLIST also share empirical spectroscopy.

At 4000 K MYTHOS's last J shell carries 1.96 ppm of Q, J>=45.5 carries
23.88 ppm, and v>=10 carries 108.72 ppm. Excited electronic manifolds carry
5.42 ppm. The paper includes quasi-bound levels as well as strictly bound
ones and excludes continuum states. Removing every level above the ATcT
ground-atom threshold reduces Q by 33.56 ppm and Cp by 0.03815 J/mol/K.
This is a defined sensitivity, not a physically complete continuum treatment
or a rigorous omitted-state bound. The paper's last-shell convergence metric
and 8000 K opacity scope do not establish full thermodynamic uncertainty.

Candidate decision and atmospheric scope
----------------------------------------

With the same NASA atomic thermal functions, natural mass and ATcT D0:

.. list-table:: Pointwise formation-constant alternatives divided by current curated
   :header-rows: 1

   * - T (K)
     - NASA thermal + ATcT
     - MYTHOS thermal + ATcT
   * - 1500
     - 1.12914110
     - 1.12922346
   * - 3000
     - 1.06582346
     - 1.06537479
   * - 4000
     - 1.05448246
     - 1.05345412

NASA/ATcT is the preferred first implementation candidate because it already
provides continuous thermal functions, is consistent with the IUPAC/Gurvich
evaluation, and differs from the modern MYTHOS Q by less than 0.1% at the reviewed nodes
in 1500--4000 K. This is a practical scoped preference, not proof that NASA is
more accurate. A newer line list's improved intensities do not by themselves
require a different thermochemical model. MYTHOS/ATcT should be retained as
a second thermal route in the candidate evaluation, including the threshold
sensitivity above. Merely changing the old JANAF energy would preserve its
known caloric defect; the next candidate must replace its thermal functions too.

The archived current model places at most 11.2391% of oxygen in OH, at 4000 K,
1 bar and C/O=0.588844; the corresponding mole fraction is 6.09116e-5.
Oxygen conservation closes below 1.5e-13. These are existing gas-equilibrium
endpoints, not an abundance response to either candidate, a photochemical
calculation or an observational validation.

The next work is a continuous candidate plus one-species finite substitution
on the fixed network, checking H/O conservation and H2O/OH/O/O2 responses.
No unavailable essential paper blocks that work or merging this source review.
Historical Gurvich pages, network covariance and a complete continuum treatment
remain provenance/uncertainty follow-ups, not unconditional merge requirements.

Reproduction and MCEG
---------------------

Run ``python mceg/reviews/oh_source_review.py`` for an offline replay, and
``pytest tests/model_evidence/oh_source_review_test.py`` for focused checks.
Original ExoMol state/PF downloads can regenerate the compact arrays with
``python mceg/reviews/oh_source_review.py --extract /path/to/downloads``;
the required local filenames are listed in ``sources.json``. Original download
hashes, compact-array hashes and inspection/access limits are recorded under
``tests/data/validation/oh_source_review/20261105``. Full papers are not vendored.

``mceg/contracts/oh_source_review.yaml`` separates reproducible source
construction, state/caloric comparison and the scoped candidate recommendation
from physical accuracy, which remains unknown. Current curated coefficients
and prior receipts remain unchanged.
