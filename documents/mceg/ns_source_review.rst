NS: corrected thermochemistry and modern state sum
==================================================

Decision and source attribution
-------------------------------

Retain the production column during this source review and prioritize a
Peebles--Marshall/JPL energy candidate. The factor-of-3.66 difference from
NASA at 1500 K is mainly an energy-evaluation difference, not a missing spin
factor. Current ``N1S1`` is an inherited FastChem fit labelled Lodders (2004);
NASA names the same neutral molecule ``SN``. No curated data are changed.

`Lodders 2004 <https://doi.org/10.1063/1.1611178>`_ corrects earlier JANAF
formation data and selects D0=463 +/-24 kJ/mol and Hf(298)=283.40 +/-24 kJ/mol.
Tables 6 and 7 were inspected. Its thermal functions follow Gurvich.
`Peebles and Marshall 2002 <https://doi.org/10.1016/S0009-2614(02)01619-6>`_
instead recommend Hf(0)=277.3 +/-2 and Hf(298)=278.0 +/-2 kJ/mol.
The paper also gives D0=468.2 kJ/mol, a directly usable source-specific
atomization-energy candidate. These are separate evaluations. `JPL 2019, note 76
<https://jpldataeval.jpl.nasa.gov/pdf/NASA-JPL%20Evaluation%2019-5.pdf>`_
identifies the corrected thermal tables and the latter energy evaluation.
The narrower theoretical energy estimate makes a candidate worthwhile;
it does not authorize silently adding a molecular Hf from one network to
atomic anchors from another. Re-evaluate the working reactions and balanced
0 K atomization energy before substitution.

Quantitative comparison
-----------------------

All formation constants refer to N(g)+S(g)->NS(g), p0=1 bar. Official NASA
headers give balanced D0=478.999947 kJ/mol using their native R=8.314510.
The source interval begins at 300 K. The current model is only evaluated
at 1500--4000 K. At 1500/3000/4000 K, current/NASA K is
0.272994/0.516732/0.610154. Substituting the Lodders molecular Hf and thermal
nodes while holding NASA atom functions fixed reproduces current K to
1.25--1.81 percent at these nodes. This residual includes the inherited fit
and atomic convention; the historical fit process is not fully recovered.
A 5.4 kJ/mol molecular H298 change alone would multiply K by
1.54185/1.24171/1.17629. This is a diagnostic, not an adopted candidate.

`SNaSH <https://doi.org/10.1093/mnras/sty939>`_ supplies 31502 levels in the
X2Pi electronic state. The distributed degeneracies satisfy
``gtot=3*(2J+1)``; dividing by 3 removes nuclear spin while preserving both
spin-orbit and parity components. The minimum energy is zero. Ground-relative
Q agrees with distributed PF values within 6.4e-7 relative, an agreement test
which does not identify the small residual's cause. The normalized Q/NASA
ratios are 1.001245/1.003197/1.005009 at 1500/3000/4000 K. With the molecular
translation mass changed from natural 46.0717 u to isotope 45.975145 u,
the corresponding fixed-atom K proxies are 0.998099/1.000045/1.001851.
This mass adjustment does not construct an isotopic mixture.

Boltzmann moments, without numerical differentiation, give Cp changes of
+0.04457/+0.11652/+0.06529 J/mol/K against common-R NASA. At 4000 K, levels
above 24524 cm-1 carry 0.001840 of the ground-state Q and J>220 carries
2.94e-6. These are included-state diagnostics, not a bound on omitted
excited electronic states. The paper discusses missing a4Pi states and
estimates their contribution below 1 percent at 5000 K; this review does
not independently validate that estimate. Small thermal differences support
retaining the present thermal family while investigating energy separately.

Footprint and evidence boundary
-------------------------------

The archived ``curated-oh-v1`` 75-case, 537-species gas-only grid replays
exactly for NS; all endpoints converged and sulfur closure is below 3e-12.
Maximum sulfur allocation to NS is 4.34539e-5 at 4000 K, 100 bar,
C/O=0.588844; its mole fraction there is 9.16157e-10. These are existing
abundances, not replacement responses. Other atmospheric regimes may differ.

The compact packet contains original download receipts, complete selected
state columns, exact source nodes, offline calculations and tests. MCEG
separates supported source attribution, normalized numerical checks and
candidate priority from unknown physical accuracy and incomplete historical
fit provenance. Neither additional inaccessible Gurvich volumes nor a full
planetary validation is a prerequisite to merging this scoped review.
