Finite C, S and N on the source hosts
=====================================

``examples/metal_silicate/cns_inventory.py`` varies finite local atom budgets
using the pinned GCE Carbon and S/N models described in
:doc:`subneptune_sulfur`. Each point solves the original host's chemistry
at its frozen temperature and a prescribed local pressure. All non-scanned
element budgets, including O and the background host elements, stay fixed.

Run the offline controls::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
      python examples/metal_silicate/cns_inventory.py

The four sequences use 0, 0.25, 1 and 2 times the selected source element
budget at 2350 K:

* Carbon: the Carbon source host at 1000 bar, with N and S absent.
* Sulfur: the S/N source host at 10000 bar with C=N=0.
* Carbon plus sulfur: the same S/N host with finite C and N=0, varying S.
* Nitrogen: the same S/N host with finite C/S, varying N. Its gas set
  contains N2, NH3 and HCN, and its melt uses the source N2 component law.

``scan_source(network, case, element, total_element_amounts_mol, ...)``
accepts a custom finite sequence in mol of atoms, complete baseline budgets,
and an optional prescribed pressure in bar. The order is retained, including
returns to zero. A changed support removes exactly the components containing
absent elements and their source reactions. Continuation guesses do not add
trace inventories. Every returned point passes fresh local reaction and
atom-balance checks; a failed point raises instead of entering the report.

Amounts and output
------------------

Each point contains the complete accepted local state and a summary:

* Atom amounts in silicate, metal and gas, including exact-zero C/N/S.
* Phase masses in g, gas mass in g and gas mean molar mass in g/mol.
* All source gas mole fractions, including CO, CO2 and CH4, and H2O/H2.
* The prescribed local pressure and the selected source model and case.

Component masses follow their full stoichiometric formulas. Gas mole totals
count molecules, and metal mole totals count atoms. The gas mass excludes
the deep silicate and metal reservoirs. Rescaling every atom budget leaves
local composition and mean molar mass unchanged and scales absolute amounts
and masses by the same factor.

Physical scope and provider handoff
-----------------------------------

These are conditional source-host mechanisms. The Carbon and S/N source
hosts and standards differ: setting S=N=0 in the latter does not recover the
former. Their empirical reaction corrections do not define a common alloy
Gibbs energy. Joint C/S output is not evidence of calibrated cross
interactions. All source temperatures extrapolate the MgO liquid fit.

N uses the pinned molecular-N2 source prescription, not the public
``n2_dasgupta2022`` total-elemental-N mass-fraction law. The source model
omits metal N and nitrides without a quantitative omission bound. It also
omits graphite, carbides and a separate sulfide phase. Gas, silicate and
metal remain present; whole-phase disappearance and competing phase
stability require separate models. The separate :doc:`subneptune_sulfide`
control retains its distinct synthetic SCSS contract.

No MELTS mapping or planetary pressure root is introduced. ExoInventory
owns absolute planetary reservoir selection and atmospheric column closure;
it can consume the accepted local amounts and gas mass at each trial
pressure. A failed property evaluation must not become a pressure bracket
endpoint. MELTS C/S/N and alloy extensions require verified provider
components, standards and partition/saturation data before that separate
physical branch can be coupled.
