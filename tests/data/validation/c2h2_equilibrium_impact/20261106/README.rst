Acetylene fixed-network candidate substitution
================================================

``ions.npz`` and ``neutral.npz`` hold current, retained-thermal ATcT and
NASA-thermal ATcT endpoints on identical 75-case grids. Current endpoints
are reused; the two candidate grids contribute 300 fresh solves.
``diagnostics.json`` records those 300 executions. ``selected_diagnostics.json``
records 18 tighter and ten interpolation-path solves. ``summary.json`` includes
two independent ion-network conditions with three FastChem models each,
plus every carbon-bearing species' maximum allocation response.
``receipt.json`` pins all solver inputs and execution outputs. This is finite
ideal-gas candidate evidence, not production adoption or a physical error band.
