"""Re-solve a small ideal assemblage with and without previous-pressure ledgers."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples/metal_silicate"))
from full_potential import ideal_phase
from phase_selection import select_metal_phase
from run_bse_common_gibbs import json_value

phases = {"silicate": ["SiO2_l", "FeO_l", "H2O_l", "H2_l"],
          "metal": ["Fe_m", "Si_m", "O_m", "H_m"],
          "gas": ["H2_g", "He_g", "H2O_g", "SiO_g"]}
formulas = {"SiO2": {"Si": 1, "O": 2}, "FeO": {"Fe": 1, "O": 1},
            "H2O": {"H": 2, "O": 1}, "H2": {"H": 2}, "Fe": {"Fe": 1},
            "Si": {"Si": 1}, "O": {"O": 1}, "H": {"H": 1}, "He": {"He": 1},
            "SiO": {"Si": 1, "O": 1}}
names = [name for group in phases.values() for name in group]
record = {"elements": ["H", "He", "O", "Si", "Fe", "C"], "phases": phases,
          "component_formulas": {name: formulas[name.split("_")[0]] for name in names}, "reactions": []}
matrix = np.array([[record["component_formulas"][name].get(e, 0.) for name in names] for e in record["elements"]])
target = np.array([1., .5, .03, .05, .1, .003, .001, .01, .3, .1, .02, .002])
callbacks, start = {}, 0
for phase, components in phases.items():
    values = target[start:start+len(components)]
    standard = -np.log(values/values.sum())
    callbacks[phase] = ideal_phase(lambda t, p, standard=standard: standard, gas=phase == "gas")
    start += len(components)
rows = []
for scale in (1., 1e24):
    budget = matrix @ target * scale
    def solve(pressure, **kwargs):
        return select_metal_phase(record, budget, 2000., pressure, callbacks, np.zeros(4), np.ones(4),
                                  convex_phase_bounds={phase: 0. for phase in phases}, **kwargs)
    donor = solve(1.)
    cold = solve(1.01)
    warm = solve(1.01, initial_component_amounts_mol=donor.metal_free_result.component_amounts_mol,
                 initial_metal_present_component_amounts_mol=donor.result.component_amounts_mol)
    assert donor.status == cold.status == warm.status == "metal_present"
    assert cold.result.accepted and warm.result.accepted
    normalized_difference = np.max(np.abs((warm.result.component_amounts_mol-cold.result.component_amounts_mol)/scale))
    assert normalized_difference < 1e-7
    rows.append({"amount_scale": scale, "donor": asdict(donor), "cold": asdict(cold), "continuation": asdict(warm),
                 "max_amount_difference_per_scale": normalized_difference,
                 "G_difference_RT_per_total_atom": (warm.result.gibbs_rt-cold.result.gibbs_rt)/budget.sum(),
                 "max_relative_atom_residual": max(np.max(np.abs((matrix @ result.result.component_amounts_mol-budget)[budget>0]/budget[budget>0])) for result in (cold,warm))})
output = {"scope": "Ideal numerical control only; no native/BSE run or speedup claim.",
          "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          "command": [sys.executable, *sys.argv], "record": record, "rows": rows,
          "file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), ROOT/"examples/metal_silicate/phase_selection.py")}}
Path(sys.argv[1]).write_text(json.dumps(json_value(output), indent=2, allow_nan=False)+"\n")
