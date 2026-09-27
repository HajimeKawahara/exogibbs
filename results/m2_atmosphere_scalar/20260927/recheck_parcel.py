"""Freshly recheck the primitive parcel scalar; no native/source/planet solve."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "examples/metal_silicate"))
from m2_finite_gas import build_atmosphere_setup, atmosphere_gauge_rt
from m2_atmosphere import make_atmosphere_phase
from run_bse_common_gibbs import source_standards_rt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
source = json.loads(Path(__file__).with_name("input.json").read_text())
t, p = source["temperature_K"], source["pressure_bar"]
setup = build_atmosphere_setup("janaf")
assert list(setup.elements) == source["elements"]
reference, _ = source_standards_rt(t, p)
phase = make_atmosphere_phase(setup, atmosphere_gauge_rt(setup, t, reference))
b = np.asarray(source["element_amounts_mol"])
parcel = phase.parcel(t, p, b)
energy, gradient = phase.energy_value_and_grad_rt(t, p, b)
state = phase(t, p, b)
error = float(np.max(np.abs(gradient - state.mu_rt)))
assert parcel["accepted"] and error < 5e-6
assert abs(energy-state.gibbs_rt) < 5e-9 * max(abs(energy), 1.)
args.output.write_text(json.dumps({"scope": __doc__, "parcel": parcel,
    "independent_energy_rt": energy, "independent_gradient_rt": gradient.tolist(),
    "maximum_gradient_error_rt": error}, indent=2)+"\n")
print("Independent primitive gradient maximum error / RT:", error)
