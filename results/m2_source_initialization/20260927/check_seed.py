"""Recheck the archived numerical mapping without a native/source solve."""
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "examples/metal_silicate"))
from m2_expanded_source import conserved_source_seed

folder = Path(__file__).resolve().parent
saved = json.loads((folder / "metal_free_seed_250bar.json").read_text())
reference = json.loads((folder / "mapped_seed_250bar.json").read_text())
assert saved["result"]["accepted"] is True
seed = conserved_source_seed(saved["record"], saved["result"]["component_amounts_mol"],
                             reference["record"], reference["budget_mol"])
np.testing.assert_array_equal(seed, reference["initial_component_amounts_mol"])
print("Saved finite source seed matches the independent mapping exactly.")
