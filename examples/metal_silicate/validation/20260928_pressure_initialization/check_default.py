"""Compare default selection payloads against the unchanged parent implementation."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PARENT = "5149a7928970a3f3ab664701ac71361f126ed438"
sys.path.insert(0, str(ROOT / "examples/metal_silicate"))
import phase_selection
from run_bse_common_gibbs import json_value

source = subprocess.check_output(["git", "show", PARENT+":examples/metal_silicate/phase_selection.py"], cwd=ROOT)
before = types.ModuleType("phase_selection_before_continuation")
sys.modules[before.__name__] = before
exec(compile(source, PARENT+":phase_selection.py", "exec"), before.__dict__)
spec = importlib.util.spec_from_file_location("ideal_control", ROOT/"tests/unittests/examples/metal_silicate_phase_selection_test.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
rows = []
for offset in (0., 10.):
    record, budget, callbacks, _ = control._ideal_assemblage(metal_offset=offset)
    for pressure in (1., 1.01):
        old, new = [json_value(asdict(module.select_metal_phase(record, budget, 2000., pressure,
                    callbacks, np.zeros(4), np.ones(4), convex_phase_bounds={p: 0. for p in callbacks})))
                    for module in (before, phase_selection)]
        assert old == new
        payload = json.dumps(new, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        rows.append({"metal_offset_rt": offset, "pressure_bar": pressure,
                     "default_payload_exact_equal": True, "status": new["status"],
                     "payload_sha256": hashlib.sha256(payload).hexdigest()})
Path(sys.argv[1]).write_text(json.dumps({"parent_head": PARENT,
    "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "command": [sys.executable, *sys.argv], "old_source_sha256": hashlib.sha256(source).hexdigest(),
    "current_source_sha256": hashlib.sha256(Path(phase_selection.__file__).read_bytes()).hexdigest(),
    "scope": "Exact default payload equality on four ideal controls; no BSE or speedup claim.",
    "rows": rows}, indent=2)+"\n")
