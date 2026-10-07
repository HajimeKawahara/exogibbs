# Empty-support condensate regression

`validation.json` contains fresh native condensate solves of the six vendored
ExoInventory inputs. The two 2350 K zero-condensate cases now converge; the
other four parcels retain positive condensate amounts. Every report independently
checks element and mass conservation at 1e-9 and gas stationarity, present-phase
equilibrium, and absent-phase stability at 1e-8. These are independent parcels,
not a retained column or a common magma/atmosphere equilibrium.

The gas solver stops on an absolute residual. Its former 1e-10 tolerance left
relative trace-element residuals of 1.25434818e-9 and 9.52147849e-9. The final
empty-support gas solve now uses 1e-14 before the existing full-catalog KKT and
caller-relative acceptance gates. No acceptance tolerance was relaxed.

Run from the repository root to write a new report without changing the saved
results (choose a new output filename if it already exists):

```bash
PYTHONPATH=src:examples/metal_silicate JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import m1_chemistry as m1

fixture = Path("tests/unittests/examples/data/m1_condensate_parcels.json")
inputs = json.loads(fixture.read_text())
setup = m1.build_setups()[1]
record = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "provenance": m1.provenance(),
    "input_fixture": str(fixture),
    "input_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "element_mass_tolerance": 1e-9,
    "chemical_tolerance": 1e-8,
    "parcels": [],
}
for case in inputs["cases"]:
    report = m1.solve_parcel(
        setup, case["temperature_k"], case["pressure_bar"],
        np.asarray(case["element_amounts_mol"]),
    )
    assert report["accepted"], case["id"]
    assert bool(np.any(np.asarray(report["condensate_amounts_mol"]) > 0)) == case["condensate_expected"]
    record["parcels"].append({"id": case["id"], "report": report})
with Path("/tmp/m1-zero-condensate-validation.json").open("x") as output:
    json.dump(record, output, indent=2, allow_nan=False)
    output.write("\n")
PY
```

Regression command: `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu pytest tests/unittests/examples/m1_chemistry_test.py -q`.
