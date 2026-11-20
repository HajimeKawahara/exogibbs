"""Keep reviewed numerical presets independent of MCEG tooling."""

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_curated_presets_and_solve_work_with_mceg_imports_blocked():
    script = '''
import importlib.abc
import sys
class BlockMCEG(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mceg" or fullname.startswith("mceg."):
            raise AssertionError("Runtime must not import MCEG")
sys.meta_path.insert(0, BlockMCEG())
from exogibbs.presets import (
    curated, curated_c2h, curated_c2h2, curated_ch2, curated_cn, curated_co2,
    curated_cos, curated_h2o, curated_hcp, curated_ho2, curated_kh, curated_oh,
    curated_ps, curated_tio, curated_tio2, curated_v1, curated_v2,
)
from exogibbs.api.gas import solve
import numpy as np
for preset in (
    curated_v1, curated_c2h, curated_tio, curated_co2, curated_h2o, curated_oh,
    curated_kh, curated_ho2, curated_cos, curated_cn, curated_ps, curated_tio2,
    curated_hcp, curated_c2h2, curated_ch2, curated_v2,
):
    candidate = preset.chemsetup()
    assert np.isfinite(candidate.hvector_func(2500.)).all()
setup = curated.chemsetup()
result, diagnostics = solve(setup, 2500., .01, setup.element_vector_reference,
                            return_diagnostics=True)
assert bool(diagnostics["converged"])
assert np.isclose(np.asarray(result.x).sum(), 1.)
assert "mceg" not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
           "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
    subprocess.run([sys.executable, "-c", script], env=env, check=True,
                   capture_output=True, text=True, timeout=120)
