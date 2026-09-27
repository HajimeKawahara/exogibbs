"""Re-solve a preserved unaccepted candidate with fresh provider callbacks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

gibbs = Path('/tmp/stage2-liquid-derivative-gibbs-20260927')
eos = Path('/tmp/stage2-liquid-derivative-eos-20260927')
inventory = Path('/tmp/stage2-finite-closure-h1e24-m1-16/inventory.json')
donor = Path('/tmp/stage2-published-closure-h1e24-janaf-16/trial_0000.json')
output = Path('/tmp/stage2-replayed-source-h1e24-janaf-liquid-audit')
output.mkdir(exist_ok=False)
sys.path[:0] = [str(gibbs/'src'), str(gibbs/'examples/metal_silicate'), str(eos/'src')]
from m2_expanded_source import build_expanded_bse_problem
from run_bse_common_gibbs import build_problem, minimize_gibbs, json_value
from phase_selection import restrict_phase_callbacks

old = json.loads(donor.read_bytes())['failed_state']
record, budget, callbacks, initial, metadata = build_expanded_bse_problem(
    inventory, eos,
    Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),
    '/tmp/exoeos-pr3-env/bin/python', temperature_k=old['temperature_K'],
    pressure_bar=old['pressure_bar'], gas_model='janaf', liquid_model='published')
assert record == old['source_internal_record']
seed = np.asarray(old['metal_selection']['metal_free_result']['component_amounts_mol'])
names = [name for values in record['phases'].values() for name in values]
matrix = np.array([[record['component_formulas'][name].get(e, 0.) for name in names]
                   for e in record['elements']])
np.testing.assert_allclose(matrix @ seed, budget, rtol=1e-12, atol=0)
phases = tuple(name for name in record['phases'] if name != 'metal')
problem = build_problem(record, budget, lambda t, p: np.zeros(len(names)), phases=phases)
report = {
    'scope': 'Fresh metal-free local solve from an unaccepted numerical candidate; no global pressure or metal selection.',
    'temperature_K': old['temperature_K'], 'pressure_bar': old['pressure_bar'],
    'donor': {'path': str(donor), 'sha256': hashlib.sha256(donor.read_bytes()).hexdigest(),
              'accepted': old['metal_selection']['metal_free_result']['accepted']},
    'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
    'checkouts': {name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                  for name, path in [('exogibbs', gibbs), ('exoeos', eos)]},
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'record': record, 'budget_mol': budget.tolist(), 'metadata': metadata,
    'initial_component_amounts_mol': seed.tolist(),
}
(output/'started.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False)+'\n')
start = time.time()
result = minimize_gibbs(problem, old['temperature_K'], old['pressure_bar'], budget,
                        restrict_phase_callbacks(record, problem, callbacks),
                        initial_component_amounts_mol=seed, maxiter=1000)
report.update(result=asdict(result), elapsed_seconds=time.time()-start)
(output/'result.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False)+'\n')
print(json.dumps({'accepted': result.accepted, 'audit_reasons': result.audit_reasons,
                  'elapsed_seconds': report['elapsed_seconds']}), flush=True)
if result.accepted:
    frozen = {'inventory_sha256': report['inventory_sha256'], 'record': record,
              'result': {'accepted': True, 'component_amounts_mol': result.component_amounts_mol},
              'provenance': {'fresh_result_path': str(output/'result.json'),
                             'sha256': hashlib.sha256((output/'result.json').read_bytes()).hexdigest(),
                             'scope': report['scope']}}
    (output/'accepted_metal_free_seed.json').write_text(json.dumps(json_value(frozen), indent=2, allow_nan=False)+'\n')
