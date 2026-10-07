"""Compare terminal diagnostics without modifying the frozen provider checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

PROVIDER = Path('/tmp/stage2-gas-only-warmup-gibbs-20260927')
OUTPUT = Path('/tmp/stage2-gas-only-warmup-benchmark')
INPUTS = {
    'janaf': Path('/tmp/stage2-reconnected-janaf250-column.json'),
    'janaf_condensed': Path('/tmp/stage2-reconnected-janaf-condensed250-column.json'),
}

def child(mode, diagnostics, temperature=None):
    start = time.perf_counter()
    sys.path[:0] = [str(PROVIDER / 'src'), str(PROVIDER / 'examples/metal_silicate')]
    import numpy as np
    import m2_atmosphere
    from m2_finite_gas import build_atmosphere_setup, catalog_sha256
    raw = INPUTS[mode].read_bytes()
    saved = json.loads(raw)['source']['source_atmosphere_parcel']
    if temperature is not None:
        saved['T_K'] = float(temperature)
    setup = build_atmosphere_setup(mode)
    assert list(setup.elements) == saved['elements']
    assert list(setup.gas_setup.species) == saved['gas_species']
    assert list(setup.condensate_setup.species) == saved['condensate_species']
    original = m2_atmosphere.CondensateEquilibriumOptions
    m2_atmosphere.CondensateEquilibriumOptions = lambda **kw: original(**{**kw, 'return_diagnostics': diagnostics})
    primitive_result = {}
    original_solve = m2_atmosphere.solve_condensate
    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        primitive_result.update(status=result.status, converged=bool(result.converged),
            acceptance_tier=result.acceptance_tier, selected_route=result.selected_route,
            diagnostics_present=result.diagnostics is not None)
        return result
    m2_atmosphere.solve_condensate = solve
    callback = m2_atmosphere.make_atmosphere_phase(setup, saved['element_gauge_rt'])
    setup_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    parcel = callback.parcel(saved['T_K'], saved['P_bar'], saved['element_amounts_mol'])
    cold_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    energy, gradient = callback.energy_value_and_grad_rt(saved['T_K'], saved['P_bar'], saved['element_amounts_mol'])
    ad_elapsed = time.perf_counter() - start
    payload = dict(schema='m2_gas_only_warmup_removal_benchmark_v1',
        scope='Fresh-process parcel at saved finite surface atom budget and gauge; no source minimization or global closure.',
        requested_temperature_k=temperature,
        gas_model=mode, return_diagnostics=diagnostics,
        provider_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROVIDER,text=True).strip(),
        provider_file_sha256={str(p.relative_to(PROVIDER)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [PROVIDER/'examples/metal_silicate/m2_atmosphere.py', PROVIDER/'src/exogibbs/equilibrium/condensate/lifecycle.py']},
        input_path=str(INPUTS[mode]), input_sha256=hashlib.sha256(raw).hexdigest(),
        catalog_sha256=catalog_sha256(setup),
        provider_status=subprocess.check_output(['git','status','--porcelain'], cwd=PROVIDER,text=True), cpu_affinity=sorted(os.sched_getaffinity(0)),
        setup_seconds=setup_elapsed, cold_parcel_seconds=cold_elapsed,
        primitive_ad_seconds=ad_elapsed, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        primitive_result=primitive_result, parcel=parcel, independent_ad_gibbs_rt=energy,
        independent_ad_elemental_potentials_rt=np.asarray(gradient).tolist())
    OUTPUT.mkdir(exist_ok=True)
    path=OUTPUT/f'{mode}_{str(diagnostics).lower()}.json'
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:payload[k] for k in ['gas_model','return_diagnostics','cold_parcel_seconds','primitive_ad_seconds','peak_rss_kib','primitive_result']}),flush=True)

def driver(temperature=None):
    OUTPUT.mkdir(exist_ok=True)
    records=[]
    for mode in (INPUTS if temperature is None else ['janaf_condensed']):
        for diagnostics in (True,):
            cmd=['taskset','-c','36',sys.executable,__file__,'--child',mode,'--diagnostics',str(diagnostics).lower()]
            if temperature is not None:
                cmd += ['--temperature-k',str(temperature)]
            env=dict(os.environ, JAX_ENABLE_X64='1',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_ENABLE_COMPILATION_CACHE='false')
            started=time.perf_counter()
            observed_rss=0
            reason=None
            with (OUTPUT/f'{mode}_{str(diagnostics).lower()}.log').open('w') as log:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    try:
                        status=Path(f'/proc/{process.pid}/status').read_text()
                        rss=int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))
                        observed_rss=max(observed_rss,rss)
                    except (FileNotFoundError,StopIteration):
                        pass
                    if observed_rss>4*1024*1024:
                        reason='RSS exceeded the 4 GiB benchmark resource cap.'
                    if time.perf_counter()-started>180:
                        reason='Elapsed time exceeded the 180 s single-parcel benchmark cap.'
                    if reason:
                        process.terminate()
                        process.wait(timeout=10)
                        break
                    time.sleep(.2)
            record=dict(gas_model=mode,return_diagnostics=diagnostics,exit_code=process.returncode,
                        wall_seconds=time.perf_counter()-started,observed_peak_rss_kib=observed_rss,interruption_reason=reason)
            records.append(record)
            (OUTPUT/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            if reason:
                break

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--child',choices=list(INPUTS))
    parser.add_argument('--diagnostics',choices=['true','false'])
    parser.add_argument('--temperature-k',type=float)
    args=parser.parse_args()
    if args.temperature_k is not None:
        OUTPUT = Path('/tmp/stage2-gas-only-warmup-benchmark-cloud1000')
    if args.child:
        child(args.child,args.diagnostics=='true',args.temperature_k)
    else:
        driver(args.temperature_k)
