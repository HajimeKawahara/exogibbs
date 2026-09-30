import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/tmp/stage2-extended-common-plane-gibbs-20260928')
output=Path('/tmp/stage2-extended-proof-suite-20260928')
eos=Path('/tmp/stage2-material-proof-eos-20260928')
manifest=output/'manifest.json'
def snapshot(path):
    return {'head':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            'status':subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()}
for index in map(int,sys.argv[1:]):
    command=['taskset','-c','44',sys.executable,'/tmp/stage2-run-test-shard.py',str(manifest),str(index),
             '--repo',str(root),'--prefix',str(output/'shard')]
    target=output/f'shard-{index:02d}-execution.json'
    assert not target.exists()
    before={'gibbs':snapshot(root),'eos':snapshot(eos)}
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    with (output/f'shard-{index:02d}-stdout.log').open('x') as stream:
        result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    receipt={'command':command,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'elapsed_seconds':time.monotonic()-clock,'exit_code':result.returncode,'checkouts_before':before,
             'checkouts_after':{'gibbs':snapshot(root),'eos':snapshot(eos)},
             'environment':{key:os.environ.get(key) for key in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','JAX_ENABLE_X64','JAX_PLATFORMS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
             'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(index,result.returncode,receipt['elapsed_seconds'],flush=True)
    if result.returncode:raise SystemExit(result.returncode)
