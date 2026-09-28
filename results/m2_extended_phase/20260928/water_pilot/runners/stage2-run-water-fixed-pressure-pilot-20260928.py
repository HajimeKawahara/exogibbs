import datetime,hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/tmp/stage2-water-fixed-pressure-pilot-20260928');out.mkdir(exist_ok=False)
gibbs=pathlib.Path('/tmp/stage2-material-proof-gibbs-20260928');eos=pathlib.Path('/tmp/stage2-combined-eos-20260928')
def pins():
 return {str(p):{key:subprocess.check_output(['git','-C',str(p),*args],text=True).strip() for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain']),('remote',['remote','get-url','origin'])]} for p in (gibbs,eos)}
settings={'PYTHONPATH':str(gibbs/'src')+':'+str(eos/'src'),'PYTHONDONTWRITEBYTECODE':'1','JAX_ENABLE_X64':'1','JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
script=pathlib.Path('/tmp/stage2-water-fixed-pressure-pilot-20260928.py')
command=['taskset','-c','40','/home/kawahara/anaconda3/envs/myenv39/bin/python',str(script),str(out/'results'),'1']
before=pins();start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
with (out/'stdout.log').open('xb') as stream:code=subprocess.run(command,env={**os.environ,**settings},cwd=gibbs,stdout=stream,stderr=subprocess.STDOUT).returncode
receipt={'command':command,'environment_overrides':settings,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'elapsed_seconds':time.monotonic()-clock,'checkouts_before':before,'checkouts_after':pins(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}}
(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
