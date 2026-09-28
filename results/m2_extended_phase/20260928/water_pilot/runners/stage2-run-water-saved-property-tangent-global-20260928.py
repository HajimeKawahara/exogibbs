import datetime,hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/tmp/stage2-water-fixed-pressure-tangent-global-20260928');out.mkdir(exist_ok=False)
root=pathlib.Path('/tmp/stage2-water-global-gibbs-20260928')
def pin():return {key:subprocess.check_output(['git','-C',str(root),*args],text=True).strip() for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain']),('remote',['remote','get-url','origin'])]}
settings={'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
script=pathlib.Path('/tmp/stage2-water-saved-property-tangent-global-20260928.py')
command=['taskset','-c','40','/home/kawahara/anaconda3/envs/myenv39/bin/python',str(script),str(out/'assessment.json'),'20000']
before=pin();start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
with (out/'stdout.log').open('xb') as stream:code=subprocess.run(command,env={**os.environ,**settings},cwd=root,stdout=stream,stderr=subprocess.STDOUT).returncode
receipt={'command':command,'environment_overrides':settings,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-clock,'exit_code':code,'checkout_before':before,'checkout_after':pin(),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
