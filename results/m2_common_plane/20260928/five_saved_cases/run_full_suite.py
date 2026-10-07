import datetime,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/tmp/stage2-common-plane-gibbs-20260928');out=pathlib.Path('/tmp/stage2-common-plane-tests-20260928');out.mkdir(exist_ok=False)
settings={'PYTHONPATH':str(root/'src')+':/tmp/stage2-alloy-insertion-eos-20260927/src','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','JAX_ENABLE_X64':'1','JAX_PLATFORMS':'cpu'}
command=['taskset','-c','40','/home/kawahara/anaconda3/envs/myenv39/bin/python','-m','pytest','-q','-rs','-p','no:cacheprovider','--basetemp','/tmp/stage2-common-plane-test-tmp-20260928','tests/unittests']
before=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
with (out/'stdout.log').open('xb') as stream: code=subprocess.run(command,cwd=root,env={**os.environ,**settings},stdout=stream,stderr=subprocess.STDOUT).returncode
receipt={'command':command,'environment_overrides':settings,'cwd':str(root),'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-clock,'exit_code':code,'commit':before,'commit_after':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'stdout_sha256':hashlib.sha256((out/'stdout.log').read_bytes()).hexdigest()}
(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
