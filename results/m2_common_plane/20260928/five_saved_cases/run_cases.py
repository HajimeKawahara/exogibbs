import datetime,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/tmp/stage2-common-plane-gibbs-20260928')
out=pathlib.Path('/tmp/stage2-common-plane-final-20260928');out.mkdir(exist_ok=False)
aggregate='/tmp/stage2-finite-response-results-20260927/examples/subneptune_taxonomy/finite_melt/validation/20260927_m2_finite_response/final_five_case_aggregate/aggregate.json'
settings={'PYTHONPATH':str(root/'src')+':/tmp/stage2-alloy-insertion-eos-20260927/src','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','JAX_ENABLE_X64':'1','JAX_PLATFORMS':'cpu'}
def check():
 return {key:subprocess.check_output(['git','-C',str(root),*args],text=True).strip() for key,args in [('commit',['rev-parse','HEAD']),('status',['status','--porcelain']),('remote',['remote','get-url','origin'])]}
for case in ['M1','GAS','CLOUD','OXYGEN','OH']:
 target=out/case;target.mkdir()
 command=['taskset','-c','40','/home/kawahara/anaconda3/envs/myenv39/bin/python',str(root/'examples/metal_silicate/run_m2_common_plane.py'),'--evidence-binding',aggregate,'--case',case,'--alloy-bound','/tmp/stage2-final-response-audit-20260927/'+case+'_alloy_insertion_bound.json','--exoeos-checkout','/tmp/stage2-alloy-insertion-eos-20260927','--output',str(target/'result.json')]
 before=check();start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
 with (target/'stdout.log').open('xb') as stream: code=subprocess.run(command,cwd=root,env={**os.environ,**settings},stdout=stream,stderr=subprocess.STDOUT).returncode
 receipt={'command':command,'environment_overrides':settings,'cwd':str(root),'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-clock,'exit_code':code,'checkouts_before':before,'checkouts_after':check(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir()}}
 (target/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(case,code,(target/'stdout.log').read_text(),flush=True)
 if code:break
