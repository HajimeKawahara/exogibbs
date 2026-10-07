import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('index',type=int);p.add_argument('--repo',required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
os.chdir(a.repo);sys.path.insert(0,a.repo)
m=json.loads(Path(a.manifest).read_text());row=m['shards'][a.index]
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==m['source_head']
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
import pytest
class Recorder:
 def __init__(self):self.nodes=[];self.reports=[]
 def pytest_collection_modifyitems(self,session,config,items):
  self.nodes=[x.nodeid for x in items]
  assert len(self.nodes)==len(set(self.nodes)) and set(self.nodes)==set(row['nodeids'])
 def pytest_runtest_logreport(self,report):
  self.reports.append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome,'duration':report.duration,'longrepr':str(report.longrepr) if report.failed or report.skipped else None})
r=Recorder();start=time.time();prefix=a.prefix+f'-{a.index:02d}'
exitcode=pytest.main(['-q','-rs','--basetemp='+prefix+'-tmp',*row['files']],plugins=[r])
Path(prefix+'-receipt.json').write_text(json.dumps({'source_head':m['source_head'],'manifest':str(Path(a.manifest).resolve()),'index':a.index,'collected_nodeids':r.nodes,'reports':r.reports,'exit_code':int(exitcode),'wall_seconds':time.time()-start},indent=2)+'\n')
sys.exit(exitcode)
