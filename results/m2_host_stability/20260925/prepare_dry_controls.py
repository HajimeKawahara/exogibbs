import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/exoinventory-m2-material-stability-20260925')
INV=ROOT/'exoinventory/examples/subneptune_taxonomy/finite_melt'
sys.path.insert(0,str(INV))
import bse_inventory as b
sys.path.insert(0,str(ROOT/'exoeos/examples'))
import melts_liquid_evaluator as e
source=json.loads((INV/'bse/source.json').read_text())
base=b.load_baseline(); output=ROOT/'stability_probe/dry_controls';output.mkdir(exist_ok=True)
for factor,temp in [(0.8,2173.15),(1.,2173.15),(1.2,2173.15),(1.,1873.15)]:
 modified=copy.deepcopy(source)
 table=modified['oxides_wt_percent'];total=table['MgO']+table['SiO2']; ratio=factor*table['MgO']/table['SiO2']
 table['SiO2']=total/(1+ratio);table['MgO']=total-table['SiO2']
 ledger=b.build_inventory(base,modified)
 oxides=dict(zip(ledger['oxide_order'],ledger['oxide_amounts_mol']))
 n=np.linalg.solve(e.NU.T,[oxides.get(k,0.) for k in b.OXIDES] if False else [next((v for k,v in oxides.items() if k.lower()==o),0.) for o in e.OXIDES])
 assert np.all(n>=0)
 selected=[i for i,x in enumerate(n) if x>0]
 names=[e.COMPONENTS[i]+'_melts' for i in selected]+['H2_dissolved']
 formulas={name:{elem:float(v) for elem,v in zip(e.ELEMENTS,e.FORMULA_MATRIX[i]) if v} for name,i in zip(names,selected)};formulas['H2_dissolved']={'H':2}
 nativeatoms=dict(zip(e.ELEMENTS,e.FORMULA_MATRIX.T@n))
 np.testing.assert_allclose([nativeatoms.get(k,0.) for k in ledger['elements']],ledger['rock_element_amounts_mol'],rtol=1e-14,atol=0)
 mgsi=ledger['rock_element_amounts_mol'][b.ELEMENTS.index('Mg')]/ledger['rock_element_amounts_mol'][b.ELEMENTS.index('Si')]
 native={
 'scope':'Supplied dry-host property control; no H/He added, no source or crystal equilibrium solved. The exported inventory retains its independent H/He entries only as unchanged bookkeeping inputs.',
 'temperature_K':temp,'pressure_bar':1.,'mg_si_factor':factor,'mg_si_atomic_ratio':mgsi,
 'record':{'elements':ledger['elements'],'phases':{'silicate':names},'component_formulas':formulas},
 'source_result':{'accepted':None,'component_amounts_mol':n[selected].tolist()+[0.]},
 'inventory':ledger,'input_parameters':base,'modified_source':modified,
 'control_provenance':{'original_source_sha256':hashlib.sha256((INV/'bse/source.json').read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256((INV/'bse_inventory.py').read_bytes()).hexdigest(),'recipe':'Redistribute the fixed MgO+SiO2 weight total to factor times the baseline atomic Mg/Si; all other oxide weights and inputs fixed. Oxygen follows oxide stoichiometry.','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 key=f'dry_mgsi_{factor:g}_{temp:g}'
 (output/f'{key}.json').write_text(json.dumps(native,indent=2,allow_nan=False)+'\n')
 print(key,mgsi)
