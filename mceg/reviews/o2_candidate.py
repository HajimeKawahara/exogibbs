"""Construct evaluated natural-O2 thermals on an explicit retained-fit gauge."""

import argparse
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson

from h2o_candidate import _evaluated, evaluated_fit
from pending_candidate import R, formation_g as evaluate_g, reaction_properties, validate_definition
from tio_candidate import _nasa

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"tests/data/validation/o2_candidate/20261106"
SOURCE=ROOT/"tests/data/validation/o2_source_review/20261106"


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    return json.loads((DATA/"candidate.json").read_text())


def formation_g(temperature, parameters: dict, xp):
    """Preserve the full NASA-proxy residual and replace molecular thermals only."""
    retained=evaluate_g(temperature,parameters["retained"],xp)
    modern=_evaluated(temperature,parameters["evaluated"],xp)
    proxy=_nasa(temperature,parameters["nasa_molecular_thermal_coefficients"],xp)
    return tuple(old+new-ref for old,new,ref in zip(retained,modern,proxy))


def extract(directory: Path) -> None:
    """Use only inspected original numeric tables, fitting Cp rather than G nodes."""
    source=json.loads((SOURCE/"sources.json").read_text())
    records=json.loads((SOURCE/"inputs.json").read_text())
    tables={};hashes={}
    for item in source["modern_downloads"]:
        path=directory/item["file"];hashes[item["file"]]=sha256(path.read_bytes()).hexdigest()
        if hashes[item["file"]]!=item["sha256"]:raise ValueError("Changed original table: "+item["file"])
        with path.open(newline="") as stream:
            tables[item["file"]]=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream) if 1500<=float(row["T/K"])<=4000]
    rows=tables["O2_NaturalThermo.csv"]
    published={"temperature_K":np.array([r["T/K"] for r in rows]),"H_relative_J_mol":np.array([r["H/[J/mol]"] for r in rows]),"S_J_mol_K":np.array([r["S/[J/(K mol)]"] for r in rows]),"Cp_J_mol_K":np.array([r["C_p/[J/(K mol)]"] for r in rows]),"expanded_U_H_J_mol":np.array([r["U-H"] for r in rows]),"expanded_U_S_J_mol_K":np.array([r["U-S"] for r in rows]),"expanded_U_Cp_J_mol_K":np.array([r["U-Cp"] for r in rows])}
    for label,filename in (("pure16","O2_ThermoFunctions.csv"),("partition","O2_PartitionFunctions.csv")):
        assert np.array_equal(published["temperature_K"],[r["T/K"] for r in tables[filename]])
        for key in tables[filename][0]:
            if key!="T/K":published[label+"__"+key]=np.array([r[key] for r in tables[filename]])
    fit=evaluated_fit(published)
    lines=(ROOT/"src/exogibbs/data/FastChem4/logK/logK.dat").read_text().splitlines()
    i=next(i for i,line in enumerate(lines) if line.startswith("O2 "))
    coefficients=list(map(float,lines[i+1].split()))
    jd0=493579.;atct=records['atct'];d0=1000*(2*atct['O_hf0_kJ_mol']-atct['O2_hf0_kJ_mol'])
    assert abs(d0-493686)<1e-8
    retained={"kind":"retained_energy","temperature_range_K":[1500,4000],"coefficients":coefficients,"delta_GH_J_mol":jd0-d0}
    nasa=records['nasa9']['O2'];proxy=list(nasa['intervals'][1]['coefficients']);assert nasa['intervals'][1]['bounds_K']==[1000.,6000.]
    proxy[7]-=(nasa['hf298_J_mol']-nasa['h298_minus_h0_J_mol'])/8.314510
    primary={"kind":"custom","module":"o2_candidate","temperature_range_K":[1500,4000],"parameters":{"retained":retained,"evaluated":fit,"nasa_molecular_thermal_coefficients":proxy}}
    DATA.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA/"published_tables.npz",**published)
    files=[Path(__file__),SOURCE/"inputs.json",SOURCE/"sources.json",SOURCE/"O-029.txt",SOURCE/"O-001.txt",SOURCE/"nasa-selected-records.txt",DATA/"published_tables.npz",ROOT/"mceg/reviews/h2o_candidate.py",ROOT/"mceg/reviews/pending_candidate.py",ROOT/"src/exogibbs/data/FastChem4/logK/logK.dat"]
    record={"id":"o2_natural2025_atct_candidate_20261106","molecule":"O2","species":"O2","baseline_dataset":"curated-ch2-v1","temperature_range_K":[1500,4000],"tracked_elements":["O"],"primary_variant":"evaluated_2025_atct","variants":[{"name":"evaluated_2025_atct","replacements":{"O2":primary}}],"selected_variants":[{"name":"retained_thermal_atct","replacements":{"O2":retained}}],"physical_energy":{"ATcT_version":"1.228","same_version_hf0_kJ_mol":atct,"retained_JANAF_D0_J_mol":jd0,"ATcT_D0_J_mol":d0,"delta_D0_J_mol":d0-jd0,"formal_fit_intercept_Ra_J_mol":R*coefficients[0]},"original_download_sha256":hashes,"source_sha256":{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},"source_doi":"10.1063/5.0273661","thermal_scope":"2025 natural-isotope Cp fit and integrated H/S anchors on1500-4000K; source tables extend to5000K. Statistical, nuclear-spin and natural translation corrections already included; no additional multiplier, symmetry or mass adjustment.","residual_scope":"Retain literal fit residual relative to NASA molecular/atomic thermal proxies plus physical JANAF D0; change only molecular thermal function and physical deltaD0. This differs from the earlier direct-source point diagnostic that omitted this residual.","uncertainty_scope":"Inverse expanded Cp uncertainty is numerical fit weighting, not an independent covariance model. Unbound-state uncertainty and complete reaction covariance remain unquantified.","curated_update":"none"}
    (DATA/"candidate.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


def validate() -> dict:
    """Check held-out source calorics and independent quadrature, then scalar/JAX."""
    model=load_candidate();params=model['variants'][0]['replacements']['O2']['parameters']
    with np.load(DATA/"published_tables.npz") as tb:
        t=tb['temperature_K'];caloric=np.array([[R*g,R*(g-tt*dg),-R*dg,-R*tt*ddg] for tt in t for g,dg,ddg in [_evaluated(tt,params['evaluated'],np)]])
        source=np.stack((tb['H_relative_J_mol']-t*tb['S_J_mol_K'],tb['H_relative_J_mol'],tb['S_J_mol_K'],tb['Cp_J_mol_K']),axis=-1)
        errors=abs(caloric-source);withheld=(t-1500)%10!=0
        quadrature_h=tb['H_relative_J_mol'][0]+cumulative_simpson(tb['Cp_J_mol_K'],x=t,initial=0)
        quadrature_s=tb['S_J_mol_K'][0]+cumulative_simpson(tb['Cp_J_mol_K']/t,x=t,initial=0)
        checks={'source_rows':len(t),'fit_Cp_rows':251,'withheld_Cp_rows':int(withheld.sum()),'maximum_source_errors':dict(zip(('G_J_mol','H_J_mol','S_J_mol_K','Cp_J_mol_K'),map(float,errors.max(axis=0)))),'withheld_Cp_max_error_J_mol_K':float(errors[withheld,3].max()),'maximum_error_over_published_expanded_U':{key:float((errors[:,i]/tb[label]).max()) for key,i,label in [('H',1,'expanded_U_H_J_mol'),('S',2,'expanded_U_S_J_mol_K'),('Cp',3,'expanded_U_Cp_J_mol_K')]},'independent_source_quadrature_H_error_J_mol':float(abs(quadrature_h-tb['H_relative_J_mol']).max()),'independent_source_quadrature_S_error_J_mol_K':float(abs(quadrature_s-tb['S_J_mol_K']).max()),'minimum_fit_Cp_J_mol_K':float(caloric[:,3].min())}
    checks['continuous']=validate_definition(model)
    checks['samples']=[{'temperature_K':t,'primary':reaction_properties(t,model['variants'][0]['replacements']['O2']),'energy_only':reaction_properties(t,model['selected_variants'][0]['replacements']['O2'])} for t in (1500.,2000.,2500.,3000.,4000.)]
    return checks


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract-dir',type=Path);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.extract_dir:extract(args.extract_dir)
    result=validate()
    if args.write:(DATA/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
