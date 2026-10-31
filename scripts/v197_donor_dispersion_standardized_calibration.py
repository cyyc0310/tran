#!/usr/bin/env python3
"""LOO-calibrated CIF intervals scaled by source/target donor disagreement."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v194_china_scenario_shape_transfer as v194
OUT=ROOT/'results/china-projected-cif-v197-dispersion-calibration-r1'
KS=(10,15);FLOOR=1e-4
METHODS=[f'k{k}_dispersion_scaled' for k in KS]
def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f8',order='C').tobytes()).hexdigest()
def quantile_higher(x,q):return float(np.quantile(np.asarray(x,dtype=float),q,method='higher'))
def calc(donors,target_mu):
    means=np.asarray([d[2] for d in donors]);shapes=np.stack([d[3] for d in donors]);n,h=shapes.shape
    result={};details={}
    for k in KS:
        residuals=np.empty_like(shapes);source_scales=np.empty_like(shapes)
        for i in range(n):
            mask=np.arange(n)!=i
            order=np.argsort(np.abs(means[mask]-means[i]),kind='stable')[:k]
            neighbor=shapes[mask][order]
            residuals[i]=shapes[i]-neighbor.mean(axis=0)
            source_scales[i]=np.maximum(neighbor.std(axis=0,ddof=1),FLOOR)
        std_scores=np.abs(residuals)/source_scales
        q=quantile_higher(std_scores.ravel(),0.90)
        target_order=np.argsort(np.abs(means-target_mu),kind='stable')[:k]
        target_neighbors=shapes[target_order]
        center=target_mu+target_neighbors.mean(axis=0)
        scale=np.maximum(target_neighbors.std(axis=0,ddof=1),FLOOR)
        half=q*scale
        result[f'k{k}_dispersion_scaled']=(center-half,center+half,center,scale)
        details[str(k)]={'q90_standardized':q,'standardized_source_residual_sha256':digest(std_scores),'source_scale_sha256':digest(source_scales),'target_scale_sha256':digest(scale),'center_sha256':digest(center)}
    return result,details

def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit(f'refusing overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=True);data=v194.read_data();tracks=['same_scenario','leave_scenario_out'];manifest=OUT/'prediction_manifest.jsonl';mh=hashlib.sha256();count=0
    with manifest.open('w') as f:
      for track in tracks:
       for scenario in v194.SCENARIOS:
        for year in v194.YEARS:
         vals,bounds=data[(scenario,year)]
         for ti in range(1,len(v194.EXPECTED_PROVINCES)):
          province=v194.EXPECTED_PROVINCES[ti]
          for mi,(lo,hi) in enumerate(bounds):
           mu=float(vals[lo:hi,ti].mean());donors=v194.make_donors(data,track,scenario,year,ti,bounds,mi,mu);intervals,detail=calc(donors,mu)
           row={'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'target_mean':mu,'donor_sha256':digest(np.stack([d[3] for d in donors])),'calibration':detail,'intervals_sha256':{m:{'lower':digest(x[0]),'upper':digest(x[1]),'center':digest(x[2]),'scale':digest(x[3])} for m,x in intervals.items()}}
           line=json.dumps(row,sort_keys=True,separators=(',',':'))+'\n';f.write(line);mh.update(line.encode());count+=1
         print(f'sealed {track}/{scenario}/{year}: {count}',flush=True)
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v194.DATA.glob('CEF data for Scenario *.xlsx'))}
    seal={'status':'SEALED_BEFORE_SCORING','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'v194_runner_sha256':hashlib.sha256((ROOT/'scripts/v194_china_scenario_shape_transfer.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'docs/V197_DONOR_DISPERSION_STANDARDIZED_CALIBRATION_PROTOCOL.md').read_bytes()).hexdigest(),'v194_protocol_sha256':hashlib.sha256((ROOT/'docs/V194_CHINA_SCENARIO_SHAPE_TRANSFER_PROTOCOL.md').read_bytes()).hexdigest(),'v194_amendment_sha256':hashlib.sha256((ROOT/'docs/V194_DATA_COMPLETENESS_AMENDMENT.md').read_bytes()).hexdigest(),'v195_protocol_sha256':hashlib.sha256((ROOT/'docs/V195_SOURCE_CV_ADAPTIVE_DONORS_PROTOCOL.md').read_bytes()).hexdigest(),'v196_protocol_sha256':hashlib.sha256((ROOT/'docs/V196_SOURCE_LOO_TIME_OF_DAY_CALIBRATION_PROTOCOL.md').read_bytes()).hexdigest(),'input_workbooks_sha256':inputs,'k':KS,'floor':FLOOR,'query_count':count,'manifest_sha256':mh.hexdigest(),'scope':'China model-projection world only; retrospective oracle monthly means'}
    (OUT/'prediction_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n');print('sealed before scoring',flush=True)
    data=v194.read_data();reader=manifest.open();scores=[]
    for track in tracks:
     for scenario in v194.SCENARIOS:
      for year in v194.YEARS:
       vals,bounds=data[(scenario,year)]
       for ti in range(1,len(v194.EXPECTED_PROVINCES)):
        province=v194.EXPECTED_PROVINCES[ti]
        for mi,(lo,hi) in enumerate(bounds):
         mu=float(vals[lo:hi,ti].mean());donors=v194.make_donors(data,track,scenario,year,ti,bounds,mi,mu);intervals,detail=calc(donors,mu);sealed=json.loads(next(reader))
         if (sealed['track'],sealed['scenario'],sealed['year'],sealed['province'],sealed['month'])!=(track,scenario,year,province,mi+1):raise AssertionError('manifest order mismatch')
         if sealed['donor_sha256']!=digest(np.stack([d[3] for d in donors])):raise AssertionError('donor hash mismatch')
         for m,x in intervals.items():
          if sealed['intervals_sha256'][m]!={'lower':digest(x[0]),'upper':digest(x[1]),'center':digest(x[2]),'scale':digest(x[3])}:raise AssertionError('prediction seal mismatch')
         y=vals[lo:hi,ti].copy();perm=y[np.random.default_rng(194_700_000+ti*100+mi).permutation(y.size)]
         if not np.isclose(float(perm.mean()),mu,atol=1e-12,rtol=0):raise AssertionError('target mean changed')
         perm_int,_=calc(donors,mu)
         if any(not all(np.array_equal(intervals[m][j],perm_int[m][j]) for j in range(4)) for m in METHODS):raise AssertionError('target shape affected predictions')
         for m,(a,b,c,s) in intervals.items():
          w=b-a;outside=np.maximum(a-y,0)+np.maximum(y-b,0)
          scores.append({'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'method':m,'k':int(m.split('_')[0][1:]),'q90_standardized':detail[str(int(m.split('_')[0][1:]))]['q90_standardized'],'level_mae':abs(float(c.mean())-mu),'shape_mae':float(np.mean(np.abs(c-y))),'shape_rmse':float(np.sqrt(np.mean((c-y)**2))),'coverage90':float(np.mean((y>=a)&(y<=b))),'width90':float(np.mean(w)),'winkler90':float(np.mean(w+20*outside)),'mean_target_scale':float(s.mean())})
       print(f'scored {track}/{scenario}/{year}',flush=True)
    if reader.readline():raise AssertionError('extra manifest rows')
    reader.close();import pandas as pd
    df=pd.DataFrame(scores);df.to_csv(OUT/'monthly_scores.csv',index=False);metrics=['level_mae','shape_mae','shape_rmse','coverage90','width90','winkler90','mean_target_scale']
    df.groupby(['track','method'])[metrics].mean().reset_index().to_csv(OUT/'summary_macro.csv',index=False);df.groupby(['track','scenario','method'])[metrics].mean().reset_index().to_csv(OUT/'summary_scenario.csv',index=False)
    print('V197 complete',OUT,flush=True);print(pd.read_csv(OUT/'summary_macro.csv').to_string(index=False),flush=True)
if __name__=='__main__':main()
