#!/usr/bin/env python3
"""Source-only LOO residual calibration for Chinese projected hourly CIF shapes."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v194_china_scenario_shape_transfer as v194
OUT=ROOT/'results/china-projected-cif-v196-loco-calibration-r1'
KS=(10,15)
METHODS=[f'k{k}_{band}' for k in KS for band in ('pooled','hour_of_day')]

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f8',order='C').tobytes()).hexdigest()
def q_higher(a,q):return float(np.quantile(np.asarray(a,dtype=np.float64),q,method='higher'))
def build(donors,target_mu):
    means=np.asarray([d[2] for d in donors]);shapes=np.stack([d[3] for d in donors]);n,h=shapes.shape
    out={}; details={}
    tod=np.arange(h)%24
    for k in KS:
        residuals=np.empty_like(shapes)
        for i in range(n):
            mask=np.arange(n)!=i
            order=np.argsort(np.abs(means[mask]-means[i]),kind='stable')[:k]
            residuals[i]=shapes[i]-shapes[mask][order].mean(axis=0)
        absr=np.abs(residuals)
        qpool=q_higher(absr.ravel(),0.90)
        qhod=np.asarray([q_higher(absr[:,tod==hh].ravel(),0.90) for hh in range(24)])
        target_order=np.argsort(np.abs(means-target_mu),kind='stable')[:k]
        center=target_mu+shapes[target_order].mean(axis=0)
        out[f'k{k}_pooled']=(center-qpool,center+qpool,center)
        qseries=qhod[tod]
        out[f'k{k}_hour_of_day']=(center-qseries,center+qseries,center)
        details[str(k)]={'source_loo_residual_sha256':digest(residuals),'q_pooled':qpool,'q_hour_of_day':qhod.tolist(),'target_center_sha256':digest(center)}
    return out,details

def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit(f'refusing overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    data=v194.read_data();tracks=['same_scenario','leave_scenario_out'];mfile=OUT/'prediction_manifest.jsonl';mh=hashlib.sha256();nq=0
    with mfile.open('w') as f:
        for track in tracks:
            for scenario in v194.SCENARIOS:
                for year in v194.YEARS:
                    vals,bounds=data[(scenario,year)]
                    for ti in range(1,len(v194.EXPECTED_PROVINCES)):
                        province=v194.EXPECTED_PROVINCES[ti]
                        for mi,(lo,hi) in enumerate(bounds):
                            mu=float(vals[lo:hi,ti].mean());donors=v194.make_donors(data,track,scenario,year,ti,bounds,mi,mu)
                            intervals,details=build(donors,mu)
                            row={'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'target_month_mean':mu,'donor_profiles':[{'scenario':d[0],'province':v194.EXPECTED_PROVINCES[d[1]]} for d in donors],'donor_sha256':digest(np.stack([d[3] for d in donors])),'loo_calibration':details,'interval_sha256':{m:{'lower':digest(x[0]),'upper':digest(x[1]),'center':digest(x[2])} for m,x in intervals.items()}}
                            line=json.dumps(row,sort_keys=True,separators=(',',':'))+'\n';f.write(line);mh.update(line.encode());nq+=1
                    print(f'sealed {track}/{scenario}/{year}: {nq}',flush=True)
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v194.DATA.glob('CEF data for Scenario *.xlsx'))}
    seal={'status':'SEALED_BEFORE_SCORING','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'v194_sha256':hashlib.sha256((ROOT/'scripts/v194_china_scenario_shape_transfer.py').read_bytes()).hexdigest(),'v194_protocol_sha256':hashlib.sha256((ROOT/'docs/V194_CHINA_SCENARIO_SHAPE_TRANSFER_PROTOCOL.md').read_bytes()).hexdigest(),'v194_amendment_sha256':hashlib.sha256((ROOT/'docs/V194_DATA_COMPLETENESS_AMENDMENT.md').read_bytes()).hexdigest(),'v195_protocol_sha256':hashlib.sha256((ROOT/'docs/V195_SOURCE_CV_ADAPTIVE_DONORS_PROTOCOL.md').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'docs/V196_SOURCE_LOO_TIME_OF_DAY_CALIBRATION_PROTOCOL.md').read_bytes()).hexdigest(),'input_workbook_sha256':inputs,'k_values':KS,'query_count':nq,'manifest_sha256':mh.hexdigest(),'scope':'modeled China projections; retrospective oracle target monthly mean; not measured-China validation'}
    (OUT/'prediction_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n');print('predictions sealed; scoring starts',flush=True)
    data=v194.read_data();reader=mfile.open();rows=[]
    for track in tracks:
      for scenario in v194.SCENARIOS:
       for year in v194.YEARS:
        vals,bounds=data[(scenario,year)]
        for ti in range(1,len(v194.EXPECTED_PROVINCES)):
         province=v194.EXPECTED_PROVINCES[ti]
         for mi,(lo,hi) in enumerate(bounds):
          mu=float(vals[lo:hi,ti].mean());donors=v194.make_donors(data,track,scenario,year,ti,bounds,mi,mu);intervals,details=build(donors,mu)
          sealed=json.loads(next(reader))
          if sealed['track']!=track or sealed['scenario']!=scenario or sealed['year']!=year or sealed['province']!=province or sealed['month']!=mi+1:raise AssertionError('manifest query order mismatch')
          if sealed['donor_sha256']!=digest(np.stack([d[3] for d in donors])):raise AssertionError('donor mismatch')
          for m,(a,b,c) in intervals.items():
           if sealed['loo_calibration'][str(int(m.split('_')[0][1:]))]['target_center_sha256']!=digest(c):raise AssertionError('center mismatch')
           expect=sealed['interval_sha256'][m]
           if expect!={'lower':digest(a),'upper':digest(b),'center':digest(c)}:raise AssertionError('interval differs from seal')
          y=vals[lo:hi,ti].copy();perm=y[np.random.default_rng(194_700_000+ti*100+mi).permutation(y.size)]
          if not np.isclose(float(perm.mean()),mu,atol=1e-12,rtol=0):raise AssertionError('permuted target mean changed')
          perm_intervals,_=build(donors,mu)
          if any(not all(np.array_equal(intervals[m][j],perm_intervals[m][j]) for j in range(3)) for m in METHODS):raise AssertionError('target-label permutation isolation failed')
          for m,(a,b,c) in intervals.items():
           w=b-a;outside=np.maximum(a-y,0)+np.maximum(y-b,0)
           rows.append({'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'method':m,'k':int(m.split('_')[0][1:]),'level_mae':abs(float(c.mean())-mu),'shape_mae':float(np.mean(np.abs(c-y))),'shape_rmse':float(np.sqrt(np.mean((c-y)**2))),'coverage90':float(np.mean((y>=a)&(y<=b))),'width90':float(np.mean(w)),'winkler90':float(np.mean(w+20*outside))})
        print(f'scored {track}/{scenario}/{year}',flush=True)
    if reader.readline():raise AssertionError('extra manifest rows')
    reader.close();import pandas as pd
    df=pd.DataFrame(rows);df.to_csv(OUT/'monthly_scores.csv',index=False);met=['level_mae','shape_mae','shape_rmse','coverage90','width90','winkler90']
    df.groupby(['track','method'])[met].mean().reset_index().to_csv(OUT/'summary_macro.csv',index=False);df.groupby(['track','scenario','method'])[met].mean().reset_index().to_csv(OUT/'summary_scenario.csv',index=False)
    print('V196 complete',OUT,flush=True);print(pd.read_csv(OUT/'summary_macro.csv').to_string(index=False),flush=True)
if __name__=='__main__':main()
