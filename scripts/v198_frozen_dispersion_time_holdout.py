#!/usr/bin/env python3
"""Time-holdout test of frozen V197 donor-dispersion normalized intervals."""
from __future__ import annotations
import calendar,hashlib,json,sys
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import v194_china_scenario_shape_transfer as v194
OUT=ROOT/'results/china-projected-cif-v198-time-holdout-r1'
YEARS=(2055,2060);FLOOR=1e-4;METHODS=['constant_anchor','k15_mean','all_available_mean','k15_law','all_available_law','loo_pooled_k15','v197_scaled_k15']
def digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8',order='C').tobytes()).hexdigest()
def quant_higher(x,q):return float(np.quantile(np.asarray(x,dtype=float),q,method='higher'))
def read_data():
    data={}
    for scenario in v194.SCENARIOS:
        p=v194.DATA/f'CEF data for Scenario {scenario}.xlsx';wb=load_workbook(p,read_only=True,data_only=True)
        for year in YEARS:
            ws=wb[str(year)];it=ws.iter_rows(values_only=True);header=next(it)
            if tuple(header)!=('Time',*v194.EXPECTED_PROVINCES):raise ValueError(f'header {scenario}/{year}')
            raw=list(it);n=8784 if calendar.isleap(year) else 8760
            if len(raw)!=n:raise ValueError(f'row count {scenario}/{year}')
            time_col=np.asarray([r[0] for r in raw],dtype=float)
            if not np.array_equal(time_col,np.arange(1,n+1)):raise ValueError(f'time sequence {scenario}/{year}')
            vals=np.asarray([[np.nan if x is None else float(x) for x in r[1:]] for r in raw],dtype=float)
            # Missing values are retained as NaN; never filled.
            if np.isinf(vals).any():raise ValueError(f'infinite value {scenario}/{year}')
            bounds=[];cur=0
            for month in range(1,13):
                end=cur+calendar.monthrange(year,month)[1]*24;bounds.append((cur,end));cur=end
            data[(scenario,year)]=(vals,bounds)
        wb.close();print('loaded',scenario,flush=True)
    return data

def complete_month(vals,bounds,month_idx,province_idx):
    lo,hi=bounds[month_idx];x=vals[lo:hi,province_idx]
    return np.isfinite(x).all()
def profile(data,scenario,year,p,bounds,mi):
    vals,_=data[(scenario,year)];lo,hi=bounds[mi];x=vals[lo:hi,p]
    if not np.isfinite(x).all():raise ValueError('incomplete donor selected')
    mu=float(x.mean());return mu,x-mu

def get_donors(data,track,scenario,year,target,bounds,mi):
    if track=='same_scenario':
        candidates=[(scenario,p) for p in range(1,len(v194.EXPECTED_PROVINCES)) if p!=target and complete_month(data[(scenario,year)][0],bounds,mi,p)]
        if len(candidates)<16:return None
    else:
        other=[s for s in v194.SCENARIOS if s!=scenario];counts=(8,8,7,7);rng=np.random.default_rng(198000000+v194.SCENARIOS.index(scenario)*1000000+year*100+target*10+mi);candidates=[]
        for donor_s,count in zip(other,counts):
            vals,_=data[(donor_s,year)]
            eligible=[p for p in range(1,len(v194.EXPECTED_PROVINCES)) if p!=target and complete_month(vals,bounds,mi,p)]
            if len(eligible)<count:return None
            chosen=rng.choice(np.asarray(eligible),size=count,replace=False);candidates.extend((donor_s,int(p)) for p in chosen)
    out=[]
    for donor_s,p in candidates:
        mu,shape=profile(data,donor_s,year,p,bounds,mi);out.append((donor_s,p,mu,shape))
    return out

def predict(mu,donors):
    means=np.asarray([d[2] for d in donors]);shapes=np.stack([d[3] for d in donors]);n,h=shapes.shape
    order=np.argsort(np.abs(means-mu),kind='stable');near=shapes[order[:15]]
    center15=mu+near.mean(axis=0);centerall=mu+shapes.mean(axis=0)
    # Source leave-one-donor-out residuals and local dispersions.
    residual=np.empty_like(shapes);src_scale=np.empty_like(shapes)
    for i in range(n):
        mask=np.arange(n)!=i;ord_i=np.argsort(np.abs(means[mask]-means[i]),kind='stable')[:15];neighbors=shapes[mask][ord_i]
        residual[i]=shapes[i]-neighbors.mean(axis=0);src_scale[i]=np.maximum(neighbors.std(axis=0,ddof=1),FLOOR)
    q_pool=quant_higher(np.abs(residual).ravel(),0.90);q_std=quant_higher((np.abs(residual)/src_scale).ravel(),0.90)
    target_scale=np.maximum(near.std(axis=0,ddof=1),FLOOR)
    interval_pooled=(center15-q_pool,center15+q_pool)
    interval_scaled=(center15-q_std*target_scale,center15+q_std*target_scale)
    return {'constant_anchor':np.full(h,mu),'k15_mean':center15,'all_available_mean':centerall,'k15_law':mu+near,'all_available_law':mu+shapes}, {'loo_pooled_k15':interval_pooled,'v197_scaled_k15':interval_scaled}, {'q_pooled':q_pool,'q_standardized':q_std,'target_scale_sha256':digest(target_scale),'source_residual_sha256':digest(residual),'source_scale_sha256':digest(src_scale)}

def crps(particles,y):
    x=np.sort(np.asarray(particles,float),axis=0);n=x.shape[0];first=np.mean(np.abs(x-y),axis=0);w=(2*np.arange(1,n+1)-n-1).reshape(-1,1)
    return float(np.mean(first-np.sum(w*x,axis=0)/(n*n)))
def interval_score(lo,hi,y):
    width=hi-lo;outside=np.maximum(lo-y,0)+np.maximum(y-hi,0)
    return {'coverage90':float(np.mean((y>=lo)&(y<=hi))),'width90':float(np.mean(width)),'winkler90':float(np.mean(width+20*outside))}
def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit(f'refusing overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=True);data=read_data();tracks=('same_scenario','leave_scenario_out');manifest=OUT/'prediction_manifest.jsonl';mh=hashlib.sha256();queries=[];excluded={}
    with manifest.open('w') as f:
      for track in tracks:
       for scenario in v194.SCENARIOS:
        for year in YEARS:
         vals,bounds=data[(scenario,year)]
         for ti in range(1,len(v194.EXPECTED_PROVINCES)):
          province=v194.EXPECTED_PROVINCES[ti]
          for mi,(lo,hi) in enumerate(bounds):
           if not complete_month(vals,bounds,mi,ti):
            excluded['target_month_incomplete']=excluded.get('target_month_incomplete',0)+1;continue
           mu=float(vals[lo:hi,ti].mean());donors=get_donors(data,track,scenario,year,ti,bounds,mi)
           if donors is None:
            excluded['donor_budget_infeasible']=excluded.get('donor_budget_infeasible',0)+1;continue
           pred,ints,diag=predict(mu,donors)
           row={'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'target_mean':mu,'donor_count':len(donors),'donors':[{'scenario':d[0],'province':v194.EXPECTED_PROVINCES[d[1]]} for d in donors],'diagnostics':diag,'point_sha256':{m:digest(pred[m]) for m in ('constant_anchor','k15_mean','all_available_mean')},'law_sha256':{m:digest(pred[m]) for m in ('k15_law','all_available_law')},'interval_sha256':{m:{'lower':digest(z[0]),'upper':digest(z[1])} for m,z in ints.items()}}
           line=json.dumps(row,sort_keys=True,separators=(',',':'))+'\n';f.write(line);mh.update(line.encode());queries.append((track,scenario,year,ti,mi,mu,len(donors))); 
      print('sealed',track,scenario,year,'total',len(queries),flush=True)
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v194.DATA.glob('CEF data for Scenario *.xlsx'))}
    seal={'status':'SEALED_BEFORE_SCORING','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'docs/V198_FROZEN_DISPERSION_METHOD_TIME_HOLDOUT_PROTOCOL.md').read_bytes()).hexdigest(),'v197_script_sha256':hashlib.sha256((ROOT/'scripts/v197_donor_dispersion_standardized_calibration.py').read_bytes()).hexdigest(),'v197_protocol_sha256':hashlib.sha256((ROOT/'docs/V197_DONOR_DISPERSION_STANDARDIZED_CALIBRATION_PROTOCOL.md').read_bytes()).hexdigest(),'v197_results_sha256':hashlib.sha256((ROOT/'docs/V197_DONOR_DISPERSION_CALIBRATION_RESULTS.md').read_bytes()).hexdigest(),'v194_missingness_preflight_sha256':hashlib.sha256((ROOT/'results/china-projected-cif-v194-shape-results-r1/missingness_preflight.json').read_bytes()).hexdigest(),'input_workbooks_sha256':inputs,'years':YEARS,'query_count':len(queries),'excluded':excluded,'manifest_sha256':mh.hexdigest(),'scope':'time holdout within simulated China grid; retrospective oracle month mean; no new-province or real-data claim'}
    (OUT/'prediction_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n');print('sealed before scoring',len(queries),excluded,flush=True)
    # Re-read source workbook data for the independent scoring pass, after full manifest seal.
    data=read_data();reader=manifest.open();rows=[]
    for track,scenario,year,ti,mi,mu,n_donors in queries:
        vals,bounds=data[(scenario,year)];lo,hi=bounds[mi];donors=get_donors(data,track,scenario,year,ti,bounds,mi)
        if donors is None or len(donors)!=n_donors:raise AssertionError('donor set changed')
        pred,ints,diag=predict(mu,donors);sealed=json.loads(next(reader));province=v194.EXPECTED_PROVINCES[ti]
        if (sealed['track'],sealed['scenario'],sealed['year'],sealed['province'],sealed['month'])!=(track,scenario,year,province,mi+1):raise AssertionError('manifest order')
        if sealed['point_sha256']!={m:digest(pred[m]) for m in ('constant_anchor','k15_mean','all_available_mean')}:raise AssertionError('point prediction hash')
        if sealed['law_sha256']!={m:digest(pred[m]) for m in ('k15_law','all_available_law')}:raise AssertionError('law hash')
        if sealed['interval_sha256']!={m:{'lower':digest(z[0]),'upper':digest(z[1])} for m,z in ints.items()}:raise AssertionError('interval hash')
        y=vals[lo:hi,ti].copy();perm=y[np.random.default_rng(198700000+ti*100+mi).permutation(len(y))]
        if not np.isclose(float(perm.mean()),mu,atol=1e-12,rtol=0):raise AssertionError('target label permutation mean')
        pred_perm,ints_perm,_=predict(mu,donors)
        if any(not np.array_equal(pred[m],pred_perm[m]) for m in pred) or any(not all(np.array_equal(ints[m][j],ints_perm[m][j]) for j in range(2)) for m in ints):raise AssertionError('target within-month isolation')
        for method,p in pred.items():
            point=p.mean(axis=0) if p.ndim==2 else p; e={'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'donor_count':len(donors),'method':method,'level_mae':abs(float(point.mean())-mu),'shape_mae':float(np.mean(np.abs(point-y))),'shape_rmse':float(np.sqrt(np.mean((point-y)**2))),'hour_mae':float(np.mean(np.abs(point-y))),'hour_rmse':float(np.sqrt(np.mean((point-y)**2))),'crps':None,'coverage90':None,'width90':None,'winkler90':None}
            if p.ndim==2:e.update(crps=crps(p,y),**interval_score(*np.quantile(p,[.05,.95],axis=0,method='linear'),y))
            rows.append(e)
        for method,(a,b) in ints.items():
            e={'track':track,'scenario':scenario,'year':year,'province':province,'month':mi+1,'donor_count':len(donors),'method':method,'level_mae':abs(float(((a+b)/2).mean())-mu),'shape_mae':float(np.mean(np.abs((a+b)/2-y))),'shape_rmse':float(np.sqrt(np.mean((((a+b)/2)-y)**2))),'hour_mae':float(np.mean(np.abs((a+b)/2-y))),'hour_rmse':float(np.sqrt(np.mean((((a+b)/2)-y)**2))),'crps':None};e.update(interval_score(a,b,y));e['winkler90']=float(np.mean((b-a)+20*(np.maximum(a-y,0)+np.maximum(y-b,0))));rows.append(e)
    if reader.readline():raise AssertionError('extra manifest')
    reader.close();import pandas as pd
    df=pd.DataFrame(rows);df.to_csv(OUT/'monthly_scores.csv',index=False);metrics=['level_mae','shape_mae','shape_rmse','hour_mae','hour_rmse','crps','coverage90','width90','winkler90']
    df.groupby(['track','method'],dropna=False)[metrics].mean().reset_index().to_csv(OUT/'summary_macro.csv',index=False);df.groupby(['track','scenario','year','method'],dropna=False)[metrics].mean().reset_index().to_csv(OUT/'summary_scenario_year.csv',index=False);df.groupby(['track','method']).size().rename('scored_month_queries').reset_index().to_csv(OUT/'score_counts.csv',index=False)
    print('V198 complete',OUT,flush=True);print(pd.read_csv(OUT/'summary_macro.csv').to_string(index=False),flush=True)
if __name__=='__main__':main()
