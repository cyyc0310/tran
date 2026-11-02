#!/usr/bin/env python3
"""Compare time-only and leave-one-BA-out risk calibration for US→GB."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_eia923_pseudotarget_transfer as pseudo
import run_official_intensity_transfer as official
import run_thermal_rate_official_increment as thermal
PROTOCOL=ROOT/'docs/CROSSFIT_DOMAIN_RISK_CALIBRATION_PROTOCOL_20261101.md'
BA=official.US_REGIONS
OUT_DEFAULT=ROOT/'results/crossfit-domain-risk-calibration-r1'
METHODS=('coarse_hgb_M','coarse_hgb_MH','fixed_ef_recomposition')

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def row_weights(frame:pd.DataFrame)->np.ndarray:
    n=frame.groupby('region').region.transform('size').to_numpy(float)
    return 1/(n*frame.region.nunique())
def fit_point(name:str,frame:pd.DataFrame,y:np.ndarray)->object:
    good=np.isfinite(y);x=frame.loc[good].reset_index(drop=True);yy=y[good];w=row_weights(x)
    if name=='coarse_hgb_M':cols=tuple(official.FEATURES)
    else:cols=tuple(official.FEATURES)+('fossil_heat_rate_mmbtu_mwh',)
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10.,random_state=412,early_stopping=False)
    model.fit(x[list(cols)],yy,sample_weight=w*len(w))
    return model

def predict_point(name:str,model:object,frame:pd.DataFrame)->np.ndarray:
    cols=tuple(official.FEATURES) if name=='coarse_hgb_M' else tuple(official.FEATURES)+('fossil_heat_rate_mmbtu_mwh',)
    return np.clip(model.predict(frame[list(cols)]),0,1000)
def fuel_for(frame:pd.DataFrame,raw:dict[str,pd.DataFrame])->np.ndarray:
    parts=[]
    for region,g in frame.groupby('region',sort=False):
        parts.append(raw[region].iloc[g.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
    return np.concatenate(parts)
def fit_recomposition(frame:pd.DataFrame,y:np.ndarray,raw:dict[str,pd.DataFrame])->dict:
    good=np.isfinite(y);q=frame.loc[good].reset_index(drop=True);return official.fit_models(q,fuel_for(q,raw))
def source_response(region:str,year:int)->np.ndarray:
    y=official.source_labels(region,years=(year,));return y

def weighted_q90(errors:pd.DataFrame)->float:
    x=errors.copy();x['weight']=1/x.groupby('region').region.transform('size')/x.region.nunique()
    x=x.sort_values('score');cum=x.weight.cumsum();idx=int(np.searchsorted(cum.to_numpy(),.9,side='left'));return float(x.score.iloc[min(idx,len(x)-1)])
def calibration_stats(parts:dict[str,np.ndarray])->dict:
    clean={r:np.asarray(arr,float)[np.isfinite(arr)] for r,arr in parts.items()}
    if any(len(arr)==0 for arr in clean.values()):raise ValueError('empty calibration region after label filtering')
    bias=float(np.mean([np.mean(v) for v in clean.values()]))
    e=pd.DataFrame([{'region':r,'score':abs(float(vv)-bias)} for r,arr in clean.items() for vv in arr])
    q90_by={r:float(g.score.quantile(.9)) for r,g in e.groupby('region')}
    return {'bias':bias,'pooled_equal_region_q90':weighted_q90(e),'max_region_q90':max(q90_by.values()),'region_q90':q90_by,'calibration_regions':len(clean),'calibration_hours':int(len(e))}
def score_interval(y:np.ndarray,point:np.ndarray,radius:float,clock:pd.Series)->dict:
    keep=np.isfinite(y)&np.isfinite(point);y=y[keep];point=point[keep];e=y-point;times=pd.to_datetime(clock[keep]).reset_index(drop=True)
    tmp=pd.DataFrame({'day':times.dt.strftime('%Y-%m-%d'),'y':y,'p':point})
    c=tmp[['y','p']]-tmp.groupby('day')[['y','p']].transform('mean')
    return {'hours':int(len(y)),'mae':float(np.mean(abs(e))),'shape_mae':float(np.mean(abs(c.y-c.p))),
            'coverage90':float(np.mean(abs(e)<=radius)),'width90':float(2*radius),
            'interval_score90':float(np.mean(2*radius+20*np.maximum(abs(e)-radius,0)))}
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT_DEFAULT);a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    cap=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH);raw,base,labels,audit=official.prepare(ROOT,cap)
    monthly=pseudo.parse_monthly(tuple(range(2020,2025)),BA);arms,_=pseudo.replace_source_features(base,monthly);prepared=arms['eia923_monthly']
    rates=pd.concat([thermal.eia_rates()[0],thermal.gb_rates()],ignore_index=True)
    for region,frame in prepared.items():
        vals=[]
        for year in frame.year:
            hit=rates.loc[rates.region.eq(region)&rates.stat_year.eq(int(year)-2),'fossil_heat_rate_mmbtu_mwh']
            vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
        frame['fossil_heat_rate_mmbtu_mwh']=vals
        if frame.fossil_heat_rate_mmbtu_mwh.isna().any():raise ValueError(f'missing rate for {region}')
    src2022={r:prepared[r].loc[prepared[r].year.eq(2022)].copy() for r in BA}
    src2023={r:prepared[r].loc[prepared[r].year.eq(2023)].copy() for r in BA}
    src_labels={r:{y:official.aligned_labels((src2022 if y==2022 else src2023)[r],official.source_labels(r,years=(y,))) for y in (2022,2023)} for r in BA}
    # Full-source model and conventional time-only residuals.
    full_train=pd.concat([src2022[r] for r in BA],ignore_index=True);full_y=np.concatenate([src_labels[r][2022] for r in BA])
    full_models={name:fit_point(name,full_train,full_y) for name in ('coarse_hgb_M','coarse_hgb_MH')}
    full_proxy=fit_recomposition(full_train,full_y,raw)
    insample={name:{} for name in METHODS};residual_rows=[]
    for r in BA:
        q=src2023[r];yy=src_labels[r][2023]
        for name in ('coarse_hgb_M','coarse_hgb_MH'):
            insample[name][r]=yy-predict_point(name,full_models[name],q)
            residual_rows.extend({'calibration':'time_only','method':name,'region':r,'residual':float(e)} for e in insample[name][r])
        insample['fixed_ef_recomposition'][r]=yy-official.predict_models(full_proxy,q)['coarse_recomposition']
        residual_rows.extend({'calibration':'time_only','method':'fixed_ef_recomposition','region':r,'residual':float(e)} for e in insample['fixed_ef_recomposition'][r])
    # Leave-one-BA-out source pseudo-target residuals; no held-BA 2022 labels enter its fit.
    oof={name:{} for name in METHODS}
    for held in BA:
        fit_regions=[r for r in BA if r!=held]
        train=pd.concat([src2022[r] for r in fit_regions],ignore_index=True);yy=np.concatenate([src_labels[r][2022] for r in fit_regions])
        models={name:fit_point(name,train,yy) for name in ('coarse_hgb_M','coarse_hgb_MH')};proxy=fit_recomposition(train,yy,raw)
        q=src2023[held];truth=src_labels[held][2023]
        for name in ('coarse_hgb_M','coarse_hgb_MH'):
            oof[name][held]=truth-predict_point(name,models[name],q)
            residual_rows.extend({'calibration':'loo_oof','method':name,'region':held,'residual':float(e)} for e in oof[name][held])
        oof['fixed_ef_recomposition'][held]=truth-official.predict_models(proxy,q)['coarse_recomposition']
        residual_rows.extend({'calibration':'loo_oof','method':'fixed_ef_recomposition','region':held,'residual':float(e)} for e in oof['fixed_ef_recomposition'][held])
        print('LOBA calibration model fitted',held,flush=True)
    calibrators={name:{'time_only':calibration_stats(insample[name]),'loo_oof':calibration_stats(oof[name])} for name in METHODS}
    (out/'source_calibration.json').write_text(json.dumps(calibrators,indent=2))
    pd.DataFrame(residual_rows).to_csv(out/'source_residuals.csv.gz',index=False,compression='gzip')
    # Cross-country point forecasts use all source BAs; intervals use source-only calibration alternatives.
    target=prepared['GB'].loc[prepared['GB'].year.eq(2024)].copy().reset_index(drop=True)
    pred={'time_utc':pd.to_datetime(target.time,utc=True).astype(str).to_numpy()}
    for name in ('coarse_hgb_M','coarse_hgb_MH'):pred[name]=predict_point(name,full_models[name],target)
    pred['fixed_ef_recomposition']=official.predict_models(full_proxy,target)['coarse_recomposition']
    pred_path=out/'US_to_GB_predictions.csv.gz';pd.DataFrame(pred).to_csv(pred_path,index=False,compression='gzip')
    seal={'source':'US seven BAs','target':'GB','target_year':2024,'path':pred_path.name,'sha256':sha(pred_path),'hours':len(target)}
    (out/'prediction_seal.json').write_text(json.dumps([seal],indent=2))
    # Freeze and verify all source inputs before the target scoring labels are opened.
    code=[Path(__file__),PROTOCOL,Path(official.__file__),Path(pseudo.__file__),Path(thermal.__file__),ROOT/'scripts/run_cross_country.py',ROOT/'src/transcif/research/cross_country.py',ROOT/'src/transcif/research/capacity_transfer.py']
    data=[ROOT/'data/eia930-hourly-co2-r1/acquisition_manifest.json',ROOT/'data/eia930-hourly-co2-r1/coverage_audit_2022_2024.json',ROOT/'results/thermal-rate-official-increment-r3/annual_fossil_heat_rates.csv']
    data += [ROOT/f'results/research/data/{r}_hourly.csv' for r in BA]
    data += [official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz',thermal.DUKES]
    data += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in BA]
    data += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz']
    data += [ROOT/'docs/evidence/eia923-monthly-pseudotarget-r1/input'/f'f923_{y}.zip' for y in (2020,2021)]
    data += [ROOT/'docs/evidence/eia923-ba-mapping-r1'/f'f923_{y}.zip' for y in (2022,2023,2024)]
    hashes={'code':{str(p):sha(p) for p in dict.fromkeys(code)},'data':{str(p):sha(p) for p in dict.fromkeys(data)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes,indent=2))
    if any(sha(Path(p))!=d for group in hashes.values() for p,d in group.items()):raise RuntimeError('code/input hash changed')
    if sha(pred_path)!=seal['sha256']:raise RuntimeError('target prediction seal mismatch')
    (out/'preseal_target_audit.json').write_text(json.dumps({'target':'GB 2024','previously_viewed':True,'read_before_prediction_seal':False,'this_run_is_development_replay':True},indent=2))
    # Target labels enter only after prediction was written and sealed.
    observed=official.aligned_labels(target,official.source_labels('GB',years=(2024,)))
    target_hash=sha(official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz')
    scored=pd.read_csv(pred_path);records=[]
    for name in METHODS:
        center_cal=calibrators[name]['time_only'];oof_cal=calibrators[name]['loo_oof']
        # same-model time radius; same-model OOF pooled and worst-BA radii; M OOF radius transferred to M+H center
        specs=[('time_only_same_model',center_cal,center_cal['pooled_equal_region_q90']),('loo_oof_pooled_same_model',oof_cal,oof_cal['pooled_equal_region_q90']),('loo_oof_worstBA_same_model',oof_cal,oof_cal['max_region_q90'])]
        if name=='coarse_hgb_MH':specs.append(('loo_oof_pooled_M_radius',oof_cal,calibrators['coarse_hgb_M']['loo_oof']['pooled_equal_region_q90']))
        for variant,bias_cal,radius in specs:
            point=scored[name].to_numpy(float)+bias_cal['bias']
            records.append({'center_model':name,'radius_policy':variant,'bias':bias_cal['bias'],'radius':float(radius),**score_interval(observed,point,float(radius),target.clock)})
    metrics=pd.DataFrame(records);metrics.to_csv(out/'target_metrics.csv',index=False)
    (out/'target_scoring_hashes.json').write_text(json.dumps({'GB_consumption_2024_halfhour.csv.gz':target_hash},indent=2))
    (out/'status.json').write_text(json.dumps({'status':'DEVELOPMENT_SOURCE_CROSSFIT_RISK_CALIBRATION','train_year':2022,'calibration_year':2023,'target_year':2024,'target_country':'GB','target_label_previously_viewed':True,'target_labels_read_after_seal':True,'calibration_domains':len(BA),'loo_oof_models_exclude_held_ba_2022_labels':True,'guarantee':'none; source-domain diagnostic only'},indent=2))
    print(metrics.to_string(index=False))
if __name__=='__main__':main()
