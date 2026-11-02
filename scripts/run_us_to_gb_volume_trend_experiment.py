from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
sys.path.insert(0, 'scripts')
import run_us_to_gb_2025_official_generation as base

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--target-labels',type=Path,default=ROOT/'docs/evidence/neso-national-intensity-2025-r2/neso_national_intensity_history.csv')
args=parser.parse_args()
OUT=args.output.resolve(); OUT.mkdir(parents=True, exist_ok=False)
TARGET_LABELS=args.target_labels.resolve()
EXPECTED_TARGET_LABEL_SHA='c6bb4e11b293457232b757b5e149db0a45f731bd1d6db27a7747baea10e42f14'
PROTOCOL=ROOT/'docs/CROSS_COUNTRY_US_TO_GB_VOLUME_TREND_PROTOCOL_20261102.md'
TREND=('volume_log_change2',)
METHODS=('anchor700','source_thermal_anchor','source_fuel_recomposition','mix_hgb','mix_hgb_volume')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def add_volume_history(raw, feat):
    p=raw.clock.dt.to_period('M')
    x=raw[list(base.FUELS)].copy(); x['period']=p.to_numpy()
    sums=x.groupby('period')[list(base.FUELS)].sum()
    counts=x.groupby('period').size()
    exp=raw.assign(period=p.to_numpy()).groupby('period').expected_hours.first()
    valid=sums.loc[(counts/exp).ge(.99)].sum(axis=1)
    logs=np.log(valid.where(valid.gt(0)))
    vals={}
    for period in feat.period.unique():
        hist=logs.reindex(pd.period_range(period-3,period-2,freq='M'))
        if hist.notna().all(): vals[period]=float(hist.iloc[-1]-hist.iloc[0])
    out=feat.copy(); out[TREND[0]]=out.period.map(vals)
    return out

def load_feature_raw(ba):
    path=base.SOURCE_DIR/'feature_inputs'/f'{ba}_generation_2022_2024.csv.gz'
    frame=pd.read_csv(path)
    frame['time']=pd.to_datetime(frame.time_utc,utc=True)
    frame['clock']=frame.time.dt.tz_convert(base.ZONES[ba]).dt.tz_localize(None)
    frame['expected_hours']=frame.clock.map(lambda t: ((t.to_period('M')+1).start_time.tz_localize(base.ZONES[ba])-t.to_period('M').start_time.tz_localize(base.ZONES[ba])).total_seconds()/3600)
    frame['region']=ba; frame['country']='US'
    for fuel in base.FUELS: frame[fuel]=frame[f'generation_{fuel}'].astype(float)
    return frame, path

def data_for(raw, labels):
    f=add_volume_history(raw,base.monthly_features(raw)); f['raw_row']=f.index
    y=labels.set_index('time').y.reindex(f.time).to_numpy(float)
    f['source_label']=y
    usable=f.dropna(subset=[*base.COARSE,*TREND,'source_label']).copy()
    y=usable.source_label.to_numpy(float)
    x, cols=base.features(usable)
    for n in TREND: x[n]=usable[n].to_numpy(float)
    return usable,x,cols,y

def weighted_q90(resids):
    vals=[]; weights=[]
    for arr in resids.values():
        a=np.asarray(arr,float); vals.extend(a.tolist()); weights.extend([1/len(resids)/len(a)]*len(a))
    order=np.argsort(vals); c=np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(vals)[order[np.searchsorted(c,.9,side='left')]])

_, source_labels, source_hashes=base.load_sources()
source_raw={}; feature_hashes={}
for ba in base.BAS:
    source_raw[ba], feature_path=load_feature_raw(ba)
    feature_hashes[str(feature_path.resolve())]=sha(feature_path)
source_hashes.update(feature_hashes)
train_y=[]; train_x0=[]; train_x1=[]; train_ba=[]; entries={}
for ba in base.BAS:
    f,x,cols,y=data_for(source_raw[ba],source_labels[ba])
    entries[ba]=(f,x,cols,y)
    m=f.clock.dt.year.le(2023).to_numpy()
    train_y.append(y[m]); train_x0.append(x.loc[m,list(cols)]); train_x1.append(x.loc[m,[*cols,*TREND]]); train_ba.extend([ba]*int(m.sum()))

x0=pd.concat(train_x0,ignore_index=True); x1=pd.concat(train_x1,ignore_index=True); ytr=np.concatenate(train_y)
ba_arr=np.asarray(train_ba); counts=pd.Series(ba_arr).value_counts().to_dict(); w=np.array([len(ytr)/len(base.BAS)/counts[b] for b in ba_arr])
opts=dict(max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10.,random_state=412,early_stopping=False)
mix=HistGradientBoostingRegressor(**opts).fit(x0,ytr,sample_weight=w)
mixv=HistGradientBoostingRegressor(**opts).fit(x1,ytr,sample_weight=w)

# Source-only thermal anchor and fuel-response recomposition.
thermal_num=thermal_den=0.; coarse_rows=[]; y_rows=[]; profile_parts=[]
for ba in base.BAS:
    f,_,_,y=entries[ba]; m=f.clock.dt.year.le(2023).to_numpy(); train=f.loc[m]
    raw=source_raw[ba].loc[train.index]
    fuel=raw[list(base.FUELS)].to_numpy(float); thermal=fuel[:,:4].sum(axis=1); total=fuel.sum(axis=1)
    w_ba=np.full(int(m.sum()),1./int(m.sum()))
    thermal_num += float(np.sum(w_ba*y[m]*total)); thermal_den += float(np.sum(w_ba*thermal))
    coarse=np.column_stack([thermal,fuel[:,5],fuel[:,4],fuel[:,6],fuel[:,7]])
    coarse_rows.append(coarse); y_rows.append(y[m])
    tmp=pd.DataFrame(coarse,columns=base.COARSE); tmp['hour']=train.clock.dt.hour.to_numpy()
    profile=tmp.groupby('hour')[list(base.COARSE)].mean().reindex(range(24)).fillna(0.)
    profile_parts.append(profile.div(profile.mean().replace(0.,np.nan)).fillna(1.).to_numpy())
thermal_ef=thermal_num/thermal_den
coarse_train=np.concatenate(coarse_rows); y_coarse=np.concatenate(y_rows)
fr=LinearRegression(positive=True,fit_intercept=False).fit(coarse_train,y_coarse,sample_weight=w)
profile=np.mean(profile_parts,axis=0)

def predict(f,x,model_name):
    if model_name=='anchor700': return 700*f.thermal.to_numpy(float)
    if model_name=='source_thermal_anchor': return thermal_ef*f.thermal.to_numpy(float)
    if model_name=='mix_hgb': return mix.predict(x[list(cols)])
    if model_name=='mix_hgb_volume': return mixv.predict(x[[*cols,*TREND]])
    q=f[list(base.COARSE)].to_numpy(float)*profile[f.clock.dt.hour.to_numpy()]
    q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
    return fr.predict(q)

cal_res={m:{} for m in METHODS}; cal_bias={}
for ba in base.BAS:
    f,x,cols_ba,y=entries[ba]; mask=f.clock.dt.year.eq(2024).to_numpy(); cf=f.loc[mask]; cx=x.loc[mask]
    for method in METHODS: cal_res[method][ba]=y[mask]-predict(cf,cx,method)
for method in METHODS:
    cal_bias[method]=float(np.mean([a.mean() for a in cal_res[method].values()]))
    cal_res[method]={ba:np.abs(a-cal_bias[method]) for ba,a in cal_res[method].items()}
radii={m:weighted_q90(cal_res[m]) for m in METHODS}

# Make prediction and seal before reading NESO 2025 response labels.
gb_raw,gb_audit=base.load_gb_inputs()
gf=add_volume_history(gb_raw,base.monthly_features(gb_raw)); gf['raw_row']=gf.index
gf=gf.loc[gf.clock.dt.year.eq(2025)].dropna(subset=[*base.COARSE,*TREND]).copy()
gx,gb_cols=base.features(gf)
for n in TREND: gx[n]=gf[n].to_numpy(float)
if len(gf)!=8760: raise ValueError(f'expected 8760 target rows with valid trend, got {len(gf)}')
preds={m:predict(gf,gx,m) for m in METHODS}
pred=pd.DataFrame({'time':gf.time,'clock':gf.clock,'observed_month':gf.observed_month,
                   'volume_log_change2':gf.volume_log_change2,
                   **{m:np.clip(preds[m],0,None) for m in METHODS}})
pred_path=OUT/'sealed_predictions.csv.gz'; pred.to_csv(pred_path,index=False,compression='gzip')
cal_path=OUT/'source_calibration.json'; cal_path.write_text(json.dumps({
    'bias':cal_bias,'radius90':radii,'calibration_year':2024,'source_ba_count':len(base.BAS),
    'calibration_rows_by_ba':{ba:int(entries[ba][0].clock.dt.year.eq(2024).sum()) for ba in base.BAS},
    'training_rows_by_ba':{ba:int(entries[ba][0].clock.dt.year.le(2023).sum()) for ba in base.BAS},
    'weighting':'equal total training weight per BA; equal BA-weighted calibration bias and pooled absolute residual quantile'
},indent=2)+'\n')
seal={'prediction_sha256':sha(pred_path),'calibration_sha256':sha(cal_path),'prediction_rows':len(pred),
      'target_labels_read_before_seal':False,'target_budget':'t-2 CN5 shares plus one scale-free two-month log-volume change feature',
      'trend_formula':'log total generation month p-2 minus log total generation month p-3; source monthly features computed from complete generation-only files before joining hourly labels',
      'source_input_hashes':source_hashes,'target_fuel_sha256':sha(base.TARGET_FUELS),
      'experiment_runner_sha256':sha(Path(__file__)),
      'protocol_runner_sha256':{str(p):sha(p) for p in [Path(base.__file__),base.PROTOCOL,PROTOCOL]}}
(OUT/'prediction_seal.json').write_text(json.dumps(seal,indent=2)+'\n')

# Open official target response only after forecast and calibration are sealed.
if sha(TARGET_LABELS)!=EXPECTED_TARGET_LABEL_SHA:
    raise ValueError('NESO 2025 label snapshot differs from the frozen official-generation run')
labels=pd.read_csv(TARGET_LABELS,parse_dates=['from']); labels['time']=pd.to_datetime(labels['from'],utc=True).dt.floor('h'); labels['y']=pd.to_numeric(labels.actual_gco2_per_kwh,errors='coerce')
labels=labels.groupby('time').y.agg(['mean','count']).reset_index(); labels=labels.loc[labels['count'].eq(2)&labels['mean'].notna(),['time','mean']].rename(columns={'mean':'y'})
joined=pred.merge(labels,on='time',how='inner',validate='one_to_one')
rows=[]; monthly=[]
for method in METHODS:
    point=joined[method].to_numpy(float)+cal_bias[method]; y=joined.y.to_numpy(float); e=y-point
    daily=pd.DataFrame({'y':y,'p':point,'day':joined.clock.dt.strftime('%Y-%m-%d').to_numpy()})
    centered=daily[['y','p']]-daily.groupby('day')[['y','p']].transform('mean')
    radius=radii[method]
    rows.append({'method':method,'hours':len(joined),'mae':float(np.mean(np.abs(e))),
                 'shape_mae':float(np.mean(np.abs(centered.y-centered.p))),
                 'bias':float(e.mean()),'coverage90':float(np.mean(np.abs(e)<=radius)),
                 'width90':2*radius,'winkler90':float(np.mean(2*radius+20*np.maximum(np.abs(e)-radius,0)))})
    for mo,part in joined.groupby(joined.clock.dt.month):
        pe=part.y.to_numpy(float)-(part[method].to_numpy(float)+cal_bias[method])
        monthly.append({'method':method,'month':int(mo),'hours':len(part),'mae':float(np.mean(np.abs(pe))),
                        'coverage90':float(np.mean(np.abs(pe)<=radius)),'width90':2*radius,
                        'winkler90':float(np.mean(2*radius+20*np.maximum(np.abs(pe)-radius,0)))})
pd.DataFrame(rows).to_csv(OUT/'overall_metrics.csv',index=False)
pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
(OUT/'scoring_manifest.json').write_text(json.dumps({
    'target_label_path':str(TARGET_LABELS),'target_label_sha256':sha(TARGET_LABELS),
    'expected_frozen_target_label_sha256':EXPECTED_TARGET_LABEL_SHA,
    'prediction_sha256_verified':sha(pred_path),'scored_hours':len(joined),
    'target_labels_read_after_prediction_seal':True
},indent=2)+'\n')
(OUT/'audit.json').write_text(json.dumps({'status':'DEVELOPMENT_METHOD_REPLICATION_OFFICIAL_GENERATION',
    'direction':'US_EIA_to_GB_NESO','source_train_years':[2022,2023],'source_calibration_year':2024,
    'target_year':2025,'target_scored_hours':len(joined),'target_labels_read_after_prediction_seal':True,
    'target_input_vintage':'NESO current revised series; strict t-2 backcast, not as-issued',
    'target_estimand':'NESO intensity.actual generation estimate','target_label_sha256':sha(TARGET_LABELS),
    'experiment_runner_sha256':sha(Path(__file__)),
    'source_training_rows_by_ba':{ba:int(entries[ba][0].clock.dt.year.le(2023).sum()) for ba in base.BAS},
    'source_calibration_rows_by_ba':{ba:int(entries[ba][0].clock.dt.year.eq(2024).sum()) for ba in base.BAS},
    'target_methods_previously_viewed':True,'no_hyperparameter_selection':True},indent=2)+'\n')
print(pd.DataFrame(rows).round(4).to_string(index=False))
print('saved',OUT)
