"""Source-held-out probe of as-of monthly mix volatility and hourly tail risk."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/disjoint-calibration-v50'
PARENT=ROOT/'results/disjoint-calibration-v50-full-mix-probe'
OUT=ROOT/'results/disjoint-calibration-v50-mix-volatility-probe'
PROTOCOL=ROOT/'docs/V50_MONTHLY_MIX_VOLATILITY_PROTOCOL.md'
SCRIPT=ROOT/'scripts/probe_v50_monthly_mix_volatility.py'
FUELS=('coal','gas','oil','nuclear','hydro','wind','solar','other')
EF=np.array([1000.,400.,800.,0.,0.,0.,0.,400.])
sys.path.insert(0,str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402


def sha(path:Path)->str:
    return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def read_json(path:Path)->dict:
    return json.loads(path.read_text())


def source_features(monthly_paths:list[Path],sources:list[str])->dict[str,dict]:
    records={r:[] for r in sources}
    for path in monthly_paths:
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            required={'region','period_start','period_end','available_at',*('fuelshare_'+f for f in FUELS)}
            if not required<=set(reader.fieldnames or []): raise ValueError('Missing monthly field')
            for row in reader:
                if row['region'] not in records: continue
                start,end,release=(pd.Timestamp(row[k]) for k in ('period_start','period_end','available_at'))
                if any(x.tzinfo is None for x in (start,end,release)): raise ValueError('Naive monthly timestamp')
                start,end,release=(x.tz_convert('UTC') for x in (start,end,release))
                if (start!=start.normalize() or start.day!=1 or end!=start+pd.offsets.MonthBegin(1)
                        or release<end): raise ValueError('Non-calendar/nonreleased month')
                shares=np.array([float(row['fuelshare_'+f]) for f in FUELS])
                if not np.isfinite(shares).all() or (shares<0).any() or not np.isclose(shares.sum(),1.,atol=1e-8,rtol=0):
                    raise ValueError('Fuel shares are not a finite simplex')
                records[row['region']].append((start,end,release,float(shares@EF)))
    out={}
    for source,rows in records.items():
        rows.sort(key=lambda x:x[0])
        if len({x[0] for x in rows})!=len(rows): raise ValueError('Duplicate source-month key')
        level=[];volatility=[];context_count=0
        for day in helper.CAL_DAYS:
            legal=[r for r in rows if r[0]>=pd.Timestamp('2022-01-01',tz='UTC') and r[0]<day and r[1]<=day and r[2]<=day]
            latest=legal[-3:]
            if len(latest)!=3: raise ValueError('Three released monthly contexts required')
            values=np.array([r[3] for r in latest],dtype=float)
            level.append(float(values.mean()))
            volatility.append(float(np.abs(np.diff(values)).mean()))
            context_count+=3
        out[source]={'level':float(np.mean(level)),'volatility':float(np.mean(volatility)),
                     'query_dates':len(helper.CAL_DAYS),'monthly_contexts':context_count}
    return out


def loo(features:np.ndarray,y:np.ndarray,sources:list[str],model:str)->list[dict]:
    rows=[]
    columns={'null':[],'level':[0],'volatility':[1],'level_plus_volatility':[0,1]}[model]
    for i,source in enumerate(sources):
        train=np.arange(len(sources))!=i
        if not columns:
            prediction=float(y[train].mean());beta=[]
        else:
            xtrain=features[train][:,columns];xtest=features[[i]][:,columns]
            center=xtrain.mean(axis=0);scale=xtrain.std(axis=0,ddof=0)
            nonzero=scale>1e-12
            xtrain=xtrain[:,nonzero];xtest=xtest[:,nonzero];center=center[nonzero];scale=scale[nonzero]
            if xtrain.shape[1]:
                ztrain=(xtrain-center)/scale;ztest=(xtest-center)/scale
                design=np.column_stack([np.ones(len(ztrain)),ztrain])
                coef=np.linalg.lstsq(design,y[train],rcond=None)[0]
                prediction=float(np.r_[1.,ztest.ravel()]@coef);beta=coef[1:].tolist()
            else:
                prediction=float(y[train].mean());beta=[]
        baseline=float(y[train].mean())
        rows.append({'held_source':source,'model':model,'observed':float(y[i]),
            'prediction':prediction,'training_mean':baseline,'absolute_error':abs(float(y[i])-prediction),
            'null_absolute_error':abs(float(y[i])-baseline),
            'deviation_sign_correct':int(np.sign(y[i]-baseline)==np.sign(prediction-baseline)),
            'standardized_coefficients':json.dumps(beta,separators=(',',':'))})
    return rows


def summarize(pred:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (outcome,model),group in pred.groupby(['outcome','model'],sort=True):
        errors=group.absolute_error.to_numpy(float);null=group.null_absolute_error.to_numpy(float)
        rows.append({'outcome':outcome,'model':model,'n_sources':len(group),
            'LOO_MAE':float(errors.mean()),'LOO_RMSE':float(np.sqrt(np.mean(errors**2))),
            'null_MAE':float(null.mean()),'null_RMSE':float(np.sqrt(np.mean(null**2))),
            'deviation_sign_accuracy':float(group.deviation_sign_correct.mean())})
    return pd.DataFrame(rows)


def contract()->dict:
    return {'version':1,'sources':list(helper.EXTERNAL),'query_dates':len(helper.CAL_DAYS),
        'models':['null','level','volatility','level_plus_volatility'],'outcomes':['p90','p99'],
        'source_level_units':True,'target_or_unseen_opened':False,'production_refits':0,
        'forecast_gain':False,'algorithm_novelty':False}


def prepare()->None:
    if OUT.exists(): raise FileExistsError('Preserve existing probe output')
    freeze=read_json(RUN/'freeze.json');parent=read_json(PARENT/'summary.json')
    if parent.get('status')!='COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE': raise ValueError('Complete full-mix parent required')
    monthly=[Path(p) for p in freeze['external_monthly']]
    for path in monthly:
        if sha(path)!=freeze['input_hashes'].get(str(path)): raise ValueError('V50 monthly source hash changed')
    paths=[RUN/'freeze.json',PARENT/'summary.json',PARENT/'source_feature.csv',PARENT/'source_heldout_predictions.csv',
           *monthly,PROTOCOL,SCRIPT]
    inputs={str(p):sha(p) for p in paths}
    if sha(PARENT/'source_feature.csv')!=parent.get('feature_csv_sha256') or sha(PARENT/'source_heldout_predictions.csv')!=parent.get('heldout_csv_sha256'):
        raise ValueError('Parent source outcome/feature artifact changed')
    OUT.mkdir(parents=True)
    frozen={**contract(),'parent_summary_sha256':sha(PARENT/'summary.json'),
        'parent_feature_sha256':sha(PARENT/'source_feature.csv'),
        'parent_predictions_sha256':sha(PARENT/'source_heldout_predictions.csv'),
        'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(SCRIPT),
        'monthly_input_hashes':{str(p):sha(p) for p in monthly},'input_hashes':inputs}
    (OUT/'freeze.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')


def run()->None:
    if any((OUT/n).exists() for n in ('complete.json','source_features.csv','heldout_predictions.csv','summary.csv')):
        raise FileExistsError('Preserve partial or completed probe outputs')
    frozen=read_json(OUT/'freeze.json')
    if any(frozen.get(k)!=v for k,v in contract().items()): raise ValueError('Frozen contract changed')
    for path,digest in frozen['input_hashes'].items():
        if sha(Path(path))!=digest: raise ValueError('Frozen V50 information source changed')
    sources=list(helper.EXTERNAL)
    feats=source_features([Path(p) for p in frozen['monthly_input_hashes']],sources)
    prior_features=pd.read_csv(PARENT/'source_feature.csv').set_index('source')
    prior_pred=pd.read_csv(PARENT/'source_heldout_predictions.csv').set_index('held_source')
    if set(feats)!=set(sources) or set(prior_pred.index)!=set(sources): raise ValueError('Ten-source parent required')
    feature_rows=[]
    for source in sources:
        level=feats[source]['level'];parent_level=float(prior_features.loc[source,'mean_legal_monthly_EF_proxy'])
        if abs(level-parent_level)>1e-9: raise ValueError('Recomputed level feature differs from frozen parent')
        feature_rows.append({'source':source,**feats[source]})
    feature_table=pd.DataFrame(feature_rows)
    x=feature_table[['level','volatility']].to_numpy(float)
    predictions=[]
    for outcome,ycol in [('p90','observed_p90_score'),('p99','observed_p99_score')]:
        y=np.array([float(prior_pred.loc[source,ycol]) for source in sources])
        for model in ('null','level','volatility','level_plus_volatility'):
            for row in loo(x,y,sources,model): predictions.append({'outcome':outcome,**row})
    pred=pd.DataFrame(predictions);summary=summarize(pred)
    feature_path=OUT/'source_features.csv';pred_path=OUT/'heldout_predictions.csv';summary_path=OUT/'summary.csv'
    feature_table.to_csv(feature_path,index=False,mode='x');pred.to_csv(pred_path,index=False,mode='x');summary.to_csv(summary_path,index=False,mode='x')
    complete={**contract(),'status':'COMPLETE_SOURCE_HELDOUT_PROBE','freeze_sha256':sha(OUT/'freeze.json'),
        'source_features_sha256':sha(feature_path),'heldout_predictions_sha256':sha(pred_path),
        'summary_sha256':sha(summary_path),'wrong_source_or_target_access':False}
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,allow_nan=False)+'\n')
    print(summary.to_string(index=False))


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run'));args=p.parse_args()
    prepare() if args.command=='prepare' else run()


if __name__=='__main__':main()
