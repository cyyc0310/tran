"""Nested source-held-out shrinkage of a monthly conditional MCStd threshold."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/disjoint-calibration-v50'
MIX=ROOT/'results/disjoint-calibration-v50-full-mix-probe'
OUT=ROOT/'results/disjoint-calibration-v50-nested-mix-q-shrinkage'
PROTOCOL=ROOT/'docs/V50_NESTED_MIX_Q_SHRINKAGE_PROTOCOL.md'
SCRIPT=ROOT/'scripts/run_v50_nested_mix_q_shrinkage.py'
sys.path.insert(0,str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402
from scripts.run_v50_coarse_mix_conditional_interval import (  # noqa: E402
    N_METHOD, equal_source_quantile, inverse, metrics)

SOURCES,EXTERNAL=helper.LEGACY,helper.EXTERNAL
METHODS=('GlobalABS','MCStd','Component','HETScale','HETLocationScale','CQRplus')
LAMBDAS=(0.,.25,.5,.75,1.)
SELECTION_COVERAGE=(.90,.95)
SELECTION_WORST=.85


def sha(path:Path)->str:
    return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def read_json(path:Path)->dict:
    return json.loads(path.read_text())


def write_json(path:Path,obj:dict)->None:
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def contract()->dict:
    return {'version':1,'sources':list(SOURCES),'external_sources':list(EXTERNAL),
        'outer_functions':list(SOURCES),'shrinkage_grid':list(LAMBDAS),'primary_level':.9,
        'inner_coverage_range':list(SELECTION_COVERAGE),'inner_worst_source_floor':SELECTION_WORST,
        'primary_arm':'B0 Scale-anchor + MCStd','target_or_unseen_opened':False,
        'point_or_risk_refits':0,'algorithm_novelty_claim':False}


def closure()->dict[str,str]:
    freeze=read_json(RUN/'freeze.json');predseal=read_json(RUN/'prediction_seal.json');calseal=read_json(RUN/'calibration_seal.json')
    if (read_json(RUN/'completed.json').get('status')!='COMPLETE'
        or predseal.get('freeze_sha256')!=sha(RUN/'freeze.json')
        or calseal.get('prediction_seal_sha256')!=sha(RUN/'prediction_seal.json')):
        raise ValueError('Complete original V50 seals required')
    mix_summary=read_json(MIX/'summary.json');
    if mix_summary.get('status')!='COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE': raise ValueError('Full-mix source feature parent incomplete')
    feature_path=MIX/'source_feature.csv';prior_prediction=MIX/'source_heldout_predictions.csv'
    if (sha(feature_path)!=mix_summary.get('feature_csv_sha256') or sha(prior_prediction)!=mix_summary.get('heldout_csv_sha256')):
        raise ValueError('Full-mix parent output hash mismatch')
    prediction_entries={(e['outer'],e['region']):e for e in predseal['entries'] if e.get('role')=='calibration' and e.get('region') in EXTERNAL}
    label_entries={e['region']:e for e in calseal['labels']}
    if set(prediction_entries)!={(o,r) for o in SOURCES for r in EXTERNAL} or set(label_entries)!=set(EXTERNAL):
        raise ValueError('Fixed 70 prediction ×10 label grid required')
    paths=[RUN/'freeze.json',RUN/'prediction_seal.json',RUN/'calibration_seal.json',RUN/'completed.json',
           MIX/'summary.json',feature_path,prior_prediction,PROTOCOL,SCRIPT]
    paths += [Path(p) for p in freeze['external_monthly']]
    paths += [RUN/'external_labels'/f'{r}.npz' for r in EXTERNAL]
    paths += [RUN/'predictions'/f'{o}__{r}.npz' for o in SOURCES for r in EXTERNAL]
    hashes={str(p):sha(p) for p in paths}
    for p in freeze['external_monthly']:
        if hashes[p]!=freeze['input_hashes'].get(p): raise ValueError('V50 monthly input differs from frozen hash')
    for r in EXTERNAL:
        entry=label_entries[r];p=RUN/'external_labels'/f'{r}.npz'
        if entry['path']!=str(p) or hashes[str(p)]!=entry['sha256']: raise ValueError('Calibration label seal mismatch')
    for o in SOURCES:
        for r in EXTERNAL:
            entry=prediction_entries[(o,r)];p=RUN/'predictions'/f'{o}__{r}.npz'
            if entry['path']!=str(p) or hashes[str(p)]!=entry['sha256']: raise ValueError('Prediction archive seal mismatch')
    return hashes


def prepare()->None:
    if OUT.exists():raise FileExistsError('Preserve an existing or partial nested-q run')
    OUT.mkdir(parents=True)
    frozen={**contract(),'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'input_hashes':closure(),'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(SCRIPT)}
    write_json(OUT/'freeze.json',frozen)


def check_freeze()->dict:
    f=read_json(OUT/'freeze.json')
    if any(f.get(k)!=v for k,v in contract().items()) or f['protocol_sha256']!=sha(PROTOCOL) or f['script_sha256']!=sha(SCRIPT) or f['input_hashes']!=closure():
        raise ValueError('Nested-q frozen contract/input/code changed')
    return f


def load_caches()->tuple[dict,dict,dict,dict]:
    freeze=read_json(RUN/'freeze.json');predseal=read_json(RUN/'prediction_seal.json');calseal=read_json(RUN/'calibration_seal.json')
    labels={};scores={};hetscores={};preds={}
    label_entries={e['region']:e for e in calseal['labels']}
    prediction_entries={(e['outer'],e['region']):e for e in predseal['entries'] if e.get('role')=='calibration' and e.get('region') in EXTERNAL}
    for source in EXTERNAL:
        lp=RUN/'external_labels'/f'{source}.npz'
        with np.load(lp,allow_pickle=False) as z:
            positions,truth,origins=z['positions'].astype(int),z['truth'].copy(),z['origins'].copy()
        keep=helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins,utc=True)),90)
        if int(keep.sum())!=freeze['lag_counts']['90'][source]:raise ValueError('90-day support mismatch')
        labels[source]=(positions[keep],truth[keep]);scores[source]={};hetscores[source]={};preds[source]={}
        for outer in SOURCES:
            pp=RUN/'predictions'/f'{outer}__{source}.npz'
            with np.load(pp,allow_pickle=False) as z:
                if z['outer'].item()!=outer or z['region'].item()!=source or z['methods'].tolist()!=list(METHODS):
                    raise ValueError('Prediction archive identity mismatch')
                pos,y=labels[source];lo0,hi0,scale=(np.take(z[k][0,1],pos,axis=0) for k in ('lo0','hi0','scales'))
                s=np.maximum(np.maximum((lo0-y)/scale,(y-hi0)/scale),0.)
                hb={k:np.take(z[k][0,N_METHOD['HETScale']],pos,axis=0) for k in ('lo0','hi0','scales')}
                hs=np.maximum(np.maximum((hb['lo0']-y)/hb['scales'],(y-hb['hi0'])/hb['scales']),0.)
                scores[source][outer]=s;hetscores[source][outer]=hs
                preds[source][outer]={'lo0':lo0.copy(),'hi0':hi0.copy(),'scale':scale.copy(),'truth':y.copy(),
                    'raw':np.take(z['raw'][0],pos,axis=0),'het':hb}
    return labels,scores,hetscores,preds


def predict_local_q(training:tuple[str,...],test:str,outer:str,features:dict[str,float],scores:dict)->float:
    x=np.array([features[s] for s in training],float)
    y=np.array([inverse(scores[s][outer],.9) for s in training],float)
    design=np.column_stack([np.ones(len(x)),x]);beta=np.linalg.lstsq(design,y,rcond=None)[0]
    return max(0.,float(np.array([1.,features[test]])@beta))


def apply_interval(stored:dict,q:float,*,het:bool=False)->dict:
    if het:
        b=stored['het'];lo=np.clip(b['lo0']-q*b['scales'],0,1000);hi=np.clip(b['hi0']+q*b['scales'],0,1000)
    else:
        lo=np.clip(stored['lo0']-q*stored['scale'],0,1000);hi=np.clip(stored['hi0']+q*stored['scale'],0,1000)
    return metrics(lo,hi,stored['truth'])


def inner_lambda(training:tuple[str,...],outer:str,features:dict[str,float],scores:dict,preds:dict)->tuple[float,list[dict]]:
    by_lambda={lam:[] for lam in LAMBDAS}
    for validation in training:
        inner=tuple(s for s in training if s!=validation)
        qg=equal_source_quantile({s:scores[s][outer] for s in inner},.9)
        ql=predict_local_q(inner,validation,outer,features,scores)
        stored=preds[validation][outer]
        for lam in LAMBDAS:
            metric=apply_interval(stored,(1-lam)*qg+lam*ql)
            by_lambda[lam].append(metric)
    details=[]
    feasible=[]
    for lam,vals in by_lambda.items():
        cov=float(np.mean([x['coverage90'] for x in vals]));worst=float(np.min([x['coverage90'] for x in vals]))
        iscore=float(np.mean([x['interval_score90'] for x in vals]));width=float(np.mean([x['width90'] for x in vals]))
        ok=SELECTION_COVERAGE[0]<=cov<=SELECTION_COVERAGE[1] and worst>=SELECTION_WORST
        details.append({'lambda':lam,'inner_source_macro_coverage':cov,'inner_worst_source_coverage':worst,
            'inner_interval_score':iscore,'inner_width':width,'feasible':bool(ok)})
        if ok:feasible.append((iscore,lam))
    if feasible: selected=min(feasible,key=lambda x:(x[0],x[1]))[1]
    else:selected=0.
    return selected,details


def run()->None:
    if any((OUT/n).exists() for n in ('completion.json','outer_metrics.csv','source_macro_metrics.csv','lambda_selection.csv')):
        raise FileExistsError('Preserve partial or completed nested-q outputs')
    frozen=check_freeze();initial=sha(OUT/'freeze.json')
    mix_features=pd.read_csv(MIX/'source_feature.csv').set_index('source')['mean_legal_monthly_EF_proxy'].to_dict()
    if set(mix_features)!=set(EXTERNAL):raise ValueError('Exact ten source monthly features required')
    labels,scores,hetscores,preds=load_caches()
    rows=[];select_rows=[]
    for outer in SOURCES:
        for held in EXTERNAL:
            training=tuple(s for s in EXTERNAL if s!=held)
            qg=equal_source_quantile({s:scores[s][outer] for s in training},.9)
            ql=predict_local_q(training,held,outer,mix_features,scores)
            lam,details=inner_lambda(training,outer,mix_features,scores,preds)
            stored=preds[held][outer]
            qs={'nested_shrink_MCStd':(1-lam)*qg+lam*ql,'global_MCStd':qg,'mix_conditional_MCStd':ql}
            for method,q in qs.items():
                rows.append({'outer':outer,'held_source':held,'method':method,'q_prediction':q,
                    'q_global':qg,'q_local':ql,'selected_lambda':lam,**apply_interval(stored,q)})
            hq=equal_source_quantile({s:hetscores[s][outer] for s in training},.9)
            rows.append({'outer':outer,'held_source':held,'method':'global_HETScale','q_prediction':hq,
                'q_global':hq,'q_local':np.nan,'selected_lambda':np.nan,
                **apply_interval(stored,hq,het=True)})
            raw=stored['raw'];center=raw.mean(axis=1,keepdims=True)
            expanded=np.sort(np.clip(center+2*(raw-center),0,1000),axis=1)
            rows.append({'outer':outer,'held_source':held,'method':'raw_anchor_factor2','q_prediction':2.,
                'q_global':np.nan,'q_local':np.nan,'selected_lambda':np.nan,
                **metrics(expanded[:,6],expanded[:,121],stored['truth'])})
            feasible_count=sum(d['feasible'] for d in details)
            for d in details:select_rows.append({'outer':outer,'held_source':held,'selected_lambda':lam,
                'selected_by_fallback':int(feasible_count==0),**d})
    frame=pd.DataFrame(rows);selection=pd.DataFrame(select_rows)
    if len(frame)!=350 or len(selection)!=350 or frame.duplicated(['outer','held_source','method']).any():
        raise ValueError('Fixed 7×10×5 outer comparison table required')
    source_macro=frame.groupby(['held_source','method'],sort=False)[
        ['coverage90','width90','interval_score90','lower_miss','upper_miss']].mean().reset_index()
    macro=source_macro.groupby('method')[['coverage90','width90','interval_score90','lower_miss','upper_miss']].mean().reset_index()
    for path,table in ((OUT/'outer_metrics.csv',frame),(OUT/'lambda_selection.csv',selection),
                       (OUT/'source_macro_metrics.csv',source_macro),(OUT/'macro_summary.csv',macro)):
        table.to_csv(path,index=False,mode='x')
    check_freeze()
    if sha(OUT/'freeze.json')!=initial:raise ValueError('Freeze changed during nested-q computation')
    write_json(OUT/'completion.json',{'status':'COMPLETE_NESTED_SOURCE_HELDOUT_DIAGNOSTIC',**contract(),
        'freeze_sha256':initial,'input_hashes':frozen['input_hashes'],
        'output_hashes':{str(p):sha(p) for p in (OUT/'outer_metrics.csv',OUT/'lambda_selection.csv',OUT/'source_macro_metrics.csv',OUT/'macro_summary.csv')},
        'target_or_unseen_opened':False,'point_or_risk_refits':0,'algorithm_novelty_claim':False})
    print(macro.to_string(index=False))
    per_fold=selection.groupby(['outer','held_source']).first()
    print('selected lambda counts:',per_fold.selected_lambda.value_counts().sort_index().to_dict(),
          'fallback folds:',int(per_fold.selected_by_fallback.sum()))


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.command=='prepare' else run()


if __name__=='__main__':main()
