"""Fixed V30 head reproduction on archived legal monthly forecasts, no hourly I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import forward_context_v31 as model
from scripts import run_coarse_context_v30 as common

OUT=ROOT/'results/forward-context-v31'
PARENT=ROOT/'docs/evidence/coarse-context-v30-verification.json'
PARENT_SHA='b3590c85d46d99510403432a24c658f1df85137831a3c46005054f0df994cf4a'
CODE=('scripts/forward_context_v31.py','scripts/run_forward_context_v31.py',
      'tests/test_forward_context_v31.py','tests/test_run_forward_context_v31.py','docs/FORWARD_CONTEXT_V31_PROTOCOL.md')
REVIEWS=tuple(ROOT/f'results/forward-context-v31-{k}-review.md' for k in ('code','python','science'))
DATES=pd.date_range('2025-01-01','2025-06-01',freq='MS',tz='UTC').as_unit('ns')
SCORE_CUTOFF=pd.Timestamp('2025-09-01',tz='UTC')
BASE=('cif_abs_error','cif_sq_error','cif_signed_error','g5_l1','pos_abs_error','projection_l1')
SECONDARY=('ef_abs_error','ef_sq_error','ef_signed_error')
METRICS=BASE+SECONDARY
OUTPUTS=('case_metrics.csv','region_metrics.csv','macro_metrics.csv','matched_contrasts.csv','source_choice.json','label_support.csv')
sha,read,write,verify,now=common.sha,common.read,common.write,common.verify,common.now


def expected_contract() -> dict:
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Fixed independently audited V30 parent required')
    evidence=read(PARENT)
    if evidence['status']!='VERIFICATION_COMPLETE' or evidence['independent_actual_audit_status']!='PASS':
        raise ValueError('Actual V30 PASS required')
    inputs={str(PARENT):PARENT_SHA}
    for field in ('producer_code_hashes','input_hashes','auditor_code_hashes','artifact_hashes'):
        for path,digest in evidence[field].items():
            if path in inputs and inputs[path]!=digest:raise ValueError('Conflicting parent mappings')
            inputs[path]=digest
    verify(inputs)
    old=read(ROOT/'results/coarse-context-v30/freeze.json');seal=read(ROOT/'results/coarse-context-v30/prediction_seal.json')
    v29=read(ROOT/'results/conditional-law-v29/freeze.json')
    head_paths={e['region']:e['audit_path'] for e in seal['entries']}
    forecast_paths=v29['monthly_prediction_paths']
    aliases=[old['monthly_path'],*head_paths.values(),*forecast_paths.values()]
    if set(head_paths)!=set(model.SOURCES) or set(forecast_paths)!=set(model.SOURCES) or any(p not in inputs for p in aliases):
        raise ValueError('Bound seven head/forecast aliases and monthly source required')
    for path in REVIEWS:
        if not path.is_file() or not path.read_text().strip():raise ValueError('Three final prelaunch reviews required')
        inputs[str(path)]=sha(path)
    return dict(version=31,sources=list(model.SOURCES),arms=list(model.ARMS),origins=DATES.asi8.tolist(),
        samples=128,head_refits=42,monthly_path=old['monthly_path'],head_source_paths=head_paths,
        forecast_paths=forecast_paths,code_hashes={str(ROOT/n):sha(ROOT/n) for n in CODE},input_hashes=inputs,
        environment=common.environment(),deployment_cutoff=model.CUTOFF.isoformat(),score_cutoff=SCORE_CUTOFF.isoformat(),
        label_minimum_coverage=.9,oracle_current_group=False,forward_forecast=True,hourly_values_read=False,
        new_training_labels_used=False,probability_coverage_evaluated=False,target_values_opened=False,
        unseen_values_opened=False,custom_method=False,china_category_mapping_verified=False,real_publication_vintage_verified=False)


def prepare() -> None:
    if OUT.exists():raise FileExistsError('Preserve V31 attempt')
    contract=expected_contract();OUT.mkdir();write(OUT/'freeze.json',dict(created_utc=now(),**contract))
    print('Prepared',len(contract['code_hashes']),'code',len(contract['input_hashes']),'input hashes',flush=True)


def check() -> dict:
    frozen=read(OUT/'freeze.json');expected=expected_contract()
    if (set(frozen)!=set(expected)|{'created_utc'} or any(frozen[k]!=v for k,v in expected.items())
            or any(frozen[k] is not v for k,v in expected.items() if isinstance(v,bool))):
        raise ValueError('Frozen forecast-only contract/closure changed')
    return frozen


def forecast_inputs(path: str) -> np.ndarray:
    with np.load(path,allow_pickle=False) as archive:
        origins=archive['origins'];draws=archive['MG_draws']
    expected=pd.date_range('2025-01-01','2025-06-30',tz='UTC').as_unit('ns').asi8
    if not np.array_equal(origins,expected) or draws.shape!=(181,128,5):raise ValueError('Fixed archived MG daily forecast grid required')
    indices=np.searchsorted(origins,DATES.asi8)
    if not np.array_equal(origins[indices],DATES.asi8):raise ValueError('Exact six month-start forecast origins required')
    return draws[indices].copy()


def predict() -> None:
    frozen=check();initial=sha(OUT/'freeze.json');folder=OUT/'predictions'
    if folder.exists() or (OUT/'prediction_seal.json').exists():raise FileExistsError('Preserve V31 prediction attempt')
    folder.mkdir();rows=common.monthly_rows(frozen);entries=[]
    for region in model.SOURCES:
        parent=read(frozen['head_source_paths'][region]);draws=forecast_inputs(frozen['forecast_paths'][region])
        result=model.fit_predict(parent,rows,region=region,origins=DATES,mg_draws=draws)
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        arrays.update(arms=np.asarray(model.ARMS),origins=DATES.asi8,keys=np.asarray([f'{region}/{at:%Y-%m}' for at in DATES]))
        path=folder/f'{region}.npz'
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        audit=path.with_suffix('.json')
        write(audit,dict(parent_head_path=frozen['head_source_paths'][region],forecast_path=frozen['forecast_paths'][region],**result['audit']))
        entries.append(dict(region=region,path=str(path),sha256=sha(path),audit_path=str(audit),audit_sha256=sha(audit)))
        print('Predicted',region,'origins',len(DATES),'same-head refits',result['audit']['fit_count'],flush=True)
    check();verify({str(OUT/'freeze.json'):initial})
    write(OUT/'prediction_seal.json',dict(created_utc=now(),freeze_sha256=initial,entries=entries,complete_grid=True,
        archives=7,arms=6,origins_per_source=6,head_refits=42,oracle_current_group=False,forward_forecast=True,
        hourly_values_read=False,new_training_labels_used=False,target_values_opened=False,unseen_values_opened=False))


def predictions() -> dict:
    seal=read(OUT/'prediction_seal.json')
    closed=('oracle_current_group','hourly_values_read','new_training_labels_used','target_values_opened','unseen_values_opened')
    if (seal['freeze_sha256']!=sha(OUT/'freeze.json') or seal['complete_grid'] is not True or seal['forward_forecast'] is not True
            or any(seal[k] is not False for k in closed) or seal['archives']!=7 or seal['arms']!=6
            or seal['origins_per_source']!=6 or seal['head_refits']!=42
            or [e['region'] for e in seal['entries']]!=list(model.SOURCES)):
        raise ValueError('Complete seven-source forecast-only seal required before labels')
    folder=OUT/'predictions'
    if (not folder.is_dir() or {p.name for p in folder.iterdir()}!={f'{s}.{e}' for s in model.SOURCES for e in ('npz','json')}
            or any(not p.is_file() or p.is_symlink() for p in folder.iterdir())):
        raise ValueError('Exact fourteen regular prediction files required')
    for e in seal['entries']:
        path=folder/f"{e['region']}.npz"
        if e['path']!=str(path) or e['audit_path']!=str(path.with_suffix('.json')):raise ValueError('Registered forecast archive aliases required')
        verify({e['path']:e['sha256'],e['audit_path']:e['audit_sha256']})
        with np.load(path,allow_pickle=False) as a:
            if (a['arms'].tolist()!=list(model.ARMS) or not np.array_equal(a['origins'],DATES.asi8)
                    or a['keys'].tolist()!=[f"{e['region']}/{at:%Y-%m}" for at in DATES]
                    or any(a[k].shape!=(6,6,128,4) or not np.isfinite(a[k]).all() for k in ('ratios_raw','ratios'))
                    or any(a[k].shape!=(6,6,128) or not np.isfinite(a[k]).all() for k in ('EF_draws','CIF_draws','projection_l1'))
                    or any(a[k].shape!=(6,6) or not np.isfinite(a[k]).all() for k in ('EF_point','CIF_point'))):
                raise ValueError('Full fixed six-arm monthly forecast grid required')
    return seal


def scoring_labels(rows: list[dict]) -> tuple[dict,pd.DataFrame]:
    wanted={f'{region}/{at:%Y-%m}' for region in model.SOURCES for at in DATES};selected={}
    for row in rows:
        if row['region'] not in model.SOURCES:continue
        start=pd.Timestamp(row['period_start']);key=f"{row['region']}/{start:%Y-%m}"
        if key not in wanted:continue
        if key in selected:raise ValueError('Duplicate monthly score label')
        selected[key]=row
    labels={};support=[]
    for region in model.SOURCES:
        for at in DATES:
            key=f'{region}/{at:%Y-%m}';item=dict(region=region,key=key,origin=at.isoformat(),status='missing',
                available_at='',coverage=np.nan,total_mwh=np.nan,ef_defined=False)
            if key not in selected:support.append(item);continue
            row=selected[key];start=model.parent._utc(row['period_start']);end=model.parent._utc(row['period_end']);release=model.parent._utc(row['available_at'])
            if start!=at or end!=at+pd.offsets.MonthBegin(1) or release<end:raise ValueError('Invalid whole-month score metadata')
            item['available_at']=release.isoformat()
            if release>SCORE_CUTOFF:item['status']='not_released';support.append(item);continue
            coverage=float(row['coverage']);item['coverage']=coverage
            if not np.isfinite(coverage) or not 0<=coverage<=1:raise ValueError('Finite valid label coverage required')
            if coverage<.9:item['status']='insufficient_coverage';support.append(item);continue
            total=float(row['total_mwh']);item['total_mwh']=total
            if not np.isfinite(total):raise ValueError('Finite label total required')
            if total<=0:item['status']='nonpositive_total';support.append(item);continue
            mix=np.array([float(row[c]) for c in model.parent.SHARES]);model.parent._simplex(mix,8)
            mass=mix[model.POS].sum();defined=bool(mass>1e-12)
            labels[key]=dict(cif=float(mix@model.EF),ef=float(mix[model.POS]@model.EF[model.POS]/mass) if defined else np.nan,
                g5=np.r_[mass,mix[model.ZERO]],ef_defined=defined)
            item.update(status='scored',ef_defined=defined);support.append(item)
    return labels,pd.DataFrame(support)


def matched_pairs() -> list[tuple[str,str]]:
    pairs=[(f'{a}-{m}',f'{b}-{m}') for m in model.MODELS for a,b in [('Size','Trend'),('Size','G5'),('Trend','G5')]]
    pairs.extend((f'{f}-ET',f'{f}-Ridge') for f in model.FAMILIES)
    if len(pairs)!=9 or len(set(pairs))!=9:raise ValueError('Fixed nine comparisons required')
    return pairs


def summarize(region: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    macro=region.groupby('arm')[[*METRICS,'cif_rmse','ef_rmse']].mean().reset_index()
    macro['scored_region_months']=region.groupby('arm').scored_months.sum().reindex(macro.arm).to_numpy()
    macro['internal_EF_region_months']=region.groupby('arm').internal_EF_months.sum().reindex(macro.arm).to_numpy()
    signals=[]
    for candidate,reference in matched_pairs()[:6]:
        a=region[region.arm.eq(candidate)].set_index('region').cif_abs_error.reindex(model.SOURCES)
        b=region[region.arm.eq(reference)].set_index('region').cif_abs_error.reindex(model.SOURCES)
        good=(b.mean()>1e-12 and a.mean()<=.95*b.mean() and int((a<b-1e-12).sum())>=4
            and bool((a<=1.1*b+1e-12).all()) and all(a[k]<=b[k]+1e-12 for k in ('MISO','ERCO')))
        signals.append(dict(candidate=candidate,reference=reference,signal=bool(good),
            improved_regions=int((a<b-1e-12).sum()),macro_cif_mae_delta=float(a.mean()-b.mean())))
    size_models=[m for m in model.MODELS if all(next(s['signal'] for s in signals if s['candidate']==f'Size-{m}' and s['reference']==f'{f}-{m}') for f in ('Trend','G5'))]
    return macro,dict(comparisons=signals,size_signal_models=size_models,substantial_forward_size_signal=bool(size_models),
        oracle_current_group=False,forward_forecast=True,probability_coverage_evaluated=False,
        hourly_stage_authorized=False,new_holdout_authorized=False,mechanism_breakthrough=False,custom_method=False)


def score() -> None:
    if any((OUT/n).exists() for n in (*OUTPUTS,'source_completed.json')):raise FileExistsError('Preserve V31 score attempt')
    frozen=check();seal=predictions();initial={str(OUT/n):sha(OUT/n) for n in ('freeze.json','prediction_seal.json')}
    labels,support=scoring_labels(common.monthly_rows(frozen));records=[]
    for entry in seal['entries']:
        with np.load(entry['path'],allow_pickle=False) as a:
            for j,key in enumerate(a['keys']):
                if key not in labels:continue
                y=labels[key];g=a['MG_draws'][j].mean(0)
                for i,arm in enumerate(model.ARMS):
                    error=float(a['CIF_point'][i,j]-y['cif']);e=float(a['EF_point'][i,j]-y['ef'])
                    records.append(dict(region=entry['region'],key=str(key),arm=arm,ef_defined=y['ef_defined'],
                        cif_abs_error=abs(error),cif_sq_error=error**2,cif_signed_error=error,
                        ef_abs_error=abs(e),ef_sq_error=e**2,ef_signed_error=e,
                        g5_l1=float(abs(g-y['g5']).sum()),pos_abs_error=float(abs(g[0]-y['g5'][0])),
                        projection_l1=float(a['projection_l1'][i,j].mean())))
        print('Scored monthly coarse targets',entry['region'],flush=True)
    case=pd.DataFrame(records)
    if (len(support)!=42 or len(case)!=len(labels)*6 or case.duplicated(['region','key','arm']).any()
            or set(case.region)!=set(model.SOURCES) or not np.isfinite(case[list(BASE)]).all().all()
            or not np.isfinite(case.loc[case.ef_defined,list(SECONDARY)]).all().all()
            or not case.loc[~case.ef_defined,list(SECONDARY)].isna().all().all()):
        raise ValueError('Complete common monthly score support required')
    region=case.groupby(['region','arm'])[list(METRICS)].mean().reset_index()
    region['cif_rmse']=np.sqrt(region.cif_sq_error);region['ef_rmse']=np.sqrt(region.ef_sq_error)
    counts=case.groupby(['region','arm']).agg(scored_months=('key','count'),internal_EF_months=('ef_defined','sum')).reset_index()
    region=region.merge(counts,on=['region','arm'],validate='one_to_one');macro,choice=summarize(region);contrasts=[]
    for candidate,reference in matched_pairs():
        paired=region[region.arm.eq(candidate)].merge(region[region.arm.eq(reference)],on='region',suffixes=('_c','_r'),validate='one_to_one')
        for row in paired.to_dict('records'):
            contrasts.append(dict(region=row['region'],candidate=candidate,reference=reference,
                **{k+'_delta':row[k+'_c']-row[k+'_r'] for k in (*METRICS,'cif_rmse','ef_rmse')}))
    if len(region)!=42 or len(macro)!=6 or len(contrasts)!=63:raise ValueError('Exact42region/6macro/63contrast grid required')
    for name,frame in [('case_metrics',case),('region_metrics',region),('macro_metrics',macro),('matched_contrasts',pd.DataFrame(contrasts)),('label_support',support)]:
        frame.to_csv(OUT/f'{name}.csv',mode='x',index=False)
    write(OUT/'source_choice.json',choice);check();predictions();verify(initial)
    write(OUT/'source_completed.json',dict(version=31,created_utc=now(),freeze_sha256=initial[str(OUT/'freeze.json')],
        prediction_seal_sha256=initial[str(OUT/'prediction_seal.json')],output_hashes={str(OUT/n):sha(OUT/n) for n in OUTPUTS},
        oracle_current_group=False,forward_forecast=True,hourly_values_read=False,new_training_labels_used=False,
        probability_coverage_evaluated=False,target_values_opened=False,unseen_values_opened=False,mechanism_breakthrough=False,custom_method=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('prepare','predict','score'))
    args=parser.parse_args();{'prepare':prepare,'predict':predict,'score':score}[args.stage]()
