"""Immutable, source-only coarse conditional diagnostics; no hourly truth I/O."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import coarse_context_v30 as model

OUT=ROOT/'results/coarse-context-v30'
PARENT=ROOT/'docs/evidence/conditional-law-v29-verification.json'
PARENT_SHA='7512d5e67a21073e42150f7ffb3b379cdd0d03f1061bdcff18348a8c9253b35f'
CODE=('scripts/coarse_context_v30.py','scripts/run_coarse_context_v30.py',
      'tests/test_coarse_context_v30.py','tests/test_run_coarse_context_v30.py','docs/COARSE_CONTEXT_V30_PROTOCOL.md')
REVIEWS=tuple(ROOT/f'results/coarse-context-v30-{name}-review.md' for name in ('code','python','science'))
METRICS=('ef_abs_error','ef_sq_error','ef_signed_error','ratio_l1','coal_abs_error','projection_l1')
OUTPUTS=('case_metrics.csv','region_metrics.csv','macro_metrics.csv','matched_contrasts.csv','source_choice.json')


def sha(path: str|Path) -> str:
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path: str|Path) -> dict:
    return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    return value


def write(path: Path,value: dict) -> None:
    with path.open('x') as f:json.dump(json_value(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify(mapping: dict) -> None:
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('Changed bound bytes: '+path)


def environment() -> dict:
    return dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p)
        for p in ('numpy','pandas','scipy','scikit-learn')})


def expected_contract() -> dict:
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Fixed audited V29 parent required')
    parent=read(PARENT)
    if parent['status']!='VERIFICATION_COMPLETE' or parent['independent_actual_audit_status']!='PASS':
        raise ValueError('Independent actual V29 PASS required')
    inputs={str(PARENT):PARENT_SHA}
    for name in ('producer_code_hashes','input_hashes','auditor_code_hashes','artifact_hashes'):
        for path,digest in parent[name].items():
            if path in inputs and inputs[path]!=digest:raise ValueError('Conflicting parent mappings')
            inputs[path]=digest
    verify(inputs)
    frozen=read(ROOT/'results/conditional-law-v29/freeze.json')
    if frozen['monthly_path'] not in inputs:raise ValueError('Bound source monthly alias required')
    for path in REVIEWS:
        if not path.is_file() or not path.read_text().strip():raise ValueError('Three final prelaunch reviews required')
        inputs[str(path)]=sha(path)
    return dict(version=30,sources=list(model.SOURCES),arms=list(model.ARMS),
        feature_dimensions={k:len(v) for k,v in model.FEATURE_NAMES.items()},
        monthly_path=frozen['monthly_path'],code_hashes={str(ROOT/n):sha(ROOT/n) for n in CODE},input_hashes=inputs,
        environment=environment(),label_start=model.START.isoformat(),label_end_exclusive=model.END.isoformat(),
        deployment_cutoff=model.CUTOFF.isoformat(),history_rows=3,fit_count=70,
        oracle_current_group=True,forward_forecast=False,hourly_values_read=False,
        china_category_mapping_verified=False,real_publication_vintage_verified=False,
        target_values_opened=False,unseen_values_opened=False,custom_method=False)


def prepare() -> None:
    if OUT.exists():raise FileExistsError('Preserve V30 attempt')
    contract=expected_contract();OUT.mkdir();write(OUT/'freeze.json',dict(created_utc=now(),**contract))
    print('Prepared',len(contract['code_hashes']),'code',len(contract['input_hashes']),'input hashes',flush=True)


def check() -> dict:
    frozen=read(OUT/'freeze.json');expected=expected_contract()
    if (set(frozen)!=set(expected)|{'created_utc'} or any(frozen[k]!=v for k,v in expected.items())
            or any(frozen[k] is not v for k,v in expected.items() if isinstance(v,bool))):
        raise ValueError('Frozen source, closure or conditional diagnostic contract changed')
    return frozen


def monthly_rows(frozen: dict) -> list[dict]:
    with Path(frozen['monthly_path']).open(newline='') as f:return list(csv.DictReader(f))


def predict() -> None:
    frozen=check();freeze_hash=sha(OUT/'freeze.json')
    folder=OUT/'predictions'
    if folder.exists() or (OUT/'prediction_seal.json').exists():raise FileExistsError('Preserve V30 prediction attempt')
    folder.mkdir();rows=monthly_rows(frozen);cases=model.build_cases(rows)
    if set(cases['regions'])!=set(model.SOURCES):raise ValueError('All seven visible source cohorts required')
    entries=[]
    for region in model.SOURCES:
        result=model.fit_predict(rows,cases,outer=region);qi=result['query_ids']
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        arrays.update({k:np.asarray(v) for k,v in result['query_provenance'].items()})
        arrays.update(arms=np.asarray(model.ARMS),keys=np.asarray([cases['keys'][i] for i in qi]),
            origins=cases['origin_ns'][qi],regions=cases['regions'][qi])
        path=folder/f'{region}.npz'
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        audit=folder/f'{region}.json'
        write(audit,dict(region=region,**result['audit'],fits=result['fits'],cohort=cases['audit'],
            all_case_keys=cases['keys'],all_case_regions=cases['regions'],feature_dimensions=frozen['feature_dimensions'],
            oracle_current_group=True,forward_forecast=False,hourly_values_read=False))
        entries.append(dict(region=region,path=str(path),sha256=sha(path),audit_path=str(audit),
            audit_sha256=sha(audit),rows=len(qi),query_keys=arrays['keys'].tolist()))
        print('Predicted',region,'cases',len(qi),'fits',len(result['fits']),flush=True)
    check();verify({str(OUT/'freeze.json'):freeze_hash})
    write(OUT/'prediction_seal.json',dict(created_utc=now(),freeze_sha256=freeze_hash,entries=entries,
        complete_grid=True,archives=7,arms=10,fit_count=70,eligible_cases=len(cases['keys']),
        oracle_current_group=True,forward_forecast=False,hourly_values_read=False,
        target_values_opened=False,unseen_values_opened=False))


def predictions() -> dict:
    seal=read(OUT/'prediction_seal.json')
    if (seal['freeze_sha256']!=sha(OUT/'freeze.json') or seal['complete_grid'] is not True
            or seal['archives']!=7 or seal['arms']!=10 or seal['fit_count']!=70
            or seal['oracle_current_group'] is not True
            or any(seal[k] is not False for k in ('forward_forecast','hourly_values_read','target_values_opened','unseen_values_opened'))
            or [e['region'] for e in seal['entries']]!=list(model.SOURCES)):
        raise ValueError('Complete conditional seven-source seal required before score labels')
    folder=OUT/'predictions';expected={f'{s}.{ext}' for s in model.SOURCES for ext in ('npz','json')}
    if (not folder.is_dir() or {p.name for p in folder.iterdir()}!=expected
            or any(not p.is_file() or p.is_symlink() for p in folder.iterdir())):
        raise ValueError('Exact fourteen regular prediction files required')
    count=0
    for entry in seal['entries']:
        path=folder/f"{entry['region']}.npz"
        if entry['path']!=str(path) or entry['audit_path']!=str(path.with_suffix('.json')):
            raise ValueError('Registered archive/audit aliases required')
        verify({entry['path']:entry['sha256'],entry['audit_path']:entry['audit_sha256']})
        with np.load(path,allow_pickle=False) as a:
            n=entry['rows'];count+=n
            if (n<1 or a['arms'].tolist()!=list(model.ARMS) or a['keys'].tolist()!=entry['query_keys']
                    or a['ratios'].shape!=(10,n,4) or a['ratios_raw'].shape!=(10,n,4)
                    or a['EF_point'].shape!=(10,n) or a['projection_l1'].shape!=(10,n)
                    or not np.isfinite(a['projection_l1']).all() or not np.isfinite(a['EF_point']).all()
                    or not np.isfinite(a['ratios']).all() or not np.isfinite(a['ratios_raw']).all()
                    or not np.all(a['regions']==entry['region'])):
                raise ValueError('Complete ten-head coarse archive required')
    if count!=seal['eligible_cases']:raise ValueError('Complete common visible cohort required')
    return seal


def matched_pairs() -> list[tuple[str,str]]:
    pairs=[]
    for name in model.MODELS:
        for candidate,reference in zip(model.FAMILIES[1:],model.FAMILIES[:-1]):pairs.append((f'{candidate}-{name}',f'{reference}-{name}'))
        pairs.extend((f'{family}-{name}',f'G5-{name}') for family in model.FAMILIES[2:])
    pairs.extend((f'{family}-ET',f'{family}-Ridge') for family in model.FAMILIES)
    if len(pairs)!=19 or len(set(pairs))!=19:raise ValueError('Fixed nineteen comparisons required')
    return pairs


def summarize(regional: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    macro=regional.groupby('arm')[[*METRICS,'ef_rmse']].mean().reset_index()
    signals=[]
    for candidate,reference in matched_pairs()[:14]:
        a=regional[regional.arm.eq(candidate)].set_index('region').ef_abs_error.reindex(model.SOURCES)
        b=regional[regional.arm.eq(reference)].set_index('region').ef_abs_error.reindex(model.SOURCES)
        good=(b.mean()>1e-12 and a.mean()<=.95*b.mean() and int((a<b-1e-12).sum())>=4
            and bool((a<=1.1*b+1e-12).all()) and all(a[r]<=b[r]+1e-12 for r in ('MISO','ERCO')))
        signals.append(dict(candidate=candidate,reference=reference,signal=bool(good),
            improved_regions=int((a<b-1e-12).sum()),macro_ef_mae_delta=float(a.mean()-b.mean())))
    adjacent_context={(f'{a}-{m}',f'{b}-{m}') for m in model.MODELS
                      for a,b in zip(model.FAMILIES[2:],model.FAMILIES[1:-1])}
    return macro,dict(comparisons=signals,substantial_context_signal=any(s['signal'] and (s['candidate'],s['reference']) in adjacent_context for s in signals),
        oracle_current_group=True,forward_forecast=False,hourly_stage_authorized=False,new_holdout_authorized=False,
        mechanism_breakthrough=False,custom_method=False)


def score() -> None:
    if any((OUT/n).exists() for n in (*OUTPUTS,'source_completed.json')):raise FileExistsError('Preserve V30 score attempt')
    frozen=check();seal=predictions();initial={str(OUT/n):sha(OUT/n) for n in ('freeze.json','prediction_seal.json')}
    rows=monthly_rows(frozen);lookup={}
    for row in rows:
        if row['region'] in model.SOURCES:
            start=pd.Timestamp(row['period_start']);lookup[f"{row['region']}/{start.year:04d}-{start.month:02d}"]=row
    records=[]
    for entry in seal['entries']:
        with np.load(entry['path'],allow_pickle=False) as a:
            y=np.stack([model.response(lookup[k]) for k in a['keys']]);ef=y@model.EF[model.POS]
            for i,arm in enumerate(model.ARMS):
                error=a['EF_point'][i]-ef
                for j,key in enumerate(a['keys']):
                    records.append(dict(region=entry['region'],key=str(key),arm=arm,ef_abs_error=float(abs(error[j])),
                        ef_sq_error=float(error[j]**2),ef_signed_error=float(error[j]),
                        ratio_l1=float(abs(a['ratios'][i,j]-y[j]).sum()),coal_abs_error=float(abs(a['ratios'][i,j,0]-y[j,0])),
                        projection_l1=float(a['projection_l1'][i,j])))
        print('Scored coarse labels',entry['region'],flush=True)
    case=pd.DataFrame(records)
    if (len(case)!=seal['eligible_cases']*10 or case.duplicated(['region','key','arm']).any()
            or not np.isfinite(case[list(METRICS)]).all().all()):raise ValueError('Complete finite common-cohort coarse score grid required')
    region=case.groupby(['region','arm'])[list(METRICS)].mean().reset_index();region['ef_rmse']=np.sqrt(region.ef_sq_error)
    macro,choice=summarize(region);contrasts=[]
    for candidate,reference in matched_pairs():
        paired=region[region.arm.eq(candidate)].merge(region[region.arm.eq(reference)],on='region',suffixes=('_c','_r'),validate='one_to_one')
        for row in paired.to_dict('records'):
            contrasts.append(dict(region=row['region'],candidate=candidate,reference=reference,
                **{k+'_delta':row[k+'_c']-row[k+'_r'] for k in (*METRICS,'ef_rmse')}))
    if len(region)!=70 or len(macro)!=10 or len(contrasts)!=133:raise ValueError('Exact70/10/133 coarse tables required')
    for name,frame in [('case_metrics',case),('region_metrics',region),('macro_metrics',macro),('matched_contrasts',pd.DataFrame(contrasts))]:
        frame.to_csv(OUT/f'{name}.csv',index=False,mode='x')
    write(OUT/'source_choice.json',choice);check();predictions();verify(initial)
    write(OUT/'source_completed.json',dict(version=30,created_utc=now(),freeze_sha256=initial[str(OUT/'freeze.json')],
        prediction_seal_sha256=initial[str(OUT/'prediction_seal.json')],output_hashes={str(OUT/n):sha(OUT/n) for n in OUTPUTS},
        oracle_current_group=True,forward_forecast=False,hourly_values_read=False,target_values_opened=False,
        unseen_values_opened=False,mechanism_breakthrough=False,custom_method=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('prepare','predict','score'))
    args=parser.parse_args();{'prepare':prepare,'predict':predict,'score':score}[args.stage]()
