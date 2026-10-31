"""Immutable source-only local-packet pilot; prepare/predict/score are separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import local_sketch_v17 as model
from scripts.run_information_value_v10 import environment,score_arrays

SOURCES,PACKETS,MODELS=model.SOURCES,model.PACKETS,model.MODELS
LAGS,FACTORS=(7,30),(.75,1.,1.25,1.5,2.)
METRICS=('crps','empirical_coverage90','empirical_width90','empirical_interval_score90',
         'empirical_lower_tail_miss','empirical_upper_tail_miss','point_level_mae','point_shape_mae')
OUT=ROOT/'results/local-sketch-v17'
OLD=ROOT/'results/weather-coarse-v14-round1/prefit_freeze.json'
OLD_MANIFEST=ROOT/'docs/evidence/weather-coarse-v14-verification.json'
CODE=('scripts/local_sketch_v17.py','scripts/run_local_sketch_v17.py',
      'tests/test_local_sketch_v17.py','docs/LOCAL_SKETCH_V17_PROTOCOL.md')


def sha(path: str|Path) -> str:
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path: Path) -> dict[str,Any]:return json.loads(path.read_text())


def write(path: Path,value: dict[str,Any]) -> None:
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def now() -> str:return datetime.now(timezone.utc).isoformat()


def verify(hashes: dict[str,str]) -> None:
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Changed frozen bytes: '+p)


def config() -> dict[str,Any]:
    return dict(version=17,kind='source_only_local_sketch_pilot',sources=list(SOURCES),
        packets=PACKETS,models=list(MODELS),lags=list(LAGS),factors=list(FACTORS),samples=model.SAMPLES,
        seed=model.SEED,point_trees=model.POINT_TREES,point_leaf=model.POINT_LEAF,
        qrf_trees=model.RISK_TREES,qrf_leaf=model.RISK_LEAF,qrf_max_features=.7,qrf_bootstrap=True,
        ridge_alpha=30.,history_calendar_days=28,ewma_scale_days=14,periods=model.PERIODS,
        target_values_opened=False,unseen_values_opened=False,source_aggregation_reads_2025_hourly=True,
        prediction_reads_2025_fine_labels=False,publication_vintages_verified=False,
        quantization_noise_phase_stress_completed=False,independent_confirmation=False)


def source_days(path: Path,start: str,stop: str) -> list[tuple[pd.Timestamp,np.ndarray]]:
    """Filter registered source timestamps before numeric fine-label conversion."""
    region=path.stem.removesuffix('_hourly')
    if region not in SOURCES:raise ValueError('Only registered source hourly inputs allowed')
    with path.open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if start<=r['time']<stop]
    t=pd.DataFrame(rows)
    if t.empty:return []
    dates=pd.DatetimeIndex(pd.to_datetime(t.time,utc=True,errors='raise'))
    if (dates.has_duplicates or not dates.equals(dates.floor('h')) or not t.region.eq(region).all()
            or not set(t.valid)<={'True','False'}):raise ValueError('Unique registered UTC source hours required')
    answer=[]
    for day in sorted(dates.floor('D').unique()):
        take=np.flatnonzero(dates.floor('D')==day)
        if len(take)!=24 or not t.iloc[take].valid.eq('True').all():continue
        take=take[np.argsort(dates[take].asi8)]
        if not np.array_equal(dates[take].asi8,pd.date_range(day,periods=24,freq='h').asi8):raise ValueError('Complete UTC day required')
        a=t.iloc[take][['generation_'+f for f in model.FUELS]].to_numpy(float)
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a.sum(1)<=0):raise ValueError('Bad source generation')
        answer.append((day,a))
    return answer


def prepare() -> None:
    """Disclosed isolated daily aggregation; never run implicitly during scoring."""
    if (OUT/'freeze.json').exists() or (OUT/'packets').exists():raise FileExistsError('Preserve existing/partial experiment')
    prior,evidence=read(OLD),read(OLD_MANIFEST)
    if prior['sources']!=list(SOURCES):raise ValueError('Changed inherited source registry')
    paths=[OLD,OLD_MANIFEST,Path(sys.executable).resolve()];native={}
    for region in SOURCES:
        p=Path(prior['source'])/f'{region}_hourly.csv'
        verify({str(p):prior['input_hashes'][str(p)]});paths.append(p)
        rel=f'results/weather-coarse-v14-round1/predictions/source-{region}-B-native_recompose.npz'
        p=ROOT/rel;verify({str(p):evidence['artifact_hashes'][rel]});paths.append(p);native[region]=str(p)
    code={**prior['code_hashes'],**{str(ROOT/p):sha(ROOT/p) for p in CODE}}
    verify(code)
    env=environment();envhash=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()
    frozen={**config(), 'created_utc':now(),'source':prior['source'],'native_paths':native,
            'code_hashes':code,'input_hashes':{str(p):sha(p) for p in paths},'environment':env,'environment_sha256':envhash}
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'freeze.json',frozen);freeze_sha=sha(OUT/'freeze.json')
    folder=OUT/'packets';folder.mkdir();records={p:[] for p in PACKETS}
    for region in SOURCES:
        for day,a in source_days(Path(frozen['source'])/f'{region}_hourly.csv','2022-','2025-07'):
            for name,values in model.packets_from_day(a).items():
                records[name].append(dict(region=region,day=day.isoformat(),**{f'v{i}':float(v) for i,v in enumerate(values)}))
    hashes={};counts={}
    for name,rows in records.items():
        t=pd.DataFrame(rows).sort_values(['region','day'])
        if t.duplicated(['region','day']).any():raise ValueError('Duplicate packet day')
        p=folder/f'{name}.csv';t.to_csv(p,index=False,mode='x');hashes[str(p)]=sha(p);counts[name]=len(t)
    if len(set(counts.values()))!=1:raise ValueError('Packets must share the complete-day mask')
    verify({**code,**frozen['input_hashes'],str(OUT/'freeze.json'):freeze_sha})
    write(OUT/'data_seal.json',dict(created_utc=now(),freeze_sha256=freeze_sha,packet_hashes=hashes,
        region_days=counts,source_hourly_aggregation_disclosed=True))


def check() -> tuple[dict,dict]:
    frozen,data=read(OUT/'freeze.json'),read(OUT/'data_seal.json')
    # JSON roundtrip normalizes tuple-valued period endpoints.
    if any(frozen[k]!=v for k,v in json.loads(json.dumps(config())).items()):raise ValueError('Changed frozen pilot contract')
    env=environment()
    if frozen['environment']!=env or frozen['environment_sha256']!=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest():
        raise ValueError('Changed execution environment')
    expected={str(OUT/'packets'/f'{p}.csv') for p in PACKETS}
    if data['freeze_sha256']!=sha(OUT/'freeze.json') or set(data['packet_hashes'])!=expected or data['source_hourly_aggregation_disclosed'] is not True:
        raise ValueError('Packet data seal mismatch')
    verify({**frozen['code_hashes'],**frozen['input_hashes'],**data['packet_hashes']})
    return frozen,data


def stream(origin: pd.Timestamp) -> np.ndarray:
    stamp=int(origin.value)
    return np.random.default_rng(np.random.SeedSequence([model.SEED,0,stamp&0xffffffff,(stamp>>32)&0xffffffff])).random((model.SAMPLES,8))[:,1]


def checked_features(histories: dict,packet: str,region: str,lag: int,day: pd.Timestamp) -> tuple:
    """Keep the shared raw-energy anchor identical to D2 before any model use."""
    x,anchor,audit=histories[packet,region,lag].features(day)
    _,reference,_=histories['D2',region,lag].features(day)
    difference=float(abs(anchor-reference))
    if not np.isfinite(difference) or difference>1e-9:
        raise ValueError(f'Common D2 anchor mismatch: {packet}/{region}/{lag}/{day}')
    return x,anchor,audit,difference


def predict() -> None:
    frozen,_=check();parents={str(OUT/n):sha(OUT/n) for n in ('freeze.json','data_seal.json')}
    if (OUT/'predictions').exists() or (OUT/'prediction_seal.json').exists():raise FileExistsError('Preserve prediction attempt')
    folder=OUT/'predictions';folder.mkdir()
    fine={r:source_days(Path(frozen['source'])/f'{r}_hourly.csv','2024-01-25','2025-01-01') for r in SOURCES}
    histories={}
    for packet,n in PACKETS.items():
        t=pd.read_csv(OUT/'packets'/f'{packet}.csv',dtype={f'v{i}':object for i in range(n)})
        for region in SOURCES:
            for lag in LAGS:histories[packet,region,lag]=model.PacketHistory(t.loc[t.region.eq(region)],packet,lag)
    days=pd.date_range('2025-01-01','2025-06-30',tz='UTC');uniforms=np.stack([stream(d) for d in days]);entries=[]
    def dataset(packet: str,lag: int,held: str,period: str) -> tuple:
        xs=[];ys=[];anchors=[];labels=[];dates=[];differences=[];counts={r:0 for r in SOURCES}
        start,stop=[pd.Timestamp(v,tz='UTC') for v in model.PERIODS[period]]
        for region in SOURCES:
            if region==held:continue
            for day,a in fine[region]:
                if not start<=day<stop:continue
                x,anchor,_,difference=checked_features(histories,packet,region,lag,day)
                xs.append(x);anchors.append(anchor);ys.append((a/a.sum(1)[:,None])@model.EF)
                differences.append(difference)
                labels.append(region);dates.append(day);counts[region]+=1
        if len(set(labels))<2:raise ValueError('Each training period needs two actual source regions')
        return np.stack(xs),np.stack(ys),np.asarray(anchors),np.asarray(labels),pd.DatetimeIndex(dates),max(differences),counts
    for held in SOURCES:
        with np.load(frozen['native_paths'][held],allow_pickle=False) as z:
            origins,raw,ids=z['origins'].copy(),z['raw_ensemble'].copy(),z['donor_indices'].copy()
        if not np.array_equal(origins,days.asi8) or raw.shape!=(181,128,24):raise ValueError('Native common forecast support changed')
        p=folder/f'source-{held}-native.npz'
        with p.open('xb') as handle:np.savez_compressed(handle,origins=origins,raw=raw,donor_ids=ids,models=np.array(['native']))
        entries.append(dict(region=held,lag=0,packet='native',path=str(p),sha256=sha(p),models=['native']))
        for lag in LAGS:
            for packet in PACKETS:
                training={period:dataset(packet,lag,held,period) for period in model.PERIODS}
                px,py,pa,pr,pd_,_,_=training['proper'];fx,fy,fa,fr,fd,_,_=training['riskfit'];cx,cy,ca,cr,cd,_,_=training['riskpool']
                query=[checked_features(histories,packet,held,lag,d) for d in days]
                xq,aq=np.stack([v[0] for v in query]),np.asarray([v[1] for v in query])
                points={kind:model.PointModel(kind).fit(px,py,pa,pr,pd_,exclude=held) for kind in ('ridge','ET')}
                raws=[];centers=[];extra={};model_audits={}
                for name in MODELS:
                    point=points['ridge' if name.startswith('ridge') else 'ET']
                    fit_errors=fy-point.predict(fx,fa);pool_errors=cy-point.predict(cx,ca)
                    kind='hetero' if name.endswith('hetero') else 'QRF' if name.endswith('QRF') else 'global'
                    risk=model.RiskModel(kind).fit(fx,fit_errors,fr,fd,cx,pool_errors,cr,cd,exclude=held)
                    center=point.predict(xq,aq);prediction=risk.draw(center,xq,uniforms)
                    raws.append(prediction.pop('raw'));centers.append(center)
                    for k,v in prediction.items():extra[name+'__'+k]=v
                    extra.update({name+'__pool_errors':pool_errors,name+'__pool_regions':cr.astype('U'),
                        name+'__pool_days':cd.as_unit('ns').asi8,name+'__riskfit_days':fd.as_unit('ns').asi8,
                        name+'__riskfit_regions':fr.astype('U')})
                    model_audits[name]=dict(point_regions=list(point.regions),riskfit_regions=list(risk.fit_regions),
                        riskpool_regions=list(risk.regions),empty_leaf_tree_queries=int(prediction['empty_leaf_tree_queries']))
                p=folder/f'source-{held}-{lag}-{packet}.npz'
                with p.open('xb') as handle:np.savez_compressed(handle,origins=origins,models=np.array(MODELS),
                    raw=np.stack(raws),centers=np.stack(centers),features=xq,anchors=aq,residual_uniforms=uniforms,**extra)
                audit=p.with_suffix('.json')
                write(audit,dict(held_region=held,lag=lag,packet=packet,cohorts={k:v[-1] for k,v in training.items()},
                    empty_allowed_regions={k:[r for r in SOURCES if r!=held and v[-1][r]==0] for k,v in training.items()},
                    common_anchor_max_abs_difference={**{k:v[-2] for k,v in training.items()},'query':max(v[3] for v in query)},
                    models=model_audits,query_availability=[v[2] for v in query],source2025fine_labels_read=False))
                entries.append(dict(region=held,lag=lag,packet=packet,path=str(p),sha256=sha(p),
                    models=list(MODELS),audit_path=str(audit),audit_sha256=sha(audit)))
                print('Predicted',held,lag,packet,flush=True)
    check();verify(parents)
    write(OUT/'prediction_seal.json',dict(created_utc=now(),freeze_sha256=parents[str(OUT/'freeze.json')],
        entries=entries,source2025fine_labels_read=False,target_values_opened=False,complete_grid=True))


def check_predictions(seal: dict) -> None:
    expected={(r,0,'native') for r in SOURCES}|{(r,l,p) for r in SOURCES for l in LAGS for p in PACKETS}
    actual=[(e['region'],e['lag'],e['packet']) for e in seal['entries']]
    if (len(actual)!=len(expected) or set(actual)!=expected or seal['complete_grid'] is not True
            or seal['source2025fine_labels_read'] is not False or seal['target_values_opened'] is not False):
        raise ValueError('Exact63prediction grid required before fine evaluation')
    for e in seal['entries']:
        name=f"source-{e['region']}-native.npz" if e['lag']==0 else f"source-{e['region']}-{e['lag']}-{e['packet']}.npz"
        p=(OUT/'predictions'/name).resolve()
        if Path(e['path']).resolve()!=p or e['models']!=(['native'] if e['lag']==0 else list(MODELS)):
            raise ValueError('Unregistered prediction path or model list')
        verify({e['path']:e['sha256']})
        with np.load(e['path'],allow_pickle=False) as a:
            expected_shape=(181,128,24) if e['lag']==0 else (4,181,128,24)
            if (a['raw'].shape!=expected_shape or not np.isfinite(a['raw']).all()
                    or not np.array_equal(a['origins'],pd.date_range('2025-01-01','2025-06-30',tz='UTC').asi8)
                    or a['models'].tolist()!=e['models']):raise ValueError('Invalid prediction array/grid')
            if e['lag']:
                if (a['centers'].shape!=(4,181,24) or a['anchors'].shape!=(181,)
                        or not np.isfinite(a['centers']).all() or not np.isfinite(a['anchors']).all()
                        or a['residual_uniforms'].shape!=(181,128)):
                    raise ValueError('Invalid prediction center/uniforms')
                for name in MODELS:
                    regions,dates=a[name+'__pool_regions'],a[name+'__pool_days']
                    weights=a[name+'__conditional_weights'];ids=a[name+'__residual_indices'];errors=a[name+'__pool_errors']
                    if (len(set(regions))<2 or not set(regions)<=set(SOURCES)-{e['region']}
                            or dates.shape!=(len(regions),) or errors.shape!=(len(regions),24)
                            or np.any(dates<pd.Timestamp('2024-10-01',tz='UTC').value)
                            or np.any(dates>=pd.Timestamp('2025-01-01',tz='UTC').value)
                            or weights.shape!=(181,len(regions)) or not np.isfinite(weights).all() or np.any(weights<0)
                            or not np.allclose(weights.sum(1),1.,rtol=0,atol=1e-12)
                            or ids.shape!=(181,128) or not np.issubdtype(ids.dtype,np.integer)
                            or np.any(ids<0) or np.any(ids>=len(regions)) or not np.isfinite(errors).all()):
                        raise ValueError('Invalid source-only out-of-sample riskpool')
        if e['lag']:
            if Path(e['audit_path']).resolve()!=p.with_suffix('.json'):raise ValueError('Unregistered model audit path')
            verify({e['audit_path']:e['audit_sha256']})


def selection(regional: pd.DataFrame) -> dict:
    keys=['lag','packet','model','factor'];macro=regional.groupby(keys)[list(METRICS)].mean()
    macro['worst_coverage']=regional.groupby(keys).empirical_coverage90.min();macro=macro.reset_index()
    macro['violation']=np.maximum(.9-macro.empirical_coverage90,0)+np.maximum(macro.empirical_coverage90-.95,0)+np.maximum(.85-macro.worst_coverage,0)
    def choose(packet: str) -> dict:
        part=macro[(macro.packet==packet)&macro.lag.eq(0 if packet=='native' else 7)]
        passed=part[part.violation<=1e-12];pool=passed if len(passed) else part
        row=pool.sort_values(([] if len(passed) else ['violation'])+['crps','model','factor']).iloc[0]
        return {**row.to_dict(),'passed':bool(row.violation<=1e-12)}
    picked={p:choose(p) for p in ('native',*PACKETS)}
    main=picked['T8'];refs=[picked[p] for p in ('D2','D8','native')]
    gate=bool(main['passed'] and all(main['crps']<=.95*r['crps'] and main['empirical_width90']<=r['empirical_width90']
                and main['empirical_interval_score90']<r['empirical_interval_score90'] for r in refs))
    control=macro[macro.lag.eq(7)&macro.packet.eq('G5')&macro.model.eq(main['model'])&macro.factor.eq(main['factor'])]
    if len(control)!=1:raise ValueError('Matched G5 control missing')
    g5=control.iloc[0]
    timing=bool(main['passed'] and main['crps']<g5.crps and main['empirical_width90']<=g5.empirical_width90
                and main['empirical_interval_score90']<g5.empirical_interval_score90)
    return dict(created_utc=now(),selections=picked,T8_same_model_same_factor_G5=control.iloc[0].to_dict(),
        source_information_gate=gate,H_timing_gain_detected=timing,target_test_authorized=False,mechanism_breakthrough=False,
        independent_confirmation=False,quantization_noise_phase_stress_completed=False)


def score() -> None:
    if (OUT/'source_completed.json').exists() or (OUT/'day_metrics.csv').exists():raise FileExistsError('Preserve completed/partial scores')
    frozen,_=check();seal=read(OUT/'prediction_seal.json')
    if seal['freeze_sha256']!=sha(OUT/'freeze.json'):raise ValueError('Prediction freeze binding changed')
    check_predictions(seal);parents={str(OUT/n):sha(OUT/n) for n in ('freeze.json','data_seal.json','prediction_seal.json')};rows=[]
    for region in SOURCES:
        observed=source_days(Path(frozen['source'])/f'{region}_hourly.csv','2025-01-01','2025-07-01')
        lookup={d.value:(a/a.sum(1)[:,None])@model.EF for d,a in observed}
        for e in (e for e in seal['entries'] if e['region']==region):
            with np.load(e['path'],allow_pickle=False) as z:
                origins=z['origins'];keep=np.isin(origins,list(lookup));y=np.stack([lookup[t] for t in origins[keep]])
                arrays=z['raw'][None] if e['lag']==0 else z['raw']
                for name,raw in zip(e['models'],arrays):
                    paths=np.clip(raw[keep],0,1000);point=paths.mean(1)
                    for factor in FACTORS:
                        scaled=np.clip(point[:,None]+factor*(paths-point[:,None]),0,1000);metrics=score_arrays(point,scaled,y)
                        rows.extend(dict(region=region,origin=pd.Timestamp(int(t),tz='UTC').isoformat(),lag=e['lag'],
                            packet=e['packet'],model=name,factor=factor,**{k:float(v[i]) for k,v in metrics.items() if k in METRICS})
                            for i,t in enumerate(origins[keep]))
    day=pd.DataFrame(rows);keys=['region','lag','packet','model','factor']
    if len(day[['region','origin']].drop_duplicates())!=1145 or day.duplicated(keys+['origin']).any() or not np.isfinite(day[list(METRICS)]).all().all():
        raise ValueError('Changed common fine-evaluation cohort')
    for _,part in day.groupby(keys[1:]):
        if len(part)!=1145:raise ValueError('Each configuration must have all1145regiondays')
    regional=day.groupby(keys)[list(METRICS)].mean().reset_index();macro=regional.groupby(keys[1:])[list(METRICS)].mean().reset_index()
    outputs=[]
    for n,t in [('day_metrics',day),('region_metrics',regional),('macro_metrics',macro)]:
        p=OUT/f'{n}.csv';t.to_csv(p,index=False,mode='x');outputs.append(p)
    write(OUT/'source_choice.json',selection(regional));outputs.append(OUT/'source_choice.json')
    check();check_predictions(seal);verify(parents)
    write(OUT/'source_completed.json',dict(created_utc=now(),version=17,freeze_sha256=parents[str(OUT/'freeze.json')],
        prediction_seal_sha256=parents[str(OUT/'prediction_seal.json')],output_hashes={str(p):sha(p) for p in outputs},
        target_values_opened=False,unseen_values_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','predict','score'),required=True)
    {'prepare':prepare,'predict':predict,'score':score}[p.parse_args().stage]()
