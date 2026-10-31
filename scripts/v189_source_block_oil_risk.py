"""Pre-registered source-block deletion stress test of V188's oil-risk exception."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from source_block_oil_risk import SourceBlockOil,OIL,HOURLY,COMPONENTS,guard
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v171_two_scale_zero import score_probability,aggregate,METRICS
from v188_forward_zero_risk import KINDS as OLD_KINDS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v188-forward-zero-risk-r1'
AUDIT=ROOT/'results/v188-forward-zero-risk-audit-r1'
PRE=ROOT/'results/v189-source-block-oil-preflight-r2'
OUT=ROOT/'results/v189-source-block-oil-r2'
BASELINE='__baseline__'
SCORE_COLUMNS=('fold','period','profile_id','region','month','hours','fuel','scenario',
               'input_fuel_positive','current_fuel_positive','affected','method',*METRICS)


def score_frame(rows):
    return pd.DataFrame(rows,columns=SCORE_COLUMNS)


def empty_schema_contract():
    import io
    frame=score_frame([])
    restored=pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    if list(restored.columns)!=list(SCORE_COLUMNS) or len(restored) or not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError('empty conditional score schema')


def dependencies():
    hashes={}
    for folder in (BASE,AUDIT):
        meta=verify(folder)
        for k,v in meta['input_hashes'].items():
            if k in hashes and hashes[k]!=v:raise ValueError('upstream hashes conflict')
            hashes[k]=v
        for k in ('summary.json',*meta['output_hashes']):hashes[str((folder/k).relative_to(ROOT))]=sha(folder/k)
    for path in (Path(__file__),ROOT/'scripts/source_block_oil_risk.py',ROOT/'docs/V189_SOURCE_BLOCK_OIL_PROTOCOL.md',ROOT/'docs/V189_EMPTY_CONDITIONAL_ADDENDUM.md'):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('dependency changed')
    return hashes


def roster():
    frame=pd.read_csv(ROOT/'results/v159-fuel-activity-source-pilot-r1/source_sampling.csv')
    result=sorted(set(frame.region))
    if len(result)!=29 or BASELINE in result:raise ValueError('fixed source deletion roster')
    return [BASELINE,*result]


def oil_scores(data,rec,scenario,hourly,components,affected):
    q=int(rec.profile_id);row=data.index.iloc[q];states=data.states[int(row.offset):int(row.stop)]
    y=data.raw[int(row.offset):int(row.stop),OIL]==0;local=data.local(q,'asof_30d')
    identity={'fold':int(rec.fold),'period':rec.period,'profile_id':q,'region':rec.region,'month':rec.month,
              'hours':len(states),'fuel':'oil','scenario':scenario,'input_fuel_positive':bool(data.M[local,OIL]>0),
              'current_fuel_positive':bool(data.M[q,OIL]>0),'affected':bool(affected)}
    out={'hourly':[identity|{'method':m,**score_probability(p[states],y)} for m,p in hourly.items()],
         'components':[identity|{'method':m,**score_probability(p[states],y)} for m,p in components.items()] if data.M[q,OIL]>0 else []}
    return out


def aggregates(rows):
    regional=[];macro=[]
    for scenario,group in rows.groupby('scenario',sort=True):
        for period,t in [('H2',group),('Q3',group.loc[group.period.eq('Q3')]),('Q4',group.loc[group.period.eq('Q4')])]:
            r,m=aggregate(t)
            for frame in (r,m):frame['scenario']=scenario;frame['period']=period;frame['cohort']=frame.cohort.replace({'all29':'all_available'})
            regional.append(r);macro.append(m)
    return pd.concat(regional,ignore_index=True),pd.concat(macro,ignore_index=True)


def load_fold(fold):
    folder=BASE/f'fold_{fold}'
    with np.load(folder/'training_matrices.npz',allow_pickle=False) as packed:arrays={k:packed[k].copy() for k in packed.files}
    with np.load(folder/'probabilities.npz',allow_pickle=False) as packed:old={k:packed[k].copy() for k in packed.files}
    proper=json.loads((folder/'training_provenance.json').read_text())['proper_regions']
    return arrays,pd.read_csv(folder/'queries.csv'),old,proper


def build_fold(data,short,long,fold,preflight=False):
    arrays,queries,old,proper=load_fold(fold);fits={};proof={};predicted={}
    for deleted in (BASELINE,*proper):
        model=SourceBlockOil(arrays,data.states,deleted)
        for q in queries.profile_id.iloc[[0,-1]]:guard(model,data,short,long,int(q))
        fits[deleted]=model.fingerprint();proof[deleted]=model.provenance()
        outputs=[model.predict(short,long,int(q))[:2] for q in queries.profile_id]
        predicted[deleted]={kind:np.asarray([np.stack([values[i][m] for m in names]) for values in outputs]) for i,(kind,names) in enumerate([('hourly',HOURLY),('components',COMPONENTS)])}
        if deleted==BASELINE:
            for kind,names in [('hourly',HOURLY),('components',COMPONENTS)]:
                expected=np.stack([old[kind][:,list(OLD_KINDS[kind]).index(m),:,OIL] for m in names],axis=1)
                if not np.array_equal(predicted[deleted][kind],expected):raise ValueError('V188 oil probabilities not bit-exact')
        if deleted!=BASELINE and not np.any(arrays['region']==deleted):
            if any(not np.array_equal(a,predicted[BASELINE][k]) for k,a in predicted[deleted].items()):raise ValueError('empty-source placebo changed probability')
    if preflight:return fits,proof
    expected=json.loads((PRE/'model_fingerprints.json').read_text())[str(fold)]
    if fits!=expected:raise ValueError('preflight model fingerprint changed')
    # Populate all global scenarios before accessing any query outcome.
    scenarios=roster();probabilities={k:np.stack([predicted[s if s in proper else BASELINE][k] for s in scenarios]) for k in ('hourly','components')}
    rows={k:[] for k in probabilities}
    for si,scenario in enumerate(scenarios):
        for j,rec in enumerate(queries.itertuples(index=False)):
            hourly={m:probabilities['hourly'][si,j,k] for k,m in enumerate(HOURLY)}
            components={m:probabilities['components'][si,j,k] for k,m in enumerate(COMPONENTS)}
            scored=oil_scores(data,rec,scenario,hourly,components,scenario in proper and bool(np.any(arrays['region']==scenario)))
            for kind,v in scored.items():rows[kind].extend(v)
    return fits,proof,queries,probabilities,{k:score_frame(v) for k,v in rows.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    dest=PRE if args.preflight else OUT
    if dest.exists() and (args.preflight or not args.resume):raise FileExistsError(dest)
    empty_schema_contract()
    hashes=dependencies();data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT)
    long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    if args.preflight:
        fits={};proof={}
        with threadpool_limits(limits=2):
            for fold in range(6):
                fits[str(fold)],proof[str(fold)]=build_fold(data,short,long,fold,True)
                print(f'V189 preflight fold {fold} complete',flush=True)
        if dependencies()!=hashes:raise ValueError('preflight input changed')
        dest.mkdir()
        for name,value in [('model_fingerprints',fits),('training_provenance',proof)]:
            (dest/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        result={'assertion_failures':0,'fit_blocks':122,'forests_fitted':488,'future_query_raw_guards':244,'held_outcomes_scored':False,
                'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in dest.iterdir()}}
    else:
        if verify(PRE)['input_hashes']!=hashes:raise ValueError('preflight dependencies changed')
        for name in ('summary.json','model_fingerprints.json','training_provenance.json'):hashes[str((PRE/name).relative_to(ROOT))]=sha(PRE/name)
        manifest=json.dumps({'input_hashes':hashes,'scenarios':roster(),'hourly_methods':HOURLY,'components':COMPONENTS},indent=2,sort_keys=True)+'\n'
        if dest.exists():
            if (dest/'run_manifest.json').read_text()!=manifest or (dest/'summary.json').exists():raise ValueError('resume manifest/completed')
        else:dest.mkdir();(dest/'run_manifest.json').write_text(manifest)
        with threadpool_limits(limits=2):
            for fold in range(6):
                block=dest/f'fold_{fold}'
                if block.exists():
                    if verify(block)['manifest_sha256']!=sha(dest/'run_manifest.json'):raise ValueError('saved block manifest')
                    continue
                fits,proof,queries,probabilities,frames=build_fold(data,short,long,fold)
                staging=Path(tempfile.mkdtemp(prefix='.v189-source-block-',dir=OUT.parent))
                for name,value in [('model_fingerprints',fits),('training_provenance',proof)]:
                    (staging/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
                np.savez_compressed(staging/'probabilities.npz',profile_id=queries.profile_id.to_numpy(int),scenario=np.asarray(roster(),str),**probabilities)
                queries.to_csv(staging/'queries.csv',index=False)
                for k,frame in frames.items():
                    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all() or frame.duplicated(['scenario','profile_id','method']).any():raise ValueError('score finite/keys')
                    frame.to_csv(staging/f'{k}_scores.csv',index=False)
                meta={'assertion_failures':0,'fold':fold,'manifest_sha256':sha(dest/'run_manifest.json'),'fit_blocks':len(fits),'output_hashes':{p.name:sha(p) for p in staging.iterdir()}}
                (staging/'summary.json').write_text(json.dumps(meta,indent=2)+'\n');staging.rename(block)
                print(f'V189 formal fold {fold} complete',flush=True)
        counts={}
        for kind,expected in [('hourly',23700),('components',5490)]:
            rows=pd.concat([pd.read_csv(dest/f'fold_{f}/{kind}_scores.csv') for f in range(6)],ignore_index=True)
            if len(rows)!=expected or rows.duplicated(['scenario','profile_id','method']).any():raise ValueError('root denominator/keys')
            r,m=aggregates(rows)
            for name,frame in [('scores',rows),('region_metrics',r),('macro_metrics',m)]:frame.to_csv(dest/f'{kind}_{name}.csv',index=False)
            counts[kind]=len(rows)
        if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('formal input changed')
        result={'assertion_failures':0,'fit_blocks':122,'forests_fitted':488,'scenario_query_forecasts':4740,'score_rows':counts,
                'input_hashes':hashes,'output_hashes':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()},
                'scope':'Targeted post-V188 development stress test, not new independent data, causal attribution or CIF improvement.'}
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
