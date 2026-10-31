"""Six-fold fixed-July-origin zero-risk experiment on the frozen H2 roster."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extended_monthly_information import ExtendedMonthlyInformation
from forward_zero_risk import ForwardZeroRisk, METHODS, MONTH_METHODS, COMPONENTS, check_inputs
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify, FUELS
from v171_two_scale_zero import aggregate,score_query,METRICS

ROOT=Path(__file__).resolve().parents[1]
INV=ROOT/'results/v188-forward-risk-inventory-r1'
PRE=ROOT/'results/v188-forward-zero-risk-preflight-r1'
OUT=ROOT/'results/v188-forward-zero-risk-r1'
KINDS={'hourly':METHODS,'monthly':MONTH_METHODS,'components':COMPONENTS}


def checked_hashes():
    upstream=verify(INV)
    hashes=upstream['input_hashes'].copy()
    paths=[INV/'summary.json',*[INV/k for k in upstream['output_hashes']],Path(__file__),
           ROOT/'scripts/forward_zero_risk.py',ROOT/'scripts/monthly_risk_features.py',
           ROOT/'scripts/monthly_history_zero_risk.py',ROOT/'scripts/two_scale_zero.py',
           ROOT/'scripts/hourly_fuel_data.py',ROOT/'scripts/v171_two_scale_zero.py',
           ROOT/'scripts/v154_hourly_strong_baselines.py',ROOT/'scripts/v170_fuel_marginal_diagnosis.py',
           ROOT/'scripts/v151_hourly_law.py',ROOT/'results/v144-canonical-profile-cache-r1/profiles.npz']
    for p in paths:
        key=str(p.relative_to(ROOT));value=sha(p)
        if key in hashes and hashes[key]!=value:raise ValueError('upstream hash conflict')
        hashes[key]=value
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('input changed')
    return hashes


def make_predictor(data,short,long,fold):
    meta=json.loads((INV/'summary.json').read_text())
    return ForwardZeroRisk(data,short,long,fold,pd.read_csv(INV/'source_inventory.csv'),
                           json.loads((INV/'selections.json').read_text()),pd.read_csv(INV/'queries.csv'),
                           pd.Timestamp(meta['cutoff']))


def aggregate_periods(rows):
    regional,macro=[],[]
    for period,table in [('H2',rows),('Q3',rows.loc[rows.period.eq('Q3')]),('Q4',rows.loc[rows.period.eq('Q4')])]:
        r,m=aggregate(table)
        for frame in (r,m):frame['period']=period;frame['cohort']=frame.cohort.replace({'all29':'all_available'})
        regional.append(r);macro.append(m)
    return pd.concat(regional,ignore_index=True),pd.concat(macro,ignore_index=True)


def provenance(model):
    return {'proper_regions':model.fit,'fine_regions':model.fine_fit,
            'source_sample_count':len(model.raw_ids),'monthly_training_profiles':len(model.month_ids),
            'positive_fuel_training':{fuel:{'hours':int(model.positive[:,f].sum()),
                'regions':sorted(set(model.regions[model.positive[:,f]]))} for f,fuel in enumerate(FUELS)}}


def run_fold(data,short,long,fold,destination,manifest_sha,preflight=False):
    model=make_predictor(data,short,long,fold)
    for q in model.queries.profile_id.iloc[[0,-1]]:check_inputs(model,int(q))
    fingerprint=model.fingerprint()
    # Exercise the identical persisted prefix schema without reading held truth.
    prefix=pd.DataFrame(model.prefixes())
    roundtrip=pd.read_csv(io.StringIO(prefix.to_csv(index=False)))
    if prefix.to_csv(index=False)!=roundtrip.to_csv(index=False):raise ValueError('prefix CSV roundtrip')
    if preflight:return fingerprint,provenance(model)
    if fingerprint!=json.loads((PRE/'model_fingerprints.json').read_text())[str(fold)]:raise ValueError('preflight fit changed')
    saved={k:[] for k in KINDS};rows={k:[] for k in KINDS};contexts=[]
    # Construct every forecast before any held query outcome read.
    predictions=[]
    for rec in model.queries.itertuples(index=False):
        outputs=model.predict(int(rec.profile_id));predictions.append((rec,outputs[:3]))
        for kind,values in zip(KINDS,outputs[:3]):saved[kind].append(np.stack([values[m] for m in KINDS[kind]]))
        c=outputs[3];contexts.append({'profile_id':int(rec.profile_id),'local':int(c['local']),
                                    'prefix':[int(p) for p in c['prefix']]})
    for rec,outputs in predictions:
        scored=score_query(data,rec,*outputs)
        for kind in KINDS:rows[kind].extend({'fold':fold,'period':rec.period}|r for r in scored[kind])
    frames={k:pd.DataFrame(v) for k,v in rows.items()}
    for kind,frame in frames.items():
        if frame.duplicated(['profile_id','method','fuel']).any() or not np.isfinite(frame[list(METRICS)]).all().all():raise ValueError('score finite/keys')
        if kind!='components' and len(frame)!=len(model.queries)*len(KINDS[kind])*8:raise ValueError('score denominator')
    staging=Path(tempfile.mkdtemp(prefix='.v188-forward-',dir=OUT.parent))
    for kind,frame in frames.items():frame.to_csv(staging/f'{kind}_scores.csv',index=False)
    np.savez_compressed(staging/'probabilities.npz',profile_id=model.queries.profile_id.to_numpy(int),**{k:np.asarray(v) for k,v in saved.items()})
    np.savez_compressed(staging/'training_matrices.npz',**model.arrays())
    prefix.to_csv(staging/'training_prefix.csv',index=False)
    model.queries.to_csv(staging/'queries.csv',index=False)
    for name,value in [('model_fingerprints',fingerprint),('contexts',contexts),('training_provenance',provenance(model))]:
        (staging/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    result={'assertion_failures':0,'fold':fold,'manifest_sha256':manifest_sha,'forests_fitted':56,
            **provenance(model),'score_rows':{k:len(v) for k,v in frames.items()},
            'output_hashes':{p.name:sha(p) for p in staging.iterdir()}}
    (staging/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');staging.rename(destination)
    print(f'V188 fold {fold} complete: {result["score_rows"]}',flush=True)
    return fingerprint,provenance(model)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    destination=PRE if args.preflight else OUT
    if destination.exists() and (args.preflight or not args.resume):raise FileExistsError(destination)
    hashes=checked_hashes()
    data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT)
    long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    if args.preflight:
        fingerprints={};coverage={}
        with threadpool_limits(limits=2):
            for fold in range(6):
                fingerprints[str(fold)],coverage[str(fold)]=run_fold(data,short,long,fold,destination,'',True)
                print(f'V188 preflight fold {fold} complete',flush=True)
        if checked_hashes()!=hashes:raise ValueError('preflight input changed')
        destination.mkdir()
        for name,value in [('model_fingerprints',fingerprints),('training_provenance',coverage)]:
            (destination/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        result={'assertion_failures':0,'forests_fitted':336,'future_query_raw_guards':12,'held_outcomes_scored':False,
                'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in destination.iterdir()}}
    else:
        if verify(PRE)['input_hashes']!=hashes:raise ValueError('preflight dependencies differ')
        for name in ('summary.json','model_fingerprints.json','training_provenance.json'):hashes[str((PRE/name).relative_to(ROOT))]=sha(PRE/name)
        text=json.dumps({'input_hashes':hashes,'methods':KINDS,'cutoff':'2024-07-01','query_count':158},indent=2,sort_keys=True)+'\n'
        if destination.exists():
            if (destination/'run_manifest.json').read_text()!=text or (destination/'summary.json').exists():raise ValueError('resume manifest/completed')
        else:destination.mkdir();(destination/'run_manifest.json').write_text(text)
        with threadpool_limits(limits=2):
            for fold in range(6):
                block=destination/f'fold_{fold}'
                if block.exists():
                    if verify(block)['manifest_sha256']!=sha(destination/'run_manifest.json'):raise ValueError('block manifest')
                else:run_fold(data,short,long,fold,block,sha(destination/'run_manifest.json'))
        counts={}
        for kind in KINDS:
            rows=pd.concat([pd.read_csv(destination/f'fold_{f}/{kind}_scores.csv') for f in range(6)],ignore_index=True)
            if rows.duplicated(['profile_id','method','fuel']).any():raise ValueError('root duplicates')
            regional,macro=aggregate_periods(rows)
            for name,frame in [('scores',rows),('region_metrics',regional),('macro_metrics',macro)]:frame.to_csv(destination/f'{kind}_{name}.csv',index=False)
            counts[kind]=len(rows)
        if counts['hourly']!=158*14*8 or counts['monthly']!=158*4*8:raise ValueError('root denominator')
        if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('formal inputs changed')
        result={'assertion_failures':0,'forests_fitted':336,'query_forecasts':158,'score_rows':counts,
                'input_hashes':hashes,'output_hashes':{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},
                'scope':'Previously viewed data, constrained within-2024 forward split; not online-vintage confirmation or CIF performance.'}
    (destination/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
