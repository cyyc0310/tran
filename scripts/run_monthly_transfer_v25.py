"""Source-only monthly-budget controls: immutable prepare/predict/score stages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import monthly_transfer_v25 as model
from scripts import run_local_sketch_v17 as base
from scripts.native_center_v18 import scaled_paths

OUT = ROOT/'results/monthly-transfer-v25'
MANIFEST = ROOT/'docs/evidence/native-hindcast-v20r1-verification.json'
PARENT = ROOT/'results/native-hindcast-v20r1'
PARENT_SHA = '7a39eb59d98ce3323e443c49e3ead987813be855f49d3c21865155d71beaec8f'
CODE = ('scripts/monthly_transfer_v25.py', 'scripts/run_monthly_transfer_v25.py',
        'tests/test_monthly_transfer_v25.py', 'tests/test_run_monthly_transfer_v25.py',
        'docs/MONTHLY_TRANSFER_V25_PROTOCOL.md')
REVIEWS = tuple(ROOT/f'results/monthly-transfer-v25-{r}-review.md' for r in ('code','python','science'))
DAYS = pd.date_range('2025-01-01', '2025-06-30', tz='UTC')
FACTORS = (.75, 1., 1.25, 1.5, 2.)
METRICS = base.METRICS
sha, read, write, verify, now = base.sha, base.read, base.write, base.verify, base.now
OUTPUTS = ('day_metrics.csv','region_metrics.csv','macro_metrics.csv','matched_contrasts.csv','source_choice.json')


def absolute(mapping: dict) -> dict:
    return {str(Path(p) if Path(p).is_absolute() else ROOT/p): h for p,h in mapping.items()}


def parent() -> tuple[dict, dict, dict]:
    if sha(MANIFEST) != PARENT_SHA:
        raise ValueError('Audited R1 final manifest digest required')
    evidence = read(MANIFEST)
    if evidence.get('numerical_audit_status') != 'PASS':
        raise ValueError('R1 independent numerical PASS required')
    hashes = {}
    for key in ('frozen_code_hashes','frozen_input_hashes','artifact_hashes'):
        hashes.update(absolute(evidence[key]))
    verify(hashes)
    frozen, seal, done = (read(PARENT/n) for n in ('freeze.json','verification_seal.json','completed.json'))
    if (seal['entries'] != frozen['entries'] or seal['banks'] != frozen['banks']
            or done['verification_seal_sha256'] != sha(PARENT/'verification_seal.json')
            or any(x['freeze_sha256'] != sha(PARENT/'freeze.json') for x in (seal,done))):
        raise ValueError('R1 aliases and completed receipt required')
    return frozen, absolute(evidence['frozen_code_hashes']), {**hashes, str(MANIFEST): PARENT_SHA}


def expected_contract() -> dict:
    prior, code, inputs = parent()
    if any(not p.is_file() or not p.read_text().strip() for p in REVIEWS):
        raise ValueError('Three independent prelaunch reviews required')
    for path in REVIEWS:
        inputs[str(path)] = sha(path)
    bank_lookup = {tuple(e['excluded']): e['path'] for e in prior['banks']}
    queries = {e['outer']:e['path'] for e in prior['entries'] if e['kind']=='query_copy'}
    if set(queries) != set(model.SOURCES) or len(bank_lookup) != 21:
        raise ValueError('Exact R1 banks/query registry')
    bank_paths = {r:[bank_lookup[tuple(sorted((r,s)))] for s in [q for q in model.SOURCES if q!=r][:2]] for r in model.SOURCES}
    for name in CODE:
        code[str(ROOT/name)] = sha(ROOT/name)
    monthly = str(Path(prior['source'])/'monthly.csv')
    if monthly not in inputs or any(p not in inputs for p in queries.values()):
        raise ValueError('Parent must bind source month and query aliases')
    for paths in bank_paths.values():
        if any(p not in inputs for p in paths):
            raise ValueError('Parent must bind all bank aliases')
    return dict(version=25,source=prior['source'],
        sources=list(model.SOURCES),arms=list(model.ARMS),factors=list(FACTORS),samples=128,seed=17023,
        code_hashes=code,input_hashes=inputs,environment=base.environment(),
        banks_by_outer=bank_paths,query_paths=queries,monthly_path=monthly,
        target_values_opened=False,unseen_values_opened=False,custom_method=False,
        prediction_reads_new_fine=False,score_reads_source_2025H1_only_after_complete_seal=True,
        publication_vintages_verified=False,annual_only_evaluated=False)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve partial or completed V25 output')
    contract=expected_contract()
    OUT.mkdir()
    write(OUT/'freeze.json',dict(created_utc=now(),**contract))
    print('Prepared',len(contract['code_hashes']),'code',len(contract['input_hashes']),'input hashes',flush=True)


def check() -> dict:
    frozen = read(OUT/'freeze.json')
    expected=expected_contract()
    if (set(frozen) != set(expected)|{'created_utc'}
            or any(frozen[k] != v for k,v in expected.items())
            or any(frozen[k] is not v for k,v in expected.items() if isinstance(v,bool))):
        raise ValueError('Frozen source, aliases, closure or experiment contract changed')
    return frozen


def load(path: str|Path) -> dict:
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy(order='K') for k in z.files}


def monthly_prefix(path: Path) -> pd.DataFrame:
    selected = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['region'] not in model.SOURCES:
                raise ValueError('Only source monthly rows allowed')
            end, available = pd.Timestamp(row['period_end']), pd.Timestamp(row['available_at'])
            if end.tzinfo is None or available.tzinfo is None:
                raise ValueError('Explicit coarse timezone required')
            if end <= DAYS[-1] and available <= DAYS[-1]:
                selected.append({k:row[k] for k in ('region','period_start','period_end','available_at',*model.M8_COLUMNS)})
    frame = pd.DataFrame(selected)
    for k in ('period_start','period_end','available_at'):
        frame[k] = pd.to_datetime(frame[k],utc=True,errors='raise')
    frame[list(model.M8_COLUMNS)] = frame[list(model.M8_COLUMNS)].astype(float)
    return frame


def predict() -> None:
    frozen = check(); initial = sha(OUT/'freeze.json')
    if (OUT/'predictions').exists() or (OUT/'prediction_seal.json').exists():
        raise FileExistsError('Preserve V25 prediction attempt')
    folder=OUT/'predictions'; folder.mkdir()
    monthly = monthly_prefix(Path(frozen['monthly_path']))
    visible = {b:model.group_monthly(monthly,b) for b in ('M8','MG')}
    entries = []
    for region in model.SOURCES:
        old=load(frozen['query_paths'][region])
        if not np.array_equal(old['origins'],DAYS.asi8):
            raise ValueError('Exact181 original query dates required')
        paths=frozen['banks_by_outer'][region]
        bank=model.union_bank(load(paths[0]),load(paths[1]),old,outer=region)
        laws={b:model.coarse_law(visible[b],region,DAYS,budget=b) for b in ('M8','MG')}
        result=model.predict(bank,laws,DAYS)
        native=result['arms']['M8-recompose_conditional']
        error=float(np.max(abs(native['raw']-old['raw'])))
        if not np.array_equal(native['donor_ids'],old['donor_ids']) or error>1e-9:
            write(OUT/f'native-replay-failure-{region}.json',dict(max_abs_error=error,
                donor_ids_equal=bool(np.array_equal(native['donor_ids'],old['donor_ids']))))
            raise ValueError('M8 strong native identity failed before any seal')
        arrays=dict(origins=DAYS.asi8,arms=np.array(model.ARMS),
            bank_regions=bank['regions'],bank_days=bank['days'],bank_row_mapping=bank['row_mapping'])
        for key in ('raw','donor_ids','fallback_ids','permutation','fine_draws'):
            arrays[key]=np.stack([result['arms'][a][key] for a in model.ARMS])
        for b in ('M8','MG'):
            for key in ('mix','draws','metadata'):
                arrays[b+'_'+key]=laws[b][key]
        path=folder/f'{region}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,**arrays)
        audit_path=folder/f'{region}.json'
        write(audit_path,dict(region=region,outer_excluded=True,bank_rows=len(bank['days']),
            bank_paths=paths,query_path=frozen['query_paths'][region],native_max_abs_error=error,
            native_ids_exact=True,forecast_audits={b:laws[b]['audit'] for b in laws},**result['audit']))
        entries.append(dict(region=region,path=str(path),sha256=sha(path),audit_path=str(audit_path),audit_sha256=sha(audit_path)))
        print('Predicted',region,'native_error',error,flush=True)
        del result,arrays,bank,old
    check(); verify({str(OUT/'freeze.json'):initial})
    write(OUT/'prediction_seal.json',dict(created_utc=now(),freeze_sha256=initial,entries=entries,
        complete_grid=True,archives=7,raw_arms=9,prediction_reads_new_fine=False,
        target_values_opened=False,unseen_values_opened=False))


def predictions() -> dict:
    seal=read(OUT/'prediction_seal.json')
    if (seal['freeze_sha256'] != sha(OUT/'freeze.json') or seal['complete_grid'] is not True
            or seal['archives'] != 7 or seal['raw_arms'] != 9
            or any(seal[k] is not False for k in ('prediction_reads_new_fine','target_values_opened','unseen_values_opened'))
            or [e['region'] for e in seal['entries']] != list(model.SOURCES)):
        raise ValueError('Complete seven-region seal required before source truth')
    for e in seal['entries']:
        if (Path(e['path']).resolve() != (OUT/'predictions'/f"{e['region']}.npz").resolve()
                or Path(e['audit_path']).resolve() != (OUT/'predictions'/f"{e['region']}.json").resolve()):
            raise ValueError('Registered archive and audit paths required')
        verify({e['path']:e['sha256'],e['audit_path']:e['audit_sha256']})
        a=load(e['path'])
        if (a['arms'].tolist()!=list(model.ARMS) or not np.array_equal(a['origins'],DAYS.asi8)
                or a['raw'].shape!=(9,181,128,24) or not np.isfinite(a['raw']).all()
                or a['fine_draws'].shape!=(9,181,128,8)):
            raise ValueError('Complete nine-arm source archive required')
    return seal


def summarize(regional: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    macro=regional.groupby(['arm','factor'])[list(METRICS)].mean().reset_index()
    worst=regional.groupby(['arm','factor']).empirical_coverage90.min()
    macro=macro.merge(worst.rename('worst_region_coverage90'),on=['arm','factor'],validate='one_to_one')
    macro['qualified']=macro.empirical_coverage90.between(.9,.95)&macro.worst_region_coverage90.ge(.85)
    choices={}
    for arm,table in macro.groupby('arm'):
        table=table.copy()
        table['violation']=(.9-table.empirical_coverage90).clip(lower=0)+(table.empirical_coverage90-.95).clip(lower=0)+(.85-table.worst_region_coverage90).clip(lower=0)
        good=table.loc[table.qualified]
        selected=(good.sort_values(['crps','factor']) if len(good) else table.sort_values(['violation','crps','factor'])).iloc[0]
        choices[arm]={k:(bool(v) if isinstance(v,(bool,np.bool_)) else float(v)) for k,v in selected.items() if k!='arm'}
    return macro,dict(choices=choices,custom_method=False,mechanism_breakthrough=False,
        new_holdout_authorized=False,full_probabilistic_controls_completed=False)


def score() -> None:
    if any((OUT/n).exists() for n in (*OUTPUTS,'source_completed.json')):
        raise FileExistsError('Preserve V25 score attempt')
    frozen=check(); seal=predictions()
    start={str(OUT/n):sha(OUT/n) for n in ('freeze.json','prediction_seal.json')}
    rows=[]
    for e in seal['entries']:
        region=e['region']; a=load(e['path'])
        truth={d.value:(g/g.sum(1)[:,None])@model.EF for d,g in base.source_days(Path(frozen['source'])/f'{region}_hourly.csv','2025-01-01','2025-07-01')}
        keep=np.isin(a['origins'],list(truth)); origins=a['origins'][keep]
        y=np.stack([truth[int(d)] for d in origins])
        for i,arm in enumerate(model.ARMS):
            for factor in FACTORS:
                point,paths,clip=scaled_paths(a['raw'][i,keep],factor)
                scores=base.score_arrays(point,paths,y)
                rows.extend(dict(region=region,origin=pd.Timestamp(int(d),tz='UTC').isoformat(),arm=arm,factor=factor,
                    final_spread_clipping_rate=float(clip[j]),**{k:float(scores[k][j]) for k in METRICS}) for j,d in enumerate(origins))
        print('Scored',region,flush=True)
    day=pd.DataFrame(rows)
    if (len(day)!=51525 or len(day[['region','origin']].drop_duplicates())!=1145
            or day.duplicated(['region','origin','arm','factor']).any()
            or not np.isfinite(day.select_dtypes('number')).all().all()):
        raise ValueError('Exact1145-day/45-curve source cohort required')
    regional=day.groupby(['region','arm','factor'])[list(METRICS)].mean().reset_index()
    macro,choice=summarize(regional)
    contrasts=[]
    comparisons=[('MG-recompose_conditional',b) for b in ('MG-anchor_conditional','MG-recompose_global','MG-conditional_shuffled','MG-conditional_plugin_split')]+[('M8-recompose_conditional',b) for b in ('M8-anchor_conditional','M8-recompose_global','M8-conditional_shuffled')]
    for candidate,reference in comparisons:
        left=regional[regional.arm.eq(candidate)]; right=regional[regional.arm.eq(reference)]
        joined=left.merge(right,on=['region','factor'],suffixes=('_candidate','_reference'),validate='one_to_one')
        for row in joined.to_dict('records'):
            contrasts.append(dict(region=row['region'],factor=row['factor'],candidate=candidate,reference=reference,
                **{k+'_delta':row[k+'_candidate']-row[k+'_reference'] for k in METRICS}))
    for name,table in dict(day_metrics=day,region_metrics=regional,macro_metrics=macro,matched_contrasts=pd.DataFrame(contrasts)).items():
        table.to_csv(OUT/(name+'.csv'),index=False,mode='x')
    write(OUT/'source_choice.json',choice)
    check(); predictions(); verify(start)
    write(OUT/'source_completed.json',dict(version=25,created_utc=now(),freeze_sha256=start[str(OUT/'freeze.json')],
        prediction_seal_sha256=start[str(OUT/'prediction_seal.json')],output_hashes={str(OUT/n):sha(OUT/n) for n in OUTPUTS},
        target_values_opened=False,unseen_values_opened=False,custom_method=False,mechanism_breakthrough=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('prepare','predict','score'))
    args=parser.parse_args()
    {'prepare':prepare,'predict':predict,'score':score}[args.stage]()
