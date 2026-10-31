"""Source-only coarse-supervision time-window controls, immutable stages."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import coarse_supervision_v28 as model
from scripts import monthly_transfer_v25 as monthly
from scripts import run_monthly_transfer_v25 as prior
from scripts.native_center_v18 import scaled_paths

OUT = ROOT/'results/coarse-supervision-v28'
PARENT = ROOT/'docs/evidence/group-regression-v27-r1-verification.json'
V27 = ROOT/'results/group-regression-v27'
PARENT_SHA = '25f0df9f032d1d0d8ecd1169be6ed2ebc06700bf5e5ec26f3d3ce19de8f4b18e'
CODE = ('scripts/coarse_supervision_v28.py','scripts/run_coarse_supervision_v28.py',
        'tests/test_coarse_supervision_v28.py','tests/test_run_coarse_supervision_v28.py',
        'docs/COARSE_SUPERVISION_V28_PROTOCOL.md')
REVIEWS = tuple(ROOT/f'results/coarse-supervision-v28-{r}-review.md' for r in ('code','python','science'))
DAYS, FACTORS, METRICS = prior.DAYS, prior.FACTORS, prior.METRICS
sha, read, write, verify, now = prior.sha, prior.read, prior.write, prior.verify, prior.now
OUTPUTS = prior.OUTPUTS


def expected_contract() -> dict:
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('Fixed independently verified V27 R1 manifest required')
    evidence=read(PARENT)
    if evidence.get('status')!='VERIFICATION_COMPLETE' or evidence.get('independent_actual_audit_status')!='PASS':
        raise ValueError('V27 actual independent PASS required')
    inputs={str(PARENT):PARENT_SHA}
    for key in ('producer_code_hashes','input_hashes','auditor_code_hashes','artifact_hashes'):
        for path,digest in evidence[key].items():
            if path in inputs and inputs[path]!=digest:
                raise ValueError('Conflicting parent hash mappings')
            inputs[path]=digest
    verify(inputs)
    p=read(prior.OUT/'freeze.json'); seal=read(prior.OUT/'prediction_seal.json')
    if seal['freeze_sha256']!=sha(prior.OUT/'freeze.json') or [e['region'] for e in seal['entries']]!=list(monthly.SOURCES):
        raise ValueError('Fixed V25 seven-query aliases required')
    background=read(V27/'prediction_seal.json')
    if (background['freeze_sha256']!=sha(V27/'freeze.json')
            or [e['region'] for e in background['entries']]!=list(monthly.SOURCES)):
        raise ValueError('Audited V27 seven-region background aliases required')
    for path in REVIEWS:
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('Three prelaunch reviews required')
        inputs[str(path)]=sha(path)
    return dict(version=28,source=p['source'],sources=list(monthly.SOURCES),arms=list(model.ARMS),
        factors=list(FACTORS),samples=128,seed=17023,code_hashes={str(ROOT/n):sha(ROOT/n) for n in CODE},
        input_hashes=inputs,environment=prior.base.environment(),
        banks_by_outer=p['banks_by_outer'],query_paths=p['query_paths'],
        monthly_prediction_paths={e['region']:e['path'] for e in seal['entries']},
        background_prediction_paths={e['region']:e['path'] for e in background['entries']},
        monthly_path=p['monthly_path'],fixed_deployment_cutoff='2025-01-01T00:00:00+00:00',
        coarse_supervision_windows=['S24','L24','LR24'],
        annual_values_consumed=False,china_category_mapping_verified=False,
        target_values_opened=False,unseen_values_opened=False,custom_method=False,
        prediction_reads_new_fine=False,publication_vintages_verified=False,annual_only_evaluated=False)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve V28 attempt')
    contract=expected_contract(); OUT.mkdir()
    write(OUT/'freeze.json',dict(created_utc=now(),**contract))
    print('Prepared',len(contract['code_hashes']),'code',len(contract['input_hashes']),'input hashes',flush=True)


def check() -> dict:
    frozen=read(OUT/'freeze.json'); expected=expected_contract()
    if (set(frozen)!=set(expected)|{'created_utc'} or any(frozen[k]!=v for k,v in expected.items())
            or any(frozen[k] is not v for k,v in expected.items() if isinstance(v,bool))):
        raise ValueError('Frozen source, aliases, closure or contract changed')
    return frozen


def predict() -> None:
    frozen=check(); initial=sha(OUT/'freeze.json')
    if (OUT/'predictions').exists() or (OUT/'prediction_seal.json').exists():
        raise FileExistsError('Preserve V28 prediction attempt')
    folder=OUT/'predictions'; folder.mkdir()
    with Path(frozen['monthly_path']).open(newline='') as f:
        monthly_rows=list(csv.DictReader(f))  # Each fit filters strings before hidden fuel conversion.
    entries=[]
    for region in monthly.SOURCES:
        original=prior.load(frozen['query_paths'][region])
        pair=frozen['banks_by_outer'][region]
        bank=monthly.union_bank(prior.load(pair[0]),prior.load(pair[1]),original,outer=region)
        old=prior.load(frozen['monthly_prediction_paths'][region])
        if old['arms'].tolist()!=list(monthly.ARMS) or not np.array_equal(old['origins'],DAYS.asi8):
            raise ValueError('Exact V25 prediction registry required')
        k=old['arms'].tolist().index('MG-recompose_conditional')
        baseline={name:old[name][k] for name in ('raw','fine_draws','donor_ids','fallback_ids')}
        background=prior.load(frozen['background_prediction_paths'][region])
        context=model.fit_heads(bank,monthly_rows,outer=region)
        result=context.predict(baseline,old['MG_draws'],DAYS,background)
        fields=tuple(result['arms'][model.ARMS[0]])
        arrays={key:np.stack([result['arms'][a][key] for a in model.ARMS]) for key in fields}
        arrays.update({key:value for key,value in result.items() if isinstance(value,np.ndarray)})
        arrays.update(origins=DAYS.asi8,arms=np.asarray(model.ARMS),donor_ids=baseline['donor_ids'],
            fallback_ids=baseline['fallback_ids'],bank_regions=bank['regions'],bank_days=bank['days'])
        path=folder/f'{region}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,**arrays)
        audit_path=folder/f'{region}.json'
        write(audit_path,dict(region=region,parent_prediction_path=frozen['monthly_prediction_paths'][region],
            background_prediction_path=frozen['background_prediction_paths'][region],
            **result['audit']))
        entries.append(dict(region=region,path=str(path),sha256=sha(path),audit_path=str(audit_path),audit_sha256=sha(audit_path)))
        print('Predicted',region,flush=True)
        del old,result,arrays,bank,original
    check(); verify({str(OUT/'freeze.json'):initial})
    write(OUT/'prediction_seal.json',dict(created_utc=now(),freeze_sha256=initial,entries=entries,
        complete_grid=True,archives=7,raw_arms=21,prediction_reads_new_fine=False,
        target_values_opened=False,unseen_values_opened=False))


def predictions() -> dict:
    seal=read(OUT/'prediction_seal.json')
    if (seal['freeze_sha256']!=sha(OUT/'freeze.json') or seal['complete_grid'] is not True
            or seal['archives']!=7 or seal['raw_arms']!=21
            or any(seal[k] is not False for k in ('prediction_reads_new_fine','target_values_opened','unseen_values_opened'))
            or [e['region'] for e in seal['entries']]!=list(monthly.SOURCES)):
        raise ValueError('Complete seven-region seal required before source truth')
    for e in seal['entries']:
        if (Path(e['path']).resolve()!=(OUT/'predictions'/f"{e['region']}.npz").resolve()
                or Path(e['audit_path']).resolve()!=(OUT/'predictions'/f"{e['region']}.json").resolve()):
            raise ValueError('Registered archive and audit paths required')
        verify({e['path']:e['sha256'],e['audit_path']:e['audit_sha256']})
        a=prior.load(e['path'])
        if (a['arms'].tolist()!=list(model.ARMS) or not np.array_equal(a['origins'],DAYS.asi8)
                or a['raw'].shape!=(21,181,128,24) or not np.isfinite(a['raw']).all()
                or a['fine_draws'].shape!=(21,181,128,8)):
            raise ValueError('Complete 21-arm source archive required')
    return seal


def matched_pairs() -> list[tuple[str,str]]:
    pairs=[]
    for window in ('S24','L24','LR24'):
        for name in ('Ridge','ET'):
            prefix=f'{window}-{name}-'
            pairs.extend([(prefix+'residual_recompose',prefix+'residual_anchor'),
                          (prefix+'residual_recompose',prefix+'plugin_recompose')])
    for name in ('Ridge','ET'):
        for variant in ('plugin_recompose','residual_anchor','residual_recompose'):
            pairs.extend([(f'L24-{name}-{variant}',f'S24-{name}-{variant}'),
                          (f'LR24-{name}-{variant}',f'L24-{name}-{variant}')])
        pairs.extend([(f'S24-{name}-residual_recompose',f'V27-{name}-residual_recompose'),
                      (f'V27-{name}-residual_recompose','MG-recompose_conditional')])
    if len(pairs)!=28 or len(set(pairs))!=28:
        raise ValueError('Fixed28 matched comparisons required')
    return pairs


def score() -> None:
    if any((OUT/n).exists() for n in (*OUTPUTS,'source_completed.json')):
        raise FileExistsError('Preserve V28 score attempt')
    frozen=check(); seal=predictions()
    start={str(OUT/n):sha(OUT/n) for n in ('freeze.json','prediction_seal.json')}
    rows=[]
    for e in seal['entries']:
        region=e['region']; a=prior.load(e['path'])
        truth={d.value:(g/g.sum(1)[:,None])@monthly.EF for d,g in prior.base.source_days(
            Path(frozen['source'])/f'{region}_hourly.csv','2025-01-01','2025-07-01')}
        keep=np.isin(a['origins'],list(truth)); origins=a['origins'][keep]
        y=np.stack([truth[int(d)] for d in origins])
        for i,arm in enumerate(model.ARMS):
            for factor in FACTORS:
                point,paths,clip=scaled_paths(a['raw'][i,keep],factor)
                scores=prior.base.score_arrays(point,paths,y)
                rows.extend(dict(region=region,origin=pd.Timestamp(int(d),tz='UTC').isoformat(),arm=arm,factor=factor,
                    final_spread_clipping_rate=float(clip[j]),**{k:float(scores[k][j]) for k in METRICS}) for j,d in enumerate(origins))
        print('Scored',region,flush=True)
    day=pd.DataFrame(rows)
    if (len(day)!=120225 or len(day[['region','origin']].drop_duplicates())!=1145
            or day.duplicated(['region','origin','arm','factor']).any()
            or not np.isfinite(day.select_dtypes('number')).all().all()):
        raise ValueError('Exact1145-day/105-curve source cohort required')
    regional=day.groupby(['region','arm','factor'])[list(METRICS)].mean().reset_index()
    if len(regional)!=735:
        raise ValueError('Exact735 region curves required')
    macro,choice=prior.summarize(regional)
    if len(macro)!=105:
        raise ValueError('Exact105 macro curves required')
    comparisons=matched_pairs()
    contrasts=[]
    for candidate,reference in comparisons:
        joined=regional[regional.arm.eq(candidate)].merge(regional[regional.arm.eq(reference)],
            on=['region','factor'],suffixes=('_candidate','_reference'),validate='one_to_one')
        for row in joined.to_dict('records'):
            contrasts.append(dict(region=row['region'],factor=row['factor'],candidate=candidate,reference=reference,
                **{k+'_delta':row[k+'_candidate']-row[k+'_reference'] for k in METRICS}))
    if len(contrasts)!=980:
        raise ValueError('Exact980 matched contrast rows required')
    for name,table in dict(day_metrics=day,region_metrics=regional,macro_metrics=macro,matched_contrasts=pd.DataFrame(contrasts)).items():
        table.to_csv(OUT/(name+'.csv'),index=False,mode='x')
    write(OUT/'source_choice.json',choice)
    check(); predictions(); verify(start)
    write(OUT/'source_completed.json',dict(version=28,created_utc=now(),freeze_sha256=start[str(OUT/'freeze.json')],
        prediction_seal_sha256=start[str(OUT/'prediction_seal.json')],output_hashes={str(OUT/n):sha(OUT/n) for n in OUTPUTS},
        target_values_opened=False,unseen_values_opened=False,custom_method=False,mechanism_breakthrough=False,
        annual_values_consumed=False,china_category_mapping_verified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('prepare','predict','score'))
    args=parser.parse_args()
    {'prepare':prepare,'predict':predict,'score':score}[args.stage]()
