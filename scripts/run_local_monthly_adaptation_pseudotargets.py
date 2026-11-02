#!/usr/bin/env python3
"""US BA leave-one-out validation of label-free monthly composition adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_dual_track_monthly_adaptation as dual
import run_eia923_pseudotarget_transfer as eia_monthly
import run_official_intensity_transfer as official
from run_cross_country import ROOT


PROTOCOL=ROOT/'docs/LOCAL_MONTHLY_ADAPTATION_PSEUDOTARGET_PROTOCOL_20261101.md'
BA_REGIONS=official.US_REGIONS
ADAPTERS=('observed_m','persistence','seasonal','local_ridge','source_ridge')
TRACKS=('generation_proxy','consumption_official')
POINTS=('coarse_boosting','fixed_ef_recomposition')


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)

    capacity=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw,base,_,audit=official.prepare(ROOT,capacity)
    monthly=eia_monthly.parse_monthly(tuple(range(2020,2025)),BA_REGIONS)
    prepared_arms,support=eia_monthly.replace_source_features(base,monthly)
    observed=prepared_arms['eia923_monthly']
    tables=dual.monthly_tables(monthly)
    code_files=[Path(__file__),PROTOCOL,ROOT/'scripts/run_dual_track_monthly_adaptation.py',
        ROOT/'scripts/run_eia923_pseudotarget_transfer.py',ROOT/'scripts/run_official_intensity_transfer.py',
        ROOT/'scripts/run_cross_country.py',ROOT/'src/transcif/research/cross_country.py',
        ROOT/'src/transcif/research/monthly_adaptation.py']
    input_files=[ROOT/'data/neso-gb/df_fuel_ckan.csv',official.EIA_ROOT/'acquisition_manifest.json',
        official.EIA_ROOT/'coverage_audit_2022_2024.json',official.CAPACITY_TABLE_PATH]
    input_files += [eia_monthly.eia_archive(y) for y in range(2020,2025)]
    input_files += [ROOT/f'results/research/data/{r}_hourly.csv' for r in BA_REGIONS]
    input_files += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in BA_REGIONS]
    hashes={'code':{str(p):sha(p) for p in dict.fromkeys(code_files)},
        'inputs':{str(p):sha(p) for p in dict.fromkeys(input_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'input_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    (out/'common_support.json').write_text(json.dumps(support,indent=2)+'\n')

    pending=[];seals=[];calibrations=[]
    for heldout in BA_REGIONS:
        sources=[r for r in BA_REGIONS if r!=heldout]
        source_tables={r:tables[r] for r in sources}
        for adapter in ADAPTERS:
            if adapter=='observed_m':
                features={r:observed[r].copy() for r in BA_REGIONS}
            else:
                features={r:dual.transform(observed[r],tables[r],adapter,source_tables)
                          for r in BA_REGIONS}
            for track in TRACKS:
                source_labels=dual.label_series(track,raw,(2022,2023))
                models,fuel_models=dual.fit_response(sources,features,raw,source_labels)
                calibration=dual.calibration_for(sources,features,models,fuel_models,source_labels)
                calibrations.append({'pseudo_target':heldout,'adapter':adapter,'track':track,
                                     'calibration':calibration})
                query=features[heldout].loc[features[heldout].year.eq(2024)].copy().reset_index(drop=True)
                point=official.point_predictions(models,fuel_models,query)
                data={'time_utc':pd.to_datetime(query.time,utc=True).astype(str).to_numpy(),**point}
                for method in POINTS:
                    center=point[method]+calibration[method]['bias']
                    radius=calibration[method]['radius90']
                    data[f'{method}_lo']=center-radius;data[f'{method}_hi']=center+radius
                path=out/f'{heldout}_{adapter}_{track}_predictions.csv.gz'
                pd.DataFrame(data).to_csv(path,index=False,compression='gzip')
                seal={'pseudo_target':heldout,'source_regions':sources,'adapter':adapter,'track':track,
                    'path':path.name,'sha256':sha(path),'predicted_rows':len(query)}
                seals.append(seal);pending.append((seal,query,calibration))
        print('sealed BA pseudo-target',heldout,flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2)+'\n')
    (out/'source_calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    for group in ('code','inputs'):
        if any(sha(Path(p))!=v for p,v in hashes[group].items()):
            raise RuntimeError(f'{group} hash mismatch')
    if any(sha(out/s['path'])!=s['sha256'] for s in seals):
        raise RuntimeError('sealed prediction hash mismatch')

    rows=[];target_hashes={}
    for seal,query,calibration in pending:
        region=seal['pseudo_target'];track=seal['track']
        if track=='consumption_official':
            target_labels={region:official.source_labels(region,years=(2024,))}
            target_path=official.TARGET_LABEL_ROOT/f'{region}_consumption_2024.csv.gz'
        else:
            target_labels=dual.label_series(track,raw,(2024,))
            target_path=ROOT/f'results/research/data/{region}_hourly.csv'
        target_hashes[f'{track}:{region}']=sha(target_path)
        truth=dual.align(query,target_labels)
        stored=pd.read_csv(out/seal['path'])
        for method in POINTS:
            point=stored[method].to_numpy(float)+calibration[method]['bias']
            lo=stored[f'{method}_lo'].to_numpy(float);hi=stored[f'{method}_hi'].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                take=query.clock.dt.month.eq(month).to_numpy()&np.isfinite(truth)
                if not take.any():continue
                error=truth[take]-point[take]
                day=pd.DataFrame({'y':truth[take],'p':point[take],
                    'd':query.loc[take,'clock'].dt.strftime('%Y-%m-%d').to_numpy()})
                centered=day[['y','p']]-day.groupby('d')[['y','p']].transform('mean')
                rows.append({'pseudo_target':region,'adapter':seal['adapter'],'track':track,
                    'method':method,'month':int(month),'hours':int(take.sum()),
                    'mae':float(np.abs(error).mean()),
                    'shape_mae':float(np.abs(centered.y-centered.p).mean()),
                    'coverage90':float(((truth[take]>=lo[take])&(truth[take]<=hi[take])).mean()),
                    'width90':float(np.mean(hi[take]-lo[take])),
                    'interval_score90':float(np.mean(hi[take]-lo[take]+
                        20*np.maximum(lo[take]-truth[take],0)+
                        20*np.maximum(truth[take]-hi[take],0)))})
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'monthly_metrics.csv',index=False)
    target_hash_path=out/'target_scoring_hashes.json'
    target_hash_path.write_text(json.dumps(target_hashes,indent=2)+'\n')
    fields=['mae','shape_mae','coverage90','width90','interval_score90']
    region=metrics.groupby(['pseudo_target','adapter','track','method'])[fields].mean().reset_index()
    region.to_csv(out/'pseudo_target_metrics.csv',index=False)
    macro=region.groupby(['adapter','track','method'])[fields].mean().reset_index()
    macro.to_csv(out/'macro_metrics.csv',index=False)
    verify={'code_files_checked':len(hashes['code']),'input_files_checked':len(hashes['inputs']),
        'code_hash_mismatches':[p for p,v in hashes['code'].items() if sha(Path(p))!=v],
        'input_hash_mismatches':[p for p,v in hashes['inputs'].items() if sha(Path(p))!=v],
        'sealed_predictions':len(seals),'prediction_rows':sum(s['predicted_rows'] for s in seals),
        'prediction_hash_mismatches':[s['path'] for s in seals if sha(out/s['path'])!=s['sha256']],
        'target_label_files':len(target_hashes),
        'target_label_hash_mismatches':[p for p,v in target_hashes.items()
            if sha((ROOT/f'results/research/data/{p.split(":")[1]}_hourly.csv') if p.startswith('generation_proxy:')
                else official.TARGET_LABEL_ROOT/f'{p.split(":")[1]}_consumption_2024.csv.gz')!=v],
        'target_predictions_sealed_before_label_scoring':True}
    (out/'post_run_verification.json').write_text(json.dumps(verify,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps({'status':'DEVELOPMENT_US_BA_PSEUDOTARGET_MONTHLY_ADAPTATION',
        'pseudo_targets':list(BA_REGIONS),'source_train_year':2022,'source_calibration_year':2023,
        'target_year':2024,'tracks':list(TRACKS),'target_labels_previously_viewed':True,
        'prediction_scoring_after_seal':True,
        'monthly_vintage':'EIA-923 final annual files, lagged t-2 use simulated, not as-issued'},
        indent=2)+'\n')
    print(macro.to_string(index=False))


if __name__=='__main__':
    main()
