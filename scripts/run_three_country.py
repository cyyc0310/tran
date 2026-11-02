"""Same-period single-source vs two-source country-held-out development."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transcif.research.multicountry import (
    FACTORS,FINE,HISTORY,METHODS,coarse_values,fit,load_countries,predict,public_features,public_months,weights,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_cross_country import ROOT,digest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--us-pool',choices=['base','nuclear_free','nuclear_control'],default='base')
    parser.add_argument('--source-repair',type=Path)
    parser.add_argument('--capacity-table',type=Path)
    parser.add_argument('--reference-profile',type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'src/transcif/research/multicountry.py',
           ROOT/'src/transcif/research/cross_country.py',ROOT/'src/transcif/data/smard.py',
           ROOT/'src/transcif/research/monthly_adaptation.py',ROOT/'docs/THREE_COUNTRY_TRANSFER_PROTOCOL.md']
    if args.us_pool!='base':
        files += [ROOT/'src/transcif/research/source_support.py',ROOT/'docs/SOURCE_SUPPORT_EXPANSION_PROTOCOL.md']
    if args.source_repair:
        if args.us_pool=='base':raise ValueError('source repair requires an expansion pool')
        files += [ROOT/'docs/SOURCE_CALIBRATION_RECOVERY_PROTOCOL.md',
                  ROOT/'src/transcif/research/proxy_recovery.py']
    methods=METHODS
    capacity_table=None
    if args.reference_profile and (not args.capacity_table or args.us_pool=='base'):
        raise ValueError('fixed reference profiles require a capacity expansion pool')
    if args.capacity_table:
        if args.us_pool!='base' and not args.reference_profile:
            raise ValueError('capacity support comparison requires the frozen base profiles')
        from transcif.research import capacity_transfer
        capacity_table=capacity_transfer.load_capacity(args.capacity_table)
        for name,expected in capacity_table['input_hashes'].items():
            if digest(ROOT/name)!=expected:raise ValueError(f'capacity input changed: {name}')
        methods=(*METHODS,*capacity_transfer.METHODS)
        files += [args.capacity_table.resolve(),ROOT/'src/transcif/research/capacity_transfer.py',
                  ROOT/'docs/PUBLIC_CAPACITY_TRANSFER_PROTOCOL.md']
        files += [ROOT/name for name in capacity_table['input_hashes']]
        if args.us_pool!='base':
            files += [ROOT/'docs/CAPACITY_SOURCE_SUPPORT_PROTOCOL.md']
    if args.reference_profile:
        files += list(args.reference_profile.resolve().glob('*.joblib'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    (out/'startup_code_hashes.json').write_text(json.dumps(hashes,indent=2))
    raw,audit=load_countries(ROOT)
    original_regions=set(raw)
    if args.us_pool!='base':
        from transcif.research.source_support import POOLS,ZONES,extra_sources,matched_counts,match_added
        overrides={'LGEE':args.source_repair} if args.source_repair else None
        additions,addition_audit=extra_sources(ROOT,overrides);raw.update(additions);audit.update(addition_audit)
    features={};cohorts=[]
    for r,f in raw.items():
        v=public_features(public_months(f),f.clock)
        v['country'],v['region']=f.country,f.region
        v['year']=v.clock.dt.year;v['raw_row']=v.index
        v=v.dropna(subset=list(HISTORY))
        if capacity_table is not None:v=capacity_transfer.add_capacity(v,capacity_table)
        features[r]=v
        for year in (2022,2023):
            for month,n in v.loc[v.year.eq(year)].groupby(v.clock.dt.month).size().items():
                cohorts.append({'region':r,'country':f.country.iloc[0],'year':year,'month':int(month),'hours':int(n)})
    pd.DataFrame(cohorts).to_csv(out/'cohorts.csv',index=False)
    if args.us_pool!='base':
        train_counts=matched_counts(features,2022)
        cal_counts=matched_counts(features,2023,True)
        (out/'pool_layout.json').write_text(json.dumps({'pool':args.us_pool,'selected_additions':POOLS[args.us_pool],
            'train_monthly_count_per_added_region':train_counts,'cal_monthly_count_per_added_region':cal_counts},indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit,indent=2))
    records,seals=[],[]
    for target in ('DE','GB','US'):
        others=[c for c in ('DE','GB','US') if c!=target]
        for source_countries in [(others[0],),(others[1],),tuple(others)]:
            tag='+'.join(source_countries)+'_to_'+target
            regions=[r for r,f in raw.items() if f.country.iloc[0] in source_countries and
                     (r in original_regions or (args.us_pool!='base' and r in POOLS[args.us_pool]))]
            train_frames=[features[r].loc[features[r].year.eq(2022)] for r in regions]
            if args.us_pool!='base':
                train_frames=[match_added(f,train_counts) if r in ZONES else f for r,f in zip(regions,train_frames)]
            train=pd.concat(train_frames,ignore_index=True)
            fuel=np.concatenate([raw[r].loc[f.raw_row,list(FINE)].to_numpy() for r,f in zip(regions,train_frames)])
            coarse=np.concatenate([coarse_values(raw[r].loc[f.raw_row]) for r,f in zip(regions,train_frames)])
            models=fit(train,fuel,coarse)
            if args.reference_profile:
                reference=joblib.load(args.reference_profile/f'{tag}.joblib')
                models['profile']=reference['profile'].copy()
            capacity_models=capacity_transfer.fit(train,fuel,models) if capacity_table is not None else None
            joblib.dump(models,out/f'{tag}.joblib')
            if capacity_models is not None:joblib.dump(capacity_models,out/f'{tag}_capacity.joblib')
            cal_frames=[features[r].loc[features[r].year.eq(2023)&features[r].clock.dt.month.le(6)] for r in regions]
            if args.us_pool!='base':
                cal_frames=[match_added(f,cal_counts) if r in ZONES else f for r,f in zip(regions,cal_frames)]
            calibration=pd.concat(cal_frames,ignore_index=True)
            cal_fuel=np.concatenate([raw[r].loc[f.raw_row,list(FINE)].to_numpy() for r,f in zip(regions,cal_frames)])
            cal_y=cal_fuel@FACTORS/cal_fuel.sum(axis=1)
            cal_point=predict(models,calibration)
            if capacity_models is not None:cal_point.update(capacity_transfer.predict(capacity_models,calibration))
            w=weights(calibration)
            radii={m:weighted_quantile(np.abs(cal_y-cal_point[m]),w,.9) for m in methods}
            (out/f'{tag}_metadata.json').write_text(json.dumps({'source_countries':source_countries,
                'target_country':target,'training_hours':len(train),'radii':radii,
                'macro_ef':models['macro_ef'],'ratio_ef':models['ratio_ef'],
                'profile_reference':str(args.reference_profile) if args.reference_profile else None},indent=2))
            for r,f in features.items():
                if raw[r].country.iloc[0]!=target:
                    continue
                if r not in original_regions:continue
                query=f.loc[f.year.eq(2023)&f.clock.dt.month.gt(6)]
                predictions=predict(models,query)
                if capacity_models is not None:predictions.update(capacity_transfer.predict(capacity_models,query))
                path=out/f'{tag}_{r}.csv'
                pd.DataFrame({'time':raw[r].loc[query.raw_row,'time'].astype(str),**predictions}).to_csv(path,index=False)
                seals.append({'path':path.name,'sha256':digest(path)})
                records.append((tag,target,r,query,path,radii))
            print(tag,'sealed',flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2))
    rows=[]
    for tag,target,r,query,path,radii in records:
        predictions=pd.read_csv(path)
        fuel=raw[r].loc[query.raw_row,list(FINE)].to_numpy()
        y=fuel@FACTORS/fuel.sum(axis=1)
        for method in methods:
            p=predictions[method].to_numpy();radius=radii[method]
            for month in sorted(query.clock.dt.month.unique()):
                mask=query.clock.dt.month.eq(month).to_numpy();error=y[mask]-p[mask]
                temp=pd.DataFrame({'day':query.loc[mask].clock.dt.strftime('%Y-%m-%d').to_numpy(),
                                   'y':y[mask],'p':p[mask]})
                centered=temp[['y','p']]-temp.groupby('day')[['y','p']].transform('mean')
                rows.append({'configuration':tag,'target':target,'region':r,'method':method,'month':int(month),
                             'hours':int(mask.sum()),'mae':float(np.abs(error).mean()),
                             'shape_mae':float(np.abs(centered.y-centered.p).mean()),
                             'coverage90':float((np.abs(error)<=radius).mean()),'width90':float(2*radius),
                             'interval_score90':float(np.mean(2*radius+20*np.maximum(np.abs(error)-radius,0)))})
    monthly=pd.DataFrame(rows);monthly.to_csv(out/'monthly_metrics.csv',index=False)
    fields=['mae','shape_mae','coverage90','width90','interval_score90']
    regional=monthly.groupby(['configuration','target','region','method'])[fields].mean().reset_index()
    regional.to_csv(out/'region_metrics.csv',index=False)
    country=regional.groupby(['configuration','target','method'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv',index=False)
    if any(digest(ROOT/k)!=v for k,v in hashes.items()):raise RuntimeError('code changed during run')
    for item in audit.values():
        for key in ['path'] if 'path' in item else []:
            hashes[item[key]]=digest(ROOT/item[key])
    for year in (2022,2023):
        name=f'results/smard-de-{year}-full.json';hashes[name]=digest(ROOT/name)
    (out/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(country.to_string(index=False))


if __name__=='__main__':main()
