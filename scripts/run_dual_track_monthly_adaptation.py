#!/usr/bin/env python3
"""Paired monthly-composition adaptation for generation and consumption targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_eia923_pseudotarget_transfer as eia_monthly
import run_official_intensity_transfer as official
from run_cross_country import ROOT
from transcif.research import cross_country, monthly_adaptation


PROTOCOL = ROOT/'docs/DUAL_TRACK_MONTHLY_ADAPTATION_PROTOCOL_20261101.md'
DIRECTIONS = (('US', 'GB'), ('GB', 'US'))
TRACKS = ('generation_proxy', 'consumption_official')
ADAPTERS = ('observed_m', 'persistence', 'seasonal', 'local_ridge', 'source_ridge')
POINTS = ('source_mean', 'calendar_ridge', 'coarse_ridge', 'coarse_boosting',
          'fixed_ef_recomposition')
EF = cross_country.EF


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def neso_monthly_table() -> pd.DataFrame:
    path = ROOT/'data/neso-gb/df_fuel_ckan.csv'
    columns = ['DATETIME','GAS','COAL','NUCLEAR','WIND','WIND_EMB','HYDRO','BIOMASS','OTHER','SOLAR']
    raw = pd.read_csv(path, usecols=columns)
    raw['stamp'] = pd.to_datetime(raw.DATETIME, utc=True, errors='raise')
    raw['clock'] = raw.stamp.dt.tz_convert('Europe/London').dt.tz_localize(None).dt.floor('h')
    raw['coal'], raw['gas'], raw['oil'] = raw.COAL, raw.GAS, 0.
    raw['other'] = raw.BIOMASS + raw.OTHER
    raw['nuclear'], raw['hydro'] = raw.NUCLEAR, raw.HYDRO
    raw['wind'], raw['solar'] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    fuels = raw[list(cross_country.FUELS)].to_numpy(float)
    raw = raw.loc[np.isfinite(fuels).all(axis=1) & (fuels >= 0).all(axis=1)].copy()
    hourly = raw.groupby('clock')[list(cross_country.FUELS)].mean()
    hourly = hourly.loc[raw.groupby('clock').size().eq(2)]
    period = hourly.index.to_period('M')
    totals = hourly.groupby(period).sum()
    expected = pd.Series({p: ((p+1).start_time.tz_localize('Europe/London')-
                                p.start_time.tz_localize('Europe/London')).total_seconds()/3600
                         for p in totals.index})
    observed = hourly.groupby(period).size()
    totals = totals.loc[observed.reindex(totals.index).to_numpy() >= .99*expected.reindex(totals.index).to_numpy()]
    totals['thermal'] = totals[['coal','gas','oil','other']].sum(axis=1)
    totals = totals[list(cross_country.COARSE)]
    return totals.div(totals.sum(axis=1),axis=0)


def monthly_tables(monthly_eia: pd.DataFrame) -> dict[str,pd.DataFrame]:
    tables = {'GB': neso_monthly_table()}
    for ba, group in monthly_eia.groupby('ba',sort=True):
        frame=group.set_index('observed_month')[list(cross_country.COARSE)].copy()
        frame.index=pd.PeriodIndex(frame.index,freq='M')
        tables[ba]=frame.sort_index()
    return tables


def transform(frame: pd.DataFrame, table: pd.DataFrame, method: str,
              source_tables: dict[str,pd.DataFrame] | None = None,
              source_cutoff: pd.Period | None = None) -> pd.DataFrame:
    """Use observed t-2 for M, otherwise a strict as-of forecast of month t mix."""
    out=frame.copy()
    if method=='observed_m':
        return out
    source_model=None
    if method=='source_ridge':
        if source_tables is None:
            raise ValueError('source Ridge requires source month histories')
        cutoff=source_cutoff or pd.Period('2022-12',freq='M')
        bounded={r:t.loc[t.index<=cutoff] for r,t in source_tables.items()}
        source_model=monthly_adaptation.fit_source(bounded)
    cache={}
    for period,positions in out.groupby('period',sort=False).groups.items():
        period=pd.Period(period,freq='M')
        if method=='source_ridge' and period.year<=2022:
            # Rolling-origin source features: the month being predicted is
            # excluded from the regression fit, as it must be at deployment.
            cutoff=period-2
            bounded={r:t.loc[t.index<=cutoff] for r,t in source_tables.items()}
            local_source=monthly_adaptation.fit_source(bounded)
        else:
            local_source=source_model
        cache[period]=monthly_adaptation.forecast(table,period,local_source,
                      'source_ridge' if method=='source_ridge' else method)
        values=cache[period]
        out.loc[positions,list(cross_country.COARSE)]=values
    for category in cross_country.COARSE:
        for suffix in ('sin','cos'):
            out[f'{category}_hour_{suffix}']=out[category]*out[f'hour_{suffix}']
    return out


def label_series(track: str, raw: dict[str,pd.DataFrame], years: tuple[int,...]) -> dict[str,pd.Series]:
    if track=='consumption_official':
        return {region:official.source_labels(region,years=years)
                for region in (*official.US_REGIONS,'GB')}
    result={}
    for region in (*official.US_REGIONS,'GB'):
        frame=raw[region]
        time=pd.DatetimeIndex(pd.to_datetime(frame.time,utc=True,errors='raise'))
        fuel=frame[list(official.FUELS)].to_numpy(float)
        total=fuel.sum(axis=1)
        keep=time.year.isin(years)&np.isfinite(fuel).all(axis=1)&(total>0)
        result[region]=pd.Series(fuel[keep]@EF/total[keep],index=time[keep],name='label').sort_index()
    return result


def align(frame:pd.DataFrame,labels:dict[str,pd.Series])->np.ndarray:
    region=frame.region.iloc[0]
    return official.aligned_labels(frame,labels[region])


def fit_response(regions:list[str],features:dict[str,pd.DataFrame],raw:dict[str,pd.DataFrame],
                 labels:dict[str,pd.Series]):
    frames=[];values=[];frame_regions=[]
    for region in regions:
        query=features[region].loc[features[region].year.eq(2022)].copy()
        truth=align(query,labels)
        keep=np.isfinite(truth)
        if keep.any():
            frames.append(query.loc[keep]);values.append(truth[keep])
            frame_regions.append(region)
    train=pd.concat(frames,ignore_index=True)
    y=np.concatenate(values)
    models=official.train_official(train,y)
    fuels=np.concatenate([raw[r].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float)
                          for r,q in zip(frame_regions,frames)])
    fuel_models=official.fit_models(train.reset_index(drop=True),fuels)
    return models,fuel_models


def calibration_for(regions:list[str],features:dict[str,pd.DataFrame],models:dict,
                    fuel_models:dict,labels:dict)->dict:
    h1=[];h2=[]
    for region in regions:
        f=features[region]
        for year,container in ((2023,h1),(2023,h2)):
            months=f.clock.dt.month
            mask=f.year.eq(year)&(months.le(6) if container is h1 else months.gt(6))
            q=f.loc[mask].copy()
            truth=align(q,labels)
            pred=official.point_predictions(models,fuel_models,q)
            container.append((region,{method:truth-pred[method] for method in POINTS}))
    result={}
    for method in POINTS:
        region_means=[float(v[method][np.isfinite(v[method])].mean()) for _,v in h1]
        bias=float(np.mean(region_means))
        residuals=[];groups=[]
        for region,preds in h2:
            values=preds[method]
            valid=np.isfinite(values)
            residuals.extend((values[valid]-bias).tolist())
            groups.extend([region]*int(valid.sum()))
        weights=monthly_adaptation.region_weights(np.asarray(groups))
        radius=monthly_adaptation.weighted_quantile(np.abs(np.asarray(residuals)),weights,.9)
        result[method]={'bias':bias,'radius90':radius,'calibration_regions':len(regions),
                        'h1_hours':int(sum(np.isfinite(v[method]).sum() for _,v in h1)),
                        'h2_hours':len(residuals)}
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    capacity=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw,base,_,audit=official.prepare(ROOT,capacity)
    mix=eia_monthly.parse_monthly(tuple(range(2020,2025)),official.US_REGIONS)
    prepared_by_arm,support=eia_monthly.replace_source_features(base,mix)
    tables=monthly_tables(mix)
    code_files=[Path(__file__),PROTOCOL,Path(eia_monthly.__file__),Path(official.__file__),
        ROOT/'scripts/run_cross_country.py',ROOT/'src/transcif/research/cross_country.py',
        ROOT/'src/transcif/research/monthly_adaptation.py']
    input_files=[ROOT/'data/neso-gb/df_fuel_ckan.csv',official.EIA_ROOT/'acquisition_manifest.json',
        official.EIA_ROOT/'coverage_audit_2022_2024.json',official.CAPACITY_TABLE_PATH]
    input_files += [eia_monthly.eia_archive(year) for year in range(2020,2025)]
    input_files += [ROOT/f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    input_files += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in official.US_REGIONS]
    input_files += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz']
    hashes={'code':{str(p):sha(p) for p in dict.fromkeys(code_files)},
            'inputs':{str(p):sha(p) for p in dict.fromkeys(input_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'input_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    (out/'common_support.json').write_text(json.dumps(support,indent=2)+'\n')

    pending=[];seals=[];calibration_records=[]
    for source_country,target_country in DIRECTIONS:
        source_regions=[r for r,f in prepared_by_arm['eia923_monthly'].items() if f.country.iloc[0]==source_country]
        target_regions=[r for r,f in prepared_by_arm['eia923_monthly'].items() if f.country.iloc[0]==target_country]
        source_tables={r:tables[r] for r in source_regions}
        for adapter in ADAPTERS:
            features={}
            for region,frame in prepared_by_arm['eia923_monthly'].items():
                if adapter=='observed_m':
                    features[region]=frame.copy()
                else:
                    features[region]=transform(frame,tables[region],adapter,source_tables)
            for track in TRACKS:
                source_labels=label_series(track,raw,(2022,2023))
                models,fuel_models=fit_response(source_regions,features,raw,source_labels)
                calibration=calibration_for(source_regions,features,models,fuel_models,source_labels)
                calibration_records.append({'source':source_country,'target':target_country,
                    'adapter':adapter,'track':track,'calibration':calibration})
                for region in target_regions:
                    query=features[region].loc[features[region].year.eq(2024)].copy().reset_index(drop=True)
                    prediction=official.point_predictions(models,fuel_models,query)
                    path=out/f'{source_country}_{region}_{adapter}_{track}_predictions.csv.gz'
                    payload={'time_utc':pd.to_datetime(query.time,utc=True).astype(str).to_numpy(),
                             'observed_month':query.observed_month.to_numpy(),**prediction}
                    for method in POINTS:
                        bias=calibration[method]['bias'];radius=calibration[method]['radius90']
                        center=prediction[method]+bias
                        payload[f'{method}_lo']=center-radius;payload[f'{method}_hi']=center+radius
                    pd.DataFrame(payload).to_csv(path,index=False,compression='gzip')
                    seal={'source':source_country,'target':target_country,'region':region,
                          'adapter':adapter,'track':track,'path':path.name,'sha256':sha(path),
                          'hours':len(query)}
                    seals.append(seal);pending.append((seal,query,calibration))
        print(source_country,'to',target_country,'all monthly adapters and tracks sealed',flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2)+'\n')
    (out/'source_calibration.json').write_text(json.dumps(calibration_records,indent=2)+'\n')
    for group in ('code','inputs'):
        if any(sha(Path(path))!=digest for path,digest in hashes[group].items()):
            raise RuntimeError(f'{group} hash mismatch during run')
    if any(sha(out/s['path'])!=s['sha256'] for s in seals):
        raise RuntimeError('sealed target prediction hash mismatch')

    records=[]
    for seal,query,calibration in pending:
        region=seal['region'];track=seal['track']
        if track=='consumption_official':
            target_labels={region:official.source_labels(region,years=(2024,))}
        else:
            target_labels=label_series(track,raw,(2024,))
        truth=align(query,target_labels)
        pred=pd.read_csv(out/seal['path'])
        for method in POINTS:
            point=pred[method].to_numpy(float)+calibration[method]['bias']
            lo=pred[f'{method}_lo'].to_numpy(float);hi=pred[f'{method}_hi'].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                take=query.clock.dt.month.eq(month).to_numpy()&np.isfinite(truth)
                if not take.any():continue
                err=truth[take]-point[take]
                day=pd.DataFrame({'y':truth[take],'p':point[take],
                    'd':query.loc[take,'clock'].dt.strftime('%Y-%m-%d').to_numpy()})
                centered=day[['y','p']]-day.groupby('d')[['y','p']].transform('mean')
                records.append({'direction':seal['source']+'_to_'+seal['target'],
                    'target_region':region,'adapter':seal['adapter'],'track':track,
                    'method':method,'month':int(month),'hours':int(take.sum()),
                    'mae':float(np.abs(err).mean()),
                    'shape_mae':float(np.abs(centered.y-centered.p).mean()),
                    'coverage90':float(((truth[take]>=lo[take])&(truth[take]<=hi[take])).mean()),
                    'width90':float(np.mean(hi[take]-lo[take])),
                    'interval_score90':float(np.mean(hi[take]-lo[take]+
                         20*np.maximum(lo[take]-truth[take],0)+
                         20*np.maximum(truth[take]-hi[take],0)))})
    monthly=pd.DataFrame(records);monthly.to_csv(out/'monthly_metrics.csv',index=False)
    fields=['mae','shape_mae','coverage90','width90','interval_score90']
    regional=monthly.groupby(['direction','target_region','adapter','track','method'])[fields].mean().reset_index()
    regional.to_csv(out/'region_metrics.csv',index=False)
    endpoint=regional.groupby(['direction','adapter','track','method'])[fields].mean().reset_index()
    endpoint.to_csv(out/'endpoint_metrics.csv',index=False)
    (out/'status.json').write_text(json.dumps({'status':'DEVELOPMENT_DUAL_TRACK_MONTHLY_ADAPTATION',
        'directions':['US_to_GB','GB_to_US'],'target_year':2024,'source_train_year':2022,
        'source_calibration':'2023H1 bias + 2023H2 weighted absolute-residual q90',
        'target_monthly_inputs':'EIA-923 final archives for US, NESO history for GB, exact t-2 index',
        'monthly_vintage':'EIA-923 final annual files; lagged use simulated, not historical as-issued',
        'target_labels_previously_viewed':True,'tracks':list(TRACKS),
        'target_labels_scored_after_prediction_seal':True},indent=2)+'\n')
    print(endpoint.groupby(['track','adapter','method'])[fields].mean().to_string())


if __name__=='__main__':
    main()
